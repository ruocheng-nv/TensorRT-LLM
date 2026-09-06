# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage-4 long-horizon B/E state canary on the serving-equivalent LLM API path.

Acceptance shape (Stage 4, criterion 3): before the full GSM8K gate, config B
and config E each generate **>= 2048 new tokens** for at least two fixed
prompts and must be **exactly deterministic-equivalent token-by-token**, with
periodic finite per-stream Sinkhorn sums/norms, valid expanded pool
indices/``-1`` masks, and isolated KDA/MLA/pool cache state. The pass
condition compares TensorRT-LLM B with TensorRT-LLM E — native-HF tokens are
not consulted (``reference_tier=reduced_source``); the two prompts are the
frozen iteration-21 manifest's canary prompts, reused for their
implementation-independent provenance (only their ``input_ids`` are read).

Engine geometry is the literal Goal-4.2 graded trtllm-serve pair:
``max_batch_size=4, max_num_tokens=4096, max_seq_len=4096``, KV cache
``enable_block_reuse=false, max_tokens=16384`` — the canary rows (prompt
<= 50 tokens + 2048 new) fit that serving budget with headroom, which is the
point: the serving configuration itself sustains the long horizon with zero
truncation (``ignore_eos`` + ``max_tokens`` = the window, so every row must
finish ``length`` after exactly its budget).

Evidence channels:

* **Tokens** — three sequential rows: ``canary0``, ``canary1``, then
  ``canary0_rerun`` (the first prompt again, through recycled slots/pages).
  Within a config, ``canary0_rerun`` must reproduce ``canary0`` bitwise
  (slot-reuse isolation at a 2048-token horizon); across configs, every row's
  full token sequence must match B<->E exactly.
* **Runtime state** — ``GLM53_STATE_DIGEST_DIR`` (the Stage-3 hook): every PP
  rank digests each decode step's input KDA conv/ssm slot state and sparse
  latent/index pages. Per-row per-step digests are merged and hashed; the E
  run compares its full per-step sha256 trajectory against B's (bitwise
  isolated-state equivalence), and ``canary0`` vs ``canary0_rerun`` proves
  in-config slot isolation.
* **Sinkhorn / pool health (config B only)** — ``GLM53_STREAM_PROBE_DIR``
  enables the model's periodic decode-time probe: per-stream sums/L2 norms of
  the four hyper-connection streams, Sinkhorn ``comb`` row/column sums, and
  ``post`` weights per local layer, plus the sparse layers' expanded pool
  index validity (every non-sentinel index in ``[0, kv_len)``, every row
  covered, ``-1`` sentinels honest). The probe is read-only eager
  observation; it is NOT enabled for E, where CUDA-graph replay does not
  re-run Python — E's state health is instead proven by its bitwise state/
  token equality with the probed B run plus the digest hook's own finite
  invariants.
* **CUDA-graph hard path (config E)** — capture lines >= 1, zero
  eager-fallback markers, and the per-rank ladder audit covering every
  decode-only batch size (1..4) on every PP rank.

Every exit path writes the JSON summary and ``<summary>.exit.txt``.

Run (B first, then E with --compare-with):
    python tests/unittest/_torch/modeling/glm5_next_long_state_canary.py \
        --pp 4 --config B --probe-dir <dir> --state-digest-dir <dir> \
        --summary <b.json> --metrics-out <b.pt>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Tuple

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 4))
sys.path.insert(0, _REPO)
os.environ["PYTHONPATH"] = _REPO + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ["TLLM_LOG_LEVEL"] = "INFO"

from glm5_next_driver_preflight import (  # noqa: E402  (script-dir import)
    audit_graph_ladder,
    disk_preflight,
    expected_graph_batch_sizes,
)
from glm5_next_llm_api_logit_replay import (  # noqa: E402  (script-dir import)
    check_cuda_graph_hard_path,
    merge_state_digests,
    resolve_moe_parallel,
    sha256_of,
    tee_output_to,
)

# The two literal per-layer schedules pinned in Stage 1 (goal1.1): 45 layers,
# sparse attention at zero-based 3, 7, ..., 43, linear attention elsewhere.
# The probe-coverage audit requires the rank files to jointly cover them.
NUM_LAYERS = 45
SPARSE_LAYER_IDS = tuple(range(3, NUM_LAYERS, 4))

# Serving-equivalent engine geometry — the Goal-4.2 graded trtllm-serve pair.
ENGINE_MAX_BATCH_SIZE = 4
ENGINE_MAX_NUM_TOKENS = 4096
ENGINE_MAX_SEQ_LEN = 4096
ENGINE_KV_MAX_TOKENS = 16384

# The 20-round FP32 Sinkhorn drives comb toward doubly stochastic but does
# not reach exact row/col sums of 1.0 on sharp 4x4 matrices: the iteration-39
# wiring run measured a max |sum - 1| of 0.084 across 12 probed steps x 45
# layers x 4 ranks on the real checkpoint (and the Stage-1 "1.3e-6 residual"
# was agreement WITH native HF, not exact double-stochasticity). The hard
# health gate here is FINITENESS (the acceptance criterion's own bar); this
# bound only catches catastrophic accumulation drift (sums collapsing toward
# 0 or exploding), set 3x above the measured real-checkpoint envelope.
# Observed extrema are always reported verbatim in the evidence.
COMB_SUM_TOL = 0.25


def load_canary_prompts(canary_ref: str) -> List[Dict[str, Any]]:
    """The two frozen canary prompts' ``input_ids`` (reference logits unused)."""
    import torch

    blob = torch.load(canary_ref, map_location="cpu", weights_only=False)
    rows = [
        {"name": f"canary{r['index']}", "input_ids": [int(x) for x in r["input_ids"]]}
        for r in blob["canary"]
    ]
    if len(rows) < 2:
        raise AssertionError(f"long-state canary needs >=2 fixed prompts, found {len(rows)}")
    return rows


def expected_min_probe_steps(num_rows: int, new_tokens: int, every: int) -> int:
    """Lower bound on distinct probed decode steps a healthy run must record.

    Each row of ``new_tokens`` greedy tokens performs ``new_tokens - 1``
    decode forwards (token 1 comes from prefill), every rank sees each decode
    forward once, and the probe fires on every ``every``-th one. Two steps of
    slack absorb counter phase relative to engine warmup forwards.
    """
    return max(1, num_rows * (new_tokens - 1) // max(1, every) - 2)


def validate_probe_records(
    records_by_rank: Dict[str, List[Dict[str, Any]]],
    *,
    min_probe_steps: int,
    hc_layers_expected: Tuple[int, ...] = tuple(range(NUM_LAYERS)),
    pool_layers_expected: Tuple[int, ...] = SPARSE_LAYER_IDS,
    comb_tol: float = COMB_SUM_TOL,
) -> Tuple[Dict[str, Any], List[str]]:
    """Audit the merged stream/pool probe records.

    Pure function over parsed JSONL records so the audit itself is
    unit-testable without CUDA. Checks: every hc record's streams/comb/post
    are finite with Sinkhorn row/column sums within ``comb_tol`` of 1.0;
    every pool record has all non-sentinel indices in range and every request
    row covered; the rank files jointly cover every decoder layer (hc) and
    every sparse layer (pool); and the number of distinct probed steps meets
    the periodicity contract.
    """
    problems: List[str] = []
    hc_layers, pool_layers = set(), set()
    hc_steps, pool_steps = set(), set()
    hc_count = pool_count = 0
    comb_dev_max = 0.0
    stream_l2_max = 0.0
    post_min, post_max = float("inf"), float("-inf")
    sentinel_total = 0
    nonsentinel_max = -1

    for rank, records in sorted(records_by_rank.items()):
        for rec in records:
            where = f"rank {rank} probe_step {rec.get('probe_step')} layer {rec.get('layer')}"
            if rec.get("kind") == "hc":
                hc_count += 1
                hc_layers.add(int(rec["layer"]))
                hc_steps.add(int(rec["probe_step"]))
                if not (rec.get("streams_finite") and rec.get("comb_finite")):
                    problems.append(f"stream_probe: non-finite streams/comb at {where}")
                if not rec.get("post_finite"):
                    problems.append(f"stream_probe: non-finite post weights at {where}")
                devs = [
                    abs(float(rec[k]) - 1.0)
                    for k in (
                        "comb_row_sum_min",
                        "comb_row_sum_max",
                        "comb_col_sum_min",
                        "comb_col_sum_max",
                    )
                ]
                comb_dev_max = max(comb_dev_max, *devs)
                if max(devs) > comb_tol:
                    problems.append(
                        f"stream_probe: Sinkhorn comb row/col sum deviates {max(devs):.3g} "
                        f"(> {comb_tol}) from 1.0 at {where}"
                    )
                stream_l2_max = max(stream_l2_max, *(float(x) for x in rec["stream_l2_max"]))
                post_min = min(post_min, float(rec["post_min"]))
                post_max = max(post_max, float(rec["post_max"]))
            elif rec.get("kind") == "pool":
                pool_count += 1
                pool_layers.add(int(rec["layer"]))
                pool_steps.add(int(rec["probe_step"]))
                if not rec.get("all_in_range"):
                    problems.append(f"pool_probe: expanded index outside [0, kv_len) at {where}")
                rows_covered = rec.get("rows_covered") or []
                lens = rec.get("kv_lens") or []
                for row, (covered, kv_len) in enumerate(zip(rows_covered, lens)):
                    if int(kv_len) >= 1 and not covered:
                        problems.append(
                            f"pool_probe: request row {row} (kv_len={kv_len}) selected no "
                            f"visible position at {where}"
                        )
                sentinel_total += sum(int(x) for x in rec.get("sentinel_counts") or [])
                nonsentinel_max = max(
                    nonsentinel_max, *(int(x) for x in rec.get("nonsentinel_max") or [-1])
                )

    missing_hc = sorted(set(hc_layers_expected) - hc_layers)
    if missing_hc:
        problems.append(f"stream_probe: rank files never probed decoder layer(s) {missing_hc}")
    missing_pool = sorted(set(pool_layers_expected) - pool_layers)
    if missing_pool:
        problems.append(f"pool_probe: rank files never probed sparse layer(s) {missing_pool}")
    probed_steps = len(hc_steps)
    if probed_steps < min_probe_steps:
        problems.append(
            f"stream_probe: only {probed_steps} distinct probed steps, periodicity "
            f"contract requires >= {min_probe_steps}"
        )
    if not pool_count:
        problems.append("pool_probe: no pool records at all")

    evidence = {
        "rank_files": sorted(records_by_rank),
        "hc_records": hc_count,
        "pool_records": pool_count,
        "distinct_probed_steps_hc": probed_steps,
        "distinct_probed_steps_pool": len(pool_steps),
        "min_probe_steps_required": min_probe_steps,
        "hc_layers_covered": len(hc_layers),
        "pool_layers_covered": sorted(pool_layers),
        "missing_hc_layers": missing_hc,
        "missing_pool_layers": missing_pool,
        "comb_sum_max_abs_dev_from_1": comb_dev_max,
        "comb_sum_tolerance": comb_tol,
        "stream_l2_max": stream_l2_max,
        "post_min": post_min if post_min != float("inf") else None,
        "post_max": post_max if post_max != float("-inf") else None,
        "sentinel_slots_total": sentinel_total,
        "nonsentinel_index_max": nonsentinel_max,
    }
    return evidence, problems


def read_probe_records(probe_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    records: Dict[str, List[Dict[str, Any]]] = {}
    for fname in sorted(os.listdir(probe_dir)):
        if not (fname.startswith("rank") and fname.endswith(".jsonl")):
            continue
        with open(os.path.join(probe_dir, fname)) as fh:
            records[fname[len("rank") : -len(".jsonl")]] = [
                json.loads(line) for line in fh if line.strip()
            ]
    return records


def slot_reuse_isolation(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[str]]:
    """Within-config isolation: ``canary0_rerun`` must reproduce ``canary0``.

    The rerun decodes through slots/pages that ``canary1`` (a different
    prompt) occupied in between, so bitwise token AND per-step state equality
    means recycled KDA/MLA/pool state carries nothing across requests.
    """
    by_name = {r["name"]: r for r in rows}
    problems: List[str] = []
    first, rerun = by_name.get("canary0"), by_name.get("canary0_rerun")
    if first is None or rerun is None:
        return {"checked": False}, ["slot_reuse: canary0/canary0_rerun rows missing"]
    tokens_equal = first["trt_tokens"] == rerun["trt_tokens"]
    if not tokens_equal:
        fork = next(
            (i for i, (a, b) in enumerate(zip(first["trt_tokens"], rerun["trt_tokens"])) if a != b),
            min(len(first["trt_tokens"]), len(rerun["trt_tokens"])),
        )
        problems.append(f"slot_reuse: canary0_rerun tokens fork from canary0 at step {fork}")
    state_equal = first.get("state_sha256_steps") == rerun.get("state_sha256_steps")
    if not state_equal:
        problems.append("slot_reuse: canary0_rerun per-step state digests differ from canary0")
    evidence = {
        "checked": True,
        "tokens_bitwise_equal": tokens_equal,
        "state_bitwise_equal": state_equal,
        "steps": len(first["trt_tokens"]),
        "state_steps": len(first.get("state_sha256_steps") or []),
    }
    return evidence, problems


def compare_canary_runs(
    baseline: Dict[str, Any], current: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[str]]:
    """Full-window B-vs-E equivalence on tokens, finish reasons, and state."""
    problems: List[str] = []
    if not baseline.get("ok"):
        problems.append("be_equivalence: baseline run has ok=false")
    base_rows = {r["name"]: r for r in baseline.get("rows", [])}
    rows_compared = tokens_steps = state_steps = 0
    mismatches: List[Dict[str, Any]] = []
    for row in current.get("rows", []):
        name = row["name"]
        base = base_rows.get(name)
        if base is None:
            problems.append(f"be_equivalence: row {name} missing from baseline")
            continue
        rows_compared += 1
        if base["trt_tokens"] != row["trt_tokens"]:
            fork = next(
                (
                    i
                    for i, (a, b) in enumerate(zip(base["trt_tokens"], row["trt_tokens"]))
                    if a != b
                ),
                min(len(base["trt_tokens"]), len(row["trt_tokens"])),
            )
            mismatches.append({"row": name, "field": "trt_tokens", "step": fork})
        tokens_steps += len(row["trt_tokens"])
        if base.get("finish_reason") != row.get("finish_reason"):
            mismatches.append({"row": name, "field": "finish_reason", "step": None})
        b_state = base.get("state_sha256_steps") or []
        e_state = row.get("state_sha256_steps") or []
        if len(b_state) != len(e_state):
            problems.append(
                f"be_equivalence: {name} has {len(b_state)} baseline vs {len(e_state)} "
                f"current state steps"
            )
        else:
            state_steps += len(e_state)
            first_bad = next((i for i, (a, b) in enumerate(zip(b_state, e_state)) if a != b), None)
            if first_bad is not None:
                mismatches.append({"row": name, "field": "state_sha256", "step": first_bad})
    if mismatches:
        problems.append(
            f"be_equivalence: {len(mismatches)} B/E mismatches (first: {mismatches[0]})"
        )
    evidence = {
        "baseline_configuration": (baseline.get("config") or {}).get("configuration"),
        "rows_compared": rows_compared,
        "token_steps_compared": tokens_steps,
        "state_steps_compared": state_steps,
        "bitwise_identical": not mismatches and not problems,
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:8],
    }
    return evidence, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/dev/shm/GLM-5.3-Flash")
    parser.add_argument("--pp", type=int, default=4)
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="tensor_parallel_size; the Stage-6 TP4/EP4 terminal canary passes "
        "--tp 4 --pp 1 --ep 4 (Mapping resolution mirrors the serve/replay "
        "drivers via resolve_moe_parallel)",
    )
    parser.add_argument(
        "--ep",
        type=int,
        default=None,
        help="moe_expert_parallel_size; omit for TP-only MoE (moe_ep_size=1)",
    )
    parser.add_argument("--config", choices=["B", "E"], default="B")
    parser.add_argument("--new-tokens", type=int, default=2048)
    parser.add_argument(
        "--canary-ref",
        default="agent-flow/workspace/glm-5.3-flash-bringup/reports/hf_gsm8k_reference.pt",
    )
    parser.add_argument("--summary", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument(
        "--state-digest-dir",
        required=True,
        help="fresh dir for the per-step runtime-state digests (GLM53_STATE_DIGEST_DIR)",
    )
    parser.add_argument(
        "--probe-dir",
        default=None,
        help="fresh dir for the periodic Sinkhorn/pool probe (GLM53_STREAM_PROBE_DIR); "
        "config B only — the eager baseline is where activation probes observe "
        "real decode compute",
    )
    parser.add_argument("--probe-every", type=int, default=128)
    parser.add_argument(
        "--compare-with",
        default=None,
        help="baseline (config B) summary JSON: assert full-window bitwise B/E "
        "token/state equivalence against it",
    )
    args = parser.parse_args()
    if args.probe_dir and args.config != "B":
        parser.error("--probe-dir is a config-B (eager) evidence channel")
    try:
        moe_tp, moe_ep, mapping_label, moe_llm_kwargs = resolve_moe_parallel(
            args.tp, args.pp, args.ep
        )
    except ValueError as exc:
        parser.error(str(exc))

    for opt, path in (
        ("--state-digest-dir", args.state_digest_dir),
        ("--probe-dir", args.probe_dir),
    ):
        if not path:
            continue
        path = os.path.abspath(path)
        os.makedirs(path, exist_ok=True)
        if any(name.endswith(".jsonl") for name in os.listdir(path)):
            parser.error(f"{opt} {path} already contains .jsonl files; use a fresh dir")
    args.state_digest_dir = os.path.abspath(args.state_digest_dir)
    os.environ["GLM53_STATE_DIGEST_DIR"] = args.state_digest_dir
    if args.probe_dir:
        args.probe_dir = os.path.abspath(args.probe_dir)
        os.environ["GLM53_STREAM_PROBE_DIR"] = args.probe_dir
        os.environ["GLM53_STREAM_PROBE_EVERY"] = str(args.probe_every)

    runlog = args.summary + ".runlog.txt"
    tee_output_to(runlog)

    preflight, preflight_problems = disk_preflight(args.summary)
    print(f"[canary] disk preflight: {json.dumps(preflight['measured'])}", flush=True)
    if preflight_problems:
        failed = {
            "config": {"configuration": args.config},
            "disk_preflight": preflight,
            "failure_class": "infrastructure",
            "ok": False,
            "problems": preflight_problems,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.summary)), exist_ok=True)
        with open(args.summary, "w") as fh:
            json.dump(failed, fh, indent=2)
        with open(args.summary + ".exit.txt", "w") as fh:
            fh.write("1\n")
        print(f"[canary] disk preflight FAILED: {preflight_problems}", flush=True)
        return 1

    import torch

    import tensorrt_llm
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.inputs import TokensPrompt
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    assert tensorrt_llm.__file__.startswith(_REPO), (
        f"stale tensorrt_llm package resolved: {tensorrt_llm.__file__}"
    )

    started = time.time()
    enabled = args.config == "E"
    summary: Dict[str, Any] = {
        "config": {
            "model": args.model,
            "tensor_parallel_size": args.tp,
            "pipeline_parallel_size": args.pp,
            "world_size": args.tp * args.pp,
            "moe_tensor_parallel_size": moe_tp,
            "moe_expert_parallel_size": moe_ep,
            "mapping_label": mapping_label,
            "configuration": args.config,
            "cuda_graph": enabled,
            "overlap_scheduler": enabled,
            "new_tokens": args.new_tokens,
            "engine": {
                "max_batch_size": ENGINE_MAX_BATCH_SIZE,
                "max_num_tokens": ENGINE_MAX_NUM_TOKENS,
                "max_seq_len": ENGINE_MAX_SEQ_LEN,
                "kv_enable_block_reuse": False,
                "kv_max_tokens": ENGINE_KV_MAX_TOKENS,
                "geometry_provenance": "Goal-4.2 graded trtllm-serve pair (serving-equivalent)",
            },
            "state_digest_dir": args.state_digest_dir,
            "probe_dir": args.probe_dir,
            "probe_every": args.probe_every if args.probe_dir else None,
            "decode": {
                "temperature": 0,
                "top_k": 1,
                "ignore_eos": True,
                "detokenize": False,
                "stop_semantics": "max_tokens == the window; every row must finish 'length'",
            },
        },
        "package": {
            "tensorrt_llm_version": tensorrt_llm.__version__,
            "tensorrt_llm_file": tensorrt_llm.__file__,
        },
        "disk_preflight": preflight,
        "provenance": {
            "driver_sha256": sha256_of(os.path.abspath(__file__)),
            "modeling_glm5_next_sha256": sha256_of(
                os.path.join(_REPO, "tensorrt_llm/_torch/models/modeling_glm5_next.py")
            ),
        },
        "rows": [],
        "ok": False,
        "problems": ["run did not complete"],
    }

    llm = None
    state_store: Dict[str, Any] = {}
    try:
        prompts = load_canary_prompts(args.canary_ref)[:2]
        # Sequential execution order: prompt A, prompt B, prompt A again. The
        # rerun decodes through slots/pages the middle request recycled.
        plan = [
            (prompts[0]["name"], prompts[0]["input_ids"]),
            (prompts[1]["name"], prompts[1]["input_ids"]),
            (prompts[0]["name"] + "_rerun", prompts[0]["input_ids"]),
        ]
        max_prompt = max(len(ids) for _, ids in plan)
        budget_needed = max_prompt + args.new_tokens
        assert budget_needed + 8 <= ENGINE_MAX_SEQ_LEN, (
            f"serving budget cannot hold the canary: prompt {max_prompt} + "
            f"{args.new_tokens} new tokens vs max_seq_len {ENGINE_MAX_SEQ_LEN}"
        )
        summary["config"]["budget"] = {
            "max_prompt_tokens": max_prompt,
            "new_tokens": args.new_tokens,
            "needed_seq_len": budget_needed,
            "max_seq_len": ENGINE_MAX_SEQ_LEN,
        }

        problems: List[str] = []
        graph_config = CudaGraphConfig() if enabled else None
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tp,
            pipeline_parallel_size=args.pp,
            max_seq_len=ENGINE_MAX_SEQ_LEN,
            max_batch_size=ENGINE_MAX_BATCH_SIZE,
            max_num_tokens=ENGINE_MAX_NUM_TOKENS,
            kv_cache_config=KvCacheConfig(
                enable_block_reuse=False, max_tokens=ENGINE_KV_MAX_TOKENS
            ),
            disable_overlap_scheduler=not enabled,
            cuda_graph_config=graph_config,
            **moe_llm_kwargs,
        )
        summary["load_seconds"] = round(time.time() - started, 1)
        print(f"[canary] engine up in {summary['load_seconds']}s", flush=True)

        sampling = SamplingParams(
            max_tokens=args.new_tokens,
            temperature=0,
            top_k=1,
            ignore_eos=True,
            detokenize=False,
        )
        for name, input_ids in plan:
            t0 = time.time()
            out = llm.generate([TokensPrompt(prompt_token_ids=list(input_ids))], sampling)[0]
            comp = out.outputs[0]
            trt_tokens = [int(t) for t in comp.token_ids]
            row = {
                "name": name,
                "prompt_len": len(input_ids),
                "trt_tokens": trt_tokens,
                "trt_generated": len(trt_tokens),
                "finish_reason": comp.finish_reason,
                "stop_reason": comp.stop_reason,
                "seconds": round(time.time() - t0, 1),
                "tokens_sha256": hashlib.sha256(json.dumps(trt_tokens).encode()).hexdigest(),
            }
            summary["rows"].append(row)
            print(
                f"[canary] {name}: generated {row['trt_generated']}/{args.new_tokens} "
                f"finish={row['finish_reason']} ({row['seconds']}s)",
                flush=True,
            )
            if len(trt_tokens) != args.new_tokens:
                problems.append(
                    f"{name}: generated {len(trt_tokens)} tokens, contract requires "
                    f"exactly {args.new_tokens} (truncation or early stop)"
                )
            if comp.finish_reason != "length":
                problems.append(
                    f"{name}: finish_reason={comp.finish_reason!r} "
                    f"(stop_reason={comp.stop_reason!r}), contract requires 'length'"
                )

        rows_spec = [(name, len(ids), args.new_tokens) for name, ids in plan]
        state_store, state_problems, state_info = merge_state_digests(
            args.state_digest_dir, rows_spec
        )
        problems.extend(state_problems)
        expected_state = sum(steps - 1 for _n, _p, steps in rows_spec)
        attached = 0
        for row in summary["rows"]:
            entries = state_store.get(row["name"]) or []
            shas = [
                hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
                for entry in entries
            ]
            row["state_sha256_steps"] = shas
            row["state_rollup_sha256"] = hashlib.sha256("".join(shas).encode()).hexdigest()
            attached += len(shas)
        state_info["attached_state_steps"] = attached
        state_info["expected_state_steps"] = expected_state
        summary["state_digest"] = state_info
        if attached != expected_state:
            problems.append(
                f"state_digest: attached {attached} per-step state records, contract "
                f"requires exactly {expected_state} (= rows * (new_tokens - 1))"
            )

        iso_evidence, iso_problems = slot_reuse_isolation(summary["rows"])
        summary["slot_reuse_isolation"] = iso_evidence
        problems.extend(iso_problems)

        if args.probe_dir:
            records = read_probe_records(args.probe_dir)
            probe_evidence, probe_problems = validate_probe_records(
                records,
                min_probe_steps=expected_min_probe_steps(
                    len(plan), args.new_tokens, args.probe_every
                ),
            )
            summary["stream_pool_probe"] = probe_evidence
            problems.extend(probe_problems)
            print(
                f"[canary] probe: hc={probe_evidence['hc_records']} "
                f"pool={probe_evidence['pool_records']} "
                f"steps={probe_evidence['distinct_probed_steps_hc']} "
                f"comb_dev={probe_evidence['comb_sum_max_abs_dev_from_1']:.3g}",
                flush=True,
            )

        evidence, hard_path_problems = check_cuda_graph_hard_path(runlog, enabled)
        summary["cuda_graph_hard_path"] = evidence
        problems.extend(hard_path_problems)
        ladder_sizes = expected_graph_batch_sizes(
            graph_config.batch_sizes if graph_config else [],
            engine_max_batch_size=ENGINE_MAX_BATCH_SIZE,
            max_num_tokens=ENGINE_MAX_NUM_TOKENS,
        )
        ladder, ladder_problems = audit_graph_ladder(
            runlog,
            enabled=enabled,
            expected_sizes=ladder_sizes,
            engine_max_batch_size=ENGINE_MAX_BATCH_SIZE,
            # Every MPI rank (tp*pp of them) logs its own [RANK k] capture
            # line; the audit requires ALL of them to have captured the full
            # ladder, so a single TP rank that silently decoded eager is caught.
            pp_size=args.tp * args.pp,
        )
        summary["graph_ladder"] = ladder
        problems.extend(ladder_problems)

        if args.compare_with:
            with open(args.compare_with) as fh:
                baseline = json.load(fh)
            be_evidence, be_problems = compare_canary_runs(baseline, summary)
            summary["be_equivalence"] = be_evidence
            problems.extend(be_problems)
            print(
                f"[canary] be_equivalence: rows={be_evidence['rows_compared']} "
                f"tokens={be_evidence['token_steps_compared']} "
                f"state={be_evidence['state_steps_compared']} "
                f"bitwise_identical={be_evidence['bitwise_identical']}",
                flush=True,
            )

        summary["problems"] = problems
        summary["ok"] = not problems
        print(f"[canary] ok={summary['ok']} problems={len(problems)}", flush=True)
    except BaseException as exc:  # noqa: BLE001 — recorded, then exit code
        summary["ok"] = False
        summary["error"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[
            -4000:
        ]
        summary["problems"] = summary.get("problems") or []
        summary["problems"].append(f"exception: {type(exc).__name__}: {exc}")
        print(f"[canary] FAILED: {type(exc).__name__}: {exc}", flush=True)
    finally:
        summary["total_seconds"] = round(time.time() - started, 1)
        torch.save({"state_store": state_store}, args.metrics_out)
        with open(args.summary, "w") as fh:
            json.dump(summary, fh, indent=2)
        code = 0 if summary.get("ok") else 1
        with open(args.summary + ".exit.txt", "w") as fh:
            fh.write(f"{code}\n")
        print(f"[canary] wrote {args.summary} (exit {code})", flush=True)
        if llm is not None:
            llm.shutdown()
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
