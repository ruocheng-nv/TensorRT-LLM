# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""source_logit_replay + generation_parity on the REAL LLM API runtime path.

Stage-2 criterion 2 (feedback-aligned diagnostic parity) needs, on the
implemented TensorRT-LLM path (normal LLM API, PP-sharded real checkpoint,
KVCacheManagerV2):

- >=5 fixed real prompts from the implementation-independent frozen manifest,
  deterministic greedy, per-step logit metrics (max_abs / mean_abs / cosine)
  on the HF-comparable prefix, finite logits over the FULL window
  (source_logit_replay), and
- 2x512 + 5x32 generation windows retained COMPLETELY: every generated step
  keeps TRT-side evidence (token, fp32-logit sha256, stats, top-8 ids/values,
  full-vocab finiteness) so config B can be compared directly with config E
  across the whole window (generation_parity + B/E deterministic equivalence).

Exact native-HF/TRT token equality is DIAGNOSTIC ONLY (human acceptance
override 2026-09-03T09:55:48): HF/TRT forks are recorded honestly in
``hf_fork_diagnostics`` and per-row fork records but do not fail the run.
Past the first fork the HF trajectory continues from a different prefix, so
HF-vs-TRT logits stop being comparable there — the driver stops emitting HF
metrics at that point but KEEPS retaining per-step TRT evidence to the end of
the window. The pass conditions are: reference-window contract, exact token
counts, finish_reason=='length', full-window finiteness, complete retention
(``--expect-total-steps``), CUDA-graph hard-path evidence, and (for the E
run, via ``--compare-with``) full-window bitwise B/E equivalence.

The HF side is NOT re-run: this driver replays against the stored native-HF
references produced earlier in the ladder (their per-step logits are on
disk), so tokenizer/chat-template/thinking-mode identity is by construction —
the stored ``input_ids`` are fed to TensorRT-LLM verbatim as TokensPrompt.

Decoding contract (recorded in the JSON):
- explicit greedy: ``temperature=0`` AND ``top_k=1`` (no sampling);
- identical stop semantics on both sides: a reference row is REJECTED unless
  it stores >= the required steps (so HF did not early-stop inside the
  window), and TensorRT-LLM runs with ``ignore_eos=True`` + ``max_tokens`` =
  the required steps, so every row must finish with ``finish_reason ==
  'length'`` after exactly that many steps. No silent ``min()`` shortening:
  a too-short reference or a short TRT generation is a hard failure.

CUDA-graph hard path (config E): the driver forces ``TLLM_LOG_LEVEL=INFO``
and tees its own stdout/stderr (which the MPI-spawned PP workers inherit) to
``<summary>.runlog.txt``. After generation it greps that log and asserts
- >=1 "Running CUDA graph capture for ..." worker line (real capture), and
- zero decode-graph fallback warnings ("falling back to eager ..."),
and records the matched lines verbatim in the JSON. Decode batches here are
single-request, decode-only (bs=1), i.e. graph-eligible shapes by
construction, so capture + no-fallback means the decode steps replayed the
graph. For config B it asserts the log shows NO capture lines (true baseline).

References:
- reports/hf_gsm8k_reference.pt      -> ``canary``: 2 rows x 512 steps
- reports/goal1.5-logs/hf_badprompts.pt -> ``rows``: 4 gsm8k rows (16/27/73/85)

Every exit path writes the JSON summary and ``<summary>.exit.txt``. Compact
per-step metrics (not full logits) are saved to ``--metrics-out`` (.pt).

Run (B config, all eight GPUs free):
    python tests/unittest/_torch/modeling/glm5_next_llm_api_logit_replay.py \
        --pp 8 --config B \
        --summary reports/goal1.5-logs/llm_api_logit_replay_b.json \
        --metrics-out reports/goal1.5-logs/llm_api_logit_replay_b_metrics.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 4))
sys.path.insert(0, _REPO)
os.environ["PYTHONPATH"] = _REPO + os.pathsep + os.environ.get("PYTHONPATH", "")
# INFO must be on before tensorrt_llm imports so worker ranks inherit it: the
# CUDA-graph capture evidence is an INFO log line from each worker's engine.
os.environ["TLLM_LOG_LEVEL"] = "INFO"

from glm5_next_driver_preflight import (  # noqa: E402  (script-dir import)
    audit_graph_ladder,
    disk_preflight,
    expected_graph_batch_sizes,
)

CAPTURE_MARK = "Running CUDA graph capture for"
FALLBACK_MARKS = (
    "falling back to eager",
    "Failed to allocate the CUDA graph padding dummy",
)


def tee_output_to(path: str) -> None:
    """Duplicate this process's stdout+stderr into ``path`` via tee.

    The MPI-spawned PP worker ranks inherit these fds, so their engine logs
    (including the CUDA-graph capture INFO lines) land in the same file, which
    the driver reads back for hard-path evidence.
    """
    tee = subprocess.Popen(["tee", path], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), 1)
    os.dup2(tee.stdin.fileno(), 2)


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_reference_rows(canary_path: str, badprompts_path: str):
    """Six fixed prompts with stored native-HF per-step fp32 logits."""
    import torch

    rows = []
    canary = torch.load(canary_path, map_location="cpu", weights_only=False)
    for r in canary["canary"]:
        rows.append(
            {
                "name": f"canary{r['index']}",
                "input_ids": [int(x) for x in r["input_ids"]],
                "hf_tokens": [int(x) for x in r["generated_token_ids"]],
                "hf_step_logits": r["generated_step_logits"],  # [steps, vocab] fp32
                "role": "canary512",
            }
        )
    bad = torch.load(badprompts_path, map_location="cpu", weights_only=False)
    for r in bad["rows"]:
        # gsm8k fixtures carry dataset rows keyed by sample index; the margin-
        # screened parity fixtures carry raw prompts. Name honestly by source.
        prefix = "short" if "prompt" in r else "gsm8k"
        rows.append(
            {
                "name": f"{prefix}{r['index']}",
                "input_ids": [int(x) for x in r["input_ids"]],
                "hf_tokens": [int(x) for x in r["generated_token_ids"]],
                "hf_step_logits": r["generated_step_logits"].float(),
                "role": "short32",
            }
        )
    return rows


def compare_row(row, trt_tokens, trt_logits, steps):
    """Full-window per-step retention + HF comparison on the shared prefix.

    EVERY generated step keeps TRT-side evidence — token, argmax, sha256 of
    the fp32 logit bytes (the bitwise B-vs-E comparator), max/mean, top-8
    ids/values, full-vocab finiteness — so the two TRT configurations can be
    compared directly across the whole window. HF-vs-TRT metrics are
    additionally recorded while the prefixes agree (up to and including the
    first fork step, where both sides still saw the identical prefix); past
    the fork the HF trajectory continues from a different prefix and its
    logits stop being comparable, so those steps carry ``hf_comparable=False``
    and no HF metric — but retention does NOT stop there.
    """
    import torch

    hf_tokens = row["hf_tokens"][:steps]
    hf_logits = row["hf_step_logits"]
    per_step = []
    first_divergence = None
    fork = None
    for i in range(min(steps, len(trt_tokens))):
        trt_l = trt_logits[i].float()
        top8 = trt_l.topk(8)
        rec = {
            "step": i,
            "trt_token": int(trt_tokens[i]),
            "trt_argmax": int(trt_l.argmax()),
            "logit_sha256": hashlib.sha256(trt_l.contiguous().numpy().tobytes()).hexdigest(),
            "trt_max": float(trt_l.max()),
            "trt_mean": float(trt_l.mean()),
            "trt_top8_ids": [int(x) for x in top8.indices],
            "trt_top8_values": [float(x) for x in top8.values],
            "finite": bool(torch.isfinite(trt_l).all()),
            "hf_comparable": first_divergence is None,
        }
        if first_divergence is None:
            hf_l = hf_logits[i].float()
            # The stored HF references are generate()-style processed scores
            # and carry -inf at a few suppressed token ids (1536 = 3 ids x 512
            # steps on the canary rows). Metrics compare only the reference-
            # finite vocab entries, else max_abs=inf / cosine=nan misreads as
            # TRT corruption; the TRT finiteness check above stays full-vocab.
            ref_finite = torch.isfinite(hf_l)
            d = (trt_l[ref_finite] - hf_l[ref_finite]).abs()
            cos = torch.nn.functional.cosine_similarity(
                hf_l[ref_finite].reshape(1, -1).double(), trt_l[ref_finite].reshape(1, -1).double()
            )
            # Drift restricted to the reference's top-8 tokens: the argmax-
            # relevant slice of the vocabulary, and the quantity the manifest's
            # predeclared per_step_top8_max_abs bound is evaluated on.
            top8_idx = hf_l.masked_fill(~ref_finite, float("-inf")).topk(8).indices
            rec.update(
                {
                    "token_match": int(trt_tokens[i]) == int(hf_tokens[i]),
                    "max_abs": float(d.max()),
                    "mean_abs": float(d.mean()),
                    "cosine": float(cos),
                    "top8_max_abs": float((trt_l[top8_idx] - hf_l[top8_idx]).abs().max()),
                    "ref_masked_entries": int((~ref_finite).sum()),
                }
            )
            if int(trt_tokens[i]) != int(hf_tokens[i]):
                first_divergence = i
                ht, tt = int(hf_tokens[i]), int(trt_tokens[i])
                fork = {
                    "step": i,
                    "hf_token": ht,
                    "trt_token": tt,
                    "hf_logit_at_hf": float(hf_l[ht]),
                    "hf_logit_at_trt": float(hf_l[tt]),
                    "trt_logit_at_hf": float(trt_l[ht]),
                    "trt_logit_at_trt": float(trt_l[tt]),
                    "hf_separation": float(hf_l[ht] - hf_l[tt]),
                    "trt_separation": float(trt_l[tt] - trt_l[ht]),
                }
        per_step.append(rec)
    hf_steps = [s for s in per_step if s["hf_comparable"]]
    matched = sum(1 for s in hf_steps if s["token_match"])
    return {
        "name": row["name"],
        "role": row["role"],
        # steps_compared keeps its historical meaning: HF-comparable prefix.
        "steps_compared": len(hf_steps),
        "hf_steps_compared": len(hf_steps),
        "steps_retained": len(per_step),
        "steps_requested": steps,
        "trt_generated": len(trt_tokens),
        "trt_tokens": [int(t) for t in trt_tokens],
        "matched": matched,
        "exact": first_divergence is None and len(hf_steps) == steps,
        "first_divergence": first_divergence,
        "fork": fork,
        # Finiteness is a FULL-WINDOW contract; HF metric aggregates cover the
        # HF-comparable prefix only (no HF reference exists past the fork).
        "all_finite": all(s["finite"] for s in per_step),
        "max_abs_max": max((s["max_abs"] for s in hf_steps), default=None),
        "cosine_min": min((s["cosine"] for s in hf_steps), default=None),
        "top8_max_abs_max": max((s["top8_max_abs"] for s in hf_steps), default=None),
        "cosine_min_first_64": min((s["cosine"] for s in hf_steps[:64]), default=None),
    }, per_step


def check_cuda_graph_hard_path(runlog_path: str, enabled: bool):
    """Grep the run log for real capture / fallback evidence.

    Returns (evidence_dict, problems). Decode batches in this driver are
    single-request decode-only (bs=1) — graph-eligible by construction — so
    "capture happened" + "no eager fallback" means decode steps replayed the
    captured graph. Config B must show zero capture lines.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    time.sleep(2)  # let tee drain
    capture_lines, fallback_lines = [], []
    with open(runlog_path, "r", errors="replace") as fh:
        for line in fh:
            if CAPTURE_MARK in line:
                capture_lines.append(line.rstrip()[-240:])
            if any(m in line for m in FALLBACK_MARKS):
                fallback_lines.append(line.rstrip()[-240:])
    evidence = {
        "runlog": runlog_path,
        "capture_line_count": len(capture_lines),
        "capture_lines": capture_lines[:16],
        "fallback_line_count": len(fallback_lines),
        "fallback_lines": fallback_lines[:16],
        "decode_batch_shape": "single-request decode-only (bs=1), graph-eligible",
    }
    problems = []
    if enabled:
        if not capture_lines:
            problems.append(
                "cuda_graph_hard_path: no 'Running CUDA graph capture' line in the "
                "run log — capture did not happen"
            )
        if fallback_lines:
            problems.append(
                f"cuda_graph_hard_path: {len(fallback_lines)} eager-fallback warnings in the run log"
            )
    else:
        if capture_lines:
            problems.append("baseline config B unexpectedly captured CUDA graphs")
    return evidence, problems


def merge_state_digests(digest_dir, rows_spec):
    """Merge per-rank runtime-state digest JSONLs into per-row decode records.

    ``rows_spec`` is ``[(row_name, prompt_len, steps)]`` in execution order.
    A row of ``steps`` greedy tokens performs ONE prefill forward — which
    already produces token 1 from the prompt — and ``steps - 1`` decode
    forwards; ``prepare()`` writes one digest record per decode forward,
    carrying ``cached == [prompt_len + k - 1]`` before decode k. A row's
    state series is therefore exactly ``steps - 1`` records whose cached
    values run ``prompt_len .. prompt_len + steps - 2``: the input state of
    every decode step. Token 1 has no decode-state input by construction (it
    is sampled from prefill logits over a just-written cache), and the final
    decode's cache write feeds no later step.

    Rows ran sequentially, one engine request each, so records are grouped by
    request id and groups are matched to rows in order of first appearance; a
    group must reproduce a row's exact cached trace to be assigned (engine
    warmup dummies produce short ``cached == 0`` traces that match no row).
    Returns (state_store, problems, info): ``state_store[name][j]`` maps
    ``rank<k>/<layer>`` to the digest of the state consumed by decode j+1.
    """
    problems = []
    rank_files = sorted(
        f for f in os.listdir(digest_dir) if f.startswith("rank") and f.endswith(".jsonl")
    )
    if not rank_files:
        return {}, [f"state_digest: no rank*.jsonl files in {digest_dir}"], {}

    per_rank_rows = {}
    trailing_by_rank = {}
    for fname in rank_files:
        rank = fname[len("rank") : -len(".jsonl")]
        with open(os.path.join(digest_dir, fname)) as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        groups = {}
        order = []
        for rec in records:
            if rec.get("batch") != 1 or rec.get("num_contexts") != 0 or rec.get("seq_lens") != [1]:
                continue
            rid = tuple(rec.get("request_ids") or ())
            if rid not in groups:
                groups[rid] = []
                order.append(rid)
            groups[rid].append(rec)
        assigned = {}
        trailing = 0
        gi = 0
        for name, plen, steps in rows_spec:
            want = [plen + t for t in range(steps - 1)]
            found = None
            while gi < len(order):
                cand = groups[order[gi]]
                gi += 1
                trace = [r["cached"][0] for r in cand]
                if trace == want:
                    found = cand
                    break
                # An overlap-pipelined engine may issue one extra trailing
                # prepare after the final decode; keep only the shared window
                # so baseline and enabled runs stay step-symmetric, and count
                # the trailing record rather than dropping it silently.
                if trace == want + [plen + steps - 1]:
                    found = cand[: steps - 1]
                    trailing += 1
                    break
            if found is None:
                problems.append(
                    f"state_digest: rank {rank}: no request trace matches {name} "
                    f"(prompt_len={plen}, expected {steps - 1} decode records with "
                    f"cached {plen}..{plen + steps - 2})"
                )
                continue
            assigned[name] = found
        per_rank_rows[rank] = assigned
        trailing_by_rank[rank] = trailing

    state_store = {}
    total = 0
    for name, _plen, steps in rows_spec:
        combined = []
        for j in range(steps - 1):
            entry = {}
            for rank in sorted(per_rank_rows):
                row_records = per_rank_rows[rank].get(name)
                if row_records is None:
                    continue
                for lkey, lval in row_records[j]["layers"].items():
                    entry[f"rank{rank}/{lkey}"] = lval
            if not entry:
                problems.append(f"state_digest: {name} decode {j + 1}: no rank contributed state")
            for lkey, lval in entry.items():
                blob = json.dumps(lval)
                if '"error"' in blob:
                    problems.append(f"state_digest: {name} decode {j + 1} {lkey}: {blob[:160]}")
            combined.append(entry)
        state_store[name] = combined
        total += len(combined)
    info = {
        "dir": digest_dir,
        "rank_files": len(rank_files),
        "rows": {name: len(state_store.get(name, [])) for name, _p, _s in rows_spec},
        "total_state_steps": total,
        "trailing_post_final_records_by_rank": trailing_by_rank,
        "state_semantics": (
            "one record per decode forward = the input state that decode step "
            "consumes; steps-1 records per row because token 1 is produced by "
            "the prefill forward and has no decode-state input"
        ),
    }
    return state_store, problems, info


BE_STEP_FIELDS = (
    "trt_token",
    "trt_argmax",
    "logit_sha256",
    "trt_max",
    "trt_mean",
    "trt_top8_ids",
    "trt_top8_values",
    "finite",
    "hf_comparable",
    "token_match",
    "max_abs",
    "mean_abs",
    "cosine",
    "top8_max_abs",
    "ref_masked_entries",
)


def compare_with_baseline(b_summary_path, b_metrics_path, summary, per_step_store, expect_total):
    """Full-window B-vs-E deterministic-equivalence check.

    Loads the baseline (config B) summary JSON + per-step metrics archive and
    compares them with THIS run's rows/metrics on every retained step: full
    token sequences, fork records, per-step fp32-logit sha256 (bitwise proof)
    and every retained metric/state field. Returns (evidence_dict, problems).
    """
    import torch

    problems = []
    with open(b_summary_path) as fh:
        base = json.load(fh)
    base_metrics = torch.load(b_metrics_path, map_location="cpu", weights_only=False)
    if not base.get("ok"):
        problems.append(f"be_equivalence: baseline run {b_summary_path} has ok=false")
    base_rows = {r["name"]: r for r in base.get("rows", [])}
    mismatches = []
    steps_checked = 0
    rows_checked = 0
    state_steps_checked = 0
    baseline_state_steps = 0
    current_state_steps = 0
    for r in summary["rows"]:
        name = r["name"]
        br = base_rows.get(name)
        if br is None:
            problems.append(f"be_equivalence: row {name} missing from baseline summary")
            continue
        rows_checked += 1
        if br.get("trt_tokens") != r.get("trt_tokens"):
            bt, et = br.get("trt_tokens") or [], r.get("trt_tokens") or []
            first = next(
                (i for i in range(min(len(bt), len(et))) if bt[i] != et[i]),
                min(len(bt), len(et)),
            )
            mismatches.append({"row": name, "step": first, "field": "trt_tokens"})
        for field in ("first_divergence", "fork", "finish_reason"):
            if br.get(field) != r.get(field):
                mismatches.append({"row": name, "step": None, "field": field})
        bp = base_metrics.get(name)
        ep = per_step_store.get(name)
        if bp is None or ep is None or len(bp) != len(ep):
            problems.append(
                f"be_equivalence: {name} retains {len(bp) if bp else 0} baseline vs "
                f"{len(ep) if ep else 0} current per-step records"
            )
            continue
        for i, (bs, es) in enumerate(zip(bp, ep)):
            steps_checked += 1
            for field in BE_STEP_FIELDS:
                if bs.get(field) != es.get(field):
                    mismatches.append({"row": name, "step": i, "field": field})
            # Runtime-state digests are compared only when both sides carry
            # them: a state-digest run checked against a digest-free clean
            # baseline still proves tokens/logits/metrics equality (i.e. that
            # digest mode did not perturb compute) without false mismatches.
            if "state_sha256" in bs:
                baseline_state_steps += 1
            if "state_sha256" in es:
                current_state_steps += 1
            if "state_sha256" in bs and "state_sha256" in es:
                state_steps_checked += 1
                if bs["state_sha256"] != es["state_sha256"]:
                    mismatches.append({"row": name, "step": i, "field": "state_sha256"})
                elif bs.get("state") != es.get("state"):
                    mismatches.append({"row": name, "step": i, "field": "state"})
    if mismatches:
        problems.append(
            f"be_equivalence: {len(mismatches)} B/E mismatches (first: {mismatches[0]})"
        )
    if expect_total is not None and steps_checked != expect_total:
        problems.append(
            f"be_equivalence: compared {steps_checked} per-step records, "
            f"contract requires exactly {expect_total}"
        )
    # Coverage rule fires only when the BASELINE carries state: comparing a
    # state-digest run against the digest-free clean baseline is a legitimate
    # non-perturbation check (tokens/logits/metrics only), not a coverage gap.
    if baseline_state_steps and not (
        baseline_state_steps == current_state_steps == state_steps_checked
    ):
        problems.append(
            f"be_equivalence: runtime-state coverage mismatch — baseline carries "
            f"{baseline_state_steps} state steps, current run {current_state_steps}, "
            f"jointly compared {state_steps_checked}; a state-carrying baseline "
            f"requires the current run to cover the identical step positions"
        )
    evidence = {
        "baseline_summary": b_summary_path,
        "baseline_metrics": b_metrics_path,
        "baseline_configuration": base.get("config", {}).get("configuration"),
        "rows_compared": rows_checked,
        "steps_compared": steps_checked,
        "expected_steps": expect_total,
        "bitwise_identical": not mismatches,
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:8],
        "fields_compared": list(BE_STEP_FIELDS),
        "baseline_state_steps": baseline_state_steps,
        "current_state_steps": current_state_steps,
        "state_steps_compared": state_steps_checked,
        "state_fields_compared": ["state_sha256", "state"],
    }
    return evidence, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/dev/shm/GLM-5.3-Flash")
    parser.add_argument("--pp", type=int, default=8)
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="tensor_parallel_size. Stage-5 TP4 real-runtime replay passes "
        "--tp 4 --pp 1; the world size (tp*pp) is what the per-rank CUDA-graph "
        "ladder audit requires every rank to have captured.",
    )
    parser.add_argument("--config", choices=["B", "E"], default="B")
    parser.add_argument("--short-steps", type=int, default=32)
    parser.add_argument("--canary-steps", type=int, default=512)
    parser.add_argument(
        "--canary-ref",
        default="agent-flow/workspace/glm-5.3-flash-bringup/reports/hf_gsm8k_reference.pt",
    )
    parser.add_argument(
        "--badprompts-ref",
        default="agent-flow/workspace/glm-5.3-flash-bringup/reports/goal1.5-logs/hf_badprompts.pt",
    )
    parser.add_argument("--summary", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument(
        "--expect-total-steps",
        type=int,
        default=None,
        help="hard-assert the retained full-window evidence covers exactly "
        "this many steps summed over all rows (2x512 + 5x32 manifest = 1184)",
    )
    parser.add_argument(
        "--compare-with",
        default=None,
        help="baseline (config B) summary JSON: assert full-window bitwise "
        "B/E deterministic equivalence against it",
    )
    parser.add_argument(
        "--compare-with-metrics",
        default=None,
        help="baseline (config B) per-step metrics .pt (required with --compare-with)",
    )
    parser.add_argument(
        "--state-digest-dir",
        default=None,
        help="fresh directory for per-step runtime-state digests: exports "
        "GLM53_STATE_DIGEST_DIR so every PP worker appends rank<k>.jsonl "
        "records (KDA conv/ssm slots + sparse latent/index pages) at each "
        "decode prepare(); merged post-run into the per-step metrics",
    )
    parser.add_argument(
        "--serve-geometry",
        action="store_true",
        help="use the Goal-4.2 graded trtllm-serve engine geometry "
        "(max_batch_size=4, max_seq_len=4096) instead of this driver's "
        "historical max_batch_size=2 / fitted max_seq_len — the Stage-4 "
        "serving-equivalent diagnostics run with this flag so the engine "
        "caps match the graded serve pair exactly",
    )
    args = parser.parse_args()
    if bool(args.compare_with) != bool(args.compare_with_metrics):
        parser.error("--compare-with and --compare-with-metrics must be given together")
    if args.state_digest_dir:
        sd = os.path.abspath(args.state_digest_dir)
        os.makedirs(sd, exist_ok=True)
        if any(name.endswith(".jsonl") for name in os.listdir(sd)):
            parser.error(f"--state-digest-dir {sd} already contains .jsonl files; use a fresh dir")
        # Must be in the environment BEFORE the LLM() constructor spawns the
        # MPI worker ranks; they inherit it exactly like TLLM_LOG_LEVEL.
        os.environ["GLM53_STATE_DIGEST_DIR"] = sd
        args.state_digest_dir = sd

    runlog = args.summary + ".runlog.txt"
    tee_output_to(runlog)

    # Infrastructure gate BEFORE the engine exists: a full overlay kills a
    # long replay mid-session with Errno 28 (observed at QA iteration 34) at
    # a point where all retained evidence is lost.
    preflight, preflight_problems = disk_preflight(args.summary)
    print(f"[replay] disk preflight: {json.dumps(preflight['measured'])}", flush=True)
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
        print(f"[replay] disk preflight FAILED: {preflight_problems}", flush=True)
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
    summary = {
        "config": {
            "model": args.model,
            "tensor_parallel_size": args.tp,
            "pipeline_parallel_size": args.pp,
            "world_size": args.tp * args.pp,
            "configuration": args.config,
            "cuda_graph": enabled,
            "overlap_scheduler": enabled,
            "short_steps": args.short_steps,
            "canary_steps": args.canary_steps,
            "state_digest_dir": args.state_digest_dir,
            "decode": {
                "temperature": 0,
                "top_k": 1,
                "ignore_eos": True,
                "detokenize": False,
                "stop_semantics": (
                    "identical on both sides: reference rows must store >= the "
                    "required steps (HF did not stop inside the window) and TRT "
                    "must finish_reason=='length' after exactly that many steps"
                ),
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
            "hf_weight_loader_sha256": sha256_of(
                os.path.join(_REPO, "tensorrt_llm/_torch/models/checkpoints/hf/weight_loader.py")
            ),
        },
        "hf_exactness_policy": (
            "diagnostic-only per human acceptance override 2026-09-03T09:55:48: "
            "HF/TRT token forks are recorded in hf_fork_diagnostics and per-row "
            "fork records but are not pass conditions"
        ),
        "rows": [],
        "hf_fork_diagnostics": [],
        "ok": False,
        "problems": ["run did not complete"],
    }

    llm = None
    per_step_store = {}
    try:
        rows = load_reference_rows(args.canary_ref, args.badprompts_ref)
        assert len(rows) >= 6, f"expected >=6 reference rows, found {len(rows)}"
        n_canary = sum(1 for r in rows if r["role"] == "canary512")
        n_short = sum(1 for r in rows if r["role"] == "short32")
        assert n_canary >= 2, f"generation_parity needs >=2 canary rows, found {n_canary}"
        assert n_short + n_canary >= 5, "source_logit_replay needs >=5 prompts"

        problems = []
        # Reference-length contract: NO silent min(). A row whose stored HF
        # trajectory is shorter than the required window is a hard failure —
        # it would otherwise weaken the 32/512-step criteria invisibly.
        for row in rows:
            need = args.canary_steps if row["role"] == "canary512" else args.short_steps
            if len(row["hf_tokens"]) < need or len(row["hf_step_logits"]) < need:
                problems.append(
                    f"{row['name']}: reference too short for the contract — "
                    f"tokens={len(row['hf_tokens'])}, logits={len(row['hf_step_logits'])}, "
                    f"required={need}"
                )
        if problems:
            summary["problems"] = problems
            raise AssertionError("; ".join(problems))

        max_prompt = max(len(r["input_ids"]) for r in rows)
        need_seq = max_prompt + args.canary_steps + 64

        engine_max_batch_size = 4 if args.serve_geometry else 2
        engine_max_seq_len = 4096 if args.serve_geometry else max(2048, need_seq)
        assert need_seq <= engine_max_seq_len, (
            f"reference windows need {need_seq} sequence tokens, engine cap is {engine_max_seq_len}"
        )
        summary["config"]["engine"] = {
            "max_batch_size": engine_max_batch_size,
            "max_num_tokens": 4096,
            "max_seq_len": engine_max_seq_len,
            "kv_enable_block_reuse": False,
            "kv_max_tokens": 16384,
            "serve_geometry": bool(args.serve_geometry),
        }
        graph_config = CudaGraphConfig() if enabled else None
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tp,
            pipeline_parallel_size=args.pp,
            max_seq_len=engine_max_seq_len,
            max_batch_size=engine_max_batch_size,
            max_num_tokens=4096,
            kv_cache_config=KvCacheConfig(enable_block_reuse=False, max_tokens=16384),
            gather_generation_logits=True,
            disable_overlap_scheduler=not enabled,
            cuda_graph_config=graph_config,
        )
        summary["load_seconds"] = round(time.time() - started, 1)
        print(f"[replay] engine up in {summary['load_seconds']}s", flush=True)

        for row in rows:
            steps = args.canary_steps if row["role"] == "canary512" else args.short_steps
            sampling = SamplingParams(
                max_tokens=steps,
                temperature=0,
                top_k=1,
                ignore_eos=True,
                detokenize=False,
                return_generation_logits=True,
            )
            t0 = time.time()
            out = llm.generate([TokensPrompt(prompt_token_ids=list(row["input_ids"]))], sampling)[0]
            comp = out.outputs[0]
            trt_tokens = list(comp.token_ids)
            gen_logits = comp.generation_logits
            assert gen_logits is not None, "generation_logits not returned"
            gen_logits = gen_logits.squeeze(0) if gen_logits.dim() == 3 else gen_logits
            result, per_step = compare_row(row, trt_tokens, gen_logits.cpu(), steps)
            result["seconds"] = round(time.time() - t0, 1)
            result["finish_reason"] = comp.finish_reason
            result["stop_reason"] = comp.stop_reason
            summary["rows"].append(result)
            per_step_store[row["name"]] = per_step
            print(
                f"[replay] {row['name']}: {result['matched']}/{result['hf_steps_compared']} "
                f"hf-prefix matched, retained={result['steps_retained']}/{steps}, "
                f"exact={result['exact']} first_div={result['first_divergence']} "
                f"max_abs={result['max_abs_max']} cos_min={result['cosine_min']} "
                f"finish={result['finish_reason']} ({result['seconds']}s)",
                flush=True,
            )
            # Stop-semantics contract: ignore_eos + max_tokens=steps means the
            # ONLY legitimate finish is 'length' after exactly `steps` tokens.
            if len(trt_tokens) != steps:
                problems.append(
                    f"{row['name']}: TRT generated {len(trt_tokens)} tokens, contract "
                    f"requires exactly {steps}"
                )
            if comp.finish_reason != "length":
                problems.append(
                    f"{row['name']}: finish_reason={comp.finish_reason!r} (stop_reason="
                    f"{comp.stop_reason!r}), contract requires 'length'"
                )
            if not result["all_finite"]:
                problems.append(f"{row['name']}: non-finite logits in the retained window")
            if result["steps_retained"] != steps:
                problems.append(
                    f"{row['name']}: retained {result['steps_retained']} per-step records, "
                    f"contract requires exactly {steps}"
                )
            if not result["exact"]:
                fork = result["fork"] or {}
                summary["hf_fork_diagnostics"].append(
                    f"{row['name']}: HF/TRT fork at step {result['first_divergence']} "
                    f"(hf_sep={fork.get('hf_separation')}, trt_sep={fork.get('trt_separation')}) "
                    f"— diagnostic only per the acceptance override; TRT retention "
                    f"continued through step {result['steps_retained'] - 1}"
                )

        if args.state_digest_dir:
            rows_spec = [
                (
                    row["name"],
                    len(row["input_ids"]),
                    args.canary_steps if row["role"] == "canary512" else args.short_steps,
                )
                for row in rows
            ]
            state_store, state_problems, state_info = merge_state_digests(
                args.state_digest_dir, rows_spec
            )
            problems.extend(state_problems)
            # steps-1 state records per row: token 1 (step index 0) is
            # prefill-produced and has no decode-state input; state record j
            # is the input state of the decode that emits step index j+1.
            expected_state_steps = sum(steps_needed - 1 for _n, _p, steps_needed in rows_spec)
            state_total = 0
            for name, _plen, steps_needed in rows_spec:
                step_states = state_store.get(name) or []
                per_step = per_step_store.get(name) or []
                if len(step_states) != steps_needed - 1 or len(per_step) != steps_needed:
                    problems.append(
                        f"state_digest: {name} has {len(step_states)} state records for "
                        f"{len(per_step)} retained steps; contract requires exactly "
                        f"{steps_needed - 1} (= steps-1, see state_semantics)"
                    )
                for j, entry in enumerate(step_states):
                    if j + 1 >= len(per_step):
                        break
                    rec = per_step[j + 1]
                    rec["state"] = entry
                    rec["state_sha256"] = hashlib.sha256(
                        json.dumps(entry, sort_keys=True).encode()
                    ).hexdigest()
                    state_total += 1
            state_info["attached_state_steps"] = state_total
            state_info["expected_state_steps"] = expected_state_steps
            state_info["row_state_sha256"] = {
                name: hashlib.sha256(
                    "".join(
                        rec.get("state_sha256", "") for rec in (per_step_store.get(name) or [])
                    ).encode()
                ).hexdigest()
                for name, _p, _s in rows_spec
            }
            summary["state_digest"] = state_info
            if state_total != expected_state_steps:
                problems.append(
                    f"state_digest: attached {state_total} per-step state records, "
                    f"contract requires exactly {expected_state_steps} (sum of steps-1 "
                    f"over all rows)"
                )
            print(
                f"[replay] state_digest: ranks={state_info['rank_files']} "
                f"attached={state_total} problems={len(state_problems)}",
                flush=True,
            )

        totals = {
            "steps_requested": sum(r["steps_requested"] for r in summary["rows"]),
            "steps_generated": sum(r["trt_generated"] for r in summary["rows"]),
            "steps_retained": sum(r["steps_retained"] for r in summary["rows"]),
            "hf_steps_compared": sum(r["hf_steps_compared"] for r in summary["rows"]),
        }
        summary["totals"] = totals
        if totals["steps_retained"] != totals["steps_requested"]:
            problems.append(
                f"retention shortfall: {totals['steps_retained']} per-step records "
                f"retained vs {totals['steps_requested']} requested"
            )
        if args.expect_total_steps is not None:
            for key in ("steps_generated", "steps_retained"):
                if totals[key] != args.expect_total_steps:
                    problems.append(
                        f"totals.{key}={totals[key]}, contract requires exactly "
                        f"{args.expect_total_steps}"
                    )

        evidence, hard_path_problems = check_cuda_graph_hard_path(runlog, enabled)
        summary["cuda_graph_hard_path"] = evidence
        problems.extend(hard_path_problems)
        # Silent-eager rejection: assert the captured ladder covers every
        # decode-only batch size the engine can schedule (1..max_batch_size)
        # — an uncovered size would run eager with NO warning line.
        ladder_sizes = expected_graph_batch_sizes(
            graph_config.batch_sizes if graph_config else [],
            engine_max_batch_size=engine_max_batch_size,
            max_num_tokens=4096,
        )
        ladder, ladder_problems = audit_graph_ladder(
            runlog,
            enabled=enabled,
            expected_sizes=ladder_sizes,
            engine_max_batch_size=engine_max_batch_size,
            # Every MPI rank (tp*pp of them) logs its own [RANK k] capture line;
            # the audit requires ALL of them to have captured the full ladder,
            # so a single TP rank that silently decoded eager is caught.
            pp_size=args.tp * args.pp,
        )
        summary["graph_ladder"] = ladder
        problems.extend(ladder_problems)

        if args.compare_with:
            be_evidence, be_problems = compare_with_baseline(
                args.compare_with,
                args.compare_with_metrics,
                summary,
                per_step_store,
                args.expect_total_steps,
            )
            summary["be_equivalence"] = be_evidence
            problems.extend(be_problems)
            print(
                f"[replay] be_equivalence: rows={be_evidence['rows_compared']} "
                f"steps={be_evidence['steps_compared']} "
                f"bitwise_identical={be_evidence['bitwise_identical']}",
                flush=True,
            )

        summary["problems"] = problems
        summary["ok"] = not problems
        print(f"[replay] ok={summary['ok']} problems={len(problems)}", flush=True)
    except BaseException as exc:  # noqa: BLE001 — recorded, then exit code
        summary["ok"] = False
        summary["error"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[
            -4000:
        ]
        summary["problems"] = summary.get("problems") or []
        summary["problems"].append(f"exception: {type(exc).__name__}: {exc}")
        print(f"[replay] FAILED: {type(exc).__name__}: {exc}", flush=True)
    finally:
        summary["total_seconds"] = round(time.time() - started, 1)
        torch.save(per_step_store, args.metrics_out)
        with open(args.summary, "w") as fh:
            json.dump(summary, fh, indent=2)
        code = 0 if summary.get("ok") else 1
        with open(args.summary + ".exit.txt", "w") as fh:
            fh.write(f"{code}\n")
        print(f"[replay] wrote {args.summary} (exit {code})", flush=True)
        if llm is not None:
            llm.shutdown()
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
