# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime increment R4: the first normal LLM API smoke on the real checkpoint.

Baseline (B) configuration exactly as the acceptance criteria define it:
``cuda_graph=false`` (``cuda_graph_config=None`` -- the LLM API default is
*enabled*, so this must be explicit) and ``overlap_scheduler=false``
(``disable_overlap_scheduler=True``). ``--config E`` runs the enabled matrix
leg instead: ``CudaGraphConfig()`` plus overlap scheduling, with hard-path
evidence collected from the run's own log (the driver forces
``TLLM_LOG_LEVEL=INFO``, tees its stdout/stderr -- which the MPI-spawned PP
workers inherit -- to ``<summary>.runlog.txt``, then asserts at least one
"Running CUDA graph capture" worker line and zero eager-fallback warnings,
recording the matched lines in the JSON). Decode batches here are
decode-only bs<=4 shapes inside the default capture list, so capture plus
no-fallback means decode steps replayed the graph. Config B asserts the
same log shows NO capture line (a true baseline).

Pipeline parallelism carries the 328 GB
checkpoint: PP=8 keeps one FP8 shard per rank. Two host-memory fixes make that
fit (both proved by ``probe_rank0_load.py`` / unit tests): construction is now
``MetaInitMode``-safe (``Glm5NextLayerNorm`` + ``torch.empty`` MoE experts), so
each rank builds only meta parameters and materializes just its shard on GPU
instead of real-allocating the whole model in host RAM (the earlier global
host-OOM SIGKILL); and checkpoint reading is streamed lazily for glm5_next so
no rank materializes the full checkpoint on the host during load either.

The smoke exercises real prefill, multi-step decode/cache reuse, and a second
batch over the same engine (slot reuse), and writes a JSON summary with the
exact config, package provenance, prompts, generated token ids/texts,
finiteness/emptiness checks, timings, and a host/GPU memory time series
sampled through the whole run. The summary and an exit-marker file
(``<summary>.exit.txt``) are written on EVERY exit path, including engine
construction failures, so callers never have to infer the outcome.

Goal 3.4 (production runtime conversion) extends the same driver:

* **Production-stack provenance.** The worker ranks are separate MPI
  processes, so the driver harvests the model's own INFO lines from the
  shared runlog and REQUIRES: the per-rank ``glm5_next runtime stack:`` line
  naming ``GlmKpoolSparseAttention`` (production sparse MLA) and
  ``TRTLLMGenFusedMoE`` (production block-FP8 MoE); the one-shot
  ``glm5_next KDA dispatch`` lines proving ``prefill=trtllm::kda_prefill``
  and the Triton decode step actually ran; the per-rank
  ``glm5_next load report:`` accounting line; and config_utils' KDA-hybrid
  fp32-state override line (proof the GLM V2 hybrid cache-parameter path
  built the manager). A batch-1 prompt is ~8 chunks long specifically so the
  CuTe production prefill (>=4 total 64-token chunks) dispatches in-engine.
* **Interleaved requests + mid-decode cancellation.** Three async requests
  with a ``--cancel-tokens`` budget are submitted together; after
  ``--cancel-after-seconds`` the middle one is aborted mid-decode. The two
  survivors must complete their full budget, the aborted request must end
  with ``finish_reason='cancelled'`` and strictly fewer tokens than the
  budget.
* **Slot reuse + determinism on identical compositions.** batch2 and (after
  the abort) batch3 repeat batch1's exact composition and must reproduce its
  greedy tokens per prompt. The composition is held fixed on purpose: token
  equality across *different* batch compositions is not an engine guarantee
  (GEMM reduction shapes change with batch size, and KDA prefill legitimately
  routes CuTe vs the small-batch torch scan by total chunk count — a
  documented last-ulp difference), and the iteration-21 override rejects
  gating low-margin forks. With composition fixed, any divergence is real
  state corruption; batch3-vs-batch1 isolates cancellation as the only
  variable.

Run (all eight GPUs must be free):
    python tests/unittest/_torch/modeling/glm5_next_llm_api_smoke.py \
        --pp 8 --config B --summary reports/goal1.5-logs/llm_api_smoke_b_pp8.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback

# Script-style invocation puts this file's directory (not the repo root) at
# sys.path[0], and a stale non-editable tensorrt_llm wheel in dist-packages
# shadows the editable in-repo package — which is the only one that knows
# Glm5NextForConditionalGeneration. Pin the repo root for this process AND
# export it via PYTHONPATH so the MPI-spawned PP worker ranks resolve the same
# in-repo package.
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 4))
sys.path.insert(0, _REPO)
os.environ["PYTHONPATH"] = _REPO + os.pathsep + os.environ.get("PYTHONPATH", "")
# INFO must be on before tensorrt_llm imports so the MPI-spawned worker ranks
# inherit it: the CUDA-graph capture evidence for --config E is an INFO log
# line emitted by each worker's model engine at warmup.
os.environ["TLLM_LOG_LEVEL"] = "INFO"

from glm5_next_driver_preflight import (  # noqa: E402  (script-dir import)
    audit_graph_ladder,
    disk_preflight,
    expected_graph_batch_sizes,
)
from glm5_next_llm_api_logit_replay import resolve_moe_parallel  # noqa: E402  (script-dir import)

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


def check_cuda_graph_hard_path(runlog_path: str, enabled: bool):
    """Grep the run log for real capture / fallback evidence.

    Returns (evidence_dict, problems). Decode batches in this smoke are
    decode-only shapes at batch sizes <= max_batch_size=4, all inside the
    default capture list, so "capture happened" plus "no eager fallback"
    means the decode steps replayed captured graphs. Config B must show
    zero capture lines (a true baseline).
    """
    sys.stdout.flush()
    sys.stderr.flush()
    time.sleep(2)  # let tee drain
    capture_lines, fallback_lines = [], []
    try:
        with open(runlog_path, "r", errors="replace") as fh:
            for line in fh:
                if CAPTURE_MARK in line:
                    capture_lines.append(line.rstrip()[-240:])
                if any(m in line for m in FALLBACK_MARKS):
                    fallback_lines.append(line.rstrip()[-240:])
    except OSError as exc:
        return {"runlog": runlog_path, "error": str(exc)}, [f"hard-path log unreadable: {exc}"]
    evidence = {
        "runlog": runlog_path,
        "capture_line_count": len(capture_lines),
        "capture_lines": capture_lines[:16],
        "fallback_line_count": len(fallback_lines),
        "fallback_lines": fallback_lines[:16],
        "decode_batch_shape": "decode-only bs<=4, inside the default capture list",
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


PROVENANCE_MARKS = {
    "runtime_stack": "glm5_next runtime stack:",
    "kda_dispatch": "glm5_next KDA dispatch",
    "load_report": "glm5_next load report:",
    "hybrid_v2_dtype": "KDA hybrid: overriding mamba_ssm_cache_dtype",
}


def harvest_production_provenance(runlog_path: str):
    """Collect the worker ranks' production-stack evidence from the runlog.

    PP/MPI worker processes cannot be introspected from the driver; the model
    and the cache-parameter path publish one-shot INFO lines instead (see the
    module docstring), and this reads them back. Returns (evidence, problems).
    """
    found = {key: [] for key in PROVENANCE_MARKS}
    try:
        with open(runlog_path, errors="replace") as fh:
            for line in fh:
                for key, mark in PROVENANCE_MARKS.items():
                    if mark in line:
                        found[key].append(line.rstrip()[-300:])
    except OSError as exc:
        return {"runlog": runlog_path, "error": str(exc)}, [f"provenance log unreadable: {exc}"]

    problems = []
    stack_text = " ".join(found["runtime_stack"])
    if "GlmKpoolSparseAttention" not in stack_text:
        problems.append(
            "provenance: production sparse-MLA backend GlmKpoolSparseAttention "
            "is not named in any worker's runtime-stack line"
        )
    if "TRTLLMGenFusedMoE" not in stack_text:
        problems.append(
            "provenance: production MoE backend TRTLLMGenFusedMoE is not named "
            "in any worker's runtime-stack line"
        )
    dispatch_text = " ".join(found["kda_dispatch"])
    if "prefill=trtllm::kda_prefill" not in dispatch_text:
        problems.append(
            "provenance: production KDA prefill (trtllm::kda_prefill) never "
            "dispatched — the >=4-chunk long prompt should have routed to it"
        )
    if "decode=trtllm::causal_conv1d_update+triton_kda_delta_step" not in dispatch_text:
        problems.append("provenance: production KDA decode step never dispatched")
    if not found["load_report"]:
        problems.append("provenance: no per-rank 'glm5_next load report' accounting line")
    if not found["hybrid_v2_dtype"]:
        problems.append(
            "provenance: config_utils' KDA-hybrid fp32-state override line is "
            "missing — the glm5_next V2 hybrid cache-parameter path did not run"
        )
    evidence = {key: {"count": len(lines), "lines": lines[:12]} for key, lines in found.items()}
    return evidence, problems


def _one_memory_sample(started: float) -> dict:
    """Host MemAvailable, fattest-python RSS, and per-GPU used MiB."""
    sample = {"t": round(time.time() - started, 1)}
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    sample["host_avail_gib"] = round(int(line.split()[1]) / 1024**2, 1)
                    break
    except OSError:
        pass
    max_rss = 0
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                with open(f"/proc/{pid}/comm") as fh:
                    if "python" not in fh.read():
                        continue
                with open(f"/proc/{pid}/statm") as fh:
                    max_rss = max(max_rss, int(fh.read().split()[1]) * page)
            except OSError:
                continue
        sample["max_python_rss_gib"] = round(max_rss / 1024**3, 1)
    except (OSError, ValueError):
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        sample["gpu_used_mib"] = [int(x) for x in out.stdout.split()]
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return sample


def _start_memory_sampler(samples: list, started: float, period_s: float = 10.0):
    """Print + record a host/GPU memory sample every ``period_s`` seconds."""
    stop = threading.Event()

    def run():
        while not stop.is_set():
            try:
                sample = _one_memory_sample(started)
                samples.append(sample)
                print(
                    f"[mem] t={sample['t']}s avail={sample.get('host_avail_gib')}GiB "
                    f"max_py_rss={sample.get('max_python_rss_gib')}GiB "
                    f"gpus={sample.get('gpu_used_mib')}",
                    flush=True,
                )
            except Exception:
                pass  # the sampler must never kill the run
            stop.wait(period_s)

    thread = threading.Thread(target=run, daemon=True, name="mem-sampler")
    thread.start()
    return stop, thread


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/dev/shm/GLM-5.3-Flash")
    parser.add_argument("--pp", type=int, default=8)
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="tensor_parallel_size; the Stage-6 TP4/EP4 smoke passes "
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
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--cancel-tokens",
        type=int,
        default=512,
        help="decode budget for the interleaved cancellation leg; must be "
        "large enough that the abort lands mid-decode (config E replays "
        "graphs at ~30 ms/step at bs=3, so 512 tokens is ~16 s against the "
        "4 s default abort delay)",
    )
    parser.add_argument(
        "--cancel-after-seconds",
        type=float,
        default=4.0,
        help="how long after submission the middle interleaved request is aborted",
    )
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    try:
        moe_tp, moe_ep, mapping_label, moe_llm_kwargs = resolve_moe_parallel(
            args.tp, args.pp, args.ep
        )
    except ValueError as exc:
        parser.error(str(exc))

    runlog = args.summary + ".runlog.txt"
    tee_output_to(runlog)

    # Infrastructure gate BEFORE the engine exists: a full overlay kills a
    # long run mid-session with Errno 28 (observed) at a point where all
    # evidence is lost. A failed gate is an infrastructure condition, not
    # model evidence.
    preflight, preflight_problems = disk_preflight(args.summary)
    print(f"[smoke] disk preflight: {json.dumps(preflight['measured'])}", flush=True)
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
        print(f"[smoke] disk preflight FAILED: {preflight_problems}", flush=True)
        return 1

    import tensorrt_llm
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    assert tensorrt_llm.__file__.startswith(_REPO), (
        f"stale tensorrt_llm package resolved: {tensorrt_llm.__file__} "
        f"(expected the in-repo tree under {_REPO})"
    )

    started = time.time()
    enabled = args.config == "E"
    samples: list = []
    summary = {
        "config": {
            "model": args.model,
            "tensor_parallel_size": args.tp,
            "pipeline_parallel_size": args.pp,
            "world_size": args.tp * args.pp,
            "moe_tensor_parallel_size": moe_tp,
            "moe_expert_parallel_size": moe_ep,
            "mapping_label": mapping_label,
            "configuration": args.config,
            "max_seq_len": args.max_seq_len,
            "max_batch_size": 4,
            "cuda_graph": enabled,
            "overlap_scheduler": enabled,
            "kv_cache": {"enable_block_reuse": False, "max_tokens": 16384},
            "cancel_tokens": args.cancel_tokens,
            "cancel_after_seconds": args.cancel_after_seconds,
            "legs": [
                "batch1 (4 prompts, one >=8-chunk long prompt)",
                "batch2 (identical composition repeat: slot reuse + determinism)",
                "cancel (3 interleaved async, middle aborted mid-decode)",
                "batch3 (identical composition repeat after cancellation)",
            ],
        },
        "package": {
            "tensorrt_llm_version": tensorrt_llm.__version__,
            "tensorrt_llm_file": tensorrt_llm.__file__,
        },
        "disk_preflight": preflight,
        "phases": {},
        "memory_samples": samples,
        "ok": False,
        "problems": ["run did not complete"],
    }
    sampler_stop, sampler_thread = _start_memory_sampler(samples, started)

    llm = None
    graph_config = CudaGraphConfig() if enabled else None
    try:
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tp,
            pipeline_parallel_size=args.pp,
            max_seq_len=args.max_seq_len,
            max_batch_size=4,
            max_num_tokens=4096,
            kv_cache_config=KvCacheConfig(enable_block_reuse=False, max_tokens=16384),
            # B: overlap_scheduler=false, cuda_graph=false (the LLM API default
            # is ON, so None must be explicit). E: overlap + CudaGraphConfig().
            disable_overlap_scheduler=not enabled,
            cuda_graph_config=graph_config,
            **moe_llm_kwargs,
        )
        summary["phases"]["load_seconds"] = round(time.time() - started, 1)
        print(f"[smoke] engine up in {summary['phases']['load_seconds']}s", flush=True)

        # Greedy by default: SamplingParams with no temperature/top_p/top_k.
        sampling = SamplingParams(max_tokens=args.max_new_tokens)

        # The last prompt is deliberately long (~450 tokens, >= 8 64-token
        # chunks): the CuTe production KDA prefill requires >= 4 total chunks
        # in the varlen batch, so this prompt forces `trtllm::kda_prefill` to
        # dispatch inside the engine even if the scheduler runs it alone. The
        # short prompts may legitimately route to the documented small-batch
        # torch-scan fallback; the provenance check requires the production
        # path to have run at least once.
        prompts = [
            "The capital of France is",
            "2, 4, 6, 8,",
            "Water is composed of hydrogen and",
            "Count upward without stopping: " + " ".join(str(i) for i in range(1, 220)) + " and",
        ]
        t0 = time.time()
        first = llm.generate(prompts, sampling)
        summary["phases"]["batch1_seconds"] = round(time.time() - t0, 1)

        # A second, IDENTICAL batch on the same engine: released slots must be
        # re-acquirable and the engine must be deterministic for an identical
        # workload. The composition is repeated exactly on purpose — greedy
        # token equality across DIFFERENT batch compositions is not an engine
        # guarantee (GEMM reduction shapes change with batch size, and the
        # KDA prefill legitimately routes CuTe vs the small-batch torch scan
        # by total chunk count; both round differently at the last bf16 ulp),
        # and the iteration-21 acceptance override explicitly rejects gating
        # on such low-margin forks. Identical composition -> identical
        # schedules, kernels and shapes, so any token divergence here is real
        # engine-state corruption, not dtype noise.
        t0 = time.time()
        second = llm.generate(prompts, sampling)
        summary["phases"]["batch2_seconds"] = round(time.time() - t0, 1)

        def rows(outputs):
            out = []
            for o in outputs:
                token_ids = list(o.outputs[0].token_ids)
                out.append(
                    {
                        "prompt": o.prompt,
                        "token_ids": token_ids,
                        "text": o.outputs[0].text,
                        "num_new_tokens": len(token_ids),
                    }
                )
            return out

        summary["batch1"] = rows(first)
        summary["batch2"] = rows(second)

        # Leg 3 — interleaved async requests with a mid-decode cancellation.
        # Three requests share the engine concurrently (scheduler interleaving
        # at bs=3); the middle one is aborted while decoding. The two
        # survivors must not be perturbed and must finish their full budget.
        cancel_budget = args.cancel_tokens
        # end_id=-1 (the pattern of test_llm_pytorch.py::test_llm_abort_request)
        # disables EOS stopping for this leg only, so the abort target is
        # guaranteed to still be decoding at the abort instant and both
        # survivors deterministically run to exactly the full budget — which
        # also keeps bs=3 -> bs=2 decode running long after the abort.
        cancel_sampling = SamplingParams(max_tokens=cancel_budget, end_id=-1)
        cancel_prompts = [
            "Explain, step by step, how rain forms:",
            "List the powers of two in order: 2, 4, 8,",
            "Describe a lighthouse keeper's morning routine:",
        ]
        t0 = time.time()
        futures = [llm.generate_async(p, cancel_sampling) for p in cancel_prompts]
        time.sleep(args.cancel_after_seconds)
        raced_to_completion = bool(futures[1].finished)
        futures[1].abort()
        survivors = [futures[0].result(), futures[2].result()]
        aborted = futures[1].result()
        summary["phases"]["cancel_leg_seconds"] = round(time.time() - t0, 1)
        summary["cancel"] = {
            "budget": cancel_budget,
            "aborted_after_seconds": args.cancel_after_seconds,
            "aborted_prompt": cancel_prompts[1],
            "aborted_new_tokens": len(aborted.outputs[0].token_ids),
            "aborted_finish_reason": aborted.outputs[0].finish_reason,
            "raced_to_completion": raced_to_completion,
            "survivor_new_tokens": [len(s.outputs[0].token_ids) for s in survivors],
            "survivor_finish_reasons": [s.outputs[0].finish_reason for s in survivors],
            "survivor_texts_nonempty": [bool(s.outputs[0].text.strip()) for s in survivors],
        }
        print(f"[smoke] cancel leg: {summary['cancel']}", flush=True)

        # Leg 4 — slot reuse AFTER cancellation: the aborted request's mamba
        # slot and cache pages must be reclaimed cleanly, leaving the engine
        # bit-identical for the same identical-composition batch. Comparing
        # batch3 to batch1 with the composition held fixed isolates
        # cancellation as the only variable.
        t0 = time.time()
        third = llm.generate(prompts, sampling)
        summary["phases"]["batch3_seconds"] = round(time.time() - t0, 1)
        summary["batch3"] = rows(third)

        problems = []
        for name in ("batch1", "batch2"):
            for row in summary[name]:
                if row["num_new_tokens"] < args.max_new_tokens:
                    problems.append(
                        f"{name}: only {row['num_new_tokens']} tokens for {row['prompt']!r}"
                    )
                if not row["text"].strip():
                    problems.append(f"{name}: empty text for {row['prompt']!r}")
        for i in range(len(prompts)):
            if summary["batch1"][i]["token_ids"] != summary["batch2"][i]["token_ids"]:
                problems.append(
                    f"identical repeated batch diverged at prompt {i} (slot-reuse nondeterminism)"
                )

        cancel = summary["cancel"]
        if cancel["raced_to_completion"]:
            problems.append(
                "cancellation raced to completion before the abort — no "
                "mid-flight cancellation was exercised (raise --cancel-tokens "
                "or lower --cancel-after-seconds)"
            )
        if not 0 < cancel["aborted_new_tokens"] < cancel_budget:
            problems.append(
                f"aborted request produced {cancel['aborted_new_tokens']} tokens; "
                f"a mid-decode cut requires 0 < n < {cancel_budget}"
            )
        if cancel["aborted_finish_reason"] != "cancelled":
            problems.append(
                f"aborted finish_reason={cancel['aborted_finish_reason']!r} (expected 'cancelled')"
            )
        for i, n in enumerate(cancel["survivor_new_tokens"]):
            if n != cancel_budget:
                problems.append(f"cancel-leg survivor {i} produced {n}/{cancel_budget} tokens")
        if not all(cancel["survivor_texts_nonempty"]):
            problems.append("cancel-leg survivor produced empty text")
        for i in range(len(prompts)):
            if summary["batch3"][i]["token_ids"] != summary["batch1"][i]["token_ids"]:
                problems.append(
                    f"post-cancellation identical batch diverged at prompt {i} "
                    "(slot reuse after abort corrupted engine state)"
                )

        evidence, hard_path_problems = check_cuda_graph_hard_path(runlog, enabled)
        summary["cuda_graph_hard_path"] = evidence
        problems.extend(hard_path_problems)
        # Silent-eager rejection: a decode-only batch whose size has no
        # captured graph runs eager WITHOUT any warning line, so the
        # fallback grep above cannot see it. Assert instead that the
        # engine's captured ladder covers every schedulable decode batch
        # size (1..max_batch_size), recomputed from this driver's own
        # config with the engine's filtering rule.
        ladder_sizes = expected_graph_batch_sizes(
            graph_config.batch_sizes if graph_config else [],
            engine_max_batch_size=4,
            max_num_tokens=4096,
        )
        ladder, ladder_problems = audit_graph_ladder(
            runlog,
            enabled=enabled,
            expected_sizes=ladder_sizes,
            engine_max_batch_size=4,
            # Every MPI rank (tp*pp of them) logs its own [RANK k] capture
            # line; the audit requires ALL of them to have captured the full
            # ladder, so a single TP rank that silently decoded eager is caught.
            pp_size=args.tp * args.pp,
        )
        summary["graph_ladder"] = ladder
        problems.extend(ladder_problems)
        provenance, provenance_problems = harvest_production_provenance(runlog)
        summary["production_provenance"] = provenance
        problems.extend(provenance_problems)
        summary["problems"] = problems
        summary["ok"] = not problems
        for name in ("batch1", "batch2", "batch3"):
            for row in summary[name]:
                print(f"[smoke] {name} {row['prompt'][:80]!r} -> {row['text']!r}", flush=True)
        print(f"[smoke] ok={summary['ok']} problems={problems}", flush=True)
    except BaseException as exc:  # noqa: BLE001 — recorded, then re-raised as exit code
        summary["ok"] = False
        summary["error"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[
            -4000:
        ]
        summary["problems"] = [f"exception: {type(exc).__name__}: {exc}"]
        print(f"[smoke] FAILED: {type(exc).__name__}: {exc}", flush=True)
    finally:
        sampler_stop.set()
        sampler_thread.join(timeout=5)
        summary["phases"]["total_seconds"] = round(time.time() - started, 1)
        # JSON + exit marker are written on EVERY path: a constructor crash
        # must still leave machine-readable evidence for chained callers.
        with open(args.summary, "w") as fh:
            json.dump(summary, fh, indent=2)
        code = 0 if summary.get("ok") else 1
        with open(args.summary + ".exit.txt", "w") as fh:
            fh.write(f"{code}\n")
        print(f"[smoke] wrote {args.summary} (exit {code})", flush=True)
        if llm is not None:
            llm.shutdown()

    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
