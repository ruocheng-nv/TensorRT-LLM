# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GSM8K evaluation on the REAL LLM API runtime path (configs B and E).

Stage-1 criterion 5 (fixed-100 accuracy_canary) and the Stage-3 terminal gate
both need GSM8K measured on the implemented runtime — PP-sharded real
checkpoint, KVCacheManagerV2, deterministic greedy — not on the diagnostic
module driver. This driver renders the same prompts as the HF reference
(`glm5_next_gsm8k.render`, same tokenizer/chat template/thinking mode),
submits them in FIXED sequential chunks of ``--batch-size`` (stable batch
composition, matching the HF reference's static batching), decodes greedily
(explicit ``temperature=0, top_k=1``) with the same stop-token set, and
scores with the shared ``extract_answer``/``score`` helpers.

Config B = eager baseline (``cuda_graph_config=None``,
``disable_overlap_scheduler=True``); config E = ``CudaGraphConfig()`` +
overlap scheduling, with cuda_graph_hard_path evidence collected exactly as
the smoke/replay drivers do (INFO log tee -> capture-line/fallback grep,
embedded in the JSON).

Zero-truncation contract: ``--max-new-tokens`` must be a measured budget; any
row with ``finish_reason == 'length'`` is counted as truncated and makes the
run fail (`ok=false`) unless ``--allow-truncation`` is set (diagnostic use).

Every exit path writes the JSON summary and ``<summary>.exit.txt``; per-sample
rows (token ids, completions, predictions) go to ``--out`` (.pt).
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
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONPATH"] = _REPO + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ["TLLM_LOG_LEVEL"] = "INFO"

CAPTURE_MARK = "Running CUDA graph capture for"
FALLBACK_MARKS = (
    "falling back to eager",
    "Failed to allocate the CUDA graph padding dummy",
)


def tee_output_to(path: str) -> None:
    tee = subprocess.Popen(["tee", path], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), 1)
    os.dup2(tee.stdin.fileno(), 2)


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_cuda_graph_hard_path(runlog_path: str, enabled: bool):
    sys.stdout.flush()
    sys.stderr.flush()
    time.sleep(2)
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
    }
    problems = []
    if enabled:
        if not capture_lines:
            problems.append("cuda_graph_hard_path: no capture line — capture did not happen")
        if fallback_lines:
            problems.append(f"cuda_graph_hard_path: {len(fallback_lines)} eager-fallback warnings")
    elif capture_lines:
        problems.append("baseline config B unexpectedly captured CUDA graphs")
    return evidence, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/dev/shm/GLM-5.3-Flash")
    parser.add_argument("--pp", type=int, default=8)
    parser.add_argument("--config", choices=["B", "E"], default="B")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, required=True, help="measured budget")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eos-token-ids", type=int, nargs="*", default=[154820, 154827, 154829])
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--allow-truncation", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    runlog = args.summary + ".runlog.txt"
    tee_output_to(runlog)

    import torch
    from glm5_next_gsm8k import (
        DEFAULT_REASONING_EFFORT,
        extract_answer,
        load_samples,
        render,
        score,
    )
    from transformers import AutoTokenizer

    import tensorrt_llm
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.inputs import TokensPrompt
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    assert tensorrt_llm.__file__.startswith(_REPO), (
        f"stale tensorrt_llm package resolved: {tensorrt_llm.__file__}"
    )
    effort = args.reasoning_effort or DEFAULT_REASONING_EFFORT

    started = time.time()
    enabled = args.config == "E"
    summary = {
        "config": {
            "model": args.model,
            "pipeline_parallel_size": args.pp,
            "configuration": args.config,
            "cuda_graph": enabled,
            "overlap_scheduler": enabled,
            "samples": args.samples,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "eos_token_ids": list(args.eos_token_ids),
            "reasoning_effort": effort,
            "decode": {"temperature": 0, "top_k": 1},
            "runtime": "LLM API, KVCacheManagerV2, sequential fixed-size chunks",
        },
        "package": {
            "tensorrt_llm_version": tensorrt_llm.__version__,
            "tensorrt_llm_file": tensorrt_llm.__file__,
        },
        "provenance": {
            "driver_sha256": sha256_of(os.path.abspath(__file__)),
            "modeling_glm5_next_sha256": sha256_of(
                os.path.join(_REPO, "tensorrt_llm/_torch/models/modeling_glm5_next.py")
            ),
        },
        "ok": False,
        "problems": ["run did not complete"],
    }

    llm = None
    rows = []
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        samples = load_samples(args.samples)
        prompts = []
        for sample in samples:
            text = render(tokenizer, sample["question"], effort)
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            prompts.append((sample, ids))
        max_prompt = max(len(ids) for _, ids in prompts)

        llm = LLM(
            model=args.model,
            tensor_parallel_size=1,
            pipeline_parallel_size=args.pp,
            max_seq_len=max_prompt + args.max_new_tokens + 64,
            max_batch_size=args.batch_size,
            max_num_tokens=8192,
            kv_cache_config=KvCacheConfig(enable_block_reuse=False),
            disable_overlap_scheduler=not enabled,
            cuda_graph_config=CudaGraphConfig() if enabled else None,
        )
        summary["load_seconds"] = round(time.time() - started, 1)
        print(f"[gsm8k] engine up in {summary['load_seconds']}s", flush=True)

        sampling = SamplingParams(
            max_tokens=args.max_new_tokens,
            temperature=0,
            top_k=1,
            stop_token_ids=list(args.eos_token_ids),
        )
        for chunk_start in range(0, len(prompts), args.batch_size):
            chunk = prompts[chunk_start : chunk_start + args.batch_size]
            t0 = time.time()
            outs = llm.generate(
                [TokensPrompt(prompt_token_ids=list(ids)) for _, ids in chunk], sampling
            )
            dt = time.time() - t0
            for (sample, ids), out in zip(chunk, outs):
                comp = out.outputs[0]
                completion = comp.text
                predicted = extract_answer(completion)
                rows.append(
                    {
                        "index": sample["index"],
                        "question": sample["question"],
                        "label": sample["label"],
                        "prompt_tokens": len(ids),
                        "generated_token_ids": list(comp.token_ids),
                        "num_generated": len(comp.token_ids),
                        "completion": completion,
                        "predicted": predicted,
                        "correct": predicted == sample["label"],
                        "finish_reason": comp.finish_reason,
                        "truncated": comp.finish_reason == "length",
                    }
                )
            done = min(chunk_start + args.batch_size, len(prompts))
            n_correct = sum(r["correct"] for r in rows)
            print(
                f"[gsm8k] {done}/{len(prompts)} done, correct so far {n_correct}, chunk {dt:.1f}s",
                flush=True,
            )

        result = score(rows)
        truncated = [r["index"] for r in rows if r["truncated"]]
        wrong = [r["index"] for r in rows if not r["correct"]]
        summary["score"] = result
        summary["num_correct"] = int(sum(r["correct"] for r in rows))
        summary["num_samples"] = len(rows)
        summary["truncated_indices"] = truncated
        summary["wrong_indices"] = wrong
        problems = []
        if truncated and not args.allow_truncation:
            problems.append(f"{len(truncated)} truncated rows at budget {args.max_new_tokens}")
        evidence, hard_path_problems = check_cuda_graph_hard_path(runlog, enabled)
        summary["cuda_graph_hard_path"] = evidence
        problems.extend(hard_path_problems)
        summary["problems"] = problems
        summary["ok"] = not problems
        print(
            f"[gsm8k] score {summary['num_correct']}/{summary['num_samples']} "
            f"truncated={len(truncated)} wrong={wrong} ok={summary['ok']}",
            flush=True,
        )
    except BaseException as exc:  # noqa: BLE001
        summary["ok"] = False
        summary["error"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[
            -4000:
        ]
        summary["problems"] = [f"exception: {type(exc).__name__}: {exc}"]
        print(f"[gsm8k] FAILED: {type(exc).__name__}: {exc}", flush=True)
    finally:
        summary["total_seconds"] = round(time.time() - started, 1)
        torch.save({"rows": rows, "config": summary["config"]}, args.out)
        with open(args.summary, "w") as fh:
            json.dump(summary, fh, indent=2)
        code = 0 if summary.get("ok") else 1
        with open(args.summary + ".exit.txt", "w") as fh:
            fh.write(f"{code}\n")
        print(f"[gsm8k] wrote {args.summary} (exit {code})", flush=True)
        if llm is not None:
            llm.shutdown()
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
