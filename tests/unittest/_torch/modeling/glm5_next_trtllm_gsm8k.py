# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the long-horizon canary and fixed-100 GSM8K on the TensorRT-LLM path.

The counterpart to ``glm5_next_hf_gsm8k_reference.py``. Both read the same
sample set, prompt rendering, stop tokens and answer extraction from
``glm5_next_gsm8k``, so any score difference is the model rather than the
harness.

Run (from the repo root)::

    python tests/unittest/_torch/modeling/glm5_next_trtllm_gsm8k.py \\
        --reference agent-flow/workspace/glm-5.3-flash-bringup/reports/hf_gsm8k_reference.pt \\
        --out agent-flow/workspace/glm-5.3-flash-bringup/reports/trtllm_gsm8k.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Entry-point scripts are launched as `python <this file>`, which seeds sys.path
# with *this* directory rather than the cwd -- without the repo root the import
# below silently resolves to an installed tensorrt_llm wheel instead of the
# checkout under test, or fails outright.
sys.path.insert(
    1,
    os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
    ),
)

from glm5_next_full_model import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    Glm5NextGenerator,
    attach_caches,
    load_full_model,
)
from glm5_next_gsm8k import (  # noqa: E402
    BATCH_SIZE,
    DEFAULT_REASONING_EFFORT,
    NUM_SAMPLES,
    classify_divergence,
    compare_runs,
    extract_answer,
    load_samples,
    render,
    score,
    select,
)


def canary(generator, tokenizer, reference, max_new_tokens: int) -> List[Dict[str, Any]]:
    """Long-horizon parity: teacher-forced against the native-generate golden.

    Teacher forcing keeps both paths on the same prefix for the whole horizon,
    so a disagreement at step k is reported at step k rather than forking the
    sequence. Over 512 steps a free-running comparison would diverge once and
    then be measuring two different texts.
    """
    rows = []
    for item in reference["canary"]:
        prompt_ids = item["input_ids"].tolist()
        forced = item["generated_token_ids"].tolist()
        steps = min(max_new_tokens, len(forced))
        began = time.time()
        prefill_logits, step_logits, tokens = generator.generate(
            [prompt_ids], steps, forced=[forced]
        )
        theirs = forced[:steps]
        # The golden was generated with `min_new_tokens`, so HuggingFace's own
        # logits processor drove the stop tokens to -inf at every step to hold
        # the full horizon. Reproducing that mask here is required for the
        # comparison to be like-for-like: without it this path may pick a stop
        # token the reference was forbidden to pick, and the resulting -inf
        # arithmetic makes both the separation and the noise band infinite --
        # which the tie test then waves through as `inf <= inf`.
        masked = torch.isneginf(item["generated_step_logits"])
        mine = []
        divergences = []
        for step in range(steps):
            got = (prefill_logits[0] if step == 0 else step_logits[0, step - 1]).float().cpu()
            got = got.masked_fill(masked[step], float("-inf"))
            mine.append(int(got.argmax()))
            ref = item["generated_step_logits"][step].float()
            # One shared, non-tautological verdict per divergent step (defined in
            # glm5_next_gsm8k.classify_divergence): confidence is calibrated on
            # competitive tokens OTHER than the inverted pair, plus a
            # reference-only bf16-ULP measure. Returns None when the argmax agrees.
            verdict = classify_divergence(got, ref)
            if verdict is None:
                continue
            verdict["step"] = step
            divergences.append(verdict)
        rows.append(
            {
                "index": item["index"],
                "prompt": item["prompt"],
                "num_steps": steps,
                "num_matching": sum(1 for a, b in zip(mine, theirs) if a == b),
                "first_divergence": divergences[0]["step"] if divergences else None,
                "first_confident_divergence": next(
                    (d["step"] for d in divergences if d["confident"]), None
                ),
                "num_divergences": len(divergences),
                "num_confident_divergences": sum(1 for d in divergences if d["confident"]),
                "divergences": divergences,
                "all_finite": bool(
                    torch.isfinite(prefill_logits).all() and torch.isfinite(step_logits).all()
                ),
                "seconds": time.time() - began,
                "tokens": mine,
            }
        )
        print(
            f"[trt] canary {item['index']}: {rows[-1]['num_matching']}/{steps} matched, "
            f"{rows[-1]['num_confident_divergences']} confident divergences, "
            f"{rows[-1]['seconds']:.1f}s",
            flush=True,
        )
    return rows


def run_gsm8k(
    generator,
    tokenizer,
    samples,
    max_new_tokens: int,
    batch_size: int,
    effort: str,
    eos_token_ids,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        prompts = [
            tokenizer(render(tokenizer, s["question"], effort), add_special_tokens=False)[
                "input_ids"
            ]
            for s in chunk
        ]
        began = time.time()
        results = generator.generate_until_eos(
            prompts, max_new_tokens, eos_token_ids, progress_every=100
        )
        for sample, result in zip(chunk, results):
            text = tokenizer.decode(result["tokens"], skip_special_tokens=True)
            rows.append(
                {
                    **sample,
                    "completion": text,
                    "num_generated": result["num_generated"],
                    "predicted": extract_answer(text),
                    "truncated": not result["stopped_on_eos"],
                }
            )
        print(
            f"[trt] gsm8k {len(rows)}/{len(samples)} ({time.time() - began:.1f}s for {len(chunk)})",
            flush=True,
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    # Optional so the GSM8K half can run alongside the reference producer -- the
    # two models fit on the visible GPUs together, and only the canary and the
    # score comparison actually need the golden.
    parser.add_argument("--reference", default=None, help="the HF golden .pt")
    parser.add_argument(
        "--eos-token-ids",
        type=int,
        nargs="*",
        default=None,
        help="stop tokens when no reference is supplied; defaults to generation_config",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--canary-tokens", type=int, default=512)
    parser.add_argument("--gsm8k-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument(
        "--gsm8k-indices",
        type=int,
        nargs="*",
        default=None,
        help="evaluate only these dataset indices; a subset score is a diagnostic, "
        "never the fixed-100 gate",
    )
    parser.add_argument("--gsm8k-max-new-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--skip-canary", action="store_true")
    parser.add_argument("--skip-gsm8k", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    reference: Dict[str, Any] = {}
    if args.reference is not None:
        reference = torch.load(args.reference, map_location="cpu", weights_only=False)
        if reference["reasoning_effort"] != args.reasoning_effort:
            raise ValueError(
                f"reference used reasoning_effort={reference['reasoning_effort']!r} but this "
                f"run asks for {args.reasoning_effort!r}; a config difference would read as a "
                "model gap"
            )
        eos_token_ids = reference["eos_token_id"]
    elif args.eos_token_ids:
        eos_token_ids = list(args.eos_token_ids)
    else:
        from transformers import GenerationConfig

        eos = GenerationConfig.from_pretrained(args.checkpoint).eos_token_id
        eos_token_ids = list(eos) if isinstance(eos, (list, tuple)) else [eos]
    if not args.skip_canary and "canary" not in reference:
        raise ValueError("--skip-canary is required when the reference has no canary section")

    loaded = load_full_model(args.checkpoint, progress=True)
    payload: Dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "reasoning_effort": args.reasoning_effort,
        "batch_size": args.batch_size,
        "gsm8k_max_new_tokens": args.gsm8k_max_new_tokens,
        "eos_token_id": eos_token_ids,
        "load_seconds": loaded.load_seconds,
        "load_report": loaded.load_report,
        "decode": {"do_sample": False, "num_beams": 1, "temperature": 0, "top_k": 1},
    }

    if not args.skip_canary:
        longest = max(int(c["input_ids"].numel()) for c in reference["canary"])
        attach_caches(loaded, max_batch_size=1, max_seq_len=longest + args.canary_tokens + 64)
        generator = Glm5NextGenerator(loaded)
        payload["canary"] = canary(generator, tokenizer, reference, args.canary_tokens)
        payload["canary_tokens"] = args.canary_tokens
        generator.close()

    if not args.skip_gsm8k:
        samples = load_samples(args.gsm8k_samples)
        if args.gsm8k_indices:
            samples = select(samples, args.gsm8k_indices)
            payload["gsm8k_indices"] = list(args.gsm8k_indices)
        prompt_lengths = [
            len(
                tokenizer(
                    render(tokenizer, s["question"], args.reasoning_effort),
                    add_special_tokens=False,
                )["input_ids"]
            )
            for s in samples
        ]
        budget = max(prompt_lengths) + args.gsm8k_max_new_tokens + 64
        attach_caches(loaded, max_batch_size=args.batch_size, max_seq_len=budget)
        generator = Glm5NextGenerator(loaded)
        rows = run_gsm8k(
            generator,
            tokenizer,
            samples,
            args.gsm8k_max_new_tokens,
            args.batch_size,
            args.reasoning_effort,
            eos_token_ids,
        )
        generator.close()
        payload["gsm8k"] = rows
        payload["gsm8k_score"] = score(rows)
        payload["cache_budget"] = budget
        payload["max_prompt_tokens"] = max(prompt_lengths)
        print("[trt] gsm8k score: " + json.dumps(payload["gsm8k_score"]), flush=True)
        if "gsm8k" in reference:
            payload["reference_score"] = score(reference["gsm8k"])
            payload["discriminating"] = compare_runs(reference["gsm8k"], rows)
            print("[trt] reference score: " + json.dumps(payload["reference_score"]), flush=True)
            print(
                f"[trt] reference-right/trtllm-wrong: "
                f"{[d['index'] for d in payload['discriminating']]}",
                flush=True,
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(payload, args.out)
    print(f"[trt] wrote {args.out}", flush=True)

    if args.summary:
        summary = {k: v for k, v in payload.items() if k not in ("canary", "gsm8k")}
        if "gsm8k" in payload:
            summary["gsm8k_rows"] = [
                {k: r[k] for k in ("index", "label", "predicted", "num_generated", "truncated")}
                for r in payload["gsm8k"]
            ]
        if "canary" in payload:
            summary["canary_rows"] = [
                {
                    k: c[k]
                    for k in (
                        "index",
                        "num_steps",
                        "num_matching",
                        "first_divergence",
                        "num_divergences",
                        "num_confident_divergences",
                        "all_finite",
                        "seconds",
                    )
                }
                for c in payload["canary"]
            ]
        with open(args.summary, "w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        print(f"[trt] wrote {args.summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
