# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Produce the native-HF goldens Stage 1's remaining gates need.

Two artefacts, one model load, because loading the real checkpoint dominates
the cost:

* **long-horizon canary golden** -- native ``generate()`` out to >=512 new
  tokens for a couple of fixed prompts, with per-step logits. The existing
  Goal-1.1 fixture stops at 64, which is not enough to catch the drift a
  long generation exposes.
* **fixed-100 GSM8K reference** -- the PyTorch-side score the TensorRT-LLM run
  is compared against, on the same sample indices with the same rendering,
  stop behaviour and decode budget.

Both use ``generate()`` itself rather than a hand-written decode loop, so no
reimplementation of generation sits between the model and the golden.

Run (from the repo root)::

    python tests/unittest/_torch/modeling/glm5_next_hf_gsm8k_reference.py \\
        --out agent-flow/workspace/glm-5.3-flash-bringup/reports/hf_gsm8k_reference.pt
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

from glm5_next_gsm8k import (  # noqa: E402
    BATCH_SIZE,
    DEFAULT_REASONING_EFFORT,
    MAX_SEQ_LEN,
    NUM_SAMPLES,
    extract_answer,
    load_samples,
    render,
    score,
    select,
)

DEFAULT_CHECKPOINT = "/dev/shm/GLM-5.3-Flash"
#: The two canary prompts, taken from the Goal-1.1 fixed set so the short and
#: long horizons are measured on the same text.
CANARY_PROMPTS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many "
    "clips in May. How many clips did Natalia sell altogether in April and May?",
    "Explain in two sentences why the sky appears blue.",
]


def build_model(checkpoint: str):
    """Load exactly the model Goal 1.1's fixture was produced from.

    Two details are inherited rather than re-decided, so this golden and the
    existing 64-token one describe the same model: the checkpoint declares
    ``Glm5NextForConditionalGeneration``, whose native auto class is
    ``AutoModelForImageTextToText`` (the vision tower is built but never
    invoked on text-only input), and the quantization exclusion list carries
    Goal 1.1's corrective additions.
    """
    from glm5_next_hf_reference import _build_quantization_config, verify_quantization_split
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    config = AutoConfig.from_pretrained(checkpoint)
    started = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        quantization_config=_build_quantization_config(config),
    )
    model.eval()
    print(f"[ref] loaded in {time.time() - started:.1f}s", flush=True)
    split = verify_quantization_split(model, checkpoint)
    print(
        f"[ref] quantization split: {split['fp8_linear_modules']} FP8, "
        f"{split['bf16_linear_modules']} BF16",
        flush=True,
    )
    return tokenizer, config, model


@torch.no_grad()
def run_canary(tokenizer, model, max_new_tokens: int, effort: str) -> List[Dict[str, Any]]:
    """Native generate() out to ``max_new_tokens``, keeping per-step logits."""
    device = next(model.parameters()).device
    rows = []
    for index, prompt in enumerate(CANARY_PROMPTS):
        text = render(tokenizer, prompt, effort)
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        ids = encoded["input_ids"].to(device)
        started = time.time()
        out = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,  # never stop early: the canary needs the full horizon
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True,
        )
        generated = out.sequences[0, ids.shape[1] :]
        # scores[j] is the distribution that produced generated[j].
        step_logits = torch.stack([s[0].float().cpu() for s in out.scores], dim=0)
        rows.append(
            {
                "index": index,
                "prompt": prompt,
                "rendered": text,
                "input_ids": ids[0].cpu(),
                "generated_token_ids": generated.cpu(),
                "generated_step_logits": step_logits,
                "generated_text": tokenizer.decode(generated, skip_special_tokens=False),
                "seconds": time.time() - started,
            }
        )
        print(
            f"[ref] canary {index}: {generated.numel()} tokens in {rows[-1]['seconds']:.1f}s",
            flush=True,
        )
    return rows


@torch.no_grad()
def run_gsm8k(
    tokenizer, model, samples, max_new_tokens: int, batch_size: int, effort: str
) -> List[Dict[str, Any]]:
    """Batched greedy generation over the fixed sample set."""
    device = next(model.parameters()).device
    eos = model.generation_config.eos_token_id
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = eos[0] if isinstance(eos, (list, tuple)) else eos
    rows: List[Dict[str, Any]] = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        texts = [render(tokenizer, s["question"], effort) for s in chunk]
        # Left padding so every sequence's last position is the real final
        # token; right padding would make generate() continue from pad.
        tokenizer.padding_side = "left"
        batch = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        batch = {k: v.to(device) for k, v in batch.items()}
        began = time.time()
        out = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=pad,
        )
        completions = out[:, batch["input_ids"].shape[1] :]
        for sample, ids in zip(chunk, completions):
            text = tokenizer.decode(ids, skip_special_tokens=True)
            eos_ids = eos if isinstance(eos, (list, tuple)) else [eos]
            stopped = bool((ids[..., None] == torch.tensor(eos_ids, device=ids.device)).any())
            rows.append(
                {
                    **sample,
                    "completion": text,
                    "num_generated": int(ids.numel()),
                    "predicted": extract_answer(text),
                    # A sample that hit the budget without emitting EOS was cut
                    # off; its score measures the budget, not the model.
                    "truncated": not stopped,
                }
            )
        done = len(rows)
        print(
            f"[ref] gsm8k {done}/{len(samples)} ({time.time() - began:.1f}s for {len(chunk)})",
            flush=True,
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--canary-tokens", type=int, default=512)
    parser.add_argument("--gsm8k-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument(
        "--gsm8k-indices",
        type=int,
        nargs="*",
        default=None,
        help="evaluate only these dataset indices. Chunking is deterministic in "
        "dataset order, so a shard whose boundaries are multiples of the batch "
        "size reproduces the exact batch packing of one serial run -- which is "
        "what lets a full-dataset sweep run sharded without opening a "
        "packing-order gap against the other path",
    )
    parser.add_argument("--gsm8k-max-new-tokens", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--skip-canary", action="store_true")
    parser.add_argument("--skip-gsm8k", action="store_true")
    args = parser.parse_args()

    tokenizer, config, model = build_model(args.checkpoint)
    payload: Dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "reasoning_effort": args.reasoning_effort,
        "gsm8k_max_new_tokens": args.gsm8k_max_new_tokens,
        "batch_size": args.batch_size,
        "decode": {"do_sample": False, "num_beams": 1},
        "eos_token_id": model.generation_config.eos_token_id,
    }

    if not args.skip_canary:
        payload["canary"] = run_canary(tokenizer, model, args.canary_tokens, args.reasoning_effort)
        payload["canary_tokens"] = args.canary_tokens

    if not args.skip_gsm8k:
        samples = load_samples(args.gsm8k_samples)
        if args.gsm8k_indices:
            samples = select(samples, args.gsm8k_indices)
            payload["gsm8k_indices"] = list(args.gsm8k_indices)
        rows = run_gsm8k(
            tokenizer,
            model,
            samples,
            args.gsm8k_max_new_tokens,
            args.batch_size,
            args.reasoning_effort,
        )
        payload["gsm8k"] = rows
        payload["gsm8k_score"] = score(rows)
        print("[ref] gsm8k score: " + json.dumps(payload["gsm8k_score"]), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(payload, args.out)
    print(f"[ref] wrote {args.out}", flush=True)

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
                    "index": r["index"],
                    "num_generated": int(r["generated_token_ids"].numel()),
                    "seconds": r["seconds"],
                    "text_head": r["generated_text"][:200],
                }
                for r in payload["canary"]
            ]
        with open(args.summary, "w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        print(f"[ref] wrote {args.summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
