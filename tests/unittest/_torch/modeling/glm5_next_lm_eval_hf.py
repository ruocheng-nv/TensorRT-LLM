# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native-HF GSM8K reference through TensorRT-LLM's *own* lm-eval harness.

The accuracy criteria demand that the HF reference and the TensorRT-LLM path
be scored under one identical configuration: same sample set, same rendering,
same stop behaviour, same extraction, same metric. ``trtllm-eval`` measures
the TensorRT-LLM side through :class:`tensorrt_llm.evaluate.GSM8K` (an
lm-evaluation-harness wrapper that *shuffles the dataset with its own seed*
before applying ``num_samples``), so the only way to give the HF side the
literally-same harness is to run the same evaluator class and swap the
engine underneath.

That is what this driver does: it instantiates ``tensorrt_llm.evaluate.GSM8K``
with the same (dataset_path, num_samples, random_seed, chat-template,
system-prompt, fewshot) parameters the ``trtllm-eval`` invocation uses, and
hands it an adapter that exposes the tiny ``llm`` surface
``LmEvalWrapper`` consumes (``.tokenizer``, ``generate_async(...)`` ->
future-like with ``.result().outputs[0].text``) backed by **native
``model.generate()``** on the real checkpoint — no hand-written decode loop,
so the golden-generate policy is satisfied by construction. Prompt strings
arrive already rendered by the wrapper's own ``apply_chat_template`` (the
same code path the TensorRT-LLM side goes through), and are tokenized with
``add_special_tokens=False``, which is verified equivalent to the LLM API's
tokenization for this tokenizer (it adds no special tokens either way).

Stop behaviour: the engine side stops generation at the task's ``until``
strings; native ``generate()`` has no per-row string stopper, so this driver
generates to EOS/budget and post-truncates the *text* at the earliest
``until`` occurrence — the scored text is identical under both mechanisms.
EOS ids come from the checkpoint's ``generation_config`` exactly as the LLM
API consumes them. Rows that hit the token budget without a natural stop are
counted as truncated and fail the run (zero-truncation contract).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 4))
sys.path.insert(0, _REPO)
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONPATH"] = _REPO + os.pathsep + os.environ.get("PYTHONPATH", "")


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _PendingOutput:
    """Future-like handle; generation happens batched on first .result()."""

    class _Completion:
        def __init__(self):
            self.text: str = ""
            self.token_ids: List[int] = []
            self.finish_reason: Optional[str] = None

    def __init__(self, runner: "_BatchedHfRunner", prompt: str, sampling_params):
        self._runner = runner
        self.prompt = prompt
        self.sampling_params = sampling_params
        self.outputs = [self._Completion()]
        self.done = False

    def result(self):
        if not self.done:
            self._runner.flush()
        return self


class _BatchedHfRunner:
    """The minimal ``llm`` surface LmEvalWrapper needs, over native HF.

    ``generate_async`` queues; ``flush`` executes every queued request in
    fixed sequential chunks of ``batch_size`` (left-padded batched native
    ``generate``), honouring each request's max_tokens/stop from its
    SamplingParams. All queued requests share one (max_tokens, stop) pair by
    construction (one task, one CLI config); this is asserted, not assumed.
    """

    def __init__(self, model, tokenizer, batch_size: int, max_prompt_tokens: int):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = int(batch_size)
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.pending: List[_PendingOutput] = []
        self.completed_rows: List[Dict[str, Any]] = []
        self.chunk_seconds: List[float] = []

    # -- LmEvalWrapper surface -------------------------------------------
    def generate_async(self, prompt, sampling_params=None, streaming=False):
        assert not streaming, "streaming is not part of this reference driver"
        assert isinstance(prompt, str), f"expected a rendered prompt string, got {type(prompt)}"
        out = _PendingOutput(self, prompt, sampling_params)
        self.pending.append(out)
        return out

    def shutdown(self):
        pass

    # -- execution --------------------------------------------------------
    def flush(self):
        import torch

        todo = [p for p in self.pending if not p.done]
        if not todo:
            return
        max_tokens = {int(p.sampling_params.max_tokens) for p in todo}
        stops = {tuple(p.sampling_params.stop or ()) for p in todo}
        assert len(max_tokens) == 1, f"mixed max_tokens in one flush: {max_tokens}"
        assert len(stops) == 1, f"mixed stop sets in one flush: {stops}"
        budget = max_tokens.pop()
        stop_strings = list(stops.pop())

        eos_ids = self.model.generation_config.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        device = next(self.model.parameters()).device
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        for start in range(0, len(todo), self.batch_size):
            chunk = todo[start : start + self.batch_size]
            began = time.time()
            texts = [p.prompt for p in chunk]
            batch = self.tokenizer(
                texts, return_tensors="pt", padding=True, add_special_tokens=False
            )
            prompt_lens = batch["attention_mask"].sum(dim=1).tolist()
            assert max(prompt_lens) <= self.max_prompt_tokens, (
                f"prompt of {max(prompt_lens)} tokens exceeds the declared cap "
                f"{self.max_prompt_tokens}; refusing to silently truncate"
            )
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                out = self.model.generate(
                    **batch,
                    max_new_tokens=budget,
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=eos_ids,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            completions = out[:, batch["input_ids"].shape[1] :]
            dt = time.time() - began
            self.chunk_seconds.append(dt)
            for pending, row, n_prompt in zip(chunk, completions, prompt_lens):
                ids = row.tolist()
                # Strip trailing padding after this row's own EOS.
                cut = len(ids)
                natural_stop = False
                for j, tok in enumerate(ids):
                    if tok in eos_ids:
                        cut = j  # exclude the eos token itself, like the engine
                        natural_stop = True
                        break
                ids = ids[:cut]
                text = self.tokenizer.decode(ids, skip_special_tokens=False)
                # Engine-equivalent stop-string semantics: scored text ends
                # before the earliest ``until`` occurrence.
                stop_hit = None
                for s in stop_strings:
                    at = text.find(s)
                    if at != -1 and (stop_hit is None or at < stop_hit[1]):
                        stop_hit = (s, at)
                if stop_hit is not None:
                    text = text[: stop_hit[1]]
                    natural_stop = True
                completion = pending.outputs[0]
                completion.text = text
                completion.token_ids = ids
                completion.finish_reason = "stop" if natural_stop else "length"
                pending.done = True
                self.completed_rows.append(
                    {
                        "prompt_tokens": int(n_prompt),
                        "generated_tokens": len(ids),
                        "budget": budget,
                        "truncated": not natural_stop,
                        "stop_string_hit": stop_hit[0] if stop_hit else None,
                    }
                )
            done = len([p for p in self.pending if p.done])
            print(
                f"[lm-eval-hf] {done}/{len(self.pending)} rows, chunk {dt:.1f}s",
                flush=True,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="/dev/shm/GLM-5.3-Flash")
    parser.add_argument("--dataset-path", default="gsm8k", help="local HF datasets id/path")
    parser.add_argument("--num-samples", type=int, default=None, help="None = full dataset")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--chat-template-kwargs", default=None, help="JSON dict")
    parser.add_argument("--max-output-length", type=int, required=True)
    parser.add_argument("--max-input-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", required=True, help="per-sample artifacts land here")
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    import torch
    from glm5_next_hf_gsm8k_reference import build_model

    import tensorrt_llm
    from tensorrt_llm.evaluate import GSM8K
    from tensorrt_llm.sampling_params import SamplingParams

    assert tensorrt_llm.__file__.startswith(_REPO), (
        f"stale tensorrt_llm package resolved: {tensorrt_llm.__file__}"
    )

    started = time.time()
    chat_template_kwargs = (
        json.loads(args.chat_template_kwargs) if args.chat_template_kwargs else None
    )
    summary: Dict[str, Any] = {
        "config": {
            "checkpoint": args.checkpoint,
            "harness": "tensorrt_llm.evaluate.GSM8K (same class trtllm-eval uses)",
            "engine": "native HF model.generate (AutoModelForImageTextToText, device_map=auto)",
            "dataset_path": args.dataset_path,
            "num_samples": args.num_samples,
            "random_seed": args.random_seed,
            "num_fewshot": args.num_fewshot,
            "apply_chat_template": True,
            "system_prompt": args.system_prompt,
            "chat_template_kwargs": chat_template_kwargs,
            "max_output_length": args.max_output_length,
            "max_input_length": args.max_input_length,
            "batch_size": args.batch_size,
            "decode": {"do_sample": False, "num_beams": 1, "native_generate": True},
        },
        "package": {
            "tensorrt_llm_version": tensorrt_llm.__version__,
            "torch_version": torch.__version__,
        },
        "provenance": {"driver_sha256": sha256_of(os.path.abspath(__file__))},
        "ok": False,
        "problems": ["run did not complete"],
    }

    try:
        tokenizer, _config, model = build_model(args.checkpoint)
        summary["load_seconds"] = round(time.time() - started, 1)
        import transformers

        summary["package"]["transformers_version"] = transformers.__version__

        evaluator = GSM8K(
            dataset_path=args.dataset_path,
            num_samples=args.num_samples,
            random_seed=args.random_seed,
            apply_chat_template=True,
            fewshot_as_multiturn=False,
            num_fewshot=args.num_fewshot,
            system_prompt=args.system_prompt,
            chat_template_kwargs=chat_template_kwargs,
            log_samples=True,
            output_path=args.output_dir,
        )
        runner = _BatchedHfRunner(
            model, tokenizer, batch_size=args.batch_size, max_prompt_tokens=args.max_input_length
        )
        sampling_params = SamplingParams(
            max_tokens=args.max_output_length,
            truncate_prompt_tokens=args.max_input_length,
            temperature=0,
            top_k=1,
        )
        score = evaluator.evaluate(runner, sampling_params, sampling_override=True)

        truncated = [i for i, r in enumerate(runner.completed_rows) if r["truncated"]]
        summary["score_mean_of_filters"] = float(score)
        summary["num_rows"] = len(runner.completed_rows)
        summary["truncated_rows"] = truncated
        summary["max_generated_tokens"] = max(
            (r["generated_tokens"] for r in runner.completed_rows), default=0
        )
        summary["stop_string_hits"] = sum(1 for r in runner.completed_rows if r["stop_string_hit"])
        problems = []
        if truncated:
            problems.append(f"{len(truncated)} rows hit the {args.max_output_length}-token budget")
        summary["problems"] = problems
        summary["ok"] = not problems
        print(
            f"[lm-eval-hf] score(mean of filters) {score:.2f} rows {len(runner.completed_rows)} "
            f"truncated {len(truncated)} ok={summary['ok']}",
            flush=True,
        )
    except BaseException as exc:  # noqa: BLE001
        summary["error"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[
            -4000:
        ]
        summary["problems"] = [f"exception: {type(exc).__name__}: {exc}"]
        print(f"[lm-eval-hf] FAILED: {type(exc).__name__}: {exc}", flush=True)
    finally:
        summary["total_seconds"] = round(time.time() - started, 1)
        os.makedirs(os.path.dirname(os.path.abspath(args.summary)), exist_ok=True)
        with open(args.summary, "w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        code = 0 if summary.get("ok") else 1
        with open(args.summary + ".exit.txt", "w") as fh:
            fh.write(f"{code}\n")
        print(f"[lm-eval-hf] wrote {args.summary} (exit {code})", flush=True)
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
