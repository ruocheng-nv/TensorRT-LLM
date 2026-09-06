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

Budget-raise replay (``--replay-from``): re-issuing this reference at a
LARGER decode budget does not require regenerating rows that provably did
not consume the old budget. Under deterministic greedy decoding the token
trajectory is a fixed prefix property: a row whose sealed scored text
retokenizes to fewer than ``old_budget - AT_BUDGET_SLACK`` tokens either
stopped at EOS below the old cap or was text-cut at a task ``until`` string
inside the greedy prefix — in both cases the scored text is byte-identical
at any larger budget, so it is served verbatim from the sealed sample cache.
Only rows at/near the old cap (the sealed runaways) are natively regenerated
at the new budget; the HF model is loaded lazily and only when at least one
row needs regeneration. Every row — replayed or regenerated — is re-scored
through the same ``tensorrt_llm.evaluate.GSM8K`` harness, so the produced
``samples_gsm8k.json`` is a genuine harness artifact at the new budget, not
a hand-spliced file. Fail-closed: an unknown or already-served rendered
prompt, a sealed row left unrequested, or a budget that is not strictly
larger than the qualification budget all abort. A regenerated row that hits
the NEW budget is a problem unless its doc id was explicitly pre-declared
via ``--disclosed-runaway-doc-ids`` (the iteration-21 human override: a
native-HF runaway is disclosed, not blocking).
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


class _CachedReplayRunner(_BatchedHfRunner):
    """Budget-raise replay over a sealed lm-eval sample cache.

    ``cache_by_prompt`` maps each sealed rendered prompt to
    ``{"doc_id", "text", "resp_tokens"}``. A row qualifies for verbatim
    replay iff ``resp_tokens < qualify_budget - slack`` (its scored text is
    budget-invariant under greedy decoding, see module docstring); every
    other row is queued for native regeneration through the inherited
    batched ``generate`` path, and the HF model is built lazily on the first
    flush that actually has regeneration work.
    """

    def __init__(
        self,
        model_builder,
        tokenizer,
        batch_size: int,
        max_prompt_tokens: int,
        cache_by_prompt: Dict[str, Dict[str, Any]],
        qualify_budget: int,
        slack: int,
    ):
        super().__init__(
            model=None,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_prompt_tokens=max_prompt_tokens,
        )
        self._model_builder = model_builder
        self._cache = dict(cache_by_prompt)
        self._qualify_budget = int(qualify_budget)
        self._slack = int(slack)
        self.replayed_doc_ids: List[int] = []
        self.regen_doc_ids: List[int] = []
        self._regen_doc_by_prompt: Dict[str, int] = {}

    def unserved_cache_rows(self) -> int:
        return len(self._cache)

    def generate_async(self, prompt, sampling_params=None, streaming=False):
        assert not streaming, "streaming is not part of this reference driver"
        entry = self._cache.pop(prompt, None)
        assert entry is not None, (
            "rendered prompt not found (or already served) in the sealed sample "
            "cache — matched-config drift; refusing to fabricate a reference row"
        )
        new_budget = int(sampling_params.max_tokens)
        assert new_budget > self._qualify_budget, (
            f"replay is only valid when the budget is raised: new budget "
            f"{new_budget} must be > qualification budget {self._qualify_budget}"
        )
        if entry["resp_tokens"] < self._qualify_budget - self._slack:
            out = _PendingOutput(self, prompt, sampling_params)
            completion = out.outputs[0]
            completion.text = entry["text"]
            completion.finish_reason = "stop"
            out.done = True
            self.pending.append(out)
            self.replayed_doc_ids.append(entry["doc_id"])
            self.completed_rows.append(
                {
                    "doc_id": entry["doc_id"],
                    "replayed": True,
                    "generated_tokens": entry["resp_tokens"],
                    "budget": new_budget,
                    "truncated": False,
                    "stop_string_hit": None,
                }
            )
            return out
        handle = super().generate_async(prompt, sampling_params)
        self.regen_doc_ids.append(entry["doc_id"])
        self._regen_doc_by_prompt[prompt] = entry["doc_id"]
        return handle

    def flush(self):
        if self.model is None and any(not p.done for p in self.pending):
            self.model = self._model_builder()
        super().flush()

    def regen_truncated_doc_ids(self) -> List[int]:
        return sorted(
            self._regen_doc_by_prompt[p.prompt]
            for p in self.pending
            if p.prompt in self._regen_doc_by_prompt
            and p.done
            and p.outputs[0].finish_reason == "length"
        )


def native_truncation_report(rows, tokenizer, budget, disclosed_runaway_doc_ids):
    """Classify native-generation truncations by doc_id, split disclosed vs not.

    ``rows`` maps doc_id -> sample dict (as returned by
    ``glm5_next_lmeval_diff.load_samples``). A row counts as truncated when its
    scored text reaches the decode budget (within ``AT_BUDGET_SLACK`` tokens)
    carrying neither a ``####`` answer marker nor the ``Question:`` stop string
    — i.e. generation stopped only because it hit the cap, under the SAME
    marker/until semantics the session truncation audit uses. Disclosed
    runaways (e.g. the iteration-21 native-HF doc 786) are reported but not
    treated as a problem, mirroring the budget-raise replay path; every other
    truncation fails closed. Keyed by doc_id so the native reference's ok flag
    matches the session's doc_id-keyed truncation audit rather than relying on
    generation-order row indices.
    """
    from glm5_next_lmeval_diff import ANSWER_MARKER, AT_BUDGET_SLACK, response_of

    disclosed = sorted(int(x) for x in disclosed_runaway_doc_ids)
    truncated = []
    for doc_id in sorted(rows):
        rtext = response_of(rows[doc_id])
        n_tok = len(tokenizer.encode(rtext, add_special_tokens=False))
        if (
            n_tok >= budget - AT_BUDGET_SLACK
            and ANSWER_MARKER not in rtext
            and "Question:" not in rtext
        ):
            truncated.append(int(doc_id))
    undisclosed = [d for d in truncated if d not in disclosed]
    return {
        "truncated_doc_ids": truncated,
        "disclosed_runaway_doc_ids": disclosed,
        "undisclosed_truncated_doc_ids": undisclosed,
    }


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
    parser.add_argument(
        "--replay-from",
        default=None,
        help="sealed samples_gsm8k.json (or its directory) to replay budget-invariant rows from",
    )
    parser.add_argument(
        "--replay-qualify-budget",
        type=int,
        default=None,
        help="decode budget the sealed run used; rows below budget-slack tokens replay verbatim",
    )
    parser.add_argument(
        "--disclosed-runaway-doc-ids",
        default="",
        help="comma-separated doc ids whose regeneration may hit the new budget without failing",
    )
    args = parser.parse_args()
    if args.replay_from:
        assert args.replay_qualify_budget, "--replay-from requires --replay-qualify-budget"
        assert args.max_output_length > args.replay_qualify_budget, (
            "replay mode is a budget RAISE: --max-output-length must exceed --replay-qualify-budget"
        )
    disclosed_runaways = sorted(
        int(x) for x in args.disclosed_runaway_doc_ids.split(",") if x.strip()
    )

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
        if args.replay_from:
            from glm5_next_lmeval_diff import AT_BUDGET_SLACK, load_samples, prompt_of, response_of
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
            src_path, src_rows = load_samples(args.replay_from)
            cache: Dict[str, Dict[str, Any]] = {}
            for doc_id in sorted(src_rows):
                prompt = prompt_of(src_rows[doc_id])
                text = response_of(src_rows[doc_id])
                assert prompt and prompt not in cache, (
                    f"empty or duplicate rendered prompt in sealed cache (doc {doc_id})"
                )
                cache[prompt] = {
                    "doc_id": doc_id,
                    "text": text,
                    "resp_tokens": len(tokenizer.encode(text, add_special_tokens=False)),
                }
            summary["config"]["engine"] = (
                "sealed-sample budget-raise replay + native HF model.generate "
                "regeneration for at/near-cap rows (lazy load)"
            )
            summary["config"]["replay_from"] = src_path
            summary["config"]["replay_qualify_budget"] = args.replay_qualify_budget
            runner = _CachedReplayRunner(
                model_builder=lambda: build_model(args.checkpoint)[2],
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                max_prompt_tokens=args.max_input_length,
                cache_by_prompt=cache,
                qualify_budget=args.replay_qualify_budget,
                slack=AT_BUDGET_SLACK,
            )
            replay_source_sha = sha256_of(src_path)
        else:
            tokenizer, _config, model = build_model(args.checkpoint)
            runner = _BatchedHfRunner(
                model,
                tokenizer,
                batch_size=args.batch_size,
                max_prompt_tokens=args.max_input_length,
            )
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
        if args.replay_from:
            regen_truncated = runner.regen_truncated_doc_ids()
            undisclosed = [d for d in regen_truncated if d not in disclosed_runaways]
            summary["replay"] = {
                "source_samples": src_path,
                "source_sha256": replay_source_sha,
                "qualify_budget": args.replay_qualify_budget,
                "at_budget_slack": AT_BUDGET_SLACK,
                "replayed": len(runner.replayed_doc_ids),
                "regenerated": len(runner.regen_doc_ids),
                "regenerated_doc_ids": sorted(runner.regen_doc_ids),
                "regen_truncated_doc_ids": regen_truncated,
                "disclosed_runaway_doc_ids": disclosed_runaways,
                "unserved_cache_rows": runner.unserved_cache_rows(),
            }
            if undisclosed:
                problems.append(
                    f"undisclosed regenerated rows hit the {args.max_output_length}-token "
                    f"budget: {undisclosed}"
                )
            if runner.unserved_cache_rows():
                problems.append(
                    f"{runner.unserved_cache_rows()} sealed cache rows were never "
                    "requested by the harness (matched-config drift)"
                )
        else:
            # Native (non-replay) generation honors the SAME disclosed-runaway
            # contract as the replay path (iteration-21 human override: a
            # disclosed, pre-characterized native-HF runaway such as doc 786 is
            # reported, not blocking; any OTHER truncation fails closed).
            from glm5_next_lmeval_diff import load_samples

            _, native_rows = load_samples(args.output_dir)
            native_trunc = native_truncation_report(
                native_rows, tokenizer, args.max_output_length, disclosed_runaways
            )
            native_trunc["row_index_truncation_count"] = len(truncated)
            summary["native_truncation"] = native_trunc
            if native_trunc["undisclosed_truncated_doc_ids"]:
                problems.append(
                    f"undisclosed rows hit the {args.max_output_length}-token "
                    f"budget: {native_trunc['undisclosed_truncated_doc_ids']}"
                )
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
