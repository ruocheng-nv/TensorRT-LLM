# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Produce the native-HuggingFace reference fixture for GLM-5.3-Flash.

This is rung one of the reference ladder.  The TensorRT-LLM implementation is
later compared against the artifacts written here, so this script must use the
*native* HuggingFace path only:

* ``AutoModelForCausalLM.from_pretrained`` on the real checkpoint -- the repo's
  pinned ``transformers`` (5.16.0.dev0) contains ``glm5_next``, so the model's
  own ``generate()`` runs unmodified.  No hand-written prefill/decode loop is
  involved, so no separate golden-generate fixture is required to anchor one.
* the checkpoint's own tokenizer and chat template, with no shims, monkeypatches
  or edits to the installed ``transformers``.

It writes three things:

``prompts``      the fixed prompt set, rendered token ids, and decode settings
``logits``       final-position logits and greedy argmax for every prompt
``generation``   per-step logits and greedy tokens for ``--max-new-tokens`` steps
``activations``  hidden states entering representative decoder layers and their
                 attention / MLP sub-blocks, for later source_activation_replay

Run it directly:

    python glm5_next_hf_reference.py --out /path/to/fixture.pt
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

import torch

DEFAULT_CHECKPOINT = "/dev/shm/GLM-5.3-Flash"

# Five fixed prompts.  Two are GSM8K-shaped so the fixture exercises the same
# reasoning behaviour the accuracy gate scores; three are short general prompts
# that keep the parity comparison cheap.
FIXED_PROMPTS: List[str] = [
    "What is the capital of France? Answer in one word.",
    "Natalia sold clips to 48 of her friends in April, and then she sold half "
    "as many clips in May. How many clips did Natalia sell altogether in April "
    "and May?",
    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes "
    "of babysitting. How much did she earn?",
    "List the first five prime numbers, separated by commas.",
    "Explain in two sentences why the sky appears blue.",
]

# Layers chosen from the literal schedules: first/middle/last of each attention
# type, plus a dense MLP layer and an early and a late routed MLP layer.
DEFAULT_CAPTURE_LAYERS = [0, 2, 3, 22, 23, 43, 44]

# Long-context mode.  The pool indexer selects `index_topk / index_kpool` = 512
# compressed pools, so its selection budget only *binds* above index_topk = 2048
# valid tokens; below that every causal candidate survives and the top-k stage
# is a no-op.  The five fixed prompts are 23-50 tokens, so they exercise the
# indexer only in its degenerate regime.
#
# These paragraphs are concatenated with substituted numbers until the rendered
# prompt crosses the requested length.  Substitution matters: repeating one
# identical paragraph would give the model a degenerate, highly periodic context
# whose attention and pool-score distribution is not representative.  Varying
# the quantities keeps the text coherent while keeping every paragraph distinct.
LONG_CONTEXT_PARAGRAPHS: List[str] = [
    "A delivery company operates {a} vans. Each van completes {b} routes per "
    "day, and every route covers {c} kilometres. The company pays a fuel "
    "allowance of ${d} for each kilometre driven. Work out the total fuel "
    "allowance the company pays in a single day, and then state how much it "
    "would pay over a working week of five days.",
    "A library received a donation of {a} books. Librarians placed {b} of them "
    "on the ground floor, and split the remainder evenly between {c} reading "
    "rooms upstairs. Each reading room already held {d} books before the "
    "donation arrived. How many books does each upstairs reading room hold "
    "once the donation has been shelved?",
    "During a science fair, {a} students each built {b} models. Judges awarded "
    "{c} points for every model that used recycled material, and {d} points "
    "for every other model. If exactly half of all the models used recycled "
    "material, how many points did the judges award in total?",
    "A bakery sells loaves for ${a} each and rolls for ${b} each. On Saturday "
    "it sold {c} loaves and {d} rolls, and on Sunday it sold twice as many "
    "loaves but only half as many rolls. Calculate the bakery's total takings "
    "across the two days.",
    "The blue colour of the daytime sky comes from Rayleigh scattering. Air "
    "molecules are far smaller than the wavelength of visible light, and the "
    "amount of light they scatter rises steeply as wavelength falls, roughly "
    "as the inverse fourth power. Short-wavelength blue light is therefore "
    "redirected across the sky far more strongly than long-wavelength red "
    "light, which mostly continues straight through. Near sunset the path "
    "through the atmosphere lengthens, so much of the blue is scattered away "
    "before the light reaches an observer and the remaining direct beam looks "
    "orange or red.",
    "A train leaves a station carrying {a} passengers. At the first stop {b} "
    "passengers get off and {c} get on. At the second stop a quarter of the "
    "passengers then aboard get off, and nobody boards. At the third stop {d} "
    "passengers get on. How many passengers are aboard when the train leaves "
    "the third stop?",
]


# --------------------------------------------------------------------------
# Correcting HuggingFace's exclusion normalization for this checkpoint
# --------------------------------------------------------------------------
#
# `FineGrainedFP8HfQuantizer._normalize_modules_to_not_convert` rewrites the
# checkpoint's `modules_to_not_convert` entries through the model's weight
# renamings, then `should_convert_module` keeps a module in BF16 when the
# pattern prefix-matches or the module name ends with it.
#
# For GLM-5.3-Flash that pipeline drops exactly two modules.  The renaming rule
# for the forget gate is written for *weight keys* and so requires a trailing
# dot::
#
#     WeightRenaming(r"self_attn\.f_a_proj\.", r"self_attn.forget_gate.f_a_proj.")
#
# but the skip-list entries are *module paths* with no trailing dot
# (`model.layers.0.self_attn.f_a_proj`), so the rename never fires.  The entry
# stays `model.layers.N.self_attn.f_a_proj` while the real module is
# `model.language_model.layers.N.self_attn.forget_gate.f_a_proj`, which neither
# prefix-matches nor ends with it.
#
# The consequence is not benign: those modules become `FP8Linear`, whose
# `weight` parameter is declared `float8_e4m3fn` and whose `weight_scale_inv` is
# `torch.empty(...)`.  The checkpoint's BF16 forget-gate weights get cast to
# e4m3 and paired with uninitialised scales -- HuggingFace reports them as
# "MISSING ... newly initialized".  That silently corrupts the forget gate of
# all 34 KDA layers, which would make this "reference" worthless.
#
# The fix uses HuggingFace's own public API -- pass a corrected
# `modules_to_not_convert` to `from_pretrained` -- and adds the two bare module
# names, which `should_convert_module` matches via its `endswith` rule.  It is
# provably the right call: the checkpoint carries no `weight_scale_inv` for
# these tensors, so they are BF16 by the checkpoint's own account.  Nothing in
# the installed `transformers` is patched, shimmed, or edited.
EXTRA_MODULES_TO_NOT_CONVERT = ["f_a_proj", "f_b_proj"]


def _build_quantization_config(config):
    """Return a FineGrainedFP8Config with the corrected exclusion list."""
    from transformers import FineGrainedFP8Config

    raw = getattr(config, "quantization_config", None)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raw = raw.to_dict() if hasattr(raw, "to_dict") else dict(vars(raw))

    skip = list(raw.get("modules_to_not_convert") or [])
    added = [name for name in EXTRA_MODULES_TO_NOT_CONVERT if name not in skip]
    skip.extend(added)
    print(
        f"[ref] modules_to_not_convert: {len(skip) - len(added)} from checkpoint "
        f"+ {len(added)} corrective {added}",
        flush=True,
    )
    return FineGrainedFP8Config(
        activation_scheme=raw.get("activation_scheme", "dynamic"),
        weight_block_size=tuple(raw.get("weight_block_size", (128, 128))),
        modules_to_not_convert=skip,
    )


def verify_quantization_split(model, checkpoint_path: str) -> Dict[str, Any]:
    """Assert the loaded FP8/BF16 split matches the checkpoint exactly.

    Every module HuggingFace turned into an ``FP8Linear`` must have a real
    ``weight_scale_inv`` in the checkpoint, and every module it left alone must
    not.  This is what catches an exclusion-matching regression before it can
    quietly poison the reference, instead of relying on a load-report warning
    that is easy to scroll past.
    """
    from transformers.integrations.finegrained_fp8 import FP8Linear

    index_file = os.path.join(checkpoint_path, "model.safetensors.index.json")
    with open(index_file) as fh:
        weight_map = json.load(fh)["weight_map"]

    fp8_without_scale: List[str] = []
    bf16_with_scale: List[str] = []
    num_fp8 = num_plain = 0
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        key = f"{name}.weight"
        if key not in weight_map:
            continue
        has_scale = f"{key}_scale_inv" in weight_map
        if isinstance(module, FP8Linear):
            num_fp8 += 1
            if not has_scale:
                fp8_without_scale.append(name)
        else:
            num_plain += 1
            if has_scale:
                bf16_with_scale.append(name)

    if fp8_without_scale or bf16_with_scale:
        raise RuntimeError(
            "loaded FP8/BF16 split disagrees with the checkpoint:\n"
            f"  quantized but has no checkpoint scale ({len(fp8_without_scale)}): "
            f"{fp8_without_scale[:6]}\n"
            f"  left in BF16 but has a checkpoint scale ({len(bf16_with_scale)}): "
            f"{bf16_with_scale[:6]}"
        )
    return {"fp8_linear_modules": num_fp8, "bf16_linear_modules": num_plain}


def _resolve_layers(model) -> torch.nn.ModuleList:
    """Return the text decoder layer list, whatever the wrapper depth is."""
    for path in (
        ("model", "language_model", "layers"),
        ("model", "layers"),
        ("language_model", "layers"),
    ):
        node: Any = model
        ok = True
        for attr in path:
            if not hasattr(node, attr):
                ok = False
                break
            node = getattr(node, attr)
        if ok:
            return node
    raise RuntimeError("could not locate the text decoder layer list on the model")


class ActivationCapture:
    """Capture the inputs *and* outputs of selected decoder layers and sub-blocks.

    Inputs are what a replay test feeds to its own implementation of the module;
    outputs are what the in-model HuggingFace module actually produced for that
    input, so a standalone reconstruction can be checked against the real thing
    rather than only against another standalone build.
    """

    DEFAULT_MODULES = ("self_attn", "mlp", "input_layernorm", "post_attention_layernorm")

    def __init__(self, layers, layer_ids: List[int], modules: tuple[str, ...] | None = None):
        self.layer_ids = layer_ids
        self.modules = tuple(modules) if modules is not None else self.DEFAULT_MODULES
        self.handles = []
        self.store: Dict[str, torch.Tensor] = {}
        self._enabled = False
        for idx in layer_ids:
            layer = layers[idx]
            targets = [(f"layer{idx}", layer)]
            for attr in self.modules:
                if hasattr(layer, attr):
                    targets.append((f"layer{idx}.{attr}", getattr(layer, attr)))
            for name, module in targets:
                self.handles.append(
                    module.register_forward_pre_hook(
                        self._make_pre_hook(f"{name}.input"), with_kwargs=True
                    )
                )
                self.handles.append(
                    module.register_forward_hook(self._make_post_hook(f"{name}.output"))
                )

    @staticmethod
    def _first_tensor(value) -> torch.Tensor | None:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (tuple, list)):
            for item in value:
                if isinstance(item, torch.Tensor):
                    return item
        return None

    def _save(self, name: str, tensor) -> None:
        if not self._enabled or name in self.store:
            return
        tensor = self._first_tensor(tensor)
        if isinstance(tensor, torch.Tensor):
            self.store[name] = tensor.detach().to("cpu", torch.float32).clone()

    def _make_pre_hook(self, name: str):
        def hook(module, args, kwargs=None):
            tensor = args[0] if args else None
            if tensor is None and kwargs:
                for key in ("hidden_states", "hidden_streams", "x"):
                    if key in kwargs:
                        tensor = kwargs[key]
                        break
            self._save(name, tensor)
            return None

        return hook

    def _make_post_hook(self, name: str):
        def hook(module, args, output):
            self._save(name, output)
            return None

        return hook

    def start(self) -> None:
        self.store = {}
        self._enabled = True

    def stop(self) -> Dict[str, torch.Tensor]:
        self._enabled = False
        return self.store

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def build_long_context_prompt(tokenizer, min_tokens: int) -> tuple[str, str]:
    """Return ``(prompt, rendered)`` for a coherent prompt of >= ``min_tokens``.

    Paragraphs cycle through ``LONG_CONTEXT_PARAGRAPHS`` with the embedded
    quantities substituted per repetition, so no two paragraphs are identical.
    The rendered length is measured with the checkpoint's own chat template
    after every addition, so the returned prompt is guaranteed to clear the
    threshold rather than merely estimated to.
    """
    parts: List[str] = [
        "Read the following collection of exercises and notes carefully. "
        "Afterwards, answer only the final question.",
    ]
    step = 0
    while True:
        template = LONG_CONTEXT_PARAGRAPHS[step % len(LONG_CONTEXT_PARAGRAPHS)]
        parts.append(
            f"Item {step + 1}. "
            + template.format(a=17 + step * 3, b=4 + (step % 7), c=9 + (step % 5), d=2 + step)
        )
        step += 1
        if step % len(LONG_CONTEXT_PARAGRAPHS) == 0:
            prompt = "\n\n".join(parts + [FINAL_LONG_CONTEXT_QUESTION])
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            n = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
            if n >= min_tokens:
                return prompt, rendered
        if step > 4000:  # unreachable for any sane min_tokens; guards the loop
            raise RuntimeError(f"could not reach {min_tokens} tokens")


FINAL_LONG_CONTEXT_QUESTION = (
    "Now answer only this: in Item 1, what is the total fuel allowance the "
    "delivery company pays in a single day?"
)


def _run_long_context(args, model, tokenizer, device, layers, load_seconds, split, config) -> int:
    """Capture one long prefill so the k-pool indexer runs where its budget binds."""
    prompt, rendered = build_long_context_prompt(tokenizer, args.long_context_tokens)
    enc = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"].to(device)
    n_tokens = int(ids.shape[1])
    print(f"[ref] long-context prompt: {n_tokens} tokens", flush=True)

    capture = ActivationCapture(layers, args.capture_layers, modules=args.capture_modules)
    capture.start()
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    activations = capture.stop()
    capture.close()

    final_logits = out.logits[0, -1].detach().to("cpu", torch.float32)
    greedy = int(final_logits.argmax())
    text_cfg = config.text_config
    fixture: Dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "mode": "long_context_prefill",
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "capture_layers": args.capture_layers,
        "capture_modules": list(capture.modules),
        "index_topk": text_cfg.index_topk,
        "index_kpool": text_cfg.index_kpool,
        "load_seconds": load_seconds,
        "quantization_split": split,
        "extra_modules_to_not_convert": EXTRA_MODULES_TO_NOT_CONVERT,
        "prompts": [
            {
                "index": 0,
                "prompt": prompt,
                "rendered": rendered,
                "input_ids": ids[0].detach().cpu(),
                "num_input_tokens": n_tokens,
                "prefill_final_logits": final_logits,
                "prefill_greedy_token": greedy,
                "activations": activations,
            }
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(fixture, args.out)
    print(
        f"[ref] wrote long-context fixture to {args.out} "
        f"({len(activations)} activations, greedy={greedy!r} -> "
        f"{tokenizer.decode([greedy])!r})",
        flush=True,
    )
    if args.summary:
        with open(args.summary, "w") as fh:
            json.dump(
                {
                    "checkpoint": args.checkpoint,
                    "mode": "long_context_prefill",
                    "transformers_version": fixture["transformers_version"],
                    "torch_version": fixture["torch_version"],
                    "num_input_tokens": n_tokens,
                    "index_topk": text_cfg.index_topk,
                    "index_kpool": text_cfg.index_kpool,
                    "select_k": text_cfg.index_topk // text_cfg.index_kpool,
                    "pool_budget_binds": n_tokens > text_cfg.index_topk,
                    "capture_layers": args.capture_layers,
                    "capture_modules": list(capture.modules),
                    "captured_activations": sorted(activations.keys()),
                    "prefill_greedy_token": greedy,
                    "prefill_greedy_text": tokenizer.decode([greedy]),
                    "prefill_logits_finite": bool(torch.isfinite(final_logits).all()),
                    "prefill_logits_absmax": float(final_logits.abs().max()),
                    "load_seconds": load_seconds,
                },
                fh,
                indent=2,
            )
        print(f"[ref] wrote summary to {args.summary}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", required=True, help="path for the .pt fixture")
    parser.add_argument("--summary", default=None, help="optional .json summary path")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--capture-layers",
        type=int,
        nargs="*",
        default=DEFAULT_CAPTURE_LAYERS,
    )
    parser.add_argument(
        "--capture-prompts",
        type=int,
        default=2,
        help="capture layer activations for the first N prompts (fixture size)",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="load the model and run one forward, then exit (feasibility check)",
    )
    parser.add_argument(
        "--long-context-tokens",
        type=int,
        default=0,
        help=(
            "when > 0, ignore the fixed prompt set and instead capture a single "
            "prefill on a coherent prompt of at least this many tokens, so the "
            "pool indexer runs in the regime where its 512-pool budget binds"
        ),
    )
    parser.add_argument(
        "--capture-modules",
        nargs="*",
        default=None,
        help=(
            "restrict the hooked sub-modules (default: self_attn mlp "
            "input_layernorm post_attention_layernorm). Long-context "
            "activations are large, so narrowing this controls fixture size."
        ),
    )
    args = parser.parse_args()

    # GLM-5.3-Flash declares Glm5NextForConditionalGeneration, so its native auto
    # class is AutoModelForImageTextToText, not AutoModelForCausalLM. We drive it
    # with text-only inputs; the vision tower is constructed but never invoked.
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

    started = time.time()
    print(f"[ref] loading tokenizer from {args.checkpoint}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    config = AutoConfig.from_pretrained(args.checkpoint)

    print("[ref] loading model (device_map=auto) ...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.checkpoint,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        quantization_config=_build_quantization_config(config),
    )
    model.eval()
    load_seconds = time.time() - started
    print(f"[ref] model loaded in {load_seconds:.1f}s", flush=True)

    split = verify_quantization_split(model, args.checkpoint)
    print(
        f"[ref] quantization split verified against checkpoint: "
        f"{split['fp8_linear_modules']} FP8Linear, {split['bf16_linear_modules']} BF16 linear",
        flush=True,
    )

    device = next(model.parameters()).device
    layers = _resolve_layers(model)
    print(f"[ref] decoder layers: {len(layers)}", flush=True)

    # Render prompts with the checkpoint's own chat template. The template
    # unconditionally appends '<think>' after '<|assistant|>', so thinking mode
    # is always on for this checkpoint -- there is no enable_thinking toggle.
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in FIXED_PROMPTS
    ]
    encoded = [tokenizer(text, return_tensors="pt", add_special_tokens=False) for text in rendered]

    if args.probe_only:
        ids = encoded[0]["input_ids"].to(device)
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=False)
        print(
            f"[ref] probe ok: logits {tuple(out.logits.shape)} "
            f"finite={bool(torch.isfinite(out.logits).all())} "
            f"argmax={int(out.logits[0, -1].argmax())}",
            flush=True,
        )
        return 0

    if args.long_context_tokens > 0:
        return _run_long_context(
            args, model, tokenizer, device, layers, load_seconds, split, config
        )

    capture = ActivationCapture(layers, args.capture_layers, modules=args.capture_modules)
    fixture: Dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "decode": {
            "do_sample": False,
            "num_beams": 1,
            "temperature": None,
            "top_k": None,
            "top_p": None,
            "max_new_tokens": args.max_new_tokens,
        },
        "eos_token_id": config.text_config.eos_token_id,
        "capture_layers": args.capture_layers,
        "prompts": [],
        "load_seconds": load_seconds,
        "quantization_split": split,
        "extra_modules_to_not_convert": EXTRA_MODULES_TO_NOT_CONVERT,
    }

    summary_rows = []
    for i, (prompt, text, enc) in enumerate(zip(FIXED_PROMPTS, rendered, encoded)):
        ids = enc["input_ids"].to(device)
        attention_mask = torch.ones_like(ids)
        print(f"[ref] prompt {i}: {ids.shape[1]} tokens", flush=True)

        # Activations are only needed from a couple of prompts -- they dominate
        # the fixture size and the replay coverage comes from the layer/type
        # spread, not from the prompt count.
        want_activations = i < args.capture_prompts
        if want_activations:
            capture.start()
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=attention_mask, use_cache=False)
        activations = capture.stop() if want_activations else {}
        final_logits = out.logits[0, -1].detach().to("cpu", torch.float32)
        greedy = int(final_logits.argmax())

        with torch.no_grad():
            gen = model.generate(
                input_ids=ids,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.max_new_tokens,
                return_dict_in_generate=True,
                output_logits=True,
            )
        step_logits = torch.stack(
            [logit[0].detach().to("cpu", torch.float32) for logit in gen.logits]
        )
        new_tokens = gen.sequences[0, ids.shape[1] :].detach().cpu()

        fixture["prompts"].append(
            {
                "index": i,
                "prompt": prompt,
                "rendered": text,
                "input_ids": ids[0].detach().cpu(),
                "prefill_final_logits": final_logits,
                "prefill_greedy_token": greedy,
                "generated_token_ids": new_tokens,
                "generated_step_logits": step_logits,
                "generated_text": tokenizer.decode(new_tokens, skip_special_tokens=False),
                "activations": activations,
            }
        )
        summary_rows.append(
            {
                "index": i,
                "prompt": prompt,
                "num_input_tokens": int(ids.shape[1]),
                "prefill_greedy_token": greedy,
                "prefill_greedy_text": tokenizer.decode([greedy]),
                "prefill_logits_finite": bool(torch.isfinite(final_logits).all()),
                "prefill_logits_absmax": float(final_logits.abs().max()),
                "num_generated": int(new_tokens.numel()),
                "generated_prefix": tokenizer.decode(new_tokens[:24], skip_special_tokens=False),
                "step_logits_finite": bool(torch.isfinite(step_logits).all()),
                "captured_activations": sorted(activations.keys()),
            }
        )
        print(
            f"[ref]   greedy={greedy!r} -> {summary_rows[-1]['prefill_greedy_text']!r}", flush=True
        )

    capture.close()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(fixture, args.out)
    print(f"[ref] wrote fixture to {args.out}", flush=True)

    if args.summary:
        with open(args.summary, "w") as fh:
            json.dump(
                {
                    "checkpoint": args.checkpoint,
                    "transformers_version": fixture["transformers_version"],
                    "torch_version": fixture["torch_version"],
                    "decode": fixture["decode"],
                    "eos_token_id": fixture["eos_token_id"],
                    "capture_layers": args.capture_layers,
                    "load_seconds": load_seconds,
                    "total_seconds": time.time() - started,
                    "prompts": summary_rows,
                },
                fh,
                indent=2,
            )
        print(f"[ref] wrote summary to {args.summary}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
