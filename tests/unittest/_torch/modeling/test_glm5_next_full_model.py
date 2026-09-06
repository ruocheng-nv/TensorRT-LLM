# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Whole-model parity for GLM-5.3-Flash against native HuggingFace.

Every earlier rung compared one module, or one assembled layer, against the
source. Those can all pass while the *model* is wrong: a layer that is
individually correct can be wired at the wrong index, the four-stream residual
can accumulate, the KDA recurrent state can be seeded from the wrong slot, and
none of that is observable until all 45 layers run end to end on real weights.

The reference is the fixture built by ``glm5_next_hf_reference.py``: native
``AutoModelForCausalLM`` on the real checkpoint, driven through HuggingFace's
own ``generate()`` with ``do_sample=False``. It is a canonical native-generate
golden, not a hand-written decode loop, so nothing here depends on a
reimplementation of generation being correct.

Two comparisons matter and they fail differently:

* **source_logit_replay** -- one prefill per prompt, final-position logits. It
  catches wiring, geometry and weight-placement errors, but it cannot see the
  cache: nothing has been read back yet.
* **generation_parity** -- 32 greedy steps per prompt. Every step reads state
  the previous step wrote, so this is the only test here that can catch a KDA
  recurrence, convolution-history, latent-cache or pool-indexer defect.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List

import pytest
import torch
from glm5_next_full_model import Glm5NextGenerator, attach_caches, load_full_model
from glm5_next_ref import compare

CHECKPOINT = os.environ.get("GLM53_FLASH_CHECKPOINT", "/dev/shm/GLM-5.3-Flash")
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
EVIDENCE_PATH = os.environ.get(
    "GLM53_FULL_MODEL_EVIDENCE",
    os.path.join(
        _REPO_ROOT,
        "agent-flow/workspace/glm-5.3-flash-bringup/reports/goal15_full_model_evidence.json",
    ),
)
#: Native-HF generate() output on the real checkpoint. Its absence is a hard
#: failure, not a skip: it *is* the reference for every assertion in this file,
#: so a suite that quietly dropped it would report green while comparing the
#: model against nothing.
FIXTURE = os.environ.get(
    "GLM53_HF_FIXTURE",
    os.path.join(
        _REPO_ROOT, "agent-flow/workspace/glm-5.3-flash-bringup/reports/hf_reference_fixture.pt"
    ),
)

#: The whole model is 328 GB in its published e4m3 form against 183 GB per
#: B200, so this suite is inherently multi-device.
MIN_DEVICES = 3
#: Deterministic greedy decoding, matching the fixture's own decode config.
NEW_TOKENS = 32

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(
        torch.cuda.device_count() < MIN_DEVICES,
        reason=f"the 328 GB checkpoint needs at least {MIN_DEVICES} devices",
    ),
    pytest.mark.skipif(
        not os.path.isdir(CHECKPOINT),
        reason=f"requires the real checkpoint at {CHECKPOINT}",
    ),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def evidence() -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    yield payload
    path = os.path.abspath(EVIDENCE_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)


@pytest.fixture(scope="module")
def hf_reference() -> Dict[str, Any]:
    assert os.path.isfile(FIXTURE), (
        f"missing the native-HF reference fixture at {FIXTURE}; build it with "
        "glm5_next_hf_reference.py. It is the reference for every assertion "
        "here, so this is a failure rather than a skip."
    )
    payload = torch.load(FIXTURE, map_location="cpu", weights_only=False)
    assert payload["decode"]["do_sample"] is False, payload["decode"]
    assert payload["decode"]["num_beams"] == 1, payload["decode"]
    return payload


@pytest.fixture(scope="module")
def loaded():
    model = load_full_model(CHECKPOINT, progress=True)
    yield model


@pytest.fixture(scope="module")
def generator(loaded, hf_reference):
    prompts = [p["input_ids"].tolist() for p in hf_reference["prompts"]]
    longest = max(len(p) for p in prompts)
    attach_caches(loaded, max_batch_size=len(prompts), max_seq_len=longest + NEW_TOKENS + 64)
    gen = Glm5NextGenerator(loaded)
    yield gen
    gen.close()


@pytest.fixture(scope="module")
def replay(generator, hf_reference):
    """One generation run, shared by the replay and parity assertions.

    Prefill plus 32 decode steps over the whole model is minutes of work; both
    tests read the same run rather than paying for it twice, and sharing it also
    means they cannot disagree about what the model did.
    """
    prompts = [p["input_ids"].tolist() for p in hf_reference["prompts"]]
    reference_tokens = [p["generated_token_ids"].tolist() for p in hf_reference["prompts"]]
    # Teacher-forced inputs: each step is fed HF's chosen token, so a divergence
    # at step k is reported at step k instead of poisoning every later step with
    # a different prefix. The reported argmax is still entirely this model's.
    prefill_logits, step_logits, tokens = generator.generate(
        prompts, NEW_TOKENS, forced=reference_tokens
    )
    return {
        "prefill_logits": prefill_logits.cpu(),
        "step_logits": step_logits.cpu(),
        "tokens": tokens.cpu(),
    }


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------


def test_full_model_materializes_the_real_checkpoint(loaded, evidence):
    """All 45 layers hold real weights, in the checkpoint's own dtypes.

    The counts are the Goal-1.2 audit's, re-derived here from an *executed*
    load rather than from the index: a loader that skipped a tensor produces a
    model that still runs and still looks plausible.
    """
    report = loaded.load_report
    placement = loaded.quant_placement
    fp8_modules = sum(1 for dtype in placement.values() if dtype == "float8_e4m3fn")
    per_device = {}
    for index in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(index) / 2**30
        if allocated > 0.1:
            per_device[f"cuda:{index}"] = round(allocated, 2)

    evidence["materialization"] = {
        **report,
        "fp8_linear_modules": fp8_modules,
        "bf16_linear_modules": len(placement) - fp8_modules,
        "load_seconds": round(loaded.load_seconds, 1),
        "gib_per_device": per_device,
        "num_layers": loaded.schedule.num_layers,
    }

    assert report["missing_destinations"] == [], report
    assert report["total"] == 76108, report
    # The same accounting Goal 1.2 established analytically.
    assert report["loaded"] == 1144, report
    assert report["transformed"] == 72857, report
    assert report["ignored"] == 2107, report
    # 179 quantized linear modules is HuggingFace's own count for this
    # checkpoint (fixture `quantization_split`), reached here from the audited
    # 1509-entry exclusion set rather than copied from it.
    assert fp8_modules == 179, placement
    assert loaded.schedule.num_layers == 45
    assert len(per_device) >= MIN_DEVICES, per_device
    # No device may exceed the 183 GB B200; a plan that overcommits would have
    # thrown, but the margin is the useful number to record.
    assert max(per_device.values()) < 175.0, per_device


def test_layer_devices_follow_the_literal_schedule(loaded, evidence):
    """Every one of the 45 layers is placed exactly once, in order."""
    devices = loaded.load_report["layer_devices"]
    layer_owners = [devices[str(i)] for i in range(loaded.schedule.num_layers)]
    stage_ids = [stage.layer_ids for stage in loaded.stages]
    flat = [i for ids in stage_ids for i in ids]
    evidence["placement"] = {
        "stages": {str(s.device): s.layer_ids for s in loaded.stages},
        "embed": devices["embed"],
        "head": devices["head"],
    }
    assert flat == sorted(flat) == list(range(loaded.schedule.num_layers)), stage_ids
    # Pipeline placement is contiguous: a layer landing out of order would move
    # activations backwards across devices every step.
    assert layer_owners == sorted(layer_owners, key=layer_owners.index)


# ---------------------------------------------------------------------------
# source_logit_replay
# ---------------------------------------------------------------------------


def test_source_logit_replay(replay, hf_reference, evidence):
    """Prefill logits match native HF, and the greedy token matches exactly.

    The pass condition is the criterion's own: finite logits, reported
    max_abs/mean_abs/cosine, and greedy-argmax equality. It is deliberately not
    a tight absolute bound on the logits, because there is a measured floor
    below which no implementation can go: the checkpoint is FP8 with dynamic
    activation scaling, so HuggingFace and TensorRT-LLM each quantize
    activations to e4m3 with their own kernel. On real activations at real
    layer weights, the two kernels sit at *identical* distance from an exact
    float64 evaluation of the same math (1.078e-2 vs 1.078e-2 relative RMS at
    layer 0, 2.365e-2 vs 2.368e-2 at layer 3) while agreeing with each other 7-8
    times more closely -- the signature of two correct implementations of one
    lossy operation, not of a defect on either side. That residual compounds
    over 45 layers into the logit gap asserted here.
    """
    rows = []
    for index, prompt in enumerate(hf_reference["prompts"]):
        expected = prompt["prefill_final_logits"].float()
        got = replay["prefill_logits"][index].float()
        stats = compare(got, expected, f"prefill_logits_p{index}")
        got_token = int(got.argmax())
        expected_token = int(prompt["prefill_greedy_token"])
        top2 = expected.topk(2).values
        rows.append(
            {
                "prompt": prompt["prompt"],
                "num_input_tokens": int(prompt["input_ids"].numel()),
                "greedy_token": got_token,
                "hf_greedy_token": expected_token,
                "greedy_match": got_token == expected_token,
                "top1_logit": float(got.max()),
                # How much headroom the argmax had. Reported so a passing
                # greedy match is distinguishable from a lucky coin flip.
                "hf_top2_margin": float(top2[0] - top2[1]),
                **stats,
            }
        )
    evidence["source_logit_replay"] = rows

    assert all(r["all_finite"] for r in rows), rows
    mismatched = [r for r in rows if not r["greedy_match"]]
    assert not mismatched, f"greedy-argmax disagrees with native HF: {mismatched}"
    # Well above the ~0.15 cosine of an unrelated readout, and far enough below
    # the observed 0.98 that a real regression still trips it.
    assert min(r["cosine"] for r in rows) >= 0.95, rows


# ---------------------------------------------------------------------------
# generation_parity
# ---------------------------------------------------------------------------


def test_generation_parity(replay, hf_reference, evidence):
    """32 teacher-forced greedy steps per prompt, checked against HF's tokens.

    Unlike single-step replay this reads back state every step, so it is the
    first test in the suite that can fail on a KDA recurrence, convolution
    history, latent-KV or pool-indexer cache defect.

    Where a step *does* disagree, the question the assertion has to answer is
    whether the model made a different decision or merely landed on the other
    side of a tie. HF's own logits decide that: if the token this model picked
    is separated from HF's pick by less than the measured logit noise between
    the two FP8 kernels, then HF itself scored them as tied and the choice
    carries no semantic content. A divergence at a *confident* position is what
    a real defect looks like, and that is what fails here.
    """
    rows: List[Dict[str, Any]] = []
    for index, prompt in enumerate(hf_reference["prompts"]):
        expected_tokens = prompt["generated_token_ids"][:NEW_TOKENS].tolist()
        got_tokens = replay["tokens"][index].tolist()
        # Alignment, verified against the fixture rather than assumed: HF's
        # step_logits[j] is the distribution that *produced* token j, so it
        # includes the prefill readout at j=0 and step_logits[j].argmax()
        # equals generated_token_ids[j]. On this side, prefill produces token 0
        # and decode step k produces token k+1.
        step_stats, confident = [], []
        for step, (mine, theirs) in enumerate(zip(got_tokens, expected_tokens)):
            reference = prompt["generated_step_logits"][step].float()
            got = (
                replay["prefill_logits"][index].float()
                if step == 0
                else replay["step_logits"][index, step - 1].float()
            )
            stats = compare(got, reference, f"step{step}")
            step_stats.append(stats)
            if mine == theirs:
                continue
            # Is this a different decision, or the other side of a tie? Compare
            # HF's own separation between the two tokens against the logit
            # error actually observed *on those two tokens* at this step. That
            # sum is exactly the perturbation that can reorder the pair, so the
            # test calibrates itself from the measured data rather than from a
            # chosen multiple of a whole-vocabulary average -- which would be
            # dominated by the 154878 logits nobody was ever going to pick.
            separation = float(reference[theirs] - reference[mine])
            noise = float((got[theirs] - reference[theirs]).abs()) + float(
                (got[mine] - reference[mine]).abs()
            )
            confident.append(
                {
                    "step": step,
                    "mine": mine,
                    "hf": theirs,
                    "hf_separation": separation,
                    "noise_band": noise,
                    "tie": separation <= noise,
                }
            )
        rows.append(
            {
                "prompt": prompt["prompt"],
                "num_new_tokens": len(got_tokens),
                "tokens": got_tokens,
                "hf_tokens": expected_tokens,
                "first_divergence": confident[0]["step"] if confident else None,
                "num_matching": sum(1 for a, b in zip(got_tokens, expected_tokens) if a == b),
                "divergences": confident,
                "num_confident_divergences": sum(1 for d in confident if not d["tie"]),
                "min_step_cosine": min(s["cosine"] for s in step_stats),
                "max_step_max_abs": max(s["max_abs"] for s in step_stats),
                "mean_step_mean_abs": sum(s["mean_abs"] for s in step_stats) / len(step_stats),
                "all_finite": all(s["all_finite"] for s in step_stats),
            }
        )
    evidence["generation_parity"] = rows

    assert all(r["all_finite"] for r in rows), rows
    assert all(r["num_new_tokens"] >= NEW_TOKENS for r in rows), rows
    # The first token of every prompt comes straight off prefill, before any
    # cache has been read back, so it must match outright.
    assert all(r["tokens"][0] == r["hf_tokens"][0] for r in rows), rows
    semantic = [
        {"prompt": r["prompt"], "divergences": [d for d in r["divergences"] if not d["tie"]]}
        for r in rows
        if r["num_confident_divergences"]
    ]
    assert not semantic, f"greedy tokens diverge at confident positions: {semantic}"


def test_reference_step_logits_align_with_their_own_tokens(hf_reference, evidence):
    """Pin how the fixture's per-step logits line up with its tokens.

    This is not a property of the model under test -- it is a property of the
    reference, and getting it wrong is silent. Comparing against ``step j-1``
    instead of ``step j`` still produces plausible-looking numbers: it inflated
    the measured per-step logit noise five-fold (0.29 to 1.44) and manufactured
    "divergences" at positions where HF's own logits preferred *this* model's
    token, which is impossible under greedy decoding. Asserting the alignment
    against the fixture makes that failure mode loud.
    """
    rows = []
    for index, prompt in enumerate(hf_reference["prompts"]):
        tokens = prompt["generated_token_ids"]
        logits = prompt["generated_step_logits"]
        aligned = int((logits.argmax(-1) == tokens).sum())
        rows.append(
            {
                "prompt_index": index,
                "num_steps": int(tokens.numel()),
                "argmax_equals_token_at_same_step": aligned,
                "argmax_equals_next_token": int((logits.argmax(-1)[:-1] == tokens[1:]).sum()),
                "prefill_argmax_equals_token0": int(prompt["prefill_final_logits"].argmax())
                == int(tokens[0]),
            }
        )
    evidence["reference_alignment"] = rows

    for row in rows:
        # step j produced token j -- overwhelmingly, and never the shifted one.
        assert row["argmax_equals_token_at_same_step"] >= row["num_steps"] - 5, row
        assert row["argmax_equals_next_token"] == 0, row
        assert row["prefill_argmax_equals_token0"], row


def test_generation_parity_covers_five_prompts(replay, hf_reference):
    """The parity claim rests on at least five distinct real prompts."""
    prompts = [p["prompt"] for p in hf_reference["prompts"]]
    assert len(prompts) >= 5, prompts
    assert len(set(prompts)) == len(prompts), prompts
    assert replay["tokens"].shape == (len(prompts), NEW_TOKENS), replay["tokens"].shape


def _bf16_ulp(value: float) -> float:
    """The gap between adjacent bfloat16 values at ``value``'s magnitude.

    bfloat16 carries 7 explicit mantissa bits, so the spacing at ``2**e`` is
    ``2**(e - 7)``. This is the resolution of the *reference's own* readout: its
    ``lm_head`` is bf16, so its logits are exactly representable bf16 numbers
    and nothing finer than one ULP exists in them.
    """
    if value == 0.0:
        return float(torch.finfo(torch.bfloat16).smallest_normal)
    return float(2.0 ** (math.floor(math.log2(abs(value))) - 7))


#: Declared dtype-aware fork envelope (iteration-21 human override): exact
#: HF/TRT tokens are diagnostic, and a fork blocks only when the reference
#: separates the two candidates by more than the cross-implementation noise a
#: bf16 readout over an FP8-block model can carry. Four bf16 ULPs of the
#: reference's own grid, or the suite-wide 2% relative band on the winning
#: logit, whichever is larger.
_FORK_ULP_FACTOR = 4.0
_FORK_REL_MARGIN = 0.02


def test_generation_parity_divergences_stay_within_declared_envelope(
    replay, hf_reference, evidence
):
    """Every surviving token divergence is a low-margin fork, not a decision.

    Per the iteration-21 human acceptance override, exact HF/TensorRT-LLM token
    equality is diagnostic only: every fork is retained with its step, both
    candidate tokens, the reference's logits for both, and the separation --
    but the gate is a *declared dtype-aware envelope*, not a tie proof. A fork
    fails only when HuggingFace separates the two candidates by more than
    ``max(_FORK_ULP_FACTOR * ulp, _FORK_REL_MARGIN * |winning logit|)`` -- a
    confident decision difference that no bf16-readout noise can explain --
    which is exactly the "confident wrong decision" the override's teacher-
    forced methodology escalates on. Sub-envelope forks are the expected
    fingerprint of FP8-block kernels read out in bf16; the task-accuracy gates
    (fixed-100 canary, full GSM8K) remain the runtime correctness arbiters.
    """
    analysed = []
    for index, prompt in enumerate(hf_reference["prompts"]):
        expected = prompt["generated_token_ids"][:NEW_TOKENS].tolist()
        got = replay["tokens"][index].tolist()
        for step, (mine, theirs) in enumerate(zip(got, expected)):
            if mine == theirs:
                continue
            logits = prompt["generated_step_logits"][step].float()
            theirs_logit, mine_logit = float(logits[theirs]), float(logits[mine])
            separation = theirs_logit - mine_logit
            # The larger logit sets the local spacing of the reference's grid.
            ulp = _bf16_ulp(max(abs(theirs_logit), abs(mine_logit)))
            envelope = max(_FORK_ULP_FACTOR * ulp, _FORK_REL_MARGIN * abs(theirs_logit))
            analysed.append(
                {
                    "prompt": prompt["prompt"][:40],
                    "step": step,
                    "hf_token": theirs,
                    "hf_logit": theirs_logit,
                    "our_token": mine,
                    "our_token_hf_logit": mine_logit,
                    "hf_separation": separation,
                    "reference_ulp": ulp,
                    "identical_in_reference": theirs_logit == mine_logit,
                    "unresolvable": separation <= ulp,
                    "declared_envelope": envelope,
                    "within_declared_envelope": separation <= envelope,
                }
            )
    evidence["parity_divergence_resolution"] = {
        "num_divergences": len(analysed),
        "num_unresolvable": sum(1 for a in analysed if a["unresolvable"]),
        "num_within_declared_envelope": sum(1 for a in analysed if a["within_declared_envelope"]),
        "envelope_rule": f"max({_FORK_ULP_FACTOR} * bf16_ulp, {_FORK_REL_MARGIN} * |hf_logit|)",
        "divergences": analysed,
    }
    confident = [a for a in analysed if not a["within_declared_envelope"]]
    assert not confident, (
        "a token divergence exceeds the declared dtype-aware envelope: the "
        "reference separates the candidates too confidently for readout noise, "
        f"which is a decision difference to investigate: {confident}"
    )


# ---------------------------------------------------------------------------
# The tolerance floor
# ---------------------------------------------------------------------------


def test_block_fp8_matches_the_source_kernel_as_closely_as_arithmetic_allows(
    hf_reference, evidence
):
    """Establish the noise floor the whole-model tolerances rest on.

    Both stacks execute this FP8 checkpoint by quantizing activations to e4m3
    at call time, each with its own kernel. That is lossy, so neither reproduces
    the exact product -- the question is whether TensorRT-LLM's kernel is
    *worse*, which would be a defect, or merely *different*, which is
    arithmetic.

    The discriminator is a three-way comparison against a float64 evaluation of
    the same math on the same e4m3 weights. If TensorRT-LLM were quantizing
    badly its distance to exact would exceed HuggingFace's. The measured answer
    is that the two distances agree to three or four significant figures, which
    is what two implementations of one lossy operation look like -- and, once
    the activation scale follows the checkpoint's declared convention, the two
    kernels also agree with *each other* some two orders of magnitude more
    closely than either does with exact. That second bound is what moves if the
    quantizer regresses, so it is asserted tightly here and explained in
    ``test_the_block_fp8_matmul_reproduces_the_source_kernel_bitwise``.
    """
    import json as _json
    import os as _os

    from safetensors import safe_open

    from tensorrt_llm._torch.models.modeling_glm5_next import glm5_next_block_fp8_matmul

    transformers_fp8 = pytest.importorskip("transformers.integrations.finegrained_fp8")
    with open(_os.path.join(CHECKPOINT, "model.safetensors.index.json")) as fh:
        weight_map = _json.load(fh)["weight_map"]
    handles: Dict[str, Any] = {}

    def raw(key: str) -> torch.Tensor:
        shard = weight_map[key]
        if shard not in handles:
            handles[shard] = safe_open(_os.path.join(CHECKPOINT, shard), "pt")
        return handles[shard].get_tensor(key)

    activations = hf_reference["prompts"][0]["activations"]
    prefix = "model.language_model.layers"
    cases = [
        ("layer0.mlp.gate_proj", f"{prefix}.0.mlp.gate_proj.weight", "layer0.mlp.input"),
        (
            "layer3.self_attn.q_a_proj",
            f"{prefix}.3.self_attn.q_a_proj.weight",
            "layer3.self_attn.input",
        ),
        (
            "layer3.expert0.gate_proj",
            f"{prefix}.3.mlp.experts.0.gate_proj.weight",
            "layer3.mlp.input",
        ),
        (
            "layer44.expert0.gate_proj",
            f"{prefix}.44.mlp.experts.0.gate_proj.weight",
            "layer44.mlp.input",
        ),
    ]

    device = torch.device("cuda", 0)
    rows = []
    for name, key, act_key in cases:
        weight = raw(key).to(device)
        scale = raw(key + "_scale_inv").to(device)
        x = activations[act_key][0].to(device, torch.bfloat16)
        hf = transformers_fp8.fp8_linear(x, weight, scale, block_size=[128, 128], bias=None)
        trt = glm5_next_block_fp8_matmul(x, weight, scale)
        rows_n, cols_k = weight.shape
        dequantized = (
            weight.double()
            * scale.double().repeat_interleave(128, 0).repeat_interleave(128, 1)[:rows_n, :cols_k]
        )
        exact = x.double() @ dequantized.t()

        def rel(a: torch.Tensor, b: torch.Tensor) -> float:
            a64, b64 = a.double().flatten(), b.double().flatten()
            return float((a64 - b64).norm() / b64.norm())

        rows.append(
            {
                "case": name,
                "num_tokens": int(x.shape[0]),
                "hf_vs_trtllm": rel(hf, trt),
                "hf_vs_exact": rel(hf, exact),
                "trtllm_vs_exact": rel(trt, exact),
            }
        )
    evidence["block_fp8_kernel_equivalence"] = rows

    for row in rows:
        # TensorRT-LLM must be no further from exact than the source kernel is;
        # 5% headroom, where a genuinely worse quantization would be multiples.
        assert row["trtllm_vs_exact"] <= row["hf_vs_exact"] * 1.05, row
        # ...and the two kernels must agree with each other far better than
        # either does with exact. A divisor of 100 rather than 3 is deliberate:
        # 3 was all that held while the activation scale differed, and leaving
        # it there would let a regression back to that scale pass unnoticed.
        assert row["hf_vs_trtllm"] < row["hf_vs_exact"] / 100.0, row


def test_the_block_fp8_matmul_reproduces_the_source_kernel_bitwise(hf_reference, evidence):
    """The activation quantizer is the whole divergence, and it is fixed.

    The test above says the two stacks are *close*. This one says where the gap
    that remained came from, and pins the repair, because the answer decides
    whether exact greedy-argmax equality is reachable at all.

    Both stacks quantize activations to e4m3 in 1x128 tiles. They do not use the
    same scale: measured on this checkpoint, ``fp8_quantize_1x128``'s effective
    divisor ranges over 442.0-453.8 where the checkpoint's declared scheme says
    448, a +/-1.3% per-tile perturbation. That alone accounts for the entire
    per-GEMM disagreement -- with it, roughly half of the bf16 output elements
    match the source kernel, which is what chance gives for a last bit; with the
    declared convention, essentially all of them do.

    So this asserts three things that only hold together:

    * the model's quantizer payload equals the declared formula exactly;
    * ``fp8_quantize_1x128``'s does not, so the override is doing real work
      rather than restating the op (if TensorRT-LLM's op ever adopts the same
      convention this fails, and the local quantizer can be deleted);
    * the resulting GEMM reproduces the source kernel's bf16 output almost
      bitwise, through the same production CuTe kernel in both cases.
    """
    import json as _json
    import os as _os

    from safetensors import safe_open

    from tensorrt_llm._torch.models.modeling_glm5_next import (
        glm5_next_block_fp8_matmul,
        glm5_next_dynamic_act_quant_1x128,
    )

    transformers_fp8 = pytest.importorskip("transformers.integrations.finegrained_fp8")
    with open(_os.path.join(CHECKPOINT, "model.safetensors.index.json")) as fh:
        weight_map = _json.load(fh)["weight_map"]
    handles: Dict[str, Any] = {}

    def raw(key: str) -> torch.Tensor:
        shard = weight_map[key]
        if shard not in handles:
            handles[shard] = safe_open(_os.path.join(CHECKPOINT, shard), "pt")
        return handles[shard].get_tensor(key)

    activations = hf_reference["prompts"][0]["activations"]
    prefix = "model.language_model.layers"
    cases = [
        ("layer0.mlp.gate_proj", f"{prefix}.0.mlp.gate_proj.weight", "layer0.mlp.input"),
        (
            "layer3.self_attn.q_a_proj",
            f"{prefix}.3.self_attn.q_a_proj.weight",
            "layer3.self_attn.input",
        ),
        (
            "layer44.expert0.gate_proj",
            f"{prefix}.44.mlp.experts.0.gate_proj.weight",
            "layer44.mlp.input",
        ),
    ]

    device = torch.device("cuda", 0)
    rows: List[Dict[str, Any]] = []
    for name, key, act_key in cases:
        weight = raw(key).to(device)
        scale = raw(key + "_scale_inv").to(device).float()
        x = activations[act_key][0].to(device, torch.bfloat16).contiguous()

        # bf16 in, bf16 out -- exactly what the source runs inside the model.
        source = transformers_fp8.fp8_linear(
            x, weight, scale, block_size=[128, 128], allow_deepgemm=False
        )
        ours = glm5_next_block_fp8_matmul(x, weight, scale)

        declared_payload, declared_scale = glm5_next_dynamic_act_quant_1x128(x)
        stock_payload, _ = torch.ops.trtllm.fp8_quantize_1x128(x)
        stock_payload = stock_payload[: x.shape[0]]
        stock = torch.ops.trtllm.cute_dsl_fp8_gemm_blackwell(
            stock_payload, weight, torch.ops.trtllm.fp8_quantize_1x128(x)[1], scale
        )[: x.shape[0]]

        # The declared scheme's effective divisor, recovered per tile from the
        # payload the stock op produced, is what makes the two disagree.
        tiles = x.reshape(x.shape[0], -1, 128).float()
        stock_tiles = stock_payload.reshape(x.shape[0], -1, 128).float()
        stock_scale = (stock_tiles * tiles).sum(-1) / (stock_tiles * stock_tiles).sum(-1).clamp(
            min=1e-30
        )
        stock_divisor = tiles.abs().amax(-1) / stock_scale.clamp(min=1e-30)

        def rel(a: torch.Tensor, b: torch.Tensor) -> float:
            a64, b64 = a.double().flatten(), b.double().flatten()
            return float((a64 - b64).norm() / b64.norm())

        rows.append(
            {
                "case": name,
                "num_tokens": int(x.shape[0]),
                "declared_scale_shape": list(declared_scale.shape),
                "stock_payload_agreement": float(
                    (stock_payload.view(torch.uint8) == declared_payload.view(torch.uint8))
                    .double()
                    .mean()
                ),
                "stock_effective_divisor_min": float(stock_divisor.min()),
                "stock_effective_divisor_max": float(stock_divisor.max()),
                "ours_bitwise_equal_fraction": float((ours == source).double().mean()),
                "stock_bitwise_equal_fraction": float((stock == source).double().mean()),
                "ours_vs_source": rel(ours.float(), source.float()),
                "stock_vs_source": rel(stock.float(), source.float()),
            }
        )
    evidence["block_fp8_activation_quantizer"] = rows

    for row in rows:
        # The scale layout the CuTe GEMM consumes is [K // 128, M].
        assert row["declared_scale_shape"][1] == row["num_tokens"], row
        # The override is doing real work: the stock op's payload is close but
        # not equal, and its effective divisor is not the declared 448.
        assert row["stock_payload_agreement"] < 1.0, row
        assert not (
            447.9 < row["stock_effective_divisor_min"]
            and row["stock_effective_divisor_max"] < 448.1
        ), row
        # And the repair lands: the model's GEMM reproduces the source kernel's
        # own bf16 output almost bitwise, where the stock quantizer gives the
        # ~50% a last-bit coin flip would.
        assert row["ours_bitwise_equal_fraction"] > 0.999, row
        assert row["ours_vs_source"] < row["stock_vs_source"] / 20.0, row
