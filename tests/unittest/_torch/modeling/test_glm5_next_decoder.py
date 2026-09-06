# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rung-three calibration of the GLM-5.3-Flash hyper-connected decoder.

Companion to ``test_glm5_next_attention.py``: that file pins the two attention
modules, this one pins everything wrapped around them -- the four-stream
hyper-connection sites, the unweighted stream head, the literal dense-vs-MoE
dispatch, the FP32 noaux_tc router, and the asymmetrically clamped SwiGLU --
and then pins the assembled decoder layer against the real model.

The decisive evidence here is ``source_activation_replay`` at *whole-layer*
granularity: the Goal-1.1 fixture captured the real model's four-stream
``layerN.input`` and ``layerN.output``, so an assembled layer can be driven with
the real inputs and checked against the real outputs rather than against a
re-derivation of itself.

Every test runs on CUDA against ``/dev/shm/GLM-5.3-Flash``. There is no
toy-config or random-weight fallback; a skipped run is not evidence.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import pytest
import torch
from glm5_next_ref import CheckpointReader, compare

from tensorrt_llm._torch.models.modeling_glm5_next import (
    Glm5NextDecoderLayer,
    Glm5NextLinearAttention,
    Glm5NextMLP,
    Glm5NextMoE,
    Glm5NextSparseAttention,
    Glm5NextTopkRouter,
    clamped_swiglu,
    glm5_next_expand_streams,
    glm5_next_hyper_connection,
    glm5_next_hyper_head,
    remap_glm5_next_key,
    resolve_glm5_next_schedule,
)

CHECKPOINT = os.environ.get("GLM53_FLASH_CHECKPOINT", "/dev/shm/GLM-5.3-Flash")
LAYER_PREFIX = "model.language_model.layers"
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
#: Absolute by construction: a repo-relative default would drop the evidence
#: file wherever pytest happened to be invoked from, littering the test tree.
EVIDENCE_PATH = os.environ.get(
    "GLM53_DECODER_EVIDENCE",
    os.path.join(
        _REPO_ROOT,
        "agent-flow/workspace/glm-5.3-flash-bringup/reports/goal14_decoder_evidence.json",
    ),
)
#: Four-stream hidden states captured by hooking the real model. Its absence is
#: a hard failure rather than a skip: whole-layer source replay is the
#: pass-critical evidence for this Goal, and a suite that quietly dropped it
#: would report green while proving nothing about real inputs.
FIXTURE = os.environ.get(
    "GLM53_HF_FIXTURE",
    os.path.join(
        _REPO_ROOT, "agent-flow/workspace/glm-5.3-flash-bringup/reports/hf_reference_fixture.pt"
    ),
)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(
        not os.path.isdir(CHECKPOINT),
        reason=f"requires the real checkpoint at {CHECKPOINT}",
    ),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def device() -> torch.device:
    torch.cuda.init()
    return torch.device("cuda")


@pytest.fixture(scope="module")
def full_config():
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(CHECKPOINT)
    config.text_config._attn_implementation = "eager"
    return config


@pytest.fixture(scope="module")
def text_config(full_config):
    return full_config.text_config


@pytest.fixture(scope="module")
def schedule(full_config):
    return resolve_glm5_next_schedule(full_config)


@pytest.fixture(scope="module")
def reader() -> CheckpointReader:
    r = CheckpointReader(CHECKPOINT)
    yield r
    r.close()


@pytest.fixture(scope="module")
def hooked():
    assert os.path.isfile(FIXTURE), (
        f"missing the native-HF activation fixture at {FIXTURE}; build it with "
        "glm5_next_hf_reference.py. Whole-layer source replay is pass-critical, "
        "so this is a failure rather than a skip."
    )
    payload = torch.load(FIXTURE, map_location="cpu", weights_only=False)
    return [p for p in payload["prompts"] if p.get("activations")]


@pytest.fixture(scope="module")
def evidence() -> Dict[str, List[dict]]:
    payload: Dict[str, List[dict]] = {}
    yield payload
    path = os.path.abspath(EVIDENCE_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)


def _record(evidence, key, payload):
    evidence.setdefault(key, []).append(payload)


def _fixed(shape, device, seed=0, scale=0.05, dtype=torch.bfloat16):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(*shape, generator=gen, dtype=torch.float32) * scale).to(
        device=device, dtype=dtype
    )


# bf16 round-off band. Gates are on *relative* max_abs so the bound does not
# silently widen with activation magnitude.
REL_MAX_ABS = 0.02
MIN_COSINE = 0.9999


# ---------------------------------------------------------------------------
# Hyper-connection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layer_idx", [0, 22, 44], ids=["first", "middle", "last"])
@pytest.mark.parametrize("site", ["attn", "ffn"])
def test_hyper_connection_matches_native_hf(reader, text_config, device, evidence, layer_idx, site):
    """The reused ``mHC`` reproduces ``Glm5NextTextHyperConnection`` exactly.

    All three outputs are checked, not just the collapsed activation: ``post``
    and ``comb`` are consumed later by ``post_mapping``, so an error in either
    would otherwise only surface as a small end-of-layer drift.
    """
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextHyperConnection

    p = f"{LAYER_PREFIX}.{layer_idx}.hc_{site}"
    fn = reader.raw(f"{p}_fn").to(device)
    base = reader.raw(f"{p}_base").to(device)
    scale = reader.raw(f"{p}_scale").to(device)

    hf = Glm5NextTextHyperConnection(text_config).to(device=device, dtype=torch.bfloat16).eval()
    trt = glm5_next_hyper_connection(text_config).to(device)
    with torch.no_grad():
        hf.fn.copy_(fn.to(torch.bfloat16))
        hf.base.copy_(base.to(torch.bfloat16))
        hf.scale.copy_(scale.to(torch.bfloat16))
        trt.fn.copy_(fn.float())
        trt.base.copy_(base.float())
        trt.scale.copy_(scale.float())

    streams = _fixed((64, text_config.hc_mult, text_config.hidden_size), device, seed=layer_idx)
    with torch.no_grad():
        hf_post, hf_comb, hf_collapsed = hf(streams.unsqueeze(0))
        post, comb, collapsed = trt.pre_mapping(streams)

    m_post = compare(post.squeeze(-1).float(), hf_post[0].float(), "post")
    m_comb = compare(comb.float(), hf_comb[0].float(), "comb")
    m_collapsed = compare(collapsed.float(), hf_collapsed[0].float(), "collapsed")

    # Sinkhorn projects onto the doubly-stochastic manifold, but 20 rounds with
    # an eps added after every division do not converge exactly -- the source
    # itself leaves a residual that grows with layer depth. The check is
    # therefore against HF's *own* marginal deviation rather than an invented
    # convergence bound the source does not meet.
    row_sums = comb.float().sum(-1)
    col_sums = comb.float().sum(-2)
    hf_row_dev = float((hf_comb[0].float().sum(-1) - 1).abs().max())
    hf_col_dev = float((hf_comb[0].float().sum(-2) - 1).abs().max())
    row_dev = float((row_sums - 1).abs().max())
    col_dev = float((col_sums - 1).abs().max())
    _record(
        evidence,
        "hyper_connection",
        {
            "layer": layer_idx,
            "site": site,
            "sinkhorn_iters": trt.sinkhorn_iters,
            "post_mult_value": trt.post_mult_value,
            "eps": trt.eps,
            "norm_eps": trt.norm_eps,
            "post": m_post,
            "comb": m_comb,
            "collapsed": m_collapsed,
            "post_range": [float(post.min()), float(post.max())],
            "row_sum_dev": row_dev,
            "col_sum_dev": col_dev,
            "hf_row_sum_dev": hf_row_dev,
            "hf_col_sum_dev": hf_col_dev,
            "comb_asymmetry": float((comb - comb.transpose(-1, -2)).abs().max()),
        },
    )
    for name, m in (("post", m_post), ("comb", m_comb), ("collapsed", m_collapsed)):
        assert m["all_finite"], (name, m)
        assert m["cosine"] >= MIN_COSINE, (name, m)
        assert m["max_abs"] <= REL_MAX_ABS * max(m["ref_max_abs"], 1e-3), (name, m)
    assert row_dev <= max(1.5 * hf_row_dev, 1e-3), (row_dev, hf_row_dev)
    assert col_dev <= max(1.5 * hf_col_dev, 1e-3), (col_dev, hf_col_dev)
    # Whatever the residual, the matrix must still be far closer to
    # doubly-stochastic than a bare row-softmax, whose columns are unconstrained.
    assert col_dev < 0.1
    # post is 2*sigmoid, so it lives in (0, 2) and must actually exceed 1
    # somewhere -- otherwise the doubled range would be untested.
    assert 0.0 < float(post.min()) and float(post.max()) < 2.0
    assert float(post.max()) > 1.0


def test_hyper_connection_settings_are_pinned_by_measurement(reader, text_config, device, evidence):
    """The two model-specific ``mHC`` settings are load-bearing, not cosmetic.

    ``post_mult_value`` and ``sinkhorn_iters`` were chosen by comparing against
    the source rather than assumed, so the alternatives are shown to be wrong
    here instead of merely unmentioned.
    """
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextHyperConnection

    from tensorrt_llm._torch.modules.mhc.hyper_connection import mHC

    layer_idx = 0
    p = f"{LAYER_PREFIX}.{layer_idx}.hc_attn"
    fn = reader.raw(f"{p}_fn").to(device)
    base = reader.raw(f"{p}_base").to(device)
    scale = reader.raw(f"{p}_scale").to(device)
    hf = Glm5NextTextHyperConnection(text_config).to(device=device, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        hf.fn.copy_(fn.to(torch.bfloat16))
        hf.base.copy_(base.to(torch.bfloat16))
        hf.scale.copy_(scale.to(torch.bfloat16))
    streams = _fixed((64, text_config.hc_mult, text_config.hidden_size), device, seed=5)
    with torch.no_grad():
        hf_post, hf_comb, _ = hf(streams.unsqueeze(0))

    def run(post_mult, iters):
        m = mHC(
            mult=int(text_config.hc_mult),
            hidden_size=int(text_config.hidden_size),
            sinkhorn_iters=iters,
            eps=float(text_config.hc_eps),
            norm_eps=float(text_config.rms_norm_eps),
            sinkhorn_eps=float(text_config.hc_eps),
            post_mult_value=post_mult,
        ).to(device)
        with torch.no_grad():
            m.fn.copy_(fn.float())
            m.base.copy_(base.float())
            m.scale.copy_(scale.float())
            post, comb, _ = m.pre_mapping(streams)
        return (
            compare(post.squeeze(-1).float(), hf_post[0].float(), "post"),
            compare(comb.float(), hf_comb[0].float(), "comb"),
        )

    iters = int(text_config.hc_sinkhorn_iters)
    chosen_post, chosen_comb = run(2.0, iters)
    halved_post, _ = run(1.0, iters)
    _, short_comb = run(2.0, iters - 1)

    _record(
        evidence,
        "hyper_connection_controls",
        {
            "chosen": {"post": chosen_post, "comb": chosen_comb},
            "post_mult_1.0": halved_post,
            "sinkhorn_iters_minus_one": short_comb,
        },
    )
    # Dropping the source's factor of two on `post` is a gross error.
    assert halved_post["max_abs"] > 100 * chosen_post["max_abs"], (halved_post, chosen_post)
    # One Sinkhorn round short is subtle -- it does not move the cosine at all --
    # but it leaves an order-of-magnitude larger residual on `comb`. This is the
    # kind of drift a cosine-only gate would wave through.
    assert short_comb["max_abs"] > 5 * chosen_comb["max_abs"], (short_comb, chosen_comb)


def test_hyper_head_is_unweighted_mean(reader, text_config, device, evidence):
    """The stream head is a plain mean, and the checkpoint has no head weights.

    TensorRT-LLM's ``modules.mhc.HCHead`` is the DeepSeek-V4 variant and carries
    learned ``fn``/``base``/``scale``. Using it here would require inventing
    those weights, so the absence of any such checkpoint tensor is asserted
    rather than left implicit.
    """
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextHyperHead

    streams = _fixed((32, text_config.hc_mult, text_config.hidden_size), device, seed=3)
    with torch.no_grad():
        expected = Glm5NextTextHyperHead()(streams.unsqueeze(0))[0]
        got = glm5_next_hyper_head(streams)
    stats = compare(got.float(), expected.float(), "hyper_head")
    assert stats["max_abs"] == 0.0, stats

    head_keys = [k for k in reader.keys() if "hc_head" in k or "hyper_head" in k]
    _record(
        evidence,
        "hyper_head",
        {"exact": stats, "checkpoint_head_keys": head_keys},
    )
    assert head_keys == [], f"checkpoint unexpectedly carries head weights: {head_keys}"

    # A weighted mean with non-uniform weights must differ, so the assertion above
    # is not vacuous.
    weighted = (
        streams.float() * torch.tensor([0.1, 0.2, 0.3, 0.4], device=device).view(1, -1, 1)
    ).sum(-2)
    assert compare(weighted, expected.float(), "weighted")["max_abs"] > 0.01


def test_embedding_streams_start_identical(full_config, device):
    """Embedding expansion produces ``hc_mult`` identical, independently writable copies."""
    text = full_config.text_config
    embeds = _fixed((8, text.hidden_size), device, seed=11)
    streams = glm5_next_expand_streams(embeds, int(text.hc_mult))
    assert streams.shape == (8, text.hc_mult, text.hidden_size)
    assert streams.is_contiguous(), (
        "an expanded view aliases one row; post_mapping writes per stream"
    )
    for s in range(text.hc_mult):
        assert torch.equal(streams[:, s], embeds)
    streams[:, 0] += 1.0
    assert not torch.equal(streams[:, 0], streams[:, 1]), "streams must be independently writable"


# ---------------------------------------------------------------------------
# Feed-forward
# ---------------------------------------------------------------------------


def _load_dense_mlp(reader, text_config, layer_idx, device):
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextMLP

    p = f"{LAYER_PREFIX}.{layer_idx}.mlp"
    hf = Glm5NextTextMLP(text_config).to(device=device, dtype=torch.bfloat16).eval()
    trt = Glm5NextMLP(text_config).to(device).eval()
    with torch.no_grad():
        for name in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
            value = reader.get(f"{p}.{name}").to(device=device, dtype=torch.bfloat16)
            dict(hf.named_parameters())[name].copy_(value)
            dict(trt.named_parameters())[name].copy_(value)
    return hf, trt


@pytest.mark.parametrize("layer_idx", [0, 2], ids=["first_dense", "last_dense"])
def test_dense_mlp_matches_native_hf(reader, text_config, schedule, device, evidence, layer_idx):
    """Dense clamped-SwiGLU MLP agrees with HF on real weights."""
    assert schedule.mlp[layer_idx] == "dense"
    hf, trt = _load_dense_mlp(reader, text_config, layer_idx, device)
    x = _fixed((64, text_config.hidden_size), device, seed=layer_idx + 40)
    with torch.no_grad():
        expected = hf(x)
        got = trt(x)
    stats = compare(got.float(), expected.float(), f"dense_mlp_L{layer_idx}")
    _record(evidence, "dense_mlp", {"layer": layer_idx, **stats})
    assert stats["all_finite"]
    assert stats["cosine"] >= MIN_COSINE, stats
    assert stats["max_abs"] <= REL_MAX_ABS * stats["ref_max_abs"], stats


def test_clamped_swiglu_is_asymmetric_and_active(reader, text_config, device, evidence):
    """An extreme activation must cross the 10.0 clamp, on the correct side.

    The clamp is asymmetric: ``gate`` is bounded only from above, ``up`` on both
    sides. Ordinary SwiGLU is identical inside the clamp, so a typical-magnitude
    comparison cannot distinguish them -- this drives past the limit on purpose
    and checks the two sides separately.
    """
    limit = float(text_config.swiglu_limit)
    assert limit == 10.0
    gate = torch.tensor([-50.0, -limit - 5, 0.0, limit - 1, limit + 5, 50.0], device=device)
    up = torch.tensor([-50.0, -limit - 5, 0.0, limit - 1, limit + 5, 50.0], device=device)

    got = clamped_swiglu(gate, up, limit)
    plain = torch.nn.functional.silu(gate) * up
    # gate is NOT clamped from below: a large negative gate must pass through.
    only_upper = torch.nn.functional.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)
    both_sides = torch.nn.functional.silu(gate.clamp(min=-limit, max=limit)) * up.clamp(
        min=-limit, max=limit
    )
    _record(
        evidence,
        "swiglu_clamp",
        {
            "limit": limit,
            "clamped": got.float().tolist(),
            "plain": plain.float().tolist(),
            "symmetric_gate_variant": both_sides.float().tolist(),
        },
    )
    assert torch.equal(got, only_upper)
    assert not torch.allclose(got, plain), "the clamp never engaged"
    # Rows where gate < -limit distinguish the asymmetric clamp from a symmetric one.
    assert not torch.allclose(got, both_sides), "gate must not be clamped from below"

    # And the same asymmetry must survive through the real dense MLP: scale the
    # input until the pre-activation crosses the limit.
    hf, trt = _load_dense_mlp(reader, text_config, 0, device)
    x = _fixed((16, text_config.hidden_size), device, seed=77, scale=4.0)
    with torch.no_grad():
        pre_gate = trt.gate_proj(x)
        pre_up = trt.up_proj(x)
        expected = hf(x)
        got_mlp = trt(x)
    stats = compare(got_mlp.float(), expected.float(), "extreme_dense_mlp")
    _record(
        evidence,
        "swiglu_clamp_extreme",
        {
            "gate_max": float(pre_gate.max()),
            "gate_min": float(pre_gate.min()),
            "up_absmax": float(pre_up.abs().max()),
            "frac_gate_over_limit": float((pre_gate > limit).float().mean()),
            "frac_up_over_limit": float((pre_up.abs() > limit).float().mean()),
            **stats,
        },
    )
    assert float(pre_gate.max()) > limit, "the extreme case never crossed the gate clamp"
    assert float(pre_up.abs().max()) > limit, "the extreme case never crossed the up clamp"
    assert stats["cosine"] >= MIN_COSINE, stats


def _load_router(reader, text_config, layer_idx, device):
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextTopkRouter

    p = f"{LAYER_PREFIX}.{layer_idx}.mlp.gate"
    hf = Glm5NextTextTopkRouter(text_config).to(device=device, dtype=torch.float32).eval()
    trt = Glm5NextTopkRouter(text_config).to(device).eval()
    weight = reader.get(f"{p}.weight").to(device=device, dtype=torch.float32)
    bias = reader.get(f"{p}.e_score_correction_bias").to(device=device, dtype=torch.float32)
    with torch.no_grad():
        hf.weight.copy_(weight)
        hf.e_score_correction_bias.copy_(bias)
        trt.weight.copy_(weight)
        trt.e_score_correction_bias.copy_(bias)
    return hf, trt


@pytest.mark.parametrize("layer_idx", [3, 22, 44], ids=["first_routed", "middle", "last"])
def test_router_matches_native_hf(reader, text_config, schedule, device, evidence, layer_idx):
    """Exact expert IDs and weights, with the two orderings that are easy to invert."""
    assert schedule.mlp[layer_idx] == "sparse"
    hf, trt = _load_router(reader, text_config, layer_idx, device)
    x = _fixed((64, text_config.hidden_size), device, seed=layer_idx + 200)
    with torch.no_grad():
        hf_logits, hf_weights, hf_idx = hf(x)
        logits, weights, idx = trt(x)

    hf_sets = [set(r.tolist()) for r in hf_idx]
    trt_sets = [set(r.tolist()) for r in idx]
    exact = sum(int(a == b) for a, b in zip(hf_sets, trt_sets))
    # Compare weights on a common ordering, since topk(sorted=False) fixes none.
    hf_sorted = torch.sort(hf_idx, dim=-1).indices
    trt_sorted = torch.sort(idx, dim=-1).indices
    m_weights = compare(
        weights.gather(1, trt_sorted).float(), hf_weights.gather(1, hf_sorted).float(), "weights"
    )
    m_logits = compare(logits, hf_logits, "router_logits")

    summed = weights.sum(-1)
    _record(
        evidence,
        "router",
        {
            "layer": layer_idx,
            "exact_expert_sets": exact,
            "rows": int(idx.shape[0]),
            "top_k": int(idx.shape[1]),
            "logits": m_logits,
            "weights": m_weights,
            "logits_dtype": str(logits.dtype),
            "weight_sum_range": [float(summed.min()), float(summed.max())],
            "routed_scaling_factor": float(text_config.routed_scaling_factor),
        },
    )
    assert exact == idx.shape[0], f"expert selection differs on {idx.shape[0] - exact} rows"
    assert m_logits["max_abs"] == 0.0, m_logits
    assert m_weights["max_abs"] == 0.0, m_weights
    assert logits.dtype == torch.float32 and weights.dtype == torch.float32
    # Weights are normalized *then* scaled, so they sum to the scaling factor.
    assert torch.allclose(
        summed, torch.full_like(summed, float(text_config.routed_scaling_factor)), atol=1e-4
    )

    # Control: the correction bias participates in selection only. Adding it to
    # the returned weights instead would change them materially.
    with torch.no_grad():
        scores = torch.nn.functional.linear(x.float(), trt.weight).sigmoid()
        corrected = (scores + trt.e_score_correction_bias).gather(1, idx)
    assert compare(corrected.float(), weights.float(), "bias_in_weights")["max_abs"] > 0.1


def test_router_must_be_fp32_on_real_activations(
    reader, text_config, schedule, hooked, device, evidence
):
    """A bf16-rounded router changes the top-8 on most real tokens.

    ``moe_router_dtype`` is ``float32`` in the config and the correction bias is
    stored as F32 in the checkpoint; this pins *why* that matters instead of
    treating it as a style preference. Rounding only the router's stored values
    to bf16 -- input, arithmetic and everything else unchanged -- is enough to
    reshuffle expert selection on a majority of real tokens, and the resulting
    layer output moves by two orders of magnitude more than bf16 noise.
    """
    layer_idx = 22
    assert schedule.mlp[layer_idx] == "sparse"
    assert str(getattr(text_config, "moe_router_dtype", "float32")) == "float32"
    acts = hooked[0]["activations"]
    x = acts[f"layer{layer_idx}.mlp.input"].to(device=device, dtype=torch.bfloat16)[0]

    _, router = _load_router(reader, text_config, layer_idx, device)
    with torch.no_grad():
        _, weights, idx = router(x)
        # Round only the stored router values through bf16 and back.
        router.weight.data = router.weight.data.to(torch.bfloat16).float()
        router.e_score_correction_bias.data = router.e_score_correction_bias.data.to(
            torch.bfloat16
        ).float()
        _, weights_bf16, idx_bf16 = router(x)

    agree = sum(int(set(a.tolist()) == set(b.tolist())) for a, b in zip(idx, idx_bf16))
    bias_levels = int(
        torch.unique(
            reader.get(f"{LAYER_PREFIX}.{layer_idx}.mlp.gate.e_score_correction_bias")
            .to(torch.bfloat16)
            .float()
        ).numel()
    )
    _record(
        evidence,
        "router_fp32_requirement",
        {
            "layer": layer_idx,
            "tokens": int(idx.shape[0]),
            "rows_agreeing_after_bf16_rounding": agree,
            "distinct_bias_values_fp32": int(text_config.n_routed_experts),
            "distinct_bias_values_after_bf16": bias_levels,
            "weight_shift": compare(weights_bf16.float(), weights.float(), "weights"),
        },
    )
    # bf16 collapses the 288 distinct bias values onto a small number of levels...
    assert bias_levels < int(text_config.n_routed_experts) // 4, bias_levels
    # ...and that changes selection on most tokens.
    assert agree < idx.shape[0] // 2, (
        f"bf16 rounding left {agree}/{idx.shape[0]} rows unchanged; this control "
        "is supposed to demonstrate that FP32 routing is load-bearing"
    )


def _force_fp32_router(gate) -> None:
    """Keep HF's router in FP32 after the module has been cast to bf16.

    ``Glm5NextTextMoE(...).to(dtype=bfloat16)`` also casts ``gate.weight`` and
    the ``e_score_correction_bias`` buffer. The router's own forward re-casts
    them to FP32, but by then the precision is gone -- and this router is
    exquisitely sensitive to that: the bias sits near magnitude 10 while the
    sigmoid scores it corrects are O(1e-2), so ranking turns on gaps far below
    bf16's ~1e-2 resolution at that magnitude. Measured on layer 22 with the
    *same* input and the *same* checkpoint values, a bf16-rounded reference
    router agrees with the FP32 one on only **11 of 24** real tokens, with
    near-disjoint expert sets, and the assembled layer output then differs by
    max_abs 144.

    This must run **before** any ``copy_`` into the module: upcasting afterwards
    restores the container's dtype but not the rounded-away bits. That mistake
    is worth naming because it fails silently and looks fixed.
    """
    gate.weight.data = gate.weight.data.float()
    gate.e_score_correction_bias = gate.e_score_correction_bias.float()
    assert gate.weight.dtype == torch.float32
    assert gate.e_score_correction_bias.dtype == torch.float32


def _load_moe(reader, text_config, layer_idx, device, experts):
    """Build both MoE modules, materializing only ``experts``.

    All 288 expert slots are allocated on both sides (so the router's real ID
    space is preserved) but only the selected ones are read from the checkpoint:
    the layer's output depends on exactly those.
    """
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextMoE

    p = f"{LAYER_PREFIX}.{layer_idx}.mlp"
    dt = torch.bfloat16
    hf = Glm5NextTextMoE(text_config).to(device=device, dtype=dt).eval()
    _force_fp32_router(hf.gate)
    trt = Glm5NextMoE(text_config).to(device).eval()
    with torch.no_grad():
        hf.experts.gate_up_proj.zero_()
        hf.experts.down_proj.zero_()
        trt.gate_up_proj.zero_()
        trt.down_proj.zero_()
        for e in experts:
            gate_w = reader.get(f"{p}.experts.{e}.gate_proj.weight", device=device).to(dt)
            up_w = reader.get(f"{p}.experts.{e}.up_proj.weight", device=device).to(dt)
            down_w = reader.get(f"{p}.experts.{e}.down_proj.weight", device=device).to(dt)
            fused = torch.cat([gate_w, up_w], dim=0)
            hf.experts.gate_up_proj[e].copy_(fused)
            hf.experts.down_proj[e].copy_(down_w)
            trt.gate_up_proj[e].copy_(fused)
            trt.down_proj[e].copy_(down_w)
        for name in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
            value = reader.get(f"{p}.shared_experts.{name}", device=device).to(dt)
            dict(hf.shared_experts.named_parameters())[name].copy_(value)
            dict(trt.shared_experts.named_parameters())[name].copy_(value)
        weight = reader.get(f"{p}.gate.weight", device=device).to(torch.float32)
        bias = reader.get(f"{p}.gate.e_score_correction_bias", device=device).to(torch.float32)
        hf.gate.weight.copy_(weight)
        hf.gate.e_score_correction_bias.copy_(bias)
        trt.gate.weight.copy_(weight)
        trt.gate.e_score_correction_bias.copy_(bias)
    return hf, trt


def test_moe_matches_native_hf(reader, text_config, schedule, hooked, device, evidence):
    """Routed + shared MoE on the real hidden states entering a routed layer."""
    layer_idx = 3
    assert schedule.mlp[layer_idx] == "sparse"
    acts = hooked[0]["activations"]
    x = acts[f"layer{layer_idx}.mlp.input"].to(device=device, dtype=torch.bfloat16)[0]
    captured = acts[f"layer{layer_idx}.mlp.output"].to(device=device, dtype=torch.float32)[0]

    _, router = _load_router(reader, text_config, layer_idx, device)
    with torch.no_grad():
        _, _, idx = router(x)
    experts = sorted(set(idx.flatten().tolist()))
    hf, trt = _load_moe(reader, text_config, layer_idx, device, experts)

    with torch.no_grad():
        hf_out = hf(x).float()
        got = trt(x).float()
        # Decompose so a routed/shared mix-up cannot hide inside the total.
        shared_only = trt.shared_experts(x).float()
        routed_only = got - shared_only

    vs_hf = compare(got, hf_out, "moe_vs_hf")
    vs_model = compare(got, captured, "moe_vs_in_model")
    hf_vs_model = compare(hf_out, captured, "hf_vs_in_model")
    _record(
        evidence,
        "moe",
        {
            "layer": layer_idx,
            "num_experts_materialized": len(experts),
            "num_experts_total": int(text_config.n_routed_experts),
            "tokens": int(x.shape[0]),
            "routed_absmax": float(routed_only.abs().max()),
            "shared_absmax": float(shared_only.abs().max()),
            "trtllm_vs_standalone_hf": vs_hf,
            "trtllm_vs_in_model": vs_model,
            "standalone_hf_vs_in_model": hf_vs_model,
        },
    )
    # Both bf16 rungs must agree very tightly with each other...
    assert vs_hf["cosine"] > 0.9999, vs_hf
    # ...and drift from the FP8 in-model output by the same amount.
    assert abs(vs_model["cosine"] - hf_vs_model["cosine"]) < 1e-3, (vs_model, hf_vs_model)
    assert vs_model["cosine"] > 0.995, vs_model
    # Both contributions must be non-trivial: a dropped shared expert or a
    # dropped routed sum would still leave a plausible-looking output.
    assert float(routed_only.abs().max()) > 1e-3
    assert float(shared_only.abs().max()) > 1e-3


# ---------------------------------------------------------------------------
# Production MoE (Goal 3.3): TRTLLM-Gen fused FP8-block routed experts
# ---------------------------------------------------------------------------

#: The routed-expert layer index every production-MoE test pins.
_PROD_MOE_LAYER = 3


@pytest.fixture(scope="module")
def production_moe(reader, full_config, text_config, device):
    """One production and one fp8-emulation MoE on identical real weights.

    ``prod`` is the runtime path: ``create_moe`` with the DeepSeek noaux_tc
    routing method and ``swiglu_limit_scalar=10.0``, resolved on this SM100
    machine to ``TRTLLMGenFusedMoE`` over FP8 block scales, loaded through the
    fused layer's own quant-method loader from the raw e4m3 payloads. ``emul``
    is the Stage-1/2 verified block-FP8 torch emulation on the same bytes --
    the same-arithmetic-family rung the production kernel is judged against.
    """
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_glm5_next import build_glm5_next_quant_config

    layer_idx = _PROD_MOE_LAYER
    p = f"{LAYER_PREFIX}.{layer_idx}.mlp"
    qc = build_glm5_next_quant_config(full_config)
    mc = ModelConfig(pretrained_config=full_config, quant_config=qc, moe_backend="TRTLLM")
    prod = Glm5NextMoE(text_config, quantized=True, model_config=mc, layer_idx=layer_idx).to(device)
    prod.eval()
    emul = Glm5NextMoE(text_config, quantized=True).to(device).eval()

    fused_dict = {}
    with torch.no_grad():
        for eid in range(prod.num_experts):
            for proj, w in (("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2")):
                raw = reader.raw(f"{p}.experts.{eid}.{proj}.weight")
                scale = reader.raw(f"{p}.experts.{eid}.{proj}.weight_scale_inv")
                fused_dict[f"{eid}.{w}.weight"] = raw
                fused_dict[f"{eid}.{w}.weight_scale_inv"] = scale
                if proj == "down_proj":
                    emul.down_proj[eid].copy_(raw.to(device))
                    emul.down_proj_scale[eid].copy_(scale.to(device))
                else:
                    rows = emul.gate_up_proj.shape[1] // 2
                    start = 0 if proj == "gate_proj" else rows
                    emul.gate_up_proj[eid, start : start + rows].copy_(raw.to(device))
                    srows = emul.gate_up_proj_scale.shape[1] // 2
                    sstart = 0 if proj == "gate_proj" else srows
                    emul.gate_up_proj_scale[eid, sstart : sstart + srows].copy_(scale.to(device))
        prod.experts.load_weights([fused_dict])
        if hasattr(prod.experts, "post_load_weights"):
            prod.experts.post_load_weights()
        for name in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
            value = reader.get(f"{p}.shared_experts.{name}", device=device).to(torch.bfloat16)
            dict(prod.shared_experts.named_parameters())[name].copy_(value)
            dict(emul.shared_experts.named_parameters())[name].copy_(value)
        gate_w = reader.raw(f"{p}.gate.weight").to(device).float()
        gate_b = reader.raw(f"{p}.gate.e_score_correction_bias").to(device).float()
        for m in (prod, emul):
            m.gate.weight.copy_(gate_w)
            m.gate.e_score_correction_bias.copy_(gate_b)
    torch.cuda.synchronize()
    return prod, emul


def test_moe_production_backend_matches_native_hf(
    reader, text_config, schedule, hooked, production_moe, device, evidence
):
    """Production fused routed experts on the real hidden states of layer 3.

    Three-way comparison: the production kernel is anchored to native HF
    through the *same-arithmetic* fp8 emulation rung -- the production/HF
    cosine must sit where the emulation/HF cosine sits, so a kernel-side
    defect cannot hide inside the fp8-vs-bf16 dtype envelope.
    """
    layer_idx = _PROD_MOE_LAYER
    assert schedule.mlp[layer_idx] == "sparse"
    prod, emul = production_moe
    acts = hooked[0]["activations"]
    x = acts[f"layer{layer_idx}.mlp.input"].to(device=device, dtype=torch.bfloat16)[0]
    captured = acts[f"layer{layer_idx}.mlp.output"].to(device=device, dtype=torch.float32)[0]

    # Backend naming is part of the pass evidence, not commentary.
    backend = getattr(prod.experts, "backend", prod.experts)
    assert prod.moe_backend_name == "TRTLLMGenFusedMoE", prod.moe_backend_name
    assert type(prod.experts).__name__ == "ConfigurableMoE"
    assert getattr(backend, "has_deepseek_fp8_block_scales", False)
    assert backend.swiglu_limit_scalar == prod.swiglu_limit == 10.0

    with torch.no_grad():
        # The production layer consumes the same FP32 router GEMM bitwise.
        logits_prod = prod.gate.logits(x)
        logits_emul = emul.gate.logits(x)
        assert torch.equal(logits_prod, logits_emul)

        # The fused routing kernel (noaux_tc) must reproduce the verified
        # torch router exactly: same top-8 ID sets, same post-normalization,
        # post-scaling weights.
        _, w_ref, idx_ref = emul.gate(x)
        idx_kernel, w_kernel = prod.experts.routing_method.apply(logits_prod)
        id_sets_equal = all(
            set(idx_ref[t].tolist()) == set(idx_kernel[t].tolist()) for t in range(x.shape[0])
        )
        w_ref_sorted = torch.gather(w_ref, 1, idx_ref.argsort(dim=-1))
        w_kernel_sorted = torch.gather(w_kernel, 1, idx_kernel.long().argsort(dim=-1))
        weights_max_abs = (w_kernel_sorted - w_ref_sorted).abs().max().item()

        hf_bf16 = _load_moe(
            reader, text_config, layer_idx, device, sorted(set(idx_ref.flatten().tolist()))
        )[0]
        prod_out = prod(x).float()
        emul_out = emul(x).float()
        hf_out = hf_bf16(x).float()
        shared = prod.shared_experts(x).float()

    prod_vs_hf = compare(prod_out, hf_out, "prod_moe_vs_hf")
    emul_vs_hf = compare(emul_out, hf_out, "emul_moe_vs_hf")
    prod_vs_emul = compare(prod_out, emul_out, "prod_moe_vs_emul")
    prod_vs_model = compare(prod_out, captured, "prod_moe_vs_in_model")
    routed = prod_out - shared
    _record(
        evidence,
        "moe_production",
        {
            "layer": layer_idx,
            "tokens": int(x.shape[0]),
            "moe_backend": prod.moe_backend_name,
            "moe_layer_class": type(prod.experts).__name__,
            "op_path": "torch.ops.trtllm.fp8_block_scale_moe_runner",
            "op_backend": type(getattr(backend, "op_backend", None)).__name__,
            "activation": "clamped SwiGLU: silu(min(gate, limit)) * clamp(up, -limit, limit)",
            "swiglu_limit_scalar": float(backend.swiglu_limit_scalar),
            "quant": "FP8_BLOCK_SCALES 128x128 weights + dynamic e4m3 1x128 activations "
            "(trtllm::fp8_quantize_1x128)",
            "routing": "DeepSeekV3MoeRoutingMethod noaux_tc (trtllm::noaux_tc_op), fp32",
            "router_logits_bitwise": True,
            "top8_id_sets_equal": bool(id_sets_equal),
            "top8_weights_max_abs": weights_max_abs,
            "prod_vs_hf": prod_vs_hf,
            "emul_vs_hf": emul_vs_hf,
            "prod_vs_emul": prod_vs_emul,
            "prod_vs_in_model": prod_vs_model,
            "routed_absmax": float(routed.abs().max()),
            "shared_absmax": float(shared.abs().max()),
        },
    )
    assert id_sets_equal
    assert weights_max_abs < 1e-5, weights_max_abs
    # The production kernel must sit where the accepted fp8 emulation sits
    # relative to bf16 HF: agreement between the two fp8 paths, and no wider
    # drift from HF than the emulation's own.
    assert prod_vs_emul["all_finite"] and prod_vs_hf["all_finite"]
    assert prod_vs_emul["cosine"] > 0.998, prod_vs_emul
    assert abs(prod_vs_hf["cosine"] - emul_vs_hf["cosine"]) < 1e-3, (prod_vs_hf, emul_vs_hf)
    assert prod_vs_model["cosine"] > 0.995, prod_vs_model
    assert float(routed.abs().max()) > 1e-3 and float(shared.abs().max()) > 1e-3


def test_moe_production_extreme_activation_proves_clamp(production_moe, device, evidence):
    """The fused kernel really clamps: prod tracks the clamped emulation on
    exactly the positions where clamping changes the answer.

    ``swiglu_limit`` only matters when a gate/up pre-activation crosses 10.0,
    so the inputs are scaled until it does, and the discriminator is evaluated
    on the positions where the clamped and unclamped emulations disagree --
    everywhere else the two references coincide and prove nothing.
    """
    prod, emul = production_moe
    x = _fixed((16, emul.hidden_size), device, seed=91, scale=3.0)

    with torch.no_grad():
        prod_out = prod(x).float()
        clamped = emul(x).float()
        limit = emul.swiglu_limit
        emul.swiglu_limit = float("inf")
        unclamped = emul(x).float()
        emul.swiglu_limit = limit

    effect = (clamped - unclamped).abs()
    mask = effect > 0.05 * float(effect.max())
    assert mask.any(), "inputs never crossed the clamp; raise the scale"
    d_clamped = (prod_out - clamped).abs()[mask]
    d_unclamped = (prod_out - unclamped).abs()[mask]
    ratio = float(d_unclamped.mean() / (d_clamped.mean() + 1e-12))
    _record(
        evidence,
        "moe_production_clamp",
        {
            "masked_positions": int(mask.sum()),
            "clamp_effect_max": float(effect.max()),
            "mean_abs_vs_clamped": float(d_clamped.mean()),
            "mean_abs_vs_unclamped": float(d_unclamped.mean()),
            "discrimination_ratio": ratio,
            "swiglu_limit": limit,
        },
    )
    # On clamp-affected positions the kernel must be far closer to the clamped
    # reference than to ordinary SwiGLU.
    assert ratio > 2.0, ratio
    assert float(d_clamped.mean()) < 0.1 * float(effect.max())


def test_moe_production_decode_cuda_graph_replay(production_moe, device, evidence):
    """Captured production-MoE decode replays bitwise against eager.

    Routing runs inside the captured region from the refreshed input buffer,
    so each replay re-routes: two different token batches through one graph
    must both equal their eager counterparts bitwise.
    """
    prod, _ = production_moe
    x_buf = _fixed((2, prod.hidden_size), device, seed=101)

    def step():
        return prod(x_buf, phase="decode")

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.inference_mode(), torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        static_out = step()

    rows = []
    for trial in range(2):
        x_buf.copy_(_fixed((2, prod.hidden_size), device, seed=200 + trial))
        with torch.inference_mode():
            expected = step()
        graph.replay()
        torch.cuda.synchronize()
        equal = torch.equal(static_out, expected)
        rows.append({"trial": trial, "graph_equals_eager_bitwise": bool(equal)})
        assert equal, f"trial {trial}: production MoE graph != eager"
    _record(
        evidence,
        "moe_production_cuda_graph",
        {
            "backend": prod.moe_backend_name,
            "trials": rows,
            "captured_under": "torch.cuda.CUDAGraph",
        },
    )


def test_moe_production_block_scale_layout_and_loading(reader, production_moe, evidence):
    """The fused layer's [w3; w1] destination layout carries the checkpoint
    bytes exactly: weights and 128x128 block scales land in the documented
    halves, byte-for-byte, for spot-checked experts."""
    prod, _ = production_moe
    backend = getattr(prod.experts, "backend", prod.experts)
    num_e, inter, hidden = prod.num_experts, prod.moe_intermediate_size, prod.hidden_size
    blocks = lambda n: (n + 127) // 128  # noqa: E731

    assert tuple(backend.w3_w1_weight.shape) == (num_e, 2 * inter, hidden)
    assert backend.w3_w1_weight.dtype == torch.float8_e4m3fn
    assert tuple(backend.w2_weight.shape) == (num_e, hidden, inter)
    assert tuple(backend.w3_w1_weight_scaling_factor.shape) == (
        num_e,
        2 * blocks(inter),
        blocks(hidden),
    )
    assert backend.w3_w1_weight_scaling_factor.dtype == torch.float32
    assert tuple(backend.w2_weight_scaling_factor.shape) == (num_e, blocks(hidden), blocks(inter))

    p = f"{LAYER_PREFIX}.{_PROD_MOE_LAYER}.mlp"
    checked = []
    for eid in (0, 143, 287):
        up = reader.raw(f"{p}.experts.{eid}.up_proj.weight").to(backend.w3_w1_weight.device)
        gate = reader.raw(f"{p}.experts.{eid}.gate_proj.weight").to(backend.w3_w1_weight.device)
        down = reader.raw(f"{p}.experts.{eid}.down_proj.weight").to(backend.w2_weight.device)
        w3_half = backend.w3_w1_weight[eid, :inter]
        w1_half = backend.w3_w1_weight[eid, inter:]
        assert torch.equal(w3_half.view(torch.uint8), up.view(torch.uint8))
        assert torch.equal(w1_half.view(torch.uint8), gate.view(torch.uint8))
        assert torch.equal(backend.w2_weight[eid].view(torch.uint8), down.view(torch.uint8))
        s_up = reader.raw(f"{p}.experts.{eid}.up_proj.weight_scale_inv").to(
            backend.w3_w1_weight_scaling_factor.device
        )
        s_gate = reader.raw(f"{p}.experts.{eid}.gate_proj.weight_scale_inv").to(
            backend.w3_w1_weight_scaling_factor.device
        )
        s_down = reader.raw(f"{p}.experts.{eid}.down_proj.weight_scale_inv").to(
            backend.w2_weight_scaling_factor.device
        )
        assert torch.equal(backend.w3_w1_weight_scaling_factor[eid, : blocks(inter)], s_up)
        assert torch.equal(backend.w3_w1_weight_scaling_factor[eid, blocks(inter) :], s_gate)
        assert torch.equal(backend.w2_weight_scaling_factor[eid], s_down)
        checked.append(eid)
    _record(
        evidence,
        "moe_production_block_scales",
        {
            "experts_spot_checked": checked,
            "w3_w1_weight_shape": list(backend.w3_w1_weight.shape),
            "w3_w1_scale_shape": list(backend.w3_w1_weight_scaling_factor.shape),
            "w2_weight_shape": list(backend.w2_weight.shape),
            "w2_scale_shape": list(backend.w2_weight_scaling_factor.shape),
            "layout": "[w3(up); w1(gate)] halves, plain (non-shuffled), fp8 e4m3 + fp32 scales",
        },
    )


# ---------------------------------------------------------------------------
# Assembled decoder layer
# ---------------------------------------------------------------------------

_KDA_SHARED = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "b_proj.weight",
    "g_a_proj.weight",
    "g_b_proj.weight",
    "o_proj.weight",
)
_KDA_RENAMED = {
    "f_a_proj.weight": ("forget_gate.f_a_proj.weight", "f_a_proj.weight"),
    "f_b_proj.weight": ("forget_gate.f_b_proj.weight", "f_b_proj.weight"),
    "dt_bias": ("forget_gate.dt_bias", "dt_bias"),
    "A_log": ("forget_gate.A_log", "A_log"),
    "o_norm_weight": ("o_norm.weight", "o_norm.weight"),
}
_MLA_SHARED = (
    "q_a_proj.weight",
    "q_a_layernorm.weight",
    "q_b_proj.weight",
    "kv_a_proj_with_mqa.weight",
    "kv_a_layernorm.weight",
    "kv_b_proj.weight",
    "o_proj.weight",
    "indexer.wq_b.weight",
    "indexer.wk.weight",
    "indexer.k_norm.weight",
    "indexer.k_norm.bias",
    "indexer.weights_proj.weight",
    "indexer.index_kpool_compress_ape",
    "indexer.index_kpool_compress_gate",
)


def _load_decoder_layer(reader, text_config, schedule, layer_idx, device, experts):
    """Build HF's and TensorRT-LLM's decoder layer on identical real weights."""
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextDecoderLayer

    p = f"{LAYER_PREFIX}.{layer_idx}"
    dt = torch.bfloat16
    hf = Glm5NextTextDecoderLayer(text_config, layer_idx).to(device=device, dtype=dt).eval()
    if schedule.mlp[layer_idx] == "sparse":
        _force_fp32_router(hf.mlp.gate)
    trt = Glm5NextDecoderLayer(text_config, layer_idx, schedule).to(device).eval()
    hf_params = dict(hf.named_parameters())
    trt_params = dict(trt.named_parameters())

    with torch.no_grad():
        for name in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            value = reader.get(f"{p}.{name}", device=device).to(dt)
            hf_params[name].copy_(value)
            trt_params[name].copy_(value)

        # Hyper-connection: flat checkpoint tensors -> one mHC submodule per site.
        for site, hf_site in (("attn", "attn_hc"), ("ffn", "ffn_hc")):
            for param in ("fn", "base", "scale"):
                key = f"{p}.hc_{site}_{param}"
                # The loader used here is the model's own key mapping, so a
                # mapping regression fails this test rather than silently
                # loading via a test-local convention.
                assert remap_glm5_next_key(key) == (f"model.layers.{layer_idx}.hc_{site}.{param}")
                value = reader.raw(key).to(device)
                hf_params[f"{hf_site}.{param}"].copy_(value.to(dt))
                trt_params[f"hc_{site}.{param}"].copy_(value.float())

        # Attention
        a = f"{p}.self_attn"
        if schedule.attention[layer_idx] == "linear_attention":
            for name in _KDA_SHARED:
                value = reader.get(f"{a}.{name}", device=device).to(dt)
                hf_params[f"self_attn.{name}"].copy_(value)
                trt_params[f"self_attn.{name}"].copy_(value)
            for trt_name, (hf_name, suffix) in _KDA_RENAMED.items():
                value = reader.get(f"{a}.{suffix}", device=device)
                hf_params[f"self_attn.{hf_name}"].copy_(value.to(dt))
                trt_params[f"self_attn.{trt_name}"].copy_(
                    value.to(trt_params[f"self_attn.{trt_name}"].dtype)
                )
            conv = torch.cat(
                [reader.get(f"{a}.{n}_conv1d.weight", device=device) for n in ("q", "k", "v")],
                dim=0,
            )
            hf_params["self_attn.conv1d.weight"].copy_(
                conv.view_as(hf_params["self_attn.conv1d.weight"])
            )
            trt_params["self_attn.conv1d.weight"].copy_(
                conv.view_as(trt_params["self_attn.conv1d.weight"])
            )
        else:
            for name in _MLA_SHARED:
                value = reader.get(f"{a}.{name}", device=device).to(dt)
                hf_params[f"self_attn.{name}"].copy_(value)
                trt_params[f"self_attn.{name}"].copy_(value)

        # Feed forward
        m = f"{p}.mlp"
        if schedule.mlp[layer_idx] == "dense":
            for name in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
                value = reader.get(f"{m}.{name}", device=device).to(dt)
                hf_params[f"mlp.{name}"].copy_(value)
                trt_params[f"mlp.{name}"].copy_(value)
        else:
            hf.mlp.experts.gate_up_proj.zero_()
            hf.mlp.experts.down_proj.zero_()
            trt.mlp.gate_up_proj.zero_()
            trt.mlp.down_proj.zero_()
            for e in experts:
                gate_w = reader.get(f"{m}.experts.{e}.gate_proj.weight", device=device).to(dt)
                up_w = reader.get(f"{m}.experts.{e}.up_proj.weight", device=device).to(dt)
                down_w = reader.get(f"{m}.experts.{e}.down_proj.weight", device=device).to(dt)
                fused = torch.cat([gate_w, up_w], dim=0)
                hf.mlp.experts.gate_up_proj[e].copy_(fused)
                hf.mlp.experts.down_proj[e].copy_(down_w)
                trt.mlp.gate_up_proj[e].copy_(fused)
                trt.mlp.down_proj[e].copy_(down_w)
            for name in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
                value = reader.get(f"{m}.shared_experts.{name}", device=device).to(dt)
                hf_params[f"mlp.shared_experts.{name}"].copy_(value)
                trt_params[f"mlp.shared_experts.{name}"].copy_(value)
            weight = reader.get(f"{m}.gate.weight", device=device).to(torch.float32)
            bias = reader.get(f"{m}.gate.e_score_correction_bias", device=device).to(torch.float32)
            hf.mlp.gate.weight.copy_(weight)
            hf.mlp.gate.e_score_correction_bias.copy_(bias)
            trt.mlp.gate.weight.copy_(weight)
            trt.mlp.gate.e_score_correction_bias.copy_(bias)
    return hf, trt


def _run_trt_layer(trt, streams, text_config, device):
    """Drive one assembled layer in the context phase over packed tokens."""
    num_tokens = streams.shape[0]
    if trt.attention_type == "linear_attention":
        attn = trt.self_attn
        conv = torch.zeros(
            1, attn.conv_dim, attn.conv_kernel_size - 1, device=device, dtype=torch.bfloat16
        )
        ssm = torch.zeros(
            1, attn.num_heads, attn.head_dim, attn.head_dim, device=device, dtype=torch.float32
        )
        kwargs = dict(
            cu_seqlens=[0, num_tokens],
            slot_ids=torch.tensor([0], device=device),
            conv_pool=conv,
            ssm_pool=ssm,
            cached_lens=[0],
        )
    else:
        # The prepared-metadata carrier in miniature (pool owner + glm
        # buffers); the backend derives the paged cache state from it. See
        # test_glm5_next_attention._kpool_metadata for the real-class proof.
        from test_glm5_next_attention import _kpool_metadata, _KpoolPools

        attn = trt.self_attn
        tokens_per_block = 32
        pages = num_tokens // tokens_per_block + 2
        owner = _KpoolPools(attn, pages, tokens_per_block, device)
        metadata = _kpool_metadata(
            owner,
            block_tables=torch.arange(pages, device=device, dtype=torch.long).view(1, -1),
            kv_lens=torch.zeros(1, device=device, dtype=torch.long),
            num_contexts=1,
        )
        kwargs = dict(
            cu_seqlens=[0, num_tokens],
            cached_lens=[0],
            metadata=metadata,
        )
    return trt.forward_direct(streams, phase="prefill", **kwargs)


# Layers 0 / 3 / 22 cover every attention x MLP combination that occurs:
# linear+dense, sparse+MoE, linear+MoE. (No sparse+dense layer exists: the
# three dense MLPs are layers 0-2 and the first sparse attention is layer 3.)
@pytest.mark.parametrize("layer_idx", [0, 3, 22], ids=["linear_dense", "sparse_moe", "linear_moe"])
def test_source_activation_replay_decoder_layer(
    reader, text_config, schedule, hooked, device, evidence, layer_idx
):
    """Whole-layer replay: real four-stream input in, real four-stream output out.

    This is the integration claim for the hyper-connected decoder. Both
    hyper-connection sites, both layer norms, the attention module and the feed
    forward all run, and the result is compared against what the real model
    actually produced from the same input -- not against a re-derivation of this
    implementation.
    """
    # Every expert is materialized here, not just the ones the captured MLP
    # input selects. The replayed MLP input is not bit-identical to the captured
    # one -- it is produced by this layer's own attention and hyper-connection --
    # so a near-tied routing decision can legitimately differ between the two
    # rungs. With a subset loaded, such a token lands on a zeroed expert on one
    # side only and the layer output moves by ~144 in absolute terms. That is an
    # artifact of the fixture, not of the implementation, and loading all 288
    # experts removes it rather than papering over it with a wider tolerance.
    experts: List[int] = (
        list(range(int(text_config.n_routed_experts)))
        if schedule.mlp[layer_idx] == "sparse"
        else []
    )
    hf, trt = _load_decoder_layer(reader, text_config, schedule, layer_idx, device, experts)

    # Dispatch comes from the two literal per-layer lists. Asserting the
    # *constructed* module classes closes the loop: a layer that read the
    # attention cadence or first_k_dense_replace instead would still produce a
    # plausible output, and only the wrong module type reveals it.
    assert isinstance(
        trt.self_attn,
        Glm5NextLinearAttention
        if schedule.attention[layer_idx] == "linear_attention"
        else Glm5NextSparseAttention,
    )
    assert isinstance(trt.mlp, Glm5NextMoE if schedule.mlp[layer_idx] == "sparse" else Glm5NextMLP)

    for prompt in hooked:
        acts = prompt["activations"]
        streams_in = acts[f"layer{layer_idx}.input"].to(device=device, dtype=torch.bfloat16)[0]
        captured = acts[f"layer{layer_idx}.output"].to(device=device, dtype=torch.float32)[0]
        num_tokens = streams_in.shape[0]
        mask = torch.ones(1, num_tokens, dtype=torch.bool, device=device)
        with torch.no_grad():
            hf_out = hf(
                streams_in.unsqueeze(0),
                attention_mask=mask,
                past_key_values=None,
            )[0][0].float()
            got = _run_trt_layer(trt, streams_in, text_config, device).float()

        vs_model = compare(got, captured, "trtllm_vs_in_model")
        vs_hf = compare(got, hf_out, "trtllm_vs_standalone_hf")
        hf_vs_model = compare(hf_out, captured, "standalone_hf_vs_in_model")
        # Per-stream, so a collapse that accidentally wrote the same value into
        # all four streams cannot pass.
        per_stream = [
            compare(got[:, s], captured[:, s], f"stream{s}")["cosine"]
            for s in range(text_config.hc_mult)
        ]
        _record(
            evidence,
            "source_activation_replay_decoder_layer",
            {
                "layer": layer_idx,
                "attention_type": schedule.attention[layer_idx],
                "mlp_type": schedule.mlp[layer_idx],
                "prompt_index": prompt["index"],
                "tokens": int(num_tokens),
                "num_experts_materialized": len(experts),
                "input_absmax": float(streams_in.abs().max()),
                "trtllm_vs_in_model": vs_model,
                "trtllm_vs_standalone_hf": vs_hf,
                "standalone_hf_vs_in_model": hf_vs_model,
                "per_stream_cosine": per_stream,
                "stream_spread": float((got[:, 0] - got[:, 1]).abs().max()),
            },
        )
        label = f"L{layer_idx} p{prompt['index']}"
        assert vs_model["all_finite"], label
        # The two bf16 rungs agree tightly with each other; both may drift from
        # the FP8 in-model path, but only by the same amount.
        assert vs_hf["cosine"] > 0.999, (label, vs_hf)
        assert abs(vs_model["cosine"] - hf_vs_model["cosine"]) < 2e-3, (
            label,
            vs_model,
            hf_vs_model,
        )
        assert vs_model["cosine"] > 0.99, (label, vs_model)
        assert min(per_stream) > 0.99, (label, per_stream)
        # The four streams must genuinely differ after a hyper-connection.
        assert float((got[:, 0] - got[:, 1]).abs().max()) > 1e-3, label

    del hf, trt
    torch.cuda.empty_cache()
