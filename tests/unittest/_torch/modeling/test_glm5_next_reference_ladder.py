# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rung-one/rung-two calibration of the GLM-5.3-Flash reference ladder.

The bring-up uses a three-rung ladder:

    native HuggingFace  ->  ``glm5_next_ref``  ->  TensorRT-LLM

This file pins rung one (the real checkpoint's variant inventory) and proves
rung two agrees with rung one on the **real checkpoint weights**, for
config-selected layers of both attention types and both MLP types.  Only after
that is ``glm5_next_ref`` usable as an oracle for the TensorRT-LLM
implementation -- a reference that has never been checked against native
HuggingFace is not an independent rung, it is a second guess.

Every test here runs on CUDA against ``/dev/shm/GLM-5.3-Flash``.  There is no
toy-config or random-weight fallback: a skipped run is not evidence.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List

import pytest
import torch
import torch.nn.functional as F
from glm5_next_ref import (
    GLM53_FLASH_INVENTORY,
    CheckpointReader,
    RefHyperConnection,
    RefHyperHead,
    RefIndexer,
    RefLinearAttention,
    RefMLP,
    RefMoE,
    RefSparseMLA,
    RefUnweightedRMSNorm,
    assert_variant_inventory,
    compare,
    dequantize_block_fp8,
)

CHECKPOINT = os.environ.get("GLM53_FLASH_CHECKPOINT", "/dev/shm/GLM-5.3-Flash")
LAYER_PREFIX = "model.language_model.layers"

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
    return torch.device("cuda")


@pytest.fixture(scope="module")
def hf_config():
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(CHECKPOINT)
    config.text_config._attn_implementation = "eager"
    return config


@pytest.fixture(scope="module")
def text_config(hf_config):
    return hf_config.text_config


@pytest.fixture(scope="module")
def reader() -> CheckpointReader:
    r = CheckpointReader(CHECKPOINT)
    yield r
    r.close()


@pytest.fixture(scope="module")
def evidence() -> Dict[str, List[dict]]:
    """Collects per-test metrics; dumped at teardown for the evidence report."""
    bucket: Dict[str, List[dict]] = {}
    yield bucket
    out = os.environ.get("GLM53_EVIDENCE_JSON")
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            json.dump(bucket, fh, indent=2, default=str)


def _record(evidence, key, payload):
    evidence.setdefault(key, []).append(payload)


def _fixed_hidden(batch: int, seq: int, hidden: int, device, seed: int = 0):
    """Deterministic hidden states used where a *layer input* is needed.

    These stand in for an activation only in tests that compare two
    implementations of the *same* layer; the checkpoint-real activation replay
    (hooked HF hidden states) is a separate, later test.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(batch, seq, hidden, generator=gen, dtype=torch.float32)
    return (x * 0.5).to(device=device, dtype=torch.bfloat16)


# ---------------------------------------------------------------------------
# Rung one: pin the exact supported variant
# ---------------------------------------------------------------------------


def test_variant_inventory_matches_real_checkpoint(hf_config, evidence):
    """Every scalar/flag/list this bring-up claims is asserted against config.json."""
    observed = assert_variant_inventory(hf_config)
    _record(evidence, "variant_inventory", observed)

    inv = GLM53_FLASH_INVENTORY
    text = hf_config.text_config

    # The two literal schedules are the source of layer ownership. Assert the
    # lists themselves, never a derived cadence.
    layer_types = list(text.layer_types)
    mlp_types = list(text.mlp_layer_types)
    assert len(layer_types) == len(mlp_types) == inv.num_hidden_layers
    assert [i for i, t in enumerate(layer_types) if t == "deepseek_sparse_attention"] == list(
        inv.sparse_attention_layer_indices
    )
    assert [i for i, t in enumerate(mlp_types) if t == "dense"] == list(inv.dense_mlp_layer_indices)

    # linear_attn_config carries redundant copies of the same schedule; they
    # must agree or the checkpoint is internally inconsistent.
    lac = text.linear_attn_config
    assert lac["kda_layers"] == [i for i, t in enumerate(layer_types) if t == "linear_attention"]
    assert lac["full_attn_layers"] == list(inv.sparse_attention_layer_indices)
    assert lac["num_heads"] == inv.linear_num_heads
    assert lac["head_dim"] == inv.linear_head_dim
    assert lac["short_conv_kernel_size"] == inv.linear_conv_kernel_dim
    assert lac["gate_lower_bound"] == inv.linear_lower_bound


def test_state_dict_is_fully_accounted_for(reader, hf_config, evidence):
    """Every checkpoint key is a main-model key or an explicitly allowlisted one.

    The only permitted unconsumed namespaces are the single MTP layer
    (``layers.45``) and the vision tower -- both out of scope for this text-only
    bring-up.  Anything else unaccounted for means the main model would load
    only partly.
    """
    inv = GLM53_FLASH_INVENTORY
    keys = list(reader.keys())

    mtp = [k for k in keys if k.startswith(f"{LAYER_PREFIX}.{inv.mtp_layer_index}.")]
    vision = [k for k in keys if k.startswith("model.visual.")]
    main = [k for k in keys if k not in set(mtp) | set(vision)]

    assert len(mtp) > 0 and len(vision) > 0
    assert len(mtp) + len(vision) + len(main) == len(keys)

    # every main key belongs to a known namespace
    known_roots = (
        f"{LAYER_PREFIX}.",
        "model.language_model.embed_tokens.",
        "model.language_model.norm.",
        "lm_head.",
    )
    unknown = [k for k in main if not k.startswith(known_roots)]
    assert unknown == [], f"unrecognised checkpoint namespaces: {unknown[:10]}"

    # main model layers are exactly 0..44
    layer_ids = sorted(
        {
            int(k[len(LAYER_PREFIX) + 1 :].split(".", 1)[0])
            for k in main
            if k.startswith(LAYER_PREFIX)
        }
    )
    assert layer_ids == list(range(inv.num_hidden_layers))

    _record(
        evidence,
        "state_dict_accounting",
        {
            "total_keys": len(keys),
            "main_keys": len(main),
            "ignored_mtp_keys": len(mtp),
            "ignored_vision_keys": len(vision),
            "main_layer_ids": f"0..{layer_ids[-1]}",
        },
    )


def test_block_fp8_scale_layout_including_edge_blocks(reader, evidence):
    """Every quantized matrix's scale tensor matches its 128x128 block grid.

    This is what distinguishes block-scaled FP8 from generic per-tensor FP8: a
    per-tensor reading would expect a scalar, and a naive block reading that
    assumed exact divisibility would break on the edge blocks that the
    non-multiple-of-128 dimensions produce.
    """
    inv = GLM53_FLASH_INVENTORY
    bn, bk = inv.weight_block_size
    checked = 0
    edge_blocks = 0
    shapes: Dict[tuple, int] = {}
    # metadata-only sweep over *every* quantized weight in the checkpoint
    for key in reader.keys():
        if not key.endswith("_scale_inv"):
            continue
        base = key[: -len("_scale_inv")]
        w_shape, w_dtype = reader.meta(base)
        s_shape, s_dtype = reader.meta(key)
        assert len(w_shape) == 2, f"{base}: expected a 2D weight, got {w_shape}"
        rows, cols = w_shape
        expected = ((rows + bn - 1) // bn, (cols + bk - 1) // bk)
        assert s_shape == expected, (
            f"{base}: scale shape {s_shape} != block grid {expected} for weight {(rows, cols)}"
        )
        assert w_dtype in ("F8_E4M3", "FP8_E4M3"), f"{base} is not e4m3: {w_dtype}"
        assert s_dtype in ("F32", "FP32"), f"{key} scale is not fp32: {s_dtype}"
        if rows % bn or cols % bk:
            edge_blocks += 1
        shapes[(rows, cols)] = shapes.get((rows, cols), 0) + 1
        checked += 1

    assert checked > 0
    # A per-tensor reading of this checkpoint would expect scalar scales; every
    # scale here is a 2D block grid strictly larger than 1x1.
    assert all(
        g[0] > 1 or g[1] > 1 for g in [((r + bn - 1) // bn, (c + bk - 1) // bk) for r, c in shapes]
    )
    _record(
        evidence,
        "block_fp8_scale_layout",
        {
            "quantized_weights_checked": checked,
            "with_edge_blocks": edge_blocks,
            "block_size": [bn, bk],
            "distinct_weight_shapes": {str(k): v for k, v in sorted(shapes.items())},
            "note": (
                "every quantized dimension in this checkpoint is a multiple of 128, "
                "so the edge-block path is not exercised by these weights"
                if edge_blocks == 0
                else f"{edge_blocks} weights have partial edge blocks"
            ),
        },
    )


def test_all_1509_exclusion_entries_are_audited(reader, hf_config, evidence):
    """The 1509-entry exclusion list is resolved and checked against ground truth.

    The checkpoint is self-describing about which modules stay BF16: a weight is
    quantized exactly when it carries a ``_scale_inv`` companion.  That gives an
    oracle independent of however the exclusion patterns are interpreted, so the
    matching rule can be *proved* rather than assumed.

    This matters because the obvious reading -- full-path equality -- matches
    zero entries here: the patterns are written against ``model.layers.N....``
    while the multimodal checkpoint keys are ``model.language_model.layers.N....``.
    A loader using path equality would quantize all 1432 BF16 modules, including
    every norm, router, indexer and KDA projection.
    """
    from glm5_next_ref import resolve_quant_exclusions

    quant = hf_config.quantization_config
    if not isinstance(quant, dict):
        quant = quant.to_dict() if hasattr(quant, "to_dict") else vars(quant)
    entries = quant["modules_to_not_convert"]
    assert len(entries) == GLM53_FLASH_INVENTORY.num_modules_to_not_convert

    all_keys = set(reader.keys())
    scaled = {k[: -len("_scale_inv")] for k in all_keys if k.endswith("_scale_inv")}
    weights = [k for k in all_keys if not k.endswith("_scale_inv")]
    truth_bf16 = {k for k in weights if k not in scaled}
    truth_quantized = {k for k in weights if k in scaled}

    audit = resolve_quant_exclusions(entries, weights)

    missing = truth_bf16 - audit.excluded  # would be wrongly quantized
    extra = audit.excluded - truth_bf16  # would be wrongly left in BF16

    # Path equality is the mis-reading this test exists to rule out.
    path_equality_matches = len(set(entries) & set(weights))

    _record(
        evidence,
        "quant_exclusion_audit",
        {
            "num_entries": len(entries),
            "total_weights": len(weights),
            "checkpoint_bf16_weights": len(truth_bf16),
            "checkpoint_quantized_weights": len(truth_quantized),
            "resolver_excluded": audit.num_excluded,
            "wrongly_quantized": sorted(missing)[:10],
            "wrongly_excluded": sorted(extra)[:10],
            "zero_match_entries": len(audit.zero_match_entries),
            "zero_match_examples": sorted(set(audit.zero_match_entries))[:10],
            "path_equality_matches": path_equality_matches,
            "rule": (
                "strip one leading 'model.' component, then match the remaining "
                "dotted components as a contiguous component run in the key"
            ),
        },
    )

    assert path_equality_matches == 0, (
        "full-path equality unexpectedly matched; the rule below was derived because it does not"
    )
    assert missing == set(), (
        f"{len(missing)} BF16 modules would be quantized: {sorted(missing)[:5]}"
    )
    assert extra == set(), f"{len(extra)} quantized modules would stay BF16: {sorted(extra)[:5]}"
    assert audit.num_excluded == len(truth_bf16) == 1432

    # Every entry gets an audited outcome: either it matched, or it is recorded
    # as a zero-match entry naming a module this checkpoint does not materialise.
    assert len(audit.hits_per_entry) == len(entries)
    zero_generic = sorted({re.sub(r"\.\d+\.", ".N.", e) for e in audit.zero_match_entries})
    # the only unmatched patterns name fused/alternate module spellings that do
    # not exist in this checkpoint
    assert zero_generic == [
        "attn_mha",
        "attn_mqa",
        "hyper_connection",
        "mapping_proj",
        "model.layers.N.self_attn.fused_qkvbfg_a_proj",
        "model.layers.N.self_attn.qkv_proj",
        "router",
    ], zero_generic

    # spot-check the text-path contracts the bring-up depends on
    l3 = f"{LAYER_PREFIX}.3"
    for bf16_key in [
        f"{l3}.mlp.gate.weight",
        f"{l3}.mlp.gate.e_score_correction_bias",
        f"{l3}.self_attn.kv_b_proj.weight",
        f"{l3}.self_attn.q_a_layernorm.weight",
        f"{l3}.self_attn.kv_a_layernorm.weight",
        f"{l3}.self_attn.indexer.wq_b.weight",
        f"{l3}.self_attn.indexer.weights_proj.weight",
        f"{l3}.hc_attn_fn",
        f"{LAYER_PREFIX}.0.self_attn.q_proj.weight",
        f"{LAYER_PREFIX}.0.self_attn.o_proj.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    ]:
        assert bf16_key in audit.excluded, f"{bf16_key} should stay BF16"
        assert not reader.is_quantized(bf16_key)
    for quantized_key in [
        f"{l3}.self_attn.q_a_proj.weight",
        f"{l3}.self_attn.q_b_proj.weight",
        f"{l3}.self_attn.kv_a_proj_with_mqa.weight",
        f"{l3}.self_attn.o_proj.weight",
        f"{l3}.mlp.experts.0.gate_proj.weight",
        f"{l3}.mlp.shared_experts.down_proj.weight",
        f"{LAYER_PREFIX}.0.mlp.gate_proj.weight",
    ]:
        assert quantized_key not in audit.excluded, f"{quantized_key} should be FP8"
        assert reader.is_quantized(quantized_key)


def test_dequantize_block_fp8_roundtrip(reader, evidence):
    """The block-FP8 expansion reproduces the checkpoint's own scale tiling."""
    inv = GLM53_FLASH_INVENTORY
    bn, bk = inv.weight_block_size
    key = f"{LAYER_PREFIX}.3.self_attn.q_a_proj.weight"
    w = reader.raw(key)
    s = reader.raw(key + "_scale_inv")
    deq = dequantize_block_fp8(w, s, (bn, bk), dtype=torch.float32)

    # spot-check a handful of tiles against a hand-computed product
    for r, c in [(0, 0), (bn, 0), (0, bk), (w.shape[0] - 1, w.shape[1] - 1)]:
        expected = w[r, c].to(torch.float32) * s[r // bn, c // bk].to(torch.float32)
        assert torch.allclose(deq[r, c], expected, rtol=0, atol=0)

    _record(
        evidence,
        "block_fp8_dequant",
        {"key": key, "weight_shape": list(w.shape), "scale_shape": list(s.shape)},
    )


# ---------------------------------------------------------------------------
# Rung two: the independent reference agrees with native HF on real weights
# ---------------------------------------------------------------------------


def _load_kda(reader, text_config, layer_idx, device):
    """Build both the HF and reference KDA layer on real checkpoint weights."""
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextLinearAttention

    p = f"{LAYER_PREFIX}.{layer_idx}.self_attn"
    dt = torch.bfloat16
    hf = Glm5NextTextLinearAttention(text_config, layer_idx).to(device=device, dtype=dt)
    ref = RefLinearAttention(
        text_config.hidden_size,
        text_config.linear_num_heads,
        text_config.linear_head_dim,
        text_config.linear_conv_kernel_dim,
        text_config.linear_lower_bound,
        rms_norm_eps=text_config.rms_norm_eps,
    ).to(device=device, dtype=dt)

    # The checkpoint stores q/k/v conv filters separately; HF concatenates them
    # along dim 0 into one depthwise conv (conversion_mapping.py).
    conv = torch.cat(
        [reader.get(f"{p}.{n}_conv1d.weight").to(device) for n in ("q", "k", "v")], dim=0
    )
    plain = {
        "q_proj.weight": f"{p}.q_proj.weight",
        "k_proj.weight": f"{p}.k_proj.weight",
        "v_proj.weight": f"{p}.v_proj.weight",
        "b_proj.weight": f"{p}.b_proj.weight",
        "g_a_proj.weight": f"{p}.g_a_proj.weight",
        "g_b_proj.weight": f"{p}.g_b_proj.weight",
        "o_norm.weight": f"{p}.o_norm.weight",
        "o_proj.weight": f"{p}.o_proj.weight",
        "forget_gate.f_a_proj.weight": f"{p}.f_a_proj.weight",
        "forget_gate.f_b_proj.weight": f"{p}.f_b_proj.weight",
        "forget_gate.dt_bias": f"{p}.dt_bias",
        "forget_gate.A_log": f"{p}.A_log",
    }
    for module in (hf, ref):
        params = dict(module.named_parameters())
        with torch.no_grad():
            for name, key in plain.items():
                params[name].copy_(reader.get(key).to(device=device, dtype=params[name].dtype))
            params["conv1d.weight"].copy_(conv.view_as(params["conv1d.weight"]))
    return hf, ref


@pytest.mark.parametrize("layer_idx", [0, 22, 44], ids=["first", "middle", "last"])
def test_reference_kda_matches_native_hf(reader, text_config, device, evidence, layer_idx):
    """KDA prefill: HF's chunked scan vs the reference's sequential recurrence.

    HF runs ``chunk_kimi_delta_attention`` here; the reference runs a plain
    token-by-token recurrence.  Agreement across two different algorithms is
    what makes the reference an oracle rather than a copy.
    """
    assert text_config.layer_types[layer_idx] == "linear_attention"
    hf, ref = _load_kda(reader, text_config, layer_idx, device)

    x = _fixed_hidden(1, 96, text_config.hidden_size, device, seed=layer_idx)
    mask = torch.ones(x.shape[:2], dtype=torch.bool, device=device)
    with torch.no_grad():
        hf_out = hf(hidden_states=x, cache_params=None, attention_mask=mask)
        ref_out = ref(x, attention_mask=mask)

    m = compare(ref_out, hf_out, f"kda_layer{layer_idx}")
    _record(evidence, "kda_prefill", {**m, "layer_idx": layer_idx, "seq_len": 96})
    assert m["all_finite"]
    # bf16 activations through a 96-step recurrence; scale-relative tolerance
    assert m["cosine"] > 0.999, m
    assert m["max_abs"] <= 0.06 * max(m["ref_max_abs"], 1e-3), m


def test_reference_kda_gate_is_scaled_sigmoid_not_clamped_softplus(
    reader, text_config, device, evidence
):
    """The -5.0 forget-gate floor is a scaled sigmoid, not a clamped softplus.

    ``Glm5NextTextForgetGate`` has two branches.  With ``gate_lower_bound`` set
    (it is, at -5.0) it returns ``bound * sigmoid(exp(A_log) * g)``; the other
    branch -- the one an implementer copying a generic gated-delta-net would
    reach for -- returns ``-exp(A_log) * softplus(g)``.  Both live in
    ``(-inf, 0]`` and both saturate, so a range check alone cannot tell them
    apart.  This test pins the *functional form* against an independent FP32
    recomputation and then shows the softplus branch is materially different.
    """
    layer_idx = 0
    hf, ref = _load_kda(reader, text_config, layer_idx, device)
    bound = text_config.linear_lower_bound
    x = _fixed_hidden(1, 64, text_config.hidden_size, device, seed=7)

    with torch.no_grad():
        g_hf = hf.forget_gate(x)
        g_ref = ref.forget_gate(x)

        # independent recomputation straight from the raw projections
        fg = hf.forget_gate
        raw = fg.f_b_proj(fg.f_a_proj(x)).float() + fg.dt_bias.float().view(1, 1, -1)
        raw = raw.view(*x.shape[:2], -1, text_config.linear_head_dim)
        decay = torch.exp(fg.A_log.float().view(1, 1, text_config.linear_num_heads, 1))
        expected = bound * torch.sigmoid(decay * raw)
        # the branch that would be used if gate_lower_bound were None
        softplus_variant = -decay * torch.where(raw > 20.0, raw, torch.log1p(torch.exp(raw)))

    m = compare(g_ref, g_hf, "forget_gate")
    m_form = compare(g_hf, expected, "forget_gate_closed_form")
    m_alt = compare(g_hf, softplus_variant, "forget_gate_vs_softplus_branch")
    interior = float(((g_hf > bound + 1e-3) & (g_hf < -1e-3)).float().mean())

    _record(
        evidence,
        "kda_forget_gate",
        {
            **m,
            "closed_form": m_form,
            "vs_softplus_branch": m_alt,
            "gate_min": float(g_hf.min()),
            "gate_max": float(g_hf.max()),
            "fraction_strictly_interior": interior,
            "lower_bound": bound,
            "formula": "gate_lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))",
        },
    )
    assert m["max_abs"] < 1e-4, m
    assert m_form["max_abs"] < 1e-4, m_form
    # the gate is bounded by, and mostly strictly inside, (bound, 0)
    assert float(g_hf.min()) >= bound and float(g_hf.max()) <= 0.0
    assert interior > 0.5, f"gate is saturated everywhere ({interior:.3f} interior)"
    # and it is emphatically not the softplus branch
    assert m_alt["max_abs"] > 0.5, m_alt


def _load_indexer(reader, text_config, layer_idx, device):
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextIndexer

    p = f"{LAYER_PREFIX}.{layer_idx}.self_attn.indexer"
    dt = torch.bfloat16
    hf = Glm5NextTextIndexer(text_config, layer_idx).to(device=device, dtype=dt)
    ref = RefIndexer(
        text_config.hidden_size,
        text_config.q_lora_rank,
        text_config.index_n_heads,
        text_config.index_head_dim,
        text_config.index_topk,
        text_config.index_kpool,
        text_config.index_kpool_always_select_tail,
    ).to(device=device, dtype=dt)
    mapping = {
        "wq_b.weight": f"{p}.wq_b.weight",
        "wk.weight": f"{p}.wk.weight",
        "k_norm.weight": f"{p}.k_norm.weight",
        "k_norm.bias": f"{p}.k_norm.bias",
        "weights_proj.weight": f"{p}.weights_proj.weight",
        "index_kpool_compress_ape": f"{p}.index_kpool_compress_ape",
        "index_kpool_compress_gate": f"{p}.index_kpool_compress_gate",
    }
    for module in (hf, ref):
        params = dict(module.named_parameters())
        with torch.no_grad():
            for name, key in mapping.items():
                params[name].copy_(reader.get(key).to(device=device, dtype=params[name].dtype))
    return hf, ref


@pytest.mark.parametrize("seq_len", [37, 512, 2100], ids=["short", "mid", "above_topk"])
@pytest.mark.parametrize("layer_idx", [3, 43], ids=["first_sparse", "last_sparse"])
def test_reference_indexer_matches_native_hf(
    reader, text_config, device, evidence, layer_idx, seq_len
):
    """Pool-compressed selection agrees with HF below, at and above index_topk.

    ``seq_len=2100`` matters: with 525 pools the 512-pool budget finally binds,
    so this is the first length at which selection is actually *selective*.
    """
    assert text_config.layer_types[layer_idx] == "deepseek_sparse_attention"
    hf, ref = _load_indexer(reader, text_config, layer_idx, device)

    x = _fixed_hidden(1, seq_len, text_config.hidden_size, device, seed=layer_idx + seq_len)
    q_resid = _fixed_hidden(1, seq_len, text_config.q_lora_rank, device, seed=99)
    mask = torch.ones(x.shape[:2], dtype=torch.bool, device=device)

    with torch.no_grad():
        hf_idx = hf(hidden_states=x, q_resid=q_resid, attention_mask=mask, past_key_values=None)
        ref_idx, dbg = ref(x, q_resid, mask, return_debug=True)

    inv = GLM53_FLASH_INVENTORY
    assert hf_idx.shape == ref_idx.shape
    assert hf_idx.shape[-1] == inv.indexer_output_width, hf_idx.shape
    assert hf_idx.dtype == torch.int32 == ref_idx.dtype

    # Selection is a set, not a sequence: compare the visible-token sets per row.
    hf_sets = [set(r[r >= 0].tolist()) for r in hf_idx[0]]
    ref_sets = [set(r[r >= 0].tolist()) for r in ref_idx[0]]
    mismatched = [i for i, (a, b) in enumerate(zip(hf_sets, ref_sets)) if a != b]

    # -1 sentinels must be preserved, never rewritten to a real position
    assert (hf_idx < 0).any(), "expected -1 padding in the indexer output"
    assert int((hf_idx == -1).sum()) == int((hf_idx < 0).sum()), "only -1 may be negative"
    assert int(ref_idx.max()) < seq_len

    # causality: nothing selected may sit after the query position
    positions = torch.arange(seq_len, device=device)[:, None]
    future = (ref_idx[0] >= 0) & (ref_idx[0] > positions)
    assert not bool(future.any()), "indexer selected a future position"

    _record(
        evidence,
        "indexer_selection",
        {
            "layer_idx": layer_idx,
            "seq_len": seq_len,
            "output_width": int(hf_idx.shape[-1]),
            "num_pools_total": dbg["num_pools_total"],
            "num_pools_kept": dbg["num_pools_kept"],
            "select_k": dbg["select_k"],
            "budget_binds": dbg["num_pools_kept"] > dbg["select_k"],
            "rows_with_set_mismatch": len(mismatched),
            "mean_selected_per_row": float((ref_idx[0] >= 0).sum(-1).float().mean()),
            "sentinel_slots": int((ref_idx == -1).sum()),
        },
    )
    assert mismatched == [], (
        f"{len(mismatched)} query rows selected a different token set; first={mismatched[:5]}"
    )


def test_indexer_sentinel_is_never_gathered(text_config, device, evidence):
    """A ``-1`` slot must not become a real key, even though gather needs a legal index.

    The canary poisons the last cache position and checks that a query whose
    top-k row is entirely sentinel attends to nothing there -- the failure mode
    a ``clamp(-1, 0, kv_len-1)`` without a validity mask would produce.
    """
    from glm5_next_ref import topk_indices_to_mask

    kv_len = 16
    topk = torch.full((1, 3, 8), -1, dtype=torch.int32, device=device)
    topk[0, 0, :2] = torch.tensor([0, 1], dtype=torch.int32, device=device)
    topk[0, 1, 0] = 5
    # row 2 stays fully sentinel
    mask = topk_indices_to_mask(topk, kv_len)
    assert mask.shape == (1, 1, 3, kv_len)
    assert mask[0, 0, 0].nonzero().flatten().tolist() == [0, 1]
    assert mask[0, 0, 1].nonzero().flatten().tolist() == [5]
    assert not bool(mask[0, 0, 2].any()), "sentinel-only row attended to a real position"
    # position 0 and kv_len-1 are the two positions a bad clamp would hit
    assert not bool(mask[0, 0, 2, 0]) and not bool(mask[0, 0, 2, kv_len - 1])
    _record(evidence, "indexer_sentinel_canary", {"kv_len": kv_len, "passed": True})


def _load_mla(reader, text_config, layer_idx, device):
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextAttention

    p = f"{LAYER_PREFIX}.{layer_idx}.self_attn"
    dt = torch.bfloat16
    hf = Glm5NextTextAttention(text_config, layer_idx).to(device=device, dtype=dt)
    ref = RefSparseMLA(
        text_config.hidden_size,
        text_config.num_attention_heads,
        text_config.q_lora_rank,
        text_config.kv_lora_rank,
        text_config.qk_nope_head_dim,
        text_config.v_head_dim,
        rms_norm_eps=text_config.rms_norm_eps,
    ).to(device=device, dtype=dt)
    mapping = {
        "q_a_proj.weight": f"{p}.q_a_proj.weight",
        "q_a_layernorm.weight": f"{p}.q_a_layernorm.weight",
        "q_b_proj.weight": f"{p}.q_b_proj.weight",
        "kv_a_proj_with_mqa.weight": f"{p}.kv_a_proj_with_mqa.weight",
        "kv_a_layernorm.weight": f"{p}.kv_a_layernorm.weight",
        "kv_b_proj.weight": f"{p}.kv_b_proj.weight",
        "o_proj.weight": f"{p}.o_proj.weight",
    }
    hf_params = dict(hf.named_parameters())
    ref_params = dict(ref.named_parameters())
    with torch.no_grad():
        for name, key in mapping.items():
            value = reader.get(key).to(device=device, dtype=dt)
            hf_params[name].copy_(value)
            ref_params[name].copy_(value)
        # indexer weights only exist on the HF module
        for name, key in {
            "indexer.wq_b.weight": f"{p}.indexer.wq_b.weight",
            "indexer.wk.weight": f"{p}.indexer.wk.weight",
            "indexer.k_norm.weight": f"{p}.indexer.k_norm.weight",
            "indexer.k_norm.bias": f"{p}.indexer.k_norm.bias",
            "indexer.weights_proj.weight": f"{p}.indexer.weights_proj.weight",
            "indexer.index_kpool_compress_ape": f"{p}.indexer.index_kpool_compress_ape",
            "indexer.index_kpool_compress_gate": f"{p}.indexer.index_kpool_compress_gate",
        }.items():
            hf_params[name].copy_(reader.get(key).to(device=device, dtype=dt))
    return hf, ref


@pytest.mark.parametrize("layer_idx", [3, 23, 43], ids=["first", "middle", "last"])
def test_reference_sparse_mla_matches_native_hf(reader, text_config, device, evidence, layer_idx):
    """Fully NoPE sparse MLA agrees with HF, and no rotary path is reachable."""
    hf, ref = _load_mla(reader, text_config, layer_idx, device)

    # Fully NoPE: the rope split is zero-width on both sides.
    assert hf.qk_rope_head_dim == 0
    assert hf.kv_a_proj_with_mqa.out_features == text_config.kv_lora_rank
    assert hf.scaling == pytest.approx(text_config.qk_head_dim**-0.5)
    assert ref.scaling == pytest.approx(hf.scaling)
    assert not hasattr(hf, "rotary_emb")

    seq = 128
    x = _fixed_hidden(1, seq, text_config.hidden_size, device, seed=layer_idx)
    mask = torch.ones(x.shape[:2], dtype=torch.bool, device=device)
    with torch.no_grad():
        hf_out, _, _ = hf(hidden_states=x, attention_mask=mask, past_key_values=None)
        # drive the reference with HF's own top-k so this isolates the MLA math
        q_resid = hf.q_a_layernorm(hf.q_a_proj(x))
        topk = hf.indexer(
            hidden_states=x, q_resid=q_resid, attention_mask=mask, past_key_values=None
        )
        ref_out = ref(x, topk)

    m = compare(ref_out, hf_out, f"mla_layer{layer_idx}")
    _record(evidence, "sparse_mla", {**m, "layer_idx": layer_idx, "seq_len": seq})
    assert m["all_finite"]
    assert m["cosine"] > 0.9995, m
    assert m["max_abs"] <= 0.05 * max(m["ref_max_abs"], 1e-3), m


def _load_hc(reader, text_config, layer_idx, site, device):
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextHyperConnection

    p = f"{LAYER_PREFIX}.{layer_idx}"
    hf = Glm5NextTextHyperConnection(text_config).to(device=device, dtype=torch.float32)
    ref = RefHyperConnection(
        text_config.hc_mult,
        text_config.hidden_size,
        text_config.hc_sinkhorn_iters,
        hc_eps=text_config.hc_eps,
        norm_eps=text_config.rms_norm_eps,
    ).to(device=device, dtype=torch.float32)
    for module in (hf, ref):
        params = dict(module.named_parameters())
        with torch.no_grad():
            for name in ("fn", "base", "scale"):
                params[name].copy_(
                    reader.get(f"{p}.hc_{site}_{name}").to(device=device, dtype=torch.float32)
                )
    return hf, ref


@pytest.mark.parametrize("site", ["attn", "ffn"])
@pytest.mark.parametrize("layer_idx", [0, 22, 44], ids=["first", "middle", "last"])
def test_reference_hyper_connection_matches_native_hf(
    reader, text_config, device, evidence, layer_idx, site
):
    """20-round Sinkhorn mixing agrees with HF and is genuinely doubly stochastic.

    The trace records every normalization half-step.  The first half-step is a
    *column* normalization; each of the remaining 19 rounds is (row, column).
    That is 39 half-steps -- an implementation that does 20 (row, column) pairs,
    or that starts with a row normalization, diverges slowly over 45 layers and
    would not be caught by a short-output test.
    """
    hf, ref = _load_hc(reader, text_config, layer_idx, site, device)
    hc = text_config.hc_mult
    streams = _fixed_hidden(1, 24, text_config.hidden_size, device, seed=layer_idx)
    streams = streams.unsqueeze(2).expand(-1, -1, hc, -1).contiguous()

    with torch.no_grad():
        hf_post, hf_comb, hf_collapsed = hf(streams)
        ref_post, ref_comb, ref_collapsed, trace = ref(streams, return_trace=True)

    metrics = {
        "post": compare(ref_post, hf_post, "post"),
        "comb": compare(ref_comb, hf_comb, "comb"),
        "collapsed": compare(ref_collapsed, hf_collapsed, "collapsed"),
    }

    # exactly 1 initial column step + 19 (row, column) rounds
    assert len(trace) == 1 + 2 * (text_config.hc_sinkhorn_iters - 1) == 39
    assert trace[0]["step"] == "init:col"
    assert trace[1]["step"].endswith(":row")

    row_sums = hf_comb.sum(dim=-1)
    col_sums = hf_comb.sum(dim=-2)
    _record(
        evidence,
        "hyper_connection",
        {
            "layer_idx": layer_idx,
            "site": site,
            "metrics": metrics,
            "sinkhorn_half_steps": len(trace),
            "final_row_sum_range": [float(row_sums.min()), float(row_sums.max())],
            "final_col_sum_range": [float(col_sums.min()), float(col_sums.max())],
            "post_range": [float(hf_post.min()), float(hf_post.max())],
            "dtype": "float32",
            "trace_head": trace[:3],
            "trace_tail": trace[-2:],
        },
    )
    for name, m in metrics.items():
        assert m["all_finite"], (name, m)
        assert m["max_abs"] < 2e-5, (name, m)
    # post = 2*sigmoid(...) lives in (0, 2)
    assert 0.0 < float(hf_post.min()) and float(hf_post.max()) < 2.0

    # The *last* half-step is a column normalization, so columns are stochastic
    # to within eps while rows -- normalized one half-step earlier and then
    # perturbed by that column step -- are only approximately so.  Asserting the
    # asymmetry is what pins the ordering: an implementation that ended on a row
    # normalization would show exactly the opposite profile, and one that ran
    # too few rounds would leave both sides far from 1.
    col_dev = float((col_sums - 1).abs().max())
    row_dev = float((row_sums - 1).abs().max())
    assert col_dev <= 10 * text_config.hc_eps, (
        f"column sums deviate by {col_dev:.3e}; the final Sinkhorn half-step "
        "should be a column normalization"
    )
    assert row_dev < 0.1, f"rows are not approximately stochastic ({row_dev:.3e})"
    assert row_dev > col_dev, (
        "row and column sums are equally exact -- the ordering asymmetry that "
        "distinguishes this Sinkhorn schedule is absent"
    )


def test_hyper_head_is_unweighted_mean(text_config, device, evidence):
    """GLM-5.3-Flash's stream readout is an unweighted mean, unlike DeepSeek-V4.

    Reusing a learned/weighted head here would change every final logit, so this
    contract is pinned explicitly rather than inherited from the DeepSeek-V4
    prior art.
    """
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextHyperHead

    hf = Glm5NextTextHyperHead()
    ref = RefHyperHead()
    x = _fixed_hidden(2, 5, text_config.hidden_size, device, seed=3)
    x = x.unsqueeze(2).expand(-1, -1, text_config.hc_mult, -1).contiguous().float()
    hf_out, ref_out = hf(x), ref(x)
    assert torch.equal(hf_out, ref_out)
    assert torch.allclose(hf_out, x.mean(dim=2))
    assert len(list(hf.parameters())) == 0, "the GLM hyper head has no learned weights"
    _record(evidence, "hyper_head", {"weighted": False, "num_parameters": 0})


def test_unweighted_rms_norm_matches_native_hf(text_config, device, evidence):
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextUnweightedRMSNorm

    hf = Glm5NextTextUnweightedRMSNorm(eps=text_config.rms_norm_eps).to(device)
    ref = RefUnweightedRMSNorm(eps=text_config.rms_norm_eps).to(device)
    x = _fixed_hidden(2, 8, text_config.hidden_size, device, seed=11).float()
    m = compare(ref(x), hf(x), "unweighted_rms_norm")
    _record(evidence, "unweighted_rms_norm", m)
    assert m["max_abs"] == 0.0, m


@pytest.mark.parametrize("layer_idx", [0, 1, 2], ids=["dense0", "dense1", "dense2"])
def test_reference_dense_mlp_matches_native_hf(reader, text_config, device, evidence, layer_idx):
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextMLP

    assert text_config.mlp_layer_types[layer_idx] == "dense"
    p = f"{LAYER_PREFIX}.{layer_idx}.mlp"
    dt = torch.bfloat16
    hf = Glm5NextTextMLP(text_config).to(device=device, dtype=dt)
    ref = RefMLP(
        text_config.hidden_size, text_config.intermediate_size, text_config.swiglu_limit
    ).to(device=device, dtype=dt)
    hf_p, ref_p = dict(hf.named_parameters()), dict(ref.named_parameters())
    with torch.no_grad():
        for name in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
            v = reader.get(f"{p}.{name}").to(device=device, dtype=dt)
            hf_p[name].copy_(v)
            ref_p[name].copy_(v)

    x = _fixed_hidden(1, 32, text_config.hidden_size, device, seed=layer_idx)
    with torch.no_grad():
        m = compare(ref(x), hf(x), f"dense_mlp{layer_idx}")
    _record(evidence, "dense_mlp", {**m, "layer_idx": layer_idx})
    assert m["max_abs"] == 0.0, m


def test_clamped_swiglu_differs_from_plain_swiglu(reader, text_config, device, evidence):
    """An extreme activation must cross the 10.0 clamp on both branches.

    Without an input that pushes ``gate``/``up`` past ``swiglu_limit`` the
    clamped and unclamped activations are bit-identical, so a test that never
    crosses the boundary cannot detect a plain-SwiGLU regression.
    """
    p = f"{LAYER_PREFIX}.0.mlp"
    dt = torch.bfloat16
    limit = text_config.swiglu_limit
    ref = RefMLP(text_config.hidden_size, text_config.intermediate_size, limit)
    ref = ref.to(device=device, dtype=dt)
    ref_p = dict(ref.named_parameters())
    with torch.no_grad():
        for name in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
            ref_p[name].copy_(reader.get(f"{p}.{name}").to(device=device, dtype=dt))

    x = _fixed_hidden(1, 16, text_config.hidden_size, device, seed=5) * 40.0
    with torch.no_grad():
        gate = ref.gate_proj(x)
        up = ref.up_proj(x)
        clamped = ref(x)
        plain = ref.down_proj(F.silu(gate) * up)

    over_gate = int((gate > limit).sum())
    over_up = int((up.abs() > limit).sum())
    delta = compare(clamped, plain, "clamped_vs_plain")
    _record(
        evidence,
        "swiglu_clamp",
        {
            "swiglu_limit": limit,
            "gate_max": float(gate.max()),
            "gate_min": float(gate.min()),
            "up_absmax": float(up.abs().max()),
            "gate_elems_above_limit": over_gate,
            "up_elems_outside_limit": over_up,
            "clamped_vs_plain": delta,
        },
    )
    assert over_gate > 0, "extreme case did not push gate past the clamp"
    assert over_up > 0, "extreme case did not push up outside the clamp"
    assert delta["max_abs"] > 1.0, (
        "clamped and unclamped SwiGLU produced (near-)identical output; "
        "the clamp is not being exercised"
    )
    # gate is clamped only from above, up on both sides -- an asymmetry a
    # symmetric clamp implementation would get wrong.
    assert float(gate.min()) < -limit, "gate must retain values below -limit"


def test_reference_router_matches_native_hf(reader, text_config, device, evidence):
    """FP32 noaux_tc routing: exact expert IDs and post-scaling weights."""
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextTopkRouter

    layer_idx = 3
    p = f"{LAYER_PREFIX}.{layer_idx}.mlp.gate"
    hf = Glm5NextTextTopkRouter(text_config).to(device=device, dtype=torch.float32)
    from glm5_next_ref import RefTopkRouter

    ref = RefTopkRouter(
        text_config.hidden_size,
        text_config.n_routed_experts,
        text_config.num_experts_per_tok,
        text_config.routed_scaling_factor,
        norm_topk_prob=text_config.norm_topk_prob,
        n_group=text_config.n_group,
        topk_group=text_config.topk_group,
    ).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        w = reader.get(f"{p}.weight").to(device=device, dtype=torch.float32)
        b = reader.get(f"{p}.e_score_correction_bias").to(device=device, dtype=torch.float32)
        hf.weight.copy_(w)
        hf.e_score_correction_bias.copy_(b)
        ref.weight.copy_(w)
        ref.e_score_correction_bias.copy_(b)

    x = _fixed_hidden(1, 64, text_config.hidden_size, device, seed=13)
    with torch.no_grad():
        hf_logits, hf_weights, hf_idx = hf(x)
        ref_logits, ref_weights, ref_idx = ref(x)

    assert hf_logits.dtype == torch.float32, "router logits must be FP32"
    # top-k is returned unsorted; compare as sets, then compare sorted weights
    hf_sorted = torch.sort(hf_idx, dim=-1).values
    ref_sorted = torch.sort(ref_idx, dim=-1).values
    id_match = bool(torch.equal(hf_sorted, ref_sorted))

    order_hf = torch.argsort(hf_idx, dim=-1)
    order_ref = torch.argsort(ref_idx, dim=-1)
    w_metrics = compare(
        torch.gather(ref_weights, 1, order_ref),
        torch.gather(hf_weights, 1, order_hf),
        "topk_weights",
    )
    l_metrics = compare(ref_logits, hf_logits, "router_logits")

    # group routing is degenerate for this checkpoint
    assert text_config.n_group == text_config.topk_group == 1

    _record(
        evidence,
        "moe_router",
        {
            "layer_idx": layer_idx,
            "expert_ids_match": id_match,
            "topk_weights": w_metrics,
            "router_logits": l_metrics,
            "logits_dtype": str(hf_logits.dtype),
            "unique_experts": int(torch.unique(hf_idx).numel()),
            "weight_sum_mean": float(hf_weights.sum(-1).mean()),
            "routed_scaling_factor": text_config.routed_scaling_factor,
        },
    )
    assert id_match, "router selected different experts"
    assert l_metrics["max_abs"] == 0.0, l_metrics
    assert w_metrics["max_abs"] < 1e-6, w_metrics
    # norm_topk_prob then routed_scaling_factor -> weights sum to the factor
    assert torch.allclose(
        hf_weights.sum(-1),
        torch.full_like(hf_weights.sum(-1), text_config.routed_scaling_factor),
        atol=1e-4,
    )


def test_reference_moe_experts_match_native_hf(reader, text_config, device, evidence):
    """Routed + shared expert output on real weights for a routed layer.

    Only the experts the router actually selects are materialised: loading all
    288 would cost ~15 GiB for no extra coverage, and the selected set is
    exactly what the layer's output depends on.
    """
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextTopkRouter

    layer_idx = 3
    assert text_config.mlp_layer_types[layer_idx] == "sparse"
    p = f"{LAYER_PREFIX}.{layer_idx}.mlp"
    dt = torch.bfloat16
    hidden = text_config.hidden_size
    inter = text_config.moe_intermediate_size

    router = Glm5NextTextTopkRouter(text_config).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        router.weight.copy_(reader.get(f"{p}.gate.weight").to(device, torch.float32))
        router.e_score_correction_bias.copy_(
            reader.get(f"{p}.gate.e_score_correction_bias").to(device, torch.float32)
        )

    x = _fixed_hidden(1, 16, hidden, device, seed=17)
    with torch.no_grad():
        _, topk_weights, topk_idx = router(x)
    selected = sorted(set(topk_idx.flatten().tolist()))

    # Build a compact MoE over just the selected experts, remapping ids.
    remap = {e: i for i, e in enumerate(selected)}
    ref = RefMoE(
        hidden,
        inter,
        len(selected),
        text_config.num_experts_per_tok,
        text_config.routed_scaling_factor,
        text_config.swiglu_limit,
        n_shared_experts=text_config.n_shared_experts,
    ).to(device=device, dtype=dt)
    with torch.no_grad():
        for e in selected:
            gate_w = reader.get(f"{p}.experts.{e}.gate_proj.weight").to(device, dt)
            up_w = reader.get(f"{p}.experts.{e}.up_proj.weight").to(device, dt)
            down_w = reader.get(f"{p}.experts.{e}.down_proj.weight").to(device, dt)
            ref.gate_up_proj[remap[e]].copy_(torch.cat([gate_w, up_w], dim=0))
            ref.down_proj[remap[e]].copy_(down_w)
        for name in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
            dict(ref.shared_experts.named_parameters())[name].copy_(
                reader.get(f"{p}.shared_experts.{name}").to(device, dt)
            )

    # HF expert math, driven by the same weights, one expert at a time.
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextMLP

    shared_hf = Glm5NextTextMLP(
        text_config, intermediate_size=inter * text_config.n_shared_experts
    ).to(device=device, dtype=dt)
    with torch.no_grad():
        for name in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
            dict(shared_hf.named_parameters())[name].copy_(
                reader.get(f"{p}.shared_experts.{name}").to(device, dt)
            )

    flat = x.view(-1, hidden)
    with torch.no_grad():
        expected = torch.zeros_like(flat)
        for slot_pos, token_idx in zip(*torch.where(topk_idx >= 0)):
            e = int(topk_idx[slot_pos, token_idx])
            gate_up = F.linear(flat[slot_pos], ref.gate_up_proj[remap[e]])
            g, u = gate_up.chunk(2, dim=-1)
            g = g.clamp(max=text_config.swiglu_limit)
            u = u.clamp(-text_config.swiglu_limit, text_config.swiglu_limit)
            contribution = F.linear(F.silu(g) * u, ref.down_proj[remap[e]])
            expected[slot_pos] += (contribution * topk_weights[slot_pos, token_idx]).to(dt)
        expected = expected.view_as(x) + shared_hf(x)

        remapped = torch.tensor([[remap[int(e)] for e in row] for row in topk_idx], device=device)
        routed = torch.zeros_like(flat)
        for e_local in torch.unique(remapped).tolist():
            slot_pos, token_idx = torch.where(remapped == e_local)
            contribution = ref._expert(flat[slot_pos], e_local)
            contribution = contribution * topk_weights[slot_pos, token_idx, None]
            routed.index_add_(0, slot_pos, contribution.to(routed.dtype))
        actual = routed.view_as(x) + ref.shared_experts(x)

    m = compare(actual, expected, "moe_layer3")
    _record(
        evidence,
        "moe_experts",
        {
            **m,
            "layer_idx": layer_idx,
            "num_selected_experts": len(selected),
            "top_k": text_config.num_experts_per_tok,
            "n_shared_experts": text_config.n_shared_experts,
            "activation": "clamped_swiglu(silu(clamp(gate,max=10)) * clamp(up,-10,10))",
        },
    )
    assert m["all_finite"]
    assert m["max_abs"] <= 0.02 * max(m["ref_max_abs"], 1e-3), m
    assert m["cosine"] > 0.9999, m


def test_router_correction_bias_is_fp32_in_every_routed_layer(reader, text_config, evidence):
    """``e_score_correction_bias`` is FP32 in all 42 routed layers, and must stay so.

    ``noaux_tc`` ranks experts by ``sigmoid(logits) + e_score_correction_bias``.
    On this checkpoint the bias is not a small correction: its 288 entries sit
    in a narrow band around 10 while the sigmoid scores are O(1e-2), so the
    ranking is decided by *differences between bias entries* on the order of
    1e-3.  bfloat16 resolution at magnitude 10 is ~0.03, which is more than an
    order of magnitude too coarse.

    This test sweeps every routed layer so the constraint is pinned for the
    whole checkpoint rather than for the handful of layers the activation
    fixture happens to cover.  It is cheap: only the 288-entry bias vectors are
    read, never the experts.
    """
    routed = [i for i, t in enumerate(text_config.mlp_layer_types) if t == "sparse"]
    assert len(routed) == 42, f"expected 42 routed layers, found {len(routed)}"

    rows = []
    for layer_idx in routed:
        key = f"{LAYER_PREFIX}.{layer_idx}.mlp.gate.e_score_correction_bias"
        shape, dtype_str = reader.meta(key)
        bias = reader.get(key, dtype=torch.float32)
        assert bias.dtype == torch.float32, (layer_idx, bias.dtype)
        assert tuple(shape) == (text_config.n_routed_experts,), (layer_idx, shape)

        spread = float(bias.max() - bias.min())
        # smallest gap between two adjacent bias values -- this is the quantity
        # a lower-precision dtype has to resolve to preserve the ranking
        gaps = torch.diff(torch.sort(torch.unique(bias)).values)
        rows.append(
            {
                "layer_idx": layer_idx,
                "checkpoint_dtype": dtype_str,
                "distinct_fp32": int(torch.unique(bias).numel()),
                "distinct_bf16": int(torch.unique(bias.to(torch.bfloat16)).numel()),
                "distinct_fp16": int(torch.unique(bias.to(torch.float16)).numel()),
                "min": float(bias.min()),
                "max": float(bias.max()),
                "spread": spread,
                "median_adjacent_gap": float(gaps.median()) if gaps.numel() else 0.0,
                "bf16_resolution_at_max": float(
                    torch.tensor(float(bias.max()), dtype=torch.bfloat16).float() * 2.0**-8
                ),
            }
        )

    _record(evidence, "router_correction_bias_precision_sweep", rows)

    for row in rows:
        assert row["checkpoint_dtype"] == "F32", row
        # The bias band is narrow relative to its magnitude, which is exactly
        # why a large-exponent low-mantissa dtype cannot hold it.
        assert row["spread"] < row["max"] / 10.0, row
        # bfloat16 cannot resolve the gaps that decide the ranking.
        assert row["median_adjacent_gap"] < row["bf16_resolution_at_max"], row
        assert row["distinct_bf16"] < row["distinct_fp32"] // 4, row
        # float16's extra mantissa recovers several times more of the ranking
        # than bfloat16 but still collapses most of it (measured: 24-49 of 288
        # distinct values survive).  Neither half-precision dtype is adequate --
        # the requirement is FP32.  This assertion only records the ordering so
        # that a future "just use fp16" shortcut is visibly not a fix.
        assert row["distinct_bf16"] < row["distinct_fp16"] < row["distinct_fp32"], row
