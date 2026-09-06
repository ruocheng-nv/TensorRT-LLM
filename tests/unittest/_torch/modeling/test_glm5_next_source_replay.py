# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay real GLM-5.3-Flash activations through the independent reference.

``test_glm5_next_reference_ladder.py`` proves the reference agrees with native
HuggingFace *modules* on real checkpoint weights, but it drives them with
synthetic hidden states.  This file closes the remaining gap: it feeds the
hidden states the **real model produced on real prompts** -- captured by forward
hooks during a native ``Glm5NextForConditionalGeneration`` run -- back through
both the standalone HuggingFace module and the reference module, and compares
all three against the activation the in-model module actually emitted.

That three-way check catches two failure modes a module-only test cannot:

* a standalone module rebuilt with the wrong weights or wrong config (it would
  disagree with the captured in-model output), and
* a reference that only agrees on well-conditioned synthetic inputs (real
  activations have very different scale and outlier structure).

The fixture is produced by ``glm5_next_hf_reference.py``; point
``GLM53_HF_FIXTURE`` at it.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Dict, List

import pytest
import torch
import torch.nn.functional as F
from glm5_next_ref import (
    GLM53_FLASH_INVENTORY,
    CheckpointReader,
    RefHyperConnection,
    RefIndexer,
    RefLinearAttention,
    RefMLP,
    RefMoE,
    RefSparseMLA,
    compare,
)

CHECKPOINT = os.environ.get("GLM53_FLASH_CHECKPOINT", "/dev/shm/GLM-5.3-Flash")
FIXTURE = os.environ.get(
    "GLM53_HF_FIXTURE",
    os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            )
        ),
        "agent-flow/workspace/glm-5.3-flash-bringup/reports/hf_reference_fixture.pt",
    ),
)
LAYER_PREFIX = "model.language_model.layers"

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(
        not os.path.isdir(CHECKPOINT), reason=f"requires the checkpoint at {CHECKPOINT}"
    ),
    pytest.mark.skipif(
        not os.path.isfile(FIXTURE),
        reason=(
            f"requires the native-HF fixture at {FIXTURE}; build it with glm5_next_hf_reference.py"
        ),
    ),
]


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda")


@pytest.fixture(scope="module")
def fixture():
    return torch.load(FIXTURE, map_location="cpu", weights_only=False)


@pytest.fixture(scope="module")
def text_config():
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(CHECKPOINT)
    config.text_config._attn_implementation = "eager"
    return config.text_config


@pytest.fixture(scope="module")
def reader():
    r = CheckpointReader(CHECKPOINT)
    yield r
    r.close()


@pytest.fixture(scope="module")
def evidence():
    bucket: Dict[str, List[dict]] = {}
    yield bucket
    out = os.environ.get("GLM53_REPLAY_EVIDENCE_JSON")
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            json.dump(bucket, fh, indent=2, default=str)


def _record(evidence, key, payload):
    evidence.setdefault(key, []).append(payload)


def _activation_prompts(fixture) -> List[dict]:
    return [p for p in fixture["prompts"] if p.get("activations")]


# Modules whose projections are block-FP8 in the checkpoint run the FP8 kernel
# in-model, while both replay rungs dequantize to bf16 and use an ordinary
# matmul.  That difference is numerical, not structural, and these bounds are
# what separates the two:
#
#   * the two bf16 rungs must agree with *each other* very tightly -- this is
#     the actual correctness claim about the reference;
#   * each may drift from the in-model FP8 output by a larger, bounded amount;
#   * and they must drift by *the same* amount, since any structural error
#     would move only one of them.
FP8_REPLAY_PAIR_COSINE = 0.9999
FP8_REPLAY_MODEL_COSINE = 0.995
FP8_REPLAY_MODEL_REL_MAX_ABS = 8e-2


def _check_fp8_replay(label, m_rebuild, m_ref, m_pair) -> List[str]:
    """Return a list of failure descriptions (empty when the replay is sound)."""
    problems = []
    if m_pair["cosine"] <= FP8_REPLAY_PAIR_COSINE:
        problems.append(
            f"{label}: reference and standalone HF disagree on identical "
            f"dequantized weights (cosine {m_pair['cosine']:.7f}) -- that is a "
            f"structural difference, not FP8 error: {m_pair}"
        )
    if m_pair["max_abs"] > 0.5 * max(m_rebuild["max_abs"], 1e-6):
        problems.append(
            f"{label}: reference-vs-standalone-HF error ({m_pair['max_abs']:.4g}) is "
            f"not small relative to the FP8 gap ({m_rebuild['max_abs']:.4g})"
        )
    for name, m in (("standalone_hf", m_rebuild), ("reference", m_ref)):
        if m["cosine"] <= FP8_REPLAY_MODEL_COSINE:
            problems.append(f"{label}: {name} vs in-model cosine {m['cosine']:.6f}: {m}")
        if m["max_abs"] > FP8_REPLAY_MODEL_REL_MAX_ABS * max(m["ref_max_abs"], 1e-3):
            problems.append(f"{label}: {name} vs in-model max_abs out of envelope: {m}")
    # Both bf16 rungs must degrade against the FP8 path by the same amount; a
    # wiring error in one of them would show up as an asymmetry here.
    if abs(m_rebuild["cosine"] - m_ref["cosine"]) > 1e-3:
        problems.append(
            f"{label}: reference and standalone HF drift differently from the "
            f"in-model output ({m_ref['cosine']:.6f} vs {m_rebuild['cosine']:.6f})"
        )
    return problems


def _layers_with(fixture, suffix: str) -> List[int]:
    ids = set()
    for prompt in _activation_prompts(fixture):
        for name in prompt["activations"]:
            if name.endswith(suffix):
                ids.add(int(name.split(".")[0].removeprefix("layer")))
    return sorted(ids)


# ---------------------------------------------------------------------------
# The fixture itself is evidence: assert it is usable before relying on it
# ---------------------------------------------------------------------------


def test_native_hf_fixture_is_well_formed(fixture, evidence):
    """The native-HF run produced finite logits and a full greedy continuation."""
    prompts = fixture["prompts"]
    assert len(prompts) >= 5, "need at least five fixed prompts"
    assert fixture["decode"]["do_sample"] is False
    assert fixture["decode"]["num_beams"] == 1

    rows = []
    for p in prompts:
        logits = p["prefill_final_logits"]
        steps = p["generated_step_logits"]
        assert torch.isfinite(logits).all(), f"prompt {p['index']}: non-finite prefill logits"
        assert torch.isfinite(steps).all(), f"prompt {p['index']}: non-finite step logits"
        assert steps.shape[0] >= 32, "need at least 32 generated steps for parity"
        assert p["generated_token_ids"].numel() == steps.shape[0]
        # greedy consistency: step 0's argmax is the prefill argmax
        assert int(steps[0].argmax()) == p["prefill_greedy_token"]
        assert int(p["generated_token_ids"][0]) == p["prefill_greedy_token"]
        rows.append(
            {
                "index": p["index"],
                "num_input_tokens": int(p["input_ids"].numel()),
                "num_generated": int(steps.shape[0]),
                "prefill_greedy_token": p["prefill_greedy_token"],
                "logits_absmax": float(logits.abs().max()),
                "generated_prefix": p["generated_text"][:80],
            }
        )
    _record(
        evidence,
        "hf_fixture",
        {
            "checkpoint": fixture["checkpoint"],
            "transformers_version": fixture["transformers_version"],
            "decode": fixture["decode"],
            "capture_layers": fixture["capture_layers"],
            "prompts": rows,
        },
    )


# ---------------------------------------------------------------------------
# Replay real activations
# ---------------------------------------------------------------------------


def _load_kda(reader, text_config, layer_idx, device):
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


def test_source_activation_replay_linear_attention(reader, text_config, fixture, device, evidence):
    """Real hidden states entering KDA layers, replayed through both rungs."""
    prompts = _activation_prompts(fixture)
    assert prompts, "fixture has no captured activations"
    layer_ids = [
        i
        for i in _layers_with(fixture, "self_attn.input")
        if text_config.layer_types[i] == "linear_attention"
    ]
    assert layer_ids, "fixture captured no linear-attention layers"

    for layer_idx in layer_ids:
        hf, ref = _load_kda(reader, text_config, layer_idx, device)
        for prompt in prompts:
            acts = prompt["activations"]
            key_in = f"layer{layer_idx}.self_attn.input"
            key_out = f"layer{layer_idx}.self_attn.output"
            if key_in not in acts or key_out not in acts:
                continue
            x = acts[key_in].to(device=device, dtype=torch.bfloat16)
            captured = acts[key_out].to(device=device, dtype=torch.float32)
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=device)
            with torch.no_grad():
                hf_out = hf(hidden_states=x, cache_params=None, attention_mask=mask)
                ref_out = ref(x, attention_mask=mask)

            m_rebuild = compare(hf_out.float(), captured, "standalone_hf_vs_in_model")
            m_ref = compare(ref_out.float(), captured, "reference_vs_in_model")
            _record(
                evidence,
                "source_activation_replay_kda",
                {
                    "layer_idx": layer_idx,
                    "layer_type": "linear_attention",
                    "prompt_index": prompt["index"],
                    "seq_len": int(x.shape[1]),
                    "input_absmax": float(x.abs().max()),
                    "standalone_hf_vs_in_model": m_rebuild,
                    "reference_vs_in_model": m_ref,
                    "dtype": "bfloat16",
                    "phase": "one_shot_prefill",
                },
            )
            assert m_rebuild["all_finite"] and m_ref["all_finite"]
            assert m_rebuild["max_abs"] <= 1e-2 * max(m_rebuild["ref_max_abs"], 1e-3), m_rebuild
            assert m_ref["cosine"] > 0.999, m_ref
            assert m_ref["max_abs"] <= 6e-2 * max(m_ref["ref_max_abs"], 1e-3), m_ref


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
    shared = {
        "q_a_proj.weight": f"{p}.q_a_proj.weight",
        "q_a_layernorm.weight": f"{p}.q_a_layernorm.weight",
        "q_b_proj.weight": f"{p}.q_b_proj.weight",
        "kv_a_proj_with_mqa.weight": f"{p}.kv_a_proj_with_mqa.weight",
        "kv_a_layernorm.weight": f"{p}.kv_a_layernorm.weight",
        "kv_b_proj.weight": f"{p}.kv_b_proj.weight",
        "o_proj.weight": f"{p}.o_proj.weight",
    }
    hf_params, ref_params = dict(hf.named_parameters()), dict(ref.named_parameters())
    with torch.no_grad():
        for name, key in shared.items():
            v = reader.get(key).to(device=device, dtype=dt)
            hf_params[name].copy_(v)
            ref_params[name].copy_(v)
        for name in (
            "wq_b.weight",
            "wk.weight",
            "k_norm.weight",
            "k_norm.bias",
            "weights_proj.weight",
            "index_kpool_compress_ape",
            "index_kpool_compress_gate",
        ):
            hf_params[f"indexer.{name}"].copy_(
                reader.get(f"{p}.indexer.{name}").to(device=device, dtype=dt)
            )
    return hf, ref


def test_source_activation_replay_sparse_attention(reader, text_config, fixture, device, evidence):
    """Real hidden states entering sparse-MLA layers, including pool selection."""
    prompts = _activation_prompts(fixture)
    layer_ids = [
        i
        for i in _layers_with(fixture, "self_attn.input")
        if text_config.layer_types[i] == "deepseek_sparse_attention"
    ]
    assert layer_ids, "fixture captured no sparse-attention layers"

    failures: List[str] = []
    for layer_idx in layer_ids:
        hf, ref = _load_mla(reader, text_config, layer_idx, device)
        for prompt in prompts:
            acts = prompt["activations"]
            key_in = f"layer{layer_idx}.self_attn.input"
            key_out = f"layer{layer_idx}.self_attn.output"
            if key_in not in acts or key_out not in acts:
                continue
            x = acts[key_in].to(device=device, dtype=torch.bfloat16)
            captured = acts[key_out].to(device=device, dtype=torch.float32)
            seq = int(x.shape[1])
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=device)
            with torch.no_grad():
                hf_out, _, _ = hf(hidden_states=x, attention_mask=mask, past_key_values=None)
                q_resid = hf.q_a_layernorm(hf.q_a_proj(x))
                topk = hf.indexer(
                    hidden_states=x, q_resid=q_resid, attention_mask=mask, past_key_values=None
                )
                ref_out = ref(x, topk)

            m_rebuild = compare(hf_out.float(), captured, "standalone_hf_vs_in_model")
            m_ref = compare(ref_out.float(), captured, "reference_vs_in_model")
            m_pair = compare(ref_out.float(), hf_out.float(), "reference_vs_standalone_hf")
            selected = (topk >= 0).sum(-1).float()
            _record(
                evidence,
                "source_activation_replay_sparse_mla",
                {
                    "layer_idx": layer_idx,
                    "layer_type": "deepseek_sparse_attention",
                    "prompt_index": prompt["index"],
                    "seq_len": seq,
                    "input_absmax": float(x.abs().max()),
                    "standalone_hf_vs_in_model": m_rebuild,
                    "reference_vs_in_model": m_ref,
                    "reference_vs_standalone_hf": m_pair,
                    "quantized_projections": [
                        "q_a_proj",
                        "q_b_proj",
                        "kv_a_proj_with_mqa",
                        "o_proj",
                    ],
                    "bf16_projections": ["kv_b_proj", "q_a_layernorm", "kv_a_layernorm"],
                    "in_model_matmul": "block-FP8 e4m3 kernel",
                    "replay_matmul": "bf16 on dequantized weights",
                    "topk_output_width": int(topk.shape[-1]),
                    "mean_selected_positions": float(selected.mean()),
                    "sentinel_slots": int((topk == -1).sum()),
                    "qk_rope_head_dim": text_config.qk_rope_head_dim,
                    "phase": "one_shot_prefill",
                },
            )
            assert m_rebuild["all_finite"] and m_ref["all_finite"]
            assert int(topk.shape[-1]) == GLM53_FLASH_INVENTORY.indexer_output_width
            failures.extend(
                _check_fp8_replay(
                    f"sparse_mla L{layer_idx} p{prompt['index']}",
                    m_rebuild,
                    m_ref,
                    m_pair,
                )
            )

    assert not failures, "\n".join(failures)


def test_source_activation_replay_dense_mlp(reader, text_config, fixture, device, evidence):
    """Real hidden states entering a dense MLP, replayed through the reference."""
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextMLP

    prompts = _activation_prompts(fixture)
    layer_ids = [
        i for i in _layers_with(fixture, "mlp.input") if text_config.mlp_layer_types[i] == "dense"
    ]
    assert layer_ids, "fixture captured no dense MLP layers"

    failures: List[str] = []
    dt = torch.bfloat16
    for layer_idx in layer_ids:
        p = f"{LAYER_PREFIX}.{layer_idx}.mlp"
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

        for prompt in prompts:
            acts = prompt["activations"]
            key_in, key_out = f"layer{layer_idx}.mlp.input", f"layer{layer_idx}.mlp.output"
            if key_in not in acts or key_out not in acts:
                continue
            x = acts[key_in].to(device=device, dtype=dt)
            captured = acts[key_out].to(device=device, dtype=torch.float32)
            with torch.no_grad():
                hf_out, ref_out = hf(x), ref(x)
                gate = ref.gate_proj(x)
                up = ref.up_proj(x)
            m_rebuild = compare(hf_out.float(), captured, "standalone_hf_vs_in_model")
            m_ref = compare(ref_out.float(), captured, "reference_vs_in_model")
            m_pair = compare(ref_out.float(), hf_out.float(), "reference_vs_standalone_hf")
            _record(
                evidence,
                "source_activation_replay_dense_mlp",
                {
                    "layer_idx": layer_idx,
                    "prompt_index": prompt["index"],
                    "seq_len": int(x.shape[1]),
                    "standalone_hf_vs_in_model": m_rebuild,
                    "reference_vs_in_model": m_ref,
                    "reference_vs_standalone_hf": m_pair,
                    "quantized_projections": ["gate_proj", "up_proj", "down_proj"],
                    "in_model_matmul": "block-FP8 e4m3 kernel",
                    "replay_matmul": "bf16 on dequantized weights",
                    "gate_max_on_real_activations": float(gate.max()),
                    "up_absmax_on_real_activations": float(up.abs().max()),
                    "swiglu_limit": text_config.swiglu_limit,
                    "real_activations_cross_clamp": bool(
                        float(gate.max()) > text_config.swiglu_limit
                        or float(up.abs().max()) > text_config.swiglu_limit
                    ),
                },
            )
            assert m_rebuild["all_finite"] and m_ref["all_finite"]
            # All three dense-MLP projections are block-FP8, so the in-model path
            # runs the FP8 kernel while both replays use dequantized bf16.
            failures.extend(
                _check_fp8_replay(
                    f"dense_mlp L{layer_idx} p{prompt['index']}", m_rebuild, m_ref, m_pair
                )
            )

    assert not failures, "\n".join(failures)


def test_source_activation_replay_hyper_connection(reader, text_config, fixture, device, evidence):
    """Real four-stream residuals entering a decoder layer, mixed by both rungs."""
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextHyperConnection

    prompts = _activation_prompts(fixture)
    # decoder-layer inputs are the bare "layer<N>.input" keys (one dot), as
    # opposed to sub-block keys like "layer<N>.self_attn.input"
    layer_ids = sorted(
        {
            int(name.split(".")[0].removeprefix("layer"))
            for prompt in prompts
            for name in prompt["activations"]
            if name.count(".") == 1 and name.endswith(".input")
        }
    )
    assert layer_ids, "fixture captured no decoder-layer inputs"

    for layer_idx in layer_ids:
        for site in ("attn", "ffn"):
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
                            reader.get(f"{LAYER_PREFIX}.{layer_idx}.hc_{site}_{name}").to(
                                device=device, dtype=torch.float32
                            )
                        )

            for prompt in prompts:
                key_in = f"layer{layer_idx}.input"
                if key_in not in prompt["activations"]:
                    continue
                streams = prompt["activations"][key_in].to(device=device, dtype=torch.bfloat16)
                assert streams.shape[-2] == text_config.hc_mult, streams.shape
                with torch.no_grad():
                    hf_post, hf_comb, hf_collapsed = hf(streams)
                    ref_post, ref_comb, ref_collapsed = ref(streams)
                metrics = {
                    "post": compare(ref_post, hf_post, "post"),
                    "comb": compare(ref_comb, hf_comb, "comb"),
                    "collapsed": compare(ref_collapsed, hf_collapsed, "collapsed"),
                }
                col_dev = float((hf_comb.sum(-2) - 1).abs().max())
                row_dev = float((hf_comb.sum(-1) - 1).abs().max())
                _record(
                    evidence,
                    "source_activation_replay_hyper_connection",
                    {
                        "layer_idx": layer_idx,
                        "site": site,
                        "prompt_index": prompt["index"],
                        "seq_len": int(streams.shape[1]),
                        "hc_mult": text_config.hc_mult,
                        "sinkhorn_iters": text_config.hc_sinkhorn_iters,
                        "metrics": metrics,
                        "final_col_sum_dev": col_dev,
                        "final_row_sum_dev": row_dev,
                        "stream_absmax": float(streams.abs().max()),
                        "streams_finite": bool(torch.isfinite(streams).all()),
                    },
                )
                for name, m in metrics.items():
                    assert m["all_finite"], (name, m)
                    assert m["max_abs"] < 2e-5, (name, m)
                assert torch.isfinite(streams).all()
                assert col_dev <= 10 * text_config.hc_eps


# ---------------------------------------------------------------------------
# Routed MoE: real router decisions and real experts on real hidden states
# ---------------------------------------------------------------------------
#
# The ladder test drives the router and experts with synthetic hidden states.
# That proves the reference implements the same *function* as native HF, but it
# cannot show that the function is reached with the real routing distribution:
# random hidden states spread top-8 selection almost uniformly, while real
# activations concentrate it, and a bug in correction-bias handling or in the
# normalize-then-scale order can hide behind a uniform distribution.
#
# All 288 experts of a layer are materialized rather than remapping a selected
# subset, so every expert the real tokens actually pick is exercised with its
# own checkpoint weights and its own block-FP8 scales.  GPU-side dequantization
# keeps that at roughly 5 s and 14.5 GiB per layer.


def _build_routed_moe(reader, cfg, layer_idx, device, dt):
    """Return ``(ref, hf)`` routed-MoE rungs that share one set of weights.

    Sharing the expert tensors is deliberate: the claim under test is that the
    two *implementations* agree, so giving them separate copies of identical
    numbers would only double the memory.  Every weight is still read from the
    checkpoint exactly once, and the router/shared-expert parameters are shared
    the same way.
    """
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextMoE

    p = f"{LAYER_PREFIX}.{layer_idx}.mlp"
    n_experts = cfg.num_local_experts
    inter = cfg.moe_intermediate_size

    ref = RefMoE(
        cfg.hidden_size,
        inter,
        n_experts,
        cfg.num_experts_per_tok,
        cfg.routed_scaling_factor,
        cfg.swiglu_limit,
        n_shared_experts=cfg.n_shared_experts,
        norm_topk_prob=cfg.norm_topk_prob,
        n_group=cfg.n_group,
        topk_group=cfg.topk_group,
    ).to(device=device, dtype=dt)
    # Promote the router to FP32 *before* anything is copied into it.  The
    # checkpoint stores e_score_correction_bias as float32 with 287 distinct
    # values packed into [9.873, 10.341]; bfloat16 resolution there is ~0.03, so
    # a bf16 destination collapses them to 8 values and silently changes 7 of
    # the 8 selected experts (see
    # test_router_correction_bias_must_stay_fp32_on_real_activations).  Casting
    # after the copy does not help -- the precision is already gone.
    ref.gate.to(torch.float32)

    with torch.no_grad():
        for e in range(n_experts):
            gate_w = reader.get(f"{p}.experts.{e}.gate_proj.weight", dtype=dt, device=device)
            up_w = reader.get(f"{p}.experts.{e}.up_proj.weight", dtype=dt, device=device)
            ref.gate_up_proj[e].copy_(torch.cat([gate_w, up_w], dim=0))
            ref.down_proj[e].copy_(
                reader.get(f"{p}.experts.{e}.down_proj.weight", dtype=dt, device=device)
            )
            del gate_w, up_w
        ref.gate.weight.copy_(reader.get(f"{p}.gate.weight", device=device).to(torch.float32))
        bias = reader.get(f"{p}.gate.e_score_correction_bias", device=device)
        assert bias.dtype == torch.float32, (
            f"{p}.gate.e_score_correction_bias is stored as {bias.dtype}; the "
            f"replay assumes the checkpoint keeps it in float32"
        )
        ref.gate.e_score_correction_bias.copy_(bias)
        for name in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
            dict(ref.shared_experts.named_parameters())[name].copy_(
                reader.get(f"{p}.shared_experts.{name}", dtype=dt, device=device)
            )
    assert ref.gate.e_score_correction_bias.dtype == torch.float32
    assert ref.gate.weight.dtype == torch.float32

    # Build the HF rung on `meta` so its 14.5 GiB of expert parameters are never
    # allocated, then point every parameter at the tensors loaded above.
    with torch.device("meta"):
        hf = Glm5NextTextMoE(cfg)
    hf.experts.gate_up_proj = ref.gate_up_proj
    hf.experts.down_proj = ref.down_proj
    hf.gate.weight = ref.gate.weight
    hf.gate.e_score_correction_bias = ref.gate.e_score_correction_bias
    for name in ("gate_proj", "up_proj", "down_proj"):
        getattr(hf.shared_experts, name).weight = getattr(ref.shared_experts, name).weight
    assert not any(t.is_meta for t in hf.state_dict().values()), (
        "a Glm5NextTextMoE parameter was left on the meta device"
    )
    return ref, hf


# A routed MoE output is a weighted sum of eight expert outputs plus the shared
# expert, and on the deepest layers those terms cancel heavily (layer 44 token 1
# sums +288.8, -28.8, -7.8, ... and -56.5 down to +190).  Cancellation amplifies
# the *numerical* gap between the in-model FP8 kernel and a bf16 replay far
# beyond what it does for a single dense MLP, so a bf16-only comparison cannot
# tell "the reference math is wrong" from "FP8 is lossy here".
#
# The third rung below removes that ambiguity: it runs the reference's own
# structure on the checkpoint's *actual* e4m3 weights through HuggingFace's FP8
# kernel.  If the structure is right, that rung reproduces the captured in-model
# output tightly no matter how much cancellation there is; if the structure is
# wrong -- a mis-selected expert, a dropped shared contribution, a missing clamp
# -- it diverges just like the bf16 rungs do.
MOE_FP8_RUNG_COSINE = 0.9999
MOE_FP8_RUNG_REL_MAX_ABS = 3e-2
MOE_BF16_RUNG_COSINE = 0.995


def _fp8_linear(reader, key, x, device):
    """One projection through the same FP8 kernel the loaded model used."""
    from transformers.integrations.finegrained_fp8 import fp8_linear

    return fp8_linear(
        x,
        reader.raw(key).to(device),
        reader.raw(key + "_scale_inv").to(device),
        block_size=list(GLM53_FLASH_INVENTORY.weight_block_size),
    )


def _fp8_swiglu(reader, prefix, x, device, limit):
    gate = _fp8_linear(reader, f"{prefix}.gate_proj.weight", x, device).clamp(max=limit)
    up = _fp8_linear(reader, f"{prefix}.up_proj.weight", x, device).clamp(-limit, limit)
    return _fp8_linear(reader, f"{prefix}.down_proj.weight", F.silu(gate) * up, device)


def _fp8_routed_moe(reader, cfg, layer_idx, x, topk_weights, topk_indices, device):
    """The reference MoE structure evaluated on the real e4m3 weights.

    Routing is *not* recomputed here: it is passed in, so this rung isolates the
    expert/shared arithmetic and shares exactly the selection the other rungs
    used.
    """
    p = f"{LAYER_PREFIX}.{layer_idx}.mlp"
    flat = x.reshape(-1, x.shape[-1])
    routed = torch.zeros_like(flat)
    for expert in torch.unique(topk_indices).tolist():
        token_idx, slot_idx = torch.where(topk_indices == expert)
        contribution = _fp8_swiglu(
            reader, f"{p}.experts.{expert}", flat[token_idx], device, cfg.swiglu_limit
        )
        contribution = contribution * topk_weights[token_idx, slot_idx, None]
        routed.index_add_(0, token_idx, contribution.to(routed.dtype))
    shared = _fp8_swiglu(reader, f"{p}.shared_experts", x, device, cfg.swiglu_limit)
    return routed.view_as(x) + shared


def _check_moe_replay(label, m_ref, m_rebuild, m_pair, m_fp8) -> List[str]:
    """Return failure descriptions for one routed-MoE replay comparison."""
    problems = []
    # (1) The two bf16 implementations must agree with each other -- this is the
    #     claim that the reference implements the same function as native HF.
    if m_pair["cosine"] <= FP8_REPLAY_PAIR_COSINE:
        problems.append(
            f"{label}: reference and standalone HF disagree on identical weights "
            f"(cosine {m_pair['cosine']:.7f}); that is structural, not FP8: {m_pair}"
        )
    # (2) The same structure on the real e4m3 weights must reproduce what the
    #     loaded model actually emitted.  This is the check that catches a wrong
    #     expert set, a missing shared expert or a dropped clamp, and it stays
    #     tight even where cancellation is severe.
    if m_fp8["cosine"] <= MOE_FP8_RUNG_COSINE:
        problems.append(f"{label}: FP8-kernel rung vs in-model cosine too low: {m_fp8}")
    if m_fp8["max_abs"] > MOE_FP8_RUNG_REL_MAX_ABS * max(m_fp8["ref_max_abs"], 1e-3):
        problems.append(f"{label}: FP8-kernel rung vs in-model max_abs too large: {m_fp8}")
    # (3) The bf16 rungs may sit further out, but only in the direction
    #     quantization explains: they must still track the in-model output, and
    #     the FP8 rung must be the closer of the two.
    for name, m in (("reference", m_ref), ("standalone_hf", m_rebuild)):
        if m["cosine"] <= MOE_BF16_RUNG_COSINE:
            problems.append(f"{label}: {name} vs in-model cosine {m['cosine']:.6f}: {m}")
        if m_fp8["max_abs"] > m["max_abs"] + 1e-3:
            problems.append(
                f"{label}: the FP8-kernel rung ({m_fp8['max_abs']:.4g}) is further "
                f"from the in-model output than the bf16 {name} rung "
                f"({m['max_abs']:.4g}); the residual is not quantization error"
            )
    if abs(m_rebuild["cosine"] - m_ref["cosine"]) > 1e-3:
        problems.append(
            f"{label}: reference and standalone HF drift differently from the "
            f"in-model output ({m_ref['cosine']:.6f} vs {m_rebuild['cosine']:.6f})"
        )
    return problems


def _unclamped_routed_expert_out(ref, x, topk_weights, topk_indices):
    """Same routed-expert math as ``RefMoE`` but with the SwiGLU clamp removed.

    Used only to prove the clamp is load-bearing on a given input: if this
    matches the clamped output, that input never crossed ``swiglu_limit`` and
    the case does not qualify as extreme-activation evidence.
    """
    flat = x.reshape(-1, x.shape[-1])
    out = torch.zeros_like(flat)
    for expert in torch.unique(topk_indices).tolist():
        token_idx, slot_idx = torch.where(topk_indices == expert)
        gate_up = F.linear(flat[token_idx], ref.gate_up_proj[expert])
        gate, up = gate_up.chunk(2, dim=-1)
        contribution = F.linear(F.silu(gate) * up, ref.down_proj[expert])
        contribution = contribution * topk_weights[token_idx, slot_idx, None]
        out.index_add_(0, token_idx, contribution.to(out.dtype))
    return out.view_as(x)


def test_source_activation_replay_routed_moe(reader, text_config, fixture, device, evidence):
    """Real hidden states entering a routed MoE layer, replayed through both rungs.

    Covers the first routed layer and every later routed layer the fixture
    captured, and reports the pieces the MoE contract is actually made of: FP32
    router logits, the exact top-8 expert IDs chosen after sigmoid plus
    ``e_score_correction_bias``, the post-normalization/post-scaling weights,
    and the routed and shared contributions separately as well as summed.
    """
    prompts = _activation_prompts(fixture)
    layer_ids = [
        i for i in _layers_with(fixture, "mlp.input") if text_config.mlp_layer_types[i] == "sparse"
    ]
    assert layer_ids, "fixture captured no routed MoE layers"
    # first_k_dense_replace is a consistency check on the literal list, not the
    # source of ownership -- assert they agree rather than deriving one.
    assert layer_ids[0] == text_config.first_k_dense_replace, (
        f"first routed layer {layer_ids[0]} disagrees with "
        f"first_k_dense_replace={text_config.first_k_dense_replace}"
    )

    cfg = copy.deepcopy(text_config)
    # Glm5NextTextExperts dispatches its forward through the experts interface;
    # pin the reference (unfused) implementation so the comparison is against
    # source semantics rather than whichever kernel happens to be installed.
    cfg._experts_implementation = "eager"

    dt = torch.bfloat16
    failures: List[str] = []
    for layer_idx in layer_ids:
        ref, hf = _build_routed_moe(reader, cfg, layer_idx, device, dt)
        try:
            for prompt in prompts:
                acts = prompt["activations"]
                key_in, key_out = f"layer{layer_idx}.mlp.input", f"layer{layer_idx}.mlp.output"
                if key_in not in acts or key_out not in acts:
                    continue
                x = acts[key_in].to(device=device, dtype=dt)
                captured = acts[key_out].to(device=device, dtype=torch.float32)

                with torch.no_grad():
                    hf_logits, hf_weights, hf_idx = hf.gate(x)
                    hf_out = hf(x)
                    ref_out, dbg = ref(x, return_debug=True)

                # --- routing: logits, expert IDs, post-correction weights -----
                assert hf_logits.dtype == torch.float32, hf_logits.dtype
                assert dbg["router_logits"].dtype == torch.float32
                id_match = bool(
                    torch.equal(
                        torch.sort(hf_idx, dim=-1).values,
                        torch.sort(dbg["topk_indices"], dim=-1).values,
                    )
                )
                w_metrics = compare(
                    torch.gather(dbg["topk_weights"], 1, torch.argsort(dbg["topk_indices"], -1)),
                    torch.gather(hf_weights, 1, torch.argsort(hf_idx, -1)),
                    "topk_weights",
                )
                l_metrics = compare(dbg["router_logits"], hf_logits, "router_logits")

                # --- outputs: routed, shared, and the summed layer output -----
                with torch.no_grad():
                    fp8_out = _fp8_routed_moe(reader, cfg, layer_idx, x, hf_weights, hf_idx, device)
                m_rebuild = compare(hf_out.float(), captured, "standalone_hf_vs_in_model")
                m_ref = compare(ref_out.float(), captured, "reference_vs_in_model")
                m_pair = compare(ref_out.float(), hf_out.float(), "reference_vs_standalone_hf")
                m_fp8 = compare(fp8_out.float(), captured, "fp8_kernel_vs_in_model")

                with torch.no_grad():
                    gate_up = F.linear(
                        x.reshape(-1, x.shape[-1]), ref.gate_up_proj[int(hf_idx[0, 0])]
                    )
                    g_real, u_real = gate_up.chunk(2, dim=-1)

                _record(
                    evidence,
                    "source_activation_replay_routed_moe",
                    {
                        "layer_idx": layer_idx,
                        "prompt_index": prompt["index"],
                        "seq_len": int(x.shape[1]),
                        "is_first_routed_layer": layer_idx == layer_ids[0],
                        "num_experts": cfg.num_local_experts,
                        "top_k": cfg.num_experts_per_tok,
                        "experts_implementation": cfg._experts_implementation,
                        "router_logits_dtype": str(hf_logits.dtype),
                        "router_logits": l_metrics,
                        "expert_ids_match": id_match,
                        "unique_experts_selected": int(torch.unique(hf_idx).numel()),
                        "expert_ids_first_token": sorted(hf_idx[0].tolist()),
                        "topk_weights": w_metrics,
                        "topk_weight_sum_mean": float(hf_weights.sum(-1).mean()),
                        "routed_scaling_factor": cfg.routed_scaling_factor,
                        "norm_topk_prob": cfg.norm_topk_prob,
                        "routed_only_absmax": float(dbg["routed_only"].abs().max()),
                        "shared_only_absmax": float(dbg["shared_only"].abs().max()),
                        "routed_vs_shared_norm_ratio": float(
                            dbg["routed_only"].float().norm() / dbg["shared_only"].float().norm()
                        ),
                        "standalone_hf_vs_in_model": m_rebuild,
                        "reference_vs_in_model": m_ref,
                        "reference_vs_standalone_hf": m_pair,
                        "in_model_matmul": "block-FP8 e4m3 kernel",
                        "replay_matmul": "bf16 on dequantized weights",
                        "fp8_kernel_vs_in_model": m_fp8,
                        "fp8_rung_op_path": (
                            "transformers.integrations.finegrained_fp8.fp8_linear "
                            "(e4m3 weights, 128x128 block scales, dynamic activation scaling)"
                        ),
                        "swiglu_limit": cfg.swiglu_limit,
                        "real_gate_max": float(g_real.max()),
                        "real_up_absmax": float(u_real.abs().max()),
                        "real_activations_cross_clamp": bool(
                            float(g_real.max()) > cfg.swiglu_limit
                            or float(u_real.abs().max()) > cfg.swiglu_limit
                        ),
                    },
                )

                assert m_rebuild["all_finite"] and m_ref["all_finite"] and m_fp8["all_finite"]
                if not id_match:
                    failures.append(
                        f"routed_moe L{layer_idx} p{prompt['index']}: the two rungs "
                        f"selected different experts on real activations"
                    )
                if l_metrics["max_abs"] != 0.0:
                    failures.append(
                        f"routed_moe L{layer_idx} p{prompt['index']}: FP32 router "
                        f"logits differ: {l_metrics}"
                    )
                if w_metrics["max_abs"] > 1e-6:
                    failures.append(
                        f"routed_moe L{layer_idx} p{prompt['index']}: post-correction "
                        f"routed weights differ: {w_metrics}"
                    )
                # norm_topk_prob then routed_scaling_factor: weights sum to 2.5
                if not torch.allclose(
                    hf_weights.sum(-1),
                    torch.full_like(hf_weights.sum(-1), cfg.routed_scaling_factor),
                    atol=1e-4,
                ):
                    failures.append(
                        f"routed_moe L{layer_idx} p{prompt['index']}: routed weights do "
                        f"not sum to routed_scaling_factor"
                    )
                # The shared expert is always active, so it must contribute.
                if float(dbg["shared_only"].abs().max()) == 0.0:
                    failures.append(
                        f"routed_moe L{layer_idx} p{prompt['index']}: shared expert "
                        f"contributed nothing"
                    )
                failures.extend(
                    _check_moe_replay(
                        f"routed_moe L{layer_idx} p{prompt['index']}",
                        m_ref,
                        m_rebuild,
                        m_pair,
                        m_fp8,
                    )
                )
        finally:
            del ref, hf
            torch.cuda.empty_cache()

    assert not failures, "\n".join(failures)


def test_routed_moe_clamp_fires_on_amplified_real_activations(
    reader, text_config, fixture, device, evidence
):
    """An extreme-activation case that crosses swiglu_limit=10.0 on real weights.

    Real prompt activations do not necessarily drive the expert intermediates
    past the clamp, so an unamplified replay cannot prove the clamp is wired.
    This scales a *real* captured hidden state until the gate/up intermediates
    cross 10.0 and then requires three things: both rungs still agree, the
    clamped result differs materially from the same math with the clamp
    removed, and the clamped gate never exceeds the limit.
    """
    prompts = _activation_prompts(fixture)
    layer_ids = [
        i for i in _layers_with(fixture, "mlp.input") if text_config.mlp_layer_types[i] == "sparse"
    ]
    assert layer_ids, "fixture captured no routed MoE layers"
    layer_idx = layer_ids[0]

    cfg = copy.deepcopy(text_config)
    cfg._experts_implementation = "eager"
    dt = torch.bfloat16
    limit = cfg.swiglu_limit

    ref, hf = _build_routed_moe(reader, cfg, layer_idx, device, dt)
    try:
        prompt = prompts[0]
        x_real = prompt["activations"][f"layer{layer_idx}.mlp.input"].to(device=device, dtype=dt)

        # Grow the amplification until the intermediates actually cross the
        # clamp, so the scale is derived from the real weights rather than
        # guessed.  Routing is recomputed at each scale: sigmoid is monotonic in
        # the logits, so scaling changes which experts are picked, and the
        # clamped/unclamped comparison must use the same selection.
        chosen = None
        for scale in (4.0, 8.0, 16.0, 32.0, 64.0):
            x = x_real * scale
            with torch.no_grad():
                _, weights, idx = ref.gate(x)
                flat = x.reshape(-1, x.shape[-1])
                probe = F.linear(flat, ref.gate_up_proj[int(idx[0, 0])])
                g, u = probe.chunk(2, dim=-1)
            if float(g.max()) > limit and float(u.abs().max()) > limit:
                chosen = (scale, x, weights, idx, float(g.max()), float(u.abs().max()))
                break
        assert chosen is not None, (
            f"no tested amplification drove the expert intermediates past "
            f"swiglu_limit={limit}; the extreme-activation case did not fire"
        )
        scale, x, weights, idx, gate_max, up_absmax = chosen

        with torch.no_grad():
            ref_out, dbg = ref(x, return_debug=True)
            hf_out = hf(x)
            unclamped = _unclamped_routed_expert_out(ref, x, weights, idx)
            clamped_routed = dbg["routed_only"]
            # after the clamp the gate intermediate must be bounded
            probe = F.linear(x.reshape(-1, x.shape[-1]), ref.gate_up_proj[int(idx[0, 0])])
            g_post = probe.chunk(2, dim=-1)[0].clamp(max=limit)

        m_pair = compare(ref_out.float(), hf_out.float(), "reference_vs_standalone_hf")
        m_clamp = compare(unclamped.float(), clamped_routed.float(), "unclamped_vs_clamped")

        _record(
            evidence,
            "routed_moe_extreme_activation_clamp",
            {
                "layer_idx": layer_idx,
                "prompt_index": prompt["index"],
                "source": "real captured mlp.input, amplified",
                "amplification": scale,
                "swiglu_limit": limit,
                "gate_max_before_clamp": gate_max,
                "up_absmax_before_clamp": up_absmax,
                "gate_max_after_clamp": float(g_post.max()),
                "crossed_clamp": True,
                "reference_vs_standalone_hf": m_pair,
                "unclamped_vs_clamped": m_clamp,
                "unique_experts_selected": int(torch.unique(idx).numel()),
            },
        )

        assert m_pair["all_finite"], m_pair
        assert m_pair["cosine"] > FP8_REPLAY_PAIR_COSINE, (
            f"the two rungs disagree on a clamp-crossing input: {m_pair}"
        )
        assert float(g_post.max()) <= limit + 1e-2, float(g_post.max())
        # If removing the clamp changed nothing, the clamp was never reached and
        # this case proves nothing about clamped-vs-plain SwiGLU.
        assert m_clamp["max_abs"] > 1e-2, (
            f"removing swiglu_limit={limit} did not change the routed output; "
            f"the clamp did not fire: {m_clamp}"
        )
    finally:
        del ref, hf
        torch.cuda.empty_cache()


def test_router_correction_bias_must_stay_fp32_on_real_activations(
    reader, text_config, fixture, device, evidence
):
    """A bf16 ``e_score_correction_bias`` selects different experts. Pin FP32.

    ``noaux_tc`` ranks experts by ``sigmoid(logits) + e_score_correction_bias``.
    On this checkpoint the bias dominates that sum -- its 288 values all sit in
    ``[9.873, 10.341]`` while the sigmoid scores are ``O(1e-2)`` -- so the
    ranking is decided almost entirely by *differences between bias entries* of
    order 1e-3.  bfloat16 cannot represent those differences at magnitude ~10,
    and it does not fail loudly: routing simply falls back to the raw sigmoid
    scores and picks a different expert set.  Weight sums still equal
    ``routed_scaling_factor``, every output is finite, and cosine against the
    real layer output stays above 0.999, so only an exact expert-ID comparison
    against the real model catches it.

    This test exists because that is exactly what happened while building the
    replay above, and it is the constraint the TensorRT-LLM router loader has to
    honour: keep this buffer in FP32 regardless of the model dtype.
    """
    prompts = _activation_prompts(fixture)
    layer_ids = [
        i for i in _layers_with(fixture, "mlp.input") if text_config.mlp_layer_types[i] == "sparse"
    ]
    assert layer_ids, "fixture captured no routed MoE layers"

    top_k = text_config.num_experts_per_tok
    scale = text_config.routed_scaling_factor
    rows = []
    for layer_idx in layer_ids:
        p = f"{LAYER_PREFIX}.{layer_idx}.mlp.gate"
        w = reader.get(f"{p}.weight", device=device).to(torch.float32)
        bias32 = reader.get(f"{p}.e_score_correction_bias", device=device)
        assert bias32.dtype == torch.float32, bias32.dtype
        bias16 = bias32.to(torch.bfloat16).to(torch.float32)

        for prompt in prompts:
            key_in = f"layer{layer_idx}.mlp.input"
            if key_in not in prompt["activations"]:
                continue
            x = prompt["activations"][key_in].to(device=device, dtype=torch.bfloat16)
            with torch.no_grad():
                scores = F.linear(x.reshape(-1, text_config.hidden_size).float(), w).sigmoid()
                picked = {}
                for name, bias in (("fp32", bias32), ("bf16", bias16)):
                    idx = torch.topk(scores + bias, k=top_k, dim=-1, sorted=False)[1]
                    weights = scores.gather(1, idx)
                    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * scale
                    picked[name] = (idx, weights)

            same = int(
                (
                    torch.sort(picked["fp32"][0], -1).values
                    == torch.sort(picked["bf16"][0], -1).values
                )
                .all(-1)
                .sum()
            )
            n_tokens = picked["fp32"][0].shape[0]
            rows.append(
                {
                    "layer_idx": layer_idx,
                    "prompt_index": prompt["index"],
                    "num_tokens": n_tokens,
                    "tokens_with_identical_top_k": same,
                    "bias_dtype_in_checkpoint": str(bias32.dtype),
                    "bias_min": float(bias32.min()),
                    "bias_max": float(bias32.max()),
                    "bias_distinct_fp32": int(torch.unique(bias32).numel()),
                    "bias_distinct_bf16": int(torch.unique(bias16).numel()),
                    "max_sigmoid_score": float(scores.max()),
                    # both variants still look healthy by every aggregate metric
                    "bf16_weight_sum_ok": bool(
                        torch.allclose(
                            picked["bf16"][1].sum(-1),
                            torch.full_like(picked["bf16"][1].sum(-1), scale),
                            atol=1e-4,
                        )
                    ),
                }
            )

    assert rows, "no routed layer/prompt pair produced routing evidence"
    _record(evidence, "router_correction_bias_precision", rows)

    for row in rows:
        # The precision loss is a property of the checkpoint's own bias values:
        # FP32 resolves most experts individually, bf16 merges them into a
        # handful of buckets.
        assert row["bias_distinct_fp32"] >= text_config.num_local_experts // 2, row
        assert row["bias_distinct_bf16"] < row["bias_distinct_fp32"] // 4, row
        # ... and it is not cosmetic: it changes the selected experts.
        assert row["tokens_with_identical_top_k"] < row["num_tokens"], (
            f"bf16 bias happened to preserve every top-{top_k} set for "
            f"layer {row['layer_idx']} prompt {row['prompt_index']}; the "
            f"regression this test pins would go undetected: {row}"
        )
        # The failure is silent by every aggregate check, which is the point.
        assert row["bf16_weight_sum_ok"], row


# ---------------------------------------------------------------------------
# Long-context sparse attention: the regime where pool selection is selective
# ---------------------------------------------------------------------------
#
# The five fixed prompts are 23-50 tokens.  At 50 tokens the indexer forms 13
# size-4 pools and its 512-pool top-k keeps every one of them, so selection,
# scoring order, packed-tail rebuild and -1 padding are all inert: any
# implementation that returns "everything causal" would pass.  The ladder test
# covers seq_len=2100 with *synthetic* hidden states; this closes the last gap
# by driving the same regime with hidden states the real model produced on a
# real 2290-token prompt, where 573 pools compete for 512 slots.

LONG_FIXTURE = os.environ.get(
    "GLM53_HF_LONG_FIXTURE",
    os.path.join(os.path.dirname(FIXTURE), "hf_long_context_fixture.pt"),
)


@pytest.fixture(scope="module")
def long_fixture():
    return torch.load(LONG_FIXTURE, map_location="cpu", weights_only=False)


@pytest.mark.skipif(
    not os.path.isfile(LONG_FIXTURE),
    reason=(
        f"requires the long-context fixture at {LONG_FIXTURE}; build it with "
        f"glm5_next_hf_reference.py --long-context-tokens 2200"
    ),
)
def test_source_activation_replay_sparse_attention_long_context(
    reader, text_config, long_fixture, device, evidence
):
    """Real long-prompt activations through sparse MLA, with the pool budget binding."""
    inv = GLM53_FLASH_INVENTORY
    select_k = text_config.index_topk // text_config.index_kpool
    prompt = long_fixture["prompts"][0]
    acts = prompt["activations"]
    layer_ids = sorted(
        {
            int(name.split(".")[0].removeprefix("layer"))
            for name in acts
            if name.endswith("self_attn.input")
        }
    )
    assert layer_ids, "long-context fixture captured no attention layers"
    assert all(text_config.layer_types[i] == "deepseek_sparse_attention" for i in layer_ids), (
        f"long-context capture layers {layer_ids} are not all sparse-attention layers"
    )

    failures: List[str] = []
    for layer_idx in layer_ids:
        hf, ref = _load_mla(reader, text_config, layer_idx, device)
        try:
            key_in = f"layer{layer_idx}.self_attn.input"
            key_out = f"layer{layer_idx}.self_attn.output"
            x = acts[key_in].to(device=device, dtype=torch.bfloat16)
            captured = acts[key_out].to(device=device, dtype=torch.float32)
            seq = int(x.shape[1])
            assert seq > text_config.index_topk, (
                f"long-context prompt is only {seq} tokens; the pool budget does "
                f"not bind below index_topk={text_config.index_topk}"
            )
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=device)
            with torch.no_grad():
                hf_out, _, _ = hf(hidden_states=x, attention_mask=mask, past_key_values=None)
                q_resid = hf.q_a_layernorm(hf.q_a_proj(x))
                topk = hf.indexer(
                    hidden_states=x, q_resid=q_resid, attention_mask=mask, past_key_values=None
                )
                ref_out = ref(x, topk)

            m_rebuild = compare(hf_out.float(), captured, "standalone_hf_vs_in_model")
            m_ref = compare(ref_out.float(), captured, "reference_vs_in_model")
            m_pair = compare(ref_out.float(), hf_out.float(), "reference_vs_standalone_hf")

            # --- pool-selection contract, live for the first time on real data
            num_pools = (seq + text_config.index_kpool - 1) // text_config.index_kpool
            valid = topk >= 0
            selected_per_row = valid.sum(-1)
            positions = torch.arange(seq, device=device)[:, None]
            future = valid[0] & (topk[0] > positions)
            # the always-selected tail: each query must retain its own position
            tail_kept = bool((topk[0] == positions).any(-1).all())
            # rows past the budget must actually be pruned, not merely capped
            budget_rows = int((selected_per_row[0] < positions.squeeze(-1) + 1).sum())

            _record(
                evidence,
                "source_activation_replay_sparse_mla_long_context",
                {
                    "layer_idx": layer_idx,
                    "layer_type": "deepseek_sparse_attention",
                    "seq_len": seq,
                    "num_input_tokens": prompt["num_input_tokens"],
                    "index_topk": text_config.index_topk,
                    "index_kpool": text_config.index_kpool,
                    "select_k": select_k,
                    "num_pools_total": num_pools,
                    "pool_budget_binds": num_pools > select_k,
                    "topk_output_width": int(topk.shape[-1]),
                    "sentinel_slots": int((topk == -1).sum()),
                    "only_minus_one_is_negative": int((topk == -1).sum()) == int((topk < 0).sum()),
                    "max_selected_index": int(topk.max()),
                    "mean_selected_positions": float(selected_per_row.float().mean()),
                    "max_selected_positions": int(selected_per_row.max()),
                    "logical_width_index_topk_plus_tail": text_config.index_topk
                    + text_config.index_kpool
                    - 1,
                    "rows_above_index_topk": int(
                        (selected_per_row[0] > text_config.index_topk).sum()
                    ),
                    "rows_pruned_below_causal_count": budget_rows,
                    "tail_always_selected": tail_kept,
                    "selected_future_positions": int(future.sum()),
                    "input_absmax": float(x.abs().max()),
                    "standalone_hf_vs_in_model": m_rebuild,
                    "reference_vs_in_model": m_ref,
                    "reference_vs_standalone_hf": m_pair,
                    "in_model_matmul": "block-FP8 e4m3 kernel",
                    "replay_matmul": "bf16 on dequantized weights",
                    "phase": "one_shot_prefill",
                },
            )

            assert m_rebuild["all_finite"] and m_ref["all_finite"]
            assert int(topk.shape[-1]) == inv.indexer_output_width, topk.shape
            # The whole point of this length: selection must be selective.
            if not num_pools > select_k:
                failures.append(
                    f"long_context L{layer_idx}: {num_pools} pools <= select_k="
                    f"{select_k}; the budget still does not bind"
                )
            if budget_rows == 0:
                failures.append(
                    f"long_context L{layer_idx}: no query row was pruned below its "
                    f"causal candidate count, so top-k never discarded anything"
                )
            # The logical width is index_topk + (index_kpool - 1), and all 2051
            # slots can hold *real* positions: the 512 selected pools expand to
            # index_topk = 2048 entries and `append_visible_tail` then adds the
            # visible tail on top, up to index_kpool - 1 = 3 more.  Measured max
            # is exactly 2051, so a fixed-capacity buffer sized at index_topk
            # would truncate the always-selected tail -- the buffer Goal 1.3
            # allocates must be indexer_output_width, not index_topk.
            assert inv.indexer_output_width == text_config.index_topk + text_config.index_kpool - 1
            if int(selected_per_row.max()) > inv.indexer_output_width:
                failures.append(
                    f"long_context L{layer_idx}: a row kept "
                    f"{int(selected_per_row.max())} positions, above the logical "
                    f"width {inv.indexer_output_width}"
                )
            # -1 is a source-defined sentinel: preserved, never a real position.
            if int((topk == -1).sum()) == 0:
                failures.append(f"long_context L{layer_idx}: no -1 padding in the output")
            if int((topk == -1).sum()) != int((topk < 0).sum()):
                failures.append(f"long_context L{layer_idx}: a negative index other than -1")
            if int(topk.max()) >= seq:
                failures.append(
                    f"long_context L{layer_idx}: selected index {int(topk.max())} "
                    f"is outside the {seq}-token sequence"
                )
            if int(future.sum()) != 0:
                failures.append(
                    f"long_context L{layer_idx}: selected {int(future.sum())} future "
                    f"positions; selection is not causal"
                )
            if not tail_kept:
                failures.append(
                    f"long_context L{layer_idx}: a query row dropped its own "
                    f"position; index_kpool_always_select_tail is not honoured"
                )
            failures.extend(
                _check_fp8_replay(f"sparse_mla_long L{layer_idx}", m_rebuild, m_ref, m_pair)
            )
        finally:
            del hf, ref
            torch.cuda.empty_cache()

    assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------------
# Phase coverage: chunked prefill, token-by-token decode, cache reuse
# ---------------------------------------------------------------------------
#
# Everything above this line drives one-shot prefill: the whole prompt enters
# each module in a single call.  That regime cannot observe the state contract
# the production runtime actually depends on -- the KDA four-tap convolution
# history and recurrent state carried across chunk boundaries, and the sparse
# MLA latent-KV / packed indexer state carried across decode steps.  The risk
# register calls this out directly ("short one-shot tests pass; chunk/decode
# diverges"), and it is the cheapest place to pin the contract, because it needs
# no TensorRT-LLM cache wiring at all.
#
# Both rungs are driven independently and neither is the other's oracle:
#
#   * the HuggingFace rung uses HF's own ``DynamicCache``, which switches to
#     ``causal_conv1d_update`` + ``recurrent_kimi_delta_attention`` for
#     single-token steps and ``chunk_kimi_delta_attention`` seeded with the
#     carried recurrent state otherwise -- genuinely different kernels than the
#     one-shot path;
#   * the reference rung threads explicit ``conv_state`` / ``recurrent_state``
#     (KDA) and ``past_latent`` (MLA) through the sequential formulation.
#
# Each must reproduce *its own* one-shot output, and the reference must still
# match the activation the real model emitted.

# Schedules are described as chunk-size lists.  ``-1`` expands to "the rest of
# the sequence" and ``0`` expands to "one token at a time until the end".
PHASE_SCHEDULES = [
    ("one_shot_prefill", [-1]),
    ("prefill_then_single_decode", [-2, 1]),
    ("token_by_token_decode", [0]),
    ("sub_kernel_first_chunk", [1, -1]),
    ("uneven_sub_kernel_chunks", [3, 5, -1]),
    ("half_prefill_then_decode", [-3, 0]),
]

# A phase schedule reorders accumulation; it does not change the math.  So the
# *same-rung* comparison (phased vs that rung's own one-shot output) is pure
# bf16 round-off and gets a tight envelope -- that is the actual phase claim.
PHASE_REL_MAX_ABS = 2e-2
PHASE_MIN_COSINE = 0.9999

# The *cross-rung* comparison (phased reference vs the activation the real model
# emitted) is dominated by something else entirely: the in-model modules run
# block-FP8 kernels while both replay rungs use bf16 matmuls on dequantized
# weights.  Holding it to the phase envelope would measure quantization error,
# not phase correctness, so it reuses the envelope the corresponding one-shot
# test already established for each module.
KDA_MODEL_MIN_COSINE = 0.999
KDA_MODEL_REL_MAX_ABS = 6e-2
MLA_MODEL_MIN_COSINE = FP8_REPLAY_MODEL_COSINE
MLA_MODEL_REL_MAX_ABS = FP8_REPLAY_MODEL_REL_MAX_ABS

# The sharper cross-rung claim: phasing must not make agreement with the real
# model *worse* than one-shot already was.  A state-threading bug shows up here
# even while staying inside the (loose) quantization envelope above.
PHASE_MODEL_DEGRADATION_COSINE = 1e-4

# A negative control is only evidence if the ablation error is far outside the
# correctly-threaded error, not merely above it.
PHASE_ABLATION_MIN_RATIO = 20.0


def _expand_schedule(spec: List[int], seq_len: int) -> List[int]:
    """Turn a ``PHASE_SCHEDULES`` spec into concrete chunk sizes for ``seq_len``."""
    chunks: List[int] = []
    used = 0
    for item in spec:
        if item == 0:  # one token at a time until the end
            chunks.extend([1] * (seq_len - used))
            used = seq_len
        elif item == -1:  # all remaining tokens
            chunks.append(seq_len - used)
            used = seq_len
        elif item == -2:  # all but the last token
            chunks.append(seq_len - used - 1)
            used = seq_len - 1
        elif item == -3:  # half the sequence
            half = seq_len // 2
            chunks.append(half - used)
            used = half
        else:
            chunks.append(item)
            used += item
    if used < seq_len:
        chunks.append(seq_len - used)
    assert sum(chunks) == seq_len, f"schedule {spec} does not cover {seq_len} tokens: {chunks}"
    return [c for c in chunks if c > 0]


def _kda_hf_phased(hf, x, mask, chunks, text_config):
    """Drive the HuggingFace KDA layer through HF's own conv/recurrent cache."""
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache(config=text_config)
    outs, pos = [], 0
    with torch.no_grad():
        for n in chunks:
            outs.append(
                hf(
                    hidden_states=x[:, pos : pos + n],
                    cache_params=cache,
                    attention_mask=mask[:, pos : pos + n],
                )
            )
            pos += n
    return torch.cat(outs, dim=1), cache


def _kda_ref_phased(ref, x, mask, chunks, carry_conv=True, carry_recurrent=True, states=None):
    """Thread the reference KDA state explicitly; ablations drop one carrier.

    ``states``, when given, is filled with the final carried buffers so the
    caller can pin their shapes.
    """
    conv_state = recurrent_state = None
    outs, pos = [], 0
    with torch.no_grad():
        for n in chunks:
            out, new_conv, new_recurrent = ref(
                x[:, pos : pos + n],
                conv_state=conv_state,
                recurrent_state=recurrent_state,
                attention_mask=mask[:, pos : pos + n],
                return_state=True,
            )
            outs.append(out)
            conv_state = new_conv if carry_conv else None
            recurrent_state = new_recurrent if carry_recurrent else None
            pos += n
    if states is not None:
        states["conv"], states["recurrent"] = new_conv, new_recurrent
    return torch.cat(outs, dim=1)


def _check_phase(
    tag, m_same_rung, m_model, m_model_one_shot, bases, model_cosine, model_rel_max_abs
):
    """Return failure descriptions for one (module, prompt, schedule) triple."""
    problems: List[str] = []
    for label, m in list(m_same_rung.items()) + [("ref_vs_model", m_model)]:
        if not m["all_finite"]:
            problems.append(f"{tag} {label}: non-finite output")
    # Same-rung: only the schedule changed, so this must be round-off.
    for label, m in m_same_rung.items():
        if m["cosine"] <= PHASE_MIN_COSINE:
            problems.append(f"{tag} {label}: cosine {m['cosine']:.7f} <= {PHASE_MIN_COSINE}: {m}")
        limit = PHASE_REL_MAX_ABS * max(float(bases[label].abs().max()), 1e-3)
        if m["max_abs"] > limit:
            problems.append(f"{tag} {label}: max_abs {m['max_abs']:.6g} > {limit:.6g}: {m}")
    # Cross-rung: bounded by quantization, but phasing must not make it worse.
    if m_model["cosine"] <= model_cosine:
        problems.append(
            f"{tag} ref_vs_model: cosine {m_model['cosine']:.7f} <= {model_cosine}: {m_model}"
        )
    limit = model_rel_max_abs * max(float(bases["ref_vs_model"].abs().max()), 1e-3)
    if m_model["max_abs"] > limit:
        problems.append(f"{tag} ref_vs_model: max_abs {m_model['max_abs']:.6g} > {limit:.6g}")
    drop = m_model_one_shot["cosine"] - m_model["cosine"]
    if drop > PHASE_MODEL_DEGRADATION_COSINE:
        problems.append(
            f"{tag} ref_vs_model: phasing degraded agreement with the real model by "
            f"{drop:.3g} cosine ({m_model_one_shot['cosine']:.7f} one-shot -> "
            f"{m_model['cosine']:.7f}); state threading is lossy"
        )
    return problems


def test_source_activation_replay_kda_chunked_prefill_and_decode(
    reader, text_config, fixture, device, evidence
):
    """Real KDA activations replayed under chunked prefill and decode phases.

    A four-tap depthwise convolution needs the previous ``kernel_size - 1``
    inputs, so any chunk shorter than four tokens -- every decode step -- is
    wrong unless the convolution history crosses the boundary.  The recurrent
    state has the same requirement over the whole prefix.  Both are exercised
    here against the activation the real model produced.
    """
    prompts = _activation_prompts(fixture)
    assert prompts, "fixture has no captured activations"
    layer_ids = [
        i
        for i in _layers_with(fixture, "self_attn.input")
        if text_config.layer_types[i] == "linear_attention"
    ]
    assert layer_ids, "fixture captured no linear-attention layers"

    failures: List[str] = []
    covered_schedules = set()
    for layer_idx in layer_ids:
        hf, ref = _load_kda(reader, text_config, layer_idx, device)
        try:
            for prompt in prompts:
                acts = prompt["activations"]
                key_in = f"layer{layer_idx}.self_attn.input"
                key_out = f"layer{layer_idx}.self_attn.output"
                if key_in not in acts or key_out not in acts:
                    continue
                x = acts[key_in].to(device=device, dtype=torch.bfloat16)
                captured = acts[key_out].to(device=device, dtype=torch.float32)
                seq = int(x.shape[1])
                mask = torch.ones(x.shape[:2], dtype=torch.bool, device=device)

                with torch.no_grad():
                    hf_one = hf(hidden_states=x, cache_params=None, attention_mask=mask).float()
                    ref_one = ref(x, attention_mask=mask).float()
                    gate = ref.forget_gate(x)
                m_model_one_shot = compare(ref_one, captured, "ref_one_shot_vs_in_model")
                gate_min, gate_max = float(gate.min()), float(gate.max())
                # The -5.0 bound is a scale on a sigmoid, so the gate lives in
                # the *open* interval; a clamp implementation would touch -5.0.
                if not (text_config.linear_lower_bound < gate_min < gate_max < 0.0):
                    failures.append(
                        f"kda_phase L{layer_idx} p{prompt['index']}: forget gate "
                        f"[{gate_min:.6f}, {gate_max:.6f}] left the open interval "
                        f"({text_config.linear_lower_bound}, 0)"
                    )

                for name, spec in PHASE_SCHEDULES:
                    chunks = _expand_schedule(spec, seq)
                    hf_phased, cache = _kda_hf_phased(hf, x, mask, chunks, text_config)
                    ref_states: Dict[str, torch.Tensor] = {}
                    ref_phased = _kda_ref_phased(ref, x, mask, chunks, states=ref_states)
                    covered_schedules.add(name)

                    m_hf = compare(hf_phased.float(), hf_one, "hf_phased_vs_hf_one_shot")
                    m_ref = compare(ref_phased.float(), ref_one, "ref_phased_vs_ref_one_shot")
                    m_model = compare(ref_phased.float(), captured, "ref_phased_vs_in_model")
                    conv_state = cache.layers[layer_idx].conv_states[0]
                    recurrent_state = cache.layers[layer_idx].recurrent_states[0]
                    _record(
                        evidence,
                        "source_activation_replay_kda_phases",
                        {
                            "layer_idx": layer_idx,
                            "layer_type": "linear_attention",
                            "prompt_index": prompt["index"],
                            "seq_len": seq,
                            "phase": name,
                            "chunk_sizes": chunks if len(chunks) <= 8 else f"{len(chunks)}x1",
                            "num_chunks": len(chunks),
                            "min_chunk": min(chunks),
                            "conv_kernel_size": text_config.linear_conv_kernel_dim,
                            "sub_kernel_chunks": sum(
                                1 for c in chunks if c < text_config.linear_conv_kernel_dim
                            ),
                            "hf_phased_vs_hf_one_shot": m_hf,
                            "ref_phased_vs_ref_one_shot": m_ref,
                            "ref_phased_vs_in_model": m_model,
                            "forget_gate_min": gate_min,
                            "forget_gate_max": gate_max,
                            "forget_gate_lower_bound": text_config.linear_lower_bound,
                            "hf_conv_state_shape": list(conv_state.shape),
                            "hf_conv_state_width": int(conv_state.shape[-1]),
                            "ref_conv_state_shape": list(ref_states["conv"].shape),
                            "ref_conv_state_width": int(ref_states["conv"].shape[-1]),
                            "hf_recurrent_state_shape": list(recurrent_state.shape),
                            "hf_recurrent_state_dtype": str(recurrent_state.dtype),
                            "ref_recurrent_state_shape": list(ref_states["recurrent"].shape),
                            "ref_recurrent_state_dtype": str(ref_states["recurrent"].dtype),
                            "conv_dim": 3
                            * text_config.linear_num_heads
                            * text_config.linear_head_dim,
                            "dtype": "bfloat16",
                        },
                    )
                    # State-buffer sizing contract the production KVCacheManagerV2
                    # descriptors have to allocate.  HF keeps conv_kernel_size
                    # slots (the last holds the token just consumed, per
                    # causal_conv1d_update's roll convention) while the reference
                    # keeps only the kernel_size - 1 slots that are real left
                    # context; both reproduce one-shot prefill, so either layout
                    # is admissible and anything narrower is not.  Pinned in
                    # VariantInventory so Goal 1.3 reads one source.
                    inv = GLM53_FLASH_INVENTORY
                    if int(conv_state.shape[-1]) != inv.kda_conv_state_width:
                        failures.append(
                            f"kda_phase {name} L{layer_idx}: HF conv state width "
                            f"{int(conv_state.shape[-1])} != {inv.kda_conv_state_width}"
                        )
                    if int(ref_states["conv"].shape[-1]) != inv.kda_conv_history_width:
                        failures.append(
                            f"kda_phase {name} L{layer_idx}: reference conv history width "
                            f"{int(ref_states['conv'].shape[-1])} != {inv.kda_conv_history_width}"
                        )
                    if list(conv_state.shape[:-1]) != [x.shape[0], inv.kda_conv_dim]:
                        failures.append(
                            f"kda_phase {name} L{layer_idx}: HF conv state batch/channel "
                            f"{list(conv_state.shape[:-1])} != [{x.shape[0]}, {inv.kda_conv_dim}]"
                        )
                    expected_recurrent = [
                        x.shape[0],
                        inv.linear_num_heads,
                        inv.linear_head_dim,
                        inv.linear_head_dim,
                    ]
                    for label, state in (
                        ("hf", recurrent_state),
                        ("ref", ref_states["recurrent"]),
                    ):
                        if list(state.shape) != expected_recurrent:
                            failures.append(
                                f"kda_phase {name} L{layer_idx}: {label} recurrent state "
                                f"{list(state.shape)} != {expected_recurrent}"
                            )
                        # The recurrent state accumulates delta updates over the
                        # whole prefix; the model's bf16 would lose their tail.
                        observed_dtype = str(state.dtype).removeprefix("torch.")
                        if observed_dtype != inv.kda_recurrent_state_dtype:
                            failures.append(
                                f"kda_phase {name} L{layer_idx}: {label} recurrent state is "
                                f"{observed_dtype}, not {inv.kda_recurrent_state_dtype}"
                            )
                    failures.extend(
                        _check_phase(
                            f"kda_phase {name} L{layer_idx} p{prompt['index']}",
                            {"hf": m_hf, "ref": m_ref},
                            m_model,
                            m_model_one_shot,
                            {"hf": hf_one, "ref": ref_one, "ref_vs_model": captured},
                            KDA_MODEL_MIN_COSINE,
                            KDA_MODEL_REL_MAX_ABS,
                        )
                    )
        finally:
            del hf, ref
            torch.cuda.empty_cache()

    expected_schedules = {n for n, _ in PHASE_SCHEDULES}
    missing = expected_schedules - covered_schedules
    assert not missing, f"not every phase schedule ran: missing {sorted(missing)}"
    assert not failures, "\n".join(failures)


def test_kda_phases_require_carried_conv_and_recurrent_state(
    reader, text_config, fixture, device, evidence
):
    """Negative control: the phase test above must be able to fail.

    Drop each state carrier in turn and require the output to move far outside
    the correctly-threaded round-off.  Without this, a reference that silently
    recomputed from scratch every chunk would pass the parity test.
    """
    prompts = _activation_prompts(fixture)
    layer_ids = [
        i
        for i in _layers_with(fixture, "self_attn.input")
        if text_config.layer_types[i] == "linear_attention"
    ]
    assert layer_ids, "fixture captured no linear-attention layers"
    layer_idx = layer_ids[0]
    prompt = prompts[0]
    x = prompt["activations"][f"layer{layer_idx}.self_attn.input"].to(
        device=device, dtype=torch.bfloat16
    )
    seq = int(x.shape[1])
    mask = torch.ones(x.shape[:2], dtype=torch.bool, device=device)

    hf, ref = _load_kda(reader, text_config, layer_idx, device)
    try:
        with torch.no_grad():
            ref_one = ref(x, attention_mask=mask).float()
        # Token-by-token decode is the harshest case: every chunk is shorter
        # than the four-tap kernel and the recurrent state carries the prefix.
        chunks = _expand_schedule([0], seq)
        good = _kda_ref_phased(ref, x, mask, chunks).float()
        m_good = compare(good, ref_one, "carried_state")
        baseline = max(m_good["max_abs"], 1e-6)

        failures: List[str] = []
        ablations = {
            "no_conv_state": dict(carry_conv=False, carry_recurrent=True),
            "no_recurrent_state": dict(carry_conv=True, carry_recurrent=False),
            "no_state_at_all": dict(carry_conv=False, carry_recurrent=False),
        }
        for name, kwargs in ablations.items():
            broken = _kda_ref_phased(ref, x, mask, chunks, **kwargs).float()
            m_bad = compare(broken, ref_one, name)
            ratio = m_bad["max_abs"] / baseline
            _record(
                evidence,
                "kda_phase_negative_control",
                {
                    "layer_idx": layer_idx,
                    "prompt_index": prompt["index"],
                    "seq_len": seq,
                    "phase": "token_by_token_decode",
                    "ablation": name,
                    "carried_state_max_abs": m_good["max_abs"],
                    "ablated_max_abs": m_bad["max_abs"],
                    "ablated_cosine": m_bad["cosine"],
                    "ratio_vs_carried": ratio,
                    "required_ratio": PHASE_ABLATION_MIN_RATIO,
                },
            )
            if ratio <= PHASE_ABLATION_MIN_RATIO:
                failures.append(
                    f"ablation {name} changed the output by only {m_bad['max_abs']:.6g} "
                    f"({ratio:.1f}x the correctly-threaded {baseline:.6g}); the phase "
                    f"test cannot detect a missing state carrier"
                )
        assert not failures, "\n".join(failures)
    finally:
        del hf, ref
        torch.cuda.empty_cache()


def _load_ref_indexer(reader, text_config, layer_idx, device):
    ref = RefIndexer(
        text_config.hidden_size,
        text_config.q_lora_rank,
        text_config.index_n_heads,
        text_config.index_head_dim,
        text_config.index_topk,
        text_config.index_kpool,
        text_config.index_kpool_always_select_tail,
    ).to(device=device, dtype=torch.bfloat16)
    params = dict(ref.named_parameters())
    with torch.no_grad():
        for name in (
            "wq_b.weight",
            "wk.weight",
            "k_norm.weight",
            "k_norm.bias",
            "weights_proj.weight",
            "index_kpool_compress_ape",
            "index_kpool_compress_gate",
        ):
            params[name].copy_(
                reader.get(f"{LAYER_PREFIX}.{layer_idx}.self_attn.indexer.{name}").to(
                    device=device, dtype=torch.bfloat16
                )
            )
    return ref


def _index_rows(topk: torch.Tensor) -> List[frozenset]:
    return [frozenset(int(v) for v in row.tolist() if v >= 0) for row in topk[0]]


def test_source_activation_replay_sparse_mla_incremental_decode(
    reader, text_config, fixture, device, evidence
):
    """Real sparse-MLA activations replayed under prefill + decode cache reuse.

    Two independent state contracts are checked at once: the latent-KV cache
    (reference) versus HF's expanded key/value cache, and the packed indexer
    cache that pools are rebuilt from every step.  The indexer rows produced
    incrementally must equal the one-shot rows exactly -- if they do not, pool
    selection is not causal and a decode step is seeing a different candidate
    set than the equivalent prefill row.
    """
    inv = GLM53_FLASH_INVENTORY
    prompts = _activation_prompts(fixture)
    layer_ids = [
        i
        for i in _layers_with(fixture, "self_attn.input")
        if text_config.layer_types[i] == "deepseek_sparse_attention"
    ]
    assert layer_ids, "fixture captured no sparse-attention layers"

    from transformers.cache_utils import DynamicCache

    failures: List[str] = []
    covered_schedules = set()
    for layer_idx in layer_ids:
        hf, ref = _load_mla(reader, text_config, layer_idx, device)
        ref_indexer = _load_ref_indexer(reader, text_config, layer_idx, device)
        try:
            for prompt in prompts:
                acts = prompt["activations"]
                key_in = f"layer{layer_idx}.self_attn.input"
                key_out = f"layer{layer_idx}.self_attn.output"
                if key_in not in acts or key_out not in acts:
                    continue
                x = acts[key_in].to(device=device, dtype=torch.bfloat16)
                captured = acts[key_out].to(device=device, dtype=torch.float32)
                seq = int(x.shape[1])
                mask = torch.ones(x.shape[:2], dtype=torch.bool, device=device)

                with torch.no_grad():
                    hf_one, _, _ = hf(hidden_states=x, attention_mask=mask, past_key_values=None)
                    q_resid = hf.q_a_layernorm(hf.q_a_proj(x))
                    topk_one = hf.indexer(
                        hidden_states=x,
                        q_resid=q_resid,
                        attention_mask=mask,
                        past_key_values=None,
                    )
                    ref_one = ref(x, topk_one).float()
                hf_one = hf_one.float()
                one_shot_rows = _index_rows(topk_one)
                m_model_one_shot = compare(ref_one, captured, "ref_one_shot_vs_in_model")

                for name, spec in PHASE_SCHEDULES:
                    chunks = _expand_schedule(spec, seq)
                    covered_schedules.add(name)

                    hf_cache = DynamicCache(config=text_config)
                    hf_outs, ref_outs, phased_rows = [], [], []
                    past_latent = None
                    pos = 0
                    with torch.no_grad():
                        for n in chunks:
                            seg = x[:, pos : pos + n]
                            seg_mask = mask[:, pos : pos + n]
                            out, _, _ = hf(
                                hidden_states=seg,
                                attention_mask=seg_mask,
                                past_key_values=hf_cache,
                            )
                            hf_outs.append(out)

                            # The reference rebuilds pools from the visible
                            # prefix, mirroring what the packed indexer cache
                            # makes available, and slices the current rows.
                            prefix = x[:, : pos + n]
                            prefix_mask = mask[:, : pos + n]
                            qr = ref.q_a_layernorm(ref.q_a_proj(prefix))
                            topk_chunk = ref_indexer(prefix, qr, prefix_mask)[:, pos : pos + n]
                            ref_out, past_latent = ref(
                                seg, topk_chunk, past_latent=past_latent, return_state=True
                            )
                            ref_outs.append(ref_out)
                            phased_rows.extend(_index_rows(topk_chunk))

                            if int(topk_chunk.shape[-1]) != inv.indexer_output_width:
                                failures.append(
                                    f"mla_phase {name} L{layer_idx}: logical width "
                                    f"{int(topk_chunk.shape[-1])} != {inv.indexer_output_width}"
                                )
                            if int((topk_chunk < 0).sum()) != int((topk_chunk == -1).sum()):
                                failures.append(
                                    f"mla_phase {name} L{layer_idx}: a negative index other than -1"
                                )
                            future = topk_chunk >= (
                                pos + torch.arange(n, device=device)[None, :, None] + 1
                            )
                            if int(future.sum()) != 0:
                                failures.append(
                                    f"mla_phase {name} L{layer_idx}: selected "
                                    f"{int(future.sum())} future positions at offset {pos}"
                                )
                            pos += n

                    hf_phased = torch.cat(hf_outs, dim=1).float()
                    ref_phased = torch.cat(ref_outs, dim=1).float()
                    m_hf = compare(hf_phased, hf_one, "hf_phased_vs_hf_one_shot")
                    m_ref = compare(ref_phased, ref_one, "ref_phased_vs_ref_one_shot")
                    m_model = compare(ref_phased, captured, "ref_phased_vs_in_model")

                    indexer_keys = hf_cache.layers[layer_idx].indexer_keys
                    packed_width = int(indexer_keys.shape[-1])
                    expected_packed = inv.indexer_packed_state_width
                    _record(
                        evidence,
                        "source_activation_replay_sparse_mla_phases",
                        {
                            "layer_idx": layer_idx,
                            "layer_type": "deepseek_sparse_attention",
                            "prompt_index": prompt["index"],
                            "seq_len": seq,
                            "phase": name,
                            "chunk_sizes": chunks if len(chunks) <= 8 else f"{len(chunks)}x1",
                            "num_chunks": len(chunks),
                            "min_chunk": min(chunks),
                            "hf_phased_vs_hf_one_shot": m_hf,
                            "ref_phased_vs_ref_one_shot": m_ref,
                            "ref_phased_vs_in_model": m_model,
                            "index_rows_match_one_shot": phased_rows == one_shot_rows,
                            "topk_output_width": inv.indexer_output_width,
                            "hf_indexer_cache_shape": list(indexer_keys.shape),
                            "hf_indexer_packed_width": packed_width,
                            "expected_packed_width": expected_packed,
                            "packed_layout": "[k(index_head_dim), gate_scores(index_head_dim), valid(1)]",
                            "ref_latent_cache_shape": list(past_latent.shape),
                            "ref_latent_width": int(past_latent.shape[-1]),
                            "kv_lora_rank": text_config.kv_lora_rank,
                            "qk_rope_head_dim": text_config.qk_rope_head_dim,
                            "dtype": "bfloat16",
                        },
                    )
                    if packed_width != expected_packed:
                        failures.append(
                            f"mla_phase {name} L{layer_idx}: packed indexer cache width "
                            f"{packed_width} != 2*index_head_dim+1 = {expected_packed}"
                        )
                    if int(past_latent.shape[-1]) != inv.mla_latent_cache_width:
                        failures.append(
                            f"mla_phase {name} L{layer_idx}: latent cache width "
                            f"{int(past_latent.shape[-1])} != {inv.mla_latent_cache_width}"
                        )
                    if phased_rows != one_shot_rows:
                        differing = [
                            i for i, (a, b) in enumerate(zip(phased_rows, one_shot_rows)) if a != b
                        ]
                        failures.append(
                            f"mla_phase {name} L{layer_idx} p{prompt['index']}: incremental "
                            f"pool selection differs from one-shot at rows {differing[:8]} "
                            f"({len(differing)} of {len(one_shot_rows)}); selection is not causal"
                        )
                    failures.extend(
                        _check_phase(
                            f"mla_phase {name} L{layer_idx} p{prompt['index']}",
                            {"hf": m_hf, "ref": m_ref},
                            m_model,
                            m_model_one_shot,
                            {"hf": hf_one, "ref": ref_one, "ref_vs_model": captured},
                            MLA_MODEL_MIN_COSINE,
                            MLA_MODEL_REL_MAX_ABS,
                        )
                    )
        finally:
            del hf, ref, ref_indexer
            torch.cuda.empty_cache()

    expected_schedules = {n for n, _ in PHASE_SCHEDULES}
    missing = expected_schedules - covered_schedules
    assert not missing, f"not every phase schedule ran: missing {sorted(missing)}"
    assert not failures, "\n".join(failures)


def test_sparse_mla_phases_require_full_prefix_indexer_and_latent_cache(
    reader, text_config, fixture, device, evidence
):
    """Negative control for the sparse-MLA phase test.

    Two carriers make incremental decode correct, and both are easy to get
    wrong in a cache implementation:

    * the indexer must score pools built from the **whole visible prefix**, not
      just the tokens in the current chunk, and
    * the latent KV cache must accumulate across chunks.

    Break each in turn and require the result to leave the correctly-threaded
    round-off (or go non-finite, which is equally detectable).
    """
    prompts = _activation_prompts(fixture)
    layer_ids = [
        i
        for i in _layers_with(fixture, "self_attn.input")
        if text_config.layer_types[i] == "deepseek_sparse_attention"
    ]
    assert layer_ids, "fixture captured no sparse-attention layers"
    layer_idx = layer_ids[0]
    prompt = prompts[0]
    x = prompt["activations"][f"layer{layer_idx}.self_attn.input"].to(
        device=device, dtype=torch.bfloat16
    )
    seq = int(x.shape[1])
    mask = torch.ones(x.shape[:2], dtype=torch.bool, device=device)

    hf, ref = _load_mla(reader, text_config, layer_idx, device)
    ref_indexer = _load_ref_indexer(reader, text_config, layer_idx, device)
    try:

        def run(mla, indexer, chunks, full_prefix_indexer=True, accumulate_latent=True):
            past_latent = None
            outs, pos = [], 0
            with torch.no_grad():
                for n in chunks:
                    seg = x[:, pos : pos + n]
                    if full_prefix_indexer:
                        prefix = x[:, : pos + n]
                        pm = mask[:, : pos + n]
                        topk = indexer(prefix, mla.q_a_layernorm(mla.q_a_proj(prefix)), pm)[
                            :, pos : pos + n
                        ]
                    else:  # chunk-local: the indexer forgot the prefix
                        topk = indexer(
                            seg, mla.q_a_layernorm(mla.q_a_proj(seg)), mask[:, pos : pos + n]
                        )
                    out, new_latent = mla(seg, topk, past_latent=past_latent, return_state=True)
                    outs.append(out)
                    past_latent = new_latent if accumulate_latent else None
                    pos += n
            return torch.cat(outs, dim=1).float()

        with torch.no_grad():
            q_resid = ref.q_a_layernorm(ref.q_a_proj(x))
            ref_one = ref(x, ref_indexer(x, q_resid, mask)).float()
        chunks = _expand_schedule([0], seq)
        good = run(ref, ref_indexer, chunks)
        m_good = compare(good, ref_one, "carried_state")
        baseline = max(m_good["max_abs"], 1e-6)

        failures: List[str] = []
        ablations = {
            "chunk_local_indexer": dict(full_prefix_indexer=False),
            "no_latent_accumulation": dict(accumulate_latent=False),
        }
        for name, kwargs in ablations.items():
            broken = run(ref, ref_indexer, chunks, **kwargs)
            m_bad = compare(broken, ref_one, name)
            detected_by_nonfinite = not m_bad["all_finite"]
            ratio = float("inf") if detected_by_nonfinite else m_bad["max_abs"] / baseline
            _record(
                evidence,
                "sparse_mla_phase_negative_control",
                {
                    "layer_idx": layer_idx,
                    "prompt_index": prompt["index"],
                    "seq_len": seq,
                    "phase": "token_by_token_decode",
                    "ablation": name,
                    "carried_state_max_abs": m_good["max_abs"],
                    "ablated_max_abs": m_bad["max_abs"],
                    "ablated_cosine": m_bad["cosine"],
                    "ablated_all_finite": m_bad["all_finite"],
                    "detected_by": "non_finite" if detected_by_nonfinite else "magnitude",
                    "ratio_vs_carried": ratio,
                    "required_ratio": PHASE_ABLATION_MIN_RATIO,
                },
            )
            if not detected_by_nonfinite and ratio <= PHASE_ABLATION_MIN_RATIO:
                failures.append(
                    f"ablation {name} changed the output by only {m_bad['max_abs']:.6g} "
                    f"({ratio:.1f}x the correctly-threaded {baseline:.6g}) and stayed "
                    f"finite; the phase test cannot detect a missing carrier"
                )
        assert not failures, "\n".join(failures)
    finally:
        del hf, ref, ref_indexer
        torch.cuda.empty_cache()
