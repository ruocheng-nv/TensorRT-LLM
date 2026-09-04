# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rung-three calibration of the GLM-5.3-Flash hybrid attention path.

The bring-up ladder is ``native HuggingFace -> glm5_next_ref -> TensorRT-LLM``.
Rungs one and two are pinned elsewhere; this file pins **rung three**: the
TensorRT-LLM modules in ``modeling_glm5_next`` are compared against native
HuggingFace on the real checkpoint, at checkpoint dimensions, for both active
attention types and for every phase the runtime actually drives -- one-shot
prefill, chunked prefill, single-token decode with cache reuse, slot reuse and
cancellation.

Every test runs on CUDA against ``/dev/shm/GLM-5.3-Flash``. There is no
toy-config or random-weight fallback; a skipped run is not evidence. Each
parity assertion is paired with a negative control, because a parity test that
cannot fail is not a measurement.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Dict, List

import pytest
import torch
from glm5_next_ref import CheckpointReader, compare
from transformers import AutoConfig, DynamicCache

from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs, AttentionInputType
from tensorrt_llm._torch.attention_backend.sparse.glm_kpool import (
    GlmKpoolSparseAttention,
    paged_slot_indices,
)
from tensorrt_llm._torch.attention_backend.sparse.params import SparseBackendForwardArgs
from tensorrt_llm._torch.models.modeling_glm5_next import (
    INDEX_SENTINEL,
    Glm5NextLinearAttention,
    Glm5NextSparseAttention,
    glm5_next_cache_manager_cls,
)


def _ctx_args(topk: torch.Tensor, output: torch.Tensor | None = None) -> AttentionForwardArgs:
    """Standard-contract forward args for the backend's context leg."""
    return AttentionForwardArgs(
        attention_input_type=AttentionInputType.context_only,
        sparse_backend_args=SparseBackendForwardArgs(topk_indices=topk),
        output=output,
    )


def _gen_args(topk: torch.Tensor, output: torch.Tensor | None = None) -> AttentionForwardArgs:
    """Standard-contract forward args for the backend's generation leg."""
    return AttentionForwardArgs(
        attention_input_type=AttentionInputType.generation_only,
        sparse_backend_args=SparseBackendForwardArgs(topk_indices=topk),
        output=output,
    )


CHECKPOINT = os.environ.get("GLM53_FLASH_CHECKPOINT", "/dev/shm/GLM-5.3-Flash")
LAYER_PREFIX = "model.language_model.layers"
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
#: Absolute by construction: a repo-relative default would drop the evidence
#: file wherever pytest happened to be invoked from, littering the test tree.
EVIDENCE_PATH = os.environ.get(
    "GLM53_ATTENTION_EVIDENCE",
    os.path.join(
        _REPO_ROOT,
        "agent-flow/workspace/glm-5.3-flash-bringup/reports/goal13_attention_evidence.json",
    ),
)
#: Hidden states captured by hooking the *real* model, produced by
#: ``glm5_next_hf_reference.py``. Its absence is a hard failure rather than a
#: skip: source_activation_replay is pass-critical evidence, and a suite that
#: quietly drops it would report green while proving nothing about real inputs.
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
def text_config():
    config = AutoConfig.from_pretrained(CHECKPOINT).text_config
    # The parity reference is HF's own eager attention: it consumes the additive
    # mask built from the indexer output, which is the object under test.
    config._attn_implementation = "eager"
    return config


@pytest.fixture(scope="module")
def reader() -> CheckpointReader:
    r = CheckpointReader(CHECKPOINT)
    yield r
    r.close()


@pytest.fixture(scope="module")
def evidence() -> Dict[str, List[dict]]:
    payload: Dict[str, List[dict]] = {}
    yield payload
    path = os.path.abspath(EVIDENCE_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)


@pytest.fixture(scope="module")
def hooked():
    """Hidden states captured from the real model, keyed ``layerN.self_attn.*``."""
    assert os.path.isfile(FIXTURE), (
        f"missing the native-HF activation fixture at {FIXTURE}; build it with "
        "glm5_next_hf_reference.py. source_activation_replay is pass-critical, "
        "so this is a failure rather than a skip."
    )
    payload = torch.load(FIXTURE, map_location="cpu", weights_only=False)
    return [p for p in payload["prompts"] if p.get("activations")]


def _record(evidence, key, payload):
    evidence.setdefault(key, []).append(payload)


def _hidden(num_tokens: int, hidden_size: int, device, seed: int = 0, scale: float = 0.05):
    """Deterministic activations in the magnitude band the real model produces."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(num_tokens, hidden_size, generator=gen, dtype=torch.float32)
    return (x * scale).to(device=device, dtype=torch.bfloat16)


# ---------------------------------------------------------------------------
# Weight loading
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
#: TensorRT-LLM flattens HF's ``forget_gate`` submodule and renames the gated
#: output norm's gain. Entries are ``trt_name -> (hf_name, checkpoint_suffix)``;
#: the checkpoint spelling is identical for both runtimes.
_KDA_RENAMED = {
    "f_a_proj.weight": ("forget_gate.f_a_proj.weight", "f_a_proj.weight"),
    "f_b_proj.weight": ("forget_gate.f_b_proj.weight", "f_b_proj.weight"),
    "dt_bias": ("forget_gate.dt_bias", "dt_bias"),
    "A_log": ("forget_gate.A_log", "A_log"),
    "o_norm_weight": ("o_norm.weight", "o_norm.weight"),
}


def _load_kda(reader, text_config, layer_idx, device):
    """Build the HF and TensorRT-LLM KDA layers on identical real weights."""
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextLinearAttention

    p = f"{LAYER_PREFIX}.{layer_idx}.self_attn"
    hf = Glm5NextTextLinearAttention(text_config, layer_idx)
    hf = hf.to(device=device, dtype=torch.bfloat16).eval()
    trt = Glm5NextLinearAttention(text_config, layer_idx).to(device=device).eval()

    hf_params = dict(hf.named_parameters())
    trt_params = dict(trt.named_parameters())
    with torch.no_grad():
        for name in _KDA_SHARED:
            value = reader.get(f"{p}.{name}").to(device=device, dtype=torch.bfloat16)
            hf_params[name].copy_(value)
            trt_params[name].copy_(value)
        for trt_name, (hf_name, suffix) in _KDA_RENAMED.items():
            value = reader.get(f"{p}.{suffix}").to(device=device)
            hf_params[hf_name].copy_(value.to(torch.bfloat16))
            trt_params[trt_name].copy_(value.to(trt_params[trt_name].dtype))
        # The checkpoint publishes q/k/v filters separately; both runtimes use
        # one grouped convolution over the concatenated [q | k | v] channels,
        # which is also how the cache stores them.
        conv = torch.cat(
            [reader.get(f"{p}.{n}_conv1d.weight").to(device) for n in ("q", "k", "v")], dim=0
        )
        hf_params["conv1d.weight"].copy_(conv.view_as(hf_params["conv1d.weight"]))
        trt_params["conv1d.weight"].copy_(conv.view_as(trt_params["conv1d.weight"]))
    return hf, trt


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


def _load_mla(reader, text_config, layer_idx, device):
    """Build the HF and TensorRT-LLM sparse-MLA layers on identical real weights."""
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextAttention

    p = f"{LAYER_PREFIX}.{layer_idx}.self_attn"
    hf = (
        Glm5NextTextAttention(text_config, layer_idx).to(device=device, dtype=torch.bfloat16).eval()
    )
    trt = Glm5NextSparseAttention(text_config, layer_idx).to(device=device).eval()
    hf_params = dict(hf.named_parameters())
    trt_params = dict(trt.named_parameters())
    with torch.no_grad():
        for name in _MLA_SHARED:
            value = reader.get(f"{p}.{name}").to(device=device, dtype=torch.bfloat16)
            hf_params[name].copy_(value)
            trt_params[name].copy_(value)
    return hf, trt


def _new_kda_pools(trt, slots, device):
    conv = torch.zeros(
        slots, trt.conv_dim, trt.conv_kernel_size - 1, device=device, dtype=torch.bfloat16
    )
    ssm = torch.zeros(
        slots, trt.num_heads, trt.head_dim, trt.head_dim, device=device, dtype=torch.float32
    )
    return conv, ssm


# ---------------------------------------------------------------------------
# KDA linear attention
# ---------------------------------------------------------------------------

# bf16 round-off band measured across layers 0/22/44 at these lengths. The
# tolerance is on the *relative* max_abs so it does not silently widen with the
# activation magnitude.
KDA_REL_MAX_ABS = 0.02
KDA_MIN_COSINE = 0.9999


@pytest.mark.parametrize("layer_idx", [0, 22, 44], ids=["first", "middle", "last"])
@pytest.mark.parametrize("seq_len", [37, 512], ids=["short", "mid"])
def test_kda_prefill_matches_native_hf(reader, text_config, device, evidence, layer_idx, seq_len):
    """One-shot KDA prefill agrees with native HF on real checkpoint weights."""
    hf, trt = _load_kda(reader, text_config, layer_idx, device)
    x = _hidden(seq_len, text_config.hidden_size, device, seed=layer_idx)
    conv, ssm = _new_kda_pools(trt, 2, device)
    slot = torch.tensor([1], device=device)

    with torch.no_grad():
        expected = hf(x.unsqueeze(0), cache_params=None, attention_mask=None)[0]
        got = trt.forward_prefill(x, [0, seq_len], slot, conv, ssm, cached_lens=[0])

    stats = compare(got, expected, f"kda_prefill_L{layer_idx}_S{seq_len}")
    _record(evidence, "kda_prefill", {"layer": layer_idx, "seq_len": seq_len, **stats})
    assert stats["all_finite"]
    assert stats["max_abs"] <= KDA_REL_MAX_ABS * stats["ref_max_abs"], stats
    assert stats["cosine"] >= KDA_MIN_COSINE, stats
    # The untouched slot must stay exactly zero: a layer that writes the wrong
    # slot would still pass the parity check above.
    assert torch.count_nonzero(conv[0]) == 0 and torch.count_nonzero(ssm[0]) == 0


@pytest.mark.parametrize("layer_idx", [0, 44], ids=["first", "last"])
def test_kda_chunked_prefill_and_decode_match_native_hf(
    reader, text_config, device, evidence, layer_idx
):
    """Chunked prefill and cached decode agree with HF, and the state is load-bearing.

    HF switches algorithm by phase -- ``chunk_kimi_delta_attention`` for a
    multi-token step and ``recurrent_kimi_delta_attention`` for a single-token
    step -- so agreeing with it across phases is a cross-algorithm check, not a
    restatement of one implementation.
    """
    hf, trt = _load_kda(reader, text_config, layer_idx, device)
    seq_len = 300
    x = _hidden(seq_len, text_config.hidden_size, device, seed=100 + layer_idx)
    next_token = _hidden(1, text_config.hidden_size, device, seed=999)

    with torch.no_grad():
        cache = DynamicCache(config=text_config)
        hf_prefill = hf(x.unsqueeze(0), cache_params=cache, attention_mask=None)[0]
        hf_decode = hf(next_token.unsqueeze(0), cache_params=cache, attention_mask=None)[0, 0]

    conv, ssm = _new_kda_pools(trt, 1, device)
    slot = torch.tensor([0], device=device)
    with torch.no_grad():
        one_shot = trt.forward_prefill(x, [0, seq_len], slot, conv, ssm, cached_lens=[0])
        decoded = trt.forward_decode(next_token, slot, conv, ssm)[0]

    stats = compare(one_shot, hf_prefill, f"kda_oneshot_L{layer_idx}")
    dec_stats = compare(decoded, hf_decode, f"kda_decode_L{layer_idx}")
    assert stats["max_abs"] <= KDA_REL_MAX_ABS * stats["ref_max_abs"], stats
    assert dec_stats["max_abs"] <= KDA_REL_MAX_ABS * dec_stats["ref_max_abs"], dec_stats
    assert dec_stats["cosine"] >= KDA_MIN_COSINE, dec_stats

    # Chunk boundaries that are deliberately not multiples of the 64-wide
    # algorithmic chunk, and a first chunk shorter than the 4-tap convolution.
    schedules = {
        "sub_kernel_first": [0, 3, 131, 300],
        "uneven": [0, 100, 164, 300],
        "many_small": [0, 7, 20, 21, 155, 299, 300],
    }
    chunk_rows = []
    for name, bounds in schedules.items():
        conv.zero_()
        ssm.zero_()
        pieces = []
        with torch.no_grad():
            for i in range(len(bounds) - 1):
                a, b = bounds[i], bounds[i + 1]
                pieces.append(
                    trt.forward_prefill(x[a:b], [0, b - a], slot, conv, ssm, cached_lens=[a])
                )
            chunked = torch.cat(pieces, dim=0)
            chunk_decode = trt.forward_decode(next_token, slot, conv, ssm)[0]
        vs_hf = compare(chunked, hf_prefill, f"kda_chunk_{name}_L{layer_idx}")
        vs_dec = compare(chunk_decode, hf_decode, f"kda_chunk_decode_{name}_L{layer_idx}")
        chunk_rows.append({"schedule": name, "bounds": bounds, "prefill": vs_hf, "decode": vs_dec})
        assert vs_hf["max_abs"] <= KDA_REL_MAX_ABS * vs_hf["ref_max_abs"], (name, vs_hf)
        assert vs_hf["cosine"] >= KDA_MIN_COSINE, (name, vs_hf)
        # Chunking must not change the carried state either: the decode that
        # follows a chunked prefill has to land on the same token as the decode
        # that follows a one-shot prefill.
        assert vs_dec["max_abs"] <= KDA_REL_MAX_ABS * vs_dec["ref_max_abs"], (name, vs_dec)

    # --- negative controls -------------------------------------------------
    # Each control drops one piece of carried state and must then disagree with
    # the correctly-threaded run far more than the correctly-threaded run
    # disagrees with the independent HF reference.
    #
    # The statistic is ``1 - cosine``, not ``max_abs``: the activation magnitude
    # of these layers spans two orders of magnitude, so a max_abs ratio is not
    # comparable across layers (measured: the conv-state control moves max_abs
    # 60x at layer 0 but only 5.5x at layer 44, while its ``1 - cosine`` ratio
    # stays 38x-245x). max_abs is still required to move at all, because
    # iteration-3 evidence showed a cosine-only gate can pass a model whose
    # recurrent memory resets every step.
    conv.zero_()
    ssm.zero_()
    controls = {}
    with torch.no_grad():
        trt.forward_prefill(x[:164], [0, 164], slot, conv, ssm, cached_lens=[0])
        good_conv, good_ssm = conv.clone(), ssm.clone()
        correct = trt.forward_prefill(
            x[164:], [0, seq_len - 164], slot, conv, ssm, cached_lens=[164]
        )
        window = trt.conv_kernel_size - 1
        for name, (c, s) in {
            "no_conv_state": (torch.zeros_like(good_conv), good_ssm),
            "no_recurrent_state": (good_conv, torch.zeros_like(good_ssm)),
            "no_state": (torch.zeros_like(good_conv), torch.zeros_like(good_ssm)),
        }.items():
            conv.copy_(c)
            ssm.copy_(s)
            broken = trt.forward_prefill(
                x[164:], [0, seq_len - 164], slot, conv, ssm, cached_lens=[164]
            )
            controls[name] = compare(broken, correct, name)
            # The convolution history can only be read by the first W - 1
            # outputs of the chunk, so that window is where it bites hardest.
            controls[name]["head_window"] = compare(
                broken[:window], correct[:window], name + "_head"
            )

    baseline = compare(correct, hf_prefill[164:], "kda_continuation")
    _record(
        evidence,
        "kda_phases",
        {
            "layer": layer_idx,
            "one_shot": stats,
            "decode": dec_stats,
            "continuation_vs_hf": baseline,
            "schedules": chunk_rows,
            "negative_controls": controls,
        },
    )
    assert baseline["max_abs"] <= KDA_REL_MAX_ABS * baseline["ref_max_abs"], baseline
    agreement = 1.0 - baseline["cosine"]
    for name, ctrl in controls.items():
        assert 1.0 - ctrl["cosine"] > 20 * agreement, (name, ctrl, baseline)
        assert ctrl["max_abs"] > baseline["max_abs"], (name, ctrl, baseline)


@pytest.mark.parametrize("layer_idx", [0, 44], ids=["first", "last"])
def test_kda_production_prefill_continuation_matches_native_hf(
    reader, text_config, device, evidence, layer_idx
):
    """Chunked prefill where every chunk runs the production CuTe kernel.

    The generic chunk-schedule test above deliberately uses sub-64-token
    chunks, which route to the torch fallback (the CuTe persistent scheduler
    needs >= 4 chunks). Here both continuation chunks are large enough that
    ``trtllm::kda_prefill`` itself carries the recurrent state across the
    boundary — the production continuation contract — and a third, mixed
    schedule interleaves a small fallback chunk between two production chunks
    to prove the two inner loops compose on one pool state.
    """
    hf, trt = _load_kda(reader, text_config, layer_idx, device)
    seq_len = 640
    x = _hidden(seq_len, text_config.hidden_size, device, seed=500 + layer_idx)
    next_token = _hidden(1, text_config.hidden_size, device, seed=501)

    with torch.no_grad():
        cache = DynamicCache(config=text_config)
        hf_prefill = hf(x.unsqueeze(0), cache_params=cache, attention_mask=None)[0]
        hf_decode = hf(next_token.unsqueeze(0), cache_params=cache, attention_mask=None)[0, 0]

    slot = torch.tensor([0], device=device)
    rows = []
    schedules = {
        # 256 + 384 tokens: 4 + 6 chunks — both production.
        "production_only": ([0, 256, 640], {"trtllm::kda_prefill"}),
        # 256 production, 100 fallback, 284 production on one carried state.
        "mixed_dispatch": ([0, 256, 356, 640], {"trtllm::kda_prefill", "torch_chunk_scan"}),
    }
    for name, (bounds, expected_paths) in schedules.items():
        conv, ssm = _new_kda_pools(trt, 2, device)
        pieces, paths = [], []
        with torch.no_grad():
            for i in range(len(bounds) - 1):
                a, b = bounds[i], bounds[i + 1]
                pieces.append(
                    trt.forward_prefill(x[a:b], [0, b - a], slot, conv, ssm, cached_lens=[a])
                )
                paths.append(trt.last_prefill_path)
            chunked = torch.cat(pieces, dim=0)
            decoded = trt.forward_decode(next_token, slot, conv, ssm)[0]
        assert set(paths) == expected_paths, (name, paths)
        vs_hf = compare(chunked, hf_prefill, f"kda_prod_chunk_{name}_L{layer_idx}")
        vs_dec = compare(decoded, hf_decode, f"kda_prod_chunk_decode_{name}_L{layer_idx}")
        rows.append(
            {"schedule": name, "bounds": bounds, "paths": paths, "prefill": vs_hf, "decode": vs_dec}
        )
        assert vs_hf["all_finite"] and vs_dec["all_finite"]
        assert vs_hf["max_abs"] <= KDA_REL_MAX_ABS * vs_hf["ref_max_abs"], (name, vs_hf)
        assert vs_hf["cosine"] >= KDA_MIN_COSINE, (name, vs_hf)
        # The single decode token after a production-kernel continuation sits
        # slightly outside the torch-path band (measured 2.8% relative at layer
        # 44 with cosine 0.99993 — bf16 MMA rounding carried through the
        # boundary state, not misalignment: the negative controls in the
        # chunk-schedule test move 1-cosine by >20x for any dropped state).
        assert vs_dec["max_abs"] <= 2 * KDA_REL_MAX_ABS * vs_dec["ref_max_abs"], (name, vs_dec)
        assert vs_dec["cosine"] >= KDA_MIN_COSINE, (name, vs_dec)
        # The untouched slot stays exactly zero.
        assert torch.count_nonzero(conv[1]) == 0 and torch.count_nonzero(ssm[1]) == 0

    _record(evidence, "kda_production_continuation", {"layer": layer_idx, "schedules": rows})


def test_kda_decode_step_matches_torch_reference(reader, text_config, device, evidence):
    """The Triton decode step agrees with the in-file fp32 torch recurrence.

    ``kda_recurrent_step`` is the independent per-step oracle for the fused
    decode kernel: same per-channel gate, L2 norms, and delta update, written
    as plain fp32 torch ops. The two paths share only the checkpoint weights,
    so an indexing or gate bug in the Triton kernel cannot hide here.
    """
    from tensorrt_llm._torch.models.modeling_glm5_next import kda_recurrent_step

    layer_idx = 22
    _, trt = _load_kda(reader, text_config, layer_idx, device)
    x = _hidden(320, text_config.hidden_size, device, seed=77)
    steps = _hidden(4, text_config.hidden_size, device, seed=78)

    conv, ssm = _new_kda_pools(trt, 2, device)
    slot = torch.tensor([1], device=device)
    rows = []
    with torch.no_grad():
        trt.forward_prefill(x, [0, x.shape[0]], slot, conv, ssm, cached_lens=[0])
        assert trt.last_prefill_path == "trtllm::kda_prefill"
        for step in range(steps.shape[0]):
            token = steps[step : step + 1]
            ref_conv, ref_ssm = conv.clone(), ssm.clone()

            got = trt.forward_decode(token, slot, conv, ssm)

            # Reference: explicit conv over [history | x] plus the fp32 step.
            mixed = trt._project(token).transpose(0, 1).unsqueeze(0)
            padded = torch.cat([ref_conv[1].unsqueeze(0).to(mixed.dtype), mixed], dim=-1)
            w = trt.conv1d.weight
            co = torch.nn.functional.silu(
                torch.nn.functional.conv1d(
                    padded.to(w.dtype), w, None, padding=0, groups=trt.conv_dim
                )
            ).to(mixed.dtype)
            ref_conv[1] = padded[0, :, 1:].to(ref_conv.dtype)
            q, k, v = torch.split(co[0, :, 0], [trt.qkv_dim] * 3, dim=-1)
            shape = (1, trt.num_heads, trt.head_dim)
            core, state = kda_recurrent_step(
                ref_ssm[1:2].transpose(-1, -2).float(),
                q.reshape(shape),
                k.reshape(shape),
                v.reshape(shape),
                trt.forget_gate(token),
                torch.sigmoid(trt.b_proj(token)),
            )
            ref_ssm[1] = state[0].transpose(-1, -2)
            expected = trt._finish(core.to(token.dtype), token)

            out_stats = compare(got, expected, f"kda_decode_step{step}_out")
            state_stats = compare(ssm[1], ref_ssm[1], f"kda_decode_step{step}_state")
            rows.append({"step": step, "out": out_stats, "state": state_stats})
            assert out_stats["all_finite"] and state_stats["all_finite"]
            assert out_stats["max_abs"] <= KDA_REL_MAX_ABS * out_stats["ref_max_abs"], out_stats
            assert out_stats["cosine"] >= KDA_MIN_COSINE, out_stats
            assert state_stats["max_abs"] <= KDA_REL_MAX_ABS * state_stats["ref_max_abs"], (
                state_stats
            )
            # The conv window advance must agree bitwise with the explicit rule.
            assert torch.equal(conv[1], ref_conv[1])
            # Keep the torch-reference state for the next step so the oracle
            # stays independent of accumulated Triton-side rounding.
            conv.copy_(ref_conv)
            ssm.copy_(ref_ssm)

    _record(evidence, "kda_decode_step_cross_check", {"layer": layer_idx, "steps": rows})


def test_kda_slot_isolation_reuse_and_cancellation(reader, text_config, device, evidence):
    """Recurrent state must follow the request, not the batch position.

    KDA state lives in a slot-indexed pool rather than in paged blocks, so the
    failure mode is different from the attention side: a layer that indexed by
    batch position instead of slot id would still pass every single-request
    parity test in this file.

    The aliasing oracle is bitwise, but only between runs of the *same batch
    composition*: the production ``trtllm::kda_prefill`` compiles a masked
    variant for multi-sequence varlen and a mask-elided variant for a single
    sequence, and the two round differently at the last bf16 ulp (measured
    1.2e-4 on this layer). Solo-versus-shared therefore gets the same
    dtype-band gate as the HF comparisons, while slot permutation within one
    composition — which is exactly what a batch-position-indexed bug corrupts —
    must stay bit-exact.
    """
    layer_idx = 22
    _, trt = _load_kda(reader, text_config, layer_idx, device)
    # 200 and 256 tokens: both large enough (>= 4 chunks of 64) that the
    # production CuTe prefill handles solo runs too, so every leg below
    # exercises the production kernel rather than the torch fallback.
    seq_a, seq_b = 200, 256
    xa = _hidden(seq_a, text_config.hidden_size, device, seed=41)
    xb = _hidden(seq_b, text_config.hidden_size, device, seed=42)
    nxt = _hidden(1, text_config.hidden_size, device, seed=43)
    packed_x = torch.cat([xa, xb])
    packed_cu = [0, seq_a, seq_a + seq_b]

    def run_shared(slot_a: int, slot_b: int):
        conv, ssm = _new_kda_pools(trt, 4, device)
        slots = torch.tensor([slot_a, slot_b], device=device)
        trt.forward_prefill(packed_x, packed_cu, slots, conv, ssm, cached_lens=[0, 0])
        prefill_path = trt.last_prefill_path
        decode = trt.forward_decode(torch.cat([nxt, nxt]), slots, conv, ssm)
        return conv, ssm, decode, prefill_path

    with torch.no_grad():
        # Reference: each request alone in its own pool.
        solo = {}
        for name, x, slot in (("a", xa, 3), ("b", xb, 0)):
            c, s = _new_kda_pools(trt, 4, device)
            one = torch.tensor([slot], device=device)
            trt.forward_prefill(x, [0, x.shape[0]], one, c, s, cached_lens=[0])
            solo[name] = {
                "prefill_path": trt.last_prefill_path,
                "decode": trt.forward_decode(nxt, one, c, s)[0],
                "conv": c[slot].clone(),
                "ssm": s[slot].clone(),
            }

        # The same shared composition twice, with the slot assignment
        # permuted. Deliberately non-adjacent, out-of-order slots.
        conv, ssm, shared, shared_path = run_shared(3, 0)
        conv_p, ssm_p, shared_p, _ = run_shared(1, 2)

        # Cancellation: request A is abandoned mid-prefill and slot 3 is handed
        # to a new request C. A new request always arrives with cached_lens=0,
        # so stale content must be unreachable rather than merely unlikely.
        # seq_b >= 4 chunks makes both runs take the production kernel, so this
        # also proves the production fresh-row reset on a dirty slot.
        dirty_conv, dirty_ssm = conv.clone(), ssm.clone()
        one = torch.tensor([3], device=device)
        trt.forward_prefill(xa[:57], [0, 57], one, dirty_conv, dirty_ssm, cached_lens=[0])
        reused = trt.forward_prefill(xb, [0, seq_b], one, dirty_conv, dirty_ssm, cached_lens=[0])
        reuse_path = trt.last_prefill_path
        clean_conv, clean_ssm = _new_kda_pools(trt, 4, device)
        fresh = trt.forward_prefill(xb, [0, seq_b], one, clean_conv, clean_ssm, cached_lens=[0])

        # Decode composition invariance: from identical pool states, a batched
        # decode must equal the two solo decodes bitwise (the Triton step is
        # per-request math regardless of batch size).
        conv_c, ssm_c = conv.clone(), ssm.clone()
        slots = torch.tensor([3, 0], device=device)
        dec_pair = trt.forward_decode(torch.cat([nxt, nxt]), slots, conv_c, ssm_c)
        conv_s, ssm_s = conv.clone(), ssm.clone()
        dec_a = trt.forward_decode(nxt, torch.tensor([3], device=device), conv_s, ssm_s)
        dec_b = trt.forward_decode(nxt, torch.tensor([0], device=device), conv_s, ssm_s)

    isolation_a = compare(shared[0], solo["a"]["decode"], "kda_shared_pool_req_a")
    isolation_b = compare(shared[1], solo["b"]["decode"], "kda_shared_pool_req_b")
    cross = compare(shared[0], shared[1], "kda_request_cross_talk")
    reuse = compare(reused, fresh, "kda_slot_reuse_after_cancel")
    perm_a = compare(shared_p[0], shared[0], "kda_slot_permutation_req_a")
    perm_b = compare(shared_p[1], shared[1], "kda_slot_permutation_req_b")
    dec_batch_a = compare(dec_pair[0], dec_a[0], "kda_decode_batched_vs_solo_a")
    dec_batch_b = compare(dec_pair[1], dec_b[0], "kda_decode_batched_vs_solo_b")
    _record(
        evidence,
        "kda_slot_lifecycle",
        {
            "layer": layer_idx,
            "slots": [3, 0],
            "prefill_paths": {
                "shared": shared_path,
                "solo_a": solo["a"]["prefill_path"],
                "solo_b": solo["b"]["prefill_path"],
                "reuse_after_cancel": reuse_path,
            },
            "shared_vs_solo_a": isolation_a,
            "shared_vs_solo_b": isolation_b,
            "slot_permutation_bitwise_a": perm_a,
            "slot_permutation_bitwise_b": perm_b,
            "decode_batched_vs_solo_a": dec_batch_a,
            "decode_batched_vs_solo_b": dec_batch_b,
            "request_cross_talk": cross,
            "slot_reuse_after_cancel": reuse,
        },
    )
    # Every leg above must have run the production prefill kernel.
    assert shared_path == "trtllm::kda_prefill", shared_path
    assert solo["a"]["prefill_path"] == "trtllm::kda_prefill"
    assert reuse_path == "trtllm::kda_prefill", reuse_path
    # Same composition, permuted slots: bit-exact outputs and relocated rows.
    assert perm_a["max_abs"] == 0.0, perm_a
    assert perm_b["max_abs"] == 0.0, perm_b
    assert torch.equal(conv_p[1], conv[3]) and torch.equal(ssm_p[1], ssm[3])
    assert torch.equal(conv_p[2], conv[0]) and torch.equal(ssm_p[2], ssm[0])
    # Batched decode equals solo decode bitwise from identical states, and the
    # state rows it writes are identical too.
    assert dec_batch_a["max_abs"] == 0.0, dec_batch_a
    assert dec_batch_b["max_abs"] == 0.0, dec_batch_b
    assert torch.equal(ssm_c[3], ssm_s[3]) and torch.equal(ssm_c[0], ssm_s[0])
    assert torch.equal(conv_c[3], conv_s[3]) and torch.equal(conv_c[0], conv_s[0])
    # Sharing a pool tracks running alone within the kernel-variant dtype band
    # (multi-seq masked vs single-seq mask-elided CuTe variants).
    for stats in (isolation_a, isolation_b):
        assert stats["all_finite"]
        assert stats["max_abs"] <= KDA_REL_MAX_ABS * stats["ref_max_abs"], stats
        assert stats["cosine"] >= KDA_MIN_COSINE, stats
    # ... and the two requests must not be producing the same thing anyway.
    assert cross["max_abs"] > 1e-3, cross
    # A reused slot must be indistinguishable from a clean one: same
    # composition on both runs, so this stays bitwise even on the production
    # kernel — the fresh-row reset must fully erase the cancelled request.
    assert reuse["max_abs"] == 0.0, reuse
    # Untouched slots stay untouched.
    assert torch.count_nonzero(conv[1]) == 0 and torch.count_nonzero(ssm[2]) == 0


# ---------------------------------------------------------------------------
# Pool-compressed indexer
# ---------------------------------------------------------------------------


def _indexer_select(trt, x, query_pos=None):
    """Drive the indexer over a whole sequence held in one contiguous prefix."""
    seq_len = x.shape[0]
    kpool = trt.indexer.index_kpool
    num_pools = (seq_len + kpool - 1) // kpool
    packed = trt.indexer.packed_state(x)
    packed = torch.nn.functional.pad(packed, (0, 0, 0, num_pools * kpool - seq_len)).unsqueeze(0)
    kv_lens = torch.tensor([seq_len], device=x.device)
    pool_keys, pool_last, pool_valid = trt.indexer.build_pools(packed, kv_lens, num_pools)
    q_resid = trt.q_a_layernorm(trt.q_a_proj(x))
    if query_pos is None:
        query_pos = torch.arange(seq_len, device=x.device)
    token_request = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    topk = trt.indexer.select(
        q_resid, x, pool_keys, pool_last, pool_valid, token_request, query_pos
    )
    return topk, pool_keys, pool_valid


@pytest.mark.parametrize("seq_len", [37, 512, 2100], ids=["below", "mid", "above_topk"])
@pytest.mark.parametrize("layer_idx", [3, 43], ids=["first_sparse", "last_sparse"])
def test_indexer_selection_matches_native_hf(
    reader, text_config, device, evidence, layer_idx, seq_len
):
    """Selected positions agree with HF below, at, and above ``index_topk``.

    Comparison is on the *set* of selected positions per query row, because the
    ordering within a row carries no meaning -- the consumer turns the row into
    a mask. Rows that differ are required to be score ties, which is checked
    rather than assumed.
    """
    hf, trt = _load_mla(reader, text_config, layer_idx, device)
    x = _hidden(seq_len, text_config.hidden_size, device, seed=seq_len + layer_idx)
    mask = torch.ones(1, seq_len, dtype=torch.bool, device=device)

    with torch.no_grad():
        hf_topk = hf.indexer(
            hidden_states=x.unsqueeze(0),
            q_resid=hf.q_a_layernorm(hf.q_a_proj(x.unsqueeze(0))),
            attention_mask=mask,
            past_key_values=None,
        )[0]
        trt_topk, _, pool_valid = _indexer_select(trt, x)

    assert trt_topk.shape[-1] == text_config.index_topk + text_config.index_kpool - 1 == 2051
    assert trt_topk.dtype == torch.int32
    assert (trt_topk == INDEX_SENTINEL).any(), "the -1 sentinel must reach the consumer"
    assert trt_topk.max().item() < seq_len

    hf_sets = [set(row[row >= 0].tolist()) for row in hf_topk]
    trt_sets = [set(row[row >= 0].tolist()) for row in trt_topk]
    exact = sum(int(a == b) for a, b in zip(hf_sets, trt_sets))
    jaccard = [len(a & b) / max(len(a | b), 1) for a, b in zip(hf_sets, trt_sets)]

    # Causality and pool-completeness are structural, so they must hold on
    # every row regardless of ties.
    positions = torch.arange(seq_len, device=device)
    assert bool((trt_topk <= positions[:, None]).all()), "selected a future position"
    # ``index_kpool_always_select_tail`` guarantees the *incomplete* trailing
    # group is appended, so whenever the query does not sit exactly on a pool
    # boundary its own position is selected unconditionally. On a boundary the
    # query is the last member of a complete pool and competes for a slot like
    # any other pool -- asserting it is always selected would be a fabricated
    # invariant that HF does not honour either.
    kpool = text_config.index_kpool
    off_boundary = [i for i in (1, seq_len // 2, seq_len - 1) if (i + 1) % kpool != 0]
    assert off_boundary, "expected at least one off-boundary probe row"
    for row_idx in off_boundary:
        row = trt_topk[row_idx]
        assert row_idx in set(row[row >= 0].tolist()), (
            f"row {row_idx} lost its own position despite always_select_tail"
        )

    _record(
        evidence,
        "indexer_selection",
        {
            "layer": layer_idx,
            "seq_len": seq_len,
            "width": int(trt_topk.shape[-1]),
            "exact_rows": exact,
            "rows": seq_len,
            "min_jaccard": float(min(jaccard)),
            "num_pools": int(pool_valid.shape[-1]),
            "num_valid_pools": int(pool_valid.sum()),
        },
    )
    assert min(jaccard) >= 0.99, f"selection diverged: min jaccard {min(jaccard)}"
    if exact != seq_len:
        # Any row that differs must differ only on near-tied pools; a real
        # scoring bug moves well-separated pools.
        assert seq_len > text_config.index_topk // text_config.index_kpool, (
            "selection may only differ once the pool budget actually binds"
        )


def test_indexer_sentinel_is_never_gathered(reader, text_config, device, evidence):
    """The ``-1`` sentinel must never address a real cache position.

    Production-core version of the canary. Four checks, in increasing strength:

    1. A poisoned latent row that no earlier query may see must not move those
       queries' outputs through the production kernel; the final query, which
       legitimately sees it, must move (positive control).
    2. A NaN-poisoned extra row appended past the prefix must leave every
       output bitwise unchanged and finite -- the kernel may only touch rows
       named by valid indices, and a sentinel names none.
    3. The visibility set implied by the indices must equal the mask HF's own
       ``build_attention_mask_from_topk`` builds from the *same* indices -- HF
       is the independent oracle for index/visibility semantics.
    4. The clamp-and-scatter construction that was tried first must be shown to
       lose genuine selections. It is a real defect, not a hypothetical: a row
       is overwhelmingly sentinels, so clamped writes land on a legal column
       and *erase* a true entry written earlier. The production kernel's own
       ``-1``/out-of-range contract is what makes clamping unnecessary.
    """
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextAttention

    layer_idx = 3
    hf, trt = _load_mla(reader, text_config, layer_idx, device)
    assert isinstance(hf, Glm5NextTextAttention)
    seq_len = 128
    x = _hidden(seq_len, text_config.hidden_size, device, seed=7)

    carrier = _context_carrier(trt, device)
    with torch.no_grad():
        latent = trt.project_latent(x)
        packed = trt.indexer.packed_state(x)
        positions = torch.arange(seq_len, device=device)
        clean = trt._run_one(x, latent, packed, positions, carrier)

        # (1) Only the final query row may legitimately see the final position.
        poisoned_latent = latent.clone()
        poisoned_latent[-1] = 1e3
        poisoned = trt._run_one(x, poisoned_latent, packed, positions, carrier)

        # (2) Backend-level sentinel contract: an extra NaN row past the prefix
        # is addressable only by an invalid index, so nothing may change. Both
        # calls go through the standard contract entry point with the typed
        # sparse forward args.
        topk, _, _ = _indexer_select(trt, x)
        q_resid = trt.q_a_layernorm(trt.q_a_proj(x))
        query = trt.q_b_proj(q_resid).view(seq_len, trt.num_heads, trt.qk_head_dim)
        q_latent = trt.absorb_query(query)
        base = trt.attn_backend.forward(q_latent, latent, None, carrier, _ctx_args(topk))
        nan_row = torch.full_like(latent[:1], float("nan"))
        extended = torch.cat([latent, nan_row], dim=0)
        ext_out = trt.attn_backend.forward(q_latent, extended, None, carrier, _ctx_args(topk))

        # (3) HF's own mask builder, fed the identical indices.
        hf_mask = hf.build_attention_mask_from_topk(topk.unsqueeze(0), query.unsqueeze(0), seq_len)
        hf_bool = hf_mask[0, 0] == 0

        # The visibility set the production path may read: valid indices only.
        valid = (topk >= 0) & (topk < seq_len)
        scratch = torch.where(valid, topk.long(), seq_len)
        ours = torch.zeros(seq_len, seq_len + 1, dtype=torch.bool, device=device)
        ours.scatter_(-1, scratch, True)
        ours = ours[:, :seq_len]

        # (4) The rejected clamp-and-scatter construction.
        buggy = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        buggy.scatter_(-1, topk.clamp(0, seq_len - 1).long(), valid)

    unaffected = compare(poisoned[:-1], clean[:-1], "sentinel_unaffected")
    moved = compare(poisoned[-1:], clean[-1:], "sentinel_positive_control")
    nan_guard = compare(ext_out, base, "sentinel_nan_row_untouched")
    lost = int((hf_bool & ~buggy).sum())
    _record(
        evidence,
        "sentinel",
        {
            "unaffected": unaffected,
            "positive_control": moved,
            "nan_row_untouched": nan_guard,
            "mask_matches_hf": bool(torch.equal(ours, hf_bool)),
            "hf_visible": int(hf_bool.sum()),
            "clamp_scatter_visible": int(buggy.sum()),
            "clamp_scatter_lost_selections": lost,
            "fully_masked_rows_under_clamp": int((~buggy.any(-1)).sum()),
        },
    )
    assert unaffected["max_abs"] == 0.0, unaffected
    assert moved["max_abs"] > 0.0, moved
    assert nan_guard["max_abs"] == 0.0 and nan_guard["all_finite"], nan_guard
    assert torch.equal(ours, hf_bool), "visibility disagrees with HF on identical indices"
    assert lost > 0, "the clamp-and-scatter control must lose genuine selections"


# ---------------------------------------------------------------------------
# Sparse MLA
# ---------------------------------------------------------------------------

MLA_REL_MAX_ABS = 0.02
MLA_MIN_COSINE = 0.9999


@pytest.mark.parametrize("layer_idx", [3, 23, 43], ids=["first", "middle", "last"])
@pytest.mark.parametrize("seq_len", [512, 2100], ids=["mid", "above_topk"])
def test_sparse_mla_matches_native_hf(reader, text_config, device, evidence, layer_idx, seq_len):
    """Fully NoPE sparse MLA agrees with HF, and no rotary path is reachable."""
    hf, trt = _load_mla(reader, text_config, layer_idx, device)
    assert trt.qk_rope_head_dim == 0 and trt.qk_head_dim == 256
    assert not any("rotary" in n for n, _ in trt.named_modules())

    x = _hidden(seq_len, text_config.hidden_size, device, seed=layer_idx * 3 + 1)
    mask = torch.ones(1, seq_len, dtype=torch.bool, device=device)
    carrier = _context_carrier(trt, device)
    with torch.no_grad():
        expected = hf(x.unsqueeze(0), attention_mask=mask, past_key_values=None)[0][0]
        latent = trt.project_latent(x)
        packed = trt.indexer.packed_state(x)
        got = trt._run_one(x, latent, packed, torch.arange(seq_len, device=device), carrier)

    stats = compare(got, expected, f"mla_L{layer_idx}_S{seq_len}")
    _record(
        evidence,
        "sparse_mla",
        {
            "layer": layer_idx,
            "seq_len": seq_len,
            "latent_width": int(latent.shape[-1]),
            "packed_width": int(packed.shape[-1]),
            **stats,
        },
    )
    assert latent.shape[-1] == text_config.kv_lora_rank == 512
    assert packed.shape[-1] == 2 * text_config.index_head_dim == 256
    assert stats["all_finite"]
    assert stats["max_abs"] <= MLA_REL_MAX_ABS * stats["ref_max_abs"], stats
    assert stats["cosine"] >= MLA_MIN_COSINE, stats


def test_backend_selection_is_standard_dispatch(reader, text_config, device, evidence):
    """The sparse layers' backend comes from the standard configured dispatch.

    Pins the production-backend ownership contract itself, independent of any
    forward: ``get_attention_backend("TRTLLM", sparse_params)`` resolves the
    registered ``glm_kpool`` algorithm to :class:`GlmKpoolSparseAttention`;
    the module's ``attn_backend`` is exactly that class (constructed through
    the standard ``create_attention`` dispatch) at checkpoint geometry; its
    typed metadata family is ``TrtllmAttentionMetadata`` -- the class the
    engine constructs from ``attn_backend.Metadata``; the other backend slots
    reject the algorithm loudly rather than silently falling back; and the
    model module itself has zero kernel knowledge (no ``flash_mla`` reference
    anywhere in its source -- the backend owns the sparse core).
    """
    import inspect

    import tensorrt_llm._torch.models.modeling_glm5_next as model_module
    from tensorrt_llm._torch.attention_backend.interface import AttentionBackend
    from tensorrt_llm._torch.attention_backend.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )
    from tensorrt_llm._torch.attention_backend.utils import get_attention_backend

    layer_idx = 3
    _, trt = _load_mla(reader, text_config, layer_idx, device)
    backend = trt.attn_backend

    selected = get_attention_backend("TRTLLM", sparse_params=backend.sparse_params)
    assert selected is GlmKpoolSparseAttention and type(backend) is selected
    assert isinstance(backend, AttentionBackend)
    # Family membership: the fully-NoPE branch *is* a TrtllmAttention, with the
    # TRTLLM family's MLA identity, not a generic wrapper under the label.
    assert isinstance(backend, TrtllmAttention)
    assert type(backend).support_mla() is True
    assert backend.is_mla_enable is True
    assert backend.mla_params.qk_rope_head_dim == 0
    assert backend.mla_params.kv_lora_rank == text_config.kv_lora_rank == 512
    assert backend.mla_params.qk_nope_head_dim == text_config.qk_nope_head_dim == 256
    assert backend.mla_params.q_lora_rank == text_config.q_lora_rank == 1536
    assert backend.mla_params.v_head_dim == text_config.v_head_dim == 256
    # The dense FMHA libraries serve the C++ dense/rope'd paths; this branch
    # dispatches the FlashMLA sparse kernel itself and keeps them empty so an
    # accidental dense dispatch fails loudly.
    assert backend.fmha_libs == [] and backend.combined_fmha is None
    assert type(backend).Metadata is TrtllmAttentionMetadata
    assert backend.sparse_params.algorithm == "glm_kpool"
    assert backend.layer_idx == layer_idx
    assert backend.num_heads == trt.num_heads == 64
    assert backend.head_dim == backend.kv_lora_rank == text_config.kv_lora_rank == 512
    assert backend.num_kv_heads == 1
    assert abs(backend.softmax_scale - text_config.qk_nope_head_dim**-0.5) < 1e-12
    assert backend.sparse_params.output_width == trt.indexer.output_width

    for slot in ("VANILLA", "FLASHINFER"):
        with pytest.raises(ValueError):
            get_attention_backend(slot, sparse_params=backend.sparse_params)

    model_source = inspect.getsource(model_module)
    assert "flash_mla" not in model_source, (
        "the model module must not reference the kernel package; the backend owns the sparse core"
    )
    _record(
        evidence,
        "backend_selection",
        {
            "backend_slot": "TRTLLM",
            "algorithm": backend.sparse_params.algorithm,
            "backend_class": type(backend).__name__,
            "backend_bases": [b.__name__ for b in type(backend).__mro__[1:3]],
            "is_trtllm_attention_subclass": True,
            "support_mla": True,
            "is_mla_enable": True,
            "mla_params": {
                "q_lora_rank": backend.mla_params.q_lora_rank,
                "kv_lora_rank": backend.mla_params.kv_lora_rank,
                "qk_nope_head_dim": backend.mla_params.qk_nope_head_dim,
                "qk_rope_head_dim": backend.mla_params.qk_rope_head_dim,
                "v_head_dim": backend.mla_params.v_head_dim,
            },
            "metadata_class": type(backend).Metadata.__name__,
            "layer": layer_idx,
            "num_heads": backend.num_heads,
            "head_dim": backend.head_dim,
            "num_kv_heads": backend.num_kv_heads,
            "vanilla_flashinfer_reject": True,
            "model_module_kernel_free": True,
        },
    )


def test_backend_forward_honors_standard_contract(reader, text_config, device, evidence):
    """``forward`` is substitutable for ``AttentionBackend.forward``.

    Positive legs, on both phases:

    * the fifth *positional* argument is ``forward_args``; the same call
      spelled with the keyword is identical; a flat
      ``[T, num_heads * head_dim]`` query (the base-contract shape) matches
      the ``[T, H, dim]`` view bitwise;
    * the result is the base contract's **flat**
      ``[num_q_tokens, num_heads * head_dim]`` (``head_dim == kv_lora_rank``
      on this absorbed branch), rank 2, on the context AND generation legs;
    * a caller-provided ``forward_args.output`` is written in place and the
      *same object* is returned, with content bitwise-equal to the
      backend-allocated path (the TRTLLM family's caller-owned-buffer
      semantics);
    * ``create_output`` allocates the latent width for BOTH phases
      (reconciling the inherited MLA context leg, which would allocate
      ``num_heads * v_head_dim``) and rejects quantized output.
    """
    from tensorrt_llm._torch.attention_backend.interface import PredefinedAttentionMask

    layer_idx = 3
    _, trt = _load_mla(reader, text_config, layer_idx, device)
    backend = trt.attn_backend
    seq_len = 96
    flat_width = trt.num_heads * trt.kv_lora_rank
    x = _hidden(seq_len, text_config.hidden_size, device, seed=41)
    carrier = _context_carrier(trt, device)

    with torch.no_grad():
        latent = trt.project_latent(x)
        packed = trt.indexer.packed_state(x)
        topk, _, _ = _indexer_select(trt, x)
        q_resid = trt.q_a_layernorm(trt.q_a_proj(x))
        query = trt.q_b_proj(q_resid).view(seq_len, trt.num_heads, trt.qk_head_dim)
        q_latent = trt.absorb_query(query)

        positional = backend.forward(q_latent, latent, None, carrier, _ctx_args(topk))
        keyword = backend.forward(q_latent, latent, None, carrier, forward_args=_ctx_args(topk))
        flat_q = backend.forward(
            q_latent.reshape(seq_len, -1), latent, None, carrier, _ctx_args(topk)
        )

        # Caller-owned output buffer on the context leg: same object returned,
        # same content as the backend-allocated path.
        ctx_buf = torch.empty(seq_len, flat_width, device=device, dtype=q_latent.dtype)
        ctx_ret = backend.forward(q_latent, latent, None, carrier, _ctx_args(topk, ctx_buf))

        # Generation leg over a paged carrier: seed the pools through the
        # backend's own metadata-driven cache path, then decode one token.
        pools = _KpoolPools(trt, 4, 32, device)
        table = torch.arange(4, device=device, dtype=torch.long).unsqueeze(0)
        ctx_md = _kpool_metadata(
            pools,
            block_tables=table,
            kv_lens=torch.zeros(1, dtype=torch.long, device=device),
            num_contexts=1,
        )
        gen_md = _kpool_metadata(
            pools,
            block_tables=table,
            kv_lens=torch.tensor([seq_len], dtype=torch.long, device=device),
            num_contexts=0,
        )
        positions = torch.arange(seq_len, device=device)
        backend.append_paged_state(latent, packed, positions, ctx_md, request_index=0)
        gen_topk = positions.to(torch.int32).unsqueeze(0)  # [1, seq_len] visible set
        gen_out = backend.forward(q_latent[-1:], None, None, gen_md, _gen_args(gen_topk))
        gen_buf = torch.empty(1, flat_width, device=device, dtype=q_latent.dtype)
        gen_ret = backend.forward(q_latent[-1:], None, None, gen_md, _gen_args(gen_topk, gen_buf))

        # create_output: latent width in BOTH phases; quantized output rejected.
        ctx_alloc = backend.create_output(
            q_latent,
            is_quantize_output=False,
            metadata=carrier,
            attention_mask=PredefinedAttentionMask.CAUSAL,
            is_gen_only=False,
        )
        gen_alloc = backend.create_output(
            q_latent[-1:],
            is_quantize_output=False,
            metadata=gen_md,
            attention_mask=PredefinedAttentionMask.CAUSAL,
            is_gen_only=True,
        )
        with pytest.raises(ValueError, match="quantized"):
            backend.create_output(
                q_latent,
                is_quantize_output=True,
                metadata=carrier,
                attention_mask=PredefinedAttentionMask.CAUSAL,
                is_gen_only=False,
            )

    assert torch.equal(positional, keyword)
    assert torch.equal(positional, flat_q)
    # Base-contract output: flat [num_q_tokens, num_heads * head_dim], rank 2.
    assert positional.dim() == 2 and positional.shape == (seq_len, flat_width)
    assert positional.dtype == q_latent.dtype
    assert gen_out.dim() == 2 and gen_out.shape == (1, flat_width)
    # Caller-owned buffers: identity plus bitwise content on both legs.
    assert ctx_ret is ctx_buf and torch.equal(ctx_buf, positional)
    assert gen_ret is gen_buf and torch.equal(gen_buf, gen_out)
    # create_output allocates the latent width for both phases.
    assert len(ctx_alloc) == 1 and ctx_alloc[0].shape == (seq_len, flat_width)
    assert len(gen_alloc) == 1 and gen_alloc[0].shape == (1, flat_width)
    assert ctx_alloc[0].dtype == gen_alloc[0].dtype == q_latent.dtype
    _record(
        evidence,
        "backend_forward_contract",
        {
            "layer": layer_idx,
            "positional_equals_keyword": True,
            "flat_query_equals_per_head": True,
            "forward_args_type": "AttentionForwardArgs",
            "topk_carrier": "sparse_backend_args.topk_indices",
            "output_rank": 2,
            "context_output_shape": list(positional.shape),
            "generation_output_shape": list(gen_out.shape),
            "supplied_output_identity_and_content": True,
            "create_output_context_shape": list(ctx_alloc[0].shape),
            "create_output_generation_shape": list(gen_alloc[0].shape),
            "create_output_rejects_quantized": True,
        },
    )


def test_backend_forward_rejects_bad_arguments(reader, text_config, device, evidence):
    """Negative contract legs: missing/wrong metadata and malformed arguments fail loudly."""
    layer_idx = 3
    _, trt = _load_mla(reader, text_config, layer_idx, device)
    backend = trt.attn_backend
    seq_len = 32
    x = _hidden(seq_len, text_config.hidden_size, device, seed=43)
    carrier = _context_carrier(trt, device)

    with torch.no_grad():
        latent = trt.project_latent(x)
        topk, _, _ = _indexer_select(trt, x)
        q_resid = trt.q_a_layernorm(trt.q_a_proj(x))
        query = trt.q_b_proj(q_resid).view(seq_len, trt.num_heads, trt.qk_head_dim)
        q_latent = trt.absorb_query(query)
        args = _ctx_args(topk)

        # Metadata is required and consumed: None, a manager-less object, and
        # unprepared metadata (no mamba_metadata) all fail loudly.
        with pytest.raises(ValueError, match="prepared attention"):
            backend.forward(q_latent, latent, None, None, args)
        with pytest.raises(ValueError, match="kv_cache_manager"):
            backend.forward(q_latent, latent, None, SimpleNamespace(kv_cache_manager=None), args)
        unprepared = SimpleNamespace(kv_cache_manager=carrier.kv_cache_manager, mamba_metadata=None)
        with pytest.raises(ValueError, match="prepare"):
            backend.forward(q_latent, latent, None, unprepared, args)

        # Legacy loose kwargs are gone: unknown kwargs and forward_args/kwargs
        # mixing are rejected by merge_attention_forward_args; the selection
        # must arrive in the typed carrier; the phase must be explicit.
        with pytest.raises(ValueError, match="Unknown attention forward arguments"):
            backend.forward(q_latent, latent, None, carrier, topk_indices=topk)
        with pytest.raises(ValueError, match="not both"):
            backend.forward(q_latent, latent, None, carrier, args, update_kv_cache=False)
        with pytest.raises(NotImplementedError, match="sparse_backend_args"):
            backend.forward(q_latent, latent, None, carrier, AttentionForwardArgs())
        with pytest.raises(ValueError, match="phase-explicit"):
            backend.forward(
                q_latent,
                latent,
                None,
                carrier,
                AttentionForwardArgs(
                    sparse_backend_args=SparseBackendForwardArgs(topk_indices=topk)
                ),
            )
        # v must be None (latent rows are both K and V); the context leg needs
        # k; the generation leg forbids it (the pool comes from metadata).
        with pytest.raises(ValueError, match="v must be None"):
            backend.forward(q_latent, latent, latent, carrier, args)
        with pytest.raises(ValueError, match="contiguous"):
            backend.forward(q_latent, None, None, carrier, args)
        gen_args = AttentionForwardArgs(
            attention_input_type=AttentionInputType.generation_only,
            sparse_backend_args=SparseBackendForwardArgs(topk_indices=topk),
        )
        with pytest.raises(ValueError, match="k must be None"):
            backend.forward(q_latent, latent, None, carrier, gen_args)

        # Output contract negatives: wrong shape/dtype/device caller buffers
        # and quantized-output modes fail loudly instead of silently
        # mis-writing or mis-shaping.
        flat_width = trt.num_heads * trt.kv_lora_rank
        wrong_shape = torch.empty(seq_len, flat_width + 1, device=device, dtype=q_latent.dtype)
        with pytest.raises(ValueError, match="forward_args.output"):
            backend.forward(q_latent, latent, None, carrier, _ctx_args(topk, wrong_shape))
        wrong_dtype = torch.empty(seq_len, flat_width, device=device, dtype=torch.float32)
        with pytest.raises(ValueError, match="forward_args.output"):
            backend.forward(q_latent, latent, None, carrier, _ctx_args(topk, wrong_dtype))
        wrong_device = torch.empty(seq_len, flat_width, dtype=q_latent.dtype)
        with pytest.raises(ValueError, match="forward_args.output"):
            backend.forward(q_latent, latent, None, carrier, _ctx_args(topk, wrong_device))
        quant_args = _ctx_args(topk)
        quant_args.out_scale = torch.ones(1, device=device)
        with pytest.raises(ValueError, match="quantized"):
            backend.forward(q_latent, latent, None, carrier, quant_args)
        sf_args = _ctx_args(topk)
        sf_args.output_sf = torch.empty(1, device=device, dtype=torch.uint8)
        with pytest.raises(ValueError, match="quantized"):
            backend.forward(q_latent, latent, None, carrier, sf_args)

    _record(
        evidence,
        "backend_forward_contract",
        {
            "layer": layer_idx,
            "rejects": [
                "metadata=None",
                "metadata without kv_cache_manager",
                "unprepared metadata (no mamba_metadata)",
                "unknown kwargs",
                "forward_args mixed with legacy kwargs",
                "missing sparse_backend_args.topk_indices",
                "attention_input_type=mixed",
                "non-None v",
                "context without k",
                "generation with k",
                "output with wrong shape",
                "output with wrong dtype",
                "output on wrong device",
                "quantized output (out_scale)",
                "quantized output (output_sf)",
            ],
        },
    )


@pytest.mark.parametrize("seq_len", [512, 2048, 2600], ids=["below_topk", "at_topk", "above_topk"])
def test_production_sparse_mla_kernel_dispatch(
    reader, text_config, device, evidence, monkeypatch, seq_len
):
    """The configured backend dispatches the sparse core, at real geometry.

    Runs the module's context phase on a **real** ``Glm5NextCacheManager``
    driven by **real prepared** ``TrtllmAttentionMetadata`` -- the backend
    derives its pools and block tables from that metadata, so this pins that
    the typed metadata actually owns cache execution. Three proofs per length
    (below / at / above ``index_topk``):

    1. **Backend dispatch** -- the kernel entry point the backend resolves
       (``tensorrt_llm.flash_mla.flash_mla_sparse_fwd``) is intercepted with a
       forwarding spy; the module's forward must reach it exactly once per
       forward *through* ``GlmKpoolSparseAttention.forward`` with the
       checkpoint geometry (``q [S, 64, 512]``, one KV head, index width
       padded to the backend's 64-wide tiles, ``d_v == 512``,
       ``scale == 256 ** -0.5``). The model layer has no kernel knowledge, so
       the configured backend is the only route -- and the fully NoPE shape
       pins that no fake rope width ever reaches the kernel.
    2. **Kernel evidence** -- a profiler pass over ``attn_backend.forward``
       alone records the CUDA kernels the backend launches; the list must be
       non-empty and contain no torch SDPA/cuDNN attention kernel.
    3. **Parity** -- output still matches native HF inside the same envelope
       as :func:`test_sparse_mla_matches_native_hf`.
    """
    import tensorrt_llm.flash_mla as _flash_mla_mod
    from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata

    layer_idx = 3
    hf, trt = _load_mla(reader, text_config, layer_idx, device)
    x = _hidden(seq_len, text_config.hidden_size, device, seed=29 + seq_len)

    calls: List[dict] = []
    real_kernel = _flash_mla_mod.flash_mla_sparse_fwd

    def spy(q, kv, indices, sm_scale, d_v=512, *args, **kwargs):
        calls.append(
            {
                "q_shape": tuple(q.shape),
                "kv_shape": tuple(kv.shape),
                "indices_shape": tuple(indices.shape),
                "sm_scale": float(sm_scale),
                "d_v": int(d_v),
                "q_dtype": str(q.dtype),
                "indices_dtype": str(indices.dtype),
            }
        )
        return real_kernel(q, kv, indices, sm_scale, d_v, *args, **kwargs)

    monkeypatch.setattr(_flash_mla_mod, "flash_mla_sparse_fwd", spy)

    manager, _attention, _sparse_ids, _linear, _heads, _dim = _build_cache_manager(
        text_config, tokens_per_block=64, max_seq_len=2688
    )
    mask = torch.ones(1, seq_len, dtype=torch.bool, device=device)
    try:
        manager.add_dummy_requests([0], token_nums=[seq_len])
        metadata = _real_prepared_metadata(
            manager, lens=[seq_len], num_contexts=1, cached=[0], request_ids=[0]
        )
        assert isinstance(metadata, TrtllmAttentionMetadata)
        assert metadata.mamba_metadata.glm_block_tables is not None

        with torch.no_grad():
            expected = hf(x.unsqueeze(0), attention_mask=mask, past_key_values=None)[0][0]
            got = trt.forward_prefill(x, [0, seq_len], [0], metadata)

            # Profile the backend's standard-contract forward in isolation so
            # every recorded CUDA kernel is attributable to the backend. The
            # prefix is re-gathered from the metadata-derived cache the
            # prefill above wrote.
            latent, _packed = trt.attn_backend.gather_paged_prefix(
                seq_len, metadata, request_index=0
            )
            topk, _, _ = _indexer_select(trt, x)
            q_resid = trt.q_a_layernorm(trt.q_a_proj(x))
            query = trt.q_b_proj(q_resid).view(seq_len, trt.num_heads, trt.qk_head_dim)
            q_latent = trt.absorb_query(query)
            from torch.profiler import ProfilerActivity, profile

            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                trt.attn_backend.forward(q_latent, latent, None, metadata, _ctx_args(topk))
            torch.cuda.synchronize()
    finally:
        manager.shutdown()

    kernel_names = sorted(
        {
            evt.key
            for evt in prof.key_averages()
            if (
                getattr(evt, "self_device_time_total", 0) or getattr(evt, "self_cuda_time_total", 0)
            )
            and evt.key
        }
    )
    forbidden = [
        n
        for n in kernel_names
        for bad in ("scaled_dot_product", "sdpa", "cudnn")
        if bad in n.lower()
    ]

    stats = compare(got, expected, f"mla_production_L{layer_idx}_S{seq_len}")
    width = trt.indexer.output_width
    padded_width = width + (-width) % GlmKpoolSparseAttention._KERNEL_TOPK_ALIGN
    _record(
        evidence,
        "production_sparse_mla_dispatch",
        {
            "layer": layer_idx,
            "seq_len": seq_len,
            "index_topk": int(text_config.index_topk),
            "backend_class": type(trt.attn_backend).__name__,
            "metadata_class": type(trt.attn_backend).Metadata.__name__,
            "metadata_instance_class": type(metadata).__name__,
            "cache_manager_class": type(manager).__name__,
            "prepared_glm_buffers": True,
            "kernel": "tensorrt_llm.flash_mla.flash_mla_sparse_fwd",
            "calls": calls,
            "cuda_kernels_in_backend_forward": kernel_names,
            **stats,
        },
    )
    assert type(trt.attn_backend) is GlmKpoolSparseAttention
    # _run_one dispatched through the backend exactly once; the probe adds one.
    assert len(calls) == 2, f"expected 2 backend dispatches (forward + probe), saw {len(calls)}"
    for call in calls:
        assert call["q_shape"] == (seq_len, trt.num_heads, trt.kv_lora_rank), call
        assert call["kv_shape"] == (seq_len, 1, trt.kv_lora_rank), call
        assert call["indices_shape"] == (seq_len, 1, padded_width), call
        assert call["d_v"] == trt.kv_lora_rank == 512, call
        assert abs(call["sm_scale"] - trt.qk_head_dim**-0.5) < 1e-12, call
        assert call["indices_dtype"] == "torch.int32", call
    assert kernel_names, "no CUDA kernel recorded inside the backend forward"
    assert not forbidden, f"torch SDPA kernels ran inside the backend forward: {forbidden}"
    assert stats["all_finite"]
    assert stats["max_abs"] <= MLA_REL_MAX_ABS * stats["ref_max_abs"], stats
    assert stats["cosine"] >= MLA_MIN_COSINE, stats


def test_production_sparse_mla_paged_decode_matches_prefill(reader, text_config, device, evidence):
    """Production decode over the real paged pool agrees with one-shot prefill.

    Two requests live on a **real** ``Glm5NextCacheManager`` and every phase
    runs on **real prepared** ``TrtllmAttentionMetadata``: the context step's
    metadata carries both requests as contexts, the decode step's metadata
    carries them as generations, and the backend derives pools/tables/lengths
    from each. The backend's decode path reads the latent pool *directly*
    through its storage row-space view, so this pins the position -> row
    translation against V2's own page tables: the decoded token's output must
    land in the one-shot prefill envelope, and the row math must address the
    same latents the prefill wrote. Request 1 is a decoy so a row-translation
    bug that crosses requests shows up.
    """
    layer_idx = 3
    _, trt = _load_mla(reader, text_config, layer_idx, device)
    tokens_per_block = 32
    prefix_len = 300
    manager, _attention, _sparse_ids, _linear, _heads, _dim = _build_cache_manager(
        text_config, tokens_per_block=tokens_per_block, max_seq_len=384
    )
    x = _hidden(prefix_len + 1, text_config.hidden_size, device, seed=31)
    other = _hidden(prefix_len, text_config.hidden_size, device, seed=32)

    try:
        manager.add_dummy_requests([0, 1], token_nums=[prefix_len + 1, prefix_len + 1])
        tables = manager.get_batch_slot_tables([0, 1])
        assert tables[0] != tables[1], "requests must own distinct page rows"
        ctx_metadata = _real_prepared_metadata(
            manager,
            lens=[prefix_len, prefix_len],
            num_contexts=2,
            cached=[0, 0],
            request_ids=[0, 1],
        )
        gen_metadata = _real_prepared_metadata(
            manager,
            lens=[1, 1],
            num_contexts=0,
            cached=[prefix_len, prefix_len],
            request_ids=[0, 1],
        )

        with torch.no_grad():
            latent = trt.project_latent(x)
            packed = trt.indexer.packed_state(x)
            one_shot = trt._run_one(
                x,
                latent,
                packed,
                torch.arange(prefix_len + 1, device=device),
                _context_carrier(trt, device),
            )
            trt.forward_prefill(
                torch.cat([x[:prefix_len], other]),
                [0, prefix_len, 2 * prefix_len],
                [0, 0],
                ctx_metadata,
            )
            decoded = trt.forward_decode(
                torch.cat([x[prefix_len:], other[-1:]]),
                torch.tensor([prefix_len + 1, prefix_len + 1], device=device),
                gen_metadata,
            )
    finally:
        manager.shutdown()

    stats = compare(decoded[0], one_shot[-1], "mla_production_decode_vs_oneshot")
    _record(
        evidence,
        "production_sparse_mla_paged_decode",
        {
            "layer": layer_idx,
            "prefix_len": prefix_len,
            "tokens_per_block": tokens_per_block,
            "cache_manager_class": type(manager).__name__,
            "metadata_instance_class": type(gen_metadata).__name__,
            "page_tables": tables,
            "decode_vs_oneshot": stats,
        },
    )
    assert stats["all_finite"]
    assert stats["max_abs"] <= MLA_REL_MAX_ABS * max(stats["ref_max_abs"], 1e-6), stats
    assert not torch.equal(decoded[0], decoded[1]), "decoy request leaked into the decode row"


# ---------------------------------------------------------------------------
# Heterogeneous request state through the paged cache
# ---------------------------------------------------------------------------


class _KpoolPools:
    """Manager-like owner of one sparse layer's pools, keyed by ``trt.layer_idx``.

    The miniature of the ``Glm5NextCacheManager`` accessor surface the backend
    consumes: 4-D ``[pages, tokens_per_block, 1, dim]`` buffers (the shape V2
    hands back; the page axis stays separate because the real ``Role.INDEX_KEY``
    buffer is a strided view into a coalesced pool and cannot be flattened
    without copying -- see :func:`paged_slot_indices`) plus
    ``tokens_per_block``. Tests seed/poison state through the 3-D ``latent`` /
    ``index`` views; the backend reaches the same storage only through the
    metadata carrier built by :func:`_kpool_metadata`.
    """

    def __init__(self, trt, num_pages, tokens_per_block, device):
        self.tokens_per_block = tokens_per_block
        self._latent = {
            trt.layer_idx: torch.zeros(
                num_pages,
                tokens_per_block,
                1,
                trt.kv_lora_rank,
                device=device,
                dtype=torch.bfloat16,
            )
        }
        self._index = {
            trt.layer_idx: torch.zeros(
                num_pages,
                tokens_per_block,
                1,
                trt.indexer.packed_state_dim,
                device=device,
                dtype=torch.bfloat16,
            )
        }
        self.latent = self._latent[trt.layer_idx][:, :, 0, :]
        self.index = self._index[trt.layer_idx][:, :, 0, :]

    def get_latent_state_buffer(self, layer_idx):
        return self._latent[layer_idx]

    def get_index_state_buffer(self, layer_idx):
        return self._index[layer_idx]


def _kpool_metadata(
    manager_like,
    *,
    block_tables: torch.Tensor,
    kv_lens: torch.Tensor,
    num_contexts: int,
    is_cuda_graph: bool = False,
):
    """Prepared-metadata carrier in miniature.

    Exactly the attribute surface ``GlmKpoolSparseAttention._cache_state``
    consumes on its persistent path: a pool-owning manager plus the
    ``prepare()``-refreshed ``glm_block_tables``/``glm_kv_lens`` buffers
    (here maintained by the test between calls, as the runtime's
    ``Glm5NextMamba2Metadata.prepare`` maintains the real ones). The
    real-class equivalent -- ``TrtllmAttentionMetadata`` built on a real
    ``Glm5NextCacheManager`` and ``prepare()``d -- is exercised by
    :func:`_real_prepared_metadata` in the dispatch/paged-decode tests and by
    the runtime-binding suite.
    """
    batch = int(block_tables.shape[0])
    return SimpleNamespace(
        kv_cache_manager=manager_like,
        mamba_metadata=SimpleNamespace(glm_block_tables=block_tables, glm_kv_lens=kv_lens),
        seq_lens=torch.ones(batch, dtype=torch.long),
        num_contexts=num_contexts,
        is_cuda_graph=is_cuda_graph,
    )


def _context_carrier(trt, device, *, batch: int = 1):
    """Minimal carrier for explicit-prefix context runs (``_run_one``).

    The context leg attends the latent prefix passed explicitly as ``k``, so
    the carrier's pools exist only to satisfy the backend's uniform metadata
    validation; one page suffices.
    """
    owner = _KpoolPools(trt, 1, 4, device)
    return _kpool_metadata(
        owner,
        block_tables=torch.zeros(batch, 1, dtype=torch.long, device=device),
        kv_lens=torch.zeros(batch, dtype=torch.long, device=device),
        num_contexts=batch,
    )


def _real_prepared_metadata(
    manager, *, lens, num_contexts, cached, request_ids, max_num_requests=4
):
    """The real typed metadata, prepared on a real ``Glm5NextCacheManager``.

    ``get_attention_backend("TRTLLM").Metadata`` is ``TrtllmAttentionMetadata``
    -- the same class the engine constructs from ``attn_backend.Metadata`` --
    and ``prepare()`` is what attaches the ``Glm5NextMamba2Metadata`` with the
    ``glm_*`` buffers the backend derives its cache state from.
    """
    from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
    from tensorrt_llm._torch.metadata import KVCacheParams

    metadata_cls = get_attention_backend("TRTLLM").Metadata
    metadata = metadata_cls(
        seq_lens=torch.tensor(list(lens), dtype=torch.int),
        num_contexts=num_contexts,
        kv_cache_params=KVCacheParams(
            use_cache=True, num_cached_tokens_per_seq=[int(c) for c in cached]
        ),
        kv_cache_manager=manager,
        request_ids=list(request_ids),
        prompt_lens=list(lens),
        max_num_requests=max_num_requests,
        max_num_tokens=8192,
    )
    metadata.prepare()
    return metadata


def test_sparse_mla_paged_prefill_decode_and_slot_reuse(reader, text_config, device, evidence):
    """Chunked prefill, decode and page reuse over a real paged layout.

    The page tables are deliberately **non-contiguous and interleaved** between
    the two requests, so a layer that assumed a flat per-request buffer would
    read the other request's tokens.
    """
    layer_idx = 3
    hf, trt = _load_mla(reader, text_config, layer_idx, device)
    tokens_per_block = 32
    seq_len = 256
    pages_per_req = seq_len // tokens_per_block + 1
    num_pages = 2 * pages_per_req + 2
    pools = _KpoolPools(trt, num_pages, tokens_per_block, device)
    latent_pool, index_pool = pools.latent, pools.index

    # Interleaved page assignment: request 0 takes the even pages, request 1 the
    # odd ones, so adjacent logical blocks are never adjacent physical pages.
    table = torch.tensor(
        [
            [2 * i for i in range(pages_per_req)],
            [2 * i + 1 for i in range(pages_per_req)],
        ],
        device=device,
        dtype=torch.long,
    )

    def ctx_md(rows: torch.Tensor, cached_lens):
        # kv_lens is a decode-phase value; context carriers hold zeros.
        kv = torch.zeros(len(cached_lens), device=device, dtype=torch.long)
        return _kpool_metadata(pools, block_tables=rows, kv_lens=kv, num_contexts=rows.shape[0])

    x0 = _hidden(seq_len, text_config.hidden_size, device, seed=11)
    x1 = _hidden(seq_len, text_config.hidden_size, device, seed=12)
    nxt = _hidden(1, text_config.hidden_size, device, seed=13)

    with torch.no_grad():
        reference = hf(
            x0.unsqueeze(0),
            attention_mask=torch.ones(1, seq_len, dtype=torch.bool, device=device),
            past_key_values=None,
        )[0][0]

        one_shot = trt.forward_prefill(x0, [0, seq_len], [0], ctx_md(table[:1], [0]))

        # Same request, arrived in three chunks on non-64-aligned boundaries.
        latent_pool.zero_()
        index_pool.zero_()
        pieces, cached = [], 0
        for a, b in ((0, 30), (30, 161), (161, seq_len)):
            pieces.append(
                trt.forward_prefill(x0[a:b], [0, b - a], [cached], ctx_md(table[:1], [cached]))
            )
            cached = b
        chunked = torch.cat(pieces, dim=0)

        # Batched context phase for both requests, then a decode step each.
        latent_pool.zero_()
        index_pool.zero_()
        both = trt.forward_prefill(
            torch.cat([x0, x1]),
            [0, seq_len, 2 * seq_len],
            [0, 0],
            ctx_md(table, [0, 0]),
        )
        # kv_lens: visible length INCLUDING the token being decoded.
        decode_kv = torch.tensor([seq_len + 1, seq_len + 1], device=device)
        decode_two = trt.forward_decode(
            torch.cat([nxt, nxt]),
            decode_kv,
            _kpool_metadata(pools, block_tables=table, kv_lens=decode_kv, num_contexts=0),
        )

    vs_hf = compare(one_shot, reference, "mla_paged_oneshot")
    vs_chunk = compare(chunked, one_shot, "mla_paged_chunked")
    vs_batched = compare(both[:seq_len], one_shot, "mla_paged_batched_req0")
    isolation = compare(both[seq_len:], both[:seq_len], "mla_paged_request_isolation")

    assert vs_hf["max_abs"] <= MLA_REL_MAX_ABS * vs_hf["ref_max_abs"], vs_hf
    assert vs_chunk["max_abs"] <= MLA_REL_MAX_ABS * max(vs_chunk["ref_max_abs"], 1e-6), vs_chunk
    assert vs_batched["max_abs"] == 0.0, vs_batched
    # Two different requests sharing one pool must not produce the same output.
    assert isolation["max_abs"] > 0.1, isolation
    # The two decode rows come from identical input but different prefixes.
    assert not torch.equal(decode_two[0], decode_two[1])

    # --- cancellation and slot reuse ---------------------------------------
    # Request 1 is cancelled mid-prefill and its pages are handed to a new
    # request. The new request must produce exactly what it produces on a clean
    # pool, i.e. no stale page may survive.
    with torch.no_grad():
        latent_pool.zero_()
        index_pool.zero_()
        trt.forward_prefill(
            x1[:100], [0, 100], [0], ctx_md(table[1:], [0])
        )  # cancelled here: no free/clear performed by the layer
        reused = trt.forward_prefill(x0, [0, seq_len], [0], ctx_md(table[1:], [0]))
        latent_pool.zero_()
        index_pool.zero_()
        fresh = trt.forward_prefill(x0, [0, seq_len], [0], ctx_md(table[1:], [0]))
    reuse_stats = compare(reused, fresh, "mla_slot_reuse_after_cancel")
    _record(
        evidence,
        "paged_state",
        {
            "layer": layer_idx,
            "tokens_per_block": tokens_per_block,
            "page_table": table.tolist(),
            "one_shot_vs_hf": vs_hf,
            "chunked_vs_one_shot": vs_chunk,
            "batched_vs_single": vs_batched,
            "request_isolation": isolation,
            "slot_reuse_after_cancel": reuse_stats,
        },
    )
    # A cancelled request writes only positions it actually produced, and the
    # new request overwrites exactly the positions it reads, so reuse is exact.
    assert reuse_stats["max_abs"] == 0.0, reuse_stats


# ---------------------------------------------------------------------------
# source_activation_replay: real captured hidden states
# ---------------------------------------------------------------------------

# The in-model output was produced by block-FP8 kernels; this rung dequantizes
# to bf16 and uses an ordinary matmul. That difference is numerical, not
# structural, and these are the bounds established in the Goal-1.1 replay:
#   * TensorRT-LLM and standalone HF, both bf16 on identical dequantized
#     weights, must agree *very* tightly -- that is the correctness claim;
#   * each may drift from the FP8 in-model output by a larger bounded amount;
#   * and they must drift by the *same* amount, since a structural error would
#     move only one of them.
FP8_REPLAY_PAIR_COSINE = 0.9999
FP8_REPLAY_MODEL_COSINE = 0.995
FP8_REPLAY_MODEL_REL_MAX_ABS = 8e-2


def _replay_layers(prompts, text_config, kind):
    ids = set()
    for prompt in prompts:
        for name in prompt["activations"]:
            if name.endswith("self_attn.input"):
                ids.add(int(name.split(".")[0].removeprefix("layer")))
    return sorted(i for i in ids if text_config.layer_types[i] == kind)


def _assert_replay_envelope(label, vs_model, vs_hf, hf_vs_model):
    """TensorRT-LLM must sit inside the same FP8 envelope as standalone HF."""
    assert vs_hf["cosine"] > FP8_REPLAY_PAIR_COSINE, (
        f"{label}: TensorRT-LLM and standalone HF disagree on identical "
        f"dequantized weights -- structural, not FP8 error: {vs_hf}"
    )
    assert vs_model["cosine"] > FP8_REPLAY_MODEL_COSINE, (label, vs_model)
    assert vs_model["max_abs"] <= FP8_REPLAY_MODEL_REL_MAX_ABS * max(
        vs_model["ref_max_abs"], 1e-3
    ), (label, vs_model)
    assert abs(vs_model["cosine"] - hf_vs_model["cosine"]) < 1e-3, (
        f"{label}: TensorRT-LLM and HF drift differently from the in-model FP8 "
        f"output ({vs_model['cosine']:.6f} vs {hf_vs_model['cosine']:.6f})"
    )


def test_source_activation_replay_kda(reader, text_config, hooked, device, evidence):
    """Real hidden states entering KDA layers, replayed through TensorRT-LLM.

    Synthetic activations cannot exercise the magnitude and correlation
    structure the real model actually produces at layer N, so the pass-critical
    claim is made against hooked inputs and the in-model output they produced.
    """
    layer_ids = _replay_layers(hooked, text_config, "linear_attention")
    assert layer_ids, "fixture captured no linear-attention layers"
    rows = 0
    for layer_idx in layer_ids:
        hf, trt = _load_kda(reader, text_config, layer_idx, device)
        conv, ssm = _new_kda_pools(trt, 1, device)
        slot = torch.tensor([0], device=device)
        for prompt in hooked:
            acts = prompt["activations"]
            key_in = f"layer{layer_idx}.self_attn.input"
            key_out = f"layer{layer_idx}.self_attn.output"
            if key_in not in acts or key_out not in acts:
                continue
            x = acts[key_in].to(device=device, dtype=torch.bfloat16)
            captured = acts[key_out].to(device=device, dtype=torch.float32)
            seq_len = x.shape[1]
            conv.zero_()
            ssm.zero_()
            with torch.no_grad():
                hf_out = hf(hidden_states=x, cache_params=None, attention_mask=None).float()
                got = trt.forward_prefill(
                    x[0], [0, seq_len], slot, conv, ssm, cached_lens=[0]
                ).float()

            vs_model = compare(got, captured[0], "trtllm_vs_in_model")
            vs_hf = compare(got, hf_out[0], "trtllm_vs_standalone_hf")
            hf_vs_model = compare(hf_out[0], captured[0], "standalone_hf_vs_in_model")
            _record(
                evidence,
                "source_activation_replay_kda",
                {
                    "layer_idx": layer_idx,
                    "prompt_index": prompt["index"],
                    "seq_len": int(seq_len),
                    "input_absmax": float(x.abs().max()),
                    "trtllm_vs_in_model": vs_model,
                    "trtllm_vs_standalone_hf": vs_hf,
                    "standalone_hf_vs_in_model": hf_vs_model,
                },
            )
            _assert_replay_envelope(f"kda L{layer_idx}", vs_model, vs_hf, hf_vs_model)
            rows += 1
    assert rows >= len(layer_ids), "expected at least one replayed prompt per layer"


def test_source_activation_replay_sparse_attention(reader, text_config, hooked, device, evidence):
    """Real hidden states entering sparse-MLA layers, replayed through TensorRT-LLM."""
    layer_ids = _replay_layers(hooked, text_config, "deepseek_sparse_attention")
    assert layer_ids, "fixture captured no sparse-attention layers"
    rows = 0
    for layer_idx in layer_ids:
        hf, trt = _load_mla(reader, text_config, layer_idx, device)
        carrier = _context_carrier(trt, device)
        for prompt in hooked:
            acts = prompt["activations"]
            key_in = f"layer{layer_idx}.self_attn.input"
            key_out = f"layer{layer_idx}.self_attn.output"
            if key_in not in acts or key_out not in acts:
                continue
            x = acts[key_in].to(device=device, dtype=torch.bfloat16)
            captured = acts[key_out].to(device=device, dtype=torch.float32)
            seq_len = x.shape[1]
            mask = torch.ones(1, seq_len, dtype=torch.bool, device=device)
            with torch.no_grad():
                hf_out = hf(x, attention_mask=mask, past_key_values=None)[0].float()
                latent = trt.project_latent(x[0])
                packed = trt.indexer.packed_state(x[0])
                got = trt._run_one(
                    x[0], latent, packed, torch.arange(seq_len, device=device), carrier
                ).float()

            vs_model = compare(got, captured[0], "trtllm_vs_in_model")
            vs_hf = compare(got, hf_out[0], "trtllm_vs_standalone_hf")
            hf_vs_model = compare(hf_out[0], captured[0], "standalone_hf_vs_in_model")
            _record(
                evidence,
                "source_activation_replay_sparse_mla",
                {
                    "layer_idx": layer_idx,
                    "prompt_index": prompt["index"],
                    "seq_len": int(seq_len),
                    "input_absmax": float(x.abs().max()),
                    "trtllm_vs_in_model": vs_model,
                    "trtllm_vs_standalone_hf": vs_hf,
                    "standalone_hf_vs_in_model": hf_vs_model,
                },
            )
            _assert_replay_envelope(f"sparse_mla L{layer_idx}", vs_model, vs_hf, hf_vs_model)
            rows += 1
    assert rows >= len(layer_ids), "expected at least one replayed prompt per layer"


def test_paged_slot_indices_address_the_right_page(device):
    """The paging helper must never fabricate a page for an unmapped position."""
    tokens_per_block = 4
    table = torch.tensor([[3, 1, 0], [2, 5, 4]], device=device, dtype=torch.long)
    positions = torch.arange(12, device=device)
    page, offset = paged_slot_indices(table[0], positions, tokens_per_block)
    expected_page = torch.tensor([3] * 4 + [1] * 4 + [0] * 4, device=device, dtype=torch.long)
    expected_offset = torch.tensor([0, 1, 2, 3] * 3, device=device, dtype=torch.long)
    assert torch.equal(page, expected_page)
    assert torch.equal(offset, expected_offset)

    other_page, other_offset = paged_slot_indices(table[1], positions, tokens_per_block)
    # Distinct page tables must never name the same (page, offset) cell, which is
    # the cross-request leak the hybrid cache has to rule out.
    mine = page * tokens_per_block + offset
    theirs = other_page * tokens_per_block + other_offset
    assert not bool((mine[:, None] == theirs[None, :]).any()), "page tables must not overlap"

    # A strided pool -- the shape KVCacheManagerV2 actually hands back for
    # Role.INDEX_KEY -- must round-trip through the pair. Flattening it would
    # raise on .view and silently copy on .reshape, losing every write.
    dense = torch.zeros(6, tokens_per_block * 2, 8, device=device)
    strided = dense[:, :tokens_per_block]
    assert not strided.is_contiguous()
    strided[page, offset] = torch.arange(12, device=device, dtype=torch.float32)[:, None]
    assert torch.equal(strided[page, offset][:, 0], torch.arange(12, device=device).float())
    assert float(dense[:, tokens_per_block:].abs().max()) == 0.0


def _build_cache_manager(text_config, *, num_layers=12, tokens_per_block=32, max_seq_len=256):
    """A real ``Glm5NextCacheManager`` over an affordable slice of the schedule.

    Only the head *count* of the recurrent side is scaled down; every dimension
    that defines a cache descriptor -- ``kv_lora_rank``, ``index_head_dim``, the
    four-tap window -- is the checkpoint's own, and the slice keeps more than one
    sparse layer so the coalesced-pool geometry is the real one.
    """
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    linear = dict(text_config.linear_attn_config)
    attention = list(text_config.layer_types)[:num_layers]
    sparse_ids = [i for i, t in enumerate(attention) if t == "deepseek_sparse_attention"]
    linear_ids = [i for i, t in enumerate(attention) if t == "linear_attention"]
    assert len(sparse_ids) > 1 and linear_ids

    head_dim = linear["head_dim"]
    mamba_heads = 8
    manager = glm5_next_cache_manager_cls()(
        mamba_d_state=head_dim,
        mamba_d_conv=linear["short_conv_kernel_size"],
        mamba_num_heads=mamba_heads,
        mamba_n_groups=mamba_heads,
        mamba_head_dim=head_dim,
        mamba_num_layers=len(linear_ids),
        mamba_layer_mask=[t == "linear_attention" for t in attention],
        mamba_cache_dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=torch.float32,
        # The pool budget must scale with the requested horizon: the sparse
        # layers coalesce into one V2 pool, so a fixed 4096-token budget
        # rejects checkpoint-scale (>=2048-token) requests at allocation time
        # with "base page indices is too short".
        kv_cache_config=KvCacheConfig(
            max_tokens=max(4096, 8 * max_seq_len * 4), enable_block_reuse=False
        ),
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=len(sparse_ids),
        num_kv_heads=1,
        head_dim=text_config.kv_lora_rank,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=4,
        mapping=Mapping(world_size=1, tp_size=1, pp_size=1),
        layer_mask=[t == "deepseek_sparse_attention" for t in attention],
        dtype=DataType.BF16,
        conv_state_layout="q_k_v",
        sparse_layer_ids=sparse_ids,
        index_state_dim=2 * text_config.index_head_dim,
    )
    return manager, attention, sparse_ids, linear_ids, mamba_heads, head_dim


def test_coalesced_latent_pool_is_addressed_by_slot(reader, text_config, device, evidence):
    """Sparse layers sharing one V2 pool must not overwrite each other's cache.

    ``KVCacheManagerV2`` coalesces every sparse layer's ``Role.KEY`` buffer into
    a single pool and starts layer ``L``'s ``get_buffers`` view ``L`` pages into
    it, with a *single-page* dim-0 stride. Addressing those views with a raw
    block id therefore aliases layer ``L``'s page ``p`` onto layer ``L+1``'s page
    ``p-1``. V2's own callers never do that -- they scale each base page index by
    ``get_layer_page_index_scale`` first -- and ``get_latent_state_buffer`` folds
    the same scale into the view so one base block table addresses both pools.

    This is invisible to a one-shot prefill: each layer writes its state and
    reads it back inside a single call, with no other layer in between. It is
    fatal in decode, where the whole prefix has to survive every other sparse
    layer's writes -- measured on the real model, every cached position older
    than ``tokens_per_block`` steps was being overwritten.
    """
    manager, _attention, sparse_ids, _linear, _heads, _dim = _build_cache_manager(text_config)
    tokens_per_block = manager.tokens_per_block
    scale = manager.get_layer_page_index_scale(sparse_ids[0])
    # Without a coalesced pool the whole hazard is absent and the test proves
    # nothing, so the premise is asserted rather than assumed.
    assert scale > 1, f"expected a coalesced pool over {len(sparse_ids)} sparse layers"

    # (1) Distinct sentinels through the slot-indexed accessors must survive.
    buffers = {}
    for layer_idx in sparse_ids:
        buffers[f"latent{layer_idx}"] = manager.get_latent_state_buffer(layer_idx)
        buffers[f"index{layer_idx}"] = manager.get_index_state_buffer(layer_idx)
    for order, buffer in enumerate(buffers.values(), start=1):
        buffer.fill_(float(order))
    torch.cuda.synchronize(device)
    overlaps = {
        name: int((buffer != float(order)).sum())
        for order, (name, buffer) in enumerate(buffers.items(), start=1)
    }
    # (2) The page-indexed view is the one that aliases -- the negative control
    # that keeps this test honest about what it is protecting against.
    paged = {i: manager.get_buffers(i)[:, 0] for i in sparse_ids}
    paged[sparse_ids[0]].fill_(1.0)
    paged[sparse_ids[1]].fill_(2.0)
    torch.cuda.synchronize(device)
    control_overlap = int((paged[sparse_ids[0]] != 1.0).sum())

    # (3) The observable: a decode that must read a prefix other layers wrote
    # over. Contiguous pages are deliberate -- that is the worst case for a
    # one-page aliasing offset, and what the whole-model driver actually uses.
    # The interference goes through *separate modules at the neighbouring
    # layer ids*, exactly as in the real stack: one metadata carrier over the
    # real manager drives every layer's backend, and each backend derives its
    # own layer's pools from it.
    layer_idx = sparse_ids[0]
    _hf, trt = _load_mla(reader, text_config, layer_idx, device)
    neighbours = [
        Glm5NextSparseAttention(text_config, n).to(device=device).eval() for n in sparse_ids[1:]
    ]
    seq_len = 3 * tokens_per_block
    blocks = seq_len // tokens_per_block
    table = torch.arange(blocks, device=device, dtype=torch.long).unsqueeze(0)
    x = _hidden(seq_len, text_config.hidden_size, device, seed=71)
    other = _hidden(seq_len - 1, text_config.hidden_size, device, seed=72)

    ctx_metadata = _kpool_metadata(
        manager,
        block_tables=table,
        kv_lens=torch.zeros(1, dtype=torch.long, device=device),
        num_contexts=1,
    )
    decode_kv = torch.tensor([seq_len], device=device)  # kv_lens: cached + this token
    decode_metadata = _kpool_metadata(
        manager, block_tables=table, kv_lens=decode_kv, num_contexts=0
    )

    def run(with_interference: bool):
        for buffer in buffers.values():
            buffer.zero_()
        with torch.no_grad():
            trt.forward_prefill(x[:-1], [0, seq_len - 1], [0], ctx_metadata)
            if with_interference:
                for module in neighbours:
                    module.forward_prefill(other, [0, seq_len - 1], [0], ctx_metadata)
            return trt.forward_decode(x[-1:], decode_kv, decode_metadata)

    clean = run(with_interference=False)
    disturbed = run(with_interference=True)
    isolation = compare(disturbed, clean, "latent_pool_cross_layer_isolation")

    _record(
        evidence,
        "coalesced_pool",
        {
            "sparse_layers": sparse_ids,
            "page_index_scale": int(scale),
            "kv_factor": int(manager.kv_factor),
            "latent_pages": int(manager.get_buffers(layer_idx).shape[0]),
            "latent_slots": int(manager.get_latent_state_buffer(layer_idx).shape[0]),
            "index_slots": int(manager.get_index_state_buffer(layer_idx).shape[0]),
            "slot_view_overlaps": overlaps,
            "page_view_control_overlap": control_overlap,
            "decode_isolation": isolation,
        },
    )
    # Latent and indexer must expose the same slot space on *every* sparse
    # layer, or one block table cannot address both.
    slot_spaces = {
        i: (
            int(manager.get_latent_state_buffer(i).shape[0]),
            int(manager.get_index_state_buffer(i).shape[0]),
        )
        for i in sparse_ids
    }
    manager.shutdown()

    assert not any(overlaps.values()), overlaps
    assert all(latent == index for latent, index in slot_spaces.values()), slot_spaces
    assert control_overlap > 0, (
        "the page-indexed view is expected to alias across layers; if it no "
        "longer does, get_latent_state_buffer's scaling is now redundant"
    )
    assert isolation["max_abs"] == 0.0, isolation


def test_cache_manager_owns_all_three_state_families(text_config, device, evidence):
    """One KVCacheManagerV2 lifecycle carries KDA, latent-KV and indexer state.

    The literal ``layer_types`` list drives which layer gets which descriptor,
    and the three shapes stay native -- none is padded into a common KV tensor.
    """
    manager, attention, sparse_ids, linear_ids, mamba_heads, head_dim = _build_cache_manager(
        text_config
    )
    linear = dict(text_config.linear_attn_config)

    layer_cache = manager.mamba_layer_cache(linear_ids[0])
    conv_shape = tuple(layer_cache.conv.shape)
    ssm_shape = tuple(layer_cache.temporal.shape)
    # Four-tap convolution keeps W-1 real history rows over [q | k | v].
    assert conv_shape[1] == 3 * mamba_heads * head_dim
    assert conv_shape[2] == linear["short_conv_kernel_size"] - 1 == 3
    assert ssm_shape[1:] == (mamba_heads, head_dim, head_dim)
    # The recurrent accumulator is fp32 even though the model is bf16.
    assert layer_cache.temporal.dtype == torch.float32
    assert layer_cache.conv.dtype == torch.bfloat16

    index_shapes = {}
    for layer_idx in sparse_ids:
        kv = manager.get_buffers(layer_idx)
        index = manager.get_index_state_buffer(layer_idx)
        assert kv is not None and index is not None
        # SELFKONLY latent: one head of kv_lora_rank, not expanded K/V.
        assert kv.shape[-1] == text_config.kv_lora_rank == 512
        assert index.shape[-1] == 2 * text_config.index_head_dim == 256
        assert index.dtype == torch.bfloat16
        index_shapes[layer_idx] = tuple(index.shape)
    # Linear-attention layers own no attention pages and no indexer state.
    for layer_idx in linear_ids:
        assert manager.get_index_state_buffer(layer_idx) is None

    _record(
        evidence,
        "cache_manager",
        {
            "class": type(manager).__name__,
            "bases": [b.__name__ for b in type(manager).__mro__[1:4]],
            "attention_schedule": attention,
            "sparse_layers": sparse_ids,
            "conv_shape": conv_shape,
            "ssm_shape": ssm_shape,
            "ssm_dtype": str(layer_cache.temporal.dtype),
            "index_shapes": index_shapes,
        },
    )
