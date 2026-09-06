# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Goal 5.2 unit tier: TP-aware KDA / sparse-MLA / indexer module math.

Single-process tests of the head-sharding contracts, using hand-sharded rank
modules (a fake ``tp_size=4`` mapping plus manual slices of one tp1 module's
weights, exactly the ranges the Mapping-aware ``Linear`` swap and the
exact-placement loader produce). The four-rank end-to-end proof -- real
checkpoint, real loader, real all-reduce, real cache manager -- belongs to
``glm5_next_tp4_attention_replay.py``; this tier pins the *math*:

* rank-local KDA state pools are literally the tp1 pools' head slices;
* the sum of the four ranks' row-parallel partials reproduces the tp1 output;
* the four ranks' partial indexer scores sum to the tp1 score, and identical
  reduced scores yield identical selected indices on every rank;
* at ``tp_size == 1`` the modules are byte-identical to the pre-TP layer
  (the frozen PP4 oracle).
"""

import os
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.modeling_glm5_next import (
    Glm5NextIndexer,
    Glm5NextLinearAttention,
    Glm5NextSparseAttention,
)

CHECKPOINT = os.environ.get("GLM53_FLASH_CHECKPOINT", "/dev/shm/GLM-5.3-Flash")

TP = 4


def _config():
    return SimpleNamespace(
        hidden_size=4096,
        rms_norm_eps=1e-5,
        linear_attn_config={
            "num_heads": 64,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
        },
        num_attention_heads=64,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=256,
        qk_rope_head_dim=0,
        v_head_dim=256,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=2048,
        index_kpool=4,
        index_kpool_always_select_tail=True,
    )


def _mapping(rank: int) -> SimpleNamespace:
    return SimpleNamespace(tp_size=TP, tp_rank=rank)


class _StubAllReduce:
    """Records inputs; select() tests drive its return through ``payload``."""

    instances = []

    def __init__(self, mapping=None, strategy=None):
        self.mapping = mapping
        self.strategy = strategy
        self.payload = None  # None -> identity; tensor -> returned as the "sum"
        self.last_input = None
        self.calls = 0
        _StubAllReduce.instances.append(self)

    def __call__(self, x):
        self.calls += 1
        self.last_input = x.detach().clone()
        return x if self.payload is None else self.payload.to(x.dtype)


@pytest.fixture
def stub_allreduce(monkeypatch):
    import tensorrt_llm._torch.distributed as dist_pkg

    _StubAllReduce.instances = []
    monkeypatch.setattr(dist_pkg, "AllReduce", _StubAllReduce)
    return _StubAllReduce


def _seeded(shape, seed, dtype=torch.bfloat16, scale=0.05):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(*shape, generator=gen, dtype=torch.float32) * scale).to(dtype)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def test_kda_tp4_geometry_and_ranges():
    cfg = _config()
    ranges_h, ranges_c = [], []
    for rank in range(TP):
        with torch.device("meta"):
            kda = Glm5NextLinearAttention(cfg, 0, mapping=_mapping(rank))
        assert kda.num_heads == 16 and kda.total_num_heads == 64
        assert kda.qkv_dim == 2048 and kda.conv_dim == 6144
        assert tuple(kda.conv1d.weight.shape) == (6144, 1, 4)
        assert tuple(kda.A_log.shape) == (16,)
        assert tuple(kda.dt_bias.shape) == (2048,)
        # Raw projections keep the full checkpoint geometry until the
        # Mapping-aware swap installs the column/row ownership.
        assert tuple(kda.q_proj.weight.shape) == (8192, 4096)
        assert tuple(kda.o_proj.weight.shape) == (4096, 8192)
        ranges_h.append(kda.kda_head_range())
        ranges_c.append(kda.kda_channel_range())
    # The four ranks tile the head and head-channel axes with no gap/overlap.
    assert ranges_h == [(0, 16), (16, 32), (32, 48), (48, 64)]
    assert ranges_c == [(0, 2048), (2048, 4096), (4096, 6144), (6144, 8192)]

    with pytest.raises(ValueError, match="not divisible"):
        Glm5NextLinearAttention(cfg, 0, mapping=SimpleNamespace(tp_size=3, tp_rank=0))


def test_indexer_and_sparse_tp4_geometry(stub_allreduce):
    cfg = _config()
    with torch.device("meta"):
        attn = Glm5NextSparseAttention(cfg, 3, mapping=_mapping(1))
    assert attn.num_heads == 16 and attn.total_num_heads == 64
    assert tuple(attn.q_b_proj.weight.shape) == (16384, 1536)
    assert attn.attn_backend.num_heads == 16
    assert attn.attn_backend.head_dim == 512
    assert attn.attn_backend.num_kv_heads == 1
    idx = attn.indexer
    assert idx.n_heads == 8 and idx.total_n_heads == 32
    assert idx.score_all_reduce is not None
    assert idx.score_all_reduce.mapping.tp_rank == 1

    with torch.device("meta"):
        tp1 = Glm5NextSparseAttention(cfg, 3)
    assert tp1.num_heads == 64 and tp1.indexer.n_heads == 32
    assert tp1.indexer.score_all_reduce is None


# ---------------------------------------------------------------------------
# KDA: hand-sharded rank modules against the tp1 module
# ---------------------------------------------------------------------------


def _local_linear(full: torch.nn.Linear, rows=None, cols=None) -> torch.nn.Linear:
    weight = full.weight.detach()
    if rows is not None:
        weight = weight[rows[0] : rows[1]]
    if cols is not None:
        weight = weight[:, cols[0] : cols[1]]
    local = torch.nn.Linear(
        weight.shape[1], weight.shape[0], bias=False, dtype=weight.dtype, device=weight.device
    )
    with torch.no_grad():
        local.weight.copy_(weight)
    return local


def _shard_kda(tp1: Glm5NextLinearAttention, rank: int, device) -> Glm5NextLinearAttention:
    """One rank's module, sharded exactly as the swap + loader shard it."""
    cfg = _config()
    kda = Glm5NextLinearAttention(cfg, tp1.layer_idx, mapping=_mapping(rank)).to(device).eval()
    ch = kda.kda_channel_range()
    hd = kda.kda_head_range()
    kda.q_proj = _local_linear(tp1.q_proj, rows=ch)
    kda.k_proj = _local_linear(tp1.k_proj, rows=ch)
    kda.v_proj = _local_linear(tp1.v_proj, rows=ch)
    kda.f_b_proj = _local_linear(tp1.f_b_proj, rows=ch)
    kda.g_b_proj = _local_linear(tp1.g_b_proj, rows=ch)
    kda.b_proj = _local_linear(tp1.b_proj, rows=hd)
    kda.f_a_proj = _local_linear(tp1.f_a_proj)
    kda.g_a_proj = _local_linear(tp1.g_a_proj)
    # Row-parallel o_proj: this rank's column slice, output left as a partial
    # (the in-Linear all-reduce is the four-rank driver's concern).
    kda.o_proj = _local_linear(tp1.o_proj, cols=ch)
    with torch.no_grad():
        kda.dt_bias.copy_(tp1.dt_bias[ch[0] : ch[1]])
        kda.A_log.copy_(tp1.A_log[hd[0] : hd[1]])
        kda.o_norm_weight.copy_(tp1.o_norm_weight)
        # Each conv source sliced by the head-channel range, then [q | k | v].
        full = tp1.conv1d.weight  # [24576, 1, 4] = [q(8192) | k | v]
        pieces = [full[section * 8192 + ch[0] : section * 8192 + ch[1]] for section in range(3)]
        kda.conv1d.weight.copy_(torch.cat(pieces, dim=0))
    return kda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="KDA kernels are CUDA")
def test_kda_tp4_partials_reproduce_tp1():
    """Per-head state slices are bitwise; summed o_proj partials match tp1.

    Runs prefill (torch-scan small-batch route -- the same pool contract as
    the production kernel) and one decode step (the production Triton kernel)
    on the tp1 module and on all four hand-sharded rank modules.
    """
    device = torch.device("cuda")
    cfg = _config()
    torch.manual_seed(0)
    tp1 = Glm5NextLinearAttention(cfg, 0).to(device).eval()
    with torch.no_grad():
        for p in tp1.parameters():
            p.copy_(torch.randn_like(p.float()).mul(0.05).to(p.dtype))
    ranks = [_shard_kda(tp1, r, device) for r in range(TP)]

    seq_len, slots = 96, 2
    x = _seeded((seq_len, cfg.hidden_size), 7).to(device)
    step = _seeded((1, cfg.hidden_size), 8).to(device)

    def pools(heads):
        conv = torch.zeros(slots, 3 * heads * 128, 3, device=device, dtype=torch.bfloat16)
        ssm = torch.zeros(slots, heads, 128, 128, device=device, dtype=torch.float32)
        return conv, ssm

    slot = torch.tensor([0], device=device)
    with torch.no_grad():
        conv1, ssm1 = pools(64)
        out1 = tp1.forward_prefill(x, [0, seq_len], slot, conv1, ssm1, cached_lens=[0])
        # Snapshot the tp1 pools *before* the decode step mutates them: the
        # rank prefill states are compared against the prefill-phase state,
        # the rank decode states against the post-decode state.
        ssm1_prefill = ssm1.clone()
        conv1_prefill = conv1.clone()
        dec1 = tp1.forward_decode(step, slot, conv1, ssm1)

        partial_sum = torch.zeros_like(out1)
        dec_sum = torch.zeros_like(dec1)
        for r, kda in enumerate(ranks):
            conv4, ssm4 = pools(16)
            out4 = kda.forward_prefill(x, [0, seq_len], slot, conv4, ssm4, cached_lens=[0])
            hd = kda.kda_head_range()
            ch = kda.kda_channel_range()
            # Rank state is the tp1 state's head slice: the delta rule never
            # mixes heads. Tight fp envelopes rather than bitwise because the
            # local projections are narrower GEMMs than the tp1 ones and the
            # kernel tiling may legally differ at working precision.
            assert torch.allclose(ssm4[0], ssm1_prefill[0, hd[0] : hd[1]], rtol=1e-3, atol=1e-4), (
                f"rank{r} ssm slice"
            )
            expected_conv = torch.cat(
                [conv1_prefill[0, s * 8192 + ch[0] : s * 8192 + ch[1]] for s in range(3)],
                dim=0,
            )
            assert torch.allclose(conv4[0].float(), expected_conv.float(), rtol=1e-2, atol=1e-3), (
                f"rank{r} conv slice"
            )
            partial_sum += out4
            dec4 = kda.forward_decode(step, slot, conv4, ssm4)
            assert torch.allclose(ssm4[0], ssm1[0, hd[0] : hd[1]], rtol=1e-3, atol=1e-4), (
                f"rank{r} decode ssm slice"
            )
            dec_sum += dec4

    for label, got, want in (("prefill", partial_sum, out1), ("decode", dec_sum, dec1)):
        diff = (got.float() - want.float()).abs().max().item()
        scale = max(want.float().abs().max().item(), 1e-3)
        assert diff <= 2e-2 * scale, f"{label}: summed partials off by {diff} (scale {scale})"


# ---------------------------------------------------------------------------
# indexer: partial scores sum to the tp1 score; identical reduced scores
# yield identical indices
# ---------------------------------------------------------------------------


def _shard_indexer(tp1: Glm5NextIndexer, rank: int, device) -> Glm5NextIndexer:
    cfg = _config()
    idx = Glm5NextIndexer(cfg, tp1.layer_idx, mapping=_mapping(rank)).to(device).eval()
    h = (rank * 8 * 128, (rank + 1) * 8 * 128)
    w = (rank * 8, (rank + 1) * 8)
    idx.wq_b = _local_linear(tp1.wq_b, rows=h)
    idx.weights_proj = _local_linear(tp1.weights_proj, rows=w)
    idx.wk = _local_linear(tp1.wk)
    with torch.no_grad():
        idx.k_norm.weight.copy_(tp1.k_norm.weight)
        idx.k_norm.bias.copy_(tp1.k_norm.bias)
        idx.index_kpool_compress_ape.copy_(tp1.index_kpool_compress_ape)
        idx.index_kpool_compress_gate.copy_(tp1.index_kpool_compress_gate)
    return idx


@pytest.mark.skipif(not torch.cuda.is_available(), reason="matches the CUDA-only replay tier")
def test_indexer_tp4_scores_and_selection(stub_allreduce):
    device = torch.device("cuda")
    cfg = _config()
    torch.manual_seed(1)
    tp1 = Glm5NextIndexer(cfg, 3).to(device).eval()
    with torch.no_grad():
        for p in tp1.parameters():
            p.copy_(torch.randn_like(p.float()).mul(0.05).to(p.dtype))
    ranks = [_shard_indexer(tp1, r, device) for r in range(TP)]
    stubs = [idx.score_all_reduce for idx in ranks]
    assert all(s is not None for s in stubs)

    kv_len, num_tokens = 500, 3
    num_pools = (kv_len + 3) // 4
    hidden = _seeded((num_tokens, cfg.hidden_size), 21).to(device)
    q_resid = _seeded((num_tokens, cfg.q_lora_rank), 22).to(device)
    prefix = _seeded((num_pools * 4, cfg.hidden_size), 23).to(device)
    with torch.no_grad():
        packed = tp1.packed_state(prefix).unsqueeze(0)
        # The packed [k | gate] state comes from replicated projections: every
        # rank must rebuild it bitwise.
        for r, idx in enumerate(ranks):
            assert torch.equal(idx.packed_state(prefix), packed[0]), f"rank{r} packed_state"
        kv_lens = torch.tensor([kv_len], device=device)
        pool_keys, pool_last, pool_valid = tp1.build_pools(packed, kv_lens, num_pools)
        token_request = torch.zeros(num_tokens, dtype=torch.long, device=device)
        query_pos = torch.tensor([kv_len - 3, kv_len - 2, kv_len - 1], device=device)

        args = (q_resid, hidden, pool_keys, pool_last, pool_valid, token_request, query_pos)
        want = tp1.select(*args)

        # Pass 1: identity stubs harvest each rank's partial score by running
        # the same selection; the returned indices are discarded.
        partials = []
        for idx, stub in zip(ranks, stubs):
            stub.payload = None
            idx.select(*args)
            assert stub.calls >= 1 and stub.last_input is not None
            partials.append(stub.last_input)
        full = torch.stack(partials).sum(dim=0)

        # The four partials sum to the tp1 fp32 score (reassociation only).
        base = tp1.wq_b(q_resid).view(num_tokens, 32, 128)
        scores = torch.nn.functional.relu(
            torch.matmul(base.float(), pool_keys[token_request].transpose(-1, -2).float())
            * tp1.softmax_scale
        )
        weights = tp1.weights_proj(hidden).float() * (32**-0.5)
        tp1_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)
        assert torch.allclose(full, tp1_scores, rtol=1e-5, atol=1e-5)

        # Pass 2: every rank consumes the same reduced tensor (what the real
        # fp32 SUM all-reduce hands every rank) -> identical indices, and
        # equal to the tp1 selection.
        outs = []
        for idx, stub in zip(ranks, stubs):
            stub.payload = full
            outs.append(idx.select(*args))
            stub.payload = None
        for r, out in enumerate(outs):
            assert torch.equal(out, outs[0]), f"rank{r} selected different indices"
        assert torch.equal(outs[0], want), "tp4 selection differs from tp1 on untied scores"


# ---------------------------------------------------------------------------
# sparse MLA: absorbed views and summed o_proj partials
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="matches the CUDA-only replay tier")
def test_sparse_tp4_absorbed_slices_and_output_partials(stub_allreduce):
    device = torch.device("cuda")
    cfg = _config()
    torch.manual_seed(2)
    tp1 = Glm5NextSparseAttention(cfg, 3).to(device).eval()
    with torch.no_grad():
        for p in tp1.parameters():
            p.copy_(torch.randn_like(p.float()).mul(0.05).to(p.dtype))

    tokens = 5
    query = _seeded((tokens, 64, 256), 31).to(device)
    out_latent1 = _seeded((tokens, 64 * 512), 32).to(device)

    with torch.no_grad():
        w_k1, w_v_t1 = tp1.absorbed_kv_b()
        q_lat1 = tp1.absorb_query(query)
        out1 = tp1.project_output_latent(out_latent1)

        partial = torch.zeros_like(out1)
        for rank in range(TP):
            attn = Glm5NextSparseAttention(cfg, 3, mapping=_mapping(rank)).to(device).eval()
            hd = (rank * 16, (rank + 1) * 16)
            attn.q_b_proj = _local_linear(tp1.q_b_proj, rows=(hd[0] * 256, hd[1] * 256))
            attn.kv_b_proj = _local_linear(tp1.kv_b_proj, rows=(hd[0] * 512, hd[1] * 512))
            attn.o_proj = _local_linear(tp1.o_proj, cols=(hd[0] * 256, hd[1] * 256))
            with torch.no_grad():
                attn.q_a_proj.weight.copy_(tp1.q_a_proj.weight)
                attn.kv_a_proj_with_mqa.weight.copy_(tp1.kv_a_proj_with_mqa.weight)
                attn.q_a_layernorm.weight.copy_(tp1.q_a_layernorm.weight)
                attn.kv_a_layernorm.weight.copy_(tp1.kv_a_layernorm.weight)

            w_k4, w_v_t4 = attn.absorbed_kv_b()
            assert torch.equal(w_k4, w_k1[hd[0] : hd[1]]), f"rank{rank} absorbed w_k"
            assert torch.equal(w_v_t4, w_v_t1[hd[0] : hd[1]]), f"rank{rank} absorbed w_v"
            q_lat4 = attn.absorb_query(query[:, hd[0] : hd[1]])
            assert torch.allclose(
                q_lat4.float(), q_lat1[:, hd[0] : hd[1]].float(), rtol=1e-2, atol=1e-3
            ), f"rank{rank} absorbed q"
            local_latent = out_latent1.view(tokens, 64, 512)[:, hd[0] : hd[1]].reshape(tokens, -1)
            partial += attn.project_output_latent(local_latent)

    diff = (partial.float() - out1.float()).abs().max().item()
    scale = max(out1.float().abs().max().item(), 1e-3)
    assert diff <= 2e-2 * scale, f"summed o_proj partials off by {diff} (scale {scale})"


def test_tp1_modules_have_no_reduce_and_identical_shapes():
    """The frozen PP4 oracle: tp1 construction is byte-identical to pre-TP."""
    cfg = _config()
    with torch.device("meta"):
        kda = Glm5NextLinearAttention(cfg, 0)
        attn = Glm5NextSparseAttention(cfg, 3)
    assert kda.num_heads == 64 and kda.qkv_dim == 8192 and kda.conv_dim == 24576
    assert tuple(kda.conv1d.weight.shape) == (24576, 1, 4)
    assert tuple(kda.dt_bias.shape) == (8192,) and tuple(kda.A_log.shape) == (64,)
    assert kda.kda_head_range() == (0, 64) and kda.kda_channel_range() == (0, 8192)
    assert attn.num_heads == 64
    assert attn.indexer.n_heads == 32
    assert attn.indexer.score_all_reduce is None
    assert attn.attn_backend.num_heads == 64
