# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime-binding parity: the metadata-driven forward on real runtime state.

Four levels, cheapest first:

* ``test_runtime_context_argument_sourcing`` (CPU-logic, tiny CUDA tensors):
  pins :func:`build_glm5_next_runtime_context` plus the ``linear_kwargs`` /
  ``sparse_kwargs`` derivation against a faithful fake manager for a mixed
  context+generation batch -- including that block tables come from the
  raw-slot accessor and NOT from the scaled ``get_batch_cache_indices``.

* ``test_real_manager_slot_tables_are_raw_base_ids`` (CUDA): allocates real
  requests on a real ``Glm5NextCacheManager`` whose two sparse layers coalesce
  into one V2 pool (page-index scale > 1, the real checkpoint has scale 11)
  and proves ``get_batch_slot_tables`` returns exactly the per-request base
  page ids -- in range for the slot-major latent/index views -- while the
  standard scaled accessor returns stride-multiplied ids that would corrupt
  or overrun those views.

* ``test_runtime_forward_matches_diagnostic`` (CUDA): real request
  allocation, the runtime's own ``AttentionMetadata`` class with
  ``prepare()``, and the public ``Glm5NextForCausalLM.forward`` (embedding,
  metadata-derived runtime context, logits processor) over prefill, two
  decode/cache-reuse steps, and a mixed context+generation step. The
  reference recomputes every step through per-layer ``forward_direct`` with
  slot ids and block tables read from V2's own per-request bookkeeping
  (``kv_cache.get_base_page_indices``), so a wrong index space, wrong
  request ordering, wrong phase split, or wrong logits gather all break the
  bitwise comparison.

* ``test_model_loader_dispatch_reaches_the_audited_loader``: replays the
  runtime ``ModelLoader`` weight-population steps -- generic HF mapper
  initialization plus the signature-inspected ``_call_load_weights``
  dispatch -- against the real-checkpoint meta model, proving the exact
  ModelLoader call reaches GLM's audited loader instead of being rejected
  for an unsupported ``weight_mapper``.
"""

from __future__ import annotations

import copy
import inspect
import os
import sys
from types import SimpleNamespace
from typing import List, Sequence

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_glm5_next import (
    LINEAR_ATTENTION,
    SPARSE_ATTENTION,
    Glm5NextForCausalLM,
    build_glm5_next_runtime_context,
    resolve_glm5_next_schedule,
)

CHECKPOINT = os.environ.get("GLM53_FLASH_CHECKPOINT", "/dev/shm/GLM-5.3-Flash")


# ---------------------------------------------------------------------------
# Level 1 -- argument sourcing
# ---------------------------------------------------------------------------


class _FakeManager:
    """Minimal stand-in exposing exactly what the runtime context reads.

    Every accessor returns a distinctly-valued tensor so the test asserts the
    context selected the right buffer, not merely a same-shaped one. The two
    block-table accessors are deliberately inconsistent: the raw-slot API
    returns the true page rows while ``get_batch_cache_indices`` returns
    scale-multiplied ids (as the real V2 manager does), so the assertions
    below can only pass when the context builder consumes the raw API.
    """

    tokens_per_block = 4

    def __init__(self, device):
        self._conv = {i: torch.full((8, 3), float(i), device=device) for i in (0, 2)}
        self._ssm = {i: torch.full((8, 5), float(i) + 0.5, device=device) for i in (0, 2)}
        # Sparse buffers are [slots, tokens, heads, dim]; the context drops the
        # head axis by indexing [:, :, 0, :].
        self._latent = {1: torch.arange(8 * 6 * 1 * 4, device=device).float().reshape(8, 6, 1, 4)}
        self._index = {
            1: (torch.arange(8 * 6 * 1 * 2, device=device).float() + 100).reshape(8, 6, 1, 2)
        }
        self._pages = {10: [0, 1], 11: [2], 12: [3, 4]}

    def get_conv_states(self, layer_idx):
        return self._conv[layer_idx]

    def get_ssm_states(self, layer_idx):
        return self._ssm[layer_idx]

    def get_latent_state_buffer(self, layer_idx):
        return self._latent[layer_idx]

    def get_index_state_buffer(self, layer_idx):
        return self._index[layer_idx]

    def get_batch_slot_tables(self, request_ids):
        return [self._pages[r] for r in request_ids]

    def get_batch_cache_indices(self, request_ids, layer_idx=None):
        # Scaled ids, as V2's standard accessor returns them (scale 11 on the
        # real checkpoint). Consuming these for the slot-major views is the
        # iteration-14 defect; the block-table assertions reject them.
        return [[p * 11 for p in self._pages[r]] for r in request_ids]


def _fake_metadata(manager, seq_lens, num_contexts, cached, request_ids, slots, device):
    mamba = SimpleNamespace(state_indices=torch.tensor(slots, dtype=torch.long, device=device))
    kv_params = SimpleNamespace(num_cached_tokens_per_seq=list(cached))
    return SimpleNamespace(
        kv_cache_manager=manager,
        mamba_metadata=mamba,
        seq_lens=torch.tensor(seq_lens, dtype=torch.long),
        num_contexts=num_contexts,
        num_ctx_tokens=int(sum(seq_lens[:num_contexts])),
        request_ids=list(request_ids),
        kv_cache_params=kv_params,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="context builds a CUDA block table")
def test_runtime_context_argument_sourcing():
    """One context request (len 5) + two decodes, mixed slots and page tables."""
    device = torch.device("cuda")
    manager = _FakeManager(device)
    seq_lens = [5, 1, 1]
    cached = [0, 7, 3]
    request_ids = [10, 11, 12]
    slots = [4, 2, 9]  # deliberately not arange: the context must use these
    metadata = _fake_metadata(
        manager,
        seq_lens,
        num_contexts=1,
        cached=cached,
        request_ids=request_ids,
        slots=slots,
        device=device,
    )

    ctx = build_glm5_next_runtime_context(metadata)
    assert ctx.num_contexts == 1
    assert ctx.num_generations == 2
    assert ctx.num_ctx_tokens == 5
    assert ctx.ctx_cu_seqlens == [0, 5]
    assert ctx.cached_lens == [0, 7, 3]
    # kv_lens: cached + this step's tokens, as a device tensor (the decode
    # path may only consume device values so CUDA-graph replays see fresh
    # lengths).
    assert ctx.kv_lens.device.type == "cuda"
    assert ctx.kv_lens.tolist() == [5, 8, 4]
    assert ctx.state_indices.tolist() == [4, 2, 9]

    # Linear layer 0, prefill: slot_ids are the context slice of state_indices.
    lp = ctx.linear_kwargs(0, "prefill")
    assert lp["slot_ids"].tolist() == [4]
    assert lp["cu_seqlens"] == [0, 5]
    assert lp["cached_lens"] == [0]
    assert torch.equal(lp["conv_pool"], manager.get_conv_states(0))
    assert torch.equal(lp["ssm_pool"], manager.get_ssm_states(0))

    # Linear layer 0, decode: slot_ids are the generation slice.
    ld = ctx.linear_kwargs(0, "decode")
    assert ld["slot_ids"].tolist() == [2, 9]
    assert "cu_seqlens" not in ld

    # Sparse layers receive schedule values plus the metadata itself; the
    # backend derives every pool/table/length from it.
    sp = ctx.sparse_kwargs(1, "prefill")
    assert sp["metadata"] is metadata
    assert sp["cached_lens"] == [0]
    assert sp["cu_seqlens"] == [0, 5]
    assert set(sp) == {"metadata", "cached_lens", "cu_seqlens"}

    sd = ctx.sparse_kwargs(1, "decode")
    assert sd["metadata"] is metadata
    assert sd["kv_lens"].tolist() == [8, 4]
    assert set(sd) == {"metadata", "kv_lens"}

    # The backend's own metadata-derived cache state carries the protections
    # the context used to: RAW slot rows (the scaled get_batch_cache_indices
    # values, x11, would fail every table assertion -- the iteration-14
    # defect), full-batch tables padded to the widest request, device
    # kv_lens, and the slot-major pool views with the head axis dropped.
    from tensorrt_llm._torch.attention_backend.sparse.glm_kpool import (
        GlmKpoolSparseAttention,
        GlmKpoolSparseParams,
    )

    backend = GlmKpoolSparseAttention(
        1,
        1,
        4,
        num_kv_heads=1,
        sparse_params=GlmKpoolSparseParams(
            kv_lora_rank=4,
            qk_nope_head_dim=4,
            q_lora_rank=4,
            v_head_dim=4,
            index_topk=8,
            index_kpool=4,
        ),
    )
    state = backend._cache_state(metadata)
    assert state.tokens_per_block == 4
    assert state.num_contexts == 1
    assert state.block_tables[0].tolist() == [0, 1]
    assert state.block_tables[1].tolist() == [2, 0]
    assert state.block_tables[2].tolist() == [3, 4]
    assert state.kv_lens.device.type == "cuda"
    assert state.kv_lens.tolist() == [5, 8, 4]
    assert state.latent_pool.shape == (8, 6, 4)
    assert torch.equal(state.latent_pool, manager.get_latent_state_buffer(1)[:, :, 0, :])
    assert torch.equal(state.index_pool, manager.get_index_state_buffer(1)[:, :, 0, :])


def test_runtime_context_requires_prepared_metadata():
    """A context without mamba_metadata is a loud error, not a silent None."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manager = _FakeManager(device)
    metadata = _fake_metadata(manager, [1], 0, [0], [10], [0], device)
    metadata.mamba_metadata = None
    with pytest.raises(ValueError, match="mamba_metadata"):
        build_glm5_next_runtime_context(metadata)


# ---------------------------------------------------------------------------
# Shared small-model fixtures
# ---------------------------------------------------------------------------


def _small_glm5_next_config():
    """A tiny but structurally faithful GLM-5.3-Flash text config.

    Six layers cover every combination that occurs: linear/sparse attention
    and dense/sparse MLP, with the first three MLPs dense
    (``first_k_dense_replace=3``). The vocab and routed-expert count are
    shrunk; every ratio the modules assert (kv/q rank, head dims, kpool,
    hc_mult) is left at the checkpoint's values.
    """
    from transformers import AutoConfig

    if not os.path.isdir(CHECKPOINT):
        pytest.skip(f"requires the real config at {CHECKPOINT}")
    base = AutoConfig.from_pretrained(CHECKPOINT)
    text = copy.deepcopy(base.text_config)

    text.num_hidden_layers = 6
    text.layer_types = [
        LINEAR_ATTENTION,
        LINEAR_ATTENTION,
        LINEAR_ATTENTION,
        SPARSE_ATTENTION,
        LINEAR_ATTENTION,
        SPARSE_ATTENTION,
    ]
    text.mlp_layer_types = ["dense", "dense", "dense", "sparse", "sparse", "sparse"]
    text.first_k_dense_replace = 3
    if getattr(text, "linear_attn_config", None):
        text.linear_attn_config = dict(text.linear_attn_config)
        text.linear_attn_config["kda_layers"] = [0, 1, 2, 4]
        text.linear_attn_config["full_attn_layers"] = [3, 5]
    text.vocab_size = 512
    for attr in ("n_routed_experts", "num_experts"):
        if hasattr(text, attr):
            setattr(text, attr, 16)

    base.text_config = text
    base.num_hidden_layers = text.num_hidden_layers
    return base


def _build_small_cache_manager(
    config, schedule, device, max_seq_len, tokens_per_block=64, max_batch_size=1, mapping=None
):
    from tensorrt_llm._torch.models.modeling_glm5_next import glm5_next_cache_manager_cls
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    text = config.text_config
    linear = dict(text.linear_attn_config)
    attention = list(text.layer_types)
    mamba_mask = [t == LINEAR_ATTENTION for t in attention]
    sparse_mask = [t == SPARSE_ATTENTION for t in attention]
    sparse_ids = [i for i, on in enumerate(sparse_mask) if on]

    manager_cls = glm5_next_cache_manager_cls()
    with torch.cuda.device(device):
        return manager_cls(
            mamba_d_state=int(linear["head_dim"]),
            mamba_d_conv=int(linear["short_conv_kernel_size"]),
            mamba_num_heads=int(linear["num_heads"]),
            mamba_n_groups=int(linear["num_heads"]),
            mamba_head_dim=int(linear["head_dim"]),
            mamba_num_layers=sum(mamba_mask),
            mamba_layer_mask=mamba_mask,
            mamba_cache_dtype=torch.bfloat16,
            mamba_ssm_cache_dtype=torch.float32,
            kv_cache_config=KvCacheConfig(
                max_tokens=8 * max_seq_len * max_batch_size, enable_block_reuse=False
            ),
            kv_cache_type=CacheTypeCpp.SELFKONLY,
            num_layers=max(sum(sparse_mask), 1),
            num_kv_heads=1,
            head_dim=int(text.kv_lora_rank),
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=mapping or Mapping(world_size=1, tp_size=1, pp_size=1),
            layer_mask=sparse_mask,
            dtype=DataType.BF16,
            conv_state_layout="q_k_v",
            sparse_layer_ids=sparse_ids,
            index_state_dim=2 * int(text.index_head_dim),
        )


def _zero_cache(manager, schedule):
    for layer_id, kind in enumerate(schedule.attention):
        # Non-local layers on a pipeline-parallel rank have no state here.
        try:
            if kind == LINEAR_ATTENTION:
                conv = manager.get_conv_states(layer_id)
                ssm = manager.get_ssm_states(layer_id)
                if conv is not None:
                    conv.zero_()
                if ssm is not None:
                    ssm.zero_()
            else:
                latent = manager.get_latent_state_buffer(layer_id)
                index = manager.get_index_state_buffer(layer_id)
                if latent is not None:
                    latent.zero_()
                if index is not None:
                    index.zero_()
        except (KeyError, IndexError):
            continue


def _raw_slot_rows(manager, request_ids: Sequence[int]) -> List[List[int]]:
    """Ground-truth base page ids straight from V2's per-request bookkeeping.

    Independent of ``get_batch_slot_tables``: reads each request's
    ``get_base_page_indices`` for the sparse layer group, which is the value
    the slot-major latent/index views are defined over.
    """
    pool_id = manager._sparse_pool_id()
    rows = []
    for rid in request_ids:
        kv_cache = manager.kv_cache_map[rid]
        rows.append(
            [int(p) for p in kv_cache.get_base_page_indices(pool_id)[: kv_cache.num_blocks]]
        )
    return rows


# ---------------------------------------------------------------------------
# Level 2 -- real-manager slot tables (the iteration-14 index-space defect)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_manager_slot_tables_are_raw_base_ids():
    """Raw slot ids on a real coalesced-pool manager with scale > 1.

    Two sparse layers of identical geometry coalesce into one V2 pool, so the
    per-layer page-index scale exceeds kv_factor exactly as on the real
    checkpoint (scale 11 there). The runtime block tables must carry the raw
    base ids: the scaled accessor's values land on the wrong slot (cross-
    request corruption) and can exceed the slot-major views' bounds.
    """
    device = torch.device("cuda")
    config = _small_glm5_next_config()
    schedule = resolve_glm5_next_schedule(config)
    tokens_per_block = 64
    manager = _build_small_cache_manager(
        config,
        schedule,
        device,
        max_seq_len=256,
        tokens_per_block=tokens_per_block,
        max_batch_size=3,
    )
    try:
        request_ids = [0, 1, 2]
        # 100/40/70 tokens -> 2/1/2 blocks at 64 tokens per block.
        manager.add_dummy_requests(request_ids, token_nums=[100, 40, 70])

        sparse_ids = manager.sparse_layer_ids
        assert len(sparse_ids) >= 2
        kv_factor = int(manager.kv_factor)
        scale = manager.get_layer_page_index_scale(sparse_ids[0])
        stride = scale // kv_factor
        assert stride > 1, (
            f"coalesced pool expected: scale={scale}, kv_factor={kv_factor}; "
            "without stride > 1 this regression cannot distinguish raw from scaled ids"
        )

        raw = manager.get_batch_slot_tables(request_ids)
        ground_truth = _raw_slot_rows(manager, request_ids)
        assert raw == ground_truth

        # Multi-block coverage and distinct slots across requests.
        assert [len(r) for r in raw] == [2, 1, 2]
        flat = [p for row in raw for p in row]
        assert len(set(flat)) == len(flat), f"slot rows overlap: {raw}"

        # The standard accessor is the scaled view -- stride x the raw ids --
        # which is exactly what the runtime context must NOT consume.
        scaled = manager.get_batch_cache_indices(request_ids, layer_idx=sparse_ids[0])
        assert scaled == [[p * stride for p in row] for row in raw]
        assert scaled != raw

        # Raw ids address the slot-major views (in bounds); every nonzero
        # scaled id lands on a different slot than intended, which is the
        # cross-request corruption mode, and scaled ids in the top 1/stride of
        # the slot space would run past the views entirely.
        latent = manager.get_latent_state_buffer(sparse_ids[0])
        index = manager.get_index_state_buffer(sparse_ids[0])
        assert latent.shape[0] == index.shape[0]
        max_raw = max(flat)
        assert max_raw < latent.shape[0]
        for raw_row, scaled_row in zip(raw, scaled):
            for raw_id, scaled_id in zip(raw_row, scaled_row):
                if raw_id != 0:
                    assert scaled_id != raw_id
        # Cross-layer isolation through the slot-major views at a real slot id:
        # writing layer A's slot must not appear in layer B's same slot.
        slot = raw[0][0]
        for lid in sparse_ids[:2]:
            manager.get_latent_state_buffer(lid).zero_()
        manager.get_latent_state_buffer(sparse_ids[0])[slot].fill_(1.0)
        assert float(manager.get_latent_state_buffer(sparse_ids[1])[slot].abs().max()) == 0.0
        assert float(manager.get_latent_state_buffer(sparse_ids[0])[slot].abs().max()) == 1.0
    finally:
        manager.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_state_digest_exports_per_step_decode_state(monkeypatch, tmp_path):
    """GLM53_STATE_DIGEST_DIR makes prepare() export per-step state digests.

    On a real small ``Glm5NextCacheManager`` with one live request, a
    pure-decode ``prepare()`` appends exactly one rank record digesting every
    local KDA conv/ssm slot and every sparse layer's latent/index pages over
    the cached positions. The digest is deterministic for unchanged state,
    flips for exactly the touched state family, skips prefill batches, and
    writes nothing while the env var is unset (the production default).
    """
    import json

    from tensorrt_llm._torch.models.modeling_glm5_next import glm5_next_mamba_metadata_cls

    device = torch.device("cuda")
    config = _small_glm5_next_config()
    schedule = resolve_glm5_next_schedule(config)
    manager = _build_small_cache_manager(
        config, schedule, device, max_seq_len=256, tokens_per_block=64, max_batch_size=2
    )
    try:
        _zero_cache(manager, schedule)
        manager.add_dummy_requests([7], token_nums=[100])
        md = glm5_next_mamba_metadata_cls()(max_batch_size=2, chunk_size=128)

        def attn_md(cached, num_contexts=0):
            # Decode: one new token on `cached` history. Prefill: `cached` is 0
            # and the whole prompt is this step's tokens (consistent seq_lens /
            # num_ctx_tokens, or the base prepare()'s seq_idx build asserts).
            tokens = 1 if num_contexts == 0 else 100
            seq = torch.tensor([tokens], dtype=torch.int)
            return SimpleNamespace(
                kv_cache_manager=manager,
                kv_cache_params=SimpleNamespace(
                    num_cached_tokens_per_seq=[cached if num_contexts == 0 else 0]
                ),
                request_ids=[7],
                seq_lens=seq,
                seq_lens_cuda=seq.to(device),
                num_contexts=num_contexts,
                num_ctx_tokens=0 if num_contexts == 0 else tokens,
            )

        digest_dir = tmp_path / "digests"
        digest_dir.mkdir()

        # Default (env unset): prepare() exports nothing.
        md.prepare(attn_md(100))
        assert not list(digest_dir.iterdir())

        monkeypatch.setenv("GLM53_STATE_DIGEST_DIR", str(digest_dir))
        md.prepare(attn_md(100))
        (rank_file,) = list(digest_dir.iterdir())
        assert rank_file.name == "rank0.jsonl"

        def last_record():
            return json.loads(rank_file.read_text().splitlines()[-1])

        rec = last_record()
        assert rec["cached"] == [100] and rec["batch"] == 1 and rec["num_contexts"] == 0
        assert sorted(k for k in rec["layers"] if k.startswith("kda")) == [
            "kda0",
            "kda1",
            "kda2",
            "kda4",
        ]
        assert sorted(k for k in rec["layers"] if k.startswith("sparse")) == [
            "sparse3",
            "sparse5",
        ]
        for key in ("kda0", "kda1", "kda2", "kda4"):
            for part in ("conv", "ssm"):
                d = rec["layers"][key][part]
                assert len(d["sha256"]) == 64 and d["numel"] > 0
                assert all(f in d for f in ("sum", "abs_sum", "max", "min"))
        latent_dim = manager.get_latent_state_buffer(3).shape[-1]
        for key in ("sparse3", "sparse5"):
            for part in ("latent", "index"):
                d = rec["layers"][key][part][0]
                assert len(d["sha256"]) == 64 and d["numel"] > 0
        # All 100 cached positions are digested, at the latent width.
        assert rec["layers"]["sparse3"]["latent"][0]["numel"] == 100 * latent_dim

        # Deterministic while the state is unchanged.
        md.prepare(attn_md(100))
        assert last_record()["layers"] == rec["layers"]

        # A KDA conv write flips exactly that family's digest.
        manager.get_conv_states(0)[rec["mamba_slots"][0]].add_(1.0)
        md.prepare(attn_md(100))
        rec3 = last_record()
        assert rec3["layers"]["kda0"]["conv"]["sha256"] != rec["layers"]["kda0"]["conv"]["sha256"]
        assert rec3["layers"]["kda0"]["ssm"] == rec["layers"]["kda0"]["ssm"]
        assert rec3["layers"]["sparse3"] == rec["layers"]["sparse3"]

        # A latent page write inside the cached window flips the sparse digest.
        manager.get_latent_state_buffer(3)[rec["pages"][0][0], 0].add_(1.0)
        md.prepare(attn_md(100))
        rec4 = last_record()
        assert (
            rec4["layers"]["sparse3"]["latent"][0]["sha256"]
            != rec3["layers"]["sparse3"]["latent"][0]["sha256"]
        )
        assert rec4["layers"]["sparse3"]["index"] == rec3["layers"]["sparse3"]["index"]

        # Prefill batches are skipped: state for a not-yet-written request
        # would digest uninitialized memory.
        lines_before = len(rank_file.read_text().splitlines())
        md.prepare(attn_md(100, num_contexts=1))
        assert len(rank_file.read_text().splitlines()) == lines_before
    finally:
        manager.shutdown()


# ---------------------------------------------------------------------------
# Level 3 -- full runtime path parity (CUDA, small random-weight model)
# ---------------------------------------------------------------------------


def _reference_step_logits(
    model,
    manager,
    schedule,
    *,
    ctx_tokens: List[torch.Tensor],
    gen_tokens: torch.Tensor,
    request_ids: Sequence[int],
    cached: Sequence[int],
    tokens_per_block: int,
    device,
) -> torch.Tensor:
    """Per-layer ``forward_direct`` reference for one executor step.

    Mirrors the executor packing (contexts first, then one token per
    generation request) but derives every cache argument independently of
    ``build_glm5_next_runtime_context``: slot ids come from
    ``manager.get_state_indices`` per request and block tables from
    ``kv_cache.get_base_page_indices`` -- V2's own bookkeeping -- so the
    runtime path's metadata-to-context conversion is checked against ground
    truth rather than against itself.
    """
    inner = model.model
    num_ctx = len(ctx_tokens)
    num_gen = int(gen_tokens.numel())
    rows = _raw_slot_rows(manager, request_ids)
    width = max(len(r) for r in rows)
    table = torch.zeros(len(rows), width, dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        table[i, : len(row)] = torch.as_tensor(row, dtype=torch.long, device=device)
    slots = torch.tensor(
        [manager.get_state_indices([rid])[0] for rid in request_ids],
        dtype=torch.long,
        device=device,
    )
    cached = [int(c) for c in cached]

    ctx_cu = [0]
    for t in ctx_tokens:
        ctx_cu.append(ctx_cu[-1] + int(t.numel()))
    n_ctx_tokens = ctx_cu[-1]
    pieces = list(ctx_tokens) + ([gen_tokens] if num_gen else [])
    tokens = torch.cat(pieces) if pieces else gen_tokens
    lens = [int(t.numel()) for t in ctx_tokens] + [1] * num_gen

    # Independent metadata carrier for the sparse layers: tables come from
    # V2's own per-request bookkeeping (_raw_slot_rows) rather than from
    # build_glm5_next_runtime_context / Glm5NextMamba2Metadata.prepare, so the
    # runtime path's derivation is checked against ground truth.
    kv_lens_dev = torch.tensor(
        [c + n for c, n in zip(cached, lens)], device=device, dtype=torch.long
    )
    ref_md = SimpleNamespace(
        kv_cache_manager=manager,
        mamba_metadata=SimpleNamespace(glm_block_tables=table, glm_kv_lens=kv_lens_dev),
        seq_lens=torch.tensor(lens, dtype=torch.long),
        num_contexts=num_ctx,
        is_cuda_graph=False,
    )

    with torch.inference_mode(), torch.cuda.device(device):
        streams = inner.expand_streams(inner.embed_tokens(tokens))
        for layer_id, layer in enumerate(inner.layers):
            linear = schedule.attention[layer_id] == LINEAR_ATTENTION
            parts = []
            if num_ctx:
                if linear:
                    kwargs = dict(
                        slot_ids=slots[:num_ctx],
                        conv_pool=manager.get_conv_states(layer_id),
                        ssm_pool=manager.get_ssm_states(layer_id),
                        cu_seqlens=ctx_cu,
                        cached_lens=cached[:num_ctx],
                    )
                else:
                    kwargs = dict(
                        cached_lens=cached[:num_ctx],
                        cu_seqlens=ctx_cu,
                        metadata=ref_md,
                    )
                parts.append(
                    layer.forward_direct(streams[:n_ctx_tokens], phase="prefill", **kwargs)
                )
            if num_gen:
                if linear:
                    kwargs = dict(
                        slot_ids=slots[num_ctx:],
                        conv_pool=manager.get_conv_states(layer_id),
                        ssm_pool=manager.get_ssm_states(layer_id),
                    )
                else:
                    # kv_lens: cached + the one token being decoded, as a
                    # device tensor (the batched decode's graph contract).
                    kwargs = dict(kv_lens=kv_lens_dev[num_ctx:], metadata=ref_md)
                parts.append(layer.forward_direct(streams[n_ctx_tokens:], phase="decode", **kwargs))
            streams = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        hidden = inner.collapse_streams(streams)
        # Mirror LogitsProcessor exactly: gather last tokens, then lm_head, then float.
        last = torch.cumsum(torch.tensor(lens, dtype=torch.long, device=device), dim=0) - 1
        return model.lm_head(hidden[last]).float()


def _runtime_step_logits(
    model,
    manager,
    metadata_cls,
    *,
    lens: Sequence[int],
    num_contexts: int,
    cached: Sequence[int],
    request_ids: Sequence[int],
    prompt_lens: Sequence[int],
    tokens: torch.Tensor,
    max_num_requests: int,
    device,
) -> torch.Tensor:
    """One step through the public runtime path: metadata.prepare + forward."""
    from tensorrt_llm._torch.metadata import KVCacheParams

    attn_metadata = metadata_cls(
        seq_lens=torch.tensor(list(lens), dtype=torch.int),
        num_contexts=num_contexts,
        kv_cache_params=KVCacheParams(
            use_cache=True, num_cached_tokens_per_seq=[int(c) for c in cached]
        ),
        kv_cache_manager=manager,
        request_ids=list(request_ids),
        prompt_lens=list(prompt_lens),
        max_num_requests=max_num_requests,
        max_num_tokens=8192,
    )
    position_ids = torch.cat(
        [torch.arange(int(c), int(c) + int(n), device=device) for c, n in zip(cached, lens)]
    ).unsqueeze(0)
    with torch.inference_mode(), torch.cuda.device(device):
        attn_metadata.prepare()
        return model.forward(
            attn_metadata=attn_metadata, input_ids=tokens, position_ids=position_ids
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_runtime_forward_matches_diagnostic():
    """Public forward on prepared real metadata == ground-truth forward_direct.

    Covers prefill (two contexts, one spanning two blocks), two decode/cache-
    reuse steps, and a mixed context+generation step, on one real
    ``Glm5NextCacheManager`` with really-allocated requests. Bitwise equality
    (atol=rtol=0) at every step: both paths run the same kernels over the
    same shapes, so any difference is an argument-sourcing defect.
    """
    from tensorrt_llm._torch.attention_backend.utils import get_attention_backend

    torch.manual_seed(0)
    device = torch.device("cuda")
    config = _small_glm5_next_config()
    # Unquantized (bf16) build: both paths see identical random weights, and the
    # test asserts argument-sourcing/structure parity, not accuracy (FP8
    # numerics are covered by the Stage-1 source-replay tests).
    model_config = ModelConfig(pretrained_config=config)
    model = Glm5NextForCausalLM(model_config).to(device).eval()
    assert model.quantized is False
    # Several TRT-LLM-managed weights (LMHead, Embedding) and the MoE expert
    # parameters are built with uninitialized ``torch.empty`` (the experts
    # deliberately so — MetaInitMode routes ``empty`` to meta, which is what
    # keeps 8-rank construction host-OOM-free). Recycled allocator pages can
    # contain NaN bit patterns, so the constructor seed alone does not give a
    # finite model: overwrite every floating parameter deterministically, the
    # same way the PP wire-parity test below does.
    generator = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            if p.is_floating_point():
                p.normal_(0.0, 0.02, generator=generator)
    for p in model.parameters():
        p.requires_grad_(False)

    schedule = resolve_glm5_next_schedule(config)
    tokens_per_block = 64
    vocab = int(config.text_config.vocab_size)
    # Request 0 spans two blocks (70 > 64); request 2 arrives later as the
    # context of the mixed step.
    prompt = {
        0: torch.randint(0, vocab, (70,), device=device),
        1: torch.randint(0, vocab, (40,), device=device),
        2: torch.randint(0, vocab, (9,), device=device),
    }

    manager = _build_small_cache_manager(
        config,
        schedule,
        device,
        max_seq_len=256,
        tokens_per_block=tokens_per_block,
        max_batch_size=3,
    )
    metadata_cls = get_attention_backend(model_config.attn_backend).Metadata
    empty = torch.empty(0, dtype=prompt[0].dtype, device=device)
    try:
        # Allocate capacity for prompt + 3 decode steps up front.
        manager.add_dummy_requests([0, 1, 2], token_nums=[73, 43, 9])

        def reference_pass():
            _zero_cache(manager, schedule)
            steps = {}
            steps["A"] = _reference_step_logits(
                model,
                manager,
                schedule,
                ctx_tokens=[prompt[0], prompt[1]],
                gen_tokens=empty,
                request_ids=[0, 1],
                cached=[0, 0],
                tokens_per_block=tokens_per_block,
                device=device,
            )
            t_b1 = steps["A"].argmax(dim=-1)
            steps["B1"] = _reference_step_logits(
                model,
                manager,
                schedule,
                ctx_tokens=[],
                gen_tokens=t_b1,
                request_ids=[0, 1],
                cached=[70, 40],
                tokens_per_block=tokens_per_block,
                device=device,
            )
            t_b2 = steps["B1"].argmax(dim=-1)
            steps["B2"] = _reference_step_logits(
                model,
                manager,
                schedule,
                ctx_tokens=[],
                gen_tokens=t_b2,
                request_ids=[0, 1],
                cached=[71, 41],
                tokens_per_block=tokens_per_block,
                device=device,
            )
            t_c = steps["B2"].argmax(dim=-1)
            steps["C"] = _reference_step_logits(
                model,
                manager,
                schedule,
                ctx_tokens=[prompt[2]],
                gen_tokens=t_c,
                request_ids=[2, 0, 1],
                cached=[0, 72, 42],
                tokens_per_block=tokens_per_block,
                device=device,
            )
            return steps, (t_b1, t_b2, t_c)

        reference, (t_b1, t_b2, t_c) = reference_pass()

        # Runtime pass over the same slots/blocks from a re-zeroed cache; the
        # decode inputs are the reference's greedy tokens so every step feeds
        # both paths identical ids.
        _zero_cache(manager, schedule)
        runtime = {}
        runtime["A"] = _runtime_step_logits(
            model,
            manager,
            metadata_cls,
            lens=[70, 40],
            num_contexts=2,
            cached=[0, 0],
            request_ids=[0, 1],
            prompt_lens=[70, 40],
            tokens=torch.cat([prompt[0], prompt[1]]),
            max_num_requests=3,
            device=device,
        )
        runtime["B1"] = _runtime_step_logits(
            model,
            manager,
            metadata_cls,
            lens=[1, 1],
            num_contexts=0,
            cached=[70, 40],
            request_ids=[0, 1],
            prompt_lens=[70, 40],
            tokens=t_b1,
            max_num_requests=3,
            device=device,
        )
        runtime["B2"] = _runtime_step_logits(
            model,
            manager,
            metadata_cls,
            lens=[1, 1],
            num_contexts=0,
            cached=[71, 41],
            request_ids=[0, 1],
            prompt_lens=[70, 40],
            tokens=t_b2,
            max_num_requests=3,
            device=device,
        )
        runtime["C"] = _runtime_step_logits(
            model,
            manager,
            metadata_cls,
            lens=[9, 1, 1],
            num_contexts=1,
            cached=[0, 72, 42],
            request_ids=[2, 0, 1],
            prompt_lens=[9, 70, 40],
            tokens=torch.cat([prompt[2], t_c]),
            max_num_requests=3,
            device=device,
        )
    finally:
        manager.shutdown()

    for step in ("A", "B1", "B2", "C"):
        assert torch.isfinite(runtime[step]).all(), f"non-finite runtime logits at step {step}"
        torch.testing.assert_close(
            runtime[step],
            reference[step],
            rtol=0.0,
            atol=0.0,
            msg=f"runtime forward diverged from the forward_direct reference at step {step}",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_runtime_forward_production_kda_prefill_at_scale():
    """The public runtime path drives the *production* KDA prefill kernel.

    ``test_runtime_forward_matches_diagnostic`` uses 70/40-token prompts, which
    sit below the CuTe scheduler's 4-chunk launch floor and route to the torch
    fallback. Here the packed context batch (256 + 200 tokens = 8 chunks) makes
    every linear layer take ``trtllm::kda_prefill`` against the real
    ``Glm5NextCacheManager`` pools through really prepared
    ``Glm5NextMamba2Metadata`` — asserted per layer, not assumed — and one
    decode/cache-reuse step follows on the carried state. Reference and runtime
    run the same kernels over the same batch composition, so equality is
    bitwise, exactly as in the diagnostic-parity test above.
    """
    from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
    from tensorrt_llm._torch.models.modeling_glm5_next import Glm5NextLinearAttention

    torch.manual_seed(0)
    device = torch.device("cuda")
    config = _small_glm5_next_config()
    model_config = ModelConfig(pretrained_config=config)
    model = Glm5NextForCausalLM(model_config).to(device).eval()
    generator = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            if p.is_floating_point():
                p.normal_(0.0, 0.02, generator=generator)
    for p in model.parameters():
        p.requires_grad_(False)

    schedule = resolve_glm5_next_schedule(config)
    vocab = int(config.text_config.vocab_size)
    prompt = {
        0: torch.randint(0, vocab, (256,), device=device),
        1: torch.randint(0, vocab, (200,), device=device),
    }
    linear_modules = [
        layer.self_attn
        for layer in model.model.layers
        if isinstance(layer.self_attn, Glm5NextLinearAttention)
    ]
    assert linear_modules, "small config must contain linear-attention layers"

    manager = _build_small_cache_manager(
        config,
        schedule,
        device,
        max_seq_len=512,
        tokens_per_block=64,
        max_batch_size=2,
    )
    metadata_cls = get_attention_backend(model_config.attn_backend).Metadata
    empty = torch.empty(0, dtype=prompt[0].dtype, device=device)
    try:
        manager.add_dummy_requests([0, 1], token_nums=[260, 204])

        _zero_cache(manager, schedule)
        ref_prefill = _reference_step_logits(
            model,
            manager,
            schedule,
            ctx_tokens=[prompt[0], prompt[1]],
            gen_tokens=empty,
            request_ids=[0, 1],
            cached=[0, 0],
            tokens_per_block=64,
            device=device,
        )
        ref_paths = [m.last_prefill_path for m in linear_modules]
        t_next = ref_prefill.argmax(dim=-1)
        ref_decode = _reference_step_logits(
            model,
            manager,
            schedule,
            ctx_tokens=[],
            gen_tokens=t_next,
            request_ids=[0, 1],
            cached=[256, 200],
            tokens_per_block=64,
            device=device,
        )

        _zero_cache(manager, schedule)
        run_prefill = _runtime_step_logits(
            model,
            manager,
            metadata_cls,
            lens=[256, 200],
            num_contexts=2,
            cached=[0, 0],
            request_ids=[0, 1],
            prompt_lens=[256, 200],
            tokens=torch.cat([prompt[0], prompt[1]]),
            max_num_requests=2,
            device=device,
        )
        run_paths = [m.last_prefill_path for m in linear_modules]
        run_decode = _runtime_step_logits(
            model,
            manager,
            metadata_cls,
            lens=[1, 1],
            num_contexts=0,
            cached=[256, 200],
            request_ids=[0, 1],
            prompt_lens=[256, 200],
            tokens=t_next,
            max_num_requests=2,
            device=device,
        )
    finally:
        manager.shutdown()

    assert ref_paths == ["trtllm::kda_prefill"] * len(linear_modules), ref_paths
    assert run_paths == ["trtllm::kda_prefill"] * len(linear_modules), run_paths
    for name, got, want in (
        ("prefill", run_prefill, ref_prefill),
        ("decode", run_decode, ref_decode),
    ):
        assert torch.isfinite(got).all(), f"non-finite runtime logits at {name}"
        torch.testing.assert_close(
            got,
            want,
            rtol=0.0,
            atol=0.0,
            msg=f"runtime production-KDA forward diverged from forward_direct at {name}",
        )


# ---------------------------------------------------------------------------
# Level 4 -- the exact ModelLoader dispatch (iteration-14 defect 1)
# ---------------------------------------------------------------------------


def test_model_loader_dispatch_reaches_the_audited_loader():
    """The runtime's ModelLoader call sequence must reach GLM's loader body.

    ``ModelLoader`` always initializes a weight mapper (the generic HF one
    when no GLM-specific mapper is registered) and ``_call_load_weights``
    passes it to any ``load_weights`` whose signature names ``weight_mapper``.
    GLM's loader places keys by its own audit, so it must not advertise that
    parameter -- otherwise every real checkpoint load raises before weight
    placement. This regression replays those exact runtime steps on the
    meta-device real-config model (no weights materialized).
    """
    from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader import HfCheckpointLoader
    from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM
    from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader

    if not os.path.isdir(CHECKPOINT):
        pytest.skip(f"requires the real config at {CHECKPOINT}")

    model_config = ModelConfig.from_pretrained(CHECKPOINT)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(model_config)
    assert type(model).__name__ == "Glm5NextForCausalLM"

    # model_loader.py step 1: the mapper is always initialized, and for GLM it
    # resolves to the generic HF mapper (non-None).
    checkpoint_loader = HfCheckpointLoader()
    mapper = checkpoint_loader.get_initialized_weight_mapper(model, model_config)
    assert mapper is not None

    # model_loader.py step 2: _call_load_weights inspects the signature; the
    # audited loader must not advertise weight_mapper, or it would be handed
    # the generic mapper on every real load.
    assert "weight_mapper" not in inspect.getfullargspec(model.load_weights).args

    # The dispatched call reaches the audited loader body: the complaint is
    # about the (deliberately bogus) checkpoint content, not about a mapper.
    with pytest.raises(ValueError) as excinfo:
        ModelLoader._call_load_weights(
            None, model.load_weights, {"not.a.real.key": torch.empty(0)}, mapper
        )
    message = str(excinfo.value)
    assert "weight_mapper is not supported" not in message
    assert message.startswith("glm5_next"), message

    # An explicitly passed mapper is still rejected loudly, not swallowed.
    with pytest.raises(ValueError, match="weight_mapper is not supported"):
        model.load_weights({}, weight_mapper=mapper)


# ---------------------------------------------------------------------------
# Level 5 -- pipeline parallelism (runtime increment R4's model-side wiring)
# ---------------------------------------------------------------------------


def test_quant_swap_skips_pp_pruned_modules():
    """The block-FP8 swap must not touch modules pruned by `__pp_init__`.

    `remove_weights` clears a pruned module's parameter dict, which makes even
    `module.bias` raise AttributeError -- so a swap that visited pruned layers
    would crash at real-checkpoint load time (and would otherwise allocate
    weights for layers this rank never runs).
    """
    from torch import nn

    from tensorrt_llm._torch.models.modeling_glm5_next import glm5_next_swap_quantized_projections
    from tensorrt_llm._torch.models.modeling_utils import remove_weights

    config = _small_glm5_next_config()
    torch.manual_seed(0)
    model = Glm5NextForCausalLM(ModelConfig(pretrained_config=config))

    pruned_layer = model.model.layers[3]
    pruned_linears = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.startswith("model.layers.3.")
    ]
    assert pruned_linears
    remove_weights(pruned_layer)
    # Sanity: a pruned Linear's cleared parameter dict makes .bias raise --
    # this is exactly what an unskipped swap would trip over.
    with pytest.raises(AttributeError):
        _ = model.get_submodule(pruned_linears[0]).bias

    plan = {name: True for name in pruned_linears}
    placed = glm5_next_swap_quantized_projections(model, plan)
    # Nothing under the pruned layer was replaced or reported.
    assert not any(name.startswith("model.layers.3.") for name in placed)
    for name in pruned_linears:
        assert type(model.get_submodule(name)) is nn.Linear


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pp_two_rank_runtime_forward_matches_single_rank(monkeypatch):
    """Two virtual PP ranks reproduce the single-rank runtime path bitwise.

    Exercises the model-side pipeline-parallel wiring on real components in
    one process: identical seeded construction then `__pp_init__` pruning per
    rank, per-rank real cache managers built with the PP mapping, per-rank
    prepared metadata, and the public forward on both ranks with
    `pp_send_tensors`/`pp_recv_tensors` replaced by an in-process channel.
    The four-stream [tokens, hc_mult, hidden] tensor sent by rank 0 is
    received into rank 1's `expand_streams` buffer, and rank 1's logits must
    equal the single-rank reference bitwise over prefill, two decode steps,
    and a mixed context+generation step. NCCL transport itself is R4's
    checkpoint-scale smoke; this pins every GLM-side PP contract.
    """
    from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
    from tensorrt_llm._torch.models import modeling_utils
    from tensorrt_llm.mapping import Mapping

    device = torch.device("cuda")
    config = _small_glm5_next_config()
    # Reorder the attention kinds so each virtual rank owns both: with
    # tensor_split(6, 2) -> [0,1,2] / [3,4,5], each slice gets L,S,L. This
    # mirrors the real checkpoint under PP=8, where every 5-6 consecutive
    # layers of the L,L,L,S pattern contain both kinds; a rank with zero
    # sparse layers is not a production shape.
    config.text_config.layer_types = [
        LINEAR_ATTENTION,
        SPARSE_ATTENTION,
        LINEAR_ATTENTION,
        LINEAR_ATTENTION,
        SPARSE_ATTENTION,
        LINEAR_ATTENTION,
    ]
    linear_cfg = dict(config.text_config.linear_attn_config)
    linear_cfg["kda_layers"] = [0, 2, 3, 5]
    linear_cfg["full_attn_layers"] = [1, 4]
    config.text_config.linear_attn_config = linear_cfg
    schedule = resolve_glm5_next_schedule(config)
    tokens_per_block = 64
    vocab = int(config.text_config.vocab_size)
    prompt = {
        0: torch.randint(0, vocab, (70,), generator=torch.Generator().manual_seed(7)).to(device),
        1: torch.randint(0, vocab, (40,), generator=torch.Generator().manual_seed(8)).to(device),
        2: torch.randint(0, vocab, (9,), generator=torch.Generator().manual_seed(9)).to(device),
    }

    def build(mapping=None):
        torch.manual_seed(0)
        if mapping is None:
            model_config = ModelConfig(pretrained_config=config)
        else:
            model_config = ModelConfig(pretrained_config=config, mapping=mapping)
        model = Glm5NextForCausalLM(model_config).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model, model_config

    ref_model, ref_config = build()
    rank0_model, _ = build(Mapping(world_size=2, tp_size=1, pp_size=2, rank=0))
    rank1_model, _ = build(Mapping(world_size=2, tp_size=1, pp_size=2, rank=1))

    # TRT-LLM-managed modules (LMHead, Embedding) create weights with
    # uninitialized torch.empty rather than seeded RNG, so cross-model
    # identity cannot come from the constructor seed alone: randomize the
    # reference deterministically and copy every surviving parameter/buffer
    # into the rank models.
    generator = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        for param in ref_model.parameters():
            if param.is_floating_point():
                param.normal_(0.0, 0.02, generator=generator)
        ref_state = dict(ref_model.named_parameters())
        ref_state.update(dict(ref_model.named_buffers()))
        for rank_model in (rank0_model, rank1_model):
            local = dict(rank_model.named_parameters())
            local.update(dict(rank_model.named_buffers()))
            for name, tensor in local.items():
                tensor.copy_(ref_state[name])

    # tensor_split(6, 2) -> layers [0,1,2] on rank 0, [3,4,5] on rank 1.
    for pruned in (rank0_model.model.layers[4], rank1_model.model.layers[1]):
        assert getattr(pruned, "_weights_removed", False)
        assert not any(True for _ in pruned.parameters())
    assert getattr(rank1_model.model.embed_tokens, "_weights_removed", False)
    assert getattr(rank0_model.lm_head, "_weights_removed", False)

    channel = []

    def fake_send(tensors):
        channel.extend(t.detach().clone() for t in tensors)

    def fake_recv(tensors):
        for t in tensors:
            assert t.is_contiguous()
            t.copy_(channel.pop(0))

    monkeypatch.setattr(modeling_utils, "pp_send_tensors", fake_send)
    monkeypatch.setattr(modeling_utils, "pp_recv_tensors", fake_recv)

    metadata_cls = get_attention_backend(ref_config.attn_backend).Metadata
    managers = {
        "ref": _build_small_cache_manager(
            config,
            schedule,
            device,
            max_seq_len=256,
            tokens_per_block=tokens_per_block,
            max_batch_size=3,
        ),
        "rank0": _build_small_cache_manager(
            config,
            schedule,
            device,
            max_seq_len=256,
            tokens_per_block=tokens_per_block,
            max_batch_size=3,
            mapping=Mapping(world_size=2, tp_size=1, pp_size=2, rank=0),
        ),
        "rank1": _build_small_cache_manager(
            config,
            schedule,
            device,
            max_seq_len=256,
            tokens_per_block=tokens_per_block,
            max_batch_size=3,
            mapping=Mapping(world_size=2, tp_size=1, pp_size=2, rank=1),
        ),
    }
    models = {"ref": ref_model, "rank0": rank0_model, "rank1": rank1_model}
    try:
        for manager in managers.values():
            manager.add_dummy_requests([0, 1, 2], token_nums=[73, 43, 9])
            _zero_cache(manager, schedule)

        def run(step_kwargs):
            out = {}
            for name in ("ref", "rank0", "rank1"):
                if name == "rank0":
                    channel.clear()
                out[name] = _runtime_step_logits(
                    models[name],
                    managers[name],
                    metadata_cls,
                    max_num_requests=3,
                    device=device,
                    **step_kwargs,
                )
            assert not channel, "rank 0 sent a tensor rank 1 never received"
            torch.testing.assert_close(out["rank1"], out["ref"], rtol=0.0, atol=0.0)
            assert out["rank0"].shape == out["ref"].shape
            return out["ref"]

        logits_a = run(
            dict(
                lens=[70, 40],
                num_contexts=2,
                cached=[0, 0],
                request_ids=[0, 1],
                prompt_lens=[70, 40],
                tokens=torch.cat([prompt[0], prompt[1]]),
            )
        )
        t_b1 = logits_a.argmax(dim=-1)
        logits_b1 = run(
            dict(
                lens=[1, 1],
                num_contexts=0,
                cached=[70, 40],
                request_ids=[0, 1],
                prompt_lens=[70, 40],
                tokens=t_b1,
            )
        )
        t_b2 = logits_b1.argmax(dim=-1)
        logits_b2 = run(
            dict(
                lens=[1, 1],
                num_contexts=0,
                cached=[71, 41],
                request_ids=[0, 1],
                prompt_lens=[70, 40],
                tokens=t_b2,
            )
        )
        t_c = logits_b2.argmax(dim=-1)
        run(
            dict(
                lens=[9, 1, 1],
                num_contexts=1,
                cached=[0, 72, 42],
                request_ids=[2, 0, 1],
                prompt_lens=[9, 70, 40],
                tokens=torch.cat([prompt[2], t_c]),
            )
        )
    finally:
        for manager in managers.values():
            manager.shutdown()


# ---------------------------------------------------------------------------
# CUDA-graph capture/replay: request state must never be baked into a graph
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graphs")
def test_sparse_decode_cuda_graph_replay_tracks_lengths_and_slots():
    """Captured sparse decode replays fresh lengths/tables — no stale state.

    The decode path's CUDA-graph contract says every request-derived value is
    read from device buffers refreshed outside the captured region. Proof by
    replay: capture the batched decode ONCE over persistent input/length/table
    buffers, then replay it while (1) both requests grow a token, (2) they
    grow again, and (3) a slot is reused by a NEW shorter request with a
    *different* page table. Each replay must equal an eager call over the same
    state — outputs AND cache writes. If any length, table row, or position
    had been captured as a Python value, replays 2 and 3 would reuse replay
    1's state and diverge.
    """
    from tensorrt_llm._torch.models.modeling_glm5_next import (
        Glm5NextSparseAttention,
        get_glm5_next_text_config,
    )

    torch.manual_seed(11)
    device = torch.device("cuda")
    text = get_glm5_next_text_config(_small_glm5_next_config())
    attn = Glm5NextSparseAttention(text, layer_idx=3).to(device).eval()
    generator = torch.Generator(device=device).manual_seed(11)
    with torch.no_grad():
        for p in attn.parameters():
            if p.is_floating_point():
                p.normal_(0.0, 0.02, generator=generator)

    from test_glm5_next_attention import _kpool_metadata, _KpoolPools

    tokens_per_block = 8
    pages = 6  # 3 pages per request -> fixed capacity 24 positions each
    batch = 2
    hidden = int(text.hidden_size)

    owner = _KpoolPools(attn, pages, tokens_per_block, device)
    latent, index = owner.latent, owner.index
    # Persistent buffers the captured kernels read — the runtime's
    # prepare()-refreshed metadata buffers in miniature; the backend reaches
    # them only through the metadata carrier.
    x_buf = torch.zeros(batch, hidden, device=device, dtype=torch.bfloat16)
    kv_buf = torch.ones(batch, dtype=torch.long, device=device)
    tbl_buf = torch.tensor([[0, 1, 2], [3, 4, 5]], device=device, dtype=torch.long)
    decode_md = _kpool_metadata(owner, block_tables=tbl_buf, kv_lens=kv_buf, num_contexts=0)
    ctx_md = _kpool_metadata(
        owner,
        block_tables=tbl_buf,
        kv_lens=torch.zeros(batch, dtype=torch.long, device=device),
        num_contexts=2,
    )

    def decode():
        return attn.forward_decode(x_buf, kv_buf, decode_md)

    def seed_prefix():
        latent.zero_()
        index.zero_()
        with torch.inference_mode():
            attn.forward_prefill(prompt, [0, 11, 18], [0, 0], ctx_md)

    prompt = torch.randn(18, hidden, device=device, dtype=torch.bfloat16)
    seed_prefix()

    # Warmup runs execute for real (and pollute the cache); reseed afterwards.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), torch.inference_mode():
        kv_buf.copy_(torch.tensor([12, 8], device=device))
        for _ in range(2):
            decode()
    torch.cuda.current_stream().wait_stream(side)
    seed_prefix()

    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        static_out = decode()

    def eager_expected():
        """Eager decode over clones of the current state (same code path)."""
        clone_owner = _KpoolPools(attn, pages, tokens_per_block, device)
        clone_owner.latent.copy_(latent)
        clone_owner.index.copy_(index)
        eager_md = _kpool_metadata(
            clone_owner, block_tables=tbl_buf.clone(), kv_lens=kv_buf.clone(), num_contexts=0
        )
        with torch.inference_mode():
            out = attn.forward_decode(x_buf.clone(), kv_buf.clone(), eager_md)
        return out, clone_owner.latent, clone_owner.index

    def replay_and_check(step_x, kv_lens):
        x_buf.copy_(step_x)
        kv_buf.copy_(torch.as_tensor(kv_lens, dtype=torch.long))
        expected_out, expected_latent, expected_index = eager_expected()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(static_out, expected_out)
        assert torch.equal(latent, expected_latent)
        assert torch.equal(index, expected_index)

    gen = lambda: torch.randn(batch, hidden, device=device, dtype=torch.bfloat16)  # noqa: E731
    # Replays 1-2: both requests grow one token per step.
    replay_and_check(gen(), [12, 8])
    replay_and_check(gen(), [13, 9])

    # Replay 3: slot reuse — request 1 is replaced by a NEW request whose
    # table maps the same physical pages in a DIFFERENT order and whose
    # length restarts short. Zero its pages (release+realloc) and write a
    # fresh 5-token prefix through the new table before decoding token 6.
    new_row = torch.tensor([5, 3, 4], device=device, dtype=torch.long)
    for page in (3, 4, 5):
        latent[page].zero_()
        index[page].zero_()
    tbl_buf[1].copy_(new_row)
    with torch.inference_mode():
        fresh = torch.randn(5, hidden, device=device, dtype=torch.bfloat16)
        reprefill_md = _kpool_metadata(
            owner,
            block_tables=tbl_buf[1:2],
            kv_lens=torch.zeros(1, dtype=torch.long, device=device),
            num_contexts=1,
        )
        attn.forward_prefill(fresh, [0, 5], [0], reprefill_md)
    replay_and_check(gen(), [14, 6])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graphs")
def test_moe_decode_cuda_graph_replay_tracks_routing():
    """Captured MoE decode re-routes per replay — expert choice is not baked.

    The decode MoE path indexes expert weights by 0-d device tensors from the
    router's output, so a captured graph must re-route as inputs change. Three
    replays with different inputs (hence different top-k selections) must each
    equal the eager decode-path result for the same input.
    """
    from tensorrt_llm._torch.models.modeling_glm5_next import Glm5NextMoE, get_glm5_next_text_config

    torch.manual_seed(23)
    device = torch.device("cuda")
    text = get_glm5_next_text_config(_small_glm5_next_config())
    moe = Glm5NextMoE(text, quantized=False).to(device).eval()
    generator = torch.Generator(device=device).manual_seed(23)
    with torch.no_grad():
        for p in moe.parameters():
            if p.is_floating_point():
                p.normal_(0.0, 0.02, generator=generator)

    hidden = int(text.hidden_size)
    x_buf = torch.zeros(2, hidden, device=device, dtype=torch.bfloat16)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), torch.inference_mode():
        for _ in range(2):
            moe(x_buf, phase="decode")
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        static_out = moe(x_buf, phase="decode")

    routings = []
    for trial in range(3):
        x = torch.randn(2, hidden, device=device, dtype=torch.bfloat16)
        x_buf.copy_(x)
        with torch.inference_mode():
            expected = moe(x.clone(), phase="decode")
            _, _, topk = moe.gate(x.clone().reshape(-1, hidden))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(static_out, expected), f"trial {trial} diverged from eager"
        routings.append(topk.cpu())
    # The proof is only meaningful if the routing actually changed.
    assert not torch.equal(routings[0], routings[1]) or not torch.equal(routings[1], routings[2]), (
        "test inputs produced identical routings; nothing was proven"
    )
