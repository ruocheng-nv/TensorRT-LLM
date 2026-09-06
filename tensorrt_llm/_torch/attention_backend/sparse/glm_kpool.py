# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash k-pool sparse MLA: the fully-NoPE branch of the TRTLLM family.

``glm5_next`` sparse layers are fully NoPE MLA (``qk_rope_head_dim == 0``,
``qk_nope_head_dim == 256``, ``kv_lora_rank == 512``) whose visible set is
selected by a *pool-compressed* indexer: size-``index_kpool`` pools of keys are
scored, the top ``index_topk / index_kpool`` pools expand back into member
positions, and the incomplete tail is always appended, padded with ``-1`` to a
fixed logical width. Stock DeepSeek-V3.2 DSA selects individual keys and
assumes the 576-wide rope'd DeepSeek geometry (its SM100 absorption route
applies RoPE through ``mla_rope_append_paged_kv_assign_q``, and the shared
``MLA`` module's DSA hook requires the fused ``[q_a|kv_a|k_pe]`` projection and
``qk_rope_head_dim > 0``), so it cannot express this contract without inventing
a fake rotary width.

:class:`GlmKpoolSparseAttention` is therefore a narrow **subclass of
:class:`~..trtllm.TrtllmAttention`** -- a sibling of
:class:`~.dsa.backend.DSATrtllmAttention` inside the TRTLLM sparse family --
rather than a fork outside it:

* **Family** -- it inherits the TRTLLM backend's identity wholesale:
  ``support_mla()`` is true, ``is_mla_enable`` is true through a fully-NoPE
  :class:`~..interface.MLAParams`, and ``Metadata`` is
  :class:`~..trtllm.TrtllmAttentionMetadata`, the typed metadata class the
  engine constructs from ``attn_backend.Metadata``.
* **Selection** -- ``get_attention_backend("TRTLLM", sparse_params)`` resolves
  ``SparseParams(algorithm="glm_kpool")`` to this class next to the DSA branch
  in the sparse registry, and the standard ``create_attention(...)`` dispatch
  constructs it.
* **Contract** -- :meth:`forward` keeps the exact
  ``AttentionBackend.forward(q, k, v, metadata, forward_args)`` signature:
  options merge through :func:`~..interface.merge_attention_forward_args`
  (which rejects unknown or mixed arguments), the model layer's pool-expanded
  selection arrives through the typed
  ``AttentionForwardArgs.sparse_backend_args.topk_indices`` carrier, and the
  phase is declared through ``forward_args.attention_input_type``.
* **Cache ownership** -- every paged read and write is derived from the
  prepared ``metadata``: the one hybrid ``KVCacheManagerV2`` reached through
  ``metadata.kv_cache_manager`` owns the latent pages and the packed indexer
  state, and the per-request block tables / visible lengths come from the
  ``prepare()``-refreshed ``mamba_metadata.glm_*`` buffers. Callers never hand
  raw pool tensors to this backend.
* **Execution** -- the absorbed sparse-MLA core dispatches
  ``tensorrt_llm.flash_mla.flash_mla_sparse_fwd``, the FlashMLA sparse kernel
  of the DSA stack (``sparse/dsa/module.py``), which natively supports
  ``d_qk == d_v == 512`` and 64 query heads (``HEAD_DIM_512`` /
  ``Fwd_Sm100_Head64_Impl`` in FlashMLA's ``sparse_fwd.h``) with the same
  ``-1``/out-of-range invalid-index contract this model's indexer emits.

The model layer (``modeling_glm5_next.Glm5NextSparseAttention``) keeps the
module math -- projections, norms, the pool indexer's scoring/selection
(model-layer sparse prediction, as in MiniMax-M3), query absorption, and the
output projection -- and drives this backend only through the standard
contract entry points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import torch

from ..interface import (
    AttentionForwardArgs,
    AttentionInputType,
    MLAParams,
    PositionalEmbeddingParams,
    merge_attention_forward_args,
)
from ..trtllm import TrtllmAttention, TrtllmAttentionMetadata
from .params import SparseParams

#: Invalid-slot marker in every index tensor this backend consumes. It is the
#: FlashMLA sparse kernel's own invalid contract ("-1 or >= s_kv"), so padded
#: or unselected slots pass through unchanged; clamping them into range would
#: attend a real row.
INDEX_SENTINEL = -1


@dataclass(frozen=True)
class GlmKpoolSparseParams(SparseParams):
    """Lowered runtime parameters for the GLM k-pool sparse-MLA backend."""

    algorithm: Literal["glm_kpool"] = field(init=False, default="glm_kpool")
    #: Latent (compressed KV) width; also the absorbed query head width and
    #: the kernel's d_qk == d_v. 512 on this checkpoint.
    kv_lora_rank: int = 512
    #: Pre-absorption query/key head width; sets the softmax scale. Fully
    #: NoPE: there is no rope component on top of it.
    qk_nope_head_dim: int = 256
    #: Low-rank query bottleneck width; carried into ``MLAParams`` so the
    #: backend's MLA identity states the real checkpoint geometry.
    q_lora_rank: int = 1536
    #: Per-head value width after the absorbed V projection.
    v_head_dim: int = 256
    #: Number of key positions the expanded selection may cover.
    index_topk: int = 2048
    #: Members per compressed pool.
    index_kpool: int = 4
    #: Whether the incomplete trailing pool is always appended.
    index_always_select_tail: bool = True

    @property
    def indices_block_size(self) -> int:
        return 1

    @property
    def output_width(self) -> int:
        """Fixed logical width of the expanded index rows the model emits."""
        return self.index_topk + (self.index_kpool - 1 if self.index_always_select_tail else 0)


def _flash_mla_sparse_fwd():
    """The production sparse-MLA kernel entry point (lazy import).

    Resolved at call time so a test can intercept the module attribute to
    prove the backend is the one dispatching it.
    """
    try:
        from tensorrt_llm.flash_mla import flash_mla_sparse_fwd
    except ImportError as exc:  # pragma: no cover - wheel always bundles it
        raise RuntimeError(
            "glm_kpool sparse MLA requires tensorrt_llm.flash_mla."
            "flash_mla_sparse_fwd, which this build does not provide"
        ) from exc
    return flash_mla_sparse_fwd


def paged_slot_indices(
    block_table: torch.Tensor,
    positions: torch.Tensor,
    tokens_per_block: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(page, within_page)`` indices for ``positions`` in a paged pool.

    ``block_table`` is ``[..., max_pages]`` and ``positions`` is broadcastable
    against its leading dimensions. The pair addresses a pool shaped
    ``[num_pages, tokens_per_block, ...]``.

    Returning the pair rather than one flat ``page * tokens_per_block + offset``
    index is required, not stylistic. ``KVCacheManagerV2`` coalesces buffers
    whose per-page size differs from K/V's into a shared pool, so the
    ``Role.INDEX_KEY`` view it hands back is **strided**: its page stride is the
    slot stride, not its own payload size. Measured on this model the indexer
    buffer is ``[16, 64, 1, 256]`` with page stride 32768 against a 16384-element
    payload. A flat index is only correct for a densely packed pool, and
    flattening that view is not merely wrong -- ``.view`` raises and ``.reshape``
    silently *copies*, so every cache write would land in a temporary and be
    lost. The pair is also what the accessor's own contract prescribes.

    Callers must mask invalid positions themselves -- this deliberately does not
    invent a fallback page, because a silently wrong page is exactly the
    cross-request leak the hybrid cache has to rule out.
    """
    page = torch.div(positions, tokens_per_block, rounding_mode="floor")
    offset = positions - page * tokens_per_block
    return torch.gather(block_table, -1, page), offset


def latent_pool_rows(latent_pool: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """Row-space view of the latent pool's storage for the sparse kernel.

    ``latent_pool`` is the slot-major ``[slots, tokens_per_block, dim]`` view
    from ``Glm5NextCacheManager.get_latent_state_buffer``. V2 coalesces
    several buffers into one pool, so the slot stride is larger than the
    payload and the pool cannot be flattened with ``view`` (it would raise)
    or ``reshape`` (it would silently copy every step). The kernel, however,
    only needs *uniformly strided rows*: it reads ``kv_ptr + idx * stride``.
    Every latent row already sits at a multiple of ``dim`` elements inside
    the storage, so the whole storage is reinterpreted as ``[N, 1, dim]``
    rows and positions are translated to row ids instead.

    Returns ``(rows, base_row, rows_per_slot)`` where the row id of cache
    position ``(slot, t)`` is ``base_row + slot * rows_per_slot + t``. Rows
    belonging to other coalesced buffers are addressable but never indexed:
    only ids produced by that formula (or ``-1`` sentinels) reach the kernel.
    Pure metadata work -- no device kernel -- so it is CUDA-graph safe.
    """
    slots, tokens_per_block, dim = latent_pool.shape
    del slots, tokens_per_block
    if latent_pool.stride(2) != 1 or latent_pool.stride(1) != dim:
        raise ValueError(
            "glm_kpool latent pool rows must be contiguous within a page; got "
            f"strides {tuple(latent_pool.stride())} for dim {dim}"
        )
    slot_stride = latent_pool.stride(0)
    offset = latent_pool.storage_offset()
    if slot_stride % dim or offset % dim:
        raise ValueError(
            f"glm_kpool latent pool slot stride {slot_stride} / storage offset "
            f"{offset} are not multiples of dim {dim}; the pool has no uniform "
            "row view and the coalesced layout assumption broke"
        )
    total_rows = latent_pool.untyped_storage().nbytes() // (dim * latent_pool.element_size())
    rows = torch.as_strided(latent_pool, (total_rows, 1, dim), (dim, dim, 1), storage_offset=0)
    return rows, offset // dim, slot_stride // dim


def positions_to_pool_rows(
    positions: torch.Tensor,
    block_table: torch.Tensor,
    tokens_per_block: int,
    base_row: int,
    rows_per_slot: int,
) -> torch.Tensor:
    """Request-local cache positions -> global row ids for the sparse kernel.

    ``positions`` is int32 ``[..., width]`` with :data:`INDEX_SENTINEL` in
    invalid slots; ``block_table`` holds base slot ids per request. Sentinels
    are preserved as ``-1`` (the kernel's own invalid marker) rather than
    clamped -- a clamped sentinel would address a real row. Fixed shapes,
    gathers, and ``where`` only, so a captured decode graph replays this
    against prepare()-refreshed tables.
    """
    safe = positions.clamp(min=0).long()
    page = torch.div(safe, tokens_per_block, rounding_mode="floor")
    slot = torch.gather(block_table, -1, page)
    rows = base_row + slot * rows_per_slot + (safe - page * tokens_per_block)
    return torch.where(positions >= 0, rows, positions.long()).to(torch.int32)


@dataclass(frozen=True)
class _GlmKpoolCacheState:
    """One layer's cache state, derived from prepared attention metadata.

    ``latent_pool``/``index_pool`` are the slot-major ``[slots,
    tokens_per_block, dim]`` views over the hybrid manager's coalesced pools;
    ``block_tables``/``kv_lens`` cover the whole batch in executor order
    (contexts first). Everything here is a view or a host int -- deriving it
    launches no kernel, so it is safe inside a CUDA-graph capture as long as
    the underlying buffers are the persistent ``prepare()``-refreshed ones.
    """

    latent_pool: torch.Tensor
    index_pool: torch.Tensor
    block_tables: torch.Tensor
    kv_lens: torch.Tensor
    tokens_per_block: int
    num_contexts: int


class GlmKpoolSparseAttention(TrtllmAttention):
    """Fully-NoPE k-pool sparse-MLA branch of the TRTLLM attention family.

    See the module docstring for the family/contract rationale. Constructed
    under the standard ``create_attention(...)`` dispatch when
    ``SparseParams(algorithm="glm_kpool")`` is configured on the TRTLLM
    backend slot; the sparse registry resolves it next to
    ``DSATrtllmAttention``.
    """

    Metadata = TrtllmAttentionMetadata

    #: FlashMLA tiles the top-k axis in blocks of 64 (``B_TOPK``); index rows
    #: are padded to that multiple with ``-1`` (invalid) entries, which is
    #: semantics-free by the kernel's own contract.
    _KERNEL_TOPK_ALIGN = 64

    #: Query-head counts the FlashMLA sparse kernel instantiates
    #: (``Fwd_Sm100_Head64_Impl``/``Head128`` in ``sparse_fwd``; any other
    #: ``h_q`` raises ``Unsupported h_q``). Under tensor parallelism the local
    #: head count (16 of 64 at TP4) is below the smallest instantiation, so
    #: :meth:`_dispatch_sparse_core` zero-pads the query-head axis up to the
    #: next instantiated count and slices the output back -- the in-tree DSA
    #: precedent (``sparse/dsa/module.py`` pads its TP-local heads the same
    #: way). Attention is per-head, so zero query lanes cannot perturb real
    #: lanes; their outputs are discarded by the slice.
    _KERNEL_HEAD_COUNTS = (64, 128)

    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        quant_config=None,
        q_scaling: Optional[float] = None,
        pos_embd_params: Optional[PositionalEmbeddingParams] = None,
        mla_params: Optional[MLAParams] = None,
        skip_create_weights_in_init: bool = False,
        attention_chunk_size: Optional[int] = None,
        sparse_params: Optional[GlmKpoolSparseParams] = None,
        dtype: Optional[torch.dtype] = None,
        aux_stream: Optional[torch.cuda.Stream] = None,
        **kwargs,
    ):
        # dtype/aux_stream arrive from the standard create_attention kwargs;
        # this branch has no dtype-dependent weights and no side stream.
        del dtype, aux_stream
        # The engine-level construction path passes the llmapi config object;
        # keep it (as DSA does) without forwarding it into the base class.
        self.sparse_attention_config = kwargs.pop("sparse_attention_config", None)
        if sparse_params is None:
            raise ValueError("sparse_params is required for GlmKpoolSparseAttention")
        if not isinstance(sparse_params, GlmKpoolSparseParams):
            raise TypeError(
                f"GlmKpoolSparseAttention needs GlmKpoolSparseParams, got {type(sparse_params)}"
            )
        if head_dim != sparse_params.kv_lora_rank:
            raise ValueError(
                "glm_kpool consumes absorbed latent-space queries: head_dim "
                f"({head_dim}) must equal kv_lora_rank ({sparse_params.kv_lora_rank})"
            )
        if pos_embd_params is not None:
            raise ValueError(
                "glm_kpool is fully NoPE; positional embedding parameters have no "
                "meaning on this branch"
            )
        if mla_params is None:
            # The standard create_attention MLA path asserts qk_rope_head_dim>0
            # (the rope'd DeepSeek geometry), so this fully-NoPE branch states
            # its MLA identity itself instead of loosening the shared assert.
            mla_params = MLAParams(
                q_lora_rank=sparse_params.q_lora_rank,
                kv_lora_rank=sparse_params.kv_lora_rank,
                qk_rope_head_dim=0,
                qk_nope_head_dim=sparse_params.qk_nope_head_dim,
                v_head_dim=sparse_params.v_head_dim,
                rope_append=False,
            )
        if mla_params.qk_rope_head_dim != 0:
            raise ValueError(
                f"glm_kpool is fully NoPE; got qk_rope_head_dim={mla_params.qk_rope_head_dim}"
            )
        TrtllmAttention.__init__(
            self,
            layer_idx,
            num_heads,
            head_dim,
            num_kv_heads=num_kv_heads,
            quant_config=quant_config,
            q_scaling=q_scaling,
            pos_embd_params=None,
            mla_params=mla_params,
            skip_create_weights_in_init=skip_create_weights_in_init,
            attention_chunk_size=attention_chunk_size,
            sparse_params=sparse_params,
            **kwargs,
        )
        if self.num_kv_heads != 1:
            raise ValueError(
                f"glm_kpool latent cache is MQA-style (one KV head), got {self.num_kv_heads}"
            )
        #: Softmax scale of the *unabsorbed* q . k product over
        #: ``qk_nope_head_dim``; absorption reassociates the matmuls but the
        #: score scale is unchanged.
        self.softmax_scale = float(sparse_params.qk_nope_head_dim) ** -0.5

    def create_fmha_libs(self) -> None:
        """This branch dispatches the FlashMLA sparse kernel itself.

        The dense FMHA libraries the base class assembles serve the C++
        attention op's dense/rope'd paths; none of them expresses
        pool-compressed fully-NoPE sparse MLA, and instantiating them for a
        geometry they were never validated on (``qk_rope_head_dim == 0``,
        512-wide latent heads) would only create dead, unproven objects. The
        empty lists make any accidental dense dispatch fail loudly instead of
        silently running the wrong kernel.
        """
        self.fmha_libs = []
        self.phased_fmha_libs = []
        self.non_phased_fmha_libs = []
        self.combined_fmha = None

    @classmethod
    def support_fused_rope(cls) -> bool:
        # Fully NoPE: there is no rotary embedding anywhere on this path.
        return False

    @classmethod
    def support_fused_qkv(cls) -> bool:
        # The model layer owns the low-rank q/kv projections and absorption.
        return False

    # -- metadata-derived cache state ----------------------------------------

    def _cache_state(self, metadata) -> _GlmKpoolCacheState:
        """Derive this layer's cache state from the prepared metadata.

        The single source of cache truth for every entry point below. The
        production path is the persistent one: ``prepare()`` with the
        ``Glm5NextCacheManager`` attached refreshes the ``mamba_metadata``'s
        ``glm_block_tables``/``glm_kv_lens`` device buffers, so a captured
        decode graph replays against fresh values at stable addresses.
        Harness metadata whose manager does not attach those buffers falls
        back to an eager host-side derivation, which is refused under CUDA
        graphs exactly like the model's runtime-context builder used to.
        """
        if metadata is None:
            raise ValueError(
                "GlmKpoolSparseAttention requires the engine's prepared attention "
                "metadata (TrtllmAttentionMetadata); got None. The backend derives "
                "its cache pools, block tables, and visible lengths from it."
            )
        manager = getattr(metadata, "kv_cache_manager", None)
        if manager is None:
            raise ValueError(
                "glm_kpool metadata has no kv_cache_manager; the hybrid "
                "KVCacheManagerV2 owns the latent/indexer pools"
            )
        mamba_metadata = getattr(metadata, "mamba_metadata", None)
        if mamba_metadata is None or mamba_metadata is False:
            raise ValueError(
                "glm_kpool requires prepared metadata: call metadata.prepare() "
                "with the Glm5NextCacheManager attached (mamba_metadata is missing)"
            )
        latent = manager.get_latent_state_buffer(self.layer_idx)
        index = manager.get_index_state_buffer(self.layer_idx)
        if latent is None or index is None:
            raise ValueError(
                f"glm_kpool layer {self.layer_idx} has no latent/indexer pool on "
                "this manager; the layer schedule and the cache layout disagree"
            )
        latent = latent[:, :, 0, :]
        index = index[:, :, 0, :]
        tokens_per_block = int(manager.tokens_per_block)
        batch = int(metadata.seq_lens.shape[0])
        num_contexts = int(metadata.num_contexts)

        tables = getattr(mamba_metadata, "glm_block_tables", None)
        if tables is not None:
            # Persistent path: slices of prepare()-refreshed buffers. No
            # allocation, no H2D, no host sync -- CUDA-graph safe.
            return _GlmKpoolCacheState(
                latent_pool=latent,
                index_pool=index,
                block_tables=tables[:batch],
                kv_lens=mamba_metadata.glm_kv_lens[:batch],
                tokens_per_block=tokens_per_block,
                num_contexts=num_contexts,
            )

        # Legacy eager derivation for harness managers that do not attach the
        # GLM buffers. It allocates and copies, so it must never run inside a
        # captured region.
        if getattr(metadata, "is_cuda_graph", False):
            raise RuntimeError(
                "glm_kpool CUDA-graph execution requires the persistent "
                "prepare()-refreshed glm_block_tables/glm_kv_lens buffers; the "
                "attached mamba_metadata has no glm_block_tables"
            )
        kv_params = getattr(metadata, "kv_cache_params", None)
        if kv_params is None or kv_params.num_cached_tokens_per_seq is None:
            raise ValueError("glm_kpool requires kv_cache_params.num_cached_tokens_per_seq")
        lens = [int(n) for n in metadata.seq_lens[:batch]]
        cached = [int(n) for n in kv_params.num_cached_tokens_per_seq[:batch]]
        device = latent.device
        # Raw base-slot IDs, NOT get_batch_cache_indices: the latent/index
        # views are slot-major, and V2's standard accessor scales page ids for
        # its own flattened per-layer views.
        pages = manager.get_batch_slot_tables(list(metadata.request_ids)[:batch])
        max_pages = max((len(p) for p in pages), default=1) or 1
        block_tables = torch.zeros(batch, max_pages, dtype=torch.long, device=device)
        for row, page_ids in enumerate(pages):
            if page_ids:
                block_tables[row, : len(page_ids)] = torch.as_tensor(
                    page_ids, dtype=torch.long, device=device
                )
        kv_lens = torch.as_tensor(
            [c + n for c, n in zip(cached, lens)], dtype=torch.long, device=device
        )
        return _GlmKpoolCacheState(
            latent_pool=latent,
            index_pool=index,
            block_tables=block_tables,
            kv_lens=kv_lens,
            tokens_per_block=tokens_per_block,
            num_contexts=num_contexts,
        )

    # -- paged cache path -----------------------------------------------------

    def append_paged_state(
        self,
        latent: torch.Tensor,
        packed: torch.Tensor,
        positions: torch.Tensor,
        metadata,
        *,
        request_index: Optional[int] = None,
    ) -> None:
        """Write new tokens' latent and packed indexer state to the pools.

        ``positions`` carries the tokens' cache positions (schedule, owned by
        the model layer); which pools and tables they land in is derived from
        ``metadata``. ``request_index`` selects one context request's table
        row; ``None`` addresses the generation rows, with ``positions`` shaped
        ``[num_generations, 1]``. Callers pass only positions they own -- see
        :func:`paged_slot_indices` for why no fallback page is invented.
        """
        state = self._cache_state(metadata)
        if request_index is None:
            table = state.block_tables[state.num_contexts :]
        else:
            table = state.block_tables[request_index]
        page, offset = paged_slot_indices(table, positions, state.tokens_per_block)
        state.latent_pool[page, offset] = latent.to(state.latent_pool.dtype)
        state.index_pool[page, offset] = packed.to(state.index_pool.dtype)

    def gather_paged_prefix(
        self,
        length: int,
        metadata,
        *,
        request_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One context request's cached latent and packed-indexer prefix (host
        loop path: prefill only, never captured)."""
        state = self._cache_state(metadata)
        positions = torch.arange(length, device=state.latent_pool.device)
        page, offset = paged_slot_indices(
            state.block_tables[request_index], positions, state.tokens_per_block
        )
        return state.latent_pool[page, offset], state.index_pool[page, offset]

    def gather_packed_prefix(self, metadata) -> Tuple[torch.Tensor, int]:
        """Fixed-capacity packed-indexer prefixes for the generation rows.

        Returns ``([num_generations, padded_capacity, dim], num_pools)`` where
        ``padded_capacity`` is the buffer capacity rounded up to whole pools.
        Rows at or beyond each request's visible length gather page-0 garbage
        and are masked by *replacement* (not multiplication, so cached NaN/Inf
        garbage cannot leak through ``0 * inf``). Every shape depends only on
        buffer geometry -- CUDA-graph safe.
        """
        state = self._cache_state(metadata)
        block_tables = state.block_tables[state.num_contexts :]
        kv_lens = state.kv_lens[state.num_contexts :]
        batch = block_tables.shape[0]
        device = state.index_pool.device
        capacity = block_tables.shape[1] * state.tokens_per_block
        kpool = self.sparse_params.index_kpool
        num_pools = (capacity + kpool - 1) // kpool
        padded_cap = num_pools * kpool

        grid = torch.arange(padded_cap, device=device)
        read_pos = grid.clamp(max=capacity - 1).unsqueeze(0).expand(batch, padded_cap)
        page, offset = paged_slot_indices(block_tables, read_pos, state.tokens_per_block)
        valid = grid.unsqueeze(0) < kv_lens.unsqueeze(1)
        packed = state.index_pool[page, offset].masked_fill(~valid.unsqueeze(-1), 0)
        return packed, num_pools

    # -- sparse core ----------------------------------------------------------

    def create_output(
        self,
        q: torch.Tensor,
        *,
        is_quantize_output: bool,
        metadata=None,
        attention_mask=None,
        is_gen_only: bool = False,
        **kwargs,
    ) -> List[torch.Tensor]:
        """Allocate the standard flat output buffer for this absorbed branch.

        Reconciles the inherited ``TrtllmAttention.create_output``, whose MLA
        context leg allocates ``num_heads * v_head_dim``: the absorbed
        formulation emits the *latent* width ``num_heads * kv_lora_rank`` in
        **both** phases (the model layer applies the absorbed V projection to
        it afterwards, so ``v_head_dim`` never appears at this boundary).
        Quantized/NVFP4 output modes are not implemented on this branch and
        are rejected loudly rather than silently mis-allocated.
        """
        del metadata, attention_mask, is_gen_only, kwargs
        if is_quantize_output:
            raise ValueError(
                "glm_kpool produces a bf16 latent-space output; quantized "
                "attention output (out_scale/output_sf) is not supported"
            )
        return [q.new_empty((q.shape[0], self.num_heads * self.kv_lora_rank), dtype=q.dtype)]

    def _dispatch_sparse_core(
        self,
        q_latent: torch.Tensor,
        kv_rows: torch.Tensor,
        topk_rows: torch.Tensor,
    ) -> torch.Tensor:
        """Pad the index rows to the kernel's tiles and run the sparse kernel.

        ``q_latent`` is ``[T, H, kv_lora]``, ``kv_rows`` is ``[N, 1,
        kv_lora]``, ``topk_rows`` is int32 ``[T, width]`` with
        :data:`INDEX_SENTINEL` invalid slots. Returns the kernel's latent-space
        output ``[T, H, kv_lora]``; :meth:`_finalize_output` flattens it to the
        base-contract shape at the ``forward`` boundary.
        """
        pad = (-topk_rows.shape[-1]) % self._KERNEL_TOPK_ALIGN
        if pad:
            topk_rows = torch.nn.functional.pad(topk_rows, (0, pad), value=INDEX_SENTINEL)
        local_heads = q_latent.shape[1]
        kernel_heads = next((h for h in self._KERNEL_HEAD_COUNTS if h >= local_heads), None)
        if kernel_heads is None:
            raise ValueError(
                f"glm_kpool: {local_heads} query heads exceed every FlashMLA "
                f"sparse-kernel instantiation {self._KERNEL_HEAD_COUNTS}"
            )
        if kernel_heads != local_heads:
            # Zero-filled query lanes; per-head attention keeps them inert and
            # the slice below discards their outputs (DSA's TP-head padding).
            q_padded = q_latent.new_zeros((q_latent.shape[0], kernel_heads, q_latent.shape[2]))
            q_padded[:, :local_heads, :] = q_latent
            q_latent = q_padded
        out, _, _ = _flash_mla_sparse_fwd()(
            q_latent,
            kv_rows,
            topk_rows.unsqueeze(1),
            self.softmax_scale,
            self.kv_lora_rank,
        )
        if kernel_heads != local_heads:
            out = out[:, :local_heads, :].contiguous()
        return out

    def _finalize_output(
        self, out_latent: torch.Tensor, output: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Flatten the kernel output to ``[T, num_heads * kv_lora]``.

        With no caller buffer the kernel's own (contiguous) allocation is
        returned viewed flat -- same shape/dtype/device as
        :meth:`create_output` would allocate, without a redundant copy. A
        caller-provided ``forward_args.output`` (already validated in
        :meth:`forward`) is written in place and returned, matching the
        TRTLLM family's caller-owned-buffer semantics.
        """
        flat = out_latent.reshape(out_latent.shape[0], -1)
        if output is None:
            return flat
        output.copy_(flat)
        return output

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: TrtllmAttentionMetadata,
        forward_args: Optional[AttentionForwardArgs] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Standard ``AttentionBackend.forward`` for the k-pool sparse core.

        Arguments follow the base contract exactly:

        ``q``
            Absorbed latent-space queries, ``[T, num_heads * kv_lora]`` or the
            equivalent ``[T, num_heads, kv_lora]`` view.
        ``k``
            Context phase: the request's contiguous cached latent prefix
            ``[KV, kv_lora]`` (``num_kv_heads == 1``), gathered from the same
            metadata through :meth:`gather_paged_prefix`. Generation phase:
            ``None`` -- the kernel reads the paged latent pool directly
            through the storage row view derived from ``metadata``.
        ``v``
            Must be ``None``: the latent rows serve as both K and V, exactly
            as in absorbed MLA; the model layer applies the absorbed V
            projection to the returned latent output.
        ``metadata``
            The engine's prepared typed metadata. Required: the paged pools,
            block tables, and visible lengths are derived from it (see
            :meth:`_cache_state`); ``None`` or unprepared metadata is a loud
            error.
        ``forward_args`` / ``**kwargs``
            Merged through :func:`merge_attention_forward_args`, which
            rejects unknown kwargs and forward_args/kwargs mixing. The
            model layer's pool-expanded selection travels in the typed
            ``sparse_backend_args.topk_indices`` field (int32
            ``[T, output_width]``, request-local positions,
            :data:`INDEX_SENTINEL` padding), and
            ``attention_input_type`` declares the phase -- this backend is
            phase-explicit, so ``mixed`` is rejected. A caller-provided
            ``forward_args.output`` is validated (shape/dtype/device),
            written in place, and returned -- the TRTLLM family's
            caller-owned-buffer semantics. Quantized-output modes
            (``out_scale``/``out_scale_sf``/``output_sf``) are not
            implemented here and are rejected loudly.

        Returns the flat latent-space attention output
        ``[T, num_heads * kv_lora]`` -- the base contract's
        ``(num_q_tokens, num_heads * head_dim)`` with this backend's
        ``head_dim == kv_lora_rank``. The model layer views it per-head for
        the absorbed V projection.
        """
        forward_args = merge_attention_forward_args(forward_args, kwargs)
        sparse_args = forward_args.sparse_backend_args
        topk_indices = sparse_args.topk_indices if sparse_args is not None else None
        if topk_indices is None:
            raise NotImplementedError(
                "GlmKpoolSparseAttention needs the model layer's pool-expanded "
                "selection in forward_args.sparse_backend_args.topk_indices; the "
                "k-pool scoring/selection is module math -- see "
                "modeling_glm5_next.Glm5NextSparseAttention."
            )
        if v is not None:
            raise ValueError("glm_kpool consumes latent rows as both K and V; v must be None")
        if q.dim() == 2:
            q = q.view(q.shape[0], self.num_heads, self.head_dim)

        if (
            forward_args.out_scale is not None
            or forward_args.out_scale_sf is not None
            or forward_args.output_sf is not None
        ):
            raise ValueError(
                "glm_kpool does not support quantized attention output "
                "(out_scale/out_scale_sf/output_sf); it returns the bf16 "
                "latent-space output the model layer projects"
            )
        output = forward_args.output
        if output is not None:
            expected_shape = (q.shape[0], self.num_heads * self.kv_lora_rank)
            if (
                tuple(output.shape) != expected_shape
                or output.dtype != q.dtype
                or output.device != q.device
            ):
                raise ValueError(
                    f"glm_kpool forward_args.output must be a {expected_shape} "
                    f"tensor of dtype {q.dtype} on {q.device}; got shape "
                    f"{tuple(output.shape)}, dtype {output.dtype}, device "
                    f"{output.device}"
                )

        state = self._cache_state(metadata)
        input_type = forward_args.attention_input_type
        if input_type == AttentionInputType.context_only:
            if k is None:
                raise ValueError(
                    "glm_kpool context forward needs the request's contiguous "
                    "latent prefix as k (see gather_paged_prefix)"
                )
            kv_len = k.shape[0]
            out_latent = self._dispatch_sparse_core(
                q, k.view(kv_len, 1, self.kv_lora_rank), topk_indices
            )
            return self._finalize_output(out_latent, output)
        if input_type == AttentionInputType.generation_only:
            if k is not None:
                raise ValueError(
                    "glm_kpool generation forward reads the paged latent pool "
                    "from metadata; k must be None"
                )
            gen_tables = state.block_tables[state.num_contexts :]
            if gen_tables.shape[0] != q.shape[0]:
                raise ValueError(
                    f"glm_kpool generation forward got {q.shape[0]} query rows for "
                    f"{gen_tables.shape[0]} generation requests in the metadata"
                )
            kv_rows, base_row, rows_per_slot = latent_pool_rows(state.latent_pool)
            topk_rows = positions_to_pool_rows(
                topk_indices, gen_tables, state.tokens_per_block, base_row, rows_per_slot
            )
            return self._finalize_output(self._dispatch_sparse_core(q, kv_rows, topk_rows), output)
        raise ValueError(
            "glm_kpool forward is phase-explicit: set "
            "forward_args.attention_input_type to context_only or "
            f"generation_only, got {input_type!r}"
        )
