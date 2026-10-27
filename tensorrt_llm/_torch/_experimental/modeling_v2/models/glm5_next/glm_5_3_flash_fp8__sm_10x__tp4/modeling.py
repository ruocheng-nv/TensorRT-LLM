# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash block-FP8 on sm_100 / sm_103, tp4: model construction and forward.

The checkpoint is the composite ``Glm5NextForConditionalGeneration`` release served
text only (``disable_mm_encoder``). ``ModelingV2Glm53FlashFp8Sm10xTp4`` is the outer
model the engine builds; it holds the text decoder, whose core picks one of two
targets per step:

- ``PrefillTarget`` for steps that carry context rows;
- ``DecodeTarget`` for generation-only steps (the steps CUDA graphs capture).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Optional

import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.allgather import allgather
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.reducescatter import reducescatter
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    AttentionMetadata,
)
from tensorrt_llm._torch.attention.backends.sparse.glm_kpool.backend import latent_pool_rows
from tensorrt_llm._torch.attention.backends.sparse.glm_kpool.kernels import (
    kpool_expand,
    kpool_score,
)
from tensorrt_llm._torch.attention.backends.sparse.glm_kpool.params import (
    GlmKpoolBackendForwardArgs,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM
from tensorrt_llm._torch.models.modeling_glm5_next import (
    LINEAR_ATTENTION,
    Glm5NextForCausalLM,
    Glm5NextLinearAttention,
    Glm5NextModel,
    Glm5NextMoE,
    Glm5NextRuntimeContext,
    Glm5NextSparseAttention,
    _normalize_glm5_next_top_config,
    build_glm5_next_runtime_context,
    glm5_next_is_quantized,
    glm5_next_text_quant_config,
    resolve_glm5_next_schedule,
)
from tensorrt_llm._torch.models.modeling_glm5_next_vision import Glm5NextVLM
from tensorrt_llm._torch.models.modeling_speculative import SpecDecOneEngineForCausalLM
from tensorrt_llm._torch.modules.kimi_kda.kimi_kda_mixer import _KdaPrequantizedCore
from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.kernel_inputs import extract_routing_params
from tensorrt_llm._torch.pyexecutor.config_utils import unwrap_glm5_next_text_config
from tensorrt_llm._torch.speculative import SpecMetadata
from tensorrt_llm._utils import mpi_disabled
from tensorrt_llm.logger import logger

from . import weights as _weights
from .ops.gemm.low_m_bf16_gemm import low_m_bf16_gemm
from .ops.moe.decode_moe import MAX_ROWS as _DECODE_MOE_ROWS
from .ops.moe.decode_moe import decode_moe

# The deployment this target is written for; route() selects it on the same facts,
# and the constructor states them again for a direct construction.
_SMS = ((10, 0), (10, 3))
_TP = 4


def _fp8_linear(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    """``x @ weight.T`` for a 128 x 128 block-FP8 weight: 1 x 128 activation quantization and
    the CuTe DSL Blackwell blockwise GEMM, the pair the block-FP8 ``Linear`` runs."""
    x_fp8, x_scale = torch.ops.trtllm.fp8_quantize_1x128(x)
    return torch.ops.trtllm.cute_dsl_fp8_gemm_blackwell(x_fp8, weight, x_scale, weight_scale)


class PrefillTarget:
    """Forwards that carry context rows: the decoder stack over the packed batch.

    A step of at least ``SP_MIN_TOKENS`` rows runs sequence parallel. Rank ``r`` owns
    rows ``[r * n, (r + 1) * n)`` of the packed batch (``n = ceil(T / 4)``) and keeps
    the mHC streams of those rows only, so every boundary (post-mapping, pre-mapping,
    RMSNorm) runs over a quarter of the rows. Each sublayer is all-gather, the
    sublayer over all rows, reduce-scatter: the same bytes as the all-reduce it
    replaces. KDA, the MoE and the dense MLPs produce head- or intermediate-split
    partials as under TP. The MLA layers compute their context rows sequence
    parallel: all 64 heads for the owned rows only (indexer scoring, top-k, sparse
    attention, o_proj), and zero for the others, so the reduce-scatter leaves each
    owned row's full value. Under TP the same rows cost four times the KV gathering,
    once per 16-head shard, and every rank scored every row. Generation rows of a
    mixed step stay head-parallel. Smaller steps run the built-in forward.
    """

    SP_MIN_TOKENS = 2048

    def __init__(self, core: "Glm53FlashFp8Sm10xTp4Core") -> None:
        self.core = core
        mapping = core.model_config.mapping
        self.tp_size = mapping.tp_size
        self.tp_rank = mapping.tp_rank
        self.group = list(mapping.tp_group)
        self.process_group = mapping.tp_group_pg.boxed() if mpi_disabled() else None

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        num_tokens = (input_ids if input_ids is not None else inputs_embeds).shape[0]
        if num_tokens < self.SP_MIN_TOKENS:
            return Glm5NextModel.forward(
                self.core,
                attn_metadata=attn_metadata,
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
            )
        return self._forward_sequence_parallel(attn_metadata, input_ids, inputs_embeds)

    def _gather(self, local: torch.Tensor, num_tokens: int) -> torch.Tensor:
        return allgather(local, None, self.group, self.process_group)[:num_tokens]

    def _scatter(self, partial: torch.Tensor, rows: int) -> torch.Tensor:
        pad = rows * self.tp_size - partial.shape[0]
        if pad:
            partial = torch.nn.functional.pad(partial, (0, 0, 0, pad))
        return reducescatter(partial, None, self.group, self.process_group)

    def _forward_sequence_parallel(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor],
        inputs_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        core = self.core
        if inputs_embeds is None:
            inputs_embeds = core.embed_tokens(input_ids)
        num_tokens = inputs_embeds.shape[0]
        rows = -(-num_tokens // self.tp_size)
        first_row = self.tp_rank * rows
        own = inputs_embeds[first_row : first_row + rows]
        if own.shape[0] < rows:
            own = torch.nn.functional.pad(own, (0, 0, 0, rows - own.shape[0]))
        runtime_ctx = build_glm5_next_runtime_context(attn_metadata)

        num_layers = core.schedule.num_layers
        first = core.layers[0]
        residual = core.expand_streams(own)
        post, comb, x = first.hc_attn.pre_mapping(residual)
        x = first.input_layernorm(x)
        for layer_idx in range(num_layers):
            layer = core.layers[layer_idx]
            full = self._gather(x, num_tokens)
            if layer.attention_type == LINEAR_ATTENTION:
                partial = _kda_partial(layer.self_attn, full, attn_metadata)
            else:
                partial = self._mla_partial(layer_idx, layer.self_attn, full, runtime_ctx, rows)
            del full
            residual, post, comb, x = layer.hc_ffn.fused_hc(
                self._scatter(partial, rows),
                residual,
                post,
                comb,
                norm_weight=layer.post_attention_layernorm.weight,
                norm_eps=layer.post_attention_layernorm.variance_epsilon,
            )
            del partial
            partial = self._ffn_partial(layer.mlp, x, num_tokens)
            mlp_out = self._scatter(partial, rows)
            del partial
            if layer_idx + 1 < num_layers:
                nxt = core.layers[layer_idx + 1]
                residual, post, comb, x = nxt.hc_attn.fused_hc(
                    mlp_out,
                    residual,
                    post,
                    comb,
                    norm_weight=nxt.input_layernorm.weight,
                    norm_eps=nxt.input_layernorm.variance_epsilon,
                )
            else:
                residual = layer.hc_ffn.post_mapping(mlp_out, residual, post, comb)
            del mlp_out
        return self._gather(core.collapse_streams(residual), num_tokens)

    def _ffn_partial(self, mlp, x: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """The FFN's intermediate-split output for every row, from this rank's rows ``x``.

        Every FFN GEMM input is the 1 x 128 block-FP8 quantization of the row, a per-row
        function: each rank quantizes its own rows (and runs the router GEMM on them), and
        the gather moves FP8 rows, their scales and the router logits instead of BF16 rows
        that every rank would quantize again, once for the routed and once for the shared
        expert.
        """
        x_fp8, x_scale = torch.ops.trtllm.fp8_quantize_1x128(x)
        x_fp8 = self._gather(x_fp8.view(torch.uint8), num_tokens).view(torch.float8_e4m3fn)
        # [hidden / 128, rows] K-block-major scales: gathered row-major, returned to that layout.
        x_scale = self._gather(x_scale.t().contiguous(), num_tokens).t().contiguous()
        if isinstance(mlp, Glm5NextMoE):
            logits = self._gather(mlp.gate(x), num_tokens)
            return _routed_from_fp8(mlp, x_fp8, x_scale, logits) + _gated_mlp_from_fp8(
                mlp.shared_experts, x_fp8, x_scale
            )
        return _gated_mlp_from_fp8(mlp, x_fp8, x_scale)

    def _mla_partial(
        self,
        layer_idx: int,
        attn: Glm5NextSparseAttention,
        hidden_states: torch.Tensor,
        ctx: Glm5NextRuntimeContext,
        rows: int,
    ) -> torch.Tensor:
        """This rank's contribution to the MLA output of every row: the full value of its
        own context rows, zero for the other context rows, and the head-shard partial of
        the generation rows."""
        out = hidden_states.new_zeros(hidden_states.shape)
        num_ctx = ctx.num_ctx_tokens
        if ctx.num_contexts:
            lo = min(self.tp_rank * rows, num_ctx)
            hi = min(lo + rows, num_ctx)
            self._mla_context_rows(layer_idx, attn, hidden_states[:num_ctx], ctx, lo, hi, out)
        if ctx.num_generations:
            generation = hidden_states[num_ctx:]
            lengths = ctx.kv_lens[ctx.num_contexts :]
            if ctx.gen_tokens_per_request == 1:
                out[num_ctx:] = attn.forward_decode(
                    generation, kv_lens=lengths, metadata=ctx.metadata, reduce=False
                )
            else:
                out[num_ctx:] = attn.forward_verify(
                    generation,
                    kv_lens=lengths,
                    metadata=ctx.metadata,
                    tokens_per_request=ctx.gen_tokens_per_request,
                    reduce=False,
                )
        return out

    def _mla_context_rows(
        self,
        layer_idx: int,
        attn: Glm5NextSparseAttention,
        hidden_states: torch.Tensor,
        ctx: Glm5NextRuntimeContext,
        lo: int,
        hi: int,
        out: torch.Tensor,
    ) -> None:
        """Append every context row's latent and indexer state, then attend rows
        ``[lo, hi)`` with all heads and write their o_proj output into ``out``."""
        indexer = attn.indexer
        backend = attn.attn_backend
        metadata = ctx.metadata
        kpool = indexer.index_kpool
        sched = ctx.context_rows(kpool, hidden_states.device)
        # The cache holds every row, so the latent and indexer-key projections do too.
        q_resid, latent, packed, index_weights = attn._project_attention_and_indexer(hidden_states)
        backend.append_paged_state(
            latent, packed, sched.positions, metadata, request_ids=sched.request_ids
        )
        if sched.final_positions.shape[0]:
            backend.update_pool_keys(
                sched.final_positions,
                indexer.index_kpool_compress_ape,
                metadata,
                request_ids=sched.final_request_ids,
            )
        if lo >= hi:
            return
        full = self.core.mla_full[layer_idx]
        num_rows = hi - lo
        q_resid = q_resid[lo:hi]
        positions = sched.positions[lo:hi]
        request_ids = sched.request_ids[lo:hi]
        visible = positions + 1
        max_visible = max(
            int(ctx.cached_lens[i]) + int(ctx.ctx_cu_seqlens[i + 1]) - int(ctx.ctx_cu_seqlens[i])
            for i in range(ctx.num_contexts)
        )
        q_index = indexer.wq_b(q_resid).view(num_rows, indexer.n_heads, indexer.head_dim)
        scores = backend.score_pools(
            q_index,
            index_weights[lo:hi],
            metadata,
            q_scale=indexer.softmax_scale,
            w_scale=indexer.head_mix_scale,
            kv_lens=visible,
            rows_per_request=1,
            request_ids=request_ids,
            max_kv_len=max_visible,
        )
        num_cand = (visible // kpool).to(torch.int32)
        selected = torch.empty(num_rows, indexer.select_k, dtype=torch.int32, device=scores.device)
        indexer.pool_top_k(
            scores,
            selected,
            is_prefill=False,
            sequence_lengths=num_cand,
            scan_lengths=num_cand,
        )
        topk_rows = backend.expand_selection(
            selected, metadata, kv_lens=visible, rows_per_request=1, request_ids=request_ids
        )
        num_heads = full.w_k.shape[0]
        query = _fp8_linear(q_resid, full.q_b, full.q_b_scale).view(
            num_rows, num_heads, attn.qk_head_dim
        )
        q_latent = torch.bmm(query.transpose(0, 1), full.w_k).transpose(0, 1)
        out_latent = backend.forward(
            q_latent,
            None,
            None,
            metadata,
            AttentionForwardArgs(
                attention_input_type=AttentionInputType.context_only,
                sparse_backend_args=GlmKpoolBackendForwardArgs(topk_rows=topk_rows),
            ),
        )
        per_head = out_latent.view(num_rows, num_heads, attn.kv_lora_rank)
        values = torch.bmm(per_head.transpose(0, 1), full.w_v_t).transpose(0, 1)
        out[lo:hi] = _fp8_linear(values.reshape(num_rows, -1), full.o, full.o_scale)


def _kda_partial(attn, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata):
    """The KDA mixer's row-split o_proj output, before its all-reduce."""
    core = attn._forward_impl(hidden_states, attn_metadata)
    if isinstance(core, _KdaPrequantizedCore):
        return attn.o_proj.forward_prequantized(core.activation, core.scale)
    return attn.o_proj(core.reshape(-1, attn.proj_size))


def _routed_from_fp8(
    mlp: Glm5NextMoE, x_fp8: torch.Tensor, x_scale: torch.Tensor, logits: torch.Tensor
) -> torch.Tensor:
    """The routed experts' intermediate-split output from pre-quantized rows: the
    TRTLLM-Gen FP8 block-scale runner the MoE backend calls, with the same arguments."""
    backend = mlp.experts.backend
    assert backend.num_fused_shared_expert == 0
    routing = extract_routing_params(backend.routing_method)
    return backend.op_backend.run_fp8_block_scale_moe(
        logits,
        routing.routing_bias,
        x_fp8,
        x_scale,
        backend.w3_w1_weight,
        backend.w3_w1_weight_scaling_factor,
        backend.w2_weight,
        backend.w2_weight_scaling_factor,
        backend.num_slots,
        routing.top_k,
        backend.num_fused_shared_expert,
        routing.n_group,
        routing.topk_group,
        backend.intermediate_size_per_partition,
        backend.slot_start,
        backend.expert_size_per_partition,
        routing.routed_scaling_factor,
        backend.routing_method.routing_method_type,
        topk_weights=None,
        topk_ids=None,
        gemm1_clamp_limit=backend.act_clamp,
        output=None,
        tune_max_num_tokens=backend.max_num_tokens,
        use_dp=backend.use_dp,
    )


def _gated_mlp_from_fp8(mlp, x_fp8: torch.Tensor, x_scale: torch.Tensor) -> torch.Tensor:
    """A block-FP8 ``GatedMLP`` (``reduce_output=False``) on pre-quantized rows: its
    gate_up GEMM, its (clamped) SwiGLU and its down projection."""
    gate_up = mlp.gate_up_proj
    h1 = torch.ops.trtllm.cute_dsl_fp8_gemm_blackwell(
        x_fp8, gate_up.weight, x_scale, gate_up.weight_scale
    )
    return mlp.down_proj(mlp._apply_activation(h1))


@dataclass
class _VerifySchedule:
    """Per-step row schedule of a verify step, shared by every MLA layer of the forward.

    Row ``b * T + t`` is request ``b``'s draft position ``t``: its cache position, its
    visible length and complete-pool count, and its request's block table (repeated per
    row, as the pool scoring and expansion kernels address it). Built once per forward
    from device lengths and tables only, so CUDA graphs replay it on refreshed metadata.
    """

    positions: torch.Tensor  # [B, T] int64
    visible: torch.Tensor  # [B * T] int64
    num_cand: torch.Tensor  # [B * T] int32
    tables: torch.Tensor  # [B * T, max pages] int32

    @classmethod
    def build(cls, ctx: Glm5NextRuntimeContext, kpool: int) -> "_VerifySchedule":
        tokens = ctx.gen_tokens_per_request
        lengths = ctx.kv_lens[ctx.num_contexts :]
        steps = torch.arange(tokens, device=lengths.device)
        positions = (lengths - tokens).unsqueeze(1) + steps
        visible = (positions + 1).reshape(-1)
        tables = ctx.metadata.mamba_metadata.glm_block_tables[: ctx.kv_lens.shape[0]]
        return cls(
            positions,
            visible,
            (visible // kpool).to(torch.int32),
            tables[ctx.num_contexts :].repeat_interleave(tokens, dim=0),
        )


class DecodeTarget:
    """Generation-only forwards: every row is a generation (or verify) row.

    The built-in single-phase loop with fused mHC boundaries, except that the MLA layers
    of a verify step take ``_mla_verify``: the row schedule they all need is built once
    per step instead of once per layer.
    """

    def __init__(self, core: "Glm53FlashFp8Sm10xTp4Core") -> None:
        self.core = core

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        core = self.core
        ctx = build_glm5_next_runtime_context(attn_metadata)
        if ctx.gen_tokens_per_request == 1 or not core._fused_hc_ok():
            return Glm5NextModel.forward(
                core,
                attn_metadata=attn_metadata,
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
            )
        if inputs_embeds is None:
            inputs_embeds = core.embed_tokens(input_ids)
        sched = None
        num_layers = core.schedule.num_layers
        first = core.layers[0]
        residual = core.expand_streams(inputs_embeds)
        post, comb, x = first.hc_attn.pre_mapping(residual)
        x = first.input_layernorm(x)
        for layer_idx in range(num_layers):
            layer = core.layers[layer_idx]
            if layer.attention_type == LINEAR_ATTENTION:
                attn_out = layer.self_attn(x, attn_metadata)
            else:
                if sched is None:
                    sched = _VerifySchedule.build(ctx, layer.self_attn.indexer.index_kpool)
                attn_out = _mla_verify(layer.self_attn, x, ctx, sched)
            residual, post, comb, x = layer.hc_ffn.fused_hc(
                attn_out,
                residual,
                post,
                comb,
                norm_weight=layer.post_attention_layernorm.weight,
                norm_eps=layer.post_attention_layernorm.variance_epsilon,
            )
            del attn_out
            mlp_out = layer.run_mlp(x, None)
            if layer_idx + 1 < num_layers:
                nxt = core.layers[layer_idx + 1]
                residual, post, comb, x = nxt.hc_attn.fused_hc(
                    mlp_out,
                    residual,
                    post,
                    comb,
                    norm_weight=nxt.input_layernorm.weight,
                    norm_eps=nxt.input_layernorm.variance_epsilon,
                )
            else:
                residual = layer.hc_ffn.post_mapping(mlp_out, residual, post, comb)
            del mlp_out
        return core.collapse_streams(residual)


def _mla_verify(
    attn: Glm5NextSparseAttention,
    hidden_states: torch.Tensor,
    ctx: Glm5NextRuntimeContext,
    sched: _VerifySchedule,
) -> torch.Tensor:
    """``Glm5NextSparseAttention.forward_verify`` and its all-reduce on the step's schedule:
    the same kernels, with the schedule's positions, lengths and repeated tables."""
    indexer = attn.indexer
    backend = attn.attn_backend
    metadata = ctx.metadata
    batch, tokens = sched.positions.shape
    num_rows = batch * tokens
    q_resid, latent, packed, index_weights = attn._project_attention_and_indexer(hidden_states)
    backend.append_paged_state(
        latent.view(batch, tokens, -1), packed.view(batch, tokens, -1), sched.positions, metadata
    )
    backend.update_pool_keys(sched.positions, indexer.index_kpool_compress_ape, metadata)
    query = attn.q_b_proj(q_resid).view(num_rows, attn.num_heads, attn.qk_head_dim)
    q_index = indexer.wq_b(q_resid).view(num_rows, indexer.n_heads, indexer.head_dim)
    state = backend._cache_state(metadata)
    params = backend.sparse_params
    capacity = state.block_tables.shape[1] * state.tokens_per_block
    scores = kpool_score(
        q_index,
        index_weights,
        state.index_pool,
        sched.tables,
        sched.visible,
        state.tokens_per_block,
        num_pools_max=(capacity + params.index_kpool - 1) // params.index_kpool,
        head_dim=params.index_head_dim,
        kpool=params.index_kpool,
        q_scale=indexer.softmax_scale,
        w_scale=indexer.head_mix_scale,
        precision="bf16",
    )
    selected = torch.empty(num_rows, indexer.select_k, dtype=torch.int32, device=scores.device)
    indexer.pool_top_k(
        scores,
        selected,
        is_prefill=False,
        sequence_lengths=sched.num_cand,
        scan_lengths=sched.num_cand,
    )
    _, base_row, rows_per_slot = latent_pool_rows(state.latent_pool)
    topk_rows = kpool_expand(
        selected,
        sched.visible,
        sched.tables,
        state.tokens_per_block,
        base_row=base_row,
        rows_per_slot=rows_per_slot,
        kpool=params.index_kpool,
        out_width=params.kernel_output_width,
    )
    out_latent = backend.forward(
        attn.absorb_query(query),
        None,
        None,
        metadata,
        AttentionForwardArgs(
            attention_input_type=AttentionInputType.generation_only,
            sparse_backend_args=GlmKpoolBackendForwardArgs(topk_rows=topk_rows),
        ),
    )
    return attn.project_output_latent(out_latent, reduce=True)


# Decode rows (MTP-3 verify of up to eight requests); prefill keeps the module GEMMs.
_LOW_M_ROWS = 32


class _KdaLowMProjections(Glm5NextLinearAttention):
    """The KDA mixer with its small-M BF16 projections on ``low_m_bf16_gemm``.

    Decode and verify steps run the ``[f_a | g_a | b]`` gate projection, which sits on
    the critical path of every KDA layer (its aux-stream chain outlasts the qkv GEMM),
    and the row-parallel o_proj on the FlashInfer low-M kernels. Larger row counts take
    the parent's GEMMs.
    """

    def _project_gate_inputs(self, x: torch.Tensor):
        rows = x.reshape(-1, x.shape[-1])
        if rows.shape[0] > _LOW_M_ROWS:
            return super()._project_gate_inputs(x)
        hd, H = self.head_dim, self.num_heads
        bfa = low_m_bf16_gemm(rows, self._bfa_proj_weight).view(*x.shape[:-1], -1)
        beta = bfa[..., 2 * hd : 2 * hd + H]
        lead = bfa.shape[:-1]
        pair = bfa[..., : 2 * hd].reshape(-1, 2, hd).transpose(0, 1)
        gates = torch.bmm(pair, self._gate_b_t)  # [2, N, D]
        return beta, gates[0].reshape(*lead, -1), gates[1].reshape(*lead, -1)

    def _project_output(self, core) -> torch.Tensor:
        if isinstance(core, _KdaPrequantizedCore):
            return super()._project_output(core)
        flat = core.reshape(-1, self.proj_size)
        if flat.shape[0] > _LOW_M_ROWS:
            return super()._project_output(core)
        out = low_m_bf16_gemm(flat, self.o_proj.weight)
        if self._o_allreduce is not None:
            out = self._o_allreduce(out)
        return out


def _bf16_weight(linear: torch.nn.Module) -> torch.Tensor:
    """The Linear's weight in BF16: as stored, or dequantized from 128 x 128 block FP8."""
    weight = linear.weight
    if weight.dtype == torch.bfloat16:
        return weight
    assert weight.dtype == torch.float8_e4m3fn, f"unexpected weight dtype {weight.dtype}"
    n, k = weight.shape
    scale = linear.weight_scale
    assert scale.shape == (n // 128, k // 128), f"block scales {tuple(scale.shape)} for {n} x {k}"
    blocks = weight.float().view(n // 128, 128, k // 128, 128) * scale.float()[:, None, :, None]
    return blocks.view(n, k).to(torch.bfloat16)


def _low_m_linear(module: torch.nn.Module) -> None:
    """Run a Linear of at most ``_LOW_M_ROWS`` rows on ``low_m_bf16_gemm``.

    A block-FP8 weight gets a dequantized BF16 copy for those rows: the decode-sized
    product reads twice the bytes but skips the activation quantization, and the
    low-M kernel is still faster than quantize + blockwise FP8 GEMM there; larger
    row counts keep the FP8 GEMM.
    """
    full_rank = module.forward
    weight = _bf16_weight(module)

    def forward(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        rows = x.reshape(-1, x.shape[-1])
        if args or kwargs or rows.shape[0] > _LOW_M_ROWS:
            return full_rank(x, *args, **kwargs)
        return low_m_bf16_gemm(rows, weight).view(*x.shape[:-1], -1)

    module.forward = forward


def _low_m_lm_head(lm_head) -> None:
    """Run the vocabulary-parallel LM head GEMM of at most ``_LOW_M_ROWS`` rows (the verify
    logits and every MTP draft head) on ``low_m_bf16_gemm``; the head's gather and slicing
    around it are unchanged."""
    full_rank = lm_head.apply_linear

    def apply_linear(input, bias, lora_params=None, layer_idx=None):
        rows = input.reshape(-1, input.shape[-1])
        if (
            bias is not None
            or lora_params is not None
            or rows.shape[0] > _LOW_M_ROWS
            or lm_head._writes_output_dtype()
        ):
            return full_rank(input, bias, lora_params, layer_idx)
        return low_m_bf16_gemm(rows.contiguous(), lm_head.weight).view(*input.shape[:-1], -1)

    lm_head.apply_linear = apply_linear


def _use_low_m_projections(layer) -> None:
    attn = layer.self_attn
    if isinstance(attn, Glm5NextLinearAttention):
        assert not attn.use_full_rank_gate and attn._bfa_proj_weight is not None
        assert attn.o_proj.weight.dtype == torch.bfloat16 and attn.o_proj.bias is None
        attn.__class__ = _KdaLowMProjections
    elif isinstance(attn, Glm5NextSparseAttention):
        for linear in (
            attn.kv_a_proj_with_mqa,
            attn.q_b_proj,
            attn.o_proj,
            attn.indexer.wk_gate_wp,
            attn.indexer.wq_b,
        ):
            assert linear.bias is None
            _low_m_linear(linear)


class _DecodeMoe(Glm5NextMoE):
    """The MoE branch with decode steps of at most 4 rows on ``decode_moe``.

    Those steps take the router GEMM and three Triton launches (top-8, FC1 + SwiGLU,
    FC2 + combine, the shared expert as the ninth slot) in place of the activation
    quantization, routing, two grouped GEMMs, activation, finalize, the shared expert's
    GEMMs and the partial sum. Larger steps take the grouped-GEMM path.
    """

    def forward(
        self, x: torch.Tensor, all_rank_num_tokens: list[int] | None = None
    ) -> torch.Tensor:
        flat = x.reshape(-1, self.hidden_size)
        if flat.shape[0] > _DECODE_MOE_ROWS:
            return super().forward(x, all_rank_num_tokens)
        backend = self.experts.backend
        out = decode_moe(
            flat.contiguous(),
            self.gate(flat),
            self.gate.e_score_correction_bias,
            self.decode_scaling,
            self.swiglu_limit,
            backend.w3_w1_weight,
            backend.w3_w1_weight_scaling_factor,
            backend.w2_weight,
            backend.w2_weight_scaling_factor,
            *self.decode_shared,
        )
        if self.moe_all_reduce is not None:
            out = self.moe_all_reduce(out)
        return out.view_as(x)


@torch.no_grad()
def _use_decode_moe(moe: Glm5NextMoE, routed_scaling_factor: float) -> None:
    """Give ``moe`` the decode path, with its shared expert in the routed experts' layout."""
    backend = moe.experts.backend
    if backend.w3_w1_weight.dtype != torch.float8_e4m3fn or backend.num_fused_shared_expert:
        return
    shared = moe.shared_experts
    gate_up, gate_up_scale = shared.gate_up_proj.weight, shared.gate_up_proj.weight_scale
    assert gate_up.dtype == torch.float8_e4m3fn and gate_up.shape == backend.w3_w1_weight.shape[1:]
    # GatedMLP stores [gate; up], the routed experts [up (w3); gate (w1)].
    gate, up = gate_up.chunk(2, dim=0)
    gate_s, up_s = gate_up_scale.chunk(2, dim=0)
    moe.decode_shared = (
        torch.cat([up, gate]).contiguous(),
        torch.cat([up_s, gate_s]).float().contiguous(),
        shared.down_proj.weight.contiguous(),
        shared.down_proj.weight_scale.float().contiguous(),
    )
    moe.decode_scaling = float(routed_scaling_factor)
    moe.__class__ = _DecodeMoe


class Glm53FlashFp8Sm10xTp4Core(Glm5NextModel):
    """The decoder stack, with the per-step choice between the two targets."""

    def __init__(self, model_config: ModelConfig[PretrainedConfig]) -> None:
        super().__init__(model_config)
        self._prefill: Optional[PrefillTarget] = None
        self._decode: Optional[DecodeTarget] = None

    @torch.no_grad()
    def post_load_weights(self) -> None:
        """Build the targets once the weights are final. Reached more than once (the
        shell and the loader's module walk), so it only builds once."""
        if self._decode is not None:
            return
        # Every decoder layer, the MTP layer's included (it is appended to ``self.layers``).
        for layer in self.layers:
            _use_low_m_projections(layer)
            if isinstance(getattr(layer, "mlp", None), Glm5NextMoE):
                _use_decode_moe(layer.mlp, float(self.config.routed_scaling_factor))
        self._prefill = PrefillTarget(self)
        self._decode = DecodeTarget(self)

    def _select_target(self, attn_metadata: AttentionMetadata) -> "PrefillTarget | DecodeTarget":
        if self._decode is None:
            self.post_load_weights()
        return self._prefill if attn_metadata.num_contexts > 0 else self._decode

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self._select_target(attn_metadata).forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            **kwargs,
        )


class Glm53FlashFp8Sm10xTp4ForCausalLM(Glm5NextForCausalLM):
    """The text decoder with the checkpoint's MTP layer as its one-model drafter."""

    def __init__(self, model_config: ModelConfig[PretrainedConfig]) -> None:
        text_config = unwrap_glm5_next_text_config(model_config.pretrained_config)
        model_config = glm5_next_text_quant_config(model_config)
        assert not bool(getattr(text_config, "tie_word_embeddings", False)), (
            "the checkpoint's lm_head is its own weight, not the embedding"
        )
        _normalize_glm5_next_top_config(model_config.pretrained_config)
        SpecDecOneEngineForCausalLM.__init__(
            self,
            Glm53FlashFp8Sm10xTp4Core(model_config),
            model_config,
            hidden_size=int(text_config.hidden_size),
            vocab_size=int(text_config.vocab_size),
        )
        self.text_config = text_config
        self.schedule = resolve_glm5_next_schedule(model_config.pretrained_config)
        self.mtp_layers = ()
        spec_config = model_config.spec_config
        if spec_config is not None and spec_config.spec_dec_mode.is_mtp_one_model():
            mtp_layers = tuple(self.draft_model.mtp_layers)
            assert len(mtp_layers) == 1, f"one MTP layer (layer 45), not {len(mtp_layers)}"
            # The draft layer is reached as model.layers[45], the checkpoint's own name.
            self.model.layers.extend(mtp_layers)
            self.mtp_layers = mtp_layers
        self.quantized = glm5_next_is_quantized(model_config)
        assert self.quantized, "the block-FP8 checkpoint loads quantized"

    def post_load_weights(self) -> None:
        super().post_load_weights()
        self.model.post_load_weights()
        if not getattr(self, "_lm_head_low_m", False):
            assert self.lm_head.weight.dtype == torch.bfloat16
            _low_m_lm_head(self.lm_head)
            self._lm_head_low_m = True


class ModelingV2Glm53FlashFp8Sm10xTp4(Glm5NextVLM):
    """GLM-5.3-Flash block-FP8, sm_100 / sm_103, tp4, text only."""

    def __init__(self, model_config: ModelConfig[PretrainedConfig], *args, **kwargs) -> None:
        mapping = model_config.mapping
        assert torch.cuda.get_device_capability() in _SMS, (
            f"this target runs on sm100 / sm103, not {torch.cuda.get_device_capability()}"
        )
        assert (
            mapping.world_size == _TP
            and mapping.tp_size == _TP
            and mapping.moe_tp_size == _TP
            and not mapping.enable_attention_dp
            and mapping.pp_size == 1
        ), f"this target is tp4 with the experts split by intermediate, not {mapping}"
        assert model_config.disable_mm_encoder, "this target serves the text model only"
        config = model_config.pretrained_config
        self.original_arch = config.architectures[0]
        # PreTrainedModel's constructor, without the built-in wrapper's: that one builds
        # the built-in text decoder through the registry.
        super(Glm5NextVLM, self).__init__(config)
        self.model_config = model_config

        llm_model_config = copy.deepcopy(model_config)
        # Share the live extra_attrs dict: the attention layers register their per-layer
        # metadata there and the engine reads the same dict.
        llm_model_config.extra_attrs = model_config.extra_attrs
        llm_model_config._frozen = False
        llm_model_config.pretrained_config.architectures = ["Glm5NextForCausalLM"]
        llm_model_config.modeling_v2_target = Glm53FlashFp8Sm10xTp4ForCausalLM
        llm_model_config._frozen = True
        self.llm = AutoModelForCausalLM.from_config(llm_model_config)
        self.mm_encoder = None
        # device='cpu' keeps the constant real under meta-device construction.
        self._mm_token_ids = torch.tensor(
            [int(config.image_token_id), int(config.video_token_id)],
            dtype=torch.int32,
            device="cpu",
        )
        self.post_config()
        logger.info(f"modeling_v2: {type(self).__name__} built")

    def load_weights(self, weights: Any, **_ignored: Any) -> None:
        _weights.load_weights(self.llm, weights)
