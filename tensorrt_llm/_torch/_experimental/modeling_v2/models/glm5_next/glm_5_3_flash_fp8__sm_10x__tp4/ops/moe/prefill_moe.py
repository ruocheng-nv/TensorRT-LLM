# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The routed experts of a prefill step on a grouped block-FP8 GEMM with the checkpoint's scales.

The GEMMs are ``grouped_blockwise_fp8_gemm``: the persistent CuTe DSL blockwise kernel with
each 128-row tile mapped to its expert by a device table, FP32 1 x 128 activation and
128 x 128 weight scales as the checkpoint stores them. On these shapes it runs FC1 and FC2
in about two thirds of the TRTLLM-Gen FP8 MoE's time.

The launches, for ``T`` rows and top-8 routing:

* ``topk`` (the decode MoE's routing kernel): sigmoid scores, FP32 correction bias, top-8,
  normalization and the routed scaling factor.
* The expert counts and their 128-aligned segment starts, in torch, with no host
  synchronization. The buffers are sized for the worst case, ``8 T + 127 E`` rows.
* ``gather``: takes each (row, slot) pair's destination row in its expert's segment with an
  atomic increment, then copies the FP8 row and its scales there.
* The tile table: the expert of every 128-row tile (``searchsorted`` over the segment ends),
  -1 for tiles past an expert's rows; the GEMMs skip those.
* FC1: ``grouped_blockwise_fp8_gemm``.
* ``act``: clamped SwiGLU of the ``[up; gate]`` product, then the 1 x 128 FP8 quantization
  of the intermediate (scale ``amax / 448``).
* FC2: the same grouped GEMM.
* ``combine``: each row's weighted sum of its eight slots, added to ``addend`` (the shared
  expert's output) in FP32 and stored as BF16.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

from .decode_moe import _topk_kernel
from .grouped_blockwise_gemm import grouped_blockwise_fp8_gemm

_TOPK = 8
_ALIGN = 128
_ACT_ROWS = 4


@triton.jit
def _gather_kernel(
    x_ptr,
    xs_ptr,
    xs_stride_row,
    xs_stride_blk,
    ids_ptr,
    cursor_ptr,
    dest_ptr,
    a_ptr,
    as_ptr,
    cap,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    SLOTS: tl.constexpr,
):
    pair = tl.program_id(0)
    row = pair // TOPK
    # The pair's row in its expert's segment; the order within a segment is arbitrary.
    expert = tl.load(ids_ptr + row * SLOTS + pair % TOPK)
    dest32 = tl.atomic_add(cursor_ptr + expert, 1)
    tl.store(dest_ptr + pair, dest32)
    dest = dest32.to(tl.int64)
    words = tl.arange(0, H // 4)
    tl.store(a_ptr + dest * (H // 4) + words, tl.load(x_ptr + row * (H // 4) + words))
    blk = tl.arange(0, H // 128)
    scale = tl.load(xs_ptr + row * xs_stride_row + blk * xs_stride_blk)
    tl.store(as_ptr + blk * cap + dest, scale)


@triton.jit
def _act_kernel(
    h_ptr,
    dest_ptr,
    out_ptr,
    out_scale_ptr,
    pairs,
    cap,
    limit,
    INTER: tl.constexpr,
    NB: tl.constexpr,
    ROWS: tl.constexpr,
):
    p = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    valid = p < pairs
    dest = tl.load(dest_ptr + p, mask=valid, other=0).to(tl.int64)
    j = tl.arange(0, INTER)
    src = h_ptr + dest[:, None] * (2 * INTER) + j[None, :]
    up = tl.load(src, mask=valid[:, None], other=0.0).to(tl.float32)
    gate = tl.load(src + INTER, mask=valid[:, None], other=0.0).to(tl.float32)
    gate = tl.minimum(gate, limit)
    up = tl.minimum(tl.maximum(up, -limit), limit)
    act = (gate / (1.0 + tl.exp(-gate)) * up).reshape(ROWS, NB, 128)
    scale = tl.maximum(tl.max(tl.abs(act), axis=2), 1e-10) / 448.0
    q = (act / scale[:, :, None]).to(tl.float8e4nv).reshape(ROWS, INTER)
    tl.store(out_ptr + dest[:, None] * INTER + j[None, :], q, mask=valid[:, None])
    blk = tl.arange(0, NB)
    tl.store(out_scale_ptr + blk[None, :] * cap + dest[:, None], scale, mask=valid[:, None])


@triton.jit
def _combine_kernel(
    y_ptr,
    dest_ptr,
    wts_ptr,
    addend_ptr,
    out_ptr,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    SLOTS: tl.constexpr,
    BH: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    n = tl.program_id(1) * BH + tl.arange(0, BH)
    acc = tl.load(addend_ptr + row * H + n).to(tl.float32)
    for k in tl.static_range(TOPK):
        dest = tl.load(dest_ptr + row * TOPK + k).to(tl.int64)
        w = tl.load(wts_ptr + row * SLOTS + k)
        acc += w * tl.load(y_ptr + dest * H + n).to(tl.float32)
    tl.store(out_ptr + row * H + n, acc.to(tl.bfloat16))


class PrefillMoe(OpWrapper):
    """``addend + scaling * sum_i w_i * expert_{e_i}(x)`` for every row of a prefill step.

    ``x`` is the FP8 ``[T, H]`` row-wise 1 x 128 quantization of the rows with scales
    ``x_scale`` (``[T, H/128]`` FP32, any strides). ``logits`` are the router's FP32
    ``[T, E]`` outputs, ``bias`` the FP32 correction bias. ``w31`` / ``s31`` are the routed
    experts' ``[E, 2I, H]`` FP8 ``[up; gate]`` weights and their ``[E, 2I/128, H/128]``
    scales, ``w2`` / ``s2`` the ``[E, H, I]`` down weights and scales. ``addend`` is BF16
    ``[T, H]``. Returns BF16 ``[T, H]``.
    """

    ARCHS = frozenset({Arch.SM_100})

    # GLM-5.3-Flash at moe_tp4: hidden 4096, intermediate 2048 / 4 = 512, 288 experts.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="13K-token prompt, sequence-parallel prefill",
            spec=dict(t=13312, e=288, h=4096, i=512),
        ),
        Cell(why="smallest sequence-parallel step", spec=dict(t=2048, e=288, h=4096, i=512)),
    )

    note = """
    The FC1 output is BF16 before the SwiGLU and the intermediate is requantized 1 x 128, as in
    the reference here. Top-8 ties go to the lower expert index.
    """

    def __call__(self, x, x_scale, logits, bias, scaling, limit, w31, s31, w2, s2, addend):
        rows, h = x.shape
        num_experts = logits.shape[1]
        inter = w2.shape[2]
        dev = x.device
        ids = torch.empty((rows, _TOPK + 1), dtype=torch.int32, device=dev)
        wts = torch.empty((rows, _TOPK + 1), dtype=torch.float32, device=dev)
        _topk_kernel[(rows,)](
            logits,
            bias,
            ids,
            wts,
            scaling,
            num_experts,
            triton.next_power_of_2(num_experts),
            _TOPK,
            _TOPK + 1,
            num_warps=4,
        )
        flat = ids[:, :_TOPK].reshape(-1)
        counts = torch.zeros(num_experts, dtype=torch.int32, device=dev)
        counts.scatter_add_(0, flat.long(), torch.ones_like(flat))
        padded = (counts + _ALIGN - 1) // _ALIGN * _ALIGN
        starts = torch.cumsum(padded, 0, dtype=torch.int32) - padded
        ends = starts + counts
        cursor = starts.clone()
        dest = torch.empty_like(flat)

        cap = (rows * _TOPK + num_experts * (_ALIGN - 1) + _ALIGN - 1) // _ALIGN * _ALIGN
        a = torch.empty((cap, h), dtype=torch.float8_e4m3fn, device=dev)
        a_scale = torch.empty((h // 128, cap), dtype=torch.float32, device=dev)
        pairs = rows * _TOPK
        _gather_kernel[(pairs,)](
            x.view(torch.int32),
            x_scale,
            x_scale.stride(0),
            x_scale.stride(1),
            ids,
            cursor,
            dest,
            a.view(torch.int32),
            a_scale,
            cap,
            h,
            _TOPK,
            _TOPK + 1,
            num_warps=4,
        )
        tile_row = torch.arange(0, cap, _ALIGN, device=dev, dtype=torch.int32)
        tile_expert = torch.searchsorted(starts + padded, tile_row, right=True)
        tile_expert = tile_expert.clamp_(max=num_experts - 1).to(torch.int32)
        tile_expert = torch.where(tile_row < ends[tile_expert], tile_expert, -1)
        h1 = torch.empty((cap, 2 * inter), dtype=torch.bfloat16, device=dev)
        grouped_blockwise_fp8_gemm(a, a_scale, w31, s31, tile_expert, h1)
        del a, a_scale
        act = torch.empty((cap, inter), dtype=torch.float8_e4m3fn, device=dev)
        act_scale = torch.empty((inter // 128, cap), dtype=torch.float32, device=dev)
        _act_kernel[(triton.cdiv(pairs, _ACT_ROWS),)](
            h1, dest, act, act_scale, pairs, cap, limit, inter, inter // 128, _ACT_ROWS, num_warps=4
        )
        del h1
        y = torch.empty((cap, h), dtype=torch.bfloat16, device=dev)
        grouped_blockwise_fp8_gemm(act, act_scale, w2, s2, tile_expert, y)
        del act, act_scale
        out = torch.empty((rows, h), dtype=torch.bfloat16, device=dev)
        _combine_kernel[(rows, h // 512)](
            y, dest, wts, addend, out, h, _TOPK, _TOPK + 1, 512, num_warps=4
        )
        return out

    def reference(self, x, x_scale, logits, bias, scaling, limit, w31, s31, w2, s2, addend):
        def dequant(w, s):
            r, c = w.shape
            return (w.float().view(r // 128, 128, c // 128, 128) * s[:, None, :, None]).view(r, c)

        def quant_1x128(v):
            blocks = v.view(v.shape[0], -1, 128)
            amax = blocks.abs().amax(-1, keepdim=True).clamp(min=1e-10)
            scale = amax / 448.0
            return ((blocks / scale).to(torch.float8_e4m3fn).float() * scale).view_as(v)

        rows = x.shape[0]
        inter = w2.shape[2]
        xf = (x.float().view(rows, -1, 128) * x_scale.float()[:, :, None]).view(rows, -1)
        out = addend.float().clone()
        with true_fp32_matmul():
            score = torch.sigmoid(logits.float())
            ids = torch.topk(score + bias.float(), _TOPK, dim=-1).indices
            wts = score.gather(1, ids)
            wts = wts / wts.sum(-1, keepdim=True) * scaling
            for e in ids.unique().tolist():
                sel, slot = (ids == e).nonzero(as_tuple=True)
                h1 = (xf[sel] @ dequant(w31[e], s31[e]).t()).to(torch.bfloat16).float()
                u = h1[:, :inter].clamp(-limit, limit)
                g = h1[:, inter:].clamp(max=limit)
                act = quant_1x128(torch.nn.functional.silu(g) * u)
                y = (act @ dequant(w2[e], s2[e]).t()).to(torch.bfloat16).float()
                out.index_add_(0, sel, wts[sel, slot, None] * y)
        return out.to(torch.bfloat16)

    def is_valid(self, x, x_scale, logits, bias, scaling, limit, w31, s31, w2, s2, addend):
        rows, h = x.shape
        assert x.dtype == torch.float8_e4m3fn and x.is_contiguous()
        assert x_scale.dtype == torch.float32 and x_scale.shape == (rows, h // 128)
        assert logits.dtype == torch.float32 and logits.is_contiguous() and logits.shape[0] == rows
        assert bias.dtype == torch.float32
        for t in (w31, w2):
            assert t.dtype == torch.float8_e4m3fn and t.is_contiguous()
        for t in (s31, s2):
            assert t.dtype == torch.float32 and t.is_contiguous()
        assert (
            addend.dtype == torch.bfloat16 and addend.shape == (rows, h) and addend.is_contiguous()
        )
        assert h % 512 == 0 and w2.shape[2] % 128 == 0


prefill_moe = PrefillMoe()
