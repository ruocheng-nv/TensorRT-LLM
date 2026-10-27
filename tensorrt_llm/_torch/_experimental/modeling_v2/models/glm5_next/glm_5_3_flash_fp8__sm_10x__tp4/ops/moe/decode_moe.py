# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MoE branch of a decode step of at most 4 rows in three Triton launches (after the router GEMM).

``topk``: sigmoid scores, FP32 correction bias, top-8, normalization and the routed scaling
factor, one program per row. ``fc1``: one program per (row, expert slot, 32-wide block of the
intermediate) computes the up and gate products of the block-FP8 weight with the BF16 row in
FP32 (each 128 x 128 weight block times its scale) and the clamped SwiGLU. Slot 8 is the
shared expert, weight 1. ``fc2``: one program per (row, 32-wide block of the hidden) sums the
nine slots' down products, each weighted, and stores BF16. The rank's result is the moe_tp
partial the caller all-reduces.

The BF16 activations are never quantized, so this computes the reference product more
closely than the FP8-activation grouped GEMM it replaces at these sizes. All three launches use
programmatic dependent launch (``TRTLLM_ENABLE_PDL``): each waits for its predecessor before
reading what that predecessor wrote; the top-k releases its dependents at once, FC1 and FC2
(several waves each) only after their weight streams.
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
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

MAX_ROWS = 4
_TOPK = 8
_SLOTS = _TOPK + 1


@triton.jit
def _topk_kernel(logits_ptr, bias_ptr, ids_ptr, wts_ptr, scaling, E: tl.constexpr, EP: tl.constexpr,
                 TOPK: tl.constexpr, SLOTS: tl.constexpr):
    row = tl.program_id(0)
    e = tl.arange(0, EP)
    valid = e < E
    # PDL: let the next launch start, then wait for the router GEMM's logits.
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    logit = tl.load(logits_ptr + row * E + e, mask=valid, other=0.0).to(tl.float32)
    score = 1.0 / (1.0 + tl.exp(-logit))
    choice = tl.where(valid, score + tl.load(bias_ptr + e, mask=valid, other=0.0), float("-inf"))
    total = 0.0
    for i in tl.static_range(TOPK):
        best = tl.max(choice, axis=0)
        idx = tl.min(tl.where(choice == best, e, EP), axis=0)
        w = tl.sum(tl.where(e == idx, score, 0.0), axis=0)
        tl.store(ids_ptr + row * SLOTS + i, idx)
        tl.store(wts_ptr + row * SLOTS + i, w)
        total += w
        choice = tl.where(e == idx, float("-inf"), choice)
    for i in tl.static_range(TOPK):
        w = tl.load(wts_ptr + row * SLOTS + i)
        tl.store(wts_ptr + row * SLOTS + i, w / total * scaling)
    tl.store(ids_ptr + row * SLOTS + TOPK, -1)
    tl.store(wts_ptr + row * SLOTS + TOPK, 1.0)


@triton.jit
def _fc1_kernel(x_ptr, ids_ptr, w_ptr, s_ptr, sw_ptr, ss_ptr, act_ptr, limit, H: tl.constexpr,
                I: tl.constexpr, SLOTS: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pair = tl.program_id(0)
    pid_n = tl.program_id(1)
    row = pair // SLOTS
    tl.extra.cuda.gdc_wait()
    e = tl.load(ids_ptr + pair).to(tl.int64)
    shared = e < 0
    n = pid_n * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    # Routed [w3 (up); w1 (gate)] rows of expert e, or the shared expert's [up; gate] copy.
    base = tl.where(shared, sw_ptr, w_ptr + tl.maximum(e, 0) * (2 * I * H))
    sbase = tl.where(shared, ss_ptr, s_ptr + tl.maximum(e, 0) * (2 * I // 128) * (H // 128))
    up_row = base + n[:, None] * H
    gate_row = up_row + I * H
    s_up = sbase + (pid_n * BN // 128) * (H // 128)
    s_gate = s_up + (I // 128) * (H // 128)
    acc_u = tl.zeros((BN,), tl.float32)
    acc_g = tl.zeros((BN,), tl.float32)
    for kb in range(0, H, BK):
        x = tl.load(x_ptr + row * H + kb + k).to(tl.float32)
        su = tl.load(s_up + (kb + k) // 128)
        sg = tl.load(s_gate + (kb + k) // 128)
        acc_u += tl.sum(tl.load(up_row + kb + k[None, :]).to(tl.float32) * (x * su)[None, :], axis=1)
        acc_g += tl.sum(tl.load(gate_row + kb + k[None, :]).to(tl.float32) * (x * sg)[None, :], axis=1)
    # Released after the weight stream: earlier, the dependents' CTAs would occupy SMs this
    # multi-wave grid still needs.
    tl.extra.cuda.gdc_launch_dependents()
    g = tl.minimum(acc_g, limit)
    u = tl.minimum(tl.maximum(acc_u, -limit), limit)
    tl.store(act_ptr + pair * I + n, g / (1.0 + tl.exp(-g)) * u)


@triton.jit
def _fc2_kernel(act_ptr, ids_ptr, wts_ptr, w_ptr, s_ptr, sw_ptr, ss_ptr, out_ptr, H: tl.constexpr,
                I: tl.constexpr, SLOTS: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(0)
    pid_n = tl.program_id(1)
    n = pid_n * BN + tl.arange(0, BN)
    j = tl.arange(0, I)
    acc = tl.zeros((BN,), tl.float32)
    tl.extra.cuda.gdc_wait()
    for slot in range(SLOTS):
        pair = row * SLOTS + slot
        e = tl.load(ids_ptr + pair).to(tl.int64)
        shared = e < 0
        base = tl.where(shared, sw_ptr, w_ptr + tl.maximum(e, 0) * (H * I))
        sbase = tl.where(shared, ss_ptr, s_ptr + tl.maximum(e, 0) * (H // 128) * (I // 128))
        a = tl.load(act_ptr + pair * I + j)
        s = tl.load(sbase + (pid_n * BN // 128) * (I // 128) + j // 128)
        w = tl.load(base + n[:, None] * I + j[None, :]).to(tl.float32)
        acc += tl.sum(w * (a * s)[None, :], axis=1) * tl.load(wts_ptr + pair)
    tl.extra.cuda.gdc_launch_dependents()
    tl.store(out_ptr + row * H + n, acc.to(tl.bfloat16))


class DecodeMoe(OpWrapper):
    """``scaling * sum_i w_i * expert_{e_i}(x) + shared(x)`` for each of at most 8 rows.

    ``logits`` are the router's FP32 ``[M, E]`` outputs; ``bias`` the FP32 correction bias.
    ``w31`` / ``s31`` are the routed experts' ``[E, 2I, H]`` FP8 ``[up; gate]`` weights and
    ``[E, 2I/128, H/128]`` scales, ``w2`` / ``s2`` the ``[E, H, I]`` down weights and scales;
    ``sw31``, ``ss31``, ``sw2``, ``ss2`` the shared expert in the same per-expert layout.
    Returns BF16 ``[M, H]``.
    """

    ARCHS = frozenset({Arch.SM_100})

    # GLM-5.3-Flash at moe_tp4: hidden 4096, intermediate 2048 / 4 = 512, 288 experts.
    CELLS: tuple[Cell, ...] = (
        Cell(why="MTP draft row", spec=dict(m=1, e=288, h=4096, i=512)),
        Cell(why="MTP-3 verify of one request", spec=dict(m=4, e=288, h=4096, i=512)),
    )

    note = """
    Top-8 ties go to the lower expert index. The FC1 activations stay FP32 between the two
    launches ([M * 9, I] scratch). Rows beyond 4 are the caller's to route elsewhere: every
    (row, slot) reads its expert's weights, so as rows start sharing experts the grouped GEMM's
    one read per expert wins (measured: 8 rows already slower).
    """

    def __call__(self, x, logits, bias, scaling, limit, w31, s31, w2, s2, sw31, ss31, sw2, ss2):
        m, h = x.shape
        num_experts = logits.shape[1]
        inter = w2.shape[2]
        ids = torch.empty((m, _SLOTS), dtype=torch.int32, device=x.device)
        wts = torch.empty((m, _SLOTS), dtype=torch.float32, device=x.device)
        pdl = get_env_enable_pdl()
        _topk_kernel[(m,)](logits, bias, ids, wts, scaling, num_experts,
                           triton.next_power_of_2(num_experts), _TOPK, _SLOTS, num_warps=4,
                           launch_pdl=pdl)
        act = torch.empty((m * _SLOTS, inter), dtype=torch.float32, device=x.device)
        _fc1_kernel[(m * _SLOTS, inter // 32)](x, ids, w31, s31, sw31, ss31, act, limit, h, inter,
                                               _SLOTS, 32, 512, num_warps=4, num_stages=3,
                                               launch_pdl=pdl)
        out = torch.empty((m, h), dtype=torch.bfloat16, device=x.device)
        _fc2_kernel[(m, h // 32)](act, ids, wts, w2, s2, sw2, ss2, out, h, inter, _SLOTS, 32,
                                  num_warps=4, num_stages=3, launch_pdl=pdl)
        return out

    def reference(self, x, logits, bias, scaling, limit, w31, s31, w2, s2, sw31, ss31, sw2, ss2):
        def dequant(w, s):
            r, c = w.shape
            return (w.float().view(r // 128, 128, c // 128, 128) * s[:, None, :, None]).view(r, c)

        def expert(w_up_gate, s_up_gate, w_down, s_down, row):
            h = dequant(w_up_gate, s_up_gate) @ row
            inter = h.shape[0] // 2
            u, g = h[:inter].clamp(-limit, limit), h[inter:].clamp(max=limit)
            return dequant(w_down, s_down) @ (torch.nn.functional.silu(g) * u)

        out = []
        with true_fp32_matmul():
            score = torch.sigmoid(logits.float())
            ids = torch.topk(score + bias.float(), _TOPK, dim=-1).indices
            wts = score.gather(1, ids)
            wts = wts / wts.sum(-1, keepdim=True) * scaling
            for t in range(x.shape[0]):
                row = x[t].float()
                acc = expert(sw31, ss31, sw2, ss2, row)
                for i in range(_TOPK):
                    e = int(ids[t, i])
                    acc = acc + wts[t, i] * expert(w31[e], s31[e], w2[e], s2[e], row)
                out.append(acc)
        return torch.stack(out).to(torch.bfloat16)

    def is_valid(self, x, logits, bias, scaling, limit, w31, s31, w2, s2, sw31, ss31, sw2, ss2):
        m, h = x.shape
        assert 1 <= m <= MAX_ROWS, f"{m} rows; at most {MAX_ROWS}"
        assert x.dtype == torch.bfloat16 and x.is_contiguous()
        assert logits.dtype == torch.float32 and logits.is_contiguous() and bias.dtype == torch.float32
        for t in (w31, w2, sw31, sw2):
            assert t.dtype == torch.float8_e4m3fn and t.is_contiguous()
        for t in (s31, s2, ss31, ss2):
            assert t.dtype == torch.float32 and t.is_contiguous()
        assert h % 256 == 0 and w2.shape[2] % 128 == 0


decode_moe = DecodeMoe()
