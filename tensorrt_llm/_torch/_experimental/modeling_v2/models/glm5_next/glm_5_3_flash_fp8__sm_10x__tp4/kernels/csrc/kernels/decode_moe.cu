/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Decode MoE (M <= 4 rows): one persistent warp-specialized kernel per call.
//
// Shapes (GLM-5.3-Flash at moe_tp4): H = 4096, I = 512, E = 288, top-8 + the shared expert.
// w31 [E, 2I, H] e4m3 = [up; gate] rows, s31 [E, 2I/128, H/128]; w2 [E, H, I] e4m3, s2 [E, H/128, I/128].
//
// Arithmetic: e4m3 -> f16 (exact), BF16 x -> f16, mma.sync m16n8k16 f16 x f16 -> f32, each 128-wide K block's
// partial scaled by its block scale in f32. The FC1 -> FC2 activation (times the routing weight) is an f16
// hi + lo pair.
//
// K permutation: in a 64-wide K block, lane (g = lane / 4, t4 = lane % 4) reads bytes [16 t4, 16 t4 + 16) of
// its weight rows; MMA chunk j takes bytes 16 t4 + 4 j + {0, 1} as A columns {2 t4, 2 t4 + 1} and + {2, 3} as
// {2 t4 + 8, 2 t4 + 9}; the B fragment reads the activation at the same K positions.
//
// Schedule (one CTA per SM). Distinct experts: u = 0 is the shared expert (known before routing), u = 1..U-1
// the routed ones in first-occurrence order. Every CTA computes the routing itself (one warp per row) while
// its producer already streams a shared-expert unit.
//   FC1 phase: unit = (u, 8-row tile of I) = 8 up rows + 8 gate rows over H, two contiguous 32 KB bulk
//     copies. The first unit of a CTA is unit blockIdx, then units come from a global atomic counter (u-major).
//     8 compute warps (K slice of 512 each, x in registers) -> partials in smem -> an epilogue warp sums them,
//     applies the clamped SwiGLU and the routing weight, stores the f16 hi/lo activation, bumps done[u].
//   FC2 phase (after the CTA's FC1 claims run out): CTA b < 128 owns output rows [32 b, 32 b + 32). It streams
//     the [32 x 512] W2 tile of every expert through the same ring, adding the expert's activation (bulk copy
//     from global) once done[u] is complete; the compute warps accumulate over all experts in registers, then
//     sum the 4 K quarters in a fixed order and store BF16 (deterministic, no atomics on the output).
//   The last CTA to finish resets the global counters for the next call.
#include "decode_moe.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace glm53_fp8
{
namespace
{

constexpr int kH = 4096, kI = 512, kE = 288, kTopK = 8, kMaxM = 4;
constexpr int kMaxU = 1 + kMaxM * kTopK; // 33
constexpr int kUPad = 40;

// warp roles: producer, 4 FC1 epilogue warps (unit k -> warp k % 4, reduction buffer k % 4), 8 compute warps
// and a poller warp (FC2 phase: publishes which experts' activations are complete)
constexpr int kWarpP = 0, kWarpEpi = 1, kNumEpi = 4, kWarpC = kWarpEpi + kNumEpi, kNumC = 8, kWarpPoll = kWarpC + kNumC;
constexpr int kWarps = kWarpPoll + 1, kThreads = kWarps * 32;

// ring of 32 KB slots (+ 128 B side area): an FC1 unit takes two (8 up rows, 8 gate rows over H), an FC2
// stage one (W2 tile [32 x 512], activation, scales)
constexpr int kSlotData = 32768, kSlot = kSlotData + 128, kSlots = 5;
constexpr int kJT = 8, kUnitsPerExpert = kI / kJT; // 64
// FC2: a cluster of 2 CTAs owns 64 output rows; rank r takes the experts u = r, r + 2, ... and the two
// partial sums are added over DSMEM at the end (fixed order: rank 0 + rank 1)
constexpr int kFc2Pairs = 64, kFc2Rows = kH / kFc2Pairs;      // 64
constexpr int kW2Tile = kFc2Rows * kI;                        // 32768 = one slot
constexpr int kActBytes = 2 * kMaxM * kI * 2, kActStages = 5; // [hi: M x 512][lo: M x 512] f16
constexpr int kRedBuf = kNumC * 16 * 16;                      // [warp][16 lanes][float4]
constexpr int kRed2 = 2 * kFc2Rows * 8 * 4;                   // [K half][row][column: hi 0..3, lo 4..7]

constexpr int kOffRing = 0;
constexpr int kOffAct = kOffRing + kSlots * kSlot;
constexpr int kOffRed = kOffAct + kActStages * kActBytes;
constexpr int kOffRed2 = kOffRed + kNumEpi * kRedBuf;
constexpr int kOffPeer = kOffRed2 + kRed2;
constexpr int kSmem = kOffPeer + kFc2Rows * kMaxM * 4;

struct MoeParams
{
    __nv_bfloat16 const* x;
    float const* logits;
    float const* bias;
    uint8_t const* w31;
    float const* s31;
    uint8_t const* sw31;
    float const* ss31;
    uint8_t const* w2;
    float const* s2;
    uint8_t const* sw2;
    float const* ss2;
    __nv_bfloat16* out;
    float scaling, limit;
    int M;
};

__device__ __align__(128) __half g_act[kUPad * 2 * kMaxM * kI]; // [u][hi, lo][M][I] (M-compact)
__device__ __align__(16) int g_done[kUPad];
__device__ int g_ctr;
__device__ int g_exit;

// ---------------------------------------------------------------- PTX helpers
__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint32_t v16)
{
    uint32_t r;
    const uint16_t h = static_cast<uint16_t>(v16);
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(h));
    return r;
}

__device__ __forceinline__ void mma16816(
    float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t word(uint4 const& v, int j)
{
    return j == 0 ? v.x : j == 1 ? v.y : j == 2 ? v.z : v.w;
}

// One 16-K chunk: rows (g, g + 8) weight words wg / wg8, activation pair b0 / b1.
__device__ __forceinline__ void chunk_mma(float (&d)[4], uint32_t wg, uint32_t wg8, uint32_t b0, uint32_t b1)
{
    mma16816(d, e4m3x2_to_f16x2(wg & 0xffffu), e4m3x2_to_f16x2(wg8 & 0xffffu), e4m3x2_to_f16x2(wg >> 16),
        e4m3x2_to_f16x2(wg8 >> 16), b0, b1);
}

__device__ __forceinline__ uint32_t smem_u32(void const* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, int count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}

__device__ __forceinline__ void mbar_init_fence()
{
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar)
{
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}

__device__ __forceinline__ void bulk_g2s(void* dst, void const* src, uint32_t bytes, uint64_t* bar)
{
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(smem_u32(dst)),
        "l"(src), "r"(bytes), "r"(smem_u32(bar))
        : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity)
{
    asm volatile(
        "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}" ::
            "r"(smem_u32(bar)),
        "r"(parity)
        : "memory");
}

__device__ __forceinline__ bool mbar_test(uint64_t* bar, uint32_t parity)
{
    uint32_t ok;
    asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
                 : "=r"(ok)
                 : "r"(smem_u32(bar)), "r"(parity)
                 : "memory");
    return ok != 0;
}

__device__ __forceinline__ void named_bar(int id, int threads)
{
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

__device__ __forceinline__ int4 ld_relaxed_v4(int const* p)
{
    int4 v;
    asm volatile("ld.relaxed.gpu.global.v4.s32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p)
                 : "memory");
    return v;
}

__device__ __forceinline__ void prefetch_l2(void const* src, uint32_t bytes)
{
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
}

__device__ __forceinline__ void cluster_arrive_relaxed()
{
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}

__device__ __forceinline__ void cluster_arrive()
{
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
}

__device__ __forceinline__ void cluster_wait()
{
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

__device__ __forceinline__ void st_cluster_f32(void const* local_smem, uint32_t peer_rank, float v)
{
    uint32_t addr;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(addr) : "r"(smem_u32(local_smem)), "r"(peer_rank));
    asm volatile("st.shared::cluster.f32 [%0], %1;" ::"r"(addr), "f"(v) : "memory");
}

__device__ __forceinline__ void fence_acquire_gpu()
{
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
}

__device__ __forceinline__ void fence_proxy_async()
{
    asm volatile("fence.proxy.async.global;" ::: "memory");
}

__device__ __forceinline__ uint32_t f2ord(float f)
{
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// ---------------------------------------------------------------- kernel
__global__ void __launch_bounds__(kThreads, 1) moe_kernel(MoeParams p)
{
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ __align__(8) uint64_t full[kSlots], empty[kSlots];
    __shared__ __align__(8) uint64_t redFull[kNumEpi], redEmpty[kNumEpi];
    __shared__ __align__(8) uint64_t actFull[kActStages], actEmpty[kActStages];
    __shared__ int
        slot_tag[kSlots]; // >= 0: FC1 unit (this slot: up rows, next slot: gate rows); -2 - u: FC2 stage of u; -1: end
    __shared__ int red_unit[kNumEpi];
    __shared__ int s_route[2 + kMaxU]; // [U, expert of u = 0.. (-1 = shared)]
    __shared__ float s_wtab[kUPad * kMaxM];
    __shared__ int s_pe[kMaxM * kTopK];
    __shared__ float s_ps[kMaxM * kTopK];
    __shared__ float s_sc[kMaxM * kE]; // sigmoid scores
    __shared__ int s_ready;            // FC2 phase: number of experts (prefix) whose activation is complete
    __shared__ int s_fc2go;            // the producer entered the FC2 phase

    int const tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, t4 = lane % 4;
    int const M = p.M;
    int const bid = blockIdx.x, nblk = gridDim.x;
    int const pair = bid >> 1, rank = bid & 1; // cluster (2, 1, 1): rank = blockIdx.x % 2
    bool const has_fc2 = pair < kFc2Pairs;

    uint8_t* ring = smem + kOffRing;
    float4* red = reinterpret_cast<float4*>(smem + kOffRed);
    float* red2 = reinterpret_cast<float*>(smem + kOffRed2);
    float* peer = reinterpret_cast<float*>(smem + kOffPeer);
    uint8_t* actring = smem + kOffAct;
    cluster_arrive_relaxed(); // matched by the wait before the DSMEM exchange at the end

    auto slot_wait_empty = [&](int seq)
    {
        if (seq >= kSlots)
            mbar_wait(&empty[seq % kSlots], ((seq / kSlots) + 1) & 1);
    };
    // FC1 unit (u, jt) into slots seq, seq + 1 (producer thread)
    auto issue_fc1 = [&](int seq, int unit, int e)
    {
        const int jt = unit % kUnitsPerExpert;
        const uint8_t* W = e < 0 ? p.sw31 : p.w31 + (size_t) e * (2 * kI) * kH;
        const float* S = e < 0 ? p.ss31 : p.s31 + (size_t) e * (2 * kI / 128) * (kH / 128);
#pragma unroll 1
        for (int part = 0; part < 2; part++)
        { // up rows, gate rows
            const int sq = seq + part, slot = sq % kSlots, row0 = part * kI + jt * kJT;
            slot_wait_empty(sq);
            slot_tag[slot] = unit;
            uint8_t* dst = ring + slot * kSlot;
            mbar_expect_tx(&full[slot], kSlotData + 128);
            bulk_g2s(dst, W + (size_t) row0 * kH, kSlotData, &full[slot]);
            bulk_g2s(dst + kSlotData, S + (row0 / 128) * (kH / 128), 128, &full[slot]);
        }
    };

    // compute warps: x fragments for the warp's K slice [512 w, 512 w + 512) (raw BF16 first)
    uint2 raw[8][4];
    auto load_x = [&](int w)
    {
        const bool tok = g < M;
#pragma unroll
        for (int kb = 0; kb < 8; kb++)
#pragma unroll
            for (int j = 0; j < 4; j++)
                raw[kb][j] = tok
                    ? *reinterpret_cast<const uint2*>(p.x + (size_t) g * kH + w * 512 + kb * 64 + 16 * t4 + 4 * j)
                    : make_uint2(0u, 0u);
    };

    int orow = 0, otok = -1; // compute threads: this thread's FC2 output (row, token) and partial sum
    float ov = 0.f;

    // ---------------- prologue: barriers, routing (logits loads go out before the weight prefetch)
    if (warp == kWarpP)
    {
        if (lane == 0)
        {
            for (int i = 0; i < kSlots; i++)
            {
                mbar_init(&full[i], 1);
                mbar_init(&empty[i], kNumC);
            }
            for (int i = 0; i < kNumEpi; i++)
            {
                mbar_init(&redFull[i], kNumC);
                mbar_init(&redEmpty[i], 1);
            }
            for (int i = 0; i < kActStages; i++)
            {
                mbar_init(&actFull[i], 1);
                mbar_init(&actEmpty[i], kNumC);
            }
            mbar_init_fence();
            s_ready = 0;
            s_fc2go = 0;
            if (bid < kUnitsPerExpert)
                issue_fc1(0, bid, -1); // static first unit = shared expert
        }
        __syncwarp();
    }
    else if (warp >= kWarpC && warp < kWarpC + kMaxM)
    {
        // routing of token row t
        int const t = warp - kWarpC;
        constexpr int kPer = (kE + 31) / 32;
        cudaGridDependencySynchronize();
        float lg[kPer], bs[kPer];
        if (t < M)
        {
#pragma unroll
            for (int i = 0; i < kPer; i++)
            {
                int const e = lane + 32 * i;
                lg[i] = e < kE ? p.logits[t * kE + e] : 0.f;
                bs[i] = e < kE ? p.bias[e] : 0.f;
            }
        }
        load_x(t);
        if (t == 0)
        {
            for (int i = lane; i < kUPad * kMaxM; i += 32)
                s_wtab[i] = 0.f;
        }
        if (t < M)
        {
            // Fast path: 32-bit keys = (ordered key with the low 9 bits replaced by 511 - expert); one warp max per
            // round names the winner directly. Exact unless the 8th and 9th keys agree in their high 23 bits, in
            // which case the exact path below redoes the row with full keys.
            uint32_t kp[kPer];
            uint64_t v[kPer];
#pragma unroll
            for (int i = 0; i < kPer; i++)
            {
                int const e = lane + 32 * i;
                float const sc = 1.f / (1.f + expf(-lg[i]));
                if (e < kE)
                    s_sc[t * kE + e] = sc;
                const uint32_t ok = f2ord(sc + bs[i]);
                v[i] = e < kE ? ((uint64_t) ok << 32) | (uint32_t) ~e : 0ull;
                kp[i] = e < kE ? (ok & ~0x1ffu) | (uint32_t) (0x1ff - e) : 0u;
            }
            auto ce32 = [&](int a, int b)
            {
                const uint32_t x = kp[a], y = kp[b];
                kp[a] = max(x, y);
                kp[b] = min(x, y);
            };
            ce32(0, 1);
            ce32(3, 4);
            ce32(6, 7);
            ce32(1, 2);
            ce32(4, 5);
            ce32(7, 8);
            ce32(0, 1);
            ce32(3, 4);
            ce32(6, 7);
            ce32(2, 5);
            ce32(0, 3);
            ce32(1, 4);
            ce32(5, 8);
            ce32(3, 6);
            ce32(4, 7);
            ce32(2, 5);
            ce32(0, 3);
            ce32(1, 4);
            ce32(5, 7);
            ce32(2, 6);
            ce32(1, 3);
            ce32(4, 6);
            ce32(2, 4);
            ce32(5, 6);
            ce32(2, 3);
            uint32_t g8 = 0, g9 = 0;
#pragma unroll 1
            for (int k = 0; k <= kTopK; k++)
            {
                const uint32_t gm = __reduce_max_sync(0xffffffffu, kp[0]);
                if (k < kTopK)
                {
                    if (lane == 0)
                        s_pe[t * kTopK + k] = 0x1ff - (int) (gm & 0x1ffu);
                    if (kp[0] == gm)
                    {
#pragma unroll
                        for (int i = 0; i < kPer - 1; i++)
                            kp[i] = kp[i + 1];
                        kp[kPer - 1] = 0u;
                    }
                    g8 = gm;
                }
                else
                {
                    g9 = gm;
                }
            }
            if ((g8 >> 9) == (g9 >> 9))
            {
                // exact path: v = (ordered key << 32) | ~expert: one u64 compare orders by key, ties to the lower
                // expert sort the lane's 9 candidates descending (25 compare-exchanges, depth 9)
                auto ce = [&](int a, int b)
                {
                    const uint64_t x = v[a], y = v[b];
                    v[a] = x > y ? x : y;
                    v[b] = x > y ? y : x;
                };
                ce(0, 1);
                ce(3, 4);
                ce(6, 7);
                ce(1, 2);
                ce(4, 5);
                ce(7, 8);
                ce(0, 1);
                ce(3, 4);
                ce(6, 7);
                ce(2, 5);
                ce(0, 3);
                ce(1, 4);
                ce(5, 8);
                ce(3, 6);
                ce(4, 7);
                ce(2, 5);
                ce(0, 3);
                ce(1, 4);
                ce(5, 7);
                ce(2, 6);
                ce(1, 3);
                ce(4, 6);
                ce(2, 4);
                ce(5, 6);
                ce(2, 3);
                // 8 rounds: warp max of the heads, the winning lane pops its head
#pragma unroll 1
                for (int k = 0; k < kTopK; k++)
                {
                    const uint32_t hk = (uint32_t) (v[0] >> 32);
                    const uint32_t gm = __reduce_max_sync(0xffffffffu, hk);
                    unsigned const hit = __ballot_sync(0xffffffffu, hk == gm);
                    int wl;
                    if (__popc(hit) == 1)
                    {
                        wl = __ffs(hit) - 1;
                    }
                    else
                    { // exact key tie across lanes: the lower expert index wins
                        wl = (int) (~__reduce_max_sync(0xffffffffu, hk == gm ? (uint32_t) v[0] : 0u) & 31u);
                    }
                    if (lane == wl)
                    {
                        s_pe[t * kTopK + k] = (int) ~(uint32_t) v[0];
#pragma unroll
                        for (int i = 0; i < kPer - 1; i++)
                            v[i] = v[i + 1];
                        v[kPer - 1] = 0ull;
                    }
                }
            }
            __syncwarp();
            if (lane < kTopK)
                s_ps[t * kTopK + lane] = s_sc[t * kE + s_pe[t * kTopK + lane]];
            __syncwarp();
            if (lane < kTopK)
            {
                float const s = s_ps[t * kTopK + lane];
                float tot = s;
                tot += __shfl_xor_sync(0xffu, tot, 4);
                tot += __shfl_xor_sync(0xffu, tot, 2);
                tot += __shfl_xor_sync(0xffu, tot, 1);
                s_ps[t * kTopK + lane] = s / tot * p.scaling;
            }
        }
        __syncwarp();
        named_bar(2, kMaxM * 32);
        if (t == 0)
        {
            // distinct experts in first-occurrence order (u = 1..), the shared expert is u = 0
            bool const valid = lane < M * kTopK;
            int const e = valid ? s_pe[lane] : -1 - lane;
            float const w = valid ? s_ps[lane] : 0.f;
            unsigned const mask = __match_any_sync(0xffffffffu, e);
            int const leader = __ffs(mask) - 1;
            bool const first = valid && leader == lane;
            unsigned const fb = __ballot_sync(0xffffffffu, first);
            int const myidx = __popc(fb & ((1u << lane) - 1u));
            int const du = __shfl_sync(0xffffffffu, myidx, leader);
            if (valid)
            {
                s_wtab[(1 + du) * kMaxM + lane / kTopK] = w;
                if (first)
                    s_route[2 + du] = e;
            }
            if (lane < kMaxM)
                s_wtab[lane] = lane < M ? 1.f : 0.f;
            if (lane == 0)
            {
                s_route[0] = 1 + __popc(fb);
                s_route[1] = -1;
            }
        }
        __syncwarp();
    }

    // compute warps: this lane's FC1 B fragments, kept for every unit
    uint32_t xf[8][4][2];
    if (warp >= kWarpC)
    {
        int const w = warp - kWarpC;
        if (w >= kMaxM)
        {
            cudaGridDependencySynchronize();
            load_x(w);
        }
#pragma unroll
        for (int kb = 0; kb < 8; kb++)
#pragma unroll
            for (int j = 0; j < 4; j++)
            {
                const __half2 h0
                    = __float22half2_rn(__bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&raw[kb][j].x)));
                const __half2 h1
                    = __float22half2_rn(__bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&raw[kb][j].y)));
                xf[kb][j][0] = *reinterpret_cast<uint32_t const*>(&h0);
                xf[kb][j][1] = *reinterpret_cast<uint32_t const*>(&h1);
            }
    }
    __syncwarp();
    __syncthreads(); // routing published
    int const U = s_route[0];
    int const total1 = U * kUnitsPerExpert;

    if (warp == kWarpP)
    {
        // ---------------- producer
        if (lane == 0)
        {
            int seq = 0;
            if (bid >= kUnitsPerExpert)
                issue_fc1(0, bid, s_route[1 + bid / kUnitsPerExpert]);
            seq = 2;
            int nxt = nblk + atomicAdd(&g_ctr, 1);
            while (nxt < total1)
            {
                int const cur = nxt;
                nxt = nblk + atomicAdd(&g_ctr, 1);
                // the poller starts once the last two experts are being claimed
                if (cur >= total1 - 2 * kUnitsPerExpert)
                    *reinterpret_cast<int volatile*>(&s_fc2go) = 1;
                issue_fc1(seq, cur, s_route[1 + cur / kUnitsPerExpert]);
                seq += 2;
            }
            if (has_fc2)
            {
                // FC2 stage i = expert u = rank + 2 i: W2 tile [64 x 512] + scales into a ring slot as soon as one is
                // free; the activation into the act ring once done[u] is complete (the poller warp publishes it)
                int const base = seq;
                int const n2 = (U - rank + 1) / 2;
                const uint32_t abytes = 2u * M * kI * 2;
                int wq = 0, aq = 0, ready = 0;
                auto poll = [&]()
                {
                    const int r = *reinterpret_cast<volatile int*>(&s_ready);
                    // (no reader-side fence: the epilogue fences its activation stores before bumping done[u], so the
                    // data is in L2, where the bulk copies read it, before the counter completes; a gpu-scope fence
                    // here costs microseconds under the streaming load)
                    if (r > ready)
                    {
                        ready = r;
                        asm volatile("" ::: "memory");
                    }
                };
                *reinterpret_cast<int volatile*>(&s_fc2go) = 1;
                poll();
                while (aq < n2)
                {
                    int const wq0 = wq, aq0 = aq;
                    while (wq < n2
                        && (base + wq < kSlots
                            || mbar_test(&empty[(base + wq) % kSlots], (((base + wq) / kSlots) + 1) & 1)))
                    {
                        int const slot = (base + wq) % kSlots, u = rank + 2 * wq, e = s_route[1 + u];
                        uint8_t* dst = ring + slot * kSlot;
                        uint8_t const* W
                            = (e < 0 ? p.sw2 : p.w2 + (size_t) e * kH * kI) + (size_t) pair * kFc2Rows * kI;
                        float const* S = (e < 0 ? p.ss2 : p.s2 + (size_t) e * (kH / 128) * (kI / 128))
                            + (pair * kFc2Rows / 128) * (kI / 128);
                        slot_tag[slot] = -2 - wq;
                        mbar_expect_tx(&full[slot], kW2Tile + 16);
                        bulk_g2s(dst, W, kW2Tile, &full[slot]);
                        bulk_g2s(dst + kSlotData, S, 16, &full[slot]);
                        wq++;
                    }
                    poll();
                    while (aq < n2 && rank + 2 * aq < ready
                        && (aq < kActStages || mbar_test(&actEmpty[aq % kActStages], ((aq / kActStages) + 1) & 1)))
                    {
                        int const as = aq % kActStages;
                        mbar_expect_tx(&actFull[as], abytes);
                        bulk_g2s(actring + as * kActBytes, g_act + (size_t) (rank + 2 * aq) * 2 * M * kI, abytes,
                            &actFull[as]);
                        aq++;
                    }
                    // nothing to issue: back off so this spin does not take issue slots from the compute warps
                    if (wq == wq0 && aq == aq0)
                        __nanosleep(128);
                }
                while (wq < n2)
                { // (only if U is tiny) remaining W2 stages
                    slot_wait_empty(base + wq);
                    int const slot = (base + wq) % kSlots, u = rank + 2 * wq, e = s_route[1 + u];
                    uint8_t* dst = ring + slot * kSlot;
                    uint8_t const* W = (e < 0 ? p.sw2 : p.w2 + (size_t) e * kH * kI) + (size_t) pair * kFc2Rows * kI;
                    float const* S = (e < 0 ? p.ss2 : p.s2 + (size_t) e * (kH / 128) * (kI / 128))
                        + (pair * kFc2Rows / 128) * (kI / 128);
                    slot_tag[slot] = -2 - wq;
                    mbar_expect_tx(&full[slot], kW2Tile + 16);
                    bulk_g2s(dst, W, kW2Tile, &full[slot]);
                    bulk_g2s(dst + kSlotData, S, 16, &full[slot]);
                    wq++;
                }
                seq = base + n2;
            }
            slot_wait_empty(seq);
            slot_tag[seq % kSlots] = -1;
            mbar_arrive(&full[seq % kSlots]);
        }
    }
    else if (warp == kWarpPoll)
    {
        // ---------------- poller: publish how many experts (from u = 0) have all 64 FC1 units done. Runs from the
        // start (L2 round trips take microseconds under load), one 16-byte group of counters per poll.
        if (has_fc2 && lane == 0)
        {
            int r = 0;
            while (r < U)
            {
                const int4 d = ld_relaxed_v4(g_done + (r & ~3));
                int const dv[4] = {d.x, d.y, d.z, d.w};
                int r2 = r;
#pragma unroll
                for (int i = 0; i < 4; i++)
                    if (i >= (r & 3) && r2 == (r & ~3) + i && dv[i] >= kUnitsPerExpert)
                        r2++;
                if (r2 > r)
                {
                    r = r2;
                    *reinterpret_cast<int volatile*>(&s_ready) = r;
                }
                else
                {
                    __nanosleep(400);
                }
            }
        }
        __syncwarp();
    }
    else if (warp < kWarpC)
    {
        // ---------------- FC1 epilogue (unit k of this CTA -> warp k % 4): sum the 8 K-slice partials, SwiGLU,
        // store the activation, bump done[u]. Four warps so that their memory fences overlap.
        int const ew = warp - kWarpEpi;
        int const jr = lane / 4, t = lane % 4; // row within the tile, token
        for (int n = 0;; n++)
        {
            mbar_wait(&redFull[ew], n & 1);
            int const unit = red_unit[ew];
            if (unit < 0)
                break;
            float up = 0.f, gate = 0.f;
            float4 const* rr = red + ew * (kNumC * 16);
#pragma unroll
            for (int w = 0; w < kNumC; w++)
            {
                const float4 v = rr[w * 16 + jr * 2 + (t >> 1)];
                up += (t & 1) ? v.y : v.x;
                gate += (t & 1) ? v.w : v.z;
            }
            __syncwarp();
            if (lane == 0)
                mbar_arrive(&redEmpty[ew]);
            int const u = unit / kUnitsPerExpert, nn = (unit % kUnitsPerExpert) * kJT + jr;
            if (t < M)
            {
                float const gv = fminf(gate, p.limit);
                float const uu = fminf(fmaxf(up, -p.limit), p.limit);
                float const act = gv / (1.f + __expf(-gv)) * uu * s_wtab[u * kMaxM + t];
                const __half hi = __float2half_rn(act);
                __half* dst = g_act + (size_t) u * 2 * M * kI + (size_t) t * kI + nn;
                dst[0] = hi;
                dst[M * kI] = __float2half_rn(act - __half2float(hi));
            }
            __threadfence();
            __syncwarp();
            if (lane == 0)
                atomicAdd(&g_done[u], 1);
        }
    }
    else
    {
        // ---------------- compute warps
        int const w = warp - kWarpC;
        bool const tok = g < M;
        int const mt = w & 3, kh = w >> 2; // FC2: m-tile and K half
        float acc2[4] = {0.f, 0.f, 0.f, 0.f};
        int seq = 0, k1 = 0;
        auto close_epi = [&]() { // tell the epilogue warps that FC1 is over in this CTA
            for (int kk = k1; kk < k1 + kNumEpi; kk++)
            {
                const int rb = kk % kNumEpi;
                if (kk >= kNumEpi)
                    mbar_wait(&redEmpty[rb], ((kk / kNumEpi) + 1) & 1);
                if (w == 0 && lane == 0)
                    red_unit[rb] = -1;
                __syncwarp();
                if (lane == 0)
                    mbar_arrive(&redFull[rb]);
            }
        };
        // FC1 units first (x fragments live), then FC2 stages (x dead): two loops keep register pressure down
        int tag = 0;
        while (true)
        {
            int const slot = seq % kSlots;
            mbar_wait(&full[slot], (seq / kSlots) & 1);
            tag = slot_tag[slot];
            if (tag < 0)
                break;
            uint8_t const* st = ring + slot * kSlot;
            // FC1 unit: up rows in this slot, gate rows in the next
            int const slotg = (seq + 1) % kSlots;
            mbar_wait(&full[slotg], ((seq + 1) / kSlots) & 1);
            uint8_t const* sg_ = ring + slotg * kSlot;
            float const* scu = reinterpret_cast<float const*>(st + kSlotData);
            float const* scg = reinterpret_cast<float const*>(sg_ + kSlotData);
            uint8_t const* rowu = st + g * kH + w * 512 + 16 * t4;
            uint8_t const* rowg = sg_ + g * kH + w * 512 + 16 * t4;
            float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int kc = 0; kc < 4; kc++)
            {
                float blk[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int kb2 = 0; kb2 < 2; kb2++)
                {
                    int const kb = 2 * kc + kb2;
                    const uint4 wu = *reinterpret_cast<uint4 const*>(rowu + kb * 64);
                    const uint4 wg = *reinterpret_cast<uint4 const*>(rowg + kb * 64);
#pragma unroll
                    for (int j = 0; j < 4; j++)
                        chunk_mma(blk, word(wu, j), word(wg, j), xf[kb][j][0], xf[kb][j][1]);
                }
                float const su = scu[4 * w + kc], sg = scg[4 * w + kc];
                acc[0] = fmaf(blk[0], su, acc[0]);
                acc[1] = fmaf(blk[1], su, acc[1]);
                acc[2] = fmaf(blk[2], sg, acc[2]);
                acc[3] = fmaf(blk[3], sg, acc[3]);
            }
            __syncwarp();
            if (lane == 0)
            {
                mbar_arrive(&empty[slot]);
                mbar_arrive(&empty[slotg]);
            }
            int const rb = k1 % kNumEpi;
            if (k1 >= kNumEpi)
                mbar_wait(&redEmpty[rb], ((k1 / kNumEpi) + 1) & 1);
            if (t4 < 2)
                red[rb * (kNumC * 16) + w * 16 + g * 2 + t4] = make_float4(acc[0], acc[1], acc[2], acc[3]);
            if (w == 0 && lane == 0)
                red_unit[rb] = tag;
            __syncwarp();
            if (lane == 0)
                mbar_arrive(&redFull[rb]);
            k1++;
            seq += 2;
        }
        close_epi();
        while (tag != -1)
        {
            int const slot = seq % kSlots;
            mbar_wait(&full[slot], (seq / kSlots) & 1);
            tag = slot_tag[slot];
            if (tag == -1)
                break;
            uint8_t const* st = ring + slot * kSlot;
            // FC2 stage i (expert rank + 2 i): rows mt * 16 + (g, g + 8) of the 64, K half kh
            int const i2 = -2 - tag, as = i2 % kActStages;
            mbar_wait(&actFull[as], (i2 / kActStages) & 1);
            float const* sc = reinterpret_cast<float const*>(st + kSlotData) + 2 * kh;
            // B columns 0..3 = hi of tokens 0..3, columns 4..7 = lo of the same tokens: one MMA per chunk does both
            // (columns of tokens >= M read token 0 and are never stored)
            int const ga = (g & 3) < M ? (g & 3) : 0;
            __half const* ab = reinterpret_cast<__half const*>(actring + as * kActBytes) + (g < 4 ? 0 : M * kI)
                + ga * kI + kh * 256 + 16 * t4;
            uint8_t const* wr = st + (mt * 16 + g) * kI + kh * 256 + 16 * t4;
#pragma unroll
            for (int kc = 0; kc < 2; kc++)
            { // the half's two 128-wide K blocks
                float bk[2][4] = {};
#pragma unroll
                for (int kb2 = 0; kb2 < 2; kb2++)
                {
                    int const kb = 2 * kc + kb2;
                    const uint4 wg = *reinterpret_cast<uint4 const*>(wr + kb * 64);
                    const uint4 wg8 = *reinterpret_cast<uint4 const*>(wr + 8 * kI + kb * 64);
                    const uint4 a0 = *reinterpret_cast<uint4 const*>(ab + kb * 64);
                    const uint4 a1 = *reinterpret_cast<uint4 const*>(ab + kb * 64 + 8);
                    chunk_mma(bk[kb2], word(wg, 0), word(wg8, 0), a0.x, a0.y);
                    chunk_mma(bk[kb2], word(wg, 1), word(wg8, 1), a0.z, a0.w);
                    chunk_mma(bk[kb2], word(wg, 2), word(wg8, 2), a1.x, a1.y);
                    chunk_mma(bk[kb2], word(wg, 3), word(wg8, 3), a1.z, a1.w);
                }
#pragma unroll
                for (int c = 0; c < 4; c++)
                    acc2[c] = fmaf(bk[0][c] + bk[1][c], sc[kc], acc2[c]);
            }
            __syncwarp();
            if (lane == 0)
            {
                mbar_arrive(&empty[slot]);
                mbar_arrive(&actEmpty[as]);
            }
            seq += 1;
        }
        if (has_fc2)
        {
#pragma unroll
            for (int c = 0; c < 4; c++)
                red2[(kh * kFc2Rows + mt * 16 + g + 8 * (c >> 1)) * 8 + 2 * t4 + (c & 1)] = acc2[c];
            named_bar(3, kNumC * 32);
            int const i = tid - kWarpC * 32; // 0..255: (token, row)
            orow = i % kFc2Rows;
            otok = i / kFc2Rows;
            ov = (red2[orow * 8 + otok] + red2[orow * 8 + 4 + otok])
                + (red2[(kFc2Rows + orow) * 8 + otok] + red2[(kFc2Rows + orow) * 8 + 4 + otok]);
        }
    }

    // ---------------- the pair's two partial sums: rank 1 -> rank 0 over DSMEM, rank 0 adds and stores BF16
    __syncthreads();
    cluster_wait(); // (the peer CTA is running)
    if (has_fc2 && rank == 1 && otok >= 0 && otok < M)
        st_cluster_f32(&peer[orow * kMaxM + otok], 0, ov);
    cluster_arrive();
    cluster_wait();
    if (has_fc2 && rank == 0 && otok >= 0 && otok < M)
        p.out[(size_t) otok * kH + pair * kFc2Rows + orow] = __float2bfloat16(ov + peer[orow * kMaxM + otok]);

    // ---------------- teardown: the last CTA resets the global counters for the next call
    __syncthreads();
    if (tid == 0)
    {
        __threadfence();
        if (atomicAdd(&g_exit, 1) == nblk - 1)
        {
            g_ctr = 0;
            for (int i = 0; i < kUPad; i++)
                g_done[i] = 0;
            g_exit = 0;
            __threadfence();
        }
    }
}

} // namespace

cudaError_t decodeMoe(DecodeMoeArgs const& a, cudaStream_t stream)
{
    static int grid = 0;
    if (grid == 0)
    {
        int dev, num_sms = 0;
        cudaError_t err = cudaGetDevice(&dev);
        if (err == cudaSuccess)
            err = cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);
        if (err == cudaSuccess)
            err = cudaFuncSetAttribute(moe_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
        if (err == cudaSuccess)
            err = cudaFuncSetAttribute(moe_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 0);
        int clusters = 0;
        if (err == cudaSuccess)
        {
            cudaLaunchConfig_t cfg{};
            cfg.gridDim = dim3(num_sms & ~1);
            cfg.blockDim = dim3(kThreads);
            cfg.dynamicSmemBytes = kSmem;
            cudaLaunchAttribute attr[1];
            attr[0].id = cudaLaunchAttributeClusterDimension;
            attr[0].val.clusterDim.x = 2;
            attr[0].val.clusterDim.y = 1;
            attr[0].val.clusterDim.z = 1;
            cfg.attrs = attr;
            cfg.numAttrs = 1;
            err = cudaOccupancyMaxActiveClusters(&clusters, moe_kernel, &cfg);
        }
        if (err != cudaSuccess)
            return err;
        // every CTA must be co-resident (spin waits on other CTAs' progress)
        clusters = clusters < num_sms / 2 ? clusters : num_sms / 2;
        if (clusters < kFc2Pairs)
            return cudaErrorNotSupported;
        grid = 2 * clusters;
    }
    if (a.rows < 1 || a.rows > kMaxM)
        return cudaErrorInvalidValue;
    MoeParams p{reinterpret_cast<__nv_bfloat16 const*>(a.x), a.logits, a.bias, a.w31, a.s31, a.sw31, a.ss31, a.w2, a.s2,
        a.sw2, a.ss2, reinterpret_cast<__nv_bfloat16*>(a.out), a.scaling, a.limit, a.rows};
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = kSmem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[2];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = a.pdl ? 1 : 0;
    attr[1].id = cudaLaunchAttributeClusterDimension;
    attr[1].val.clusterDim.x = 2;
    attr[1].val.clusterDim.y = 1;
    attr[1].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 2;
    return cudaLaunchKernelEx(&cfg, moe_kernel, p);
}

} // namespace glm53_fp8
