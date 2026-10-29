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
// Decode MoE (M <= 4 rows) on tensor cores, each distinct expert of the step streamed once from HBM in
// long contiguous TMA bulk copies.
//
// Shapes (GLM-5.3-Flash at moe_tp4): H = 4096, I = 512, E = 288, top-8 + the shared expert.
// w31 [E, 2I, H] e4m3 = [up; gate] rows, s31 [E, 2I/128, H/128]; w2 [E, H, I] e4m3, s2 [E, H/128, I/128].
//
// Arithmetic: e4m3 -> f16 (exact, cvt.rn.f16x2.e4m3x2), BF16 x -> f16 (exact for |x| < 65504),
// mma.sync m16n8k16 f16 x f16 -> f32, every 128-wide K block's partial scaled by its block scale in f32.
// The FC1 -> FC2 activation (times the routing weight) is stored as an f16 hi + lo pair.
//
// K permutation: in a 64-wide K block, lane (g = lane / 4, t4 = lane % 4) reads bytes
// [16 t4, 16 t4 + 16) of its weight rows; MMA chunk j (16 K) takes bytes 16 t4 + 4 j + {0, 1} as the
// A-fragment columns {2 t4, 2 t4 + 1} and + {2, 3} as {2 t4 + 8, 2 t4 + 9}. The B fragment (column
// g = token) reads the activation at the same K positions, so the dot products are unchanged.
//
// fc1 (persistent, one CTA per SM): every CTA computes the step's routing (CTA 0 publishes it), then
//   takes units (distinct expert u, 8-row tile of I) round-robin. A unit is the tile's 8 up rows and
//   8 gate rows over all of H: two contiguous 32 KB bulk copies, three units in the ring. The 16 rows
//   form one m16 MMA tile (rows 0-7 up, 8-15 gate), so a lane holds matching up / gate sums; warp w
//   takes K [512 w, 512 w + 512) and keeps that slice of x in registers; the warps' partials are
//   summed in shared memory and warp 0 applies the clamped SwiGLU and the routing weight.
// fc2 (one CTA per 32 output rows): stage = one distinct expert's [32 rows x 512] weight tile (16 KB
//   contiguous), its activations and scales, eight stages in the ring; warp w takes m-tile w % 2 and
//   K quarter w / 2; the quarters are summed in shared memory and stored as BF16.
#include "decode_moe.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <climits>
#include <cstdint>

namespace glm53_fp8
{
namespace
{

constexpr int kH = 4096, kI = 512, kE = 288, kTopK = 8, kSlots = kTopK + 1, kMaxM = kDecodeMoeMaxRows;
constexpr int kMaxPairs = kMaxM * kSlots, kMaxU = kMaxPairs;
constexpr int kWarps = 8, kThreads = kWarps * 32;

// fc1 unit: 8 up rows + 8 gate rows over H, then 32 up + 32 gate scales (the tile's 128-row block)
constexpr int kTile = 8, kTiles = kI / kTile;
constexpr int kUnitW = 2 * kTile * kH, kUnitBytes = kUnitW + 256;
constexpr int kRing = 3;
constexpr int kFc1Smem = kRing * kUnitBytes + kWarps * 32 * 4 * 4;
// fc2 stage: one expert's [32 rows x 512] weights, act hi [M x 512] f16, act lo [M x 512] f16, 4 scales
constexpr int kFc2Ring = 8, kFc2Rows = 32;
constexpr int kRowBytes = 512;
constexpr int kFc2W = kFc2Rows * kRowBytes, kFc2A = kMaxM * kI * 2 * 2, kFc2Stage = kFc2W + kFc2A + 16;
constexpr int kFc2Smem = kFc2Ring * kFc2Stage + kWarps * 16 * 8 * 4;

struct MoeParams
{
    __nv_bfloat16 const* x; // [M, H]
    float const* logits;    // [M, E]
    float const* bias;      // [E]
    uint8_t const* w31;
    float const* s31;
    uint8_t const* sw31;
    float const* ss31;
    uint8_t const* w2;
    float const* s2;
    uint8_t const* sw2;
    float const* ss2;
    __half* actw;       // [kMaxU, 2 (hi, lo), kMaxM, I]
    int* route;         // [1 + kMaxU]: U, then the distinct experts (-1 = shared)
    float* wtab;        // [kMaxU, kMaxM]
    __nv_bfloat16* out; // [M, H]
    float scaling, limit;
    int M;
};

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

// TMA bulk copy global -> shared, completing `bytes` on bar
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

// The step's routing, computed by one CTA into shared memory: warp r < M takes row r (sigmoid scores,
// top-8 of score + bias with ties to the lower expert, weights = score / sum * scaling, the shared
// expert as slot 8 with weight 1); then the distinct experts in first-occurrence order
// (s_route = [U, experts...]) and s_wtab[u * kMaxM + t] = token t's weight on distinct expert u.
// publish: also store both to global memory (for fc2).
__device__ void route_cta(MoeParams const& p, int* s_route, float* s_wtab, bool publish)
{
    __shared__ int s_ids[kMaxPairs];
    __shared__ float s_wts[kMaxPairs];
    __shared__ int s_firstpair[kE + 1]; // per expert (shared = index 0): its first pair, INT_MAX if none
    __shared__ unsigned s_ballot[2];
    int const tid = threadIdx.x, warp = tid / 32, lane = tid % 32, M = p.M, npairs = M * kSlots;
    if (warp < M)
    {
        constexpr int kPer = (kE + 31) / 32;
        float sc[kPer], ch[kPer];
#pragma unroll
        for (int i = 0; i < kPer; i++)
        {
            int const e = lane + 32 * i;
            if (e < kE)
            {
                sc[i] = 1.f / (1.f + __expf(-p.logits[warp * kE + e]));
                ch[i] = sc[i] + p.bias[e];
            }
            else
            {
                sc[i] = 0.f;
                ch[i] = -FLT_MAX;
            }
        }
        float wsel[kTopK], total = 0.f;
        int isel[kTopK];
#pragma unroll
        for (int k = 0; k < kTopK; k++)
        {
            float bv = -FLT_MAX;
            int bi = INT_MAX;
#pragma unroll
            for (int i = 0; i < kPer; i++)
            {
                int const e = lane + 32 * i;
                if (ch[i] > bv || (ch[i] == bv && e < bi))
                {
                    bv = ch[i];
                    bi = e;
                }
            }
#pragma unroll
            for (int off = 16; off >= 1; off >>= 1)
            {
                float const ov = __shfl_xor_sync(0xffffffffu, bv, off);
                int const oi = __shfl_xor_sync(0xffffffffu, bi, off);
                if (ov > bv || (ov == bv && oi < bi))
                {
                    bv = ov;
                    bi = oi;
                }
            }
            float mine = 0.f;
#pragma unroll
            for (int i = 0; i < kPer; i++)
                if (lane + 32 * i == bi)
                {
                    mine = sc[i];
                    ch[i] = -FLT_MAX;
                }
            wsel[k] = __shfl_sync(0xffffffffu, mine, bi & 31);
            isel[k] = bi;
            total += wsel[k];
        }
        if (lane == 0)
        {
#pragma unroll
            for (int k = 0; k < kTopK; k++)
            {
                s_ids[warp * kSlots + k] = isel[k];
                s_wts[warp * kSlots + k] = wsel[k] / total * p.scaling;
            }
            s_ids[warp * kSlots + kTopK] = -1;
            s_wts[warp * kSlots + kTopK] = 1.f;
        }
    }
    for (int i = tid; i < kMaxU * kMaxM; i += blockDim.x)
        s_wtab[i] = 0.f;
    for (int i = tid; i < kE + 1; i += blockDim.x)
        s_firstpair[i] = INT_MAX;
    __syncthreads();
    if (tid < npairs)
        atomicMin(&s_firstpair[s_ids[tid] + 1], tid);
    __syncthreads();
    // pairs 0..35 live on warps 0 and 1: distinct index = first occurrences before a pair's first one
    int const f = tid < npairs ? s_firstpair[s_ids[tid] + 1] : INT_MAX;
    bool const is_first = tid < npairs && f == tid;
    unsigned const b = __ballot_sync(0xffffffffu, is_first);
    if (lane == 0 && warp < 2)
        s_ballot[warp] = b;
    __syncthreads();
    if (tid < npairs)
    {
        int const fw = f / 32, fl = f % 32;
        int const du = (fw == 1 ? __popc(s_ballot[0]) : 0) + __popc(s_ballot[fw] & ((1u << fl) - 1u));
        s_wtab[du * kMaxM + tid / kSlots] = s_wts[tid];
        if (is_first)
            s_route[1 + du] = s_ids[tid];
        if (tid == 0)
            s_route[0] = __popc(s_ballot[0]) + (npairs > 32 ? __popc(s_ballot[1]) : 0);
    }
    __syncthreads();
    if (publish)
    {
        if (tid < 1 + kMaxU)
            p.route[tid] = s_route[tid];
        if (tid < kMaxU * kMaxM)
            p.wtab[tid] = s_wtab[tid];
    }
}

__global__ void __launch_bounds__(kThreads, 1) fc1_kernel(MoeParams p)
{
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* ring = smem;
    float* red = reinterpret_cast<float*>(smem + kRing * kUnitBytes); // [kWarps][32 lanes][4]
    __shared__ int s_route[1 + kMaxU];
    __shared__ float s_wtab[kMaxU * kMaxM];
    __shared__ __align__(8) uint64_t full[kRing];
    int const tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, t4 = lane % 4;
    int const M = p.M;
    cudaGridDependencySynchronize();
    route_cta(p, s_route, s_wtab, blockIdx.x == 0);
    int const U = s_route[0];
    int const units = U * kTiles;
    int const nunits = units > (int) blockIdx.x ? (units - 1 - (int) blockIdx.x) / (int) gridDim.x + 1 : 0;
    if (nunits == 0)
    {
        cudaTriggerProgrammaticLaunchCompletion();
        return;
    }
    // thread 0 only: CTA-local unit k (global unit blockIdx.x + k * gridDim.x) into ring slot k % kRing
    auto issue = [&](int k)
    {
        const int unit = blockIdx.x + k * gridDim.x, u = unit / kTiles, n0 = (unit % kTiles) * kTile;
        const int e = s_route[1 + u];
        const uint8_t* W = e < 0 ? p.sw31 : p.w31 + (size_t) e * (2 * kI) * kH;
        const float* S = e < 0 ? p.ss31 : p.s31 + (size_t) e * (2 * kI / 128) * (kH / 128);
        uint8_t* dst = ring + (k % kRing) * kUnitBytes;
        uint64_t* bar = &full[k % kRing];
        mbar_expect_tx(bar, kUnitBytes);
        bulk_g2s(dst, W + (size_t) n0 * kH, kTile * kH, bar);                       // up rows
        bulk_g2s(dst + kTile * kH, W + (size_t) (kI + n0) * kH, kTile * kH, bar);   // gate rows
        bulk_g2s(dst + kUnitW, S + (n0 / 128) * (kH / 128), 128, bar);              // up scales
        bulk_g2s(dst + kUnitW + 128, S + ((kI + n0) / 128) * (kH / 128), 128, bar); // gate scales
    };
    if (tid == 0)
    {
        for (int i = 0; i < kRing; i++)
            mbar_init(&full[i], 1);
        mbar_init_fence();
        for (int k = 0; k < kRing && k < nunits; k++)
            issue(k);
    }
    // this lane's B fragments for the warp's K slice, kept for every unit: x[g][512 w + 64 kb + 16 t4 + 4 j ..]
    bool const tok = g < M;
    uint32_t xf[8][4][2];
    {
        uint2 raw[8][4];
#pragma unroll
        for (int kb = 0; kb < 8; kb++)
#pragma unroll
            for (int j = 0; j < 4; j++)
                raw[kb][j] = tok
                    ? *reinterpret_cast<uint2 const*>(p.x + (size_t) g * kH + warp * 512 + kb * 64 + 16 * t4 + 4 * j)
                    : make_uint2(0u, 0u);
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
    __syncthreads(); // the barriers are initialized
#pragma unroll 1
    for (int k = 0; k < nunits; k++)
    {
        uint8_t const* buf = ring + (k % kRing) * kUnitBytes;
        mbar_wait(&full[k % kRing], (k / kRing) & 1);
        float const* su = reinterpret_cast<float const*>(buf + kUnitW);
        float const* sg = su + 32;
        uint8_t const* up = buf + g * kH + warp * 512 + 16 * t4;
        uint8_t const* gt = buf + (kTile + g) * kH + warp * 512 + 16 * t4;
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int kc = 0; kc < 4; kc++)
        { // this warp's four 128-wide K blocks
            float blk[2][4] = {};
#pragma unroll
            for (int h = 0; h < 2; h++)
            {
                int const kb = 2 * kc + h;
                const uint4 wu = *reinterpret_cast<uint4 const*>(up + kb * 64);
                const uint4 wg = *reinterpret_cast<uint4 const*>(gt + kb * 64);
#pragma unroll
                for (int j = 0; j < 4; j++)
                    chunk_mma(blk[h], word(wu, j), word(wg, j), xf[kb][j][0], xf[kb][j][1]);
            }
            float const a = su[warp * 4 + kc], c = sg[warp * 4 + kc];
            acc[0] = fmaf(blk[0][0] + blk[1][0], a, acc[0]);
            acc[1] = fmaf(blk[0][1] + blk[1][1], a, acc[1]);
            acc[2] = fmaf(blk[0][2] + blk[1][2], c, acc[2]);
            acc[3] = fmaf(blk[0][3] + blk[1][3], c, acc[3]);
        }
        __syncthreads(); // every warp is done with this slot
        if (tid == 0 && k + kRing < nunits)
            issue(k + kRing);
#pragma unroll
        for (int c = 0; c < 4; c++)
            red[(warp * 32 + lane) * 4 + c] = acc[c];
        __syncthreads();
        if (warp == 0)
        {
            int const unit = blockIdx.x + k * gridDim.x, u = unit / kTiles, n0 = (unit % kTiles) * kTile;
            float s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int w = 0; w < kWarps; w++)
#pragma unroll
                for (int c = 0; c < 4; c++)
                    s[c] += red[(w * 32 + lane) * 4 + c];
#pragma unroll
            for (int c = 0; c < 2; c++)
            { // c0 / c1: up row g, tokens 2 t4 + c; c2 / c3: the gate row
                int const t = 2 * t4 + c;
                if (t < M)
                {
                    float const gv = fminf(s[2 + c], p.limit);
                    float const uu = fminf(fmaxf(s[c], -p.limit), p.limit);
                    float const act = gv / (1.f + __expf(-gv)) * uu * s_wtab[u * kMaxM + t];
                    const __half hi = __float2half_rn(act);
                    __half* dst = p.actw + (size_t) u * 2 * kMaxM * kI + (size_t) t * kI + n0 + g;
                    dst[0] = hi;
                    dst[kMaxM * kI] = __float2half_rn(act - __half2float(hi));
                }
            }
        }
    }
    cudaTriggerProgrammaticLaunchCompletion();
}

__global__ void __launch_bounds__(kThreads, 1) fc2_kernel(MoeParams p)
{
    extern __shared__ __align__(128) uint8_t smem[];
    float* red = reinterpret_cast<float*>(smem + kFc2Ring * kFc2Stage); // [kWarps][16 rows][8 tokens]
    __shared__ __align__(8) uint64_t full[kFc2Ring];
    __shared__ int s_route[1 + kMaxU];
    int const tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, t4 = lane % 4;
    int const M = p.M, n0 = blockIdx.x * kFc2Rows, nb = n0 / 128;
    int const mt = warp % 2, kq = warp / 2; // m-tile (16 rows) and K quarter (128 wide) of this warp
    cudaGridDependencySynchronize();
    if (tid < 1 + kMaxU)
        s_route[tid] = __ldcg(p.route + tid);
    __syncthreads();
    int const U = s_route[0];
    int const abytes = M * kI * 2; // one of hi / lo, f16
    // thread 0 only: stage si -> slot si % ring, distinct expert (si + blockIdx.x) % U
    auto issue = [&](int si)
    {
        const int slot = si % kFc2Ring, s = (si + blockIdx.x) % U, e = s_route[1 + s];
        uint8_t* dst = smem + slot * kFc2Stage;
        const uint8_t* W = (e < 0 ? p.sw2 : p.w2 + (size_t) e * kH * kI) + (size_t) n0 * kI;
        const __half* A = p.actw + (size_t) s * 2 * kMaxM * kI;
        const float* S = (e < 0 ? p.ss2 : p.s2 + (size_t) e * (kH / 128) * (kI / 128)) + nb * (kI / 128);
        mbar_expect_tx(&full[slot], kFc2W + 2 * abytes + 16);
        bulk_g2s(dst, W, kFc2W, &full[slot]);
        bulk_g2s(dst + kFc2W, A, abytes, &full[slot]);                          // hi [M x I]
        bulk_g2s(dst + kFc2W + kFc2A / 2, A + kMaxM * kI, abytes, &full[slot]); // lo [M x I]
        bulk_g2s(dst + kFc2W + kFc2A, S, 16, &full[slot]);
    };
    if (tid == 0)
    {
        for (int i = 0; i < kFc2Ring; i++)
            mbar_init(&full[i], 1);
        mbar_init_fence();
        for (int si = 0; si < kFc2Ring && si < U; si++)
            issue(si);
    }
    __syncthreads();
    bool const tok = g < M;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
    for (int si = 0; si < U; si++)
    {
        mbar_wait(&full[si % kFc2Ring], (si / kFc2Ring) & 1);
        uint8_t const* st = smem + (si % kFc2Ring) * kFc2Stage;
        float const sc = reinterpret_cast<float const*>(st + kFc2W + kFc2A)[kq];
        uint8_t const* wr = st + (mt * 16 + g) * kRowBytes + kq * 128 + 16 * t4;
        __half const* actHi = reinterpret_cast<__half const*>(st + kFc2W) + (tok ? g : 0) * kI + kq * 128 + 16 * t4;
        __half const* actLo = actHi + kMaxM * kI; // the lo rows start kMaxM rows after the hi rows
        float blk[4][4] = {};
#pragma unroll
        for (int h = 0; h < 2; h++)
        { // the quarter's two 64-wide K blocks
            const uint4 wg = *reinterpret_cast<uint4 const*>(wr + h * 64);
            const uint4 wg8 = *reinterpret_cast<uint4 const*>(wr + 8 * kRowBytes + h * 64);
#pragma unroll
            for (int j = 0; j < 4; j++)
            {
                uint2 hv = make_uint2(0u, 0u), lv = make_uint2(0u, 0u);
                if (tok)
                {
                    hv = *reinterpret_cast<uint2 const*>(actHi + h * 64 + 4 * j);
                    lv = *reinterpret_cast<uint2 const*>(actLo + h * 64 + 4 * j);
                }
                chunk_mma(blk[h], word(wg, j), word(wg8, j), hv.x, hv.y);
                chunk_mma(blk[2 + h], word(wg, j), word(wg8, j), lv.x, lv.y);
            }
        }
#pragma unroll
        for (int c = 0; c < 4; c++)
            acc[c] = fmaf((blk[0][c] + blk[1][c]) + (blk[2][c] + blk[3][c]), sc, acc[c]);
        __syncthreads(); // every warp is done with this slot
        if (tid == 0 && si + kFc2Ring < U)
            issue(si + kFc2Ring);
    }
    // sum the K quarters: red[warp][row][token]
#pragma unroll
    for (int c = 0; c < 4; c++)
        red[(warp * 16 + g + 8 * (c >> 1)) * 8 + 2 * t4 + (c & 1)] = acc[c];
    __syncthreads();
    if (tid < kFc2Rows * M)
    {
        int const row = tid % kFc2Rows, t = tid / kFc2Rows, m = row / 16, r = row % 16;
        float v = 0.f;
#pragma unroll
        for (int q = 0; q < 4; q++)
            v += red[((q * 2 + m) * 16 + r) * 8 + t];
        p.out[(size_t) t * kH + n0 + row] = __float2bfloat16(v);
    }
    cudaTriggerProgrammaticLaunchCompletion();
}

template <typename K>
cudaError_t launch(K kernel, dim3 grid, int smem, MoeParams const& p, cudaStream_t stream, bool pdl)
{
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, kernel, p);
}

} // namespace

cudaError_t decodeMoe(DecodeMoeArgs const& a, cudaStream_t stream)
{
    if (a.rows < 1 || a.rows > kMaxM)
    {
        return cudaErrorInvalidValue;
    }
    int dev = 0, numSms = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err == cudaSuccess)
    {
        err = cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, dev);
    }
    if (err == cudaSuccess)
    {
        err = cudaFuncSetAttribute(fc1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kFc1Smem);
    }
    if (err == cudaSuccess)
    {
        err = cudaFuncSetAttribute(fc2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kFc2Smem);
    }
    if (err != cudaSuccess)
    {
        return err;
    }
    MoeParams p{static_cast<__nv_bfloat16 const*>(a.x), a.logits, a.bias, a.w31, a.s31, a.sw31, a.ss31, a.w2, a.s2,
        a.sw2, a.ss2, static_cast<__half*>(a.actw), a.route, a.wtab, static_cast<__nv_bfloat16*>(a.out), a.scaling,
        a.limit, a.rows};
    err = launch(fc1_kernel, dim3(numSms), kFc1Smem, p, stream, a.pdl);
    if (err != cudaSuccess)
    {
        return err;
    }
    return launch(fc2_kernel, dim3(kH / kFc2Rows), kFc2Smem, p, stream, a.pdl);
}

} // namespace glm53_fp8
