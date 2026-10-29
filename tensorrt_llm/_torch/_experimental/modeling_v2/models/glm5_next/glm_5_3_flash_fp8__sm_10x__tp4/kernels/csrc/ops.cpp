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
// Python bindings of the target's CUDA kernels: tensors in, launches on the current stream.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "kernels/decode_moe.h"
#include "kernels/mhc_ar_boundary.h"

namespace
{

void decodeMoe(torch::Tensor const& x, torch::Tensor const& logits, torch::Tensor const& bias, double scaling,
    double limit, torch::Tensor const& w31, torch::Tensor const& s31, torch::Tensor const& w2, torch::Tensor const& s2,
    torch::Tensor const& sw31, torch::Tensor const& ss31, torch::Tensor const& sw2, torch::Tensor const& ss2,
    torch::Tensor& out, bool pdl)
{
    glm53_fp8::DecodeMoeArgs a{};
    a.x = x.data_ptr();
    a.logits = logits.data_ptr<float>();
    a.bias = bias.data_ptr<float>();
    a.w31 = static_cast<uint8_t const*>(w31.data_ptr());
    a.s31 = s31.data_ptr<float>();
    a.w2 = static_cast<uint8_t const*>(w2.data_ptr());
    a.s2 = s2.data_ptr<float>();
    a.sw31 = static_cast<uint8_t const*>(sw31.data_ptr());
    a.ss31 = ss31.data_ptr<float>();
    a.sw2 = static_cast<uint8_t const*>(sw2.data_ptr());
    a.ss2 = ss2.data_ptr<float>();
    a.out = out.data_ptr();
    a.scaling = static_cast<float>(scaling);
    a.limit = static_cast<float>(limit);
    a.rows = static_cast<int>(x.size(0));
    a.pdl = pdl;
    cudaError_t const err = glm53_fp8::decodeMoe(a, at::cuda::getCurrentCUDAStream(x.get_device()));
    TORCH_CHECK(err == cudaSuccess, "decode_moe launch failed: ", cudaGetErrorString(err));
}

void mhcArBoundary(torch::Tensor const& partial, torch::Tensor const& workspace, int64_t rank,
    torch::Tensor const& residual, torch::Tensor const& post, torch::Tensor const& comb, torch::Tensor const& fn,
    torch::Tensor const& scale, torch::Tensor const& base, torch::Tensor const& norm_weight, int64_t sinkhorn_iters,
    double norm_eps, double hc_eps, double sinkhorn_eps, double post_mult, double layer_input_eps,
    torch::Tensor& residual_out, torch::Tensor& post_out, torch::Tensor& comb_out, torch::Tensor& layer_input, bool pdl)
{
    glm53_fp8::MhcArBoundaryArgs a{};
    a.workspace = reinterpret_cast<void**>(workspace.data_ptr());
    a.rank = static_cast<int>(rank);
    a.partial = partial.data_ptr();
    a.residual = residual.data_ptr();
    a.post_prev = post.data_ptr<float>();
    a.comb_prev = comb.data_ptr<float>();
    a.fn = fn.data_ptr<float>();
    a.scale = scale.data_ptr<float>();
    a.base = base.data_ptr<float>();
    a.norm_weight = norm_weight.data_ptr();
    a.residual_out = residual_out.data_ptr();
    a.post_out = post_out.data_ptr<float>();
    a.comb_out = comb_out.data_ptr<float>();
    a.layer_input = layer_input.data_ptr();
    a.rows = static_cast<int>(partial.size(0));
    a.sinkhorn_iters = static_cast<int>(sinkhorn_iters);
    a.norm_eps = static_cast<float>(norm_eps);
    a.hc_eps = static_cast<float>(hc_eps);
    a.sinkhorn_eps = static_cast<float>(sinkhorn_eps);
    a.post_mult = static_cast<float>(post_mult);
    a.layer_input_eps = static_cast<float>(layer_input_eps);
    a.pdl = pdl;
    cudaError_t const err = glm53_fp8::mhcArBoundary(a, at::cuda::getCurrentCUDAStream(partial.get_device()));
    TORCH_CHECK(err == cudaSuccess, "mhc_ar_boundary launch failed: ", cudaGetErrorString(err));
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("decode_moe", &decodeMoe);
    m.attr("decode_moe_max_rows") = glm53_fp8::kDecodeMoeMaxRows;
    m.def("mhc_ar_boundary", &mhcArBoundary);
    m.attr("mhc_ar_boundary_max_rows") = glm53_fp8::kMhcArBoundaryMaxRows;
}
