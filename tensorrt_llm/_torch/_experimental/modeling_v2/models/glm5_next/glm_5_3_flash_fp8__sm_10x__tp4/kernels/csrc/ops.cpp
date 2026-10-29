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

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("decode_moe", &decodeMoe);
    m.attr("decode_moe_max_rows") = glm53_fp8::kDecodeMoeMaxRows;
}
