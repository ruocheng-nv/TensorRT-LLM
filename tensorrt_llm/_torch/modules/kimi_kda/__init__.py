# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi Delta Attention (KDA) in-tree module for TensorRT-LLM's PyTorch backend.

KDA is the linear-attention block used at ``linear_attn_config.kda_layers``
positions in the Kimi K3 text-core. It carries a short-convolution state and a
delta-rule recurrent state per layer, so it follows the hybrid-cache /
mamba ownership pattern rather than the paged-KV FMHA attention-backend
interface.

The mixer is imported lazily (PEP 562): it depends on the external ``fla``
package for its FLA fallback paths, while the in-tree kernel dispatch
(``_kda_kernels``) and the fused decode wrapper (``_kda_decode``) are
self-contained. Consumers of only the in-tree kernels — e.g. other KDA-family
models reusing ``trtllm::kda_prefill`` — must stay importable on systems
without ``fla`` installed.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .kimi_kda_mixer import KimiKDALinearAttention

__all__ = ["KimiKDALinearAttention"]


def __getattr__(name: str):
    if name == "KimiKDALinearAttention":
        from .kimi_kda_mixer import KimiKDALinearAttention

        return KimiKDALinearAttention
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
