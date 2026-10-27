# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight loading of glm-5.3-flash-fp8 / sm_10x / tp4 (rank r of 4).

The checkpoint is the block-FP8 release (128 x 128 weight blocks, ``weight_scale_inv``):
the routed and shared experts, the three dense MLPs and the MLA projections q_a, q_b,
kv_a and o_proj are FP8; KDA, the indexer, kv_b, mHC, the embedding, the LM head and the
MTP layer are BF16. The text decoder places every tensor through its exact-placement
loader, which asserts both directions of coverage; the vision tower is not loaded.
"""

from typing import Any


def load_weights(text_model, weights: Any) -> None:
    """Fill the text decoder (and its MTP layer) from the checkpoint dict for this rank."""
    text_model.load_weights(weights)
