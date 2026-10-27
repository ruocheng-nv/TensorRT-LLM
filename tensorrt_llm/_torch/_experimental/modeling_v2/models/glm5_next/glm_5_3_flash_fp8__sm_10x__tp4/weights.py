# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight loading of glm-5.3-flash-fp8 / sm_10x / tp4 (rank r of 4).

The checkpoint is the block-FP8 release (128 x 128 weight blocks, ``weight_scale_inv``):
the routed and shared experts, the three dense MLPs and the MLA projections q_a, q_b,
kv_a and o_proj are FP8; KDA, the indexer, kv_b, mHC, the embedding, the LM head and the
MTP layer are BF16. The text decoder places every tensor through its exact-placement
loader, which asserts both directions of coverage; the vision tower is not loaded.

Besides its head shard, every rank also keeps the full q_b / kv_b / o_proj of each
decoder MLA layer: the sequence-parallel prefill runs all heads over the rows the rank
owns (``modeling.PrefillTarget``).
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet

import torch

from tensorrt_llm._torch.models.modeling_glm5_next import SPARSE_ATTENTION

_MLA_FULL = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.self_attn\."
    r"(?P<proj>q_b_proj|kv_b_proj|o_proj)\.(?P<tensor>weight|weight_scale_inv)$"
)


@dataclass
class MlaFullWeights:
    """One MLA layer's head-complete projections, in the checkpoint's own encoding."""

    q_b: torch.Tensor  # [heads * qk_head_dim, q_lora] FP8
    q_b_scale: torch.Tensor  # [heads * qk_head_dim / 128, q_lora / 128] FP32
    w_k: torch.Tensor  # [heads, qk_nope, kv_lora] BF16: kv_b's key rows, per head
    w_v_t: torch.Tensor  # [heads, kv_lora, v_head] BF16: kv_b's value rows, per head, transposed
    o: torch.Tensor  # [hidden, heads * v_head] FP8
    o_scale: torch.Tensor  # [hidden / 128, heads * v_head / 128] FP32


def mla_full_weights(
    weights: Any, mla_layers: FrozenSet[int], num_heads: int, qk_nope: int, v_head: int
) -> Dict[int, MlaFullWeights]:
    """The head-complete MLA projections of ``mla_layers``, on this rank's device."""
    device = torch.device("cuda", torch.cuda.current_device())
    found: Dict[int, Dict[str, torch.Tensor]] = {}
    for key in weights.keys():
        m = _MLA_FULL.match(key)
        if m is None or int(m["layer"]) not in mla_layers:
            continue
        tensor = weights[key]
        if not isinstance(tensor, torch.Tensor):
            tensor = tensor[:]  # a lazy safetensors slice
        found.setdefault(int(m["layer"]), {})[f"{m['proj']}.{m['tensor']}"] = tensor.to(device)
    out = {}
    for layer in sorted(mla_layers):
        t = found.get(layer, {})
        assert len(t) == 5, f"layer {layer}: MLA tensors {sorted(t)}"
        kv_b = t["kv_b_proj.weight"].view(num_heads, qk_nope + v_head, -1)
        out[layer] = MlaFullWeights(
            q_b=t["q_b_proj.weight"],
            q_b_scale=t["q_b_proj.weight_scale_inv"].float(),
            w_k=kv_b[:, :qk_nope, :].contiguous(),
            w_v_t=kv_b[:, qk_nope:, :].transpose(1, 2).contiguous(),
            o=t["o_proj.weight"],
            o_scale=t["o_proj.weight_scale_inv"].float(),
        )
    return out


def load_weights(text_model, weights: Any) -> None:
    """Fill the text decoder (and its MTP layer) from the checkpoint dict for this rank."""
    core = text_model.model
    config = core.config
    schedule = core.schedule
    mla_layers = frozenset(
        i for i in range(schedule.num_layers) if schedule.attention[i] == SPARSE_ATTENTION
    )
    core.mla_full = mla_full_weights(
        weights,
        mla_layers,
        int(config.num_attention_heads),
        int(config.qk_nope_head_dim),
        int(config.v_head_dim),
    )
    text_model.load_weights(weights)
