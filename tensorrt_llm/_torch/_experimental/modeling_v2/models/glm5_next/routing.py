# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a Glm5NextForConditionalGeneration config lands. Read this file and you know.

``route`` is one forward-reading decision: the facts that fix which code runs,
compared against the deployment the target below was written for. When none fits
it raises ``NoTarget`` stating those facts. In ``auto`` the engine then uses the
built-in implementation (``modeling_glm5_next``); in ``require`` it raises that
error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from ..._router_index import NoTarget, engine_kv_cache_quant_algo, routes

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

# (layers, hidden, routed experts, q_lora_rank) of the text config. Shape-sniffing
# also matches fine-tunes of the same shape; the accuracy gate pins identity.
_SHAPE = (45, 4096, 288, 1536)

# sm_100 (B200, GB200) and sm_103 (B300, GB300).
_SMS = frozenset({(10, 0), (10, 3)})


@routes("Glm5NextForConditionalGeneration")
def route(
    model_config: ModelConfig, llm_args: TorchLlmArgs, compute_capability: Tuple[int, int]
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits.

    The target is imported where it is returned, so declining costs no import of
    modeling code.
    """
    from tensorrt_llm._torch.pyexecutor.config_utils import unwrap_glm5_next_text_config

    mapping = model_config.mapping
    spec_config = model_config.spec_config
    text = unwrap_glm5_next_text_config(model_config.pretrained_config)
    shape = (
        text.num_hidden_layers,
        text.hidden_size,
        text.n_routed_experts,
        text.q_lora_rank,
    )
    quant = model_config.quant_config
    quant_algo = None if quant is None else quant.quant_algo
    kv_cache_quant_algo = engine_kv_cache_quant_algo(model_config, llm_args)

    # tp4: tensor-parallel attention over the 4 ranks of one node, the routed experts
    # split by intermediate (moe_tp 4). A weight-layout question, which is why it
    # selects a target rather than a runtime branch.
    tp4 = (
        mapping.world_size == 4
        and mapping.tp_size == 4
        and mapping.moe_tp_size == 4
        and mapping.moe_ep_size == 1
        and not mapping.enable_attention_dp
        and mapping.pp_size == 1
        and mapping.cp_size == 1
        and mapping.gpus_per_node >= 4
    )
    # The checkpoint's MTP layer driven by the one-model MTP-Eagle loop at three
    # drafts, or no speculation.
    mtp3_or_none = spec_config is None or (
        isinstance(spec_config, MTPDecodingConfig)
        and spec_config.spec_dec_mode.is_mtp_eagle_one_model()
        and spec_config.max_draft_len == 3
        and spec_config.max_total_draft_tokens == spec_config.max_draft_len
    )

    # GLM-5.3-Flash, block-FP8 release, text only, on sm_100 / sm_103, tp4, with a
    # BF16 or FP8 latent cache.
    if (
        compute_capability in _SMS
        and shape == _SHAPE
        and quant_algo == QuantAlgo.FP8_BLOCK_SCALES
        and tp4
        and mtp3_or_none
        and kv_cache_quant_algo in (None, QuantAlgo.FP8)
        and llm_args.disable_mm_encoder
    ):
        # One line on purpose: the claims suite reads the routed targets off
        # `from .<target>.modeling import <Class>` lines.
        # fmt: off
        # isort: off
        from .glm_5_3_flash_fp8__sm_10x__tp4.modeling import ModelingV2Glm53FlashFp8Sm10xTp4
        # isort: on
        # fmt: on
        return ModelingV2Glm53FlashFp8Sm10xTp4
    raise NoTarget(
        f"compute capability {compute_capability}, config shape {shape}, quantization "
        f"{quant_algo}, world_size={mapping.world_size} tp={mapping.tp_size} "
        f"moe_tp={mapping.moe_tp_size} moe_ep={mapping.moe_ep_size} "
        f"attention_dp={mapping.enable_attention_dp} pp={mapping.pp_size} cp={mapping.cp_size}, "
        f"speculative decoding {spec_config.spec_dec_mode.name if spec_config else 'off'}"
        f"{f' max_draft_len={spec_config.max_draft_len}' if spec_config else ''}, KV cache "
        f"{kv_cache_quant_algo or 'bf16'}, disable_mm_encoder={llm_args.disable_mm_encoder}; "
        "the one glm5_next target is GLM-5.3-Flash block-FP8 on sm100 / sm103, tp4 (4 ranks of "
        "one node, tensor-parallel attention, experts split by intermediate), with one-model "
        "MTP-Eagle at max_draft_len=3 or no speculative decoding, a BF16 or FP8 latent "
        "cache (kv_cache_config.dtype auto or fp8) and the vision tower disabled (disable_mm_encoder)"
    )
