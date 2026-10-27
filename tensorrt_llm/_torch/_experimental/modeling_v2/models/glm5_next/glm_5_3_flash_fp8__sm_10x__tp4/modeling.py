# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash block-FP8 on sm_100 / sm_103, tp4: model construction and forward.

The checkpoint is the composite ``Glm5NextForConditionalGeneration`` release served
text only (``disable_mm_encoder``). ``ModelingV2Glm53FlashFp8Sm10xTp4`` is the outer
model the engine builds; it holds the text decoder, whose core picks one of two
targets per step:

- ``PrefillTarget`` for steps that carry context rows;
- ``DecodeTarget`` for generation-only steps (the steps CUDA graphs capture).
"""

from __future__ import annotations

import copy
from typing import Any, Optional

import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM
from tensorrt_llm._torch.models.modeling_glm5_next import (
    Glm5NextForCausalLM,
    Glm5NextModel,
    _normalize_glm5_next_top_config,
    glm5_next_is_quantized,
    glm5_next_text_quant_config,
    resolve_glm5_next_schedule,
)
from tensorrt_llm._torch.models.modeling_glm5_next_vision import Glm5NextVLM
from tensorrt_llm._torch.models.modeling_speculative import SpecDecOneEngineForCausalLM
from tensorrt_llm._torch.pyexecutor.config_utils import unwrap_glm5_next_text_config
from tensorrt_llm._torch.speculative import SpecMetadata
from tensorrt_llm.logger import logger

from . import weights as _weights

# The deployment this target is written for; route() selects it on the same facts,
# and the constructor states them again for a direct construction.
_SMS = ((10, 0), (10, 3))
_TP = 4


class PrefillTarget:
    """Forwards that carry context rows: the decoder stack over the packed batch."""

    def __init__(self, core: "Glm53FlashFp8Sm10xTp4Core") -> None:
        self.core = core

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        return Glm5NextModel.forward(
            self.core,
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )


class DecodeTarget:
    """Generation-only forwards: every row is a generation (or verify) row."""

    def __init__(self, core: "Glm53FlashFp8Sm10xTp4Core") -> None:
        self.core = core

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        return Glm5NextModel.forward(
            self.core,
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )


class Glm53FlashFp8Sm10xTp4Core(Glm5NextModel):
    """The decoder stack, with the per-step choice between the two targets."""

    def __init__(self, model_config: ModelConfig[PretrainedConfig]) -> None:
        super().__init__(model_config)
        self._prefill: Optional[PrefillTarget] = None
        self._decode: Optional[DecodeTarget] = None

    @torch.no_grad()
    def post_load_weights(self) -> None:
        """Build the targets once the weights are final. Reached more than once (the
        shell and the loader's module walk), so it only builds once."""
        if self._decode is not None:
            return
        self._prefill = PrefillTarget(self)
        self._decode = DecodeTarget(self)

    def _select_target(self, attn_metadata: AttentionMetadata) -> "PrefillTarget | DecodeTarget":
        if self._decode is None:
            self.post_load_weights()
        return self._prefill if attn_metadata.num_contexts > 0 else self._decode

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self._select_target(attn_metadata).forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            **kwargs,
        )


class Glm53FlashFp8Sm10xTp4ForCausalLM(Glm5NextForCausalLM):
    """The text decoder with the checkpoint's MTP layer as its one-model drafter."""

    def __init__(self, model_config: ModelConfig[PretrainedConfig]) -> None:
        text_config = unwrap_glm5_next_text_config(model_config.pretrained_config)
        model_config = glm5_next_text_quant_config(model_config)
        assert not bool(getattr(text_config, "tie_word_embeddings", False)), (
            "the checkpoint's lm_head is its own weight, not the embedding"
        )
        _normalize_glm5_next_top_config(model_config.pretrained_config)
        SpecDecOneEngineForCausalLM.__init__(
            self,
            Glm53FlashFp8Sm10xTp4Core(model_config),
            model_config,
            hidden_size=int(text_config.hidden_size),
            vocab_size=int(text_config.vocab_size),
        )
        self.text_config = text_config
        self.schedule = resolve_glm5_next_schedule(model_config.pretrained_config)
        self.mtp_layers = ()
        spec_config = model_config.spec_config
        if spec_config is not None and spec_config.spec_dec_mode.is_mtp_one_model():
            mtp_layers = tuple(self.draft_model.mtp_layers)
            assert len(mtp_layers) == 1, f"one MTP layer (layer 45), not {len(mtp_layers)}"
            # The draft layer is reached as model.layers[45], the checkpoint's own name.
            self.model.layers.extend(mtp_layers)
            self.mtp_layers = mtp_layers
        self.quantized = glm5_next_is_quantized(model_config)
        assert self.quantized, "the block-FP8 checkpoint loads quantized"

    def post_load_weights(self) -> None:
        super().post_load_weights()
        self.model.post_load_weights()


class ModelingV2Glm53FlashFp8Sm10xTp4(Glm5NextVLM):
    """GLM-5.3-Flash block-FP8, sm_100 / sm_103, tp4, text only."""

    def __init__(self, model_config: ModelConfig[PretrainedConfig], *args, **kwargs) -> None:
        mapping = model_config.mapping
        assert torch.cuda.get_device_capability() in _SMS, (
            f"this target runs on sm100 / sm103, not {torch.cuda.get_device_capability()}"
        )
        assert (
            mapping.world_size == _TP
            and mapping.tp_size == _TP
            and mapping.moe_tp_size == _TP
            and not mapping.enable_attention_dp
            and mapping.pp_size == 1
        ), f"this target is tp4 with the experts split by intermediate, not {mapping}"
        assert model_config.disable_mm_encoder, "this target serves the text model only"
        config = model_config.pretrained_config
        self.original_arch = config.architectures[0]
        # PreTrainedModel's constructor, without the built-in wrapper's: that one builds
        # the built-in text decoder through the registry.
        super(Glm5NextVLM, self).__init__(config)
        self.model_config = model_config

        llm_model_config = copy.deepcopy(model_config)
        # Share the live extra_attrs dict: the attention layers register their per-layer
        # metadata there and the engine reads the same dict.
        llm_model_config.extra_attrs = model_config.extra_attrs
        llm_model_config._frozen = False
        llm_model_config.pretrained_config.architectures = ["Glm5NextForCausalLM"]
        llm_model_config.modeling_v2_target = Glm53FlashFp8Sm10xTp4ForCausalLM
        llm_model_config._frozen = True
        self.llm = AutoModelForCausalLM.from_config(llm_model_config)
        self.mm_encoder = None
        # device='cpu' keeps the constant real under meta-device construction.
        self._mm_token_ids = torch.tensor(
            [int(config.image_token_id), int(config.video_token_id)],
            dtype=torch.int32,
            device="cpu",
        )
        self.post_config()
        logger.info(f"modeling_v2: {type(self).__name__} built")

    def load_weights(self, weights: Any, **_ignored: Any) -> None:
        _weights.load_weights(self.llm, weights)
