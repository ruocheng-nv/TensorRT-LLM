# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash target routing, driven by synthetic configs on CPU."""

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import modeling_v2_resolve
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig, TorchLlmArgs
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

_DUMMY_MODEL = "/tmp/dummy_model"
_TARGET = "models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.modeling.ModelingV2Glm53FlashFp8Sm10xTp4"
_TP4 = dict(world_size=4, tp_size=4, moe_tp_size=4, moe_ep_size=1, gpus_per_node=8)


def _qualified(target) -> str:
    """``models.<family>.<dir>.modeling.<Class>`` for a resolved target class."""
    return f"{target.__module__.split('.modeling_v2.')[1]}.{target.__name__}"


@pytest.fixture(autouse=True)
def _on_sm100(monkeypatch):
    """Route as if this were a B200, wherever the test actually runs."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))


def _llm_args(mode="auto", **kwargs) -> TorchLlmArgs:
    kwargs.setdefault("disable_mm_encoder", True)
    return TorchLlmArgs(model=_DUMMY_MODEL, modeling_v2=mode, **kwargs)


def _glm53_config(**text_overrides):
    """The composite config the GLM-5.3-Flash release declares (text half only)."""
    text = dict(
        model_type="glm5_next_text",
        num_hidden_layers=45,
        hidden_size=4096,
        n_routed_experts=288,
        q_lora_rank=1536,
    )
    text.update(text_overrides)
    config = PretrainedConfig(
        architectures=["Glm5NextForConditionalGeneration"], model_type="glm5_next"
    )
    config.text_config = PretrainedConfig(**text)
    return config


def _model_config(
    pretrained_config,
    spec_config=None,
    quant_algo=QuantAlgo.FP8_BLOCK_SCALES,
    kv_cache_quant_algo=None,
    **mapping_kwargs,
):
    return ModelConfig(
        pretrained_config=pretrained_config,
        mapping=Mapping(**(mapping_kwargs or _TP4)),
        spec_config=spec_config,
        quant_config=QuantConfig(quant_algo=quant_algo, kv_cache_quant_algo=kv_cache_quant_algo),
    )


@pytest.mark.parametrize("mode", ["auto", "require"])
@pytest.mark.parametrize(
    "spec_config", [None, MTPDecodingConfig(max_draft_len=3)], ids=["no-spec", "mtp3"]
)
@pytest.mark.parametrize("kv_cache_quant_algo", [None, QuantAlgo.FP8], ids=["bf16-kv", "fp8-kv"])
@pytest.mark.parametrize("sm", [(10, 0), (10, 3)], ids=["sm100", "sm103"])
def test_fp8_tp4_matches(monkeypatch, sm, kv_cache_quant_algo, spec_config, mode):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: sm)
    config = _model_config(
        _glm53_config(), spec_config=spec_config, kv_cache_quant_algo=kv_cache_quant_algo, **_TP4
    )
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _TARGET


@pytest.mark.parametrize(
    "config_kwargs, model_kwargs, llm_kwargs, missed",
    [
        (dict(num_hidden_layers=78), {}, {}, r"shape \(78, 4096, 288, 1536\)"),
        (dict(), dict(quant_algo=QuantAlgo.NVFP4), {}, r"quantization (QuantAlgo\.)?NVFP4"),
        (dict(), dict(kv_cache_quant_algo=QuantAlgo.NVFP4), {}, r"KV cache (QuantAlgo\.)?NVFP4"),
        (dict(), dict(world_size=4, tp_size=4, moe_ep_size=4, moe_tp_size=1), {}, "moe_ep=4"),
        (
            dict(),
            dict(world_size=4, tp_size=4, enable_attention_dp=True, moe_tp_size=4),
            {},
            "attention_dp=True",
        ),
        (dict(), dict(world_size=8, tp_size=8, moe_tp_size=8), {}, "world_size=8"),
        (dict(), dict(spec_config=MTPDecodingConfig(max_draft_len=2)), {}, "max_draft_len=2"),
        (
            dict(),
            dict(spec_config=MTPDecodingConfig(max_draft_len=3, use_mtp_vanilla=True)),
            {},
            "MTP",
        ),
        (dict(), {}, dict(disable_mm_encoder=False), "disable_mm_encoder=False"),
    ],
    ids=[
        "other-shape",
        "nvfp4",
        "nvfp4-kv",
        "expert-parallel",
        "attention-dp",
        "tp8",
        "mtp2",
        "vanilla-mtp",
        "vision-tower",
    ],
)
def test_near_misses_do_not_match(config_kwargs, model_kwargs, llm_kwargs, missed):
    config = _model_config(_glm53_config(**config_kwargs), **model_kwargs)
    assert modeling_v2_resolve(config, _llm_args("auto", **llm_kwargs)) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require", **llm_kwargs))


@pytest.mark.parametrize("sm", [(9, 0), (12, 0)], ids=["sm90", "sm120"])
def test_other_gpus_do_not_match(monkeypatch, sm):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: sm)
    config = _model_config(_glm53_config())
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
