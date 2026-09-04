# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime-binding increment R3: cache-manager selection and config plumbing.

Pins the executor-side resolution chain the LLM API will traverse for the
real checkpoint, without any model weights:

* ``is_glm5_next`` / ``is_hybrid_linear`` recognize the composite and the
  flattened text config, and nothing else;
* the literal ``layer_types`` list resolves to the pinned 34 KDA + 11 sparse
  inventory (sparse at zero-based 3, 7, ..., 43), agreeing with the model's
  own schedule resolution;
* ``ModelConfig.get_num_attention_layers`` / ``get_num_mamba_layers`` report
  11 / 34 (the hybrid manager sizes its pools from these);
* ``get_kv_cache_manager_cls`` returns the model's ``Glm5NextCacheManager``
  (a ``MambaHybridCacheManagerV2`` subclass) for V2 and refuses V1 and
  conflicting manager-preference knobs rather than selecting a manager that
  would silently drop the indexer state;
* ``extract_mamba_kv_cache_params`` maps the KDA geometry (64 heads x 128,
  four-tap conv, fp32 recurrent state) out of ``linear_attn_config``;
* ``_create_kv_cache_manager`` (the executor's construction entry) builds a
  working manager for the small 6-layer config on CUDA, with the indexer
  buffers registered and the slot-table accessor live.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor.config_utils import (
    extract_mamba_kv_cache_params,
    get_glm5_next_layer_masks,
    is_glm5_next,
    is_hybrid_linear,
    unwrap_glm5_next_text_config,
)

CHECKPOINT = os.environ.get("GLM53_FLASH_CHECKPOINT", "/dev/shm/GLM-5.3-Flash")


def _real_config():
    from transformers import AutoConfig

    if not os.path.isdir(CHECKPOINT):
        pytest.skip(f"requires the real config at {CHECKPOINT}")
    return AutoConfig.from_pretrained(CHECKPOINT)


def test_is_glm5_next_detects_both_config_shapes():
    config = _real_config()
    assert config.model_type == "glm5_next"
    assert is_glm5_next(config)
    assert is_glm5_next(config.text_config)
    assert is_hybrid_linear(config)
    assert unwrap_glm5_next_text_config(config) is config.text_config
    assert unwrap_glm5_next_text_config(config.text_config) is config.text_config

    # No false positives on unrelated configs.
    class _Other:
        model_type = "qwen3_next_text"
        layer_types = ["linear_attention"]

    assert not is_glm5_next(_Other())
    assert not is_glm5_next(object())


def test_layer_masks_match_the_pinned_literal_inventory():
    from tensorrt_llm._torch.models.modeling_glm5_next import (
        SPARSE_ATTENTION,
        resolve_glm5_next_schedule,
    )

    config = _real_config()
    full_mask, kda_mask = get_glm5_next_layer_masks(config)
    assert len(full_mask) == len(kda_mask) == 45
    assert sum(full_mask) == 11
    assert sum(kda_mask) == 34
    assert [i for i, on in enumerate(full_mask) if on] == list(range(3, 44, 4))
    # Every layer is exactly one of the two.
    assert all(f != k for f, k in zip(full_mask, kda_mask))

    # The executor-side masks agree with the model's own schedule resolution.
    schedule = resolve_glm5_next_schedule(config)
    assert full_mask == [t == SPARSE_ATTENTION for t in schedule.attention]

    # A corrupted list is rejected, not miscounted.
    bad = _real_config()
    bad.text_config.layer_types = list(bad.text_config.layer_types)
    bad.text_config.layer_types[7] = "full_attention"
    with pytest.raises(ValueError, match="exactly one of"):
        get_glm5_next_layer_masks(bad)


def test_model_config_reports_the_hybrid_layer_split():
    model_config = ModelConfig.from_pretrained(CHECKPOINT)
    assert is_glm5_next(model_config.pretrained_config)
    assert model_config.get_num_attention_layers() == 11
    assert model_config.get_num_mamba_layers() == 34


def test_extracted_mamba_params_pin_the_kda_geometry():
    config = _real_config()
    params = extract_mamba_kv_cache_params(config)
    assert params.state_size == 128
    assert params.conv_kernel == 4
    assert params.num_heads == 64
    assert params.n_groups == 64
    assert params.head_dim == 128
    assert params.num_mamba_layers == 34
    assert sum(params.target_full_attention_layer_mask) == 11
    # KDA delta-rule recurrent state stays fp32 (Stage-1 parity contract).
    assert params.mamba_ssm_cache_dtype == torch.float32


def test_kv_cache_manager_cls_resolves_to_the_glm5_manager():
    from tensorrt_llm._torch.pyexecutor._util import get_kv_cache_manager_cls
    from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import MambaHybridCacheManagerV2
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig

    model_config = ModelConfig.from_pretrained(CHECKPOINT)

    cls = get_kv_cache_manager_cls(model_config, KvCacheConfig(use_kv_cache_manager_v2=True))
    assert cls.__name__ == "Glm5NextCacheManager"
    assert issubclass(cls, MambaHybridCacheManagerV2)

    # V1 / compatibility managers cannot hold the indexer state: loud error,
    # never a silent fallback.
    with pytest.raises(ValueError, match="V2"):
        get_kv_cache_manager_cls(model_config, KvCacheConfig(use_kv_cache_manager_v2=False))
    with pytest.raises((ValueError, NotImplementedError), match="glm5_next"):
        get_kv_cache_manager_cls(
            model_config,
            KvCacheConfig(use_kv_cache_manager_v2=True),
            is_disagg=True,
        )

    monkey = os.environ.get("TLLM_MAMBA_MANAGER_PREFERENCE")
    os.environ["TLLM_MAMBA_MANAGER_PREFERENCE"] = "MIXED"
    try:
        with pytest.raises(ValueError, match="glm5_next"):
            get_kv_cache_manager_cls(model_config, KvCacheConfig(use_kv_cache_manager_v2=True))
    finally:
        if monkey is None:
            del os.environ["TLLM_MAMBA_MANAGER_PREFERENCE"]
        else:
            os.environ["TLLM_MAMBA_MANAGER_PREFERENCE"] = monkey


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_create_kv_cache_manager_builds_the_glm5_manager():
    """The executor's construction entry builds a live manager end to end.

    Uses the structurally faithful 6-layer small config (2 sparse + 4 KDA
    layers) so the pools stay tiny; asserts the pieces later increments rely
    on: KDA conv/ssm pools, slot-major latent/index views for exactly the
    sparse layers, and the raw slot-table accessor.
    """
    from test_glm5_next_runtime_binding import _small_glm5_next_config

    from tensorrt_llm._torch.model_config import _mirror_text_subconfig_attrs
    from tensorrt_llm._torch.pyexecutor._util import (
        _create_kv_cache_manager,
        get_kv_cache_manager_cls,
    )
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    config = _small_glm5_next_config()
    # The runtime's ModelConfig.from_pretrained mirrors text-config attributes
    # onto the composite config; _create_kv_cache_manager's generic prelude
    # (hidden_size, num_attention_heads) relies on that. Apply the same
    # normalization the runtime applies.
    _mirror_text_subconfig_attrs(config)
    model_config = ModelConfig(pretrained_config=config)
    kv_cache_config = KvCacheConfig(
        use_kv_cache_manager_v2=True, max_tokens=4096, enable_block_reuse=False
    )
    manager_cls = get_kv_cache_manager_cls(model_config, kv_cache_config)
    assert manager_cls.__name__ == "Glm5NextCacheManager"

    manager = _create_kv_cache_manager(
        model_engine=None,
        kv_cache_manager_cls=manager_cls,
        mapping=Mapping(world_size=1, tp_size=1, pp_size=1),
        kv_cache_config=kv_cache_config,
        tokens_per_block=64,
        max_seq_len=256,
        max_batch_size=2,
        spec_config=None,
        sparse_attention_config=None,
        max_num_tokens=512,
        max_beam_width=1,
        kv_connector_manager=None,
        model_config=model_config,
        dtype=torch.bfloat16,
        is_draft=False,
    )
    try:
        assert manager.sparse_layer_ids == [3, 5]
        assert manager.index_state_dim == 2 * int(config.text_config.index_head_dim)
        # KDA pools exist exactly on the linear layers, slot views exactly on
        # the sparse layers.
        for layer_id, layer_type in enumerate(config.text_config.layer_types):
            if layer_type == "linear_attention":
                assert manager.get_conv_states(layer_id) is not None
                assert manager.get_ssm_states(layer_id) is not None
            else:
                latent = manager.get_latent_state_buffer(layer_id)
                index = manager.get_index_state_buffer(layer_id)
                assert latent is not None and index is not None
                assert latent.shape[0] == index.shape[0]
                assert latent.shape[-1] == int(config.text_config.kv_lora_rank)
                assert index.shape[-1] == manager.index_state_dim
        # SSM recurrent state is fp32; conv state bf16 (Stage-1 contract).
        assert manager.get_ssm_states(0).dtype == torch.float32
        assert manager.get_conv_states(0).dtype == torch.bfloat16
        # The raw slot-table accessor is live on real allocations.
        manager.add_dummy_requests([7], token_nums=[70])
        rows = manager.get_batch_slot_tables([7])
        assert len(rows) == 1 and len(rows[0]) == 2
        latent = manager.get_latent_state_buffer(3)
        assert max(rows[0]) < latent.shape[0]
    finally:
        manager.shutdown()
