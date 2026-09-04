# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Goal 5.1 TP-projection registry, swap, and shard-aware loader contracts.

Cheap gates for the Stage-5 dense path: the four-rank ownership registry
resolves every named projection to its declared column/row/replicated form,
the post-construction swap leaves no raw ``nn.Linear`` anywhere, converted
modules carry Mapping/tp-mode/full-shape metadata, and the TP1 form of every
converted module is numerically identical to what the frozen PP4 evidence ran
(BF16 -> the same ``F.linear``; block-FP8 -> the same subclass and kernel).

The real four-rank construction/loading/reconstruction evidence is produced by
``glm5_next_tp4_dense_loader.py`` under ``mpirun -n 4`` in a sanctioned
session; these tests are the pure-contract layer that must stay green on every
loader change.
"""

from __future__ import annotations

import json
import os

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_glm5_next import (
    Glm5NextForCausalLM,
    Glm5NextLoadReport,
    Glm5NextTpSpec,
    build_glm5_next_quant_config,
    resolve_glm5_next_projection_spec,
)
from tensorrt_llm.mapping import Mapping

CHECKPOINT = os.environ.get("GLM53_FLASH_CHECKPOINT", "/dev/shm/GLM-5.3-Flash")

pytestmark = [
    pytest.mark.skipif(
        not os.path.isdir(CHECKPOINT), reason=f"requires the checkpoint at {CHECKPOINT}"
    ),
]

TP4_MAPPING = dict(world_size=4, tp_size=4, pp_size=1, rank=0, gpus_per_node=4)


def _mapping(moe_tp: int, moe_ep: int) -> Mapping:
    return Mapping(moe_tp_size=moe_tp, moe_ep_size=moe_ep, **TP4_MAPPING)


# ---------------------------------------------------------------------------
# Registry resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "suffix,mode,reduce_output",
    [
        ("model.layers.0.self_attn.q_proj", "column", True),
        ("model.layers.0.self_attn.k_proj", "column", True),
        ("model.layers.0.self_attn.v_proj", "column", True),
        ("model.layers.0.self_attn.f_b_proj", "column", True),
        ("model.layers.0.self_attn.g_b_proj", "column", True),
        ("model.layers.0.self_attn.b_proj", "column", True),
        ("model.layers.0.self_attn.f_a_proj", None, True),
        ("model.layers.0.self_attn.g_a_proj", None, True),
        ("model.layers.0.self_attn.o_proj", "row", True),
        ("model.layers.3.self_attn.q_a_proj", None, True),
        ("model.layers.3.self_attn.kv_a_proj_with_mqa", None, True),
        ("model.layers.3.self_attn.q_b_proj", "column", True),
        ("model.layers.3.self_attn.kv_b_proj", "column", True),
        ("model.layers.3.self_attn.o_proj", "row", True),
        ("model.layers.3.self_attn.indexer.wq_b", "column", True),
        ("model.layers.3.self_attn.indexer.weights_proj", "column", True),
        ("model.layers.3.self_attn.indexer.wk", None, True),
        ("model.layers.0.mlp.gate_proj", "column", True),
        ("model.layers.0.mlp.up_proj", "column", True),
        ("model.layers.0.mlp.down_proj", "row", True),
    ],
)
def test_registry_resolves_named_projections(suffix, mode, reduce_output):
    for mapping in (_mapping(4, 1), _mapping(1, 4)):
        spec = resolve_glm5_next_projection_spec(suffix, mapping)
        assert spec.mode == mode, suffix
        assert spec.reduce_output == reduce_output, suffix


def test_registry_shared_expert_follows_moe_layout():
    tp4 = _mapping(4, 1)
    ep4 = _mapping(1, 4)
    base = "model.layers.4.mlp.shared_experts"
    # TP4 layout: sharded like a dense MLP, but down_proj keeps a partial so
    # Glm5NextMoE can sum routed+shared before its single all-reduce.
    assert resolve_glm5_next_projection_spec(f"{base}.gate_proj", tp4) == Glm5NextTpSpec("column")
    assert resolve_glm5_next_projection_spec(f"{base}.up_proj", tp4) == Glm5NextTpSpec("column")
    assert resolve_glm5_next_projection_spec(f"{base}.down_proj", tp4) == Glm5NextTpSpec(
        "row", reduce_output=False
    )
    # TP4/EP4 layout: fully replicated, added once after the EP combine.
    for proj in ("gate_proj", "up_proj", "down_proj"):
        assert resolve_glm5_next_projection_spec(f"{base}.{proj}", ep4) == Glm5NextTpSpec(None)


def test_registry_rejects_unknown_projection():
    with pytest.raises(ValueError, match="no tensor-parallel ownership"):
        resolve_glm5_next_projection_spec("model.layers.0.self_attn.mystery_proj", _mapping(4, 1))
    # A bare shared_experts path with an unknown leaf must also refuse.
    with pytest.raises(ValueError, match="no tensor-parallel ownership"):
        resolve_glm5_next_projection_spec(
            "model.layers.4.mlp.shared_experts.mystery", _mapping(1, 4)
        )


# ---------------------------------------------------------------------------
# Swap over the real model (meta, TP1)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tp1_model():
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(CHECKPOINT)
    quant_config = build_glm5_next_quant_config(config)
    model_config = ModelConfig(
        pretrained_config=config,
        quant_config=quant_config,
        moe_backend=ModelConfig.resolve_moe_backend(
            "AUTO", "Glm5NextForConditionalGeneration", quant_config
        ),
    )
    keys = list(
        json.load(open(os.path.join(CHECKPOINT, "model.safetensors.index.json")))["weight_map"]
    )
    with torch.device("meta"):
        model = Glm5NextForCausalLM(model_config)
    placement = model.apply_quant_plan(keys)
    return model, placement


def test_swap_leaves_no_raw_linear(tp1_model):
    from tensorrt_llm._torch.modules.linear import Linear

    model, placement = tp1_model
    raw = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and not isinstance(module, Linear)
    ]
    assert raw == []
    dtypes = {}
    for dtype in placement.values():
        dtypes[dtype] = dtypes.get(dtype, 0) + 1
    # 179 audited block-FP8 projections; every other named projection is a
    # checkpoint-published BF16 module and must stay BF16.
    assert dtypes == {"float8_e4m3fn": 179, "bfloat16": 350}


def test_swap_preserves_full_shapes_and_metadata(tp1_model):
    from tensorrt_llm._torch.modules.linear import TensorParallelMode

    model, _ = tp1_model
    l0 = model.model.layers[0].self_attn
    l3 = model.model.layers[3].self_attn
    moe = model.model.layers[4].mlp
    expectations = [
        (l0.q_proj, TensorParallelMode.COLUMN, (8192, 4096), torch.bfloat16),
        (l0.b_proj, TensorParallelMode.COLUMN, (64, 4096), torch.bfloat16),
        (l0.f_a_proj, None, (128, 4096), torch.bfloat16),
        (l0.o_proj, TensorParallelMode.ROW, (4096, 8192), torch.bfloat16),
        (l3.q_a_proj, None, (1536, 4096), torch.float8_e4m3fn),
        (l3.q_b_proj, TensorParallelMode.COLUMN, (16384, 1536), torch.float8_e4m3fn),
        (l3.kv_a_proj_with_mqa, None, (512, 4096), torch.float8_e4m3fn),
        (l3.kv_b_proj, TensorParallelMode.COLUMN, (32768, 512), torch.bfloat16),
        (l3.o_proj, TensorParallelMode.ROW, (4096, 16384), torch.float8_e4m3fn),
        (l3.indexer.wq_b, TensorParallelMode.COLUMN, (4096, 1536), torch.bfloat16),
        (l3.indexer.wk, None, (128, 4096), torch.bfloat16),
        (l3.indexer.weights_proj, TensorParallelMode.COLUMN, (32, 4096), torch.bfloat16),
        (moe.shared_experts.gate_proj, None, (2048, 4096), torch.float8_e4m3fn),
        (moe.shared_experts.down_proj, None, (4096, 2048), torch.float8_e4m3fn),
    ]
    for module, mode, full_shape, dtype in expectations:
        assert module.tp_mode == mode
        assert module.glm5_full_shape == full_shape
        # TP1: local == full.
        assert tuple(module.weight.shape) == full_shape
        assert module.weight.dtype == dtype
        assert module.glm5_tp_spec is not None
    # The base-class LMHead is already Mapping-aware and vocab-column-sharded.
    assert model.lm_head.tp_mode == TensorParallelMode.COLUMN
    assert tuple(model.lm_head.weight.shape) == (154880, 4096)


def test_moe_single_reduction_wiring_tp1(tp1_model):
    model, _ = tp1_model
    moe = model.model.layers[4].mlp
    # TP1: no all-reduce object, composition unchanged.
    assert moe.moe_all_reduce is None
    assert moe.experts is not None
    # Router and correction bias stay replicated FP32 parameters.
    assert moe.gate.weight.dtype == torch.float32
    assert moe.gate.e_score_correction_bias.dtype == torch.float32


def test_load_report_records_shards():
    report = Glm5NextLoadReport()
    report.loaded = 2
    report.remote_experts = 3
    report.tp_shards["self_attn.q_proj"] = {"mode": "column", "range": [0, 2048]}
    summary = report.summary()
    assert summary["remote_experts"] == 3
    assert summary["total"] == 5
    assert summary["tp_shards"]["self_attn.q_proj"]["range"] == [0, 2048]


# ---------------------------------------------------------------------------
# GPU: shard-aware loader on one owner (TP1 degenerate path)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_linear_group_loader_tp1_owner(tp1_model):
    """The loader routes converted projections through Linear.load_weights.

    TP1 keeps slicing degenerate, so this proves the group routing, the lazy
    handover, the lm_head empty-module-path case, and bitwise parity of the
    converted module against the raw checkpoint math the PP4 oracle ran.
    """
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from glm5_next_full_model import LazyCheckpoint

    from tensorrt_llm._torch.models.modeling_glm5_next import (
        Disposition,
        _destination_owner,
        audit_glm5_next_checkpoint,
        glm5_next_block_fp8_matmul,
        remap_glm5_next_key,
    )

    model, _ = tp1_model
    weights = LazyCheckpoint(CHECKPOINT)
    audit = audit_glm5_next_checkpoint(list(weights.keys()), model.model_config.pretrained_config)
    num_layers = model.schedule.num_layers
    by_owner = {}
    for key, disposition in audit.disposition.items():
        if disposition == Disposition.IGNORED:
            continue
        dest = remap_glm5_next_key(key)
        owner = _destination_owner(dest, num_layers)
        by_owner.setdefault(owner, []).append((key, dest, disposition))

    device = torch.device("cuda", torch.cuda.current_device())
    report = Glm5NextLoadReport()
    for owner, module, prefix in (
        (0, model.model.layers[0], "model.layers.0."),
        ("head", model.lm_head, "lm_head."),
    ):
        module.to_empty(device=device)
        model._fill_module(module, owner, prefix, by_owner[owner], weights, device, report)

    # Shard records exist for every converted projection of the owner, keyed
    # by the full destination path (the lm_head owner's relative path is
    # empty, so its record lands under the bare prefix).
    assert "model.layers.0.self_attn.q_proj" in report.tp_shards
    assert "lm_head" in report.tp_shards
    assert report.tp_shards["lm_head"]["full_shape"] == [154880, 4096]

    torch.manual_seed(3)
    x = torch.randn(8, 4096, dtype=torch.bfloat16, device=device)
    l0 = model.model.layers[0].self_attn
    assert torch.equal(l0.q_proj(x), torch.nn.functional.linear(x, l0.q_proj.weight))
    mlp = model.model.layers[0].mlp
    ref = glm5_next_block_fp8_matmul(x, mlp.gate_proj.weight, mlp.gate_proj.weight_scale)
    assert torch.equal(mlp.gate_proj(x), ref)
    logits = model.lm_head(x)
    assert logits.shape == (8, 154880)
    assert bool(torch.isfinite(logits).all())
