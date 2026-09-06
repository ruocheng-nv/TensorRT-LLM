# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Goal 5.2 regression tier: every GLM TP collective is pinned to NCCL.

The iteration-42 TP4 bring-up found two real engine failures caused by the
default ``AllReduceStrategy.AUTO``: the AllReduce autotuner deadlocked all four
ranks during engine warmup while profiling the indexer's tiny FP32 score
reduction, and its selected tactic raced at decode producing intermittent NaN
logits (masked by any added host sync). The fix bakes
``AllReduceStrategy.NCCL`` into every TP collective the model constructs:

* ``Glm5NextIndexer.score_all_reduce`` (the one FP32 pool-score reduction),
* ``Glm5NextMoE.moe_all_reduce`` (the routed/shared combine in both layouts),
* every row-parallel ``Linear`` installed by
  ``glm5_next_swap_quantized_projections``.

These tests pin that invariant so a refactor cannot silently reintroduce AUTO,
and pin the TP1 counterpart: at ``tp_size == 1`` the model constructs **no**
reduction at all (the frozen PP4/tp1 oracle must stay byte-identical, with no
collective in its schedule).

Construction-only: modules are built on ``meta`` with the package ``AllReduce``
stubbed (the stub records ``mapping``/``strategy`` exactly as passed), so this
tier runs single-process with no MPI/NCCL session.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.distributed import AllReduceStrategy
from tensorrt_llm._torch.models.modeling_glm5_next import (
    Glm5NextIndexer,
    Glm5NextMoE,
    glm5_next_swap_quantized_projections,
    resolve_glm5_next_projection_spec,
)
from tensorrt_llm.mapping import Mapping

TP = 4


class _StubAllReduce:
    """Records constructor arguments; never touches MPI/NCCL."""

    instances = []

    def __init__(self, mapping=None, strategy=AllReduceStrategy.AUTO, dtype=None):
        self.mapping = mapping
        self.strategy = strategy
        self.dtype = dtype
        _StubAllReduce.instances.append(self)

    def __call__(self, x, **kwargs):
        return x


@pytest.fixture
def stub_allreduce(monkeypatch):
    import tensorrt_llm._torch.distributed as dist_pkg

    _StubAllReduce.instances = []
    # Both owners import AllReduce lazily inside __init__
    # (``from ..distributed import AllReduce``), so patching the package
    # attribute intercepts every construction site under test.
    monkeypatch.setattr(dist_pkg, "AllReduce", _StubAllReduce)
    return _StubAllReduce


def _indexer_config():
    return SimpleNamespace(
        hidden_size=4096,
        q_lora_rank=1536,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=2048,
        index_kpool=4,
        index_kpool_always_select_tail=True,
    )


def _moe_config():
    return SimpleNamespace(
        hidden_size=4096,
        intermediate_size=12288,
        moe_intermediate_size=2048,
        n_routed_experts=288,
        n_shared_experts=1,
        num_experts_per_tok=8,
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        swiglu_limit=10.0,
    )


def _mapping(tp_size: int, *, moe_tp: int = None, moe_ep: int = None) -> Mapping:
    if tp_size == 1:
        return Mapping()
    return Mapping(
        world_size=tp_size,
        tp_size=tp_size,
        pp_size=1,
        rank=0,
        gpus_per_node=tp_size,
        moe_tp_size=moe_tp if moe_tp is not None else tp_size,
        moe_ep_size=moe_ep if moe_ep is not None else 1,
    )


# ---------------------------------------------------------------------------
# indexer score reduction
# ---------------------------------------------------------------------------


def test_indexer_score_all_reduce_pinned_nccl_at_tp4(stub_allreduce):
    with torch.device("meta"):
        idx = Glm5NextIndexer(_indexer_config(), 3, mapping=_mapping(TP))
    assert idx.score_all_reduce is not None, "TP4 indexer must own the FP32 score reduction"
    assert idx.score_all_reduce.strategy == AllReduceStrategy.NCCL, (
        f"indexer score_all_reduce resolved {idx.score_all_reduce.strategy!r}, not the "
        "NCCL pin (AUTO deadlocked engine warmup and raced at decode on TP4)"
    )
    assert len(stub_allreduce.instances) == 1, "exactly one collective per indexer"


def test_indexer_tp1_builds_no_reduction(stub_allreduce):
    with torch.device("meta"):
        bare = Glm5NextIndexer(_indexer_config(), 3)
        tp1 = Glm5NextIndexer(_indexer_config(), 3, mapping=SimpleNamespace(tp_size=1, tp_rank=0))
    assert bare.score_all_reduce is None
    assert tp1.score_all_reduce is None
    assert stub_allreduce.instances == [], "tp1 indexer constructed a collective"


# ---------------------------------------------------------------------------
# MoE combine
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_create_moe(monkeypatch):
    """Replace the fused-MoE factory: strategy pinning is a constructor-side
    contract of Glm5NextMoE itself, independent of which backend resolves.
    """
    import importlib

    # The fused_moe package re-exports the ``create_moe`` *function* as a
    # package attribute, shadowing the submodule on attribute access; the
    # model's lazy ``from ..moe.fused_moe.create_moe import create_moe``
    # resolves through sys.modules, so the patch must land on the real module.
    create_moe_mod = importlib.import_module("tensorrt_llm._torch.moe.fused_moe.create_moe")

    class _StubBackend:
        pass

    class _StubExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.backend = _StubBackend()

    monkeypatch.setattr(create_moe_mod, "create_moe", lambda **kwargs: _StubExperts())
    return create_moe_mod


@pytest.mark.parametrize(
    "moe_tp,moe_ep",
    [(4, 1), (1, 4)],
    ids=["tp4_layout", "tp4ep4_layout"],
)
def test_moe_all_reduce_pinned_nccl_in_both_layouts(
    stub_allreduce, stub_create_moe, moe_tp, moe_ep
):
    mapping = _mapping(TP, moe_tp=moe_tp, moe_ep=moe_ep)
    model_config = SimpleNamespace(mapping=mapping)
    with torch.device("meta"):
        moe = Glm5NextMoE(_moe_config(), quantized=True, model_config=model_config, layer_idx=4)
    assert moe.moe_all_reduce is not None, "tp_size=4 MoE must own its combine reduction"
    assert moe.moe_all_reduce.strategy == AllReduceStrategy.NCCL, (
        f"moe_all_reduce resolved {moe.moe_all_reduce.strategy!r}, not the NCCL pin"
    )
    strategies = [inst.strategy for inst in stub_allreduce.instances]
    assert strategies == [AllReduceStrategy.NCCL], strategies


def test_moe_tp1_and_diagnostic_build_no_reduction(stub_allreduce, stub_create_moe):
    with torch.device("meta"):
        production_tp1 = Glm5NextMoE(
            _moe_config(),
            quantized=True,
            model_config=SimpleNamespace(mapping=_mapping(1)),
            layer_idx=4,
        )
        diagnostic = Glm5NextMoE(_moe_config())
    assert production_tp1.moe_all_reduce is None
    assert diagnostic.moe_all_reduce is None
    assert stub_allreduce.instances == [], "tp1/diagnostic MoE constructed a collective"


# ---------------------------------------------------------------------------
# swapped Mapping-aware Linears
# ---------------------------------------------------------------------------


class _ProjectionTree(nn.Module):
    """Minimal module tree covering every swap-relevant ownership class.

    The spec registry matches dotted suffixes like ``.self_attn.q_proj``, so
    the projections sit under a ``layer0`` parent to give every name a leading
    path segment (as in the real ``model.layers.<i>`` namespace).
    """

    def __init__(self):
        super().__init__()
        self.layer0 = nn.Module()
        self.layer0.self_attn = nn.Module()
        self.layer0.self_attn.q_proj = nn.Linear(4096, 8192, bias=False, dtype=torch.bfloat16)
        self.layer0.self_attn.o_proj = nn.Linear(8192, 4096, bias=False, dtype=torch.bfloat16)
        self.layer0.self_attn.q_a_proj = nn.Linear(4096, 1536, bias=False, dtype=torch.bfloat16)
        self.layer0.mlp = nn.Module()
        self.layer0.mlp.gate_proj = nn.Linear(4096, 12288, bias=False, dtype=torch.bfloat16)
        self.layer0.mlp.down_proj = nn.Linear(12288, 4096, bias=False, dtype=torch.bfloat16)

    @property
    def attn(self):
        return self.layer0.self_attn

    @property
    def ffn(self):
        return self.layer0.mlp


class _SharedExpertTree(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer4 = nn.Module()
        self.layer4.mlp = nn.Module()
        self.layer4.mlp.shared_experts = nn.Module()
        self.layer4.mlp.shared_experts.gate_proj = nn.Linear(
            4096, 2048, bias=False, dtype=torch.bfloat16
        )
        self.layer4.mlp.shared_experts.down_proj = nn.Linear(
            2048, 4096, bias=False, dtype=torch.bfloat16
        )


def test_swap_pins_every_constructed_allreduce_to_nccl(stub_allreduce):
    with torch.device("meta"):
        root = _ProjectionTree()
        glm5_next_swap_quantized_projections(root, plan={}, mapping=_mapping(TP))

    from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode

    rows = {"self_attn.o_proj": root.attn.o_proj, "mlp.down_proj": root.ffn.down_proj}
    for name, mod in rows.items():
        assert isinstance(mod, Linear), name
        assert mod.tp_mode == TensorParallelMode.ROW, name
        assert mod.reduce_output, f"{name} must reduce inside the Linear (one all-reduce)"
        assert mod.all_reduce is not None, name
        assert mod.all_reduce.strategy == AllReduceStrategy.NCCL, (
            f"{name} row reduction resolved {mod.all_reduce.strategy!r}, not the NCCL pin"
        )
    # The uniform pin also covers any AllReduce a non-row mode constructs
    # (column/replicated construct-but-never-invoke): nothing under the swap
    # may carry AUTO.
    for inst in stub_allreduce.instances:
        assert inst.strategy == AllReduceStrategy.NCCL, (
            f"a swapped Linear constructed an AllReduce with {inst.strategy!r}"
        )
    assert stub_allreduce.instances, "expected at least the two row reductions"


def test_swap_tp4_shared_expert_down_stays_partial(stub_allreduce):
    with torch.device("meta"):
        root = _SharedExpertTree()
        glm5_next_swap_quantized_projections(
            root, plan={}, mapping=_mapping(TP, moe_tp=4, moe_ep=1)
        )
    down = root.layer4.mlp.shared_experts.down_proj
    assert not down.reduce_output, (
        "TP4-layout shared down_proj must keep a partial; Glm5NextMoE owns the single combine"
    )
    assert down.all_reduce is None


def test_swap_tp1_builds_no_tp_reduction(stub_allreduce):
    with torch.device("meta"):
        root = _ProjectionTree()
        glm5_next_swap_quantized_projections(root, plan={}, mapping=_mapping(1))
    for mod in (root.attn.o_proj, root.ffn.down_proj):
        assert mod.tp_size == 1
        assert mod.all_reduce is None or mod.all_reduce.mapping.tp_size == 1, (
            "tp1 swap must not construct a multi-rank reduction"
        )


def test_spec_registry_declares_single_reduction_per_branch():
    """The row set is exactly the branch outputs; every other spec never
    reduces — the structural half of the exactly-one-reduction invariant.
    """
    mapping = _mapping(TP)
    rows = {".self_attn.o_proj", ".mlp.down_proj"}
    for suffix in rows:
        spec = resolve_glm5_next_projection_spec(f"model.layers.0{suffix}", mapping)
        assert spec.mode == "row" and spec.reduce_output
    for suffix in (
        ".self_attn.q_proj",
        ".self_attn.kv_b_proj",
        ".self_attn.indexer.wq_b",
        ".mlp.gate_proj",
    ):
        spec = resolve_glm5_next_projection_spec(f"model.layers.0{suffix}", mapping)
        assert spec.mode == "column"
    for suffix in (".self_attn.q_a_proj", ".self_attn.indexer.wk"):
        spec = resolve_glm5_next_projection_spec(f"model.layers.0{suffix}", mapping)
        assert spec.mode is None
    shared = resolve_glm5_next_projection_spec(
        "model.layers.4.mlp.shared_experts.down_proj", mapping
    )
    assert shared.mode == "row" and not shared.reduce_output
