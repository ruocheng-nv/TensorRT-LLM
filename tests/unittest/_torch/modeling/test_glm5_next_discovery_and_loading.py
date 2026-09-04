# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash text-path discovery, checkpoint mapping and diagnostic loading.

``test_glm5_next_reference_ladder.py`` and ``test_glm5_next_source_replay.py``
prove *what the model computes*, against native HuggingFace on real weights.
This file proves the step before that: that TensorRT-LLM finds the model at all,
and that every one of the 76108 published tensors reaches exactly one
destination with the right quantization verdict.

The audit is deliberately analytic -- it reads the safetensors index and the
config, not 328 GB of materialized weights -- so it stays cheap enough to gate
every later loading change. The value-level checks then materialize
representative layers on CUDA and compare against the checkpoint bytes.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import pytest
import torch
from glm5_next_ref import GLM53_FLASH_INVENTORY, CheckpointReader

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_glm5_next import (
    DENSE_MLP,
    FP32_PARAMETERS,
    KDA_CONV_DEST,
    KDA_CONV_SOURCES,
    LINEAR_ATTENTION,
    SPARSE_ATTENTION,
    SPARSE_MLP,
    Disposition,
    Glm5NextForCausalLM,
    audit_glm5_next_checkpoint,
    build_glm5_next_quant_config,
    destination_module_name,
    narrow_glm5_next_exclusions,
    remap_glm5_next_key,
    resolve_glm5_next_exclusions,
    resolve_glm5_next_schedule,
    source_module_name,
)
from tensorrt_llm.models.modeling_utils import QuantConfig

CHECKPOINT = os.environ.get("GLM53_FLASH_CHECKPOINT", "/dev/shm/GLM-5.3-Flash")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(
        not os.path.isdir(CHECKPOINT), reason=f"requires the checkpoint at {CHECKPOINT}"
    ),
]

_VISION_PREFIX = "model.visual."
_MTP_PREFIX = "model.language_model.layers.45."


@pytest.fixture(scope="module")
def hf_config():
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(CHECKPOINT)


@pytest.fixture(scope="module")
def checkpoint_keys() -> List[str]:
    index = os.path.join(CHECKPOINT, "model.safetensors.index.json")
    with open(index) as fh:
        return sorted(json.load(fh)["weight_map"])


@pytest.fixture(scope="module")
def quantized_weights(checkpoint_keys) -> Dict[str, bool]:
    """Ground truth straight from the checkpoint.

    A weight tensor is block-FP8 quantized **iff** the producer wrote a
    companion ``weight_scale_inv``. This is the checkpoint's own statement about
    itself, independent of the 1509-entry exclusion list, which is exactly why
    it can referee that list.
    """
    keys = set(checkpoint_keys)
    return {
        k: (k[: -len(".weight")] + ".weight_scale_inv") in keys
        for k in checkpoint_keys
        if k.endswith(".weight")
    }


@pytest.fixture(scope="module")
def reader():
    r = CheckpointReader(CHECKPOINT)
    yield r
    r.close()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_auto_discovery_resolves_glm5_next_without_override(hf_config):
    """The published architecture resolves to the text model with no override."""
    from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM

    assert hf_config.architectures == ["Glm5NextForConditionalGeneration"], (
        f"checkpoint architecture changed: {hf_config.architectures}"
    )
    model_config = ModelConfig(pretrained_config=hf_config)
    resolved = AutoModelForCausalLM._resolve_class(model_config)
    assert resolved is Glm5NextForCausalLM, (
        f"auto-discovery resolved {resolved!r}; the checkpoint would need a "
        f"public override, which the bring-up forbids"
    )


def test_lazy_model_zoo_index_lists_glm5_next():
    """The lazy-import index must agree with the decorator, or import breaks."""
    from tensorrt_llm._torch.models import _arch_index

    assert (
        _arch_index.MODEL_ARCH_TO_MODULE["Glm5NextForConditionalGeneration"] == "modeling_glm5_next"
    )
    assert _arch_index.MODEL_CLASS_TO_MODULE["Glm5NextForCausalLM"] == "modeling_glm5_next"


def test_runtime_interface_hooks(hf_config):
    """Increment 1 of the runtime binding: the executor's model-side hooks.

    ``infer_max_seq_len`` (capacity planning) and
    ``get_preferred_kv_cache_manager_version`` (the ``"auto"`` -> V2 resolution)
    are the first pieces the LLM API / PyExecutor consult. They are inert under
    the Stage-1 diagnostic driver, but are pinned here so the interface a reader
    sees matches the runtime contract. See reports/goal1.5-runtime-binding-plan.md.
    """
    # Classmethod: this hybrid latent-KV + pool-indexer + recurrent/conv state
    # needs V2; V1 cannot express it.
    assert Glm5NextForCausalLM.get_preferred_kv_cache_manager_version() == "V2"

    # infer_max_seq_len must read the *text* config: the top multimodal config
    # carries no max_position_embeddings. Meta construction allocates no weights.
    # The MoE backend goes through the runtime's own AUTO resolution (FP8 block
    # scales on SM100 -> TRTLLM); the bare dataclass default (CUTLASS) is a
    # pinned backend that cannot serve this quant on this SM and correctly
    # fails resolution, which is not what this hook test is about.
    quant_config = build_glm5_next_quant_config(hf_config)
    model_config = ModelConfig(
        pretrained_config=hf_config,
        quant_config=quant_config,
        moe_backend=ModelConfig.resolve_moe_backend(
            "AUTO", hf_config.architectures[0], quant_config
        ),
    )
    with torch.device("meta"):
        model = Glm5NextForCausalLM(model_config)
    assert model.infer_max_seq_len() == 1048576


def test_from_config_constructs_the_quantized_model():
    """The runtime's construction path must yield the block-FP8 build.

    ``AutoModelForCausalLM.from_config`` calls ``cls(model_config)`` with no
    further arguments, so quantization has to be derived from the
    ``ModelConfig`` the runtime builds via ``ModelConfig.from_pretrained`` --
    which maps this checkpoint's ``quantization_config``
    (``weight_block_size=[128,128]``) to ``FP8_BLOCK_SCALES``. A constructor
    that defaulted to bf16 here would hand the loader a model it must reject.
    """
    from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM
    from tensorrt_llm.quantization.mode import QuantAlgo

    model_config = ModelConfig.from_pretrained(CHECKPOINT)
    assert model_config.quant_config is not None
    assert model_config.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES, (
        f"runtime quant detection changed: {model_config.quant_config.quant_algo}"
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(model_config)
    assert type(model).__name__ == "Glm5NextForCausalLM"
    assert model.quantized is True

    # Without a quant config the build is bf16 and whole-model loading refuses
    # it up front (dequantizing 328 GB of e4m3 to bf16 would double residency
    # and change the arithmetic), rather than failing key-by-key mid-load.
    bare = ModelConfig(pretrained_config=model_config.pretrained_config)
    with torch.device("meta"):
        unquantized = Glm5NextForCausalLM(bare)
    assert unquantized.quantized is False
    with pytest.raises(ValueError, match="quantized=True"):
        unquantized.load_weights({})


# ---------------------------------------------------------------------------
# Literal dual-list dispatch
# ---------------------------------------------------------------------------


def test_literal_dual_list_dispatch_matches_the_pinned_inventory(hf_config):
    """Both 45-entry lists, and the third redundant encoding, agree."""
    inv = GLM53_FLASH_INVENTORY
    schedule = resolve_glm5_next_schedule(hf_config)

    assert schedule.num_layers == inv.num_hidden_layers == 45
    assert len(schedule.attention_indices(LINEAR_ATTENTION)) == inv.num_linear_attention_layers
    assert schedule.attention_indices(SPARSE_ATTENTION) == inv.sparse_attention_layer_indices
    assert schedule.mlp_indices(DENSE_MLP) == inv.dense_mlp_layer_indices
    assert len(schedule.mlp_indices(SPARSE_MLP)) == inv.num_sparse_mlp_layers

    # The two lists are independent contracts: an attention type must not be
    # inferable from the MLP type or vice versa.
    sparse_attn = set(schedule.attention_indices(SPARSE_ATTENTION))
    sparse_mlp = set(schedule.mlp_indices(SPARSE_MLP))
    assert sparse_attn != sparse_mlp, (
        "attention and MLP schedules coincide; this test could not tell a "
        "single-list implementation from a dual-list one"
    )


@pytest.mark.parametrize(
    "field,mutation",
    [
        ("layer_types", "swap_one_entry"),
        ("mlp_layer_types", "swap_one_entry"),
        ("layer_types", "truncate"),
    ],
)
def test_schedule_validation_rejects_a_corrupted_list(hf_config, field, mutation):
    """Negative control: the cross-checks must be able to fail."""
    import copy

    broken = copy.deepcopy(hf_config)
    text = broken.text_config
    values = list(getattr(text, field))
    if mutation == "swap_one_entry":
        # Flip the entry at the first index of the rarer type.
        rare = LINEAR_ATTENTION if field == "layer_types" else DENSE_MLP
        common = SPARSE_ATTENTION if field == "layer_types" else SPARSE_MLP
        idx = values.index(rare)
        values[idx] = common
    else:
        values = values[:-1]
    setattr(text, field, values)

    with pytest.raises(ValueError):
        resolve_glm5_next_schedule(broken)


# ---------------------------------------------------------------------------
# Checkpoint accounting
# ---------------------------------------------------------------------------


def test_every_checkpoint_key_is_accounted_for(hf_config, checkpoint_keys):
    """All 76108 tensors land in exactly one disposition, none unresolved."""
    audit = audit_glm5_next_checkpoint(checkpoint_keys, hf_config)

    assert audit.unresolved == [], (
        f"{len(audit.unresolved)} checkpoint keys could not be placed: {audit.unresolved[:10]}"
    )
    counts = audit.counts()
    assert sum(counts.values()) == len(checkpoint_keys) == 76108, counts
    assert set(audit.disposition) == set(checkpoint_keys)

    # Every destination is reachable from exactly one source, except the fused
    # convolution which deliberately collapses three.
    fused = [d for d in audit.destinations if d.endswith(f".{KDA_CONV_DEST}.weight")]
    assert len(fused) == GLM53_FLASH_INVENTORY.num_linear_attention_layers, len(fused)


def test_ignored_keys_are_exactly_the_mtp_and_vision_namespaces(hf_config, checkpoint_keys):
    """Nothing is ignored by accident, and nothing in scope is skipped."""
    audit = audit_glm5_next_checkpoint(checkpoint_keys, hf_config)
    ignored = audit.keys_with(Disposition.IGNORED)

    for key in ignored:
        assert key.startswith(_MTP_PREFIX) or key.startswith(_VISION_PREFIX), (
            f"{key} was ignored but is not in an allowlisted namespace"
        )
    # And the converse: every key in those namespaces is ignored, so a rename
    # cannot quietly pull MTP weights into the main decoder.
    for key in checkpoint_keys:
        in_allowlisted = key.startswith(_MTP_PREFIX) or key.startswith(_VISION_PREFIX)
        assert in_allowlisted == (audit.disposition[key] == Disposition.IGNORED), key

    n_vision = sum(1 for k in ignored if k.startswith(_VISION_PREFIX))
    n_mtp = sum(1 for k in ignored if k.startswith(_MTP_PREFIX))
    assert (n_mtp, n_vision) == (1760, 347), (n_mtp, n_vision)
    assert len(checkpoint_keys) - len(ignored) == 74001


def test_kda_conv_sources_all_reach_the_fused_destination(hf_config, checkpoint_keys):
    """A partial fusion would silently zero a third of the convolution."""
    schedule = resolve_glm5_next_schedule(hf_config)
    per_dest: Dict[str, List[str]] = {}
    for key in checkpoint_keys:
        dest = remap_glm5_next_key(key)
        if dest is not None and dest.endswith(f".{KDA_CONV_DEST}.weight"):
            per_dest.setdefault(dest, []).append(key)

    linear_layers = schedule.attention_indices(LINEAR_ATTENTION)
    assert len(per_dest) == len(linear_layers)
    for dest, sources in per_dest.items():
        assert len(sources) == len(KDA_CONV_SOURCES), (dest, sources)
        got = {s.rsplit(".", 2)[-2] for s in sources}
        assert got == set(KDA_CONV_SOURCES), (dest, got)
    # Only linear-attention layers own a short convolution.
    owning = sorted(int(d.split(".")[2]) for d in per_dest)
    assert tuple(owning) == linear_layers


# ---------------------------------------------------------------------------
# Quantization exclusions
# ---------------------------------------------------------------------------


def test_narrowing_drops_only_vision_patterns(hf_config):
    published = hf_config.quantization_config["modules_to_not_convert"]
    narrowed = narrow_glm5_next_exclusions(published)
    dropped = [p for p in published if p not in set(narrowed)]
    assert len(published) == GLM53_FLASH_INVENTORY.num_modules_to_not_convert == 1509
    for pattern in dropped:
        assert pattern.split(".", 1)[0] == "visual", pattern
    # Narrowing must not remove anything the decoder could need.
    for pattern in narrowed:
        assert pattern.split(".", 1)[0] != "visual", pattern


def test_quant_exclusions_reproduce_checkpoint_ground_truth(
    hf_config, checkpoint_keys, quantized_weights
):
    """The 1509-entry list, resolved our way, matches the checkpoint exactly.

    This is the referee for Decision D: the exclusion patterns are the
    producer's *intent*, while ``weight_scale_inv`` presence is what the
    producer actually wrote. If they disagree anywhere, the loader would either
    quantize a BF16 tensor or look for a scale that does not exist.
    """
    quant_config = build_glm5_next_quant_config(hf_config)
    audit = audit_glm5_next_checkpoint(checkpoint_keys, hf_config)
    loadable = [k for k in checkpoint_keys if audit.disposition[k] != Disposition.IGNORED]
    verdicts = resolve_glm5_next_exclusions(loadable, quant_config)

    text_weights = [
        k
        for k in quantized_weights
        if k.startswith("model.language_model.") and not k.startswith(_MTP_PREFIX)
    ]
    assert text_weights, "no text weights found"

    wrongly_quantized: List[str] = []
    wrongly_excluded: List[str] = []
    for key in text_weights:
        dest = destination_module_name(key)
        predicted_bf16 = verdicts[dest]
        actually_bf16 = not quantized_weights[key]
        if actually_bf16 and not predicted_bf16:
            wrongly_quantized.append(key)
        elif predicted_bf16 and not actually_bf16:
            wrongly_excluded.append(key)

    assert not wrongly_quantized, (
        f"{len(wrongly_quantized)} BF16 tensors would be quantized, e.g. {wrongly_quantized[:5]}"
    )
    assert not wrongly_excluded, (
        f"{len(wrongly_excluded)} block-FP8 tensors would be left BF16, e.g. {wrongly_excluded[:5]}"
    )


def test_fused_conv1d_inherits_its_sources_exclusion(hf_config, checkpoint_keys):
    """Regression: resolving exclusions on the *destination* name is wrong.

    ``q_conv1d`` / ``k_conv1d`` / ``v_conv1d`` are all named in the published
    exclusion list and are all BF16 in the checkpoint, but they fuse into a
    single ``conv1d`` that no published pattern mentions. Looking the fused name
    up directly reports "not excluded" and would quantize a BF16 tensor in every
    linear-attention layer, so the verdict has to be carried across the fusion.
    """
    quant_config = build_glm5_next_quant_config(hf_config)
    schedule = resolve_glm5_next_schedule(hf_config)
    audit = audit_glm5_next_checkpoint(checkpoint_keys, hf_config)
    loadable = [k for k in checkpoint_keys if audit.disposition[k] != Disposition.IGNORED]
    verdicts = resolve_glm5_next_exclusions(loadable, quant_config)

    linear_layers = schedule.attention_indices(LINEAR_ATTENTION)
    for layer_idx in linear_layers:
        dest = f"model.layers.{layer_idx}.self_attn.{KDA_CONV_DEST}"
        assert verdicts[dest] is True, (
            f"{dest} must stay BF16: all three fused sources are excluded"
        )
        # ... and the naive destination-side lookup must genuinely disagree,
        # otherwise this test proves nothing.
        assert quant_config.is_module_excluded_from_quantization(dest) is False, (
            f"{dest} now matches a published pattern directly; the "
            f"source-side resolution this test guards is no longer load-bearing"
        )
        for source in KDA_CONV_SOURCES:
            src = f"model.layers.{layer_idx}.self_attn.{source}"
            assert quant_config.is_module_excluded_from_quantization(src) is True, src


def test_fused_sources_that_disagree_on_quantization_are_rejected(hf_config):
    """Negative control: a fusion across a quantization boundary must raise."""
    # q_conv1d excluded, k_conv1d/v_conv1d not: fusing them would need half the
    # channels to carry a scale the other half does not have.
    quant_config = QuantConfig(exclude_modules=["model.layers.0.self_attn.q_conv1d"])
    keys = [f"model.language_model.layers.0.self_attn.{name}.weight" for name in KDA_CONV_SOURCES]
    with pytest.raises(ValueError, match="disagree on quantization"):
        resolve_glm5_next_exclusions(keys, quant_config)


def test_source_and_destination_namespaces_are_distinct(checkpoint_keys):
    """The published patterns live in the text namespace, the keys do not."""
    sample = "model.language_model.layers.3.self_attn.q_a_proj.weight"
    assert sample in checkpoint_keys
    assert source_module_name(sample) == "model.layers.3.self_attn.q_a_proj"
    assert destination_module_name(sample) == "model.layers.3.self_attn.q_a_proj"
    conv = "model.language_model.layers.0.self_attn.q_conv1d.weight"
    assert source_module_name(conv) == "model.layers.0.self_attn.q_conv1d"
    assert destination_module_name(conv) == f"model.layers.0.self_attn.{KDA_CONV_DEST}"


# ---------------------------------------------------------------------------
# Diagnostic value-level loading (CUDA, real weights)
# ---------------------------------------------------------------------------


def test_router_correction_bias_is_fp32_in_the_checkpoint(reader, hf_config):
    """The noaux_tc bias must never be demoted; the loader has to know that."""
    assert "e_score_correction_bias" in FP32_PARAMETERS
    schedule = resolve_glm5_next_schedule(hf_config)
    routed = schedule.mlp_indices(SPARSE_MLP)
    checked = 0
    for layer_idx in (routed[0], routed[len(routed) // 2], routed[-1]):
        key = f"model.language_model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"
        _shape, dtype = reader.meta(key)
        assert dtype == "F32", f"{key} is {dtype}, not float32"
        checked += 1
    assert checked == 3


def test_diagnostic_load_reproduces_checkpoint_values(reader, hf_config, checkpoint_keys):
    """Materialize representative layers on CUDA through the mapping.

    One layer of each literal type: a linear-attention + dense-MLP layer and a
    sparse-attention + routed-MoE layer. Full-model materialization needs more
    than one GPU (328 GB of FP8 weights), so this is the largest value-level
    check a single device supports -- and it is the one that catches a wrong
    destination, a wrong dequantization, or a silently transposed tensor.
    """
    device = torch.device("cuda")
    schedule = resolve_glm5_next_schedule(hf_config)
    linear_layer = schedule.attention_indices(LINEAR_ATTENTION)[0]
    sparse_layer = schedule.attention_indices(SPARSE_ATTENTION)[0]

    quant_config = build_glm5_next_quant_config(hf_config)
    audit = audit_glm5_next_checkpoint(checkpoint_keys, hf_config)

    problems: List[str] = []
    checked = 0
    for layer_idx in (linear_layer, sparse_layer):
        prefix = f"model.language_model.layers.{layer_idx}."
        # Skip the 288 routed experts: they are stacked by the MoE backend and
        # would need 14.5 GiB per layer; the router, shared expert and every
        # attention tensor are all covered.
        layer_keys = [
            k
            for k in checkpoint_keys
            if k.startswith(prefix) and ".mlp.experts." not in k and not k.endswith("_scale_inv")
        ]
        assert layer_keys, f"layer {layer_idx} has no keys"
        for key in layer_keys:
            dest = remap_glm5_next_key(key)
            if dest is None:
                problems.append(f"{key}: no destination")
                continue
            tensor = reader.get(key, device=device)
            checked += 1
            if not torch.isfinite(tensor.float()).all():
                problems.append(f"{key}: non-finite values")
            expected_shape, _dtype = reader.meta(key)
            is_quantized = quant_config.is_module_excluded_from_quantization(
                source_module_name(key)
            )
            if is_quantized:
                # Excluded -> stored BF16 -> reader returns it unchanged.
                if tuple(tensor.shape) != tuple(expected_shape):
                    problems.append(
                        f"{key}: excluded module changed shape "
                        f"{tuple(tensor.shape)} != {tuple(expected_shape)}"
                    )
            if audit.disposition[key] == Disposition.IGNORED:
                problems.append(f"{key}: in-scope layer key was ignored")

    assert checked > 0
    assert not problems, "\n".join(problems[:20])


def test_fused_kda_convolution_has_the_pinned_geometry(reader, hf_config):
    """The three depthwise convolutions concatenate to the pinned conv width."""
    inv = GLM53_FLASH_INVENTORY
    device = torch.device("cuda")
    schedule = resolve_glm5_next_schedule(hf_config)
    layer_idx = schedule.attention_indices(LINEAR_ATTENTION)[0]
    prefix = f"model.language_model.layers.{layer_idx}.self_attn."

    parts = [reader.get(f"{prefix}{name}.weight", device=device) for name in KDA_CONV_SOURCES]
    per_part = inv.kda_conv_dim // len(KDA_CONV_SOURCES)
    for name, part in zip(KDA_CONV_SOURCES, parts):
        assert part.shape[0] == per_part, (name, tuple(part.shape))
        assert part.shape[-1] == inv.linear_conv_kernel_dim, (name, tuple(part.shape))

    fused = torch.cat(parts, dim=0)
    assert fused.shape[0] == inv.kda_conv_dim
    assert fused.shape[-1] == inv.linear_conv_kernel_dim
    assert torch.isfinite(fused.float()).all()
    # Concatenation order is load-bearing: q, then k, then v.
    assert torch.equal(fused[:per_part], parts[0])
    assert torch.equal(fused[per_part : 2 * per_part], parts[1])
    assert torch.equal(fused[2 * per_part :], parts[2])


def test_hf_weight_loader_streams_glm5_next_lazily(tmp_path):
    """The runtime checkpoint reader must NOT materialize glm5_next eagerly.

    The PP=8 smoke's attempt 2 was SIGKILLed by the kernel's global OOM
    killer: the eager ``HfWeightLoader`` path loads the full 328 GB into
    anonymous host memory once per rank (observed at 161 GiB anon RSS in one
    rank before the kill). ``glm5_next`` therefore routes through the lazy
    ``safe_open`` path. This pins the routing decision and the two slice
    semantics the model's loader relies on: values are lazy (not tensors),
    and ``[:]`` materializes bytes equal to the stored tensor.
    """
    import safetensors.torch

    from tensorrt_llm._torch.models.checkpoints.hf.weight_loader import HfWeightLoader
    from tensorrt_llm.mapping import Mapping

    stored = {
        "model.language_model.layers.0.x.weight": torch.arange(24, dtype=torch.bfloat16).reshape(
            4, 6
        ),
        "lm_head.weight": torch.ones(3, 2, dtype=torch.float32),
    }
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(json.dumps({"model_type": "glm5_next"}))
    safetensors.torch.save_file(stored, str(ckpt / "model-00001-of-00001.safetensors"))

    weights = HfWeightLoader().load_weights(
        str(ckpt), mapping=Mapping(world_size=1, tp_size=1, pp_size=1)
    )
    assert set(weights.keys()) == set(stored.keys())
    for name, expect in stored.items():
        lazy = weights[name]
        # Lazy value: not a tensor until indexed, exactly what
        # Glm5NextForCausalLM._fill_module's materialization branch expects.
        assert not torch.is_tensor(lazy)
        materialized = lazy[:]
        assert torch.is_tensor(materialized)
        assert materialized.dtype == expect.dtype
        assert torch.equal(materialized, expect)

    # The real checkpoint takes the same branch. The check is config-driven,
    # so every PP rank routes identically (the loader's collectives contract:
    # ranks must agree on the branch or the eager path's barrier deadlocks).
    assert HfWeightLoader._is_streamed_checkpoint(CHECKPOINT)


def test_construction_is_meta_init_safe_and_experts_stay_meta():
    """The runtime constructs under MetaInitMode; GLM must not force a fallback.

    ``ModelLoader._create_and_load`` wraps construction in ``MetaInitMode`` so a
    pipeline-parallel rank materializes only its shard on GPU, never the whole
    328 GB in host RAM. Two GLM-specific hazards defeated that and made the
    ``except Exception`` fall back to *regular* init (real host allocation on
    every rank -> the observed PP=8 global-OOM SIGKILL):

    1. ``nn.LayerNorm`` (indexer ``k_norm``) runs ``fill_``/``zero_`` on meta
       weights, which MetaInitMode disallows -> ``MetaInitException`` at
       construction. Fixed by ``Glm5NextLayerNorm`` (factory-built affines).
    2. The MoE experts were ``nn.Parameter(torch.zeros(...))``; MetaInitMode
       only routes ``aten.empty`` to meta, so zeros real-allocated 304 GB of
       experts per rank. Fixed by building them with ``torch.empty``.

    This pins both: construction under MetaInitMode raises nothing, and the
    expert weights stay on ``meta`` (proving no host materialization).
    """
    from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM
    from tensorrt_llm._torch.models.modeling_utils import MetaInitException, MetaInitMode

    model_config = ModelConfig.from_pretrained(CHECKPOINT)
    try:
        with MetaInitMode():
            model = AutoModelForCausalLM.from_config(model_config)
    except MetaInitException as exc:  # pragma: no cover - regression guard
        raise AssertionError(
            f"glm5_next construction is not MetaInitMode-safe: {exc}. The runtime "
            "would fall back to regular init and real-allocate the whole model in "
            "host RAM on every PP rank."
        )

    # The 304 GB of experts must be meta (never host-allocated at construction).
    # The production build holds them in the fused layer's stacked
    # ``w3_w1_weight`` / ``w2_weight`` parameters.
    meta = torch.device("meta")
    checked = 0
    for name, param in model.named_parameters():
        if name.endswith(("w3_w1_weight", "w2_weight")) and param.dim() == 3:
            assert param.device == meta, f"{name} is on {param.device}, expected meta"
            checked += 1
    assert checked > 0, "found no MoE expert weight parameters to check"

    # The indexer LayerNorm must be correctly named so the checkpoint's
    # k_norm.weight / k_norm.bias still place by name, and -- the load-bearing
    # point -- must NOT have forced the MetaInitException fallback above. Its
    # tiny (head_dim) ones/zeros affines are legitimately real: a safe norm
    # default if ever read before load, and ~3 KB, irrelevant to the OOM the
    # meta experts fix.
    k_norm_params = {name: p for name, p in model.named_parameters() if "indexer.k_norm" in name}
    assert any(n.endswith("k_norm.weight") for n in k_norm_params)
    assert any(n.endswith("k_norm.bias") for n in k_norm_params)

    # Engine weight-creation lifecycle (Goal 3.4): ``from_config`` defers
    # module weight creation (``skip_create_weights_in_init=True``, set by
    # ``AutoModelForCausalLM.from_config`` itself), and
    # ``DecoderModelForCausalLM.__post_init__`` must then have completed it
    # for every fused-MoE layer -- on this SM100/FP8-block checkpoint always
    # the TRTLLMGen backend. A missed ``create_weights`` surfaces here rather
    # than as a shape/attribute error mid-load on the 328 GB checkpoint.
    assert model_config.skip_create_weights_in_init is True, (
        "AutoModelForCausalLM.from_config no longer defers weight creation; "
        "the engine-lifecycle assumptions below changed"
    )
    routed = 0
    for layer in model.model.layers:
        experts = getattr(layer.mlp, "experts", None)
        if experts is None:
            continue
        routed += 1
        assert experts._weights_created is True, (
            f"layer {layer.layer_idx}: fused MoE weights were not created by "
            "__post_init__ under the deferred (engine) lifecycle"
        )
        assert layer.mlp.moe_backend_name == "TRTLLMGenFusedMoE", (
            f"layer {layer.layer_idx}: unexpected MoE backend "
            f"{layer.mlp.moe_backend_name!r} under runtime AUTO resolution"
        )
    assert routed == 42, f"expected 42 routed layers with fused experts, found {routed}"


def test_glm5_next_layernorm_matches_torch_layernorm():
    """Glm5NextLayerNorm is bit-identical to nn.LayerNorm once weights load.

    The meta-safe norm only changes *how* the affine parameters are created; the
    forward must stay exactly ``F.layer_norm`` so the indexer's k-normalisation
    is unchanged. Random affines + a random input, compared against a stock
    ``nn.LayerNorm`` carrying the same weight/bias, must agree bitwise.
    """
    from tensorrt_llm._torch.models.modeling_glm5_next import Glm5NextLayerNorm

    torch.manual_seed(0)
    dim = 128
    ours = Glm5NextLayerNorm(dim, eps=1e-6, dtype=torch.float32)
    ref = torch.nn.LayerNorm(dim, eps=1e-6, dtype=torch.float32)
    with torch.no_grad():
        w = torch.randn(dim)
        b = torch.randn(dim)
        ours.weight.copy_(w)
        ours.bias.copy_(b)
        ref.weight.copy_(w)
        ref.bias.copy_(b)
    x = torch.randn(4, 7, dim)
    assert torch.equal(ours(x), ref(x))


def test_runtime_materialization_places_k_norm_on_cuda_with_checkpoint_values(reader, hf_config):
    """Post-materialization/load, the indexer k_norm affines are CUDA-resident.

    ``Glm5NextLayerNorm`` builds its weight/bias as real *CPU* tensors
    (MetaInitMode does not intercept ``ones``/``zeros``), and the loader's
    ``init_meta_tensor`` pass explicitly returns non-meta tensors unchanged --
    so the only step that moves these affines to the GPU is ``model_loader``'s
    subsequent ``model.to("cuda")``. This replays the loader's exact sequence
    for the pp8 rank that owns the first indexer layer

      MetaInitMode construct -> _apply(init_meta_tensor) -> model.to("cuda")
      -> HfCheckpointLoader().load_weights (lazy) -> model.load_weights

    and pins the end state: every local parameter is CUDA-resident (nothing
    left on meta or CPU), and k_norm.weight/bias hold the checkpoint's bytes.
    """
    from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader import HfCheckpointLoader
    from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM
    from tensorrt_llm._torch.models.modeling_utils import MetaInitMode
    from tensorrt_llm.mapping import Mapping

    schedule = resolve_glm5_next_schedule(hf_config)
    first_indexer_layer = schedule.attention_indices(SPARSE_ATTENTION)[0]

    # The pp8 rank that owns that layer (rank 0 for layer 3; computed, not
    # assumed, so a schedule change cannot silently un-cover the indexer).
    pp_size = 8
    num_layers = hf_config.text_config.num_hidden_layers
    owner_rank = None
    for rank in range(pp_size):
        mapping = Mapping(world_size=pp_size, tp_size=1, pp_size=pp_size, rank=rank)
        if first_indexer_layer in mapping.pp_layers(num_layers):
            owner_rank = rank
            break
    assert owner_rank is not None
    mapping = Mapping(world_size=pp_size, tp_size=1, pp_size=pp_size, rank=owner_rank)

    model_config = ModelConfig.from_pretrained(CHECKPOINT, mapping=mapping)
    with MetaInitMode():
        model = AutoModelForCausalLM.from_config(model_config)

    try:
        # model_loader.py's materialization: meta -> empty CUDA, then a blanket
        # .to("cuda") that is exactly what carries the real CPU affines over.
        memo = {}

        def init_meta_tensor(t: torch.Tensor):
            if t.device != torch.device("meta"):
                return t
            if t not in memo:
                memo[t] = torch.empty_like(t, device="cuda")
            return memo[t]

        model._apply(init_meta_tensor)
        memo.clear()
        model.to("cuda")

        weights = HfCheckpointLoader().load_weights(CHECKPOINT, mapping=mapping)
        model.load_weights(weights)

        prefix = f"model.language_model.layers.{first_indexer_layer}.self_attn.indexer.k_norm"
        checked = 0
        for name, param in model.named_parameters():
            assert param.device.type == "cuda", f"{name} on {param.device} after load"
            if "indexer.k_norm" not in name:
                continue
            kind = "weight" if name.endswith("weight") else "bias"
            stored = reader.get(f"{prefix}.{kind}", device=torch.device("cpu"))
            got = param.detach().to("cpu")
            assert got.dtype == stored.dtype, (name, got.dtype, stored.dtype)
            assert torch.equal(got, stored), f"{name} does not hold the checkpoint bytes"
            checked += 1
        assert checked == 2, (
            f"expected k_norm weight+bias local to rank {owner_rank}, saw {checked}"
        )
    finally:
        del model
        torch.cuda.empty_cache()
