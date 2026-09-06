# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash (``glm5_next``) text-path model discovery and weight mapping.

The checkpoint is published as ``Glm5NextForConditionalGeneration``: a vision
tower plus a text decoder, with an extra MTP layer appended to the decoder
stack. This module brings up the **text decoder only**, so it has to answer
three questions before a single weight can be placed:

1. *Which config?* The runtime is handed the top-level multimodal config; the
   decoder contract lives in ``config.text_config``.
2. *Which module runs at layer i?* The config carries two independent literal
   45-entry lists -- ``layer_types`` (attention) and ``mlp_layer_types`` (feed
   forward) -- plus a third redundant encoding in
   ``linear_attn_config.kda_layers`` / ``full_attn_layers``. They agree on this
   checkpoint, and that agreement is asserted rather than assumed: dispatch is
   driven by the literal lists, never derived from ``first_k_dense_replace`` or
   from an attention cadence.
3. *Where does each of the 76108 checkpoint tensors go?* Every key is resolved
   to exactly one destination and one disposition -- see
   :func:`audit_glm5_next_checkpoint`.

Nothing here fabricates a value. A key that cannot be placed is an error, not a
silent drop; the only tensors allowed to go unplaced are the MTP and vision
namespaces, which are explicitly out of scope for this bring-up and are
allowlisted by exact namespace rather than by pattern.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl
from torch import nn
from transformers import PretrainedConfig

from ...logger import logger
from ...models.modeling_utils import QuantConfig
from ...quantization.mode import QuantAlgo
from ..attention_backend.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    AttentionMetadata,
)
from ..attention_backend.sparse.glm_kpool import INDEX_SENTINEL, GlmKpoolSparseParams
from ..attention_backend.sparse.params import SparseBackendForwardArgs
from ..attention_backend.utils import create_attention
from ..model_config import ModelConfig
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.rms_norm import RMSNorm
from .modeling_utils import DecoderModel, DecoderModelForCausalLM, register_auto_model

# ---------------------------------------------------------------------------
# Literal schedule vocabulary
# ---------------------------------------------------------------------------

LINEAR_ATTENTION = "linear_attention"
SPARSE_ATTENTION = "deepseek_sparse_attention"
DENSE_MLP = "dense"
SPARSE_MLP = "sparse"

#: Checkpoint namespaces this bring-up deliberately does not load. These are
#: matched as exact dotted-component prefixes, never as substrings or globs: a
#: pattern like ``*visual*`` would also swallow a decoder weight that merely
#: contained the word, and the whole point of the audit is that nothing is
#: dropped by accident.
_VISION_PREFIX = "model.visual."
_LANGUAGE_PREFIX = "model.language_model."


def get_glm5_next_text_config(config: PretrainedConfig) -> PretrainedConfig:
    """Return the text-decoder config, accepting either nesting level.

    The runtime resolves the model from the top-level ``Glm5NextConfig``, but
    every decoder contract (schedules, ranks, MoE, HC) lives on
    ``text_config``. Callers may already hold the inner config, so this is
    idempotent.
    """
    return getattr(config, "text_config", config)


@dataclass(frozen=True)
class Glm5NextSchedule:
    """The two literal per-layer dispatch lists, validated against each other."""

    attention: Tuple[str, ...]
    mlp: Tuple[str, ...]

    @property
    def num_layers(self) -> int:
        return len(self.attention)

    def attention_indices(self, kind: str) -> Tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.attention) if t == kind)

    def mlp_indices(self, kind: str) -> Tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.mlp) if t == kind)


def resolve_glm5_next_schedule(config: PretrainedConfig) -> Glm5NextSchedule:
    """Read and cross-validate the literal dispatch lists.

    Raises on any disagreement between the three redundant encodings. A model
    whose attention schedule is inferred from a cadence, or whose MLP schedule
    is inferred from ``first_k_dense_replace``, would silently place the wrong
    module (and therefore the wrong cache descriptor) at some layer; the config
    states both lists explicitly, so there is no reason to guess.
    """
    text = get_glm5_next_text_config(config)
    num_layers = int(text.num_hidden_layers)

    attention = tuple(text.layer_types)
    mlp = tuple(text.mlp_layer_types)

    for name, values, allowed in (
        ("layer_types", attention, {LINEAR_ATTENTION, SPARSE_ATTENTION}),
        ("mlp_layer_types", mlp, {DENSE_MLP, SPARSE_MLP}),
    ):
        if len(values) != num_layers:
            raise ValueError(
                f"glm5_next {name} has {len(values)} entries but num_hidden_layers={num_layers}"
            )
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"glm5_next {name} contains unsupported entries {unknown}")

    schedule = Glm5NextSchedule(attention=attention, mlp=mlp)

    # Third, redundant encoding of the attention schedule. It is not used for
    # dispatch, but a disagreement means the checkpoint is not the variant this
    # bring-up was validated against.
    linear_attn_config = getattr(text, "linear_attn_config", None) or {}
    kda_layers = linear_attn_config.get("kda_layers")
    full_attn_layers = linear_attn_config.get("full_attn_layers")
    if kda_layers is not None:
        if tuple(kda_layers) != schedule.attention_indices(LINEAR_ATTENTION):
            raise ValueError("glm5_next linear_attn_config.kda_layers disagrees with layer_types")
    if full_attn_layers is not None:
        if tuple(full_attn_layers) != schedule.attention_indices(SPARSE_ATTENTION):
            raise ValueError(
                "glm5_next linear_attn_config.full_attn_layers disagrees with layer_types"
            )

    # first_k_dense_replace is asserted against the literal list, not used to
    # build it.
    first_k_dense = getattr(text, "first_k_dense_replace", None)
    if first_k_dense is not None:
        expected_dense = tuple(range(int(first_k_dense)))
        if schedule.mlp_indices(DENSE_MLP) != expected_dense:
            raise ValueError(
                f"glm5_next mlp_layer_types dense entries "
                f"{schedule.mlp_indices(DENSE_MLP)} disagree with "
                f"first_k_dense_replace={first_k_dense}"
            )

    return schedule


# ---------------------------------------------------------------------------
# Checkpoint key mapping
# ---------------------------------------------------------------------------

#: The KDA short convolution is published as three separate depthwise
#: convolutions -- one per projection -- but is a single grouped ``conv1d`` over
#: the concatenated ``[q, k, v]`` channel axis at runtime. The concatenation
#: order is load-bearing and matches the order the projections are concatenated
#: in the source forward.
KDA_CONV_SOURCES = ("q_conv1d", "k_conv1d", "v_conv1d")
KDA_CONV_DEST = "conv1d"

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")
_CONV_RE = re.compile(r"^(model\.layers\.\d+\.self_attn\.)(q|k|v)_conv1d\.(weight|bias)$")
#: The two hyper-connection sites are published as flat per-layer tensors
#: (``hc_attn_fn``) while the runtime holds each site as one ``mHC`` submodule
#: whose parameters are ``fn`` / ``base`` / ``scale``. Only the separator moves;
#: no tensor is reshaped, split, or fused.
_HC_RE = re.compile(r"^(model\.layers\.\d+\.)hc_(attn|ffn)_(fn|base|scale)$")


class Disposition:
    """How a checkpoint tensor reaches (or does not reach) the runtime."""

    #: Placed on a destination parameter unchanged.
    LOADED = "loaded"
    #: Placed after a shape/dtype/layout transformation (conv fusion, expert
    #: stacking, block-FP8 dequantization, or a companion scale tensor).
    TRANSFORMED = "transformed"
    #: Deliberately not loaded, under an exact allowlisted namespace.
    IGNORED = "ignored"


@dataclass
class Glm5NextWeightAudit:
    """Exhaustive per-key accounting for a GLM-5.3-Flash checkpoint."""

    #: destination module/parameter name -> source key
    destinations: Dict[str, str] = field(default_factory=dict)
    #: source key -> disposition
    disposition: Dict[str, str] = field(default_factory=dict)
    #: source key -> why it was transformed / ignored
    reason: Dict[str, str] = field(default_factory=dict)
    #: keys that could not be placed at all -- always a hard error
    unresolved: List[str] = field(default_factory=list)

    def counts(self) -> Dict[str, int]:
        return dict(Counter(self.disposition.values()))

    def keys_with(self, disposition: str) -> List[str]:
        return sorted(k for k, d in self.disposition.items() if d == disposition)

    def reasons(self) -> Dict[str, int]:
        return dict(Counter(self.reason.values()))


def _ignored_reason(key: str, mtp_prefixes: Sequence[str]) -> Optional[str]:
    if key.startswith(_VISION_PREFIX):
        return "vision tower is out of scope for the text bring-up"
    for prefix in mtp_prefixes:
        if key.startswith(prefix):
            return "MTP / next-n prediction layer is out of scope"
    return None


def remap_glm5_next_key(key: str) -> Optional[str]:
    """Map one text-decoder checkpoint key to its runtime destination.

    Returns ``None`` for keys outside the text decoder (the caller decides
    whether that is an allowlisted namespace or an error).
    """
    if key == "lm_head.weight":
        return key
    if not key.startswith(_LANGUAGE_PREFIX):
        return None
    # model.language_model.<rest> -> model.<rest>: the runtime decoder is not
    # nested inside a multimodal wrapper.
    dest = "model." + key[len(_LANGUAGE_PREFIX) :]

    conv = _CONV_RE.match(dest)
    if conv is not None:
        prefix, _which, suffix = conv.groups()
        return f"{prefix}{KDA_CONV_DEST}.{suffix}"
    hc = _HC_RE.match(dest)
    if hc is not None:
        prefix, site, param = hc.groups()
        return f"{prefix}hc_{site}.{param}"
    return dest


def audit_glm5_next_checkpoint(
    keys: Iterable[str],
    config: PretrainedConfig,
) -> Glm5NextWeightAudit:
    """Resolve every checkpoint key to exactly one destination and disposition.

    This is the Goal-1.2 contract in executable form. It is deliberately
    analytic -- it needs only the safetensors index and the config, not 328 GB
    of materialized weights -- so it can gate every later loading change
    cheaply.
    """
    text = get_glm5_next_text_config(config)
    num_layers = int(text.num_hidden_layers)
    num_nextn = int(getattr(text, "num_nextn_predict_layers", 0) or 0)
    # MTP layers are appended immediately after the decoder stack.
    mtp_prefixes = tuple(f"{_LANGUAGE_PREFIX}layers.{num_layers + i}." for i in range(num_nextn))

    audit = Glm5NextWeightAudit()
    conv_sources: Dict[str, List[str]] = {}

    for key in keys:
        ignored = _ignored_reason(key, mtp_prefixes)
        if ignored is not None:
            audit.disposition[key] = Disposition.IGNORED
            audit.reason[key] = ignored
            continue

        dest = remap_glm5_next_key(key)
        if dest is None:
            audit.unresolved.append(key)
            continue

        if dest.endswith(f".{KDA_CONV_DEST}.weight") or dest.endswith(f".{KDA_CONV_DEST}.bias"):
            conv_sources.setdefault(dest, []).append(key)
            audit.disposition[key] = Disposition.TRANSFORMED
            audit.reason[key] = (
                "KDA short convolution: q/k/v depthwise weights concatenate into one grouped conv1d"
            )
            audit.destinations[dest] = dest
            continue

        if dest.endswith(".weight_scale_inv"):
            audit.disposition[key] = Disposition.TRANSFORMED
            audit.reason[key] = "block-FP8 128x128 weight scale"
            audit.destinations[dest] = key
            continue

        if ".mlp.experts." in dest:
            audit.disposition[key] = Disposition.TRANSFORMED
            audit.reason[key] = "routed expert stacked into the fused MoE layout"
            audit.destinations[dest] = key
            continue

        audit.disposition[key] = Disposition.LOADED
        audit.destinations[dest] = key

    # Every fused conv destination must have received all of its sources; a
    # partial fusion would silently zero a third of the convolution.
    for dest, sources in conv_sources.items():
        if len(sources) != len(KDA_CONV_SOURCES):
            raise ValueError(
                f"glm5_next {dest} expected {len(KDA_CONV_SOURCES)} source "
                f"tensors {KDA_CONV_SOURCES}, found {sorted(sources)}"
            )

    return audit


# ---------------------------------------------------------------------------
# Quantization exclusions
# ---------------------------------------------------------------------------

#: Parameters that must never be demoted below FP32, independent of the model
#: dtype. ``e_score_correction_bias`` is the noaux_tc router bias: its 288
#: entries sit in a ~0.5-wide band around magnitude ~10 while the sigmoid
#: scores it corrects are O(1e-2), so expert ranking turns on inter-expert gaps
#: of 4e-5 - 6e-4. bf16 resolution at that magnitude is 2e-2 - 6e-2, i.e. three
#: orders of magnitude too coarse; it collapses 234-288 distinct values to 6-11
#: and silently changes the top-8 selection on ~60% of real tokens while every
#: aggregate check (weights sum to routed_scaling_factor, finite outputs,
#: cosine vs the real layer) still passes.
FP32_PARAMETERS = ("e_score_correction_bias",)


def _is_text_pattern(pattern: str) -> bool:
    """False for patterns that can only ever match the vision tower."""
    head = pattern.split(".", 1)[0]
    return head != "visual"


def narrow_glm5_next_exclusions(patterns: Sequence[str]) -> List[str]:
    """Drop vision-only entries from the checkpoint's exclusion list.

    The published list covers the whole multimodal checkpoint. Handing the
    vision entries to the text model costs match time and can only ever produce
    a miss, so they are removed -- but by an exact head-component test, not by a
    substring search that could also delete a decoder entry.
    """
    return [p for p in patterns if _is_text_pattern(p)]


def glm5_next_quant_exclusions(config: PretrainedConfig) -> List[str]:
    """The exclusion patterns that apply to the text decoder."""
    quant_config = getattr(config, "quantization_config", None) or {}
    patterns = quant_config.get("modules_to_not_convert") or []
    return narrow_glm5_next_exclusions(patterns)


def build_glm5_next_quant_config(config: PretrainedConfig) -> QuantConfig:
    """A ``QuantConfig`` whose ``exclude_modules`` match runtime module names.

    The published patterns are written against the *text-config* namespace
    (``model.layers.N....``) while the checkpoint keys carry the multimodal
    wrapper (``model.language_model.layers.N....``). Because
    :func:`remap_glm5_next_key` strips that wrapper, the runtime module names
    line up with the patterns exactly and TensorRT-LLM's own
    ``QuantConfig.is_module_excluded_from_quantization`` resolves them without
    a model-specific matcher.
    """
    from tensorrt_llm.quantization.mode import QuantAlgo

    quant_config = getattr(config, "quantization_config", None) or {}
    block_size = tuple(quant_config.get("weight_block_size", (128, 128)))
    if block_size != (128, 128):
        raise ValueError(f"glm5_next expects 128x128 FP8 weight blocks, got {block_size}")

    return QuantConfig(
        quant_algo=QuantAlgo.FP8_BLOCK_SCALES,
        group_size=block_size[0],
        exclude_modules=glm5_next_quant_exclusions(config),
    )


def glm5_next_is_quantized(model_config: ModelConfig[PretrainedConfig]) -> bool:
    """Whether construction uses the checkpoint's block-FP8 form.

    The runtime constructs models through ``AutoModelForCausalLM.from_config``,
    which calls ``cls(model_config)`` with no further arguments -- so the
    quantization decision must live on the ``ModelConfig`` itself, exactly
    where ``ModelConfig.from_pretrained`` puts it when it reads the
    checkpoint's ``quantization_config`` (``weight_block_size=[128,128]`` maps
    to ``FP8_BLOCK_SCALES``). A constructor flag that defaulted to bf16 would
    make the runtime path build a model the loader must reject.

    This checkpoint is published in exactly one quantized form; any other
    non-None algorithm on the config is a configuration error, not a request
    for a different build.
    """
    quant = getattr(model_config, "quant_config", None)
    if quant is None or quant.quant_algo is None:
        return False
    if quant.quant_algo != QuantAlgo.FP8_BLOCK_SCALES:
        raise ValueError(
            "glm5_next supports only the published FP8_BLOCK_SCALES checkpoint "
            f"form or unquantized bf16 modules; got quant_algo={quant.quant_algo}"
        )
    return True


_PARAM_SUFFIXES = (".weight_scale_inv", ".weight", ".bias")


def _strip_param_suffix(name: str) -> str:
    for suffix in _PARAM_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def source_module_name(key: str) -> Optional[str]:
    """The checkpoint's own module name for ``key``, in the pattern namespace.

    The published exclusion patterns are written against the text-config
    namespace (``model.layers.N....``), so the multimodal wrapper is stripped
    but *no* fusion is applied: this is deliberately the pre-transformation
    spelling.
    """
    if key == "lm_head.weight":
        return "lm_head"
    if not key.startswith(_LANGUAGE_PREFIX):
        return None
    return _strip_param_suffix("model." + key[len(_LANGUAGE_PREFIX) :])


def destination_module_name(key: str) -> Optional[str]:
    """The runtime module name ``key`` is loaded into, after any fusion."""
    dest = remap_glm5_next_key(key)
    return None if dest is None else _strip_param_suffix(dest)


def resolve_glm5_next_exclusions(
    keys: Iterable[str],
    quant_config: QuantConfig,
) -> Dict[str, bool]:
    """Excluded-from-quantization verdict per *destination* module.

    Exclusions must be resolved against the **source** spelling and only then
    carried across the source-to-destination transformation. Resolving them
    against the destination silently mis-classifies every fused module: the
    KDA short convolution is published as ``q_conv1d`` / ``k_conv1d`` /
    ``v_conv1d`` -- all three excluded -- but fuses into a single ``conv1d``
    that no published pattern names, so a destination-side lookup reports "not
    excluded" and would quantize a BF16 tensor in all 34 linear-attention
    layers.

    When several sources fuse into one destination they must agree; a
    disagreement means the fusion itself is invalid (half the channels would
    need a scale the other half does not have) and is raised rather than
    resolved by a majority or a default.
    """
    verdicts: Dict[str, bool] = {}
    evidence: Dict[str, Dict[bool, List[str]]] = {}
    cache: Dict[str, bool] = {}

    for key in keys:
        source = source_module_name(key)
        dest = destination_module_name(key)
        if source is None or dest is None:
            continue
        if source not in cache:
            cache[source] = quant_config.is_module_excluded_from_quantization(source)
        verdict = cache[source]
        evidence.setdefault(dest, {}).setdefault(verdict, []).append(source)
        verdicts[dest] = verdict

    for dest, by_verdict in evidence.items():
        if len(by_verdict) > 1:
            excluded = sorted(set(by_verdict.get(True, [])))
            quantized = sorted(set(by_verdict.get(False, [])))
            raise ValueError(
                f"glm5_next {dest}: fused sources disagree on quantization -- "
                f"excluded={excluded} quantized={quantized}. Fusing tensors "
                f"with different quantization is not representable."
            )
    return verdicts


#: The checkpoint's ``weight_block_size``. Quantized matrices carry one FP32
#: scale per 128x128 weight block, with partial *edge* blocks wherever a
#: dimension is not a multiple of 128 -- ``q_b_proj``'s 1536-wide input divides
#: exactly, but ``kv_a_proj_with_mqa``'s 576 rows need a 5th block covering only
#: 64 of them, so scale shapes are always computed with a ceiling.
FP8_BLOCK: Tuple[int, int] = (128, 128)


def glm5_next_fp8_scale_shape(out_features: int, in_features: int) -> Tuple[int, int]:
    """Block-scale shape for an ``[out_features, in_features]`` FP8 matrix."""
    return (
        (out_features + FP8_BLOCK[0] - 1) // FP8_BLOCK[0],
        (in_features + FP8_BLOCK[1] - 1) // FP8_BLOCK[1],
    )


#: e4m3's largest finite magnitude, and the divisor the checkpoint's declared
#: ``activation_scheme='dynamic'`` scales activations by.
FP8_E4M3_MAX = 448.0


def glm5_next_dynamic_act_quant_1x128(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[M, K]`` activations to e4m3 in 1x128 tiles, as declared.

    ``config.json`` publishes ``activation_scheme='dynamic'`` with
    ``weight_block_size=[128, 128]``. That scheme has one arithmetic definition,
    shared by the source and by every other stack that reads these checkpoints:
    per row and per 128-wide K tile, ``scale = max(|x|) / 448`` in float32 and
    ``payload = (x / max(scale, 1e-12))`` rounded to nearest e4m3. The stored
    scale is the *unclamped* one; the clamp exists only to keep an all-zero tile
    from dividing by zero.

    This is written out rather than delegated to
    ``torch.ops.trtllm.fp8_quantize_1x128`` because that op implements a
    different scale: measured on this checkpoint its effective divisor ranges
    over 442.0-453.8 against a median of 448.03, a per-tile perturbation of
    +/-1.3%. Both are defensible quantizations -- their distances from exact
    arithmetic agree to four significant figures -- but only one of them is the
    convention the checkpoint was quantized under, and the difference is not
    academic: it is the entire measured divergence between this path and the
    source. See
    ``test_the_block_fp8_matmul_reproduces_the_source_kernel_bitwise``.

    Returns ``(payload [M, K] float8_e4m3fn, scale [K // 128, M] float32)`` --
    the transposed scale layout ``cute_dsl_fp8_gemm_blackwell`` consumes.
    """
    num_rows, num_cols = x.shape
    if num_cols % 128:
        raise ValueError(
            f"glm5_next block-FP8 needs a K that is a multiple of 128, got {num_cols}; "
            "every quantized projection on this checkpoint has one (4096, 2048, "
            "1536, 512, 8192, 12288, 16384)"
        )
    tiles = x.reshape(num_rows, num_cols // 128, 128).float()
    scale = tiles.abs().amax(dim=-1) / FP8_E4M3_MAX
    payload = (tiles / scale.clamp(min=1e-12).unsqueeze(-1)).to(torch.float8_e4m3fn)
    return payload.reshape(num_rows, num_cols), scale.t().contiguous()


def glm5_next_block_fp8_matmul(
    x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor
) -> torch.Tensor:
    """``x @ weight.T`` for a block-FP8 weight, with dynamic activation scales.

    This is the arithmetic the source actually runs, and running it rather than
    an ordinary bf16 matmul over dequantized weights is load-bearing. The
    checkpoint is ``activation_scheme='dynamic'``, so activations are quantized
    to e4m3 in 1x128 tiles *at call time*, the product accumulates per block,
    and each block is rescaled by its stored ``weight_scale_inv``.

    Measured against the source's own kernel on real checkpoint weights, this
    path reproduces its bf16 output **bitwise on 99.99-100% of elements** with a
    residual relative RMS of 1e-5 or below. Quantizing the activations with
    ``fp8_quantize_1x128`` instead leaves 4.1e-3 -- a hundred times further, and
    only ~50% of output elements bitwise equal, which is chance for a bf16 last
    bit. Dequantizing the weight to bf16 and running an ordinary matmul is
    further still. The GEMM is the same production CuTe kernel in every case;
    only the activation quantization differs (see
    :func:`glm5_next_dynamic_act_quant_1x128`).

    ``weight`` is ``[N, K]`` float8_e4m3fn and ``weight_scale`` is
    ``[ceil(N/128), ceil(K/128)]`` float32. A 2-D slice of a stacked expert
    tensor is contiguous, so routed experts use this directly.
    """
    flat = x.reshape(-1, x.shape[-1])
    if flat.dtype != torch.bfloat16:
        flat = flat.to(torch.bfloat16)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    act, act_scale = glm5_next_dynamic_act_quant_1x128(flat)
    out = torch.ops.trtllm.cute_dsl_fp8_gemm_blackwell(act, weight, act_scale, weight_scale)
    return out.view(*x.shape[:-1], out.shape[-1])


def glm5_next_quant_plan(
    keys: Iterable[str],
    quant_config: QuantConfig,
) -> Dict[str, bool]:
    """Per-destination-module verdict: ``True`` means *runs block-FP8*.

    This is the inverse reading of :func:`resolve_glm5_next_exclusions` and is
    deliberately derived from it rather than recomputed, so the runtime dtype of
    every module and the audited 1509-entry exclusion set can never disagree.
    """
    return {
        dest: not excluded
        for dest, excluded in resolve_glm5_next_exclusions(keys, quant_config).items()
    }


def audit_glm5_next_exclusion_patterns(
    patterns: Sequence[str],
    source_module_names: Sequence[str],
) -> Dict[str, Any]:
    """Report which exclusion patterns fired, so none is silently inert.

    A pattern that matches nothing is not automatically a defect -- the
    published list also carries fused spellings (``qkv_proj``,
    ``fused_qkvbfg_a_proj``) that this checkpoint does not use but a fused
    runtime layout would -- but every zero-match entry has to be visible rather
    than assumed benign. Patterns are evaluated against **source** module
    names, which is the namespace they were authored in.
    """
    unique_names = sorted(set(source_module_names))
    matched: Dict[str, int] = {p: 0 for p in patterns}
    single = {p: QuantConfig(exclude_modules=[p]) for p in patterns}
    for name in unique_names:
        for pattern, cfg in single.items():
            if cfg.is_module_excluded_from_quantization(name):
                matched[pattern] += 1
    zero = sorted(p for p, n in matched.items() if n == 0)
    return {
        "num_patterns": len(patterns),
        "num_module_names": len(unique_names),
        "num_matched": len(patterns) - len(zero),
        "num_zero_match": len(zero),
        "zero_match_patterns": zero,
        "match_counts": matched,
    }


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------


def _normalize_glm5_next_top_config(config: PretrainedConfig) -> None:
    """Give the composite multimodal config the fields the runtime reads.

    The checkpoint's top-level ``Glm5NextConfig`` carries no
    ``num_hidden_layers`` or ``torch_dtype`` -- both live on ``text_config`` --
    but ``DecoderModel``/``DecoderModelForCausalLM`` and executor capacity
    planning read them from the config they are handed. Copy them up once,
    from the text config, rather than teaching every runtime consumer about
    the wrapper. Values already present are left alone.
    """
    text = get_glm5_next_text_config(config)
    if getattr(config, "num_hidden_layers", None) is None:
        config.num_hidden_layers = int(text.num_hidden_layers)
    if getattr(config, "torch_dtype", None) is None:
        config.torch_dtype = getattr(text, "torch_dtype", None) or torch.bfloat16


@dataclass
class Glm5NextRuntimeContext:
    """Per-forward schedule arguments derived once from AttentionMetadata.

    The two attention families need different arguments: the KDA layers take
    their recurrent/conv pools and slot ids here, while the sparse layers take
    only *schedule* values (request boundaries, positions) plus the prepared
    ``metadata`` itself -- their attention backend derives every cache pool,
    block table, and visible length from that metadata. This object is built
    once per model forward; requests are packed context-first, matching the
    executor's batch layout.
    """

    manager: Any
    num_contexts: int
    num_ctx_tokens: int
    num_generations: int
    ctx_cu_seqlens: List[int]
    cached_lens: List[int]
    state_indices: torch.Tensor
    #: Per-request visible lengths (cached + this step's tokens) as a device
    #: tensor. The decode path consumes ONLY device values so that captured
    #: CUDA graphs replay against prepare()-refreshed buffers rather than
    #: Python ints baked in at capture time.
    kv_lens: torch.Tensor
    #: The engine's prepared, typed attention metadata this context was
    #: derived from; the sparse layers' attention backend consumes it as the
    #: single source of cache state.
    metadata: AttentionMetadata

    def linear_kwargs(self, layer_idx: int, phase: str) -> Dict[str, Any]:
        kwargs = {
            "conv_pool": self.manager.get_conv_states(layer_idx),
            "ssm_pool": self.manager.get_ssm_states(layer_idx),
        }
        if phase == "prefill":
            kwargs.update(
                slot_ids=self.state_indices[: self.num_contexts],
                cu_seqlens=self.ctx_cu_seqlens,
                cached_lens=self.cached_lens[: self.num_contexts],
            )
        else:
            kwargs.update(slot_ids=self.state_indices[self.num_contexts :])
        return kwargs

    def sparse_kwargs(self, layer_idx: int, phase: str) -> Dict[str, Any]:
        # Schedule only: the backend (keyed by its own layer_idx) derives the
        # slot-indexed latent/indexer views, block tables, and lengths from
        # the prepared metadata itself.
        del layer_idx
        kwargs: Dict[str, Any] = {"metadata": self.metadata}
        if phase == "prefill":
            kwargs.update(
                cached_lens=self.cached_lens[: self.num_contexts],
                cu_seqlens=self.ctx_cu_seqlens,
            )
        else:
            kwargs.update(kv_lens=self.kv_lens[self.num_contexts :])
        return kwargs


def build_glm5_next_runtime_context(
    attn_metadata: AttentionMetadata,
) -> Glm5NextRuntimeContext:
    """Derive the per-forward cache arguments from prepared metadata.

    Requires ``attn_metadata.prepare()`` to have run: that is what attaches
    ``mamba_metadata`` (the manager is a ``BaseMambaCacheManager``) and fills
    its batch-ordered ``state_indices``. ``cached_lens`` follows the runtime's
    own convention -- tokens already in the cache, excluding the ones in this
    step -- which is exactly what ``forward_prefill``/``forward_decode`` seed
    and position from.
    """
    manager = attn_metadata.kv_cache_manager
    if manager is None:
        raise ValueError("glm5_next requires a kv cache manager; got None")
    mamba_metadata = attn_metadata.mamba_metadata
    if mamba_metadata is None or mamba_metadata is False:
        raise ValueError(
            "glm5_next requires mamba_metadata; call attn_metadata.prepare() "
            "with the Glm5NextCacheManager attached"
        )
    batch = int(attn_metadata.seq_lens.shape[0])
    num_contexts = int(attn_metadata.num_contexts)

    if getattr(mamba_metadata, "glm_block_tables", None) is not None:
        # Persistent path: every tensor below is a prepare()-refreshed buffer
        # slice, so this function does no allocation, no H2D, and no host
        # sync -- it is safe to run inside CUDA graph capture, and replays
        # read the refreshed values at the same addresses.
        return Glm5NextRuntimeContext(
            manager=manager,
            num_contexts=num_contexts,
            num_ctx_tokens=int(attn_metadata.num_ctx_tokens),
            num_generations=batch - num_contexts,
            ctx_cu_seqlens=mamba_metadata.glm_ctx_cu_seqlens,
            cached_lens=mamba_metadata.glm_cached_lens_host,
            state_indices=mamba_metadata.state_indices[:batch],
            kv_lens=mamba_metadata.glm_kv_lens[:batch],
            metadata=attn_metadata,
        )

    # Legacy eager construction, kept for harnesses whose fake managers do
    # not attach the GLM metadata buffers. It allocates and copies, so it
    # must never run inside a captured region. (The sparse backend applies
    # the same rule to its own metadata-derived block tables.)
    if getattr(attn_metadata, "is_cuda_graph", False):
        raise RuntimeError(
            "glm5_next CUDA-graph execution requires the Glm5NextCacheManager's "
            "Glm5NextMamba2Metadata (persistent prepare()-refreshed buffers); "
            "the attached mamba_metadata has no glm_block_tables"
        )
    lens = attn_metadata.seq_lens.tolist()
    kv_params = attn_metadata.kv_cache_params
    if kv_params is None or kv_params.num_cached_tokens_per_seq is None:
        raise ValueError("glm5_next requires kv_cache_params.num_cached_tokens_per_seq")
    cached_lens = [int(n) for n in kv_params.num_cached_tokens_per_seq[:batch]]

    ctx_cu = [0]
    for length in lens[:num_contexts]:
        ctx_cu.append(ctx_cu[-1] + int(length))

    device = torch.device("cuda", torch.cuda.current_device())
    kv_lens = torch.as_tensor(
        [c + n for c, n in zip(cached_lens, lens)], dtype=torch.long, device=device
    )

    return Glm5NextRuntimeContext(
        manager=manager,
        num_contexts=num_contexts,
        num_ctx_tokens=int(attn_metadata.num_ctx_tokens),
        num_generations=batch - num_contexts,
        ctx_cu_seqlens=ctx_cu,
        cached_lens=cached_lens,
        state_indices=mamba_metadata.state_indices[:batch],
        kv_lens=kv_lens,
        metadata=attn_metadata,
    )


@register_auto_model("Glm5NextForConditionalGeneration")
class Glm5NextForCausalLM(DecoderModelForCausalLM):
    """Text-path entry point for GLM-5.3-Flash.

    Auto-discovery resolves the published architecture
    ``Glm5NextForConditionalGeneration`` to this class, so the text bring-up
    needs no public override or checkpoint edit. The vision tower is out of
    scope and its weights are allowlisted rather than loaded (see
    :func:`audit_glm5_next_checkpoint`).

    The class inherits the ``DecoderModelForCausalLM`` lifecycle wholesale --
    ``PostInitCaller`` construction, ``LMHead``/``LogitsProcessor``, the
    runtime ``forward(attn_metadata, ...)``, and pipeline-parallel prologue/
    epilogue handling -- and keeps only what is GLM-specific: the config
    narrowing, the literal dispatch schedule, the audited block-FP8 quant
    plan, and the checkpoint's exact-placement loader.

    The base class is deliberately unsubscripted: the generic form
    ``DecoderModelForCausalLM[Glm5NextModel, ...]`` would evaluate
    ``Glm5NextModel`` at class-definition time, before it is defined below.

    Instantiating this materializes all 45 layers including 288 routed
    experts each (~328 GB); whole-model use constructs on ``meta`` and
    materializes per owner via :meth:`load_weights`.
    """

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        text_config = get_glm5_next_text_config(model_config.pretrained_config)
        # tie_word_embeddings is false on this checkpoint, so lm_head is a real
        # weight rather than a view of the embedding; it is asserted, not assumed.
        if bool(getattr(text_config, "tie_word_embeddings", False)):
            raise ValueError(
                "glm5_next was validated with untied output embeddings; a tied "
                "checkpoint would need lm_head to alias embed_tokens"
            )
        _normalize_glm5_next_top_config(model_config.pretrained_config)
        super().__init__(
            Glm5NextModel(model_config),
            config=model_config,
            hidden_size=int(text_config.hidden_size),
            vocab_size=int(text_config.vocab_size),
        )
        # ``config`` is a base-class property (the top-level composite config);
        # the narrowed decoder contract lives here.
        self.text_config = text_config
        self.schedule = resolve_glm5_next_schedule(model_config.pretrained_config)
        self.quant_config = build_glm5_next_quant_config(model_config.pretrained_config)
        # Derived from the ModelConfig, never a constructor flag: the runtime's
        # AutoModelForCausalLM.from_config calls cls(model_config) and nothing
        # else, so this is the only place the decision can live.
        self.quantized = glm5_next_is_quantized(model_config)
        # One provenance line per rank: engine-scale runs (LLM API / serving)
        # spawn MPI workers whose model objects the driver cannot introspect,
        # so the resolved production stack is published through the worker log.
        attn_backends = sorted(
            {
                type(layer.self_attn.attn_backend).__name__
                for layer in self.model.layers
                if isinstance(layer.self_attn, Glm5NextSparseAttention)
            }
        )
        moe_backends = sorted(
            {
                layer.mlp.moe_backend_name or "diagnostic-torch"
                for layer in self.model.layers
                if isinstance(layer.mlp, Glm5NextMoE)
            }
        )
        logger.info(
            f"glm5_next runtime stack: sparse_attention={attn_backends}, "
            f"moe_backend={moe_backends}, quantized={self.quantized}, "
            f"kv_cache_manager=V2 (Glm5NextCacheManager)"
        )

    @property
    def num_hidden_layers(self) -> int:
        return self.schedule.num_layers

    # -- runtime (LLM API / PyExecutor) interface -------------------------
    #
    # These are the first pieces of the runtime binding (Goal 2.4 / Decision E):
    # the hooks the executor calls before the model is ever run. They are inert
    # under the Stage-1 diagnostic driver -- which constructs this class directly
    # -- and become live once the metadata-driven ``forward`` and the cache
    # selection plumbing land. See reports/goal1.5-runtime-binding-plan.md for the
    # full increment sequence. Kept here (not deferred) so the interface a reader
    # sees matches what the executor requires, and so it can be unit-tested now.

    def infer_max_seq_len(self) -> int:
        """Max sequence length the runtime sizes KV/mamba caches for.

        The executor calls this during capacity planning. GLM-5.3-Flash declares
        ``max_position_embeddings=1048576`` and is fully NoPE (no rope-factor
        scaling), so the value is the text config's directly.
        """
        return int(self.text_config.max_position_embeddings)

    @classmethod
    def get_preferred_kv_cache_manager_version(cls, pretrained_config=None) -> str:
        """Opt this model into ``KVCacheManagerV2``.

        The hybrid latent-KV + pool-indexer + recurrent/conv state is owned by a
        single ``Glm5NextCacheManager`` (a ``MambaHybridCacheManagerV2`` subclass,
        :func:`glm5_next_cache_manager_cls`); V1 cannot express it. This is the
        ``"auto"`` -> V2 resolution hook the runtime consults.
        """
        return "V2"

    def attention_type(self, layer_idx: int) -> str:
        """The literal attention module type for ``layer_idx``."""
        return self.schedule.attention[layer_idx]

    def mlp_type(self, layer_idx: int) -> str:
        """The literal feed-forward module type for ``layer_idx``."""
        return self.schedule.mlp[layer_idx]

    def audit_checkpoint(self, keys: Iterable[str]) -> Glm5NextWeightAudit:
        """Resolve every checkpoint key against this model's destinations."""
        return audit_glm5_next_checkpoint(keys, self.model_config.pretrained_config)

    # -- whole-model materialization --------------------------------------

    def apply_quant_plan(self, keys: Iterable[str]) -> Dict[str, str]:
        """Swap every projection to its Mapping-aware Linear, per the audited plan."""
        plan = glm5_next_quant_plan(keys, self.quant_config)
        placement = glm5_next_swap_quantized_projections(
            self, plan, mapping=self.model_config.mapping
        )
        self._quant_plan_applied = True
        return placement

    def load_weights(
        self,
        weights: Any,
        *,
        device_map: Optional[Dict[Any, Any]] = None,
        report: Optional[Glm5NextLoadReport] = None,
        **_ignored: Any,
    ) -> Glm5NextLoadReport:
        """Materialize and fill the whole text model from a raw checkpoint.

        The signature deliberately does **not** declare ``weight_mapper``:
        the runtime's ``ModelLoader._call_load_weights`` inspects the
        argument list and hands every model that names it the generic
        initialized HF mapper, and this loader places checkpoint keys by its
        own audit rather than by mapper rules. Not advertising the parameter
        is what keeps the exact ``ModelLoader`` call compatible; a mapper
        passed explicitly is rejected below rather than silently swallowed
        by ``**_ignored``.

        ``weights`` is any mapping from checkpoint key to the tensor **as
        stored** -- e4m3 payloads and their FP32 block scales are copied
        verbatim, never dequantized, because excluded modules are published in
        BF16 and quantized ones in e4m3 with a scale, with no overlap. That
        makes the load a 1:1 placement and keeps the resident model at the
        checkpoint's own 328 GB.

        ``device_map`` maps each owner -- a layer index, or ``"embed"``,
        ``"norm"``, ``"head"`` -- to a device. The model is expected to have
        been constructed on ``meta``: each owner is materialized directly onto
        its target device and filled immediately, so peak memory is one layer
        above the final footprint rather than a second full copy.

        Every destination is counted. A checkpoint tensor that finds no
        parameter, and a parameter that receives no tensor, are both errors:
        either one leaves a model that still runs and still looks plausible.
        """
        if _ignored.pop("weight_mapper", None) is not None:
            raise ValueError(
                "glm5_next uses its audited exact-placement loader; a "
                "checkpoint-format weight_mapper is not supported"
            )
        if not self.quantized:
            raise ValueError(
                "glm5_next whole-model loading requires the block-FP8 build "
                "(quantized=True). Dequantizing all 288 experts of 42 routed "
                "layers to bf16 would double the resident model to ~656 GB and "
                "move it four times further from the source's own arithmetic."
            )
        # The runtime's ModelLoader calls load_weights(weights) directly; the
        # diagnostic harness applies the plan itself first. Both end at the
        # same audited swap exactly once.
        if not getattr(self, "_quant_plan_applied", False):
            self.apply_quant_plan(list(weights.keys()))
        report = report or Glm5NextLoadReport()
        pretrained = self.model_config.pretrained_config
        num_layers = self.schedule.num_layers
        audit = audit_glm5_next_checkpoint(list(weights.keys()), pretrained)

        # Each owner is materialized and filled on its own, so peak memory is
        # one layer above the final footprint rather than a second full copy.
        targets: Dict[Any, Tuple[nn.Module, str]] = {
            **{i: (self.model.layers[i], f"model.layers.{i}.") for i in range(num_layers)},
            "embed": (self.model.embed_tokens, "model.embed_tokens."),
            "norm": (self.model.norm, "model.norm."),
            "head": (self.lm_head, "lm_head."),
        }
        # Owners pruned by pipeline parallelism (`__pp_init__` cleared their
        # parameters): their checkpoint keys belong to another rank. They are
        # counted, not silently dropped, so the per-rank report still accounts
        # for every key: loaded + transformed + ignored + skipped_remote must
        # cover the whole checkpoint on every rank.
        remote = {
            owner
            for owner, (module, _) in targets.items()
            if getattr(module, "_weights_removed", False)
        }

        by_owner: Dict[Any, List[Tuple[str, str]]] = {}
        for key, disposition in audit.disposition.items():
            if disposition == Disposition.IGNORED:
                report.ignored += 1
                continue
            dest = remap_glm5_next_key(key)
            owner = _destination_owner(dest, num_layers) if dest else None
            if owner is None:
                raise ValueError(f"glm5_next has no destination owner for {key!r} -> {dest!r}")
            if owner in remote:
                report.skipped_remote += 1
                continue
            # The loaded/transformed split is the audit's, not the loader's. The
            # two must agree by construction: a loader that re-derived it from
            # its own placement path would drift from the audited contract -- a
            # block scale placed by name is still a *transformed* tensor, since
            # it exists only because the weight beside it was quantized.
            if disposition == Disposition.TRANSFORMED:
                report.transformed += 1
            else:
                report.loaded += 1
            by_owner.setdefault(owner, []).append((key, dest, disposition))
        if audit.unresolved:
            raise ValueError(f"glm5_next cannot place {sorted(audit.unresolved)[:5]}")

        default_device = torch.device("cuda", torch.cuda.current_device())
        for owner, (module, prefix) in targets.items():
            if owner in remote:
                continue
            device = torch.device((device_map or {}).get(owner, default_device))
            module.to_empty(device=device)
            self._fill_module(
                module, owner, prefix, by_owner.get(owner, []), weights, device, report
            )
            report.devices[owner] = str(device)
        # Per-rank accounting provenance for engine-scale runs (the driver
        # cannot read another rank's Glm5NextLoadReport object; the log can).
        logger.info(
            f"glm5_next load report: loaded={report.loaded} "
            f"transformed={report.transformed} ignored={report.ignored} "
            f"skipped_remote={report.skipped_remote} "
            f"owners={len(report.devices)}"
        )
        return report

    def _fill_module(
        self,
        module: nn.Module,
        owner: Any,
        prefix: str,
        entries: Sequence[Tuple[str, str, str]],
        weights: Any,
        device: torch.device,
        report: Glm5NextLoadReport,
    ) -> None:
        """Place one materialized owner's tensors, fusing where the audit says to.

        Three destination families, three placement contracts:

        * A TensorRT-LLM ``Linear`` destination (every converted projection and
          the vocab-sharded ``LMHead``) receives its checkpoint tensors through
          the module's own ``load_weights``/``load_shard`` -- the tensors stay
          lazy safetensors slices so only this rank's contiguous weight and
          128x128 scale rows are ever materialized, and the destination's
          quant method owns the slicing.
        * Production routed experts go to the fused layer's loader, filtered to
          ``initial_local_expert_ids`` -- an expert owned by another EP rank is
          counted ``remote_experts`` and its bytes are never read.
        * Everything else (norms, HC, router, KDA gates/conv, embeddings) is an
          exact-shape replicated copy, as before.
        """
        from ..modules.linear import Linear as TrtllmLinear

        params = dict(module.named_parameters())
        params.update(dict(module.named_buffers()))
        named_modules = dict(module.named_modules())
        conv_parts: Dict[str, Dict[str, torch.Tensor]] = {}
        filled = set()
        # Production routed experts load through the fused layer's own
        # quant-method loader (which owns the [w3; w1] destination layout and
        # any kernel-side transforms), so their checkpoint tensors are
        # collected here and handed over in one call below.
        production_moe = getattr(getattr(module, "mlp", None), "experts", None) is not None
        local_expert_ids = (
            set(module.mlp.experts.initial_local_expert_ids) if production_moe else set()
        )
        moe_weights: Dict[str, torch.Tensor] = {}
        _MOE_PROJ_TO_W = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
        # {module_path: {label: lazy tensor}} plus the matching dest strings,
        # handed to each Linear's own loader after the placement loop.
        linear_groups: Dict[str, Dict[str, Any]] = {}
        linear_dests: Dict[str, Dict[str, str]] = {}

        def materialize(t: Any) -> torch.Tensor:
            # Lazy safetensors slice: HfWeightLoader streams glm5_next
            # checkpoints (328 GB would otherwise be materialized in host
            # RAM once per PP rank — the proven global-OOM kill). Indexing
            # materializes only this tensor's bytes from the mmapped file,
            # and the reference is dropped once the on-device copy lands.
            return t if torch.is_tensor(t) else t[:]

        for key, dest, disposition in sorted(entries):
            local = _apply_destination_alias(dest.removeprefix(prefix))
            lazy = weights[key]

            # The q/k/v depthwise filters fuse into one grouped conv1d, so this
            # has to be recognised on the checkpoint's own spelling: by the time
            # remap_glm5_next_key has run, all three already share one name.
            conv = _CONV_RE.match("model." + key[len(_LANGUAGE_PREFIX) :])
            if conv is not None:
                conv_parts.setdefault(conv.group(3), {})[conv.group(2)] = materialize(lazy)
                continue

            moe = _FUSED_MOE_RE.match(dest.removesuffix("_scale_inv").removesuffix(".weight"))
            if moe is not None:
                if production_moe:
                    expert_id = int(moe.group("expert"))
                    if expert_id not in local_expert_ids:
                        # Owned by another EP rank. The by_owner pass counted it
                        # under its audit disposition before expert locality was
                        # known; reclassify the same bucket back out without
                        # ever materializing the bytes.
                        if disposition == Disposition.TRANSFORMED:
                            report.transformed -= 1
                        else:
                            report.loaded -= 1
                        report.remote_experts += 1
                        continue
                    suffix = ".weight_scale_inv" if dest.endswith("_scale_inv") else ".weight"
                    w_name = _MOE_PROJ_TO_W[moe.group("proj")]
                    moe_weights[f"{expert_id}.{w_name}{suffix}"] = materialize(lazy)
                else:
                    self._place_expert(module, moe, dest, materialize(lazy), params)
                continue

            mod_path, _, leaf = local.rpartition(".")
            dest_mod = named_modules.get(mod_path)
            if isinstance(dest_mod, TrtllmLinear):
                # Keep the checkpoint's own label (load_weights_vanilla resolves
                # the DeepSeek-recipe 'weight_scale_inv' name itself). The big
                # weight stays a lazy slice so load_shard streams only this
                # rank's range; the 128x128 scale (a few KB) is materialized
                # here because the quant method inspects `.dim()` on it before
                # its own load_shard slicing.
                if dest.endswith("_scale_inv"):
                    label, value = "weight_scale_inv", materialize(lazy)
                else:
                    label, value = leaf, lazy
                linear_groups.setdefault(mod_path, {})[label] = value
                linear_dests.setdefault(mod_path, {})[label] = dest
                continue

            tensor = materialize(lazy)
            # The two KDA per-head parameters that are not Linear modules:
            # their checkpoint tensors are full-width, so at tp_size > 1 this
            # rank owns the same contiguous head (A_log) or head-channel
            # (dt_bias) slice its column-sharded projections own.
            if local in ("self_attn.A_log", "self_attn.dt_bias"):
                attn = named_modules.get("self_attn")
                if getattr(attn, "tp_size", 1) > 1:
                    full_shape = list(tensor.shape)
                    start, end = (
                        attn.kda_head_range()
                        if local.endswith("A_log")
                        else attn.kda_channel_range()
                    )
                    tensor = tensor[start:end]
                    report.tp_shards[f"{prefix}{local}"] = {
                        "mode": "column",
                        "tp_rank": int(attn.tp_rank),
                        "tp_size": int(attn.tp_size),
                        "range": [int(start), int(end)],
                        "full_shape": full_shape,
                        "local_shape": list(tensor.shape),
                        "reduce_output": False,
                        "quant": "float32",
                    }
            target = params.get(local)
            if target is None:
                raise KeyError(
                    f"glm5_next has no parameter for {key!r} (destination {dest!r}, "
                    f"looked up {local!r})"
                )
            if tuple(target.shape) != tuple(tensor.shape):
                raise ValueError(
                    f"glm5_next {dest!r}: checkpoint shape {tuple(tensor.shape)} does not "
                    f"match parameter shape {tuple(target.shape)}"
                )
            with torch.no_grad():
                target.copy_(tensor.to(device=device, dtype=target.dtype))
            filled.add(local)
            report.dtypes[dest] = str(target.dtype).removeprefix("torch.")

        for mod_path, group in sorted(linear_groups.items()):
            dest_mod = named_modules[mod_path]
            # Shards are recorded under the full destination path so every
            # layer's projection is its own accounting row, not one row per
            # suffix pattern shared by all 45 layers.
            self._load_linear_shard(
                dest_mod,
                f"{prefix}{mod_path}".rstrip("."),
                group,
                linear_dests[mod_path],
                report,
            )
            for param_name in ("weight", "weight_scale", "bias"):
                # mod_path is "" when the owner *is* the Linear (the LMHead).
                local_param = f"{mod_path}.{param_name}" if mod_path else param_name
                if local_param in params:
                    filled.add(local_param)

        for suffix, parts in conv_parts.items():
            missing = [n for n in KDA_CONV_SOURCES if n[0] not in parts]
            if missing:
                raise ValueError(f"glm5_next conv1d.{suffix} missing sources {missing}")
            target = params[f"self_attn.{KDA_CONV_DEST}.{suffix}"]
            pieces = [parts[n[0]] for n in KDA_CONV_SOURCES]
            attn = named_modules.get("self_attn")
            if getattr(attn, "tp_size", 1) > 1:
                # Each checkpoint filter is sliced by this rank's head channel
                # range *before* the [q | k | v] concatenation, so the local
                # grouped filter convolves exactly the channels the
                # column-sharded projections produce (the vLLM GLM5 KDA port
                # keeps the same per-source column ownership).
                start, end = attn.kda_channel_range()
                for i, source in enumerate(KDA_CONV_SOURCES):
                    full_shape = list(pieces[i].shape)
                    pieces[i] = pieces[i][start:end]
                    report.tp_shards[f"{prefix}self_attn.{source}"] = {
                        "mode": "column",
                        "tp_rank": int(attn.tp_rank),
                        "tp_size": int(attn.tp_size),
                        "range": [int(start), int(end)],
                        "full_shape": full_shape,
                        "local_shape": list(pieces[i].shape),
                        "reduce_output": False,
                        "quant": str(target.dtype).removeprefix("torch."),
                    }
            fused = torch.cat(pieces, dim=0)
            with torch.no_grad():
                target.copy_(fused.to(device=device, dtype=target.dtype).view_as(target))
            filled.add(f"self_attn.{KDA_CONV_DEST}.{suffix}")

        if moe_weights:
            expected = 6 * len(local_expert_ids)
            if len(moe_weights) != expected:
                raise ValueError(
                    f"glm5_next owner {owner!r}: production MoE collected "
                    f"{len(moe_weights)} expert tensors, expected {expected} "
                    f"(local experts {len(local_expert_ids)} of {module.mlp.num_experts})"
                )
            module.mlp.experts.load_weights([moe_weights])
            if hasattr(module.mlp.experts, "post_load_weights"):
                module.mlp.experts.post_load_weights()
            filled.update(n for n in params if n.startswith("mlp.experts."))

        # Routed experts are filled through the stacking rule / fused loader
        # above rather than by name. The two activation scales belong to
        # FP8_BLOCK_SCALES' static branch; this checkpoint is
        # activation_scheme='dynamic', so the scales are recomputed per call by
        # fp8_quantize_1x128 and these are never read. to_empty leaves them
        # uninitialized, so they are zeroed rather than left holding garbage
        # that a future code path could pick up.
        fused = (
            set()
            if production_moe
            else {
                "mlp.gate_up_proj",
                "mlp.down_proj",
                "mlp.gate_up_proj_scale",
                "mlp.down_proj_scale",
            }
        )
        unused = {n for n in params if n.endswith(("input_scale", "inv_input_scale"))}
        with torch.no_grad():
            for name in unused:
                params[name].zero_()
        unfilled = sorted(set(params) - filled - fused - unused)
        if unfilled:
            report.missing_destinations.extend(f"{prefix}{name}" for name in unfilled)
            raise ValueError(f"glm5_next owner {owner!r}: no checkpoint tensor reached {unfilled}")

    @staticmethod
    def _load_linear_shard(
        dest_mod: nn.Module,
        mod_path: str,
        group: Dict[str, Any],
        dests: Dict[str, str],
        report: Glm5NextLoadReport,
    ) -> None:
        """Hand one projection's checkpoint tensors to its own Linear loader.

        The destination's ``load_shard``/quant-method contract does the actual
        slicing (this rank's contiguous rows/columns, and the matching
        128x128 scale rows via ``scale_span=128``), so the loader never
        ``torch.chunk``\\ s a weight itself. The full checkpoint geometry is
        validated against the module's declared full shape first -- the local
        parameter is a shard, so the old exact-shape check cannot apply here.
        """
        full_shape = getattr(dest_mod, "glm5_full_shape", None)
        if full_shape is None and hasattr(dest_mod, "num_embeddings"):
            # The base-class LMHead is constructed Mapping-aware directly
            # rather than swapped, so it carries its own full geometry.
            full_shape = (dest_mod.num_embeddings, dest_mod.embedding_dim)
        weight = group.get("weight")
        if weight is None:
            raise ValueError(f"glm5_next projection {mod_path!r} has no checkpoint weight")
        checkpoint_shape = tuple(weight.shape if torch.is_tensor(weight) else weight.get_shape())
        if full_shape is not None and checkpoint_shape != tuple(full_shape):
            raise ValueError(
                f"glm5_next {mod_path!r}: checkpoint weight {checkpoint_shape} does not "
                f"match declared full shape {tuple(full_shape)}"
            )
        dest_mod.load_weights([group])

        tp_mode = getattr(dest_mod, "tp_mode", None)
        mode_str = getattr(tp_mode, "value", None)
        # The sharded dim's full width: rows for COLUMN, cols for ROW.
        full_dim = checkpoint_shape[0] if mode_str == "column" else checkpoint_shape[1]
        sharding = getattr(dest_mod, "tp_sharding", None)
        if isinstance(sharding, tuple):
            shard_range = (int(sharding[0]), int(sharding[1]))
        elif mode_str is not None and dest_mod.tp_size > 1:
            shard_range = (
                dest_mod._calc_shard(full_dim, dest_mod.tp_size, dest_mod.tp_rank),
                dest_mod._calc_shard(full_dim, dest_mod.tp_size, dest_mod.tp_rank + 1),
            )
        else:
            shard_range = (0, int(full_dim))
        report.tp_shards[mod_path] = {
            "mode": mode_str or "replicated",
            "tp_rank": int(dest_mod.tp_rank),
            "tp_size": int(dest_mod.tp_size),
            "range": list(shard_range),
            "full_shape": list(checkpoint_shape),
            "local_shape": list(dest_mod.weight.shape),
            "reduce_output": bool(getattr(dest_mod, "reduce_output", False)),
            "quant": str(dest_mod.weight.dtype).removeprefix("torch."),
        }
        for label, dest in dests.items():
            if label == "weight_scale_inv":
                report.dtypes[dest] = "float32"
            else:
                report.dtypes[dest] = str(dest_mod.weight.dtype).removeprefix("torch.")

    @staticmethod
    def _place_expert(
        module: nn.Module,
        match: "re.Match[str]",
        dest: str,
        tensor: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> None:
        """Stack one routed expert's projection into the fused MoE layout.

        ``gate_proj`` and ``up_proj`` share one ``[E, 2I, H]`` tensor with the
        gate rows first, which is the order ``_expert`` splits back out and the
        order the fused backends expect. The block scales follow the same
        stacking at 1/128th the resolution, so an off-by-one row split would
        show up as a scale applied to the wrong half rather than as a silent
        magnitude shift.
        """
        expert = int(match.group("expert"))
        proj = match.group("proj")
        is_scale = dest.endswith("_scale_inv")
        if proj == "down_proj":
            target = params["mlp.down_proj_scale" if is_scale else "mlp.down_proj"]
            view = target[expert]
        else:
            target = params["mlp.gate_up_proj_scale" if is_scale else "mlp.gate_up_proj"]
            rows = target.shape[1] // 2
            start = 0 if proj == "gate_proj" else rows
            view = target[expert, start : start + rows]
        if tuple(view.shape) != tuple(tensor.shape):
            raise ValueError(
                f"glm5_next {dest!r}: expert slice {tuple(view.shape)} does not match "
                f"checkpoint {tuple(tensor.shape)}"
            )
        with torch.no_grad():
            view.copy_(tensor.to(device=target.device, dtype=target.dtype))


# ---------------------------------------------------------------------------
# KDA linear attention
# ---------------------------------------------------------------------------


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """FLA-style L2 norm: ``x / sqrt(sum(x^2) + eps)``.

    The ``+ eps`` is inside the square root and is *added*, not maxed. The two
    spellings differ by ~1e-3 relative on small-norm rows, which is enough to
    move a delta-rule state over a 45-layer prefix, so the source form is kept
    verbatim rather than replaced with ``F.normalize``.
    """
    return x / torch.sqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def kda_recurrent_step(
    state: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One Kimi-delta-attention step for a batch of single-token requests.

    ``state`` is ``[N, H, K, V]`` float32 and is *not* modified in place; the
    caller decides where the new state lands. ``query``/``key`` are ``[N, H, K]``
    **before** L2 normalization, ``value`` is ``[N, H, V]``, ``g`` is the
    per-key-channel log decay ``[N, H, K]``, and ``beta`` is ``[N, H]``.

    Everything runs in float32: the state accumulates delta corrections over the
    whole prefix, so bf16 rounding here compounds across decode steps.
    """
    query = _l2norm(query.float()) * (query.shape[-1] ** -0.5)
    key = _l2norm(key.float())
    value = value.float()

    state = state * g.float().exp().unsqueeze(-1)
    recalled = (state * key.unsqueeze(-1)).sum(dim=-2)
    delta = (value - recalled) * beta.float().unsqueeze(-1)
    state = state + key.unsqueeze(-1) * delta.unsqueeze(-2)
    out = (state * query.unsqueeze(-1)).sum(dim=-2)
    return out, state


def kda_chunk_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chunked Kimi-delta-attention for one sequence, ``[1, S, H, *]`` inputs.

    This is the source's prefill algorithm (a UT-transform inside each chunk
    plus a state carry between chunks), not a rolled-out recurrence. Keeping the
    *algorithm* rather than only the result means the sequential decode path
    stays an independent cross-check of this one, which is what makes a
    prefill/decode disagreement observable instead of self-consistent.

    ``g`` is per key channel (``[1, S, H, K]``), which is the difference from a
    gated-delta-net: its decay is one scalar per head, so a scalar-gated kernel
    cannot express this and is not a valid substitute.
    """
    dtype = query.dtype
    # [1, H, S, D]
    query, key, value, g = (t.transpose(1, 2).float() for t in (query, key, value, g))
    beta = beta.transpose(1, 2).float()

    b, h, seq_len, k_dim = key.shape
    v_dim = value.shape[-1]
    pad = (chunk_size - seq_len % chunk_size) % chunk_size

    query = _l2norm(query) * (k_dim**-0.5)
    key = _l2norm(key)

    query, key, value, g = (
        torch.nn.functional.pad(t, (0, 0, 0, pad)) for t in (query, key, value, g)
    )
    beta = torch.nn.functional.pad(beta, (0, pad))
    total = seq_len + pad

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, g, k_beta, v_beta = (
        t.reshape(b, h, -1, chunk_size, t.shape[-1]) for t in (query, key, value, g, k_beta, v_beta)
    )
    beta = beta.reshape(b, h, -1, chunk_size)

    g = g.cumsum(dim=-2)
    device = query.device
    tri = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=device), 0)
    decay_mask = (g.unsqueeze(-2) - g.unsqueeze(-3)).exp()
    attn = -(k_beta.unsqueeze(-2) * key.unsqueeze(-3) * decay_mask).sum(-1).masked_fill(tri, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * attn[..., :i, :i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=device)

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp())

    if initial_state is None:
        state = torch.zeros(b, h, k_dim, v_dim, dtype=torch.float32, device=device)
    else:
        state = initial_state.float().clone()

    out = torch.zeros_like(value)
    strict = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=device), 1)
    for i in range(total // chunk_size):
        q_i, k_i, v_i, g_i = query[:, :, i], key[:, :, i], value[:, :, i], g[:, :, i]
        inter = (q_i * g_i.exp()) @ state
        intra = (q_i.unsqueeze(-2) * k_i.unsqueeze(-3) * decay_mask[:, :, i]).sum(-1)
        intra = intra.masked_fill(strict, 0)
        v_new = v_i - k_cumdecay[:, :, i] @ state
        out[:, :, i] = inter + intra @ v_new
        g_last = g_i[:, :, -1]
        state = (
            state * g_last.exp().unsqueeze(-1)
            + (k_i * (g_last.unsqueeze(-2) - g_i).exp()).transpose(-1, -2) @ v_new
        )

    out = out.reshape(b, h, -1, v_dim)[:, :, :seq_len]
    return out.transpose(1, 2).to(dtype), state


_KDA_PRODUCTION_DEPS: Optional[Dict[str, Any]] = None


def _kda_production():
    """The production KDA kernel stack (lazily imported, cached).

    These are the same in-tree kernels the Kimi K3 KDA path runs: the packed
    variable-length causal convolution and its single-token update (the exact
    ops the HF GLM5Next source itself names), the fused post-conv
    normalize/transpose, the CuTe DSL chunked-prefill dispatch, and the
    fresh-row reset. Imported lazily so model *discovery* stays light; the
    imports are hard requirements at forward time, not optional extras.
    """
    global _KDA_PRODUCTION_DEPS
    if _KDA_PRODUCTION_DEPS is None:
        from ..modules.kimi_kda._kda_kernels import KDAKernelDispatch, fused_kda_post_conv
        from ..modules.mamba.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
        from ..modules.mamba.recurrent_state_cache import reset_recurrent_state_rows

        _KDA_PRODUCTION_DEPS = {
            "KDAKernelDispatch": KDAKernelDispatch,
            "fused_kda_post_conv": fused_kda_post_conv,
            "causal_conv1d_fn": causal_conv1d_fn,
            "causal_conv1d_update": causal_conv1d_update,
            "reset_recurrent_state_rows": reset_recurrent_state_rows,
        }
    return _KDA_PRODUCTION_DEPS


#: Value-block width of the Triton decode step; head_dim must divide by it.
_KDA_DECODE_BV = 64


@triton.jit
def _kda_decode_step_kernel(
    CONV,
    G,
    BETA,
    A_LOG,
    DTB,
    SLOTS,
    STATE,
    OUT,
    conv_row_stride,
    state_slot_stride,
    state_head_stride,
    state_v_stride,
    out_row_stride,
    out_head_stride,
    scale,
    lower_bound,
    l2_eps,
    QKV: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BV: tl.constexpr,
):
    """One KDA decode step against the slot-indexed fp32 state pool.

    The fused CUDA decode kernel (``trtllm::kda_decode``) template-switches on
    head count and does not instantiate H=64 (`kdaDecode.cu` cases: 1..48, 96),
    so this layer's 64x128 geometry runs the same math as an OpenAI Triton
    kernel instead. Semantics per (request b, head h, value block):

      gate  = lower_bound * sigmoid(exp(A_log[h]) * (g + dt_bias))   per K chan
      q, k  = l2norm(post-conv rows), q scaled by K**-0.5
      S     = S * exp(gate);  S += k ⊗ ((v - S·k) * sigmoid(beta))
      out   = S · q

    All math is fp32 (the state integrates the whole prefix); no host value is
    read, so a captured CUDA graph replays against refreshed device buffers.
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    vb = tl.program_id(2)
    dk = tl.arange(0, K)
    dv = vb * BV + tl.arange(0, BV)

    base = b * conv_row_stride + h * K
    q = tl.load(CONV + base + dk).to(tl.float32)
    k = tl.load(CONV + base + QKV + dk).to(tl.float32)
    v = tl.load(CONV + base + 2 * QKV + dv).to(tl.float32)
    g = tl.load(G + b * (H * K) + h * K + dk).to(tl.float32)
    bias = tl.load(DTB + h * K + dk)
    decay_rate = tl.exp(tl.load(A_LOG + h))
    gate = lower_bound * tl.sigmoid(decay_rate * (g + bias))

    # FLA-style L2 norm: eps *added inside* the sqrt (the source form).
    q = q / tl.sqrt(tl.sum(q * q) + l2_eps) * scale
    k = k / tl.sqrt(tl.sum(k * k) + l2_eps)
    beta = tl.sigmoid(tl.load(BETA + b * H + h).to(tl.float32))

    slot = tl.load(SLOTS + b).to(tl.int64)
    state_ptr = (
        STATE + slot * state_slot_stride + h * state_head_stride + dv[:, None] * state_v_stride
    ) + dk[None, :]
    state = tl.load(state_ptr)
    state = state * tl.exp(gate)[None, :]
    recalled = tl.sum(state * k[None, :], axis=1)
    delta = (v - recalled) * beta
    state = state + k[None, :] * delta[:, None]
    out = tl.sum(state * q[None, :], axis=1)
    tl.store(state_ptr, state)
    tl.store(OUT + b * out_row_stride + h * out_head_stride + dv, out)


# One-shot (per process) KDA dispatch provenance. Engine-scale runs need the
# actually-dispatched kernel path — construction alone cannot prove it — and
# worker logs are the only channel from an MPI rank to the LLM API driver.
# At most one line per distinct path per process, so per-step decode stays
# log-silent after the first occurrence.
_KDA_DISPATCH_LOGGED: set = set()


def _log_kda_dispatch_once(path: str, layer_idx: int) -> None:
    if path not in _KDA_DISPATCH_LOGGED:
        _KDA_DISPATCH_LOGGED.add(path)
        logger.info(f"glm5_next KDA dispatch (first occurrence, layer {layer_idx}): {path}")


class Glm5NextLinearAttention(nn.Module):
    """KDA linear attention over TensorRT-LLM's packed token layout.

    Cache ownership
    ---------------
    The layer reads and writes two request-slot-indexed pools handed out by the
    cache manager, with no state of its own:

    * ``conv_pool``  ``[slots, 3 * H * D, W - 1]`` -- the causal history of the
      grouped short convolution over the concatenated ``[q | k | v]`` channels.
    * ``ssm_pool``   ``[slots, H, V, K]`` float32 -- the delta-rule recurrent
      accumulator. It is float32 even though the model is bf16, because it
      integrates delta corrections over the entire prefix.

    Both are exactly the shapes ``MambaHybridCacheManagerV2`` already allocates
    for a recurrent layer, so no new cache descriptor is introduced.

    Kernel dispatch
    ---------------
    The convolution always runs the production packed ops
    (``trtllm::causal_conv1d_fwd`` in prefill, ``trtllm::causal_conv1d_update``
    in decode) -- the same kernels the HF source itself calls, reading and
    advancing the pool in place with fresh/continuation handling. The
    delta-rule inner loop dispatches per phase:

    * prefill -> the in-tree CuTe DSL ``trtllm::kda_prefill`` against indexed
      state-pool rows whenever :class:`KDAKernelDispatch` accepts the batch
      (Blackwell + >= 4 total 64-token chunks -- the op's persistent scheduler
      launches a zero-size grid below that, by design). Smaller batches fall
      back to the HF-verified :func:`kda_chunk_prefill` torch scan on the same
      pool contract; ``last_prefill_path`` records which one ran.
    * decode -> :func:`_kda_decode_step_kernel`, always. The fused CUDA decode
      kernel does not instantiate this checkpoint's 64-head shape (see the
      kernel docstring), so the Triton step is the production decode path
      here, not a fallback.

    Tensor parallelism (Stage 5)
    ----------------------------
    With a ``mapping`` whose ``tp_size > 1`` the layer owns
    ``num_heads = total_num_heads // tp_size`` local heads (16 of 64 at TP4),
    following the Qwen3-Next / Kimi-Linear KDA precedent (each delta-rule head
    is independent, so head sharding needs no cross-rank math until
    ``o_proj``'s single row-parallel reduction). Everything derived from the
    head count is local: ``qkv_dim``/``conv_dim``, the grouped convolution and
    its ``[slots, 3 * H_local * D, W - 1]`` pool section, the
    ``[slots, H_local, V, K]`` recurrent pool, ``dt_bias``/``A_log``, and the
    Triton decode grid. The raw projections are constructed at the *full*
    checkpoint geometry and become local when
    :func:`glm5_next_swap_quantized_projections` installs the Mapping-aware
    ``Linear`` ownership (q/k/v/f_b/g_b/b column, f_a/g_a replicated, o_proj
    row + one reduction); ``conv1d``/``dt_bias``/``A_log`` are not Linear
    modules, so they are built local here and the loader slices their
    checkpoint sources by :meth:`kda_head_range`/:meth:`kda_channel_range`.
    At ``tp_size == 1`` every shape and code path is byte-identical to the
    pre-TP layer, which keeps the frozen PP4 evidence a valid oracle.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        dtype: torch.dtype = torch.bfloat16,
        mapping: Any = None,
    ):
        super().__init__()
        linear = dict(getattr(config, "linear_attn_config", None) or {})
        self.layer_idx = layer_idx
        self.hidden_size = int(config.hidden_size)
        self.total_num_heads = int(linear["num_heads"])
        self.tp_size = int(getattr(mapping, "tp_size", 1) or 1)
        self.tp_rank = int(getattr(mapping, "tp_rank", 0) or 0)
        if self.total_num_heads % self.tp_size:
            raise ValueError(
                f"glm5_next KDA has {self.total_num_heads} heads, not divisible by "
                f"tp_size {self.tp_size}"
            )
        self.num_heads = self.total_num_heads // self.tp_size
        self.head_dim = int(linear["head_dim"])
        self.conv_kernel_size = int(linear["short_conv_kernel_size"])
        self.gate_lower_bound = float(linear["gate_lower_bound"])
        self.qkv_dim = self.num_heads * self.head_dim
        self.total_qkv_dim = self.total_num_heads * self.head_dim
        self.conv_dim = 3 * self.qkv_dim
        self.rms_norm_eps = float(config.rms_norm_eps)

        lin = lambda i, o: nn.Linear(i, o, bias=False, dtype=dtype)  # noqa: E731
        # Raw projections carry the full checkpoint geometry; the Mapping-aware
        # swap divides them per the declared column/row ownership.
        self.q_proj = lin(self.hidden_size, self.total_qkv_dim)
        self.k_proj = lin(self.hidden_size, self.total_qkv_dim)
        self.v_proj = lin(self.hidden_size, self.total_qkv_dim)
        # One grouped convolution over the *local* [q | k | v]: the cache
        # stores those channels contiguously, so a single grouped conv reads
        # the pool without a gather. The checkpoint publishes the three
        # filters separately; the loader slices each by this rank's head
        # channel range before concatenating them.
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            bias=False,
            dtype=dtype,
        )
        self.f_a_proj = lin(self.hidden_size, self.head_dim)
        self.f_b_proj = lin(self.head_dim, self.total_qkv_dim)
        self.dt_bias = nn.Parameter(torch.zeros(self.qkv_dim, dtype=torch.float32))
        self.A_log = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        self.b_proj = lin(self.hidden_size, self.total_num_heads)
        self.g_a_proj = lin(self.hidden_size, self.head_dim)
        self.g_b_proj = lin(self.head_dim, self.total_qkv_dim)
        self.o_norm_weight = nn.Parameter(torch.ones(self.head_dim, dtype=dtype))
        self.o_proj = lin(self.total_qkv_dim, self.hidden_size)

        # Production kernel dispatch, decided once at construction (a per-call
        # decision could flip between capture and replay). The decode step is
        # shape-gated: the Triton kernel loads whole K rows and BV-wide value
        # blocks, so K must be a power of two and V a BV multiple -- true for
        # the one supported checkpoint (128/128) and asserted, not assumed.
        deps = _kda_production()
        self._kda_dispatch = deps["KDAKernelDispatch"](
            use_optimized_prefill=True, use_optimized_decode=False, use_optimized_verify=False
        )
        if self.head_dim & (self.head_dim - 1) or self.head_dim % _KDA_DECODE_BV:
            raise ValueError(
                f"glm5_next KDA decode kernel needs a power-of-two head_dim divisible by "
                f"{_KDA_DECODE_BV}; got {self.head_dim}"
            )
        #: Which inner loop the last forward_prefill ran:
        #: "trtllm::kda_prefill" (CuTe DSL, indexed pool) or "torch_chunk_scan".
        self.last_prefill_path: Optional[str] = None
        #: The decode inner loop is unconditional; named for evidence reports.
        self.decode_step_path = "triton::_kda_decode_step_kernel"

    # -- tensor-parallel ownership (consumed by the exact-placement loader) --

    def kda_head_range(self) -> Tuple[int, int]:
        """This rank's contiguous ``[start, end)`` on the 64-head axis (``A_log``)."""
        return (self.tp_rank * self.num_heads, (self.tp_rank + 1) * self.num_heads)

    def kda_channel_range(self) -> Tuple[int, int]:
        """This rank's ``[start, end)`` on one full per-head-channel axis.

        The 8192-wide flattened ``(head, head_dim)`` axis of ``dt_bias`` and of
        each of the three checkpoint convolution sources -- the same local-head
        slice the column-sharded q/k/v projections own, so the convolved
        channels and the recurrent heads always describe the same heads.
        """
        return (self.tp_rank * self.qkv_dim, (self.tp_rank + 1) * self.qkv_dim)

    # -- gates ------------------------------------------------------------

    def forget_gate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``gate_lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))``.

        The bound *scales a sigmoid*; it is not a clamp. The measured minimum on
        this checkpoint approaches -5.0 to within 5e-7 without ever reaching it,
        which is the signature of the scaled form -- a ``clamp(min=-5.0)`` would
        sit exactly on the bound.
        """
        gate = self.f_b_proj(self.f_a_proj(hidden_states))
        g = (gate.float() + self.dt_bias.float()).view(-1, self.num_heads, self.head_dim)
        decay = torch.exp(self.A_log.float()).view(1, self.num_heads, 1)
        return self.gate_lower_bound * torch.sigmoid(decay * g)

    def _gated_out_norm(self, core: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """FP32 RMS norm over the head dim, gated by ``sigmoid(gate)``."""
        x = core.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.rms_norm_eps)
        x = x * self.o_norm_weight.float()
        return (x * torch.sigmoid(gate.float())).to(self.o_proj.weight.dtype)

    # -- projections --------------------------------------------------------

    def _project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                self.q_proj(hidden_states),
                self.k_proj(hidden_states),
                self.v_proj(hidden_states),
            ],
            dim=-1,
        )

    def _finish(self, core: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.g_b_proj(self.g_a_proj(hidden_states)).view(-1, self.num_heads, self.head_dim)
        out = self._gated_out_norm(core, gate).reshape(-1, self.qkv_dim)
        return self.o_proj(out)

    # -- phases -----------------------------------------------------------

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Sequence[int],
        slot_ids: torch.Tensor,
        conv_pool: torch.Tensor,
        ssm_pool: torch.Tensor,
        cached_lens: Sequence[int],
    ) -> torch.Tensor:
        """Context phase for ``len(cu_seqlens) - 1`` packed requests.

        ``cached_lens[i]`` is how many tokens of request ``i`` the cache already
        holds, exactly as on the sparse-attention side. A non-zero value means
        this is a continuation chunk, so the convolution and recurrent state are
        seeded from the slot; zero means a new request, which must start from
        zero *even if the slot still holds a previous request's state*.

        The argument is required rather than inferred. A continuation seeded
        from zero reproduces one-shot prefill exactly on its first chunk and
        only diverges afterwards, and a new request seeded from a recycled slot
        is the cross-request leak this cache has to rule out -- neither is
        visible from the pool contents alone.

        Prefill is never CUDA-graph-captured (decode-only graphs), so the small
        host-derived schedule tensors built here are legal and cheap.
        """
        deps = _kda_production()
        num_requests = len(cu_seqlens) - 1
        bounds = [int(v) for v in cu_seqlens]
        seq_lens = [bounds[i + 1] - bounds[i] for i in range(num_requests)]
        total = bounds[-1]
        device = hidden_states.device

        # Packed [3D, T] channel-major projections; the production varlen
        # convolution advances the pool rows with fresh/continuation handling
        # (fresh rows are cleared by the kernel). The *returned* tensor is the
        # convolved output: it is the input buffer updated in place for the
        # dim-major T > 1 layout, but a fresh buffer for T == 1, where the
        # size-1 token axis makes ``contiguous()`` keep the token-major strides
        # and the op takes its out-of-place channel-last route.
        packed = self._project(hidden_states).transpose(0, 1).contiguous()
        slots_i32 = slot_ids.to(device=device, dtype=torch.int32).contiguous()
        has_init = torch.tensor(
            [int(c) > 0 for c in cached_lens[:num_requests]], device=device, dtype=torch.bool
        )
        packed = deps["causal_conv1d_fn"](
            packed,
            self.conv1d.weight.squeeze(1),
            None,
            query_start_loc=torch.tensor(bounds, device=device, dtype=torch.int32),
            cache_indices=slots_i32,
            has_initial_state=has_init,
            conv_states=conv_pool,
            activation="silu",
        )

        # Chunk schedule from host lengths (row = [seq id, chunk id in seq]),
        # matching fla.index.prepare_chunk_indices without its device readback.
        cu_dev = torch.tensor(bounds, device=device, dtype=torch.long)
        chunk_rows = [
            (sid, j) for sid, length in enumerate(seq_lens) for j in range((length + 63) // 64)
        ]
        chunk_indices = torch.tensor(chunk_rows, device=device, dtype=torch.long)
        use_production = self._kda_dispatch.can_use_indexed_prefill(
            state_pool=ssm_pool,
            state_indices=slots_i32,
            has_initial_states=has_init,
            cu_seqlens=cu_dev,
            num_sequences=num_requests,
            num_tokens=total,
            chunk_indices=chunk_indices,
        )

        if use_production:
            # Fresh rows must be zero before the kernel reads them as initial
            # state; continuation rows carry their state in place.
            deps["reset_recurrent_state_rows"](ssm_pool, slots_i32, has_init)
            q, k, v = deps["fused_kda_post_conv"](
                packed, num_heads=self.num_heads, head_dim=self.head_dim
            )
            g_raw = self.f_b_proj(self.f_a_proj(hidden_states)).view(
                1, total, self.num_heads, self.head_dim
            )
            beta_raw = self.b_proj(hidden_states).float().view(1, total, self.num_heads)
            core, final_state = self._kda_dispatch.prefill_chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g_raw,
                beta=beta_raw,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                scale=self.head_dim**-0.5,
                initial_state=None,
                safe_gate=True,
                lower_bound=self.gate_lower_bound,
                cu_seqlens=cu_dev,
                chunk_indices=chunk_indices,
                state_pool=ssm_pool,
                state_indices=slots_i32,
                varlen_is_aligned=all(length % 64 == 0 for length in seq_lens),
                single_sequence_length=seq_lens[0] if num_requests == 1 else None,
            )
            assert final_state is None, "indexed kda_prefill updates the pool in place"
            self.last_prefill_path = "trtllm::kda_prefill"
            _log_kda_dispatch_once("prefill=trtllm::kda_prefill", self.layer_idx)
            return self._finish(core[0], hidden_states)

        # HF-verified torch scan on the same production conv output and pool
        # contract. This is the documented small-batch route: below 4 total
        # 64-token chunks the CuTe op's persistent scheduler cannot launch.
        post = packed.transpose(0, 1)
        outputs = []
        for i in range(num_requests):
            start, end = bounds[i], bounds[i + 1]
            slot = int(slot_ids[i])
            x = hidden_states[start:end]
            q, k, v = torch.split(post[start:end], [self.qkv_dim] * 3, dim=-1)
            shape = (1, end - start, self.num_heads, self.head_dim)
            seeded = int(cached_lens[i]) > 0
            initial = ssm_pool[slot].transpose(-1, -2).unsqueeze(0) if seeded else None
            core, state = kda_chunk_prefill(
                q.reshape(shape),
                k.reshape(shape),
                v.reshape(shape),
                self.forget_gate(x).unsqueeze(0),
                torch.sigmoid(self.b_proj(x)).unsqueeze(0),
                initial_state=initial,
            )
            ssm_pool[slot] = state[0].transpose(-1, -2).to(ssm_pool.dtype)
            outputs.append(core[0])
        self.last_prefill_path = "torch_chunk_scan"
        _log_kda_dispatch_once("prefill=torch_chunk_scan(small-batch route)", self.layer_idx)
        return self._finish(torch.cat(outputs, dim=0), hidden_states)

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        slot_ids: torch.Tensor,
        conv_pool: torch.Tensor,
        ssm_pool: torch.Tensor,
    ) -> torch.Tensor:
        """Generation phase: exactly one token per request, fully batched.

        The production single-token convolution advances the pool in place and
        the Triton delta step updates the indexed fp32 state rows in place. No
        Python-level branch depends on a device value, so the whole step is
        capture-safe for CUDA graphs: every replay re-reads the (refreshed)
        slot-id buffer and pools at their captured addresses.
        """
        deps = _kda_production()
        batch = hidden_states.shape[0]
        if ssm_pool.stride(-1) != 1:
            raise ValueError("glm5_next KDA decode needs a K-contiguous recurrent pool")
        _log_kda_dispatch_once(
            "decode=trtllm::causal_conv1d_update+triton_kda_delta_step", self.layer_idx
        )

        conv_out = deps["causal_conv1d_update"](
            self._project(hidden_states),
            conv_pool,
            self.conv1d.weight.squeeze(1),
            None,
            activation="silu",
            conv_state_indices=slot_ids.to(device=conv_pool.device, dtype=torch.int32),
        ).contiguous()

        g_raw = self.f_b_proj(self.f_a_proj(hidden_states)).contiguous()
        beta_raw = self.b_proj(hidden_states).contiguous()
        core = torch.empty(
            batch, self.num_heads, self.head_dim, dtype=torch.float32, device=conv_out.device
        )
        _kda_decode_step_kernel[(batch, self.num_heads, self.head_dim // _KDA_DECODE_BV)](
            conv_out,
            g_raw,
            beta_raw,
            self.A_log,
            self.dt_bias,
            slot_ids.to(device=conv_out.device, dtype=torch.long).contiguous(),
            ssm_pool,
            core,
            conv_out.stride(0),
            ssm_pool.stride(0),
            ssm_pool.stride(1),
            ssm_pool.stride(2),
            core.stride(0),
            core.stride(1),
            self.head_dim**-0.5,
            self.gate_lower_bound,
            1e-6,
            QKV=self.qkv_dim,
            H=self.num_heads,
            K=self.head_dim,
            BV=_KDA_DECODE_BV,
            num_warps=4,
        )
        return self._finish(core.to(hidden_states.dtype), hidden_states)


# ---------------------------------------------------------------------------
# Pool-compressed sparse attention
# ---------------------------------------------------------------------------

# The -1 index sentinel and the paged/row cache addressing live with the
# glm_kpool sparse backend: they are cache-path contracts the backend owns.
# The sentinel itself is *source-defined* (the indexer emits a variable
# logical width padded to a fixed one, every unused slot carries -1) and it
# must survive to the kernel unchanged -- clamping it to 0 or ``kv_len - 1``
# silently attends a real token, which no tolerance-based check would catch.


class Glm5NextLayerNorm(nn.Module):
    """A ``nn.LayerNorm`` whose construction is safe under ``MetaInitMode``.

    ``nn.LayerNorm.reset_parameters`` runs ``ones_``/``zeros_`` (``aten.fill_`` /
    ``aten.zero_``) on freshly ``empty`` weights. Under the runtime's
    ``MetaInitMode`` those weights are ``meta`` and ``fill_``/``zero_`` are not
    in its random-init allow-list, so construction raises ``MetaInitException``.
    ``ModelLoader`` then catches it and falls back to *regular* init, which
    real-allocates the whole model in host RAM on every pipeline-parallel rank
    -- eight ranks each materializing the 328 GB checkpoint scale is exactly the
    global host-OOM SIGKILL observed in the PP=8 smoke.

    Building the affine parameters through the ``torch.ones``/``torch.zeros``
    factories avoids that failure a different way: ``MetaInitMode`` does *not*
    intercept ``ones``/``zeros``/``full`` (its ``init_ops`` allow-list is only
    ``empty``/``empty_like``), so these affines are constructed as real -- but
    tiny, ``normalized_shape``-sized -- CPU tensors, and no in-place init op
    ever touches a ``meta`` tensor, so no ``MetaInitException`` is raised.
    They stay real CPU tensors until the owner moves them (``to_empty`` /
    ``model.to(device)``) and ``load_weights`` fills them from the checkpoint.
    The forward is ``F.layer_norm`` -- bit-identical to ``nn.LayerNorm`` --
    and the parameter names (``weight``/``bias``) match the checkpoint so
    name-based loading is unchanged.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5, *, dtype=None):
        super().__init__()
        self.normalized_shape = (int(normalized_shape),)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(normalized_shape, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(normalized_shape, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.layer_norm(
            hidden_states, self.normalized_shape, self.weight, self.bias, self.eps
        )


class Glm5NextIndexer(nn.Module):
    """Pool-compressed DSA indexer (``Glm5NextTextIndexer``).

    Stock DeepSeek-V3.2 DSA selects ``index_topk`` individual keys. This one
    selects ``index_topk / index_kpool`` *compressed pools*, expands each back
    into its member positions, and always appends the incomplete tail, so the
    two are not interchangeable even though both end up with ~2048 positions.

    Cached state is the packed per-token pair ``[k(128) | gate(128)]``; pools
    are rebuilt from it on every step rather than cached, because a pool's
    membership depends on how many tokens are visible to the *query*, which
    changes every step. A cache sized to hold 512 pool keys would be the wrong
    object.

    Unlike the HF module there is no left padding here: TensorRT-LLM stores
    exactly one request's tokens per cache slot starting at position 0, so
    ``first_key`` is always 0 and the packed validity channel HF carries for
    padded batches is replaced by the request's own ``kv_len``.

    Tensor parallelism (Stage 5): with ``tp_size > 1`` the 32 scoring heads
    split ``n_heads = 32 // tp_size`` per rank (the Goal-5.1 column ownership
    of ``wq_b``/``weights_proj``), while the shared pool-key path (``wk``,
    ``k_norm``, the APE and compress gate) stays replicated so every rank
    rebuilds bitwise-identical pools. Each rank's weighted head contribution
    is a *partial sum* of the full 32-head score, so :meth:`select` runs
    exactly one FP32 SUM all-reduce over ``index_scores`` before masking and
    top-k; the reduced tensor is bitwise identical on every rank (each rank
    applies the same reduction order), so all ranks select identical
    pool/tail/-1 indices with no index exchange. vLLM and SGLang instead
    replicate the whole indexer (identical scores by redundant compute, no
    collective); this port keeps the sharded ownership the Stage-5 plan and
    Goal 5.1 landed and buys the same invariant with one small fp32
    collective -- see the Goal-5.2 design note for the file/commit citations.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        dtype: torch.dtype = torch.bfloat16,
        mapping: Any = None,
        allreduce_strategy: Any = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = int(config.hidden_size)
        self.total_n_heads = int(config.index_n_heads)
        self.tp_size = int(getattr(mapping, "tp_size", 1) or 1)
        self.tp_rank = int(getattr(mapping, "tp_rank", 0) or 0)
        if self.total_n_heads % self.tp_size:
            raise ValueError(
                f"glm5_next indexer has {self.total_n_heads} scoring heads, not "
                f"divisible by tp_size {self.tp_size}"
            )
        self.n_heads = self.total_n_heads // self.tp_size
        self.head_dim = int(config.index_head_dim)
        self.index_topk = int(config.index_topk)
        self.index_kpool = int(config.index_kpool)
        self.always_select_tail = bool(config.index_kpool_always_select_tail)
        self.softmax_scale = self.head_dim**-0.5
        self.select_k = self.index_topk // self.index_kpool
        self.output_width = self.index_topk + (
            self.index_kpool - 1 if self.always_select_tail else 0
        )

        # Raw projections at the full checkpoint geometry; the Mapping-aware
        # swap column-shards them to the local scoring heads.
        self.wq_b = nn.Linear(
            int(config.q_lora_rank), self.total_n_heads * self.head_dim, bias=False, dtype=dtype
        )
        self.wk = nn.Linear(self.hidden_size, self.head_dim, bias=False, dtype=dtype)
        self.k_norm = Glm5NextLayerNorm(self.head_dim, eps=1e-6, dtype=dtype)
        self.weights_proj = nn.Linear(self.hidden_size, self.total_n_heads, bias=False, dtype=dtype)
        self.index_kpool_compress_ape = nn.Parameter(
            torch.zeros(self.index_kpool, self.head_dim, dtype=dtype)
        )
        self.index_kpool_compress_gate = nn.Parameter(
            torch.zeros(self.head_dim, self.hidden_size, dtype=dtype)
        )
        # The one FP32 score reduction (plan Decision F), parameter-free so it
        # is invisible to the loader's accounting even as a registered
        # submodule.
        #
        # Strategy is pinned to NCCL rather than left at AUTO. AUTO routes
        # through the AllReduce autotuner during the engine's warmup pass
        # (``tunable_allreduce`` -> ``AutoTuner.choose_one`` ->
        # ``_profile_single_kernel``), which profiles collective tactics in a
        # way that deadlocked all four TP ranks here at engine bring-up: this
        # reduction is a tiny fp32 ``[num_tokens, num_pools]`` tensor whose
        # profiling diverged across ranks (the wide bf16 hidden-size KDA/MLA
        # o_proj reductions in the earlier layers autotuned cleanly, so the
        # hang is specific to this small fp32 collective, not TP collectives in
        # general). Autotuning a reduction this small buys nothing next to the
        # MLA/MoE GEMMs, and NCCL is always correct and available -- the same
        # trade-off the ``TLLM_DISABLE_ALLREDUCE_AUTOTUNE`` escape hatch makes
        # globally, scoped here to just the problematic op so the beneficial
        # large-collective autotuning elsewhere is untouched. ``select``
        # applies the reduction on identical replicated inputs on every rank,
        # so NCCL SUM keeps the bitwise-identical-across-ranks guarantee.
        self.score_all_reduce = None
        if self.tp_size > 1:
            from ..distributed import AllReduce, AllReduceStrategy

            self.score_all_reduce = AllReduce(mapping=mapping, strategy=AllReduceStrategy.NCCL)

    @property
    def packed_state_dim(self) -> int:
        """Width of the cached per-token state: ``[k | gate]``."""
        return 2 * self.head_dim

    def packed_state(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Per-token ``[k(head_dim) | gate(head_dim)]`` written to the cache."""
        k = self.k_norm(self.wk(hidden_states))
        gate = torch.nn.functional.linear(hidden_states, self.index_kpool_compress_gate)
        return torch.cat([k, gate], dim=-1)

    def build_pools(
        self,
        packed: torch.Tensor,
        kv_lens: torch.Tensor,
        num_pools: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rebuild compressed pools from cached packed state.

        ``packed`` is ``[N, num_pools * index_kpool, 2 * head_dim]``, already
        gathered from the cache with out-of-range rows zeroed. Returns pool
        keys ``[N, P, head_dim]``, the last member position ``[N, P]``, and pool
        validity ``[N, P]``.

        Only *complete* pools are candidates -- an incomplete trailing group is
        never a pool, it is the tail. The pool key is a learned softmax mixture
        over its four members using ``gate + ape`` logits, so it is not a mean
        and not the last key.
        """
        n = packed.shape[0]
        kpool = self.index_kpool
        device = packed.device
        keys, gate = torch.split(packed, [self.head_dim, self.head_dim], dim=-1)
        keys = keys.view(n, num_pools, kpool, self.head_dim)
        gate = gate.view(n, num_pools, kpool, self.head_dim)

        member_pos = torch.arange(num_pools * kpool, device=device).view(1, num_pools, kpool)
        member_valid = member_pos < kv_lens.view(n, 1, 1)
        pool_valid = member_valid.all(-1)

        logits = gate.float() + self.index_kpool_compress_ape.float().view(1, 1, kpool, -1)
        logits = logits.masked_fill(~member_valid.unsqueeze(-1), float("-inf"))
        probs = torch.nan_to_num(logits.softmax(dim=2)).to(keys.dtype)
        pool_keys = (probs * keys).sum(dim=2)
        return pool_keys, member_pos[..., -1].expand(n, num_pools), pool_valid

    def select(
        self,
        q_resid: torch.Tensor,
        hidden_states: torch.Tensor,
        pool_keys: torch.Tensor,
        pool_last_pos: torch.Tensor,
        pool_valid: torch.Tensor,
        token_request: torch.Tensor,
        query_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Score pools, select the top ``select_k``, expand, and append the tail.

        Returns int32 ``[num_tokens, output_width]`` positions into each token's
        own request cache, with :data:`INDEX_SENTINEL` in every unselected or
        invalid slot.

        At ``tp_size > 1`` the relu'd per-head scores cover only this rank's
        local heads, so their weighted combination is a partial sum of the
        source's 32-head score; the single FP32 SUM all-reduce below completes
        it *before* candidate masking and top-k. Both operate on replicated
        inputs afterwards, so every rank selects identical indices.
        """
        num_tokens = hidden_states.shape[0]
        device = hidden_states.device
        kpool = self.index_kpool

        q = self.wq_b(q_resid).view(num_tokens, self.n_heads, self.head_dim)
        keys = pool_keys[token_request]
        scores = torch.nn.functional.relu(
            torch.matmul(q.float(), keys.transpose(-1, -2).float()) * self.softmax_scale
        )
        # The head-mixing normalization is the source's over the *full* head
        # count -- the local matmul is a partial sum, not a smaller indexer.
        weights = self.weights_proj(hidden_states).float() * (self.total_n_heads**-0.5)
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)
        if self.score_all_reduce is not None:
            # Exactly one collective per selection; every rank applies the
            # same reduction order, so the result is bitwise identical across
            # ranks and the top-k below needs no further exchange.
            index_scores = self.score_all_reduce(index_scores.contiguous())

        candidates = pool_valid[token_request] & (
            pool_last_pos[token_request] <= query_pos[:, None]
        )
        index_scores = index_scores.masked_fill(~candidates, torch.finfo(index_scores.dtype).min)

        select_k = min(self.select_k, index_scores.shape[-1])
        selected = index_scores.topk(select_k, dim=-1).indices
        selected_valid = candidates.gather(-1, selected)

        members = torch.arange(kpool, device=device)
        expanded = (selected * kpool).unsqueeze(-1) + members
        expanded = expanded.masked_fill(~selected_valid.unsqueeze(-1), INDEX_SENTINEL)
        out = expanded.flatten(-2)

        if self.always_select_tail and kpool > 1:
            visible = query_pos + 1
            tail_count = visible % kpool
            tail_start = visible - tail_count
            offsets = torch.arange(kpool - 1, device=device)
            tail = tail_start[:, None] + offsets
            tail = tail.masked_fill(offsets[None] >= tail_count[:, None], INDEX_SENTINEL)
            out = torch.cat([out, tail], dim=-1)

        if out.shape[-1] < self.output_width:
            out = torch.nn.functional.pad(
                out, (0, self.output_width - out.shape[-1]), value=INDEX_SENTINEL
            )
        return out[..., : self.output_width].to(torch.int32)


class Glm5NextSparseAttention(nn.Module):
    """Fully NoPE sparse MLA with a pool-compressed indexer.

    ``qk_rope_head_dim`` is 0 on this checkpoint and ``mla_use_nope`` is true:
    there is no text rotary call and no rotary cache. ``indexer_rope_interleave``
    is present in the config but vestigial for the text path, so no rotary
    branch is created for it -- long-range position sensitivity comes from the
    causal KDA layers and the indexer's learned pool APE.

    Layering follows the attention developer guide. This module owns the
    module math only: low-rank q/kv projections and norms, the pool indexer's
    scoring/selection (model-layer sparse prediction, as in MiniMax-M3), the
    absorbed-MLA query/value reassociation, and ``o_proj``. Everything below
    that -- the paged latent/indexer cache path and the sparse-MLA core --
    belongs to ``self.attn_backend``, a
    :class:`~tensorrt_llm._torch.attention_backend.sparse.glm_kpool.GlmKpoolSparseAttention`:
    a ``TrtllmAttention`` subclass (the fully-NoPE branch of the TRTLLM sparse
    family) constructed through the standard ``create_attention(...)``
    dispatch on the configured backend slot (``ModelConfig.attn_backend``,
    default TRTLLM) with ``SparseParams(algorithm="glm_kpool")``. Its typed
    metadata family is ``TrtllmAttentionMetadata`` (``attn_backend.Metadata``),
    the class the engine constructs for this model.

    Cache ownership: one latent ``kv_lora_rank``-wide entry per token (the
    pre-``kv_b_proj`` latent, not expanded K/V) plus the indexer's packed
    ``[k | gate]`` state, both held by the one hybrid ``KVCacheManagerV2`` and
    read/written only by the backend, which derives every pool, block table,
    and visible length from the prepared attention metadata it is handed --
    this module passes schedule values and the metadata, never raw pools. The
    pool-expanded selection travels to the backend inside the standard
    ``AttentionForwardArgs.sparse_backend_args`` carrier.

    ``kv_b_proj`` is on this checkpoint's ``modules_to_not_convert`` list
    (BF16), so absorption is a reassociation of the same BF16 weights, not a
    dequantization.

    Tensor parallelism (Stage 5): with ``tp_size > 1`` this module owns
    ``num_heads = 64 // tp_size`` local query heads and the matching rows of
    the column-sharded ``q_b_proj``/``kv_b_proj`` (so the absorbed per-head
    views are local by construction), while the low-rank latents
    (``q_a``/``kv_a``) and both norms stay replicated and the row-sharded
    ``o_proj`` performs the branch's one reduction -- the DeepSeek-V3 MLA
    ownership. The latent cache and the indexer's packed state stay
    *complete* (512- and 256-wide) on every rank: ``num_kv_heads == 1`` is
    never divided, all ranks compute identical latent/packed rows from the
    replicated projections, and each rank's backend reads its own full copy.
    The backend is constructed with the local head count, exactly as MLA's
    per-rank ``num_heads // tp_size``. At ``tp_size == 1`` construction and
    math are byte-identical to the pre-TP module.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        dtype: torch.dtype = torch.bfloat16,
        attn_backend: str = "TRTLLM",
        mapping: Any = None,
        allreduce_strategy: Any = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = int(config.hidden_size)
        self.total_num_heads = int(config.num_attention_heads)
        self.tp_size = int(getattr(mapping, "tp_size", 1) or 1)
        self.tp_rank = int(getattr(mapping, "tp_rank", 0) or 0)
        if self.total_num_heads % self.tp_size:
            raise ValueError(
                f"glm5_next sparse MLA has {self.total_num_heads} heads, not divisible "
                f"by tp_size {self.tp_size}"
            )
        self.num_heads = self.total_num_heads // self.tp_size
        self.q_lora_rank = int(config.q_lora_rank)
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.qk_nope_head_dim = int(config.qk_nope_head_dim)
        self.qk_rope_head_dim = int(config.qk_rope_head_dim)
        self.v_head_dim = int(config.v_head_dim)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        if self.qk_rope_head_dim != 0:
            raise ValueError(
                "glm5_next text attention is fully NoPE; a non-zero qk_rope_head_dim "
                f"({self.qk_rope_head_dim}) would need a rotary path this bring-up does not have"
            )
        self.scaling = self.qk_head_dim**-0.5
        eps = float(config.rms_norm_eps)

        lin = lambda i, o: nn.Linear(i, o, bias=False, dtype=dtype)  # noqa: E731
        self.q_a_proj = lin(self.hidden_size, self.q_lora_rank)
        self.q_a_layernorm = RMSNorm(hidden_size=self.q_lora_rank, eps=eps, dtype=dtype)
        self.q_b_proj = lin(self.q_lora_rank, self.total_num_heads * self.qk_head_dim)
        self.kv_a_proj_with_mqa = lin(self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_a_layernorm = RMSNorm(hidden_size=self.kv_lora_rank, eps=eps, dtype=dtype)
        self.kv_b_proj = lin(
            self.kv_lora_rank, self.total_num_heads * (self.qk_nope_head_dim + self.v_head_dim)
        )
        self.o_proj = lin(self.total_num_heads * self.v_head_dim, self.hidden_size)
        self.indexer = Glm5NextIndexer(
            config, layer_idx, dtype=dtype, mapping=mapping, allreduce_strategy=allreduce_strategy
        )

        # The production sparse-MLA backend, selected through the standard
        # dispatch (`get_attention_backend(attn_backend, sparse_params)` via
        # `create_attention`). It consumes absorbed latent-space queries, so
        # its head_dim is kv_lora_rank, and the latent cache is MQA-style
        # (one KV head). AttentionBackend is not an nn.Module: this is a
        # plain attribute, invisible to state_dict/loading.
        self.attn_backend = create_attention(
            attn_backend,
            layer_idx,
            num_heads=self.num_heads,
            head_dim=self.kv_lora_rank,
            num_kv_heads=1,
            dtype=dtype,
            sparse_params=GlmKpoolSparseParams(
                kv_lora_rank=self.kv_lora_rank,
                qk_nope_head_dim=self.qk_nope_head_dim,
                q_lora_rank=self.q_lora_rank,
                v_head_dim=self.v_head_dim,
                index_topk=self.indexer.index_topk,
                index_kpool=self.indexer.index_kpool,
                index_always_select_tail=self.indexer.always_select_tail,
            ),
        )

    def project_latent(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The cached latent: ``kv_a_layernorm(kv_a_proj(x))``, width 512."""
        return self.kv_a_layernorm(self.kv_a_proj_with_mqa(hidden_states))

    def absorbed_kv_b(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-head absorbed views of ``kv_b_proj``: ``(w_k, w_v_t)``.

        ``w_k`` is ``[H, qk_nope, kv_lora]`` (queries -> latent space) and
        ``w_v_t`` is ``[H, kv_lora, v_head]`` (latent attention output -> V
        space). Score and value absorption are exact reassociations of the
        unabsorbed math: ``q . (W_k @ c) == (W_k^T @ q) . c`` and
        ``sum_j p_j (W_v @ c_j) == W_v @ sum_j p_j c_j``. Contiguous copies
        are cached after weight loading (keyed on the weight's data pointer
        and version) so the per-step bmm never re-materializes them; the
        cache is a plain tuple attribute, invisible to ``state_dict``.
        """
        weight = self.kv_b_proj.weight
        key = (weight.data_ptr(), weight._version)
        cached = self.__dict__.get("_absorbed_kv_b_cache")
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        if weight.dtype != torch.bfloat16:
            raise ValueError(
                "glm5_next sparse MLA absorption expects the BF16-excluded "
                f"kv_b_proj weight, got dtype {weight.dtype}"
            )
        per_head = weight.view(
            self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank
        )
        w_k = per_head[:, : self.qk_nope_head_dim, :].contiguous()
        w_v_t = per_head[:, self.qk_nope_head_dim :, :].transpose(1, 2).contiguous()
        self._absorbed_kv_b_cache = (key, w_k, w_v_t)
        return w_k, w_v_t

    def absorb_query(self, query: torch.Tensor) -> torch.Tensor:
        """``[T, H, qk_nope] -> [T, H, kv_lora]`` (fully NoPE: q is all nope)."""
        w_k, _ = self.absorbed_kv_b()
        return torch.bmm(query.transpose(0, 1), w_k).transpose(0, 1)

    def project_output_latent(self, out_latent: torch.Tensor) -> torch.Tensor:
        """Flat latent attention output -> V space -> ``o_proj``. ``[T, hidden]``.

        ``out_latent`` is the backend's base-contract result
        ``[T, num_heads * kv_lora]``; the per-head view for the absorbed V
        projection is this module's concern, not the backend boundary's.
        """
        _, w_v_t = self.absorbed_kv_b()
        tokens = out_latent.shape[0]
        per_head = out_latent.view(tokens, self.num_heads, self.kv_lora_rank)
        out = torch.bmm(per_head.transpose(0, 1), w_v_t)  # [H, T, v_head]
        return self.o_proj(out.transpose(0, 1).reshape(tokens, -1))

    def _run_one(
        self,
        hidden_states: torch.Tensor,
        latent_prefix: torch.Tensor,
        packed_prefix: torch.Tensor,
        query_pos: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """Select and attend for one request whose prefix is already gathered.

        ``latent_prefix`` is the request's own contiguous latent rows, so the
        selected positions *are* the backend's row ids and the sentinel passes
        straight through. The empty-row check is a host assertion; it only
        runs on the never-captured prefill path. The selection travels to the
        backend in the typed ``sparse_backend_args`` carrier of the standard
        ``AttentionBackend.forward`` contract.
        """
        kpool = self.indexer.index_kpool
        kv_len = latent_prefix.shape[0]
        num_pools = (kv_len + kpool - 1) // kpool
        pad = num_pools * kpool - kv_len
        packed = torch.nn.functional.pad(packed_prefix, (0, 0, 0, pad)).unsqueeze(0)
        kv_lens = torch.tensor([kv_len], device=hidden_states.device)
        pool_keys, pool_last, pool_valid = self.indexer.build_pools(packed, kv_lens, num_pools)

        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        query = self.q_b_proj(q_resid).view(-1, self.num_heads, self.qk_head_dim)
        token_request = torch.zeros(
            hidden_states.shape[0], dtype=torch.long, device=hidden_states.device
        )
        topk = self.indexer.select(
            q_resid, hidden_states, pool_keys, pool_last, pool_valid, token_request, query_pos
        )
        if not bool((topk >= 0).any(-1).all()):
            raise ValueError(
                "glm5_next sparse attention: a query row selected no visible key. "
                "Every query is covered either by a complete pool or by the always-"
                "selected tail, so an empty row means the pool/tail contract broke."
            )
        q_latent = self.absorb_query(query)
        out_latent = self.attn_backend.forward(
            q_latent,
            latent_prefix,
            None,
            metadata,
            AttentionForwardArgs(
                attention_input_type=AttentionInputType.context_only,
                sparse_backend_args=SparseBackendForwardArgs(topk_indices=topk),
            ),
        )
        return self.project_output_latent(out_latent)

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Sequence[int],
        cached_lens: Sequence[int],
        metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """Context phase, including continuation chunks.

        ``cached_lens[i]`` is how many tokens of request ``i`` are already in
        the cache, so a chunk is scored against the whole visible prefix rather
        than only against its own tokens. Scoring a chunk in isolation is the
        classic chunked-prefill bug here: it still passes a one-shot test.
        Only schedule values are passed here; every pool write and read goes
        through the backend's metadata-derived cache path.
        """
        outputs = []
        for i in range(len(cu_seqlens) - 1):
            start, end = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
            cached = int(cached_lens[i])
            x = hidden_states[start:end]
            positions = torch.arange(cached, cached + (end - start), device=x.device)
            self.attn_backend.append_paged_state(
                self.project_latent(x),
                self.indexer.packed_state(x),
                positions,
                metadata,
                request_index=i,
            )
            latent_prefix, packed_prefix = self.attn_backend.gather_paged_prefix(
                cached + (end - start), metadata, request_index=i
            )
            outputs.append(self._run_one(x, latent_prefix, packed_prefix, positions, metadata))
        return torch.cat(outputs, dim=0)

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        kv_lens: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """Generation phase: one token per request, fully batched.

        CUDA-graph contract: every shape here is a function of the *buffer*
        geometry (the metadata's block-table width times tokens_per_block),
        never of the current lengths, and every request-dependent value
        (``kv_lens`` and the metadata's block tables) is a device tensor
        refreshed by metadata ``prepare()`` outside the captured region. No
        ``.item()``/``.tolist()``, no per-request Python loop, no
        host->device copy, and no data-dependent branch runs on this path, so
        a captured decode graph replays correctly as lengths grow and slots
        are reused.

        ``kv_lens[i]`` is the request's visible length *including* the token
        being decoded, so the new token's position is ``kv_lens[i] - 1``; it
        must be the same prepare()-refreshed lengths the metadata carries,
        sliced to the generation rows. Positions at or beyond a request's
        ``kv_lens`` gather page-0 garbage in the indexer prefix; the backend
        masks them by replacement and the indexer's own validity masks exclude
        them from pools, selection, and the tail. The attention core never
        gathers the latent at all: the backend reads the paged pool directly
        through its storage row view derived from the metadata, and only
        selected (valid) positions are translated into row ids -- sentinels
        stay ``-1``. There is no empty-row assertion here: it would force a
        host sync, which is illegal under CUDA-graph capture -- and a decode
        query is always covered by construction, either by the
        always-selected tail (``visible % kpool != 0``) or by the final
        complete pool, whose last member *is* the query position
        (``visible % kpool == 0``).
        """
        batch = hidden_states.shape[0]
        device = hidden_states.device
        positions = (kv_lens - 1).unsqueeze(1)  # [B, 1]
        self.attn_backend.append_paged_state(
            self.project_latent(hidden_states).unsqueeze(1),
            self.indexer.packed_state(hidden_states).unsqueeze(1),
            positions,
            metadata,
        )
        packed_prefix, num_pools = self.attn_backend.gather_packed_prefix(metadata)

        pool_keys, pool_last, pool_valid = self.indexer.build_pools(
            packed_prefix, kv_lens, num_pools
        )
        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        query = self.q_b_proj(q_resid).view(batch, self.num_heads, self.qk_head_dim)
        token_request = torch.arange(batch, device=device)
        topk = self.indexer.select(
            q_resid, hidden_states, pool_keys, pool_last, pool_valid, token_request, kv_lens - 1
        )
        _glm5_pool_probe(self.layer_idx, topk, kv_lens)
        q_latent = self.absorb_query(query)
        out_latent = self.attn_backend.forward(
            q_latent,
            None,
            None,
            metadata,
            AttentionForwardArgs(
                attention_input_type=AttentionInputType.generation_only,
                sparse_backend_args=SparseBackendForwardArgs(topk_indices=topk),
            ),
        )
        return self.project_output_latent(out_latent)


# ---------------------------------------------------------------------------
# Heterogeneous request state
# ---------------------------------------------------------------------------


def _glm5_tensor_digest(t: torch.Tensor) -> Dict[str, Any]:
    """sha256 of the raw bytes plus fp64 invariants of one state slice.

    The sha256 is the bitwise comparator between two runs (baseline vs
    CUDA-graph/overlap); the invariants are the human-readable summary. Both
    are computed from the tensor as-is — no dtype cast before hashing — so
    equal digests mean bit-identical state.
    """
    import hashlib

    flat = t.detach().reshape(-1)
    if flat.numel() == 0:
        return {"sha256": hashlib.sha256(b"").hexdigest(), "numel": 0}
    if not flat.is_contiguous():
        flat = flat.contiguous()
    digest = hashlib.sha256(flat.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    f64 = flat.to(torch.float64)
    return {
        "sha256": digest,
        "numel": int(flat.numel()),
        "sum": float(f64.sum()),
        "abs_sum": float(f64.abs().sum()),
        "max": float(f64.max()),
        "min": float(f64.min()),
    }


def _glm5_export_state_digest(
    metadata, manager, *, batch, num_contexts, lens, cached, request_ids, pages
) -> None:
    """Per-step runtime-state digest export, enabled by GLM53_STATE_DIGEST_DIR.

    Diagnostic evidence path (default: env unset, single dict lookup, no-op).
    When enabled, every pipeline rank appends one JSON line per pure-decode
    ``prepare()`` call digesting the live runtime state the upcoming step
    consumes: each local KDA layer's recurrent (ssm) and four-tap convolution
    slot plus each local sparse layer's latent and indexer pages over the
    cached positions. ``prepare()`` runs outside any captured region in both
    the eager and the CUDA-graph runtime, and the digest synchronizes the
    device first, so the observed state is the completed result of every
    previously issued step — the same observation point in baseline and
    graph/overlap configurations. Prefill batches and CUDA-graph warmup/
    padding dummy batches are skipped: a slot that has not been written yet
    holds uninitialized memory whose digest is meaningless noise.
    """
    import json
    import os

    digest_dir = os.environ.get("GLM53_STATE_DIGEST_DIR")
    if not digest_dir or batch <= 0 or num_contexts > 0:
        return
    from ..modules.mamba.mamba2_metadata import CUDA_GRAPH_DUMMY_REQUEST_ID

    max_draft = getattr(manager, "speculative_num_draft_tokens", 0) or 0
    ids = [int(r) for r in list(request_ids)[:batch]]
    if any(
        CUDA_GRAPH_DUMMY_REQUEST_ID - max_draft <= rid <= CUDA_GRAPH_DUMMY_REQUEST_ID for rid in ids
    ):
        return

    torch.cuda.synchronize()
    slots = metadata.state_indices[:batch].to(torch.long)
    layers: Dict[str, Any] = {}

    for layer_id, is_mamba in enumerate(getattr(manager, "_mamba_layer_mask", []) or []):
        if not is_mamba:
            continue
        try:
            conv = manager.get_conv_states(layer_id)
            ssm = manager.get_ssm_states(layer_id)
        except (KeyError, IndexError):
            continue  # not this pipeline rank's layer
        if conv is None or ssm is None:
            continue
        layers[f"kda{layer_id}"] = {
            "conv": _glm5_tensor_digest(conv.index_select(0, slots)),
            "ssm": _glm5_tensor_digest(ssm.index_select(0, slots)),
        }

    tokens_per_block = int(getattr(manager, "tokens_per_block", 0) or 0)
    for layer_id in sorted(getattr(manager, "sparse_layer_ids", ()) or ()):
        try:
            latent = manager.get_latent_state_buffer(layer_id)
            index = manager.get_index_state_buffer(layer_id)
        except (KeyError, IndexError, ValueError):
            continue  # not this pipeline rank's layer
        if latent is None or index is None or tokens_per_block <= 0:
            continue
        lat3, idx3 = latent[:, :, 0, :], index[:, :, 0, :]
        per_req: Dict[str, List[Dict[str, Any]]] = {"latent": [], "index": []}
        for row in range(batch):
            n = int(cached[row])
            page_row = pages[row] if row < len(pages) else []
            if n <= 0 or not page_row or n > len(page_row) * tokens_per_block:
                per_req["latent"].append({"error": f"cached={n} pages={len(page_row)}"})
                per_req["index"].append({"error": f"cached={n} pages={len(page_row)}"})
                continue
            pos = torch.arange(n, device=lat3.device)
            tbl = torch.as_tensor(page_row, dtype=torch.long, device=lat3.device)
            pg = tbl[pos // tokens_per_block]
            off = pos % tokens_per_block
            per_req["latent"].append(_glm5_tensor_digest(lat3[pg, off]))
            per_req["index"].append(_glm5_tensor_digest(idx3[pg, off]))
        layers[f"sparse{layer_id}"] = per_req

    try:
        from tensorrt_llm._utils import mpi_rank

        rank = int(mpi_rank())
    except Exception:
        rank = 0
    record = {
        "num_contexts": int(num_contexts),
        "batch": int(batch),
        "request_ids": ids,
        "seq_lens": [int(x) for x in lens],
        "cached": [int(c) for c in cached],
        "mamba_slots": [int(s) for s in slots.cpu()],
        "pages": [[int(p) for p in row] for row in pages],
        "layers": layers,
    }
    with open(os.path.join(digest_dir, f"rank{rank}.jsonl"), "a") as fh:
        fh.write(json.dumps(record) + "\n")


_GLM5_STREAM_PROBE_STATE: Dict[str, Any] = {"first_layer": None, "step": 0, "active": False}


def _glm5_probe_append(record: Dict[str, Any]) -> None:
    """Append one probe record to this rank's JSONL under the probe dir."""
    import json
    import os

    try:
        from tensorrt_llm._utils import mpi_rank

        rank = int(mpi_rank())
    except Exception:
        rank = 0
    path = os.path.join(os.environ["GLM53_STREAM_PROBE_DIR"], f"rank{rank}.jsonl")
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")


def _glm5_stream_probe(
    layer_idx: int, phase: str, streams: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
) -> None:
    """Periodic decode-time HC/Sinkhorn health probe (``GLM53_STREAM_PROBE_DIR``).

    Diagnostic evidence path for the long-horizon state canary (default: env
    unset, a single dict lookup, no-op). Every ``GLM53_STREAM_PROBE_EVERY``-th
    decode forward, each pipeline rank appends one JSON line per local layer
    recording the four hyper-connection streams' per-stream sums and L2 norms,
    the Sinkhorn mixing matrix's row/column sums (doubly-stochastic health,
    finite by contract), and the ``post`` placement weights. The step counter
    keys on the rank's first local layer, so one flag covers every layer of
    the same forward. Read-only fp64 copies in eager mode; never active
    inside CUDA-graph capture (guarded below), and graph *replay* does not
    re-run Python at all — the canary driver only sets the env for the eager
    baseline configuration.
    """
    import os

    if not os.environ.get("GLM53_STREAM_PROBE_DIR") or phase != "decode":
        return
    if torch.cuda.is_current_stream_capturing():
        return
    state = _GLM5_STREAM_PROBE_STATE
    if state["first_layer"] is None:
        state["first_layer"] = layer_idx
    if layer_idx == state["first_layer"]:
        state["step"] += 1
        every = max(1, int(os.environ.get("GLM53_STREAM_PROBE_EVERY", "128")))
        state["active"] = (state["step"] - 1) % every == 0
    if not state["active"]:
        return
    f_streams = streams.detach().to(torch.float64)  # [tokens, mult, hidden]
    norms = f_streams.norm(dim=-1)
    sums = f_streams.sum(dim=-1)
    f_comb = comb.detach().to(torch.float64)  # [tokens, mult, mult]
    row_sums = f_comb.sum(dim=-1)
    col_sums = f_comb.sum(dim=-2)
    f_post = post.detach().to(torch.float64)
    _glm5_probe_append(
        {
            "kind": "hc",
            "probe_step": state["step"],
            "layer": int(layer_idx),
            "tokens": int(f_streams.shape[0]),
            "streams_finite": bool(torch.isfinite(f_streams).all()),
            "stream_l2_mean": [float(x) for x in norms.mean(dim=0)],
            "stream_l2_max": [float(x) for x in norms.max(dim=0).values],
            "stream_sum_mean": [float(x) for x in sums.mean(dim=0)],
            "comb_finite": bool(torch.isfinite(f_comb).all()),
            "comb_row_sum_min": float(row_sums.min()),
            "comb_row_sum_max": float(row_sums.max()),
            "comb_col_sum_min": float(col_sums.min()),
            "comb_col_sum_max": float(col_sums.max()),
            "post_finite": bool(torch.isfinite(f_post).all()),
            "post_min": float(f_post.min()),
            "post_max": float(f_post.max()),
        }
    )


def _glm5_pool_probe(layer_idx: int, topk: torch.Tensor, kv_lens: torch.Tensor) -> None:
    """Pool-selection validity probe for probed decode steps.

    Same env gate and step cadence as :func:`_glm5_stream_probe` (which runs
    earlier in the same layer and owns the counter). Records, per request row,
    the visible length, the sentinel count, the non-sentinel min/max, whether
    every non-sentinel expanded index lies in ``[0, kv_len)``, and whether the
    row selected at least one real position — the runtime counterpart of the
    indexer's expansion/always-tail/``-1`` contract.
    """
    import os

    if not os.environ.get("GLM53_STREAM_PROBE_DIR") or not _GLM5_STREAM_PROBE_STATE["active"]:
        return
    if torch.cuda.is_current_stream_capturing():
        return
    lens = kv_lens.detach().reshape(-1).to(torch.long)
    idx = topk.detach().to(torch.long)
    sentinel = idx < 0
    in_range = (idx >= 0) & (idx < lens.view(-1, 1))
    nonsent_max = idx.max(dim=-1).values
    nonsent_min = idx.masked_fill(sentinel, torch.iinfo(torch.long).max).min(dim=-1).values
    _glm5_probe_append(
        {
            "kind": "pool",
            "probe_step": _GLM5_STREAM_PROBE_STATE["step"],
            "layer": int(layer_idx),
            "width": int(idx.shape[-1]),
            "kv_lens": [int(x) for x in lens],
            "sentinel_counts": [int(x) for x in sentinel.sum(dim=-1)],
            "nonsentinel_min": [int(x) for x in nonsent_min],
            "nonsentinel_max": [int(x) for x in nonsent_max],
            "all_in_range": bool((in_range | sentinel).all()),
            "rows_covered": [bool(x) for x in (~sentinel).any(dim=-1)],
        }
    )


def glm5_next_mamba_metadata_cls():
    """Return the model's ``Mamba2Metadata`` subclass (lazily imported).

    CUDA graphs replay captured kernels over fixed buffer addresses; nothing
    Python re-runs at replay. Every per-step request-derived value the sparse
    layers consume in decode must therefore live in persistent device buffers
    refreshed by ``prepare()`` -- which the engine (and the CUDA-graph runner,
    before every replay) calls outside the captured region. This subclass adds
    exactly those buffers to the mamba metadata the hybrid manager already
    attaches:

    * ``glm_block_tables`` -- ``[max_batch, max_blocks_per_seq]`` base-slot
      page ids (pinned staging + one async H2D per step);
    * ``glm_kv_lens`` -- ``[max_batch]`` per-request visible lengths
      (cached + this step's tokens), same staging pattern;
    * ``glm_cached_lens_host`` / ``glm_ctx_cu_seqlens`` -- plain host values
      for the prefill path, which is never captured (decode-only graphs).

    The buffer width is fixed at first use from the manager's own
    ``max_blocks_per_seq`` so captured gathers can never need a wider table;
    a mid-run widening request is a hard error rather than a silent
    reallocation that stale graphs would keep reading.
    """
    from ..modules.mamba.mamba2_metadata import Mamba2Metadata

    class Glm5NextMamba2Metadata(Mamba2Metadata):
        def __init__(self, max_batch_size: int, chunk_size: int):
            super().__init__(max_batch_size, chunk_size)
            from tensorrt_llm._utils import prefer_pinned

            self._glm_pin = prefer_pinned()
            self.glm_block_tables: Optional[torch.Tensor] = None
            self._glm_block_tables_cpu: Optional[torch.Tensor] = None
            self.glm_kv_lens = torch.zeros(max_batch_size, dtype=torch.long, device="cuda")
            self._glm_kv_lens_cpu = torch.zeros(
                max_batch_size, dtype=torch.long, pin_memory=self._glm_pin
            )
            self.glm_cached_lens_host: List[int] = []
            self.glm_ctx_cu_seqlens: List[int] = [0]

        def _glm_ensure_tables(self, width: int) -> None:
            width = max(1, int(width))
            if self.glm_block_tables is None:
                self.glm_block_tables = torch.zeros(
                    self.max_batch_size, width, dtype=torch.long, device="cuda"
                )
                self._glm_block_tables_cpu = torch.zeros(
                    self.max_batch_size, width, dtype=torch.long, pin_memory=self._glm_pin
                )
            elif self.glm_block_tables.shape[1] < width:
                raise RuntimeError(
                    "glm5_next block-table buffer would need to grow from "
                    f"{self.glm_block_tables.shape[1]} to {width} pages mid-run; "
                    "captured CUDA graphs would keep reading the old buffer"
                )

        def prepare(self, attn_metadata) -> None:
            super().prepare(attn_metadata)
            manager = attn_metadata.kv_cache_manager
            kv_params = attn_metadata.kv_cache_params
            request_ids = attn_metadata.request_ids
            if (
                manager is None
                or not hasattr(manager, "get_batch_slot_tables")
                or kv_params is None
                or kv_params.num_cached_tokens_per_seq is None
                or request_ids is None
            ):
                return

            batch = attn_metadata.seq_lens.shape[0]
            num_contexts = int(attn_metadata.num_contexts)
            lens = [int(x) for x in attn_metadata.seq_lens[:batch]]
            cached_src = kv_params.num_cached_tokens_per_seq
            if isinstance(cached_src, torch.Tensor):
                cached = [int(x) for x in cached_src[:batch]]
            else:
                cached = [int(cached_src[i]) for i in range(batch)]

            self.glm_cached_lens_host = cached
            cu = [0]
            for length in lens[:num_contexts]:
                cu.append(cu[-1] + length)
            self.glm_ctx_cu_seqlens = cu

            self._glm_kv_lens_cpu[:batch].copy_(
                torch.as_tensor([c + n for c, n in zip(cached, lens)], dtype=torch.long)
            )
            self.glm_kv_lens[:batch].copy_(self._glm_kv_lens_cpu[:batch], non_blocking=True)

            width = int(getattr(manager, "max_blocks_per_seq", 0)) or 1
            self._glm_ensure_tables(width)
            pages = manager.get_batch_slot_tables(list(request_ids)[:batch])
            staging = self._glm_block_tables_cpu
            staging[:batch].zero_()
            for row, page_ids in enumerate(pages):
                if page_ids:
                    staging[row, : len(page_ids)].copy_(torch.as_tensor(page_ids, dtype=torch.long))
            self.glm_block_tables[:batch].copy_(staging[:batch], non_blocking=True)

            _glm5_export_state_digest(
                self,
                manager,
                batch=batch,
                num_contexts=num_contexts,
                lens=lens,
                cached=cached,
                request_ids=request_ids,
                pages=pages,
            )

    return Glm5NextMamba2Metadata


def glm5_next_cache_manager_cls():
    """Return the model's ``KVCacheManagerV2`` subclass.

    Imported lazily: ``mamba_cache_manager`` pulls in the pyexecutor, which is
    a much heavier import than model discovery needs.
    """
    from ...runtime.kv_cache_manager_v2 import BufferConfig
    from ..pyexecutor.kv_cache_manager_v2 import Role
    from ..pyexecutor.mamba_cache_manager import MambaHybridCacheManagerV2

    class Glm5NextCacheManager(MambaHybridCacheManagerV2):
        """One KVCacheManagerV2 lifecycle for all three kinds of GLM state.

        The three state families have genuinely different shapes and are kept
        that way -- none is padded into a faux common KV tensor:

        * linear-attention layers: the recurrent accumulator and the four-tap
          convolution history, carried by the inherited Mamba side with
          ``conv_state_layout='q_k_v'`` (the convolution's ``[q | k | v]``
          section order);
        * sparse-attention layers: the ``kv_lora_rank``-wide compressed latent,
          carried by the standard attention pages with ``SELFKONLY``;
        * sparse-attention layers again: the indexer's packed ``[k | gate]``
          state, added here as one extra ``Role.INDEX_KEY`` buffer per sparse
          layer.

        The extra buffer is registered through the base class's own
        ``_extra_buffers_per_layer`` hook, so allocation, block reuse, slot
        release, and disaggregated bookkeeping keep working. A second manager
        would duplicate request ownership and break exactly those lifecycles.
        """

        def __init__(
            self, *args, sparse_layer_ids: Sequence[int] = (), index_state_dim: int = 0, **kwargs
        ):
            # Set before super().__init__: the base _build_base_config calls
            # _extra_buffers_per_layer, which reads both of these.
            self.sparse_layer_ids = sorted(int(i) for i in sparse_layer_ids)
            self.index_state_dim = int(index_state_dim)
            if self.sparse_layer_ids and self.index_state_dim <= 0:
                raise ValueError(
                    "glm5_next sparse layers need a positive index_state_dim "
                    f"(got {self.index_state_dim})"
                )
            super().__init__(*args, **kwargs)

        @property
        def mamba_metadata_cls(self):
            """Metadata subclass carrying the paged-table/length buffers that
            the sparse decode reads inside CUDA graphs; refreshed by
            ``prepare()`` outside the captured region."""
            return glm5_next_mamba_metadata_cls()

        def _extra_buffers_per_layer(self, *, tokens_per_block):
            """One ``Role.INDEX_KEY`` buffer per sparse layer, keyed by local id."""
            elem_bytes = torch.tensor([], dtype=torch.bfloat16).element_size()
            size_per_block = self.index_state_dim * elem_bytes * tokens_per_block
            return {
                self.layer_offsets[layer_id]: [
                    BufferConfig(role=Role.INDEX_KEY, size=size_per_block)
                ]
                for layer_id in self.sparse_layer_ids
                if layer_id in self.layer_offsets
            }

        def get_index_state_buffer(self, layer_idx: int) -> Optional[torch.Tensor]:
            """Paged indexer state for ``layer_idx``, NHD-shaped."""
            return self.get_index_k_buffer(
                layer_idx,
                num_heads=1,
                head_dim=self.index_state_dim,
                dtype=torch.bfloat16,
                kv_layout="NHD",
            )

        def _sparse_pool_id(self) -> int:
            """The single V2 layer-group id that owns every sparse layer.

            Base page indices are per layer group, so one block table can
            address both the latent and the indexer views only because all
            sparse layers -- whose KEY and INDEX_KEY buffers share each
            layer's group -- resolve to one group. Asserted, not assumed:
            a future geometry change that splits the group must fail here
            rather than silently interleave two slot spaces.
            """
            pools = {
                self.layer_to_pool_mapping_dict[self.layer_offsets[layer_id]]
                for layer_id in self.sparse_layer_ids
                if layer_id in self.layer_offsets
            }
            if len(pools) != 1:
                raise ValueError(
                    f"glm5_next sparse layers span V2 layer groups {sorted(pools)}; "
                    "the slot-indexed latent/index views require a single group"
                )
            return pools.pop()

        def get_batch_slot_tables(self, request_ids: Sequence[int]) -> List[List[int]]:
            """Raw base-slot IDs per request, for the slot-major state views.

            ``get_batch_cache_indices`` is scaled for V2's flattened
            per-layer page views -- it returns ``base * scale // kv_factor``
            (scale is the coalesced buffers-per-slot count; 11 on the real
            checkpoint) -- so feeding its output to the slot-major views from
            :meth:`get_latent_state_buffer` / :meth:`get_index_state_buffer`
            addresses the wrong slots and eventually runs past the pool.
            Those views are indexed by the *base* page id itself, so this
            accessor requests the identity conversion from the same V2
            bookkeeping (``is_kv_aggregate=False, index_scale=1``).

            A pipeline-parallel rank whose local slice holds no sparse layer
            (e.g. the first three GLM layers are all linear attention) has no
            sparse pool at all; it gets empty rows, which no layer on that
            rank ever reads.
            """
            if not any(layer_id in self.layer_offsets for layer_id in self.sparse_layer_ids):
                return [[] for _ in request_ids]
            return self._get_batch_cache_indices_by_pool_id(
                list(request_ids),
                pool_id=self._sparse_pool_id(),
                is_kv_aggregate=False,
                index_scale=1,
            )

        def get_latent_state_buffer(self, layer_idx: int) -> Optional[torch.Tensor]:
            """Paged latent state for ``layer_idx``, addressed by **slot** id.

            ``get_buffers`` hands back a view whose dim-0 stride is a *single*
            page, but V2 coalesces every layer's ``Role.KEY`` buffer into one
            pool and starts layer ``L``'s view ``L`` pages into it. Indexing two
            layers' views with the same block id therefore makes them overlap
            almost entirely -- measured on this checkpoint, eleven sparse layers
            share a 2046-page pool and each layer's page ``p`` is the next
            layer's page ``p - 1``.

            V2's own callers never index that view with a raw block id:
            ``_get_batch_cache_indices_by_pool_id`` multiplies every base page
            index by ``get_layer_page_index_scale(layer_idx)`` first. Folding
            that scale into the *view* rather than into each caller's block
            table is exactly what :meth:`get_index_k_buffer` already does for
            ``Role.INDEX_KEY`` (``[slots, scale, ...][:, 0]``), and it is the
            only convention under which one block table can address both pools:
            the coalesced INDEX_KEY view is expressible *only* slot-indexed.

            Returns ``[num_slots, tokens_per_block, num_kv_heads, head_dim]``,
            the same rank as :meth:`get_index_state_buffer`, so a caller
            decomposes a position into one ``(slot, within_slot)`` pair and uses
            it against both.
            """
            pages = self.get_buffers(layer_idx)
            if pages is None:
                return None
            flat = pages[:, 0]  # [pages, tokens, heads, dim]
            converter = self.impl.get_page_index_converter(self.layer_offsets[layer_idx], Role.KEY)
            # ``scale`` is the buffers-per-slot count of the coalesced pool;
            # ``within_slot`` is where this layer's buffer sits inside a slot.
            scale, within_slot = int(converter.scale), int(converter.layer_offset)
            kv_factor = int(self.kv_factor)
            if scale <= kv_factor:
                return flat
            if scale % kv_factor:
                raise ValueError(
                    f"glm5_next layer {layer_idx}: page-index scale {scale} is not a "
                    f"multiple of kv_factor {kv_factor}, so the latent pool has no "
                    "slot-major view"
                )
            # ``get_buffers`` already folds the kv_factor axis in, so its dim 0
            # counts kv-aggregate pages and the slot stride is the scale in those
            # units. Slot counts are derived the way get_index_k_buffer derives
            # them, so both pools expose the same slot space.
            stride = scale // kv_factor
            slots = (flat.shape[0] + within_slot) // stride
            # Slot s lives at pool page ``within_slot + s * stride``; the last one
            # stays in range because a layer's offset inside a coalesced slot is
            # always smaller than the slot itself.
            if within_slot >= stride or (slots - 1) * stride >= flat.shape[0]:
                raise ValueError(
                    f"glm5_next layer {layer_idx}: {slots} slots at stride {stride} do "
                    f"not fit {flat.shape[0]} pages from slot offset {within_slot}"
                )
            return torch.as_strided(
                flat,
                size=(slots, *flat.shape[1:]),
                stride=(stride * flat.stride(0), *flat.stride()[1:]),
                storage_offset=flat.storage_offset(),
            )

    return Glm5NextCacheManager


# ---------------------------------------------------------------------------
# Feed-forward
# ---------------------------------------------------------------------------


def clamped_swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """The source's SwiGLU, whose clamp is **asymmetric**.

    ``gate`` is clamped only from *above* at ``+limit``; ``up`` is clamped on
    both sides. Ordinary SwiGLU is the same function everywhere inside the
    clamp, so only an activation that actually crosses ``limit`` can tell them
    apart -- which is why the tests drive one deliberately past it rather than
    trusting a typical-magnitude comparison.
    """
    return torch.nn.functional.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)


class Glm5NextMLP(nn.Module):
    """Dense clamped-SwiGLU MLP, also used as the always-active shared expert."""

    def __init__(
        self,
        config: PretrainedConfig,
        intermediate_size: Optional[int] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        hidden = int(config.hidden_size)
        inter = int(config.intermediate_size if intermediate_size is None else intermediate_size)
        self.swiglu_limit = float(config.swiglu_limit)
        self.gate_proj = nn.Linear(hidden, inter, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden, inter, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(inter, hidden, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor, phase: str = "prefill") -> torch.Tensor:
        # ``phase`` mirrors Glm5NextMoE.forward so the decoder layer can pass
        # it uniformly; a dense MLP is phase-independent.
        return self.down_proj(clamped_swiglu(self.gate_proj(x), self.up_proj(x), self.swiglu_limit))


class Glm5NextTopkRouter(nn.Module):
    """noaux_tc sigmoid router, FP32 end to end.

    Two orderings here are load-bearing and easy to invert:

    * the correction bias participates in **selection only** -- the returned
      weights are gathered from the *uncorrected* sigmoid scores;
    * normalization happens before ``routed_scaling_factor``, so the weights sum
      to the scaling factor rather than to one.

    ``n_group == topk_group == 1`` makes the group-limited stage degenerate (the
    single group is always selected), so it is asserted rather than implemented.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_experts = int(config.n_routed_experts)
        self.top_k = int(config.num_experts_per_tok)
        self.routed_scaling_factor = float(config.routed_scaling_factor)
        self.norm_topk_prob = bool(config.norm_topk_prob)
        n_group = int(getattr(config, "n_group", 1) or 1)
        topk_group = int(getattr(config, "topk_group", 1) or 1)
        if n_group != 1 or topk_group != 1:
            raise ValueError(
                f"glm5_next was validated with degenerate group routing; got "
                f"n_group={n_group} topk_group={topk_group}, which needs a real "
                "group-limited selection stage this bring-up does not implement"
            )
        if str(getattr(config, "scoring_func", "sigmoid")) != "sigmoid":
            raise ValueError(f"glm5_next expects sigmoid scoring, got {config.scoring_func!r}")

        self.weight = nn.Parameter(
            torch.zeros(self.num_experts, self.hidden_size, dtype=torch.float32)
        )
        # Kept FP32 unconditionally: the bias sits around magnitude ~10 while the
        # sigmoid scores it corrects are O(1e-2), so ranking turns on inter-expert
        # gaps of 4e-5 - 6e-4. bf16 resolution at that magnitude is ~1e-2, three
        # orders of magnitude too coarse -- it silently changes the top-8 while
        # every aggregate check still passes.
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(self.num_experts, dtype=torch.float32)
        )

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """FP32 router logits, the input both selection paths consume."""
        flat = x.reshape(-1, self.hidden_size)
        return torch.nn.functional.linear(flat.float(), self.weight.float())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = self.logits(x)
        scores = router_logits.sigmoid()
        topk_indices = torch.topk(
            scores + self.e_score_correction_bias.float(), k=self.top_k, dim=-1, sorted=False
        )[1]
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        return router_logits, topk_weights * self.routed_scaling_factor, topk_indices


class Glm5NextMoE(nn.Module):
    """Routed experts plus one always-active shared expert.

    Two execution modes share the router and shared-expert semantics:

    * **Production** (``quantized=True`` and a ``model_config`` supplied): the
      routed experts are a fused-MoE layer built through ``create_moe`` -- the
      one selection entry point -- with the DeepSeek noaux_tc routing method
      (identical math to :class:`Glm5NextTopkRouter`, including the 1e-20
      normalization eps, selection on bias-corrected scores, weights gathered
      from the uncorrected sigmoid, and post-normalization
      ``routed_scaling_factor``) and the DSV4-style uniform
      ``swiglu_limit_scalar``. On this checkpoint (FP8 block scales, SM100)
      the resolver lands on ``TRTLLMGenFusedMoE`` whose
      ``trtllm::fp8_block_scale_moe_runner`` consumes the clamp limit as
      ``gemm1_clamp_limit``; the ``AUTO`` ``moe_backend`` default resolves to
      ``TRTLLM`` for exactly this quant/SM pair in
      ``ModelConfig.resolve_moe_backend``. Cutlass cannot serve FP8 block
      scales on SM100 (its kernel stops at SM90/SM120) and DeepGemm requires
      the absent ``deep_gemm`` wheel, so a pinned non-TRTLLM backend fails
      resolution loudly rather than silently running something else.

    * **Diagnostic** (everything else, incl. direct test construction without
      a ``model_config``): the Stage-1/2 verified native-torch gather/scatter
      over stacked ``[E, 2I, H]`` / ``[E, H, I]`` expert weights, gate rows
      first. This is the independent reference rung the production tests
      compare against.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        dtype: torch.dtype = torch.bfloat16,
        quantized: bool = False,
        model_config: Optional[ModelConfig] = None,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        hidden = int(config.hidden_size)
        self.hidden_size = hidden
        self.moe_intermediate_size = int(config.moe_intermediate_size)
        self.num_experts = int(config.n_routed_experts)
        self.swiglu_limit = float(config.swiglu_limit)
        self.quantized = bool(quantized)
        self.gate = Glm5NextTopkRouter(config)
        inter = self.moe_intermediate_size
        self.experts: Optional[nn.Module] = None
        self.moe_backend_name: Optional[str] = None
        # Set on the production path below; the diagnostic path is single-rank.
        self.mapping = None
        self.moe_all_reduce = None

        if self.quantized and model_config is not None:
            # The fused routing kernel always normalizes the gathered top-8
            # weights; a checkpoint that disabled normalization would need the
            # diagnostic path instead of a silent semantic change.
            if not self.gate.norm_topk_prob:
                raise ValueError(
                    "glm5_next production MoE requires norm_topk_prob=True "
                    "(the DeepSeek noaux_tc routing kernel always normalizes)"
                )
            from ..moe.fused_moe.create_moe import create_moe
            from ..moe.fused_moe.routing import DeepSeekV3MoeRoutingMethod

            routing = DeepSeekV3MoeRoutingMethod(
                top_k=self.gate.top_k,
                n_group=1,
                topk_group=1,
                routed_scaling_factor=self.gate.routed_scaling_factor,
                # Fetched per call so the tensor follows the parameter through
                # to_empty/materialization onto its final device.
                callable_e_score_correction_bias=lambda: self.gate.e_score_correction_bias,
            )
            # The experts themselves are uniformly block-FP8; the checkpoint's
            # 1509 exclusion patterns concern *other* modules and are resolved
            # by the model's own quant plan, so the layer-scoped config the
            # fused backend sees carries only the algorithm and block size.
            experts_quant = QuantConfig(
                quant_algo=QuantAlgo.FP8_BLOCK_SCALES,
                group_size=128,
            )
            self.experts = create_moe(
                routing_method=routing,
                num_experts=self.num_experts,
                hidden_size=hidden,
                intermediate_size=inter,
                dtype=dtype,
                reduce_results=False,
                model_config=model_config,
                override_quant_config=experts_quant,
                layer_idx=layer_idx,
                swiglu_limit_scalar=self.swiglu_limit,
            )
            backend = getattr(self.experts, "backend", self.experts)
            self.moe_backend_name = type(backend).__name__
            # Four-rank composition (plan Decision F): the fused layer runs
            # with reduce_results=False, so its output is a rank partial --
            # a K-dim partial per expert in the TP4 layout (moe_tp_size=4),
            # a local-expert partial sum in the TP4/EP4 layout (moe_ep_size=4)
            # -- and this module owns the exactly-one reduction that combines
            # it (together with the TP4-layout shared-expert partial; the
            # replicated EP4-layout shared output is added *after* the combine
            # and never reduced). DeepSeek-V3 is the in-tree precedent.
            self.mapping = model_config.mapping
            self.moe_all_reduce = None
            if self.mapping.tp_size > 1:
                from ..distributed import AllReduce, AllReduceStrategy

                # NCCL rather than AUTO: the AUTO strategy routes through the
                # AllReduce autotuner, whose selected tactic raced at decode on
                # this model's TP4 collective schedule and produced
                # intermittent NaN logits (masked by any added host sync). See
                # the indexer's score_all_reduce and the swap's row Linears --
                # every TP collective in this model is pinned to NCCL so
                # correctness does not depend on the ``TLLM_DISABLE_ALLREDUCE_
                # AUTOTUNE`` env escape hatch. Bring-up is parity-first, so the
                # autotuner's perf tuning is not needed here.
                self.moe_all_reduce = AllReduce(
                    mapping=self.mapping,
                    strategy=AllReduceStrategy.NCCL,
                )
        else:
            # The 288 routed experts dominate the checkpoint: keeping them in
            # their published e4m3 form rather than dequantizing halves the
            # resident model (304 GB of 328 GB lives here) *and* is the
            # numerically closer path, since the source runs the same
            # block-scaled arithmetic.
            weight_dtype = torch.float8_e4m3fn if self.quantized else dtype
            # The 288 experts are the 304 GB bulk of the checkpoint. Build them
            # with ``torch.empty`` (not ``torch.zeros``): ``MetaInitMode``
            # intercepts ``aten.empty`` to ``meta`` but leaves ``torch.zeros``
            # real, so zeros here real-allocate the whole MoE in host RAM on
            # every PP rank during construction -- the transient that OOM-kills
            # the PP=8 load before any weight is read. ``load_weights`` runs
            # ``to_empty`` + fills every expert from the checkpoint, so the
            # construction values are discarded either way; empty just keeps
            # the meta-init path memory-free.
            self.gate_up_proj = nn.Parameter(
                torch.empty(self.num_experts, 2 * inter, hidden, dtype=weight_dtype),
                requires_grad=False,
            )
            self.down_proj = nn.Parameter(
                torch.empty(self.num_experts, hidden, inter, dtype=weight_dtype),
                requires_grad=False,
            )
            if self.quantized:
                gu_scale = glm5_next_fp8_scale_shape(2 * inter, hidden)
                down_scale = glm5_next_fp8_scale_shape(hidden, inter)
                self.gate_up_proj_scale = nn.Parameter(
                    torch.empty(self.num_experts, *gu_scale, dtype=torch.float32),
                    requires_grad=False,
                )
                self.down_proj_scale = nn.Parameter(
                    torch.empty(self.num_experts, *down_scale, dtype=torch.float32),
                    requires_grad=False,
                )
        self.shared_experts = Glm5NextMLP(
            config,
            intermediate_size=self.moe_intermediate_size * int(config.n_shared_experts),
            dtype=dtype,
        )

    def _expert(self, x: torch.Tensor, expert: int) -> torch.Tensor:
        if self.quantized:
            fused = glm5_next_block_fp8_matmul(
                x, self.gate_up_proj[expert], self.gate_up_proj_scale[expert]
            )
            gate, up = fused.chunk(2, dim=-1)
            return glm5_next_block_fp8_matmul(
                clamped_swiglu(gate, up, self.swiglu_limit),
                self.down_proj[expert],
                self.down_proj_scale[expert],
            )
        gate, up = torch.nn.functional.linear(x, self.gate_up_proj[expert]).chunk(2, dim=-1)
        return torch.nn.functional.linear(
            clamped_swiglu(gate, up, self.swiglu_limit), self.down_proj[expert]
        )

    def forward(self, x: torch.Tensor, phase: str = "prefill") -> torch.Tensor:
        flat = x.reshape(-1, self.hidden_size)
        if self.experts is not None:
            # Production: the fused layer routes internally from the FP32
            # logits (same noaux_tc math as the diagnostic path) and runs
            # FC1 -> clamped SwiGLU -> FC2 in one backend call. ``phase`` is
            # irrelevant here: one code path serves prefill and decode, and it
            # contains no host-dependent branching, so decode stays
            # CUDA-graph-capturable.
            routed = self.experts(flat, self.gate.logits(flat))
            if self.moe_all_reduce is not None:
                if self.mapping.moe_tp_size > 1:
                    # TP4 layout: routed and shared are both rank partials over
                    # the intermediate dim -- sum them, then exactly one
                    # reduction covers the whole MoE branch.
                    mixed = routed + self.shared_experts(flat)
                    return self.moe_all_reduce(mixed).view_as(x)
                # TP4/EP4 layout: the all-reduce *is* the EP combine of the
                # per-rank local-expert partial sums; the replicated shared
                # contribution is added once after it and never reduced.
                routed = self.moe_all_reduce(routed)
                return routed.view_as(x) + self.shared_experts(x)
            return routed.view_as(x) + self.shared_experts(x)
        _, topk_weights, topk_indices = self.gate(flat)
        if phase == "decode":
            routed = self._routed_decode(flat, topk_weights, topk_indices)
        else:
            routed = self._routed_prefill(flat, topk_weights, topk_indices)
        return routed.view_as(x) + self.shared_experts(x)

    def _routed_prefill(
        self, flat: torch.Tensor, topk_weights: torch.Tensor, topk_indices: torch.Tensor
    ) -> torch.Tensor:
        """Unique-expert grouping: one matmul per expert actually selected.

        Efficient for many tokens, but ``torch.unique(...).tolist()`` is a
        host sync and the per-expert token groups are data-dependent shapes --
        both illegal inside CUDA-graph capture, which is why decode uses
        :meth:`_routed_decode` instead. Prefill is never captured.
        """
        routed = torch.zeros_like(flat)
        for expert in torch.unique(topk_indices).tolist():
            token_idx, slot = torch.where(topk_indices == expert)
            contribution = self._expert(flat[token_idx], expert)
            # The routing weight is applied *after* down_proj, matching source.
            contribution = contribution * topk_weights[token_idx, slot, None].to(contribution.dtype)
            routed.index_add_(0, token_idx, contribution.to(routed.dtype))
        return routed

    def _routed_decode(
        self, flat: torch.Tensor, topk_weights: torch.Tensor, topk_indices: torch.Tensor
    ) -> torch.Tensor:
        """Static-shape routed dispatch for CUDA-graph-safe decode.

        Loops over the *fixed* ``[tokens x top_k]`` selection slots (a
        shape-static double loop) and gathers each expert's stacked weight via
        ``index_select`` with a one-element device index -- a pure device
        gather. Plain ``weight[expert]`` with a 0-d CUDA index would call
        ``.item()`` (a host sync, illegal during capture) and bake the
        capture-time expert id into the graph. Here no host sync, no
        data-dependent shape, and no Python value derived from routing ever
        occurs, so a captured graph re-routes from the router's fresh output
        on every replay. Same per-selection math as prefill's grouped path:
        ``x_t @ W_e`` rows are independent, and the routing weight is applied
        after ``down_proj``.
        """
        routed = torch.zeros_like(flat)
        for t in range(flat.shape[0]):
            x_t = flat[t : t + 1]
            for k in range(topk_indices.shape[1]):
                expert = topk_indices[t, k].view(1)
                contribution = self._expert_gathered(x_t, expert)
                routed[t : t + 1] += (contribution * topk_weights[t, k].to(contribution.dtype)).to(
                    routed.dtype
                )
        return routed

    def _expert_gathered(self, x: torch.Tensor, expert: torch.Tensor) -> torch.Tensor:
        """One expert's MLP with the expert chosen by a 1-element device index.

        Identical math to :meth:`_expert`; the only difference is how the
        stacked weights are addressed (``index_select`` device gather instead
        of Python-int indexing), which is what makes it legal inside CUDA
        graph capture.
        """
        gate_up = self.gate_up_proj.index_select(0, expert).squeeze(0)
        down = self.down_proj.index_select(0, expert).squeeze(0)
        if self.quantized:
            fused = glm5_next_block_fp8_matmul(
                x, gate_up, self.gate_up_proj_scale.index_select(0, expert).squeeze(0)
            )
            gate, up = fused.chunk(2, dim=-1)
            return glm5_next_block_fp8_matmul(
                clamped_swiglu(gate, up, self.swiglu_limit),
                down,
                self.down_proj_scale.index_select(0, expert).squeeze(0),
            )
        gate, up = torch.nn.functional.linear(x, gate_up).chunk(2, dim=-1)
        return torch.nn.functional.linear(clamped_swiglu(gate, up, self.swiglu_limit), down)


# ---------------------------------------------------------------------------
# Hyper-connected decoder
# ---------------------------------------------------------------------------


def glm5_next_hyper_connection(config: PretrainedConfig, dtype: torch.dtype = torch.bfloat16):
    """Build the shared ``mHC`` for one hyper-connection site.

    This reuses TensorRT-LLM's existing manifold-constrained hyper-connection
    rather than adding a model-local one. Two settings are model-specific and
    were pinned by measurement against the source module, not by assumption:

    * ``post_mult_value=2.0`` -- the source computes ``2 * sigmoid(...)`` for the
      block-output placement weights. Leaving the default 1.0 halves them and
      shows up as ``max_abs`` 0.96 on ``post`` against a [0.26, 1.92] range.
    * ``sinkhorn_iters`` is the config's own ``hc_sinkhorn_iters`` (20), not
      ``iters - 1``. The source runs one initial column normalization plus
      ``iters - 1`` row/column rounds, which is what this kernel's ``iters``
      counts; passing 19 leaves a measurable 1.9e-5 residual on ``comb`` versus
      1.3e-6 at 20.

    ``norm_eps`` is the decoder's ``rms_norm_eps`` because the source's
    hyper-connection input norm is constructed with it, while ``eps`` and
    ``sinkhorn_eps`` are the separate ``hc_eps``.
    """
    from ..modules.mhc.hyper_connection import mHC

    return mHC(
        mult=int(config.hc_mult),
        hidden_size=int(config.hidden_size),
        sinkhorn_iters=int(config.hc_sinkhorn_iters),
        dtype=dtype,
        eps=float(config.hc_eps),
        norm_eps=float(config.rms_norm_eps),
        sinkhorn_eps=float(config.hc_eps),
        post_mult_value=2.0,
    )


def glm5_next_expand_streams(embeds: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """``[tokens, hidden]`` -> ``[tokens, hc_mult, hidden]``.

    The streams start as exact copies of the embedding and only diverge once the
    first hyper-connection mixes them. ``contiguous()`` is required, not merely
    tidy: an expanded view aliases a single row, and ``post_mapping`` writes each
    stream separately.
    """
    return embeds.unsqueeze(-2).expand(-1, hc_mult, -1).contiguous()


def glm5_next_hyper_head(hidden_streams: torch.Tensor) -> torch.Tensor:
    """Collapse the ``hc_mult`` streams with an **unweighted** mean.

    Deliberately not ``modules.mhc.HCHead``: that head is the DeepSeek-V4
    variant and carries learned ``fn``/``base``/``scale`` weights. GLM-5.3-Flash
    has no such parameters in the checkpoint and the source head is a plain
    mean, so using the weighted head would require inventing weights.
    """
    return hidden_streams.mean(dim=-2)


class Glm5NextDecoderLayer(DecoderLayer):
    """One decoder layer: two hyper-connection sites wrapping attention and FFN.

    The residual path is *not* an ordinary add. Each site collapses the four
    streams into one sequence with the learned ``pre`` weights, runs the
    sublayer, then writes the result back across the streams as
    ``post * out + comb^T @ residual``. Both module choices come from the two
    literal per-layer lists, never from a cadence or ``first_k_dense_replace``.

    Two entry points share one implementation: the runtime ``forward``
    receives ``AttentionMetadata`` (plus the once-per-forward
    :class:`Glm5NextRuntimeContext`) and derives this layer's cache
    arguments; ``forward_direct`` takes them explicitly and is what the
    Stage-1 diagnostic driver and the module tests call. The runtime path is
    a thin argument-derivation shim over the direct path, so parity between
    them is an argument-sourcing check, not a second implementation.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        schedule: Glm5NextSchedule,
        dtype: torch.dtype = torch.bfloat16,
        quantized: bool = False,
        attn_backend: str = "TRTLLM",
        model_config: Optional[ModelConfig] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention_type = schedule.attention[layer_idx]
        self.mlp_type = schedule.mlp[layer_idx]
        eps = float(config.rms_norm_eps)

        mapping = model_config.mapping if model_config is not None else None
        allreduce_strategy = model_config.allreduce_strategy if model_config is not None else None
        if self.attention_type == LINEAR_ATTENTION:
            self.self_attn: nn.Module = Glm5NextLinearAttention(
                config, layer_idx, dtype=dtype, mapping=mapping
            )
        else:
            self.self_attn = Glm5NextSparseAttention(
                config,
                layer_idx,
                dtype=dtype,
                attn_backend=attn_backend,
                mapping=mapping,
                allreduce_strategy=allreduce_strategy,
            )
        self.mlp = (
            Glm5NextMoE(
                config,
                dtype=dtype,
                quantized=quantized,
                model_config=model_config,
                layer_idx=layer_idx,
            )
            if self.mlp_type == SPARSE_MLP
            else Glm5NextMLP(config, dtype=dtype)
        )
        self.input_layernorm = RMSNorm(hidden_size=int(config.hidden_size), eps=eps, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(
            hidden_size=int(config.hidden_size), eps=eps, dtype=dtype
        )
        self.hc_attn = glm5_next_hyper_connection(config, dtype=dtype)
        self.hc_ffn = glm5_next_hyper_connection(config, dtype=dtype)

    def forward(
        self,
        position_ids: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        runtime_ctx: Optional[Glm5NextRuntimeContext] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Runtime entry: derive this layer's cache arguments from metadata.

        The executor packs context requests first, then generation requests;
        the two phases run through :meth:`forward_direct` separately, which is
        exact because every non-attention operation here is token-local.
        ``position_ids`` is unused: the text path is fully NoPE and the
        indexer derives its positions from the cached lengths.
        """
        if runtime_ctx is None:
            runtime_ctx = build_glm5_next_runtime_context(attn_metadata)
        derive = (
            runtime_ctx.linear_kwargs
            if self.attention_type == LINEAR_ATTENTION
            else runtime_ctx.sparse_kwargs
        )
        parts = []
        if runtime_ctx.num_contexts > 0:
            parts.append(
                self.forward_direct(
                    hidden_states[: runtime_ctx.num_ctx_tokens],
                    phase="prefill",
                    **derive(self.layer_idx, "prefill"),
                )
            )
        if runtime_ctx.num_generations > 0:
            parts.append(
                self.forward_direct(
                    hidden_states[runtime_ctx.num_ctx_tokens :],
                    phase="decode",
                    **derive(self.layer_idx, "decode"),
                )
            )
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

    def forward_direct(
        self,
        hidden_streams: torch.Tensor,
        phase: str = "prefill",
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        """``hidden_streams`` is ``[num_tokens, hc_mult, hidden]``.

        ``attn_kwargs`` is forwarded verbatim to the selected attention module's
        ``forward_<phase>``; the two attention types legitimately need different
        cache arguments, and passing them through is cheaper than inventing a
        common descriptor that both would have to be squeezed into.
        """
        attention = getattr(self.self_attn, f"forward_{phase}")

        residual = hidden_streams
        post, comb, collapsed = self.hc_attn.pre_mapping(hidden_streams)
        _glm5_stream_probe(self.layer_idx, phase, residual, post, comb)
        attn_out = attention(self.input_layernorm(collapsed), **attn_kwargs)
        hidden_streams = self.hc_attn.post_mapping(attn_out, residual, post, comb)

        residual = hidden_streams
        post, comb, collapsed = self.hc_ffn.pre_mapping(hidden_streams)
        mlp_out = self.mlp(self.post_attention_layernorm(collapsed), phase=phase)
        return self.hc_ffn.post_mapping(mlp_out, residual, post, comb)


class Glm5NextModel(DecoderModel):
    """Embedding, the 45 hyper-connected decoder layers, and the final readout.

    A ``DecoderModel`` whose hidden state between layers is the four-stream
    tensor ``[num_tokens, hc_mult, hidden]`` rather than ``[num_tokens,
    hidden]``: the stream axis opens at the embedding and closes in
    :meth:`collapse_streams` (unweighted mean plus the final norm), which is
    this model's equivalent of the base class's trailing ``self.norm``.
    """

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        _normalize_glm5_next_top_config(model_config.pretrained_config)
        super().__init__(model_config)
        config = get_glm5_next_text_config(model_config.pretrained_config)
        schedule = resolve_glm5_next_schedule(model_config.pretrained_config)
        quantized = glm5_next_is_quantized(model_config)
        # Pipeline parallelism rides the base machinery wholesale: the
        # inter-layer activation is the four-stream tensor [tokens, hc_mult,
        # hidden], and both `forward_after_recv`/`forward_before_send` and
        # `pp_recv/send` are shape-agnostic — the recv buffer on a non-first
        # rank is exactly `expand_streams(embed_tokens.skip_forward(...))`,
        # which is a real contiguous [tokens, hc_mult, hidden] tensor.
        # `__pp_init__` prunes non-local layers; `load_weights` skips
        # pruned owners (see `skipped_remote`); the hybrid cache manager
        # slices its layer masks per rank from `mapping`.
        self.config = config
        self.schedule = schedule
        self.hc_mult = int(config.hc_mult)
        dtype = getattr(config, "torch_dtype", None) or torch.bfloat16

        self.embed_tokens = Embedding(int(config.vocab_size), int(config.hidden_size), dtype=dtype)
        self.layers = nn.ModuleList(
            [
                Glm5NextDecoderLayer(
                    config,
                    i,
                    schedule,
                    dtype=dtype,
                    quantized=quantized,
                    attn_backend=model_config.attn_backend,
                    model_config=model_config,
                )
                for i in range(schedule.num_layers)
            ]
        )
        self.norm = RMSNorm(
            hidden_size=int(config.hidden_size), eps=float(config.rms_norm_eps), dtype=dtype
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        runtime_ctx: Optional[Glm5NextRuntimeContext] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """The runtime forward over the executor's packed token batch.

        ``runtime_ctx`` is normally derived from ``attn_metadata`` here; the
        parameter lets a caller (or a parity test) inject a context built by
        another route. The stream axis opens once, is carried through all 45
        layers, and closes in :meth:`collapse_streams`.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        streams = self.expand_streams(inputs_embeds)
        if runtime_ctx is None:
            runtime_ctx = build_glm5_next_runtime_context(attn_metadata)
        for layer in self.layers:
            streams = layer(
                position_ids=position_ids,
                hidden_states=streams,
                attn_metadata=attn_metadata,
                runtime_ctx=runtime_ctx,
            )
        return self.collapse_streams(streams)

    def expand_streams(self, embeds: torch.Tensor) -> torch.Tensor:
        """Open the four-stream axis at the embedding."""
        return glm5_next_expand_streams(embeds, self.hc_mult)

    def collapse_streams(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        """Unweighted stream mean followed by the final RMS norm."""
        return self.norm(glm5_next_hyper_head(hidden_streams))


# ---------------------------------------------------------------------------
# Whole-model materialization
# ---------------------------------------------------------------------------

#: Destinations whose weights are consumed by a *fusion* rather than copied into
#: a same-named parameter, so the loader must reach them through a rule instead
#: of a name lookup. Everything else is placed by name.
_FUSED_MOE_RE = re.compile(
    r"^(?P<layer>model\.layers\.\d+\.mlp)\.experts\.(?P<expert>\d+)"
    r"\.(?P<proj>gate_proj|up_proj|down_proj)$"
)


def glm5_next_block_fp8_linear_cls():
    """Return the model's ``Linear`` subclass for quantized projections.

    ``Linear``'s FP8_BLOCK_SCALES method is reused wholesale -- weight and
    128x128 scale creation, checkpoint loading, tensor-parallel sharding, the
    all-reduce paths, and the production CuTe GEMM. The single override is the
    dynamic activation quantization, which uses the convention
    ``config.json`` declares rather than ``fp8_quantize_1x128``'s; see
    :func:`glm5_next_dynamic_act_quant_1x128` for the measured difference and
    why it is load-bearing. Without this the 179 quantized projections would
    carry a divergence the 288-way routed experts do not.

    Imported lazily and defined here rather than at module scope because
    ``modules.linear`` is a much heavier import than model discovery needs, and
    because this is the same shape as :func:`glm5_next_cache_manager_cls`.
    """
    from ..modules.linear import FP8BlockScalesLinearMethod, Linear

    class Glm5NextBlockFp8LinearMethod(FP8BlockScalesLinearMethod):
        """FP8_BLOCK_SCALES with the checkpoint's declared activation scale."""

        def apply(self, module, input: torch.Tensor, bias: Optional[torch.Tensor]):
            original_shape = input.shape
            if input.dim() > 2:
                input = input.reshape(-1, input.shape[-1])
            if input.dtype == torch.float8_e4m3fn:
                input = input.to(torch.bfloat16) * module.input_scale
            output = glm5_next_block_fp8_matmul(input, module.weight, module.weight_scale)
            if len(original_shape) > 2:
                output = output.reshape(*original_shape[:-1], output.shape[-1])
            if bias is not None:
                output = output + bias
            return output

    class Glm5NextBlockFp8Linear(Linear):
        """``Linear`` that resolves FP8_BLOCK_SCALES to the declared convention."""

        def get_quant_method(self, quant_config: Optional[QuantConfig] = None):
            method = super().get_quant_method(quant_config)
            if type(method) is FP8BlockScalesLinearMethod:
                return Glm5NextBlockFp8LinearMethod()
            return method

    return Glm5NextBlockFp8Linear


@dataclass(frozen=True)
class Glm5NextTpSpec:
    """Four-rank ownership of one named projection (plan Decision F).

    ``mode`` is the TensorParallelMode value as a string ("column", "row") or
    ``None`` for a replicated module; a string rather than the enum so the
    registry can be resolved and unit-tested without importing the heavy
    ``modules.linear``. ``reduce_output`` matters only for ROW: the KDA/MLA
    ``o_proj`` and the dense ``down_proj`` reduce inside the Linear (exactly
    one all-reduce per branch), while the TP4-layout shared expert's
    ``down_proj`` stays a partial so :class:`Glm5NextMoE` can sum it with the
    routed partial before its single reduction.
    """

    mode: Optional[str]
    reduce_output: bool = True


#: Suffix-keyed TP ownership for every raw ``nn.Linear`` the model constructs.
#: KDA heads, MLA heads, and the indexer's scoring heads are column-sharded;
#: branch outputs are row-sharded with one in-Linear reduction; the low-rank
#: latents (``q_a``/``kv_a``), the head-independent KDA gates (``f_a``/``g_a``)
#: and the indexer's shared 128-wide pool key (``wk`` -- plan Decision F keeps
#: the pool-key representation replicated so all ranks score identical pools)
#: stay replicated. The shared expert is resolved separately per MoE layout.
_GLM5_TP_SPECS: Tuple[Tuple[str, Glm5NextTpSpec], ...] = (
    # KDA (linear_attention) head projections.
    (".self_attn.q_proj", Glm5NextTpSpec("column")),
    (".self_attn.k_proj", Glm5NextTpSpec("column")),
    (".self_attn.v_proj", Glm5NextTpSpec("column")),
    (".self_attn.f_b_proj", Glm5NextTpSpec("column")),
    (".self_attn.g_b_proj", Glm5NextTpSpec("column")),
    (".self_attn.b_proj", Glm5NextTpSpec("column")),
    (".self_attn.f_a_proj", Glm5NextTpSpec(None)),
    (".self_attn.g_a_proj", Glm5NextTpSpec(None)),
    # Sparse MLA low-rank stack: latents replicated, per-head maps sharded.
    (".self_attn.q_a_proj", Glm5NextTpSpec(None)),
    (".self_attn.kv_a_proj_with_mqa", Glm5NextTpSpec(None)),
    (".self_attn.q_b_proj", Glm5NextTpSpec("column")),
    (".self_attn.kv_b_proj", Glm5NextTpSpec("column")),
    # Both attention families' output projection: row + one reduction.
    (".self_attn.o_proj", Glm5NextTpSpec("row")),
    # Pool indexer: 32 scoring heads split 8/rank; the shared pool key stays
    # replicated (the FP32 score all-reduce before top-k lands with Goal 5.2).
    (".self_attn.indexer.wq_b", Glm5NextTpSpec("column")),
    (".self_attn.indexer.weights_proj", Glm5NextTpSpec("column")),
    (".self_attn.indexer.wk", Glm5NextTpSpec(None)),
    # Dense MLP (and, in the TP4 layout, the shared expert -- see resolver).
    (".mlp.gate_proj", Glm5NextTpSpec("column")),
    (".mlp.up_proj", Glm5NextTpSpec("column")),
    (".mlp.down_proj", Glm5NextTpSpec("row")),
)


def resolve_glm5_next_projection_spec(name: str, mapping: Any) -> Glm5NextTpSpec:
    """TP ownership for the projection at dotted module path ``name``.

    The shared expert branches on the MoE layout carried by ``mapping``
    (plan Decision F): with ``moe_tp_size > 1`` (TP4 layout) it is TP-sharded
    like a dense MLP but its ``down_proj`` keeps a *partial* output, summed
    with the routed partial before :class:`Glm5NextMoE`'s single all-reduce;
    with ``moe_ep_size > 1`` (TP4/EP4 layout) it is replicated and added once
    after the EP combine, never reduced.

    An unknown ``nn.Linear`` path is an error, not a silent replication: every
    projection must have a declared four-rank owner or the per-rank accounting
    cannot certify "no double-owned tensor".
    """
    if ".mlp.shared_experts." in name:
        if int(getattr(mapping, "moe_tp_size", 1)) > 1:
            if name.endswith(".down_proj"):
                return Glm5NextTpSpec("row", reduce_output=False)
            if name.endswith((".gate_proj", ".up_proj")):
                return Glm5NextTpSpec("column")
        elif name.endswith((".gate_proj", ".up_proj", ".down_proj")):
            return Glm5NextTpSpec(None)
    else:
        for suffix, spec in _GLM5_TP_SPECS:
            if name.endswith(suffix):
                return spec
    raise ValueError(f"glm5_next has no tensor-parallel ownership declared for projection {name!r}")


def glm5_next_swap_quantized_projections(
    root: nn.Module,
    plan: Dict[str, bool],
    *,
    mapping: Any = None,
    disable_deep_gemm: bool = True,
) -> Dict[str, str]:
    """Replace every raw ``nn.Linear`` under ``root`` with a Mapping-aware one.

    Every named projection becomes a TensorRT-LLM ``Linear`` carrying the
    four-rank Mapping and its declared column/row/replicated ownership
    (:func:`resolve_glm5_next_projection_spec`); the audited ``plan`` decides
    per module whether that Linear is the block-FP8 subclass or the plain BF16
    one, so a module's runtime dtype cannot drift from the 1509-entry
    contract. The swap runs *after* construction and *before* materialization
    -- on ``meta`` tensors nothing is copied, and it never dequantizes. At
    ``tp_size == 1`` the converted modules are numerically identical to the
    raw ones (BF16 resolves to the same ``F.linear``; block-FP8 keeps the same
    subclass/kernel), which is what keeps the frozen PP4 evidence a valid
    oracle.

    ``disable_deep_gemm`` is on by default and is not cosmetic. On this SM100
    build the DeepGEMM path (:func:`torch.ops.trtllm.fp8_swap_ab_gemm`) returns
    non-finite values on ``cuda:0`` and raises ``CUDA_ERROR_INVALID_HANDLE`` on
    every other device, because its cached kernels are bound to a single CUDA
    context -- fatal for a model spanning eight GPUs in one process. It is kept
    on so the flag a reader sees still describes what would run if the
    activation-quantization override in
    :func:`glm5_next_block_fp8_linear_cls` were ever removed.

    Returns the ``{module_name: dtype}`` map for reporting: a loader that
    silently quantized an excluded module would otherwise be invisible. The
    per-module TP ownership lands on each replacement as ``glm5_tp_spec`` /
    ``glm5_full_shape`` and is reported by the loader's shard accounting.
    """
    from tensorrt_llm.mapping import Mapping

    from ..distributed import AllReduceStrategy
    from ..modules.linear import Linear, TensorParallelMode

    if mapping is None:
        model_config = getattr(root, "model_config", None)
        mapping = getattr(model_config, "mapping", None) or Mapping()
    fp8_cls = glm5_next_block_fp8_linear_cls()
    quant_config = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES)
    placed: Dict[str, str] = {}
    for name, module in list(root.named_modules()):
        if not isinstance(module, nn.Linear) or isinstance(module, Linear):
            continue
        if getattr(module, "_weights_removed", False):
            # Pruned by pipeline parallelism: the module belongs to another
            # rank. Its cleared parameter dict makes even `module.bias` an
            # AttributeError, and a replacement would allocate real weights
            # for a layer this rank never runs.
            continue
        spec = resolve_glm5_next_projection_spec(name, mapping)
        quantized = bool(plan.get(name, False))
        linear_cls = fp8_cls if quantized else Linear
        parent_name, _, attr = name.rpartition(".")
        parent = root.get_submodule(parent_name) if parent_name else root
        replacement = linear_cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            dtype=torch.bfloat16,
            mapping=mapping,
            tensor_parallel_mode=(TensorParallelMode(spec.mode) if spec.mode is not None else None),
            gather_output=False,
            reduce_output=spec.reduce_output,
            quant_config=quant_config if quantized else None,
            disable_deep_gemm=disable_deep_gemm,
            # Pin every row-parallel reduction to NCCL rather than the AUTO
            # autotuner: AUTO's tuned tactic raced at TP4 decode on this
            # model's collective schedule and produced intermittent NaN logits
            # (see Glm5NextMoE.moe_all_reduce and the indexer score reduction).
            # Column/replicated modes construct but never invoke the AllReduce,
            # so passing the strategy uniformly is harmless.
            allreduce_strategy=AllReduceStrategy.NCCL,
        )
        # Loader-side provenance: the full checkpoint geometry (the local
        # in/out features on the module are post-shard) and the declared
        # ownership, so shard accounting can prove union-without-overlap.
        replacement.glm5_full_shape = (module.out_features, module.in_features)
        replacement.glm5_tp_spec = spec
        setattr(parent, attr, replacement)
        placed[name] = "float8_e4m3fn" if quantized else "bfloat16"
    return placed


class Glm5NextLoadReport:
    """What the whole-model loader actually did, in numbers.

    A load that quietly skipped a tensor produces a model that still runs and
    still looks plausible, so every destination is counted rather than assumed.
    """

    def __init__(self) -> None:
        self.loaded = 0
        self.transformed = 0
        self.ignored = 0
        # Keys whose destination owner was pruned by pipeline parallelism:
        # they are loaded by the rank that owns those layers, not this one.
        self.skipped_remote = 0
        # Routed-expert keys owned by another expert-parallel rank (EP>1):
        # counted separately from PP-remote so the per-rank report can prove
        # the 72-expert local range without conflating the two remotenesses.
        self.remote_experts = 0
        self.missing_destinations: List[str] = []
        self.dtypes: Dict[str, str] = {}
        self.devices: Dict[int, str] = {}
        #: Per converted-projection shard provenance: tp mode, this rank's
        #: contiguous (start, end) range on the sharded dim, local and full
        #: shapes. The four ranks' ranges must union to the full tensor with
        #: no overlap -- the driver asserts that from these records.
        self.tp_shards: Dict[str, Dict[str, Any]] = {}

    def summary(self) -> Dict[str, Any]:
        counts = Counter(self.dtypes.values())
        return {
            "loaded": self.loaded,
            "transformed": self.transformed,
            "ignored": self.ignored,
            "skipped_remote": self.skipped_remote,
            "remote_experts": self.remote_experts,
            "total": self.loaded
            + self.transformed
            + self.ignored
            + self.skipped_remote
            + self.remote_experts,
            "missing_destinations": sorted(self.missing_destinations),
            "module_dtypes": dict(counts),
            # Keys are stringified: owners mix layer indices with "embed" /
            # "norm" / "head", and a mixed-type mapping is not JSON-sortable.
            "layer_devices": {str(owner): value for owner, value in self.devices.items()},
            "tp_shards": dict(self.tp_shards),
        }


#: Destinations whose runtime parameter is spelled differently from the
#: checkpoint's own name. Neither is a value transformation -- the audit already
#: classifies both -- so they are resolved here at placement time rather than in
#: :func:`remap_glm5_next_key`, whose output is the audited destination namespace.
_DESTINATION_ALIASES = (
    # The KDA gated output norm is a flat head-dim gain, not a submodule.
    (".o_norm.weight", ".o_norm_weight"),
    # TensorRT-LLM's block-scale parameter name for the DeepSeek recipe.
    (".weight_scale_inv", ".weight_scale"),
)


def _apply_destination_alias(dest: str) -> str:
    for suffix, replacement in _DESTINATION_ALIASES:
        if dest.endswith(suffix):
            return dest[: -len(suffix)] + replacement
    return dest


def _destination_owner(dest: str, num_layers: int) -> Any:
    """Which materialization unit owns ``dest``: a layer index or a named part."""
    match = re.match(r"^model\.layers\.(\d+)\.", dest)
    if match is not None:
        index = int(match.group(1))
        return index if index < num_layers else None
    if dest.startswith("model.embed_tokens."):
        return "embed"
    if dest.startswith("model.norm."):
        return "norm"
    if dest.startswith("lm_head."):
        return "head"
    return None
