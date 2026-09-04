# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent pure-PyTorch reference for the GLM-5.3-Flash text path.

This module is rung two of the three-rung reference ladder used to bring
GLM-5.3-Flash up on the TensorRT-LLM PyTorch backend:

    native HuggingFace  ->  this module  ->  TensorRT-LLM

It is deliberately written *from the source semantics* rather than by importing
``transformers.models.glm5_next`` classes, so that a shared helper bug cannot
make both sides of a parity check agree.  Where HuggingFace uses a fast
algorithm (the chunked KDA scan, the vectorised k-pool gather) this module uses
the obvious sequential formulation instead: the two implementations only agree
if the *semantics* agree.

Nothing here is used by production code -- it exists purely so tests have an
oracle that is independent of both HuggingFace and TensorRT-LLM.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "GLM53_FLASH_INVENTORY",
    "VariantInventory",
    "assert_variant_inventory",
    "CheckpointReader",
    "ExclusionAudit",
    "resolve_quant_exclusions",
    "dequantize_block_fp8",
    "RefUnweightedRMSNorm",
    "RefRMSNorm",
    "RefRMSNormGated",
    "RefHyperConnection",
    "RefHyperHead",
    "RefForgetGate",
    "RefLinearAttention",
    "RefIndexer",
    "RefSparseMLA",
    "RefMLP",
    "RefTopkRouter",
    "RefMoE",
    "compare",
]


# ---------------------------------------------------------------------------
# Pinned variant inventory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariantInventory:
    """Every scalar/flag/list contract this bring-up claims to support.

    Support is intentionally scoped to the exact ``/dev/shm/GLM-5.3-Flash``
    checkpoint.  Family-wide behaviour is explicitly *not* claimed; anything not
    listed here is ``Unknown`` and must be re-derived before it is relied on.
    """

    # --- global geometry -------------------------------------------------
    model_type: str = "glm5_next"
    text_model_type: str = "glm5_next_text"
    architecture: str = "Glm5NextForConditionalGeneration"
    hidden_size: int = 4096
    num_hidden_layers: int = 45
    vocab_size: int = 154880
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False
    torch_dtype: str = "bfloat16"

    # --- literal schedules ----------------------------------------------
    num_linear_attention_layers: int = 34
    num_sparse_attention_layers: int = 11
    sparse_attention_layer_indices: Tuple[int, ...] = tuple(range(3, 45, 4))
    num_dense_mlp_layers: int = 3
    num_sparse_mlp_layers: int = 42
    dense_mlp_layer_indices: Tuple[int, ...] = (0, 1, 2)
    first_k_dense_replace: int = 3
    # every layer runs its own indexer; no cross-layer top-k sharing
    indexer_types_unique: Tuple[str, ...] = ("full",)

    # --- linear attention (KDA) -----------------------------------------
    linear_num_heads: int = 64
    linear_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    linear_lower_bound: float = -5.0

    # --- sparse MLA ------------------------------------------------------
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_head_dim: int = 256
    qk_nope_head_dim: int = 256
    qk_rope_head_dim: int = 0
    v_head_dim: int = 256
    mla_use_nope: bool = True

    # --- pool-compressed DSA indexer -------------------------------------
    index_topk: int = 2048
    index_kpool: int = 4
    index_kpool_compress: bool = True
    index_kpool_always_select_tail: bool = True
    index_n_heads: int = 32
    index_head_dim: int = 128
    # index_topk // index_kpool
    index_select_k: int = 512
    # index_topk + (index_kpool - 1) -- the always-selected tail widens output
    indexer_output_width: int = 2051
    # present in config but vestigial: the text path is fully NoPE
    indexer_rope_interleave: bool = True

    # --- per-request cache state (proven by the phase-coverage replay) -----
    # These are not config fields; they are the state widths the production
    # KVCacheManagerV2 descriptors have to allocate, measured against native HF
    # and the reference under chunked prefill and token-by-token decode.
    #
    # 3 * linear_num_heads * linear_head_dim -- q, k and v share one conv1d
    kda_conv_dim: int = 24576
    # HF allocates linear_conv_kernel_dim slots; the last holds the token just
    # consumed (causal_conv1d_update's roll convention).  The reference keeps
    # only the linear_conv_kernel_dim - 1 slots that are real left context.
    # Either layout reproduces one-shot prefill; a narrower buffer cannot.
    kda_conv_state_width: int = 4
    kda_conv_history_width: int = 3
    # [B, linear_num_heads, linear_head_dim, linear_head_dim], and it must stay
    # FP32: it accumulates delta updates over the whole prefix.
    kda_recurrent_state_dtype: str = "float32"
    # The indexer caches packed per-token state, not pool keys -- pools are
    # rebuilt from it every step: [k(index_head_dim), gate(index_head_dim), valid(1)]
    indexer_packed_state_width: int = 257
    # Sparse MLA caches the pre-kv_b_proj latent, width kv_lora_rank
    mla_latent_cache_width: int = 512

    # --- hyper-connections -----------------------------------------------
    mhc: bool = True
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    # (2 + hc_mult) * hc_mult
    hc_mix: int = 24
    # hc_mult * hidden_size
    hc_fn_in: int = 16384
    hc_post_mult_value: float = 2.0

    # --- feed forward / MoE ----------------------------------------------
    hidden_act: str = "silu"
    swiglu_limit: float = 10.0
    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048
    n_routed_experts: int = 288
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    moe_router_dtype: str = "float32"
    # n_group == topk_group == 1 makes group-limited routing degenerate
    n_group: int = 1
    topk_group: int = 1

    # --- quantization -----------------------------------------------------
    quant_method: str = "fp8"
    quant_fmt: str = "e4m3"
    activation_scheme: str = "dynamic"
    weight_block_size: Tuple[int, int] = (128, 128)
    num_modules_to_not_convert: int = 1509

    # --- excluded namespaces ---------------------------------------------
    num_nextn_predict_layers: int = 1
    mtp_layer_index: int = 45
    ignored_key_prefixes: Tuple[str, ...] = (
        "model.language_model.layers.45.",
        "model.visual.",
    )


GLM53_FLASH_INVENTORY = VariantInventory()


def _get(cfg: Any, name: str) -> Any:
    if not hasattr(cfg, name):
        raise AssertionError(f"config is missing required attribute {name!r}")
    return getattr(cfg, name)


def assert_variant_inventory(
    config: Any,
    inventory: VariantInventory = GLM53_FLASH_INVENTORY,
) -> Dict[str, Any]:
    """Assert a loaded HF ``Glm5NextConfig`` matches the pinned inventory.

    ``config`` may be either the top-level ``Glm5NextConfig`` or its
    ``text_config``.  Returns a dict of the observed values so callers can embed
    them in an evidence report.

    Raises ``AssertionError`` on the first mismatch -- a config that disagrees
    with the pinned inventory is a task/config conflict, not something to be
    silently absorbed.
    """
    top = config
    text = getattr(config, "text_config", config)
    inv = inventory
    observed: Dict[str, Any] = {}

    def check(key: str, actual: Any, expected: Any) -> None:
        observed[key] = actual
        if isinstance(expected, float) and isinstance(actual, (int, float)):
            ok = math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=0.0)
        else:
            ok = actual == expected
        if not ok:
            raise AssertionError(
                f"variant inventory mismatch for {key!r}: checkpoint has {actual!r}, "
                f"pinned inventory expects {expected!r}"
            )

    if hasattr(top, "architectures") and top.architectures:
        check("architecture", top.architectures[0], inv.architecture)
    if hasattr(top, "model_type") and top is not text:
        check("model_type", top.model_type, inv.model_type)
    check("text_model_type", _get(text, "model_type"), inv.text_model_type)

    # global geometry
    check("hidden_size", _get(text, "hidden_size"), inv.hidden_size)
    check("num_hidden_layers", _get(text, "num_hidden_layers"), inv.num_hidden_layers)
    check("vocab_size", _get(text, "vocab_size"), inv.vocab_size)
    check(
        "max_position_embeddings",
        _get(text, "max_position_embeddings"),
        inv.max_position_embeddings,
    )
    check("rms_norm_eps", _get(text, "rms_norm_eps"), inv.rms_norm_eps)
    check("tie_word_embeddings", _get(text, "tie_word_embeddings"), inv.tie_word_embeddings)

    # literal schedules -- assert the *lists*, never a derived cadence
    layer_types = list(_get(text, "layer_types"))
    check("len(layer_types)", len(layer_types), inv.num_hidden_layers)
    linear_idx = [i for i, t in enumerate(layer_types) if t == "linear_attention"]
    sparse_idx = [i for i, t in enumerate(layer_types) if t == "deepseek_sparse_attention"]
    check("num_linear_attention_layers", len(linear_idx), inv.num_linear_attention_layers)
    check("num_sparse_attention_layers", len(sparse_idx), inv.num_sparse_attention_layers)
    check(
        "sparse_attention_layer_indices",
        tuple(sparse_idx),
        inv.sparse_attention_layer_indices,
    )
    if len(linear_idx) + len(sparse_idx) != len(layer_types):
        raise AssertionError(
            f"layer_types contains unsupported entries: {sorted(set(layer_types))}"
        )

    mlp_types = list(_get(text, "mlp_layer_types"))
    check("len(mlp_layer_types)", len(mlp_types), inv.num_hidden_layers)
    dense_idx = [i for i, t in enumerate(mlp_types) if t == "dense"]
    sparse_mlp_idx = [i for i, t in enumerate(mlp_types) if t == "sparse"]
    check("num_dense_mlp_layers", len(dense_idx), inv.num_dense_mlp_layers)
    check("num_sparse_mlp_layers", len(sparse_mlp_idx), inv.num_sparse_mlp_layers)
    check("dense_mlp_layer_indices", tuple(dense_idx), inv.dense_mlp_layer_indices)
    if len(dense_idx) + len(sparse_mlp_idx) != len(mlp_types):
        raise AssertionError(
            f"mlp_layer_types contains unsupported entries: {sorted(set(mlp_types))}"
        )
    # first_k_dense_replace is a consistency check, never the source of ownership
    check("first_k_dense_replace", _get(text, "first_k_dense_replace"), inv.first_k_dense_replace)
    if tuple(dense_idx) != tuple(range(inv.first_k_dense_replace)):
        raise AssertionError(
            "mlp_layer_types dense entries disagree with first_k_dense_replace: "
            f"{dense_idx} vs first {inv.first_k_dense_replace}"
        )

    indexer_types = list(_get(text, "indexer_types"))
    check("len(indexer_types)", len(indexer_types), inv.num_hidden_layers)
    check("indexer_types_unique", tuple(sorted(set(indexer_types))), inv.indexer_types_unique)

    # linear attention
    check("linear_num_heads", _get(text, "linear_num_heads"), inv.linear_num_heads)
    check("linear_head_dim", _get(text, "linear_head_dim"), inv.linear_head_dim)
    check(
        "linear_conv_kernel_dim",
        _get(text, "linear_conv_kernel_dim"),
        inv.linear_conv_kernel_dim,
    )
    check("linear_lower_bound", _get(text, "linear_lower_bound"), inv.linear_lower_bound)

    # sparse MLA -- fully NoPE
    check("num_attention_heads", _get(text, "num_attention_heads"), inv.num_attention_heads)
    check("num_key_value_heads", _get(text, "num_key_value_heads"), inv.num_key_value_heads)
    check("q_lora_rank", _get(text, "q_lora_rank"), inv.q_lora_rank)
    check("kv_lora_rank", _get(text, "kv_lora_rank"), inv.kv_lora_rank)
    check("qk_head_dim", _get(text, "qk_head_dim"), inv.qk_head_dim)
    check("qk_nope_head_dim", _get(text, "qk_nope_head_dim"), inv.qk_nope_head_dim)
    check("qk_rope_head_dim", _get(text, "qk_rope_head_dim"), inv.qk_rope_head_dim)
    check("v_head_dim", _get(text, "v_head_dim"), inv.v_head_dim)
    check("mla_use_nope", _get(text, "mla_use_nope"), inv.mla_use_nope)
    if inv.qk_rope_head_dim != 0:
        raise AssertionError("this bring-up only supports the fully NoPE text path")

    # pool-compressed indexer
    check("index_topk", _get(text, "index_topk"), inv.index_topk)
    check("index_kpool", _get(text, "index_kpool"), inv.index_kpool)
    check("index_kpool_compress", _get(text, "index_kpool_compress"), inv.index_kpool_compress)
    check(
        "index_kpool_always_select_tail",
        _get(text, "index_kpool_always_select_tail"),
        inv.index_kpool_always_select_tail,
    )
    check("index_n_heads", _get(text, "index_n_heads"), inv.index_n_heads)
    check("index_head_dim", _get(text, "index_head_dim"), inv.index_head_dim)
    check("index_select_k", inv.index_topk // inv.index_kpool, inv.index_select_k)
    check(
        "indexer_output_width",
        inv.index_topk + inv.index_kpool - 1,
        inv.indexer_output_width,
    )

    # per-request cache state widths (measured in the phase-coverage replay;
    # re-derived here so the pinned values cannot drift from the config)
    check(
        "kda_conv_dim",
        3 * inv.linear_num_heads * inv.linear_head_dim,
        inv.kda_conv_dim,
    )
    check("kda_conv_state_width", inv.linear_conv_kernel_dim, inv.kda_conv_state_width)
    check("kda_conv_history_width", inv.linear_conv_kernel_dim - 1, inv.kda_conv_history_width)
    check(
        "indexer_packed_state_width",
        2 * inv.index_head_dim + 1,
        inv.indexer_packed_state_width,
    )
    check("mla_latent_cache_width", inv.kv_lora_rank, inv.mla_latent_cache_width)

    # hyper-connections
    check("mhc", _get(text, "mhc"), inv.mhc)
    check("hc_mult", _get(text, "hc_mult"), inv.hc_mult)
    check("hc_sinkhorn_iters", _get(text, "hc_sinkhorn_iters"), inv.hc_sinkhorn_iters)
    check("hc_eps", _get(text, "hc_eps"), inv.hc_eps)
    check("hc_mix", (2 + inv.hc_mult) * inv.hc_mult, inv.hc_mix)
    check("hc_fn_in", inv.hc_mult * inv.hidden_size, inv.hc_fn_in)

    # feed forward / MoE
    check("hidden_act", _get(text, "hidden_act"), inv.hidden_act)
    check("swiglu_limit", _get(text, "swiglu_limit"), inv.swiglu_limit)
    check("intermediate_size", _get(text, "intermediate_size"), inv.intermediate_size)
    check("moe_intermediate_size", _get(text, "moe_intermediate_size"), inv.moe_intermediate_size)
    check("n_routed_experts", _get(text, "n_routed_experts"), inv.n_routed_experts)
    check("n_shared_experts", _get(text, "n_shared_experts"), inv.n_shared_experts)
    check("num_experts_per_tok", _get(text, "num_experts_per_tok"), inv.num_experts_per_tok)
    check("scoring_func", _get(text, "scoring_func"), inv.scoring_func)
    check("topk_method", _get(text, "topk_method"), inv.topk_method)
    check(
        "routed_scaling_factor",
        _get(text, "routed_scaling_factor"),
        inv.routed_scaling_factor,
    )
    check("norm_topk_prob", _get(text, "norm_topk_prob"), inv.norm_topk_prob)
    check("moe_router_dtype", _get(text, "moe_router_dtype"), inv.moe_router_dtype)
    check("n_group", _get(text, "n_group"), inv.n_group)
    check("topk_group", _get(text, "topk_group"), inv.topk_group)

    # MTP is out of scope but must be *accounted for*, not hidden
    check(
        "num_nextn_predict_layers",
        _get(text, "num_nextn_predict_layers"),
        inv.num_nextn_predict_layers,
    )

    # quantization
    quant = getattr(top, "quantization_config", None)
    if quant is None:
        raise AssertionError("checkpoint config has no quantization_config")
    if not isinstance(quant, dict):
        quant = quant.to_dict() if hasattr(quant, "to_dict") else vars(quant)
    check("quant_method", quant["quant_method"], inv.quant_method)
    check("quant_fmt", quant["fmt"], inv.quant_fmt)
    check("activation_scheme", quant["activation_scheme"], inv.activation_scheme)
    check("weight_block_size", tuple(quant["weight_block_size"]), inv.weight_block_size)
    check(
        "num_modules_to_not_convert",
        len(quant["modules_to_not_convert"]),
        inv.num_modules_to_not_convert,
    )

    return observed


# ---------------------------------------------------------------------------
# Checkpoint access: selective, block-FP8 aware
# ---------------------------------------------------------------------------


def dequantize_block_fp8(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: Tuple[int, int] = (128, 128),
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Expand a 128x128 block-scaled e4m3 weight back to ``dtype``.

    ``scale_inv`` has one entry per ``block_size`` tile of ``weight``; the final
    row/column tile is a partial ("edge") block whenever the weight dimension is
    not a multiple of the block size.  The expansion below therefore crops to
    the weight shape rather than assuming exact divisibility.
    """
    if weight.ndim != 2:
        raise ValueError(f"expected a 2D weight, got shape {tuple(weight.shape)}")
    rows, cols = weight.shape
    bn, bk = block_size
    exp_rows = (rows + bn - 1) // bn
    exp_cols = (cols + bk - 1) // bk
    if tuple(scale_inv.shape) != (exp_rows, exp_cols):
        raise ValueError(
            f"scale_inv shape {tuple(scale_inv.shape)} does not match the "
            f"{bn}x{bk} block grid {(exp_rows, exp_cols)} implied by weight shape "
            f"{(rows, cols)}"
        )
    scale = scale_inv.to(torch.float32)
    scale = scale.repeat_interleave(bn, dim=0).repeat_interleave(bk, dim=1)
    scale = scale[:rows, :cols]
    return (weight.to(torch.float32) * scale).to(dtype)


@dataclass
class ExclusionAudit:
    """Result of resolving ``modules_to_not_convert`` against real key names."""

    excluded: set
    hits_per_entry: Dict[str, int]
    zero_match_entries: List[str]
    ambiguous_entries: Dict[str, int] = field(default_factory=dict)

    @property
    def num_excluded(self) -> int:
        return len(self.excluded)


def resolve_quant_exclusions(
    entries: Iterable[str],
    keys: Iterable[str],
) -> ExclusionAudit:
    """Resolve GLM-5.3-Flash's ``modules_to_not_convert`` list against real keys.

    Matching is **not** path equality.  The list is authored against the
    standalone text/vision naming (``model.layers.N....``, ``visual.blocks.N....``)
    while the multimodal checkpoint inserts a container level
    (``model.language_model.layers.N....``, ``model.visual.blocks.N....``), so
    a path-equality reading matches *zero* entries and would quantize every
    module the checkpoint keeps in BF16.

    The audited rule is:

    1. strip a single leading ``model.`` component from the entry, then
    2. match when the entry's remaining dotted components appear as a
       **contiguous run** of components in the checkpoint key.

    Component-wise matching (rather than raw substring matching) is what keeps
    ``q_proj`` from also matching ``kv_a_proj_with_mqa`` and keeps the bare
    entries (``visual``, ``lm_head``, ``dt_bias``, ``weights_proj``) precise.

    This resolver is validated against the checkpoint's own ground truth -- a
    weight is BF16 exactly when it carries no ``_scale_inv`` companion -- so it
    is a *derived* rule with an independent oracle, not a guess.
    """
    normalized: List[Tuple[str, List[str]]] = []
    for entry in entries:
        parts = entry.split(".")
        if parts and parts[0] == "model":
            parts = parts[1:]
        normalized.append((entry, parts))

    def matches(sub: List[str], comps: List[str]) -> bool:
        n = len(sub)
        if n == 0 or n > len(comps):
            return False
        return any(comps[i : i + n] == sub for i in range(len(comps) - n + 1))

    excluded = set()
    hits: Dict[str, int] = {entry: 0 for entry, _ in normalized}
    for key in keys:
        comps = key.split(".")
        matched = False
        for entry, sub in normalized:
            if matches(sub, comps):
                hits[entry] += 1
                matched = True
        if matched:
            excluded.add(key)

    zero = [entry for entry, count in hits.items() if count == 0]
    return ExclusionAudit(excluded=excluded, hits_per_entry=hits, zero_match_entries=zero)


class CheckpointReader:
    """Read a *subset* of a sharded safetensors checkpoint.

    The GLM-5.3-Flash checkpoint is ~306 GiB, so module-level parity tests load
    only the tensors they actually need.  Block-FP8 weights are transparently
    dequantized when ``dequantize=True``.
    """

    def __init__(self, checkpoint_path: str, block_size: Tuple[int, int] = (128, 128)):
        self.path = checkpoint_path
        self.block_size = block_size
        index_file = os.path.join(checkpoint_path, "model.safetensors.index.json")
        with open(index_file) as fh:
            self.weight_map: Dict[str, str] = json.load(fh)["weight_map"]
        self._handles: Dict[str, Any] = {}

    def __contains__(self, key: str) -> bool:
        return key in self.weight_map

    def keys(self) -> Iterable[str]:
        return self.weight_map.keys()

    def _handle(self, shard: str):
        from safetensors import safe_open

        if shard not in self._handles:
            self._handles[shard] = safe_open(
                os.path.join(self.path, shard), framework="pt", device="cpu"
            )
        return self._handles[shard]

    def close(self) -> None:
        self._handles.clear()

    def raw(self, key: str) -> torch.Tensor:
        if key not in self.weight_map:
            raise KeyError(f"{key!r} is not present in the checkpoint index")
        return self._handle(self.weight_map[key]).get_tensor(key)

    def meta(self, key: str) -> Tuple[Tuple[int, ...], str]:
        """Return ``(shape, dtype_str)`` without materializing the tensor.

        Lets an audit sweep every one of the checkpoint's 76k entries instead of
        sampling a prefix of them.
        """
        if key not in self.weight_map:
            raise KeyError(f"{key!r} is not present in the checkpoint index")
        sl = self._handle(self.weight_map[key]).get_slice(key)
        return tuple(sl.get_shape()), sl.get_dtype()

    def get(
        self,
        key: str,
        dequantize: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Fetch one tensor, dequantizing block-FP8 weights when requested.

        ``device`` moves the raw e4m3 payload and its block scales *before* the
        expansion, so the float32 intermediate is allocated there rather than on
        the host.  Measured on this checkpoint that is ~14x faster, which is
        what makes materializing all 288 experts of a routed layer affordable.
        """
        tensor = self.raw(key)
        if device is not None:
            tensor = tensor.to(device)
        scale_key = key + "_scale_inv"
        if dequantize and scale_key in self.weight_map:
            scale = self.raw(scale_key)
            if device is not None:
                scale = scale.to(device)
            return dequantize_block_fp8(tensor, scale, self.block_size, dtype=dtype)
        if tensor.dtype == torch.float8_e4m3fn:
            raise ValueError(
                f"{key!r} is stored as float8_e4m3fn but has no {scale_key!r}; "
                "refusing to guess a scale"
            )
        return tensor

    def is_quantized(self, key: str) -> bool:
        return (key + "_scale_inv") in self.weight_map

    def prefix(self, prefix: str) -> List[str]:
        return sorted(k for k in self.weight_map if k.startswith(prefix))

    def load_module(
        self,
        module: nn.Module,
        mapping: Dict[str, str],
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cpu",
    ) -> None:
        """Copy checkpoint tensors into ``module`` params/buffers.

        ``mapping`` maps ``module`` parameter names to checkpoint keys.
        """
        own = dict(module.named_parameters())
        own.update(dict(module.named_buffers()))
        for param_name, ckpt_key in mapping.items():
            if param_name not in own:
                raise KeyError(f"{param_name!r} is not a parameter/buffer of {module}")
            target = own[param_name]
            value = self.get(ckpt_key, dequantize=True, dtype=dtype)
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"shape mismatch loading {ckpt_key!r} into {param_name!r}: "
                    f"{tuple(value.shape)} vs {tuple(target.shape)}"
                )
            with torch.no_grad():
                target.copy_(value.to(device=device, dtype=target.dtype))


# ---------------------------------------------------------------------------
# Reference modules
# ---------------------------------------------------------------------------


class RefUnweightedRMSNorm(nn.Module):
    """RMS norm with no learned gain (``Glm5NextTextUnweightedRMSNorm``)."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().square().mean(-1, keepdim=True)
        return x * torch.rsqrt(rms + self.eps).to(x.dtype)


class RefRMSNorm(nn.Module):
    """Weighted RMS norm, FP32 accumulation, cast back before the gain."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x32 = x.to(torch.float32)
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x32.to(in_dtype)


class RefRMSNormGated(nn.Module):
    """Strict-FP32 gated RMS norm used on the KDA output."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x32 = x.to(torch.float32)
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        x32 = self.weight.to(torch.float32) * x32
        x32 = x32 * torch.sigmoid(gate.to(torch.float32))
        return x32.to(in_dtype)


class RefHyperConnection(nn.Module):
    """Manifold-constrained hyper-connection (mHC) pre-mapping.

    Reproduces ``Glm5NextTextHyperConnection`` exactly, including:

    * the *unweighted* RMS norm on the flattened streams, using ``rms_norm_eps``
      (1e-5) -- **not** ``hc_eps``;
    * ``pre = sigmoid(...) + hc_eps`` and ``post = 2 * sigmoid(...)``;
    * ``comb = softmax(logits, dim=-1) + hc_eps`` followed by **one** column
      normalization (``sum(dim=-2)``), then ``hc_sinkhorn_iters - 1`` rounds of
      (row ``sum(dim=-1)``, column ``sum(dim=-2)``).  For 20 iterations that is
      39 normalization half-steps in total, and the *first* half-step is a
      column normalization, not a row one.
    * everything above in FP32; only ``collapsed`` is cast back.
    """

    def __init__(
        self,
        hc_mult: int,
        hidden_size: int,
        sinkhorn_iters: int,
        hc_eps: float = 1e-6,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.hc_mult = hc_mult
        self.hidden_size = hidden_size
        self.sinkhorn_iters = sinkhorn_iters
        self.hc_eps = hc_eps
        self.input_norm = RefUnweightedRMSNorm(eps=norm_eps)
        mix = (2 + hc_mult) * hc_mult
        self.fn = nn.Parameter(torch.empty(mix, hc_mult * hidden_size))
        self.base = nn.Parameter(torch.empty(mix))
        self.scale = nn.Parameter(torch.empty(3))

    def forward(
        self, hidden_streams: torch.Tensor, return_trace: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hc = self.hc_mult
        eps = self.hc_eps
        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        mixed = F.linear(flat, self.fn.float())
        pre_w, post_w, comb_w = mixed.split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = self.base.float().split([hc, hc, hc * hc])
        pre_s, post_s, comb_s = self.scale.float().unbind(0)

        pre = torch.sigmoid(pre_w * pre_s + pre_b) + eps
        post = 2 * torch.sigmoid(post_w * post_s + post_b)

        comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_s + comb_b.view(hc, hc)
        comb = torch.softmax(comb_logits, dim=-1) + eps

        trace: List[Dict[str, Any]] = []

        def record(step: str) -> None:
            if return_trace:
                trace.append(
                    {
                        "step": step,
                        "row_sum_min": comb.sum(dim=-1).min().item(),
                        "row_sum_max": comb.sum(dim=-1).max().item(),
                        "col_sum_min": comb.sum(dim=-2).min().item(),
                        "col_sum_max": comb.sum(dim=-2).max().item(),
                    }
                )

        # initial half-step is a COLUMN normalization
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        record("init:col")
        for i in range(self.sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
            record(f"round{i}:row")
            comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
            record(f"round{i}:col")

        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
        if return_trace:
            return post, comb, collapsed, trace
        return post, comb, collapsed

    @staticmethod
    def post_mix(
        branch_out: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """Write a branch output back into the ``hc_mult`` residual streams."""
        dtype = residual.dtype
        return post.to(dtype).unsqueeze(-1) * branch_out.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual
        )


class RefHyperHead(nn.Module):
    """GLM-5.3-Flash collapses the streams with an *unweighted* mean.

    This is the one place GLM-5.3-Flash differs from DeepSeek-V4, whose head is
    a learned weighted readout.  Reusing a weighted head here would silently
    change the final logits.
    """

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        return hidden_streams.mean(dim=-2)


class RefForgetGate(nn.Module):
    """KDA forget gate with the ``-5.0`` safe lower bound.

    Note the bound is a *scale on a sigmoid*, not a clamp: the result is
    ``gate_lower_bound * sigmoid(exp(A_log) * g)``, so ``g`` lands in
    ``(-5, 0)`` smoothly.  Treating it as ``clamp(min=-5)`` would be wrong.
    """

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, lower_bound: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        qkv_dim = num_heads * head_dim
        self.f_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.f_b_proj = nn.Linear(head_dim, qkv_dim, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(qkv_dim))
        self.A_log = nn.Parameter(torch.empty(num_heads))
        self.lower_bound = lower_bound

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        b, s = hidden_states.shape[:2]
        gate = self.f_b_proj(self.f_a_proj(hidden_states))
        g = (gate.float() + self.dt_bias.float().view(1, 1, -1)).view(b, s, -1, self.head_dim)
        decay = torch.exp(self.A_log.float().view(1, 1, self.num_heads, 1))
        return self.lower_bound * torch.sigmoid(decay * g)


def _ref_l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """FLA-style L2 norm: ``x / sqrt(sum(x^2) + eps)`` (add, not max)."""
    return x / torch.sqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def ref_kda_recurrent(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sequential Kimi-delta-attention scan -- the obvious formulation.

    HuggingFace runs a *chunked* algorithm during prefill and this recurrent one
    only for single-token decode.  Checking the chunked path against this scan
    is therefore a genuine cross-algorithm check, not a tautology.

    Shapes: q/k ``[B, S, H, K]``, v ``[B, S, H, V]``, g ``[B, S, H, K]``
    (per-key-channel decay), beta ``[B, S, H]``.
    """
    dtype = query.dtype
    query, key, value, g, beta = (t.float() for t in (query, key, value, g, beta))
    query = _ref_l2norm(query)
    key = _ref_l2norm(key)
    b, s, h, k_dim = key.shape
    v_dim = value.shape[-1]
    query = query * (query.shape[-1] ** -0.5)

    state = (
        torch.zeros(b, h, k_dim, v_dim, dtype=torch.float32, device=key.device)
        if initial_state is None
        else initial_state.float().clone()
    )
    out = torch.zeros(b, s, h, v_dim, dtype=torch.float32, device=key.device)
    for i in range(s):
        q_i, k_i, v_i = query[:, i], key[:, i], value[:, i]
        decay = g[:, i].exp().unsqueeze(-1)  # [B, H, K, 1]
        state = state * decay
        recalled = (state * k_i.unsqueeze(-1)).sum(dim=-2)  # [B, H, V]
        delta = (v_i - recalled) * beta[:, i].unsqueeze(-1)
        state = state + k_i.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, i] = (state * q_i.unsqueeze(-1)).sum(dim=-2)
    return out.to(dtype), state


class RefLinearAttention(nn.Module):
    """KDA linear-attention layer (``Glm5NextTextLinearAttention``)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int,
        lower_bound: float,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv_dim = num_heads * head_dim
        self.conv_kernel_size = conv_kernel_size
        self.conv_dim = self.qkv_dim * 3

        self.q_proj = nn.Linear(hidden_size, self.qkv_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.qkv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.qkv_dim, bias=False)
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            bias=False,
            kernel_size=conv_kernel_size,
            groups=self.conv_dim,
            padding=conv_kernel_size - 1,
        )
        self.forget_gate = RefForgetGate(hidden_size, num_heads, head_dim, lower_bound)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.g_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.g_b_proj = nn.Linear(head_dim, self.qkv_dim, bias=False)
        self.o_norm = RefRMSNormGated(head_dim, eps=rms_norm_eps)
        self.o_proj = nn.Linear(self.qkv_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
        recurrent_state: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ):
        if attention_mask is not None:
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)
        b, s = hidden_states.shape[:2]

        mixed = torch.cat(
            [self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)],
            dim=-1,
        ).transpose(1, 2)  # [B, 3*qkv, S]

        weight = self.conv1d.weight.squeeze(1)  # [conv_dim, kernel]
        if conv_state is None:
            history = torch.zeros(
                b,
                self.conv_dim,
                self.conv_kernel_size - 1,
                dtype=mixed.dtype,
                device=mixed.device,
            )
        else:
            history = conv_state.to(mixed.dtype)
        padded = torch.cat([history, mixed], dim=-1)
        new_conv_state = padded[:, :, -(self.conv_kernel_size - 1) :].clone()
        conv_out = F.conv1d(
            padded.to(weight.dtype), weight.unsqueeze(1), None, padding=0, groups=self.conv_dim
        )
        conv_out = F.silu(conv_out[:, :, -s:]).to(mixed.dtype)

        query, key, value = torch.split(conv_out.transpose(1, 2), [self.qkv_dim] * 3, dim=-1)
        shape = (b, s, self.num_heads, self.head_dim)
        query, key, value = query.view(shape), key.view(shape), value.view(shape)

        g = self.forget_gate(hidden_states)
        beta = torch.sigmoid(self.b_proj(hidden_states))

        core_out, new_recurrent_state = ref_kda_recurrent(
            query, key, value, g, beta, initial_state=recurrent_state
        )

        gate = self.g_b_proj(self.g_a_proj(hidden_states)).view(shape)
        out = self.o_norm(core_out, gate).reshape(b, s, -1)
        out = self.o_proj(out)
        if return_state:
            return out, new_conv_state, new_recurrent_state
        return out


class RefIndexer(nn.Module):
    """Pool-compressed DSA indexer (``Glm5NextTextIndexer``).

    Written as an explicit per-pool construction so it does not share the
    gather/mask trickery of the HuggingFace implementation.  Emits int32 indices
    of width ``index_topk + index_kpool - 1`` with ``-1`` in every unselected or
    invalid slot -- the ``-1`` is *source-defined*, must survive to the consumer,
    and must be masked (never clamped to a real position) before any gather.
    """

    def __init__(
        self,
        hidden_size: int,
        q_lora_rank: int,
        n_heads: int,
        head_dim: int,
        index_topk: int,
        index_kpool: int,
        always_select_tail: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.index_topk = index_topk
        self.index_kpool = index_kpool
        self.always_select_tail = always_select_tail
        self.softmax_scale = head_dim**-0.5

        self.wq_b = nn.Linear(q_lora_rank, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(hidden_size, head_dim, bias=False)
        self.k_norm = nn.LayerNorm(head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(hidden_size, n_heads, bias=False)
        self.index_kpool_compress_ape = nn.Parameter(torch.zeros(index_kpool, head_dim))
        self.index_kpool_compress_gate = nn.Parameter(torch.zeros(head_dim, hidden_size))

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        attention_mask: torch.Tensor,
        return_debug: bool = False,
    ):
        b, s = hidden_states.shape[:2]
        device = hidden_states.device
        pool = self.index_kpool

        q = self.wq_b(q_resid).view(b, s, self.n_heads, self.head_dim)
        k = self.k_norm(self.wk(hidden_states))  # [B, S, head_dim]
        gate_scores = F.linear(hidden_states, self.index_kpool_compress_gate)  # [B, S, head_dim]
        valid = attention_mask.bool()  # [B, S]

        # causal & padding visibility, [B, S(query), S(key)]
        positions = torch.arange(s, device=device)
        visible = (positions[None, :] >= positions[:, None]).T  # key <= query
        visible = visible[None].expand(b, s, s) & valid[:, None, :]

        # --- build pools, starting at the first *real* token -------------
        num_pools = (s + pool - 1) // pool
        first_key = torch.where(
            valid.any(-1),
            valid.long().argmax(-1),
            torch.full((b,), s, dtype=torch.long, device=device),
        )
        offsets = torch.arange(num_pools * pool, device=device).view(1, num_pools, pool)
        pool_indices = first_key[:, None, None] + offsets  # [B, P, pool]

        in_range = pool_indices < s
        safe = pool_indices.clamp(0, s - 1)
        bidx = torch.arange(b, device=device)[:, None, None]
        member_valid = valid[bidx, safe] & in_range  # [B, P, pool]
        pool_valid = member_valid.all(-1)  # only *complete* pools are candidates
        pool_indices = pool_indices.masked_fill(~member_valid, -1)

        # learned weighted average of the member keys
        logits = gate_scores[bidx, safe].float() + self.index_kpool_compress_ape.float()[None, None]
        logits = logits.masked_fill(~member_valid[..., None], float("-inf"))
        probs = torch.nan_to_num(logits.softmax(dim=2)).to(k.dtype)
        pool_keys = (probs * k[bidx, safe]).sum(dim=2)  # [B, P, head_dim]

        keep = pool_valid.any(0)
        pool_keys, pool_indices, pool_valid = (
            pool_keys[:, keep],
            pool_indices[:, keep],
            pool_valid[:, keep],
        )

        # --- score pools (not keys) --------------------------------------
        scores = torch.matmul(q.float(), pool_keys.transpose(-1, -2).float().unsqueeze(1))
        scores = F.relu(scores * self.softmax_scale)  # [B, S, n_heads, P]
        weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float()
        weights = weights * (self.n_heads**-0.5)
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)  # [B, S, P]

        # a pool is selectable only if its last member is visible to the query
        pool_end = pool_indices[..., -1].clamp(0, s - 1)
        pool_visible = visible.gather(-1, pool_end[:, None, :].expand(b, s, -1))
        candidates = pool_visible & pool_valid[:, None]
        index_scores = index_scores.masked_fill(~candidates, torch.finfo(index_scores.dtype).min)

        select_k = min(self.index_topk // pool, index_scores.shape[-1])
        selected = index_scores.topk(select_k, dim=-1).indices  # [B, S, K]
        selected_valid = candidates.gather(-1, selected)
        selected_indices = pool_indices[bidx, selected]  # [B, S, K, pool]

        topk = selected_indices.flatten(-2)
        topk = topk.masked_fill(
            ~selected_valid[..., None].expand_as(selected_indices).flatten(-2), -1
        )

        width = self.index_topk
        tail_indices = None
        if self.always_select_tail and pool > 1:
            max_tail = pool - 1
            tail_offsets = torch.arange(max_tail, device=device)
            visible_count = visible.long().sum(-1)  # [B, S]
            tail_count = visible_count.remainder(pool)
            tail_start = first_key[:, None] + visible_count - tail_count
            tail_indices = tail_start[..., None] + tail_offsets  # [B, S, pool-1]
            tail_ok = (tail_offsets[None, None, :] < tail_count[..., None]) & tail_indices.lt(s)
            tail_seen = visible.gather(-1, tail_indices.clamp(0, s - 1))
            tail_indices = tail_indices.masked_fill(~(tail_ok & tail_seen), -1)
            topk = torch.cat([topk, tail_indices], dim=-1)
            width += max_tail

        topk = F.pad(topk, (0, max(0, width - topk.shape[-1])), value=-1)[..., :width]
        topk = topk.masked_fill(~attention_mask.bool()[..., None], -1)
        topk = topk.to(torch.int32)

        if return_debug:
            return topk, {
                "num_pools_total": int(num_pools),
                "num_pools_kept": int(pool_keys.shape[1]),
                "select_k": int(select_k),
                "pool_keys": pool_keys,
                "index_scores": index_scores,
                "selected_pools": selected,
                "pool_indices": pool_indices,
                "tail_indices": tail_indices,
                "output_width": int(width),
            }
        return topk


def topk_indices_to_mask(topk_indices: torch.Tensor, kv_length: int) -> torch.Tensor:
    """Turn ``-1``-padded top-k indices into a boolean ``[B, 1, S, KV]`` mask.

    The ``-1`` slots are dropped by the validity test *before* the scatter; the
    clamp exists only so ``scatter_add_`` receives a legal index.  Reinterpreting
    ``-1`` as position 0 (or as ``kv_length - 1``) would silently attend to a
    real token.
    """
    valid = topk_indices.ge(0) & topk_indices.lt(kv_length)
    safe = topk_indices.clamp(0, kv_length - 1)
    counts = torch.zeros(
        topk_indices.shape[0],
        topk_indices.shape[1],
        kv_length,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    counts.scatter_add_(-1, safe, valid.to(torch.int32))
    return counts.ne(0).unsqueeze(1)


class RefSparseMLA(nn.Module):
    """Fully NoPE sparse MLA (``Glm5NextTextAttention``).

    ``qk_rope_head_dim == 0``: there is no rotary call, no rotary cache, and no
    positional term anywhere in this module.  Positional information reaches the
    model through the causal KDA layers and the indexer's learned pool APE.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        v_head_dim: int,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim  # rope dim is 0
        self.v_head_dim = v_head_dim
        self.scaling = self.qk_head_dim**-0.5

        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_a_layernorm = RefRMSNorm(q_lora_rank, eps=rms_norm_eps)
        self.q_b_proj = nn.Linear(q_lora_rank, num_heads * self.qk_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(hidden_size, kv_lora_rank, bias=False)
        self.kv_a_layernorm = RefRMSNorm(kv_lora_rank, eps=rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=False
        )
        self.o_proj = nn.Linear(num_heads * v_head_dim, hidden_size, bias=False)

    def project_q(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        b, s = hidden_states.shape[:2]
        q = self.q_b_proj(q_resid).view(b, s, self.num_heads, self.qk_head_dim).transpose(1, 2)
        return q_resid, q

    def project_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return the compressed latent KV, ``[B, S, kv_lora_rank]``."""
        return self.kv_a_layernorm(self.kv_a_proj_with_mqa(hidden_states))

    def expand_kv(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, s = latent.shape[:2]
        kv = self.kv_b_proj(latent).view(b, s, self.num_heads, -1).transpose(1, 2)
        k, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        return k, v

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        past_latent: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ):
        b, s = hidden_states.shape[:2]
        _, q = self.project_q(hidden_states)
        latent = self.project_kv(hidden_states)
        if past_latent is not None:
            latent = torch.cat([past_latent, latent], dim=1)
        key, value = self.expand_kv(latent)
        kv_len = key.shape[2]

        mask = topk_indices_to_mask(topk_indices, kv_len)
        logits = torch.matmul(q.float(), key.float().transpose(2, 3)) * self.scaling
        logits = logits.masked_fill(~mask, torch.finfo(torch.float32).min)
        probs = torch.softmax(logits, dim=-1).to(q.dtype)
        out = torch.matmul(probs, value).transpose(1, 2).reshape(b, s, -1)
        out = self.o_proj(out)
        if return_state:
            return out, latent
        return out


class RefMLP(nn.Module):
    """Clamped-SwiGLU dense MLP.

    The clamp is asymmetric and easy to get wrong: ``gate`` is clamped **only
    from above** at ``+limit`` while ``up`` is clamped on **both** sides.  A test
    that never drives an activation past 10.0 cannot tell this apart from plain
    SwiGLU.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, swiglu_limit: float):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x).clamp(min=None, max=self.swiglu_limit)
        up = self.up_proj(x).clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(F.silu(gate) * up)


class RefTopkRouter(nn.Module):
    """noaux_tc sigmoid router with FP32 logits and an e-score correction bias.

    ``n_group == topk_group == 1`` makes the group-limited stage degenerate: the
    single group is always selected, so the group mask is all-ones.  The routed
    weights are gathered from the *uncorrected* sigmoid scores, normalized, and
    only then scaled by ``routed_scaling_factor``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        routed_scaling_factor: float,
        norm_topk_prob: bool = True,
        n_group: int = 1,
        topk_group: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.n_group = n_group
        self.topk_group = topk_group
        self.weight = nn.Parameter(torch.zeros(num_experts, hidden_size))
        self.register_buffer("e_score_correction_bias", torch.zeros(num_experts))

    def forward(self, hidden_states: torch.Tensor):
        x = hidden_states.view(-1, self.hidden_size)
        router_logits = F.linear(x.float(), self.weight.float())
        scores = router_logits.sigmoid()
        corrected = scores + self.e_score_correction_bias

        if self.n_group > 1:
            per_group = self.num_experts // self.n_group
            group_scores = corrected.view(-1, self.n_group, per_group).topk(2, dim=-1)[0].sum(-1)
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            keep = group_mask.unsqueeze(-1).expand(-1, self.n_group, per_group)
            keep = keep.reshape(-1, self.num_experts).bool()
            corrected = corrected.masked_fill(~keep, float("-inf"))

        topk_indices = torch.topk(corrected, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return router_logits, topk_weights, topk_indices


class RefMoE(nn.Module):
    """Routed experts plus one always-active shared expert."""

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        top_k: int,
        routed_scaling_factor: float,
        swiglu_limit: float,
        n_shared_experts: int = 1,
        norm_topk_prob: bool = True,
        n_group: int = 1,
        topk_group: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.swiglu_limit = swiglu_limit
        self.gate = RefTopkRouter(
            hidden_size,
            num_experts,
            top_k,
            routed_scaling_factor,
            norm_topk_prob=norm_topk_prob,
            n_group=n_group,
            topk_group=topk_group,
        )
        # [E, 2*I, H] with gate rows first, matching the HF fused layout
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * moe_intermediate_size, hidden_size)
        )
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, moe_intermediate_size))
        self.shared_experts = RefMLP(
            hidden_size, moe_intermediate_size * n_shared_experts, swiglu_limit
        )

    def _expert(self, x: torch.Tensor, expert: int) -> torch.Tensor:
        gate_up = F.linear(x, self.gate_up_proj[expert])
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return F.linear(F.silu(gate) * up, self.down_proj[expert])

    def forward(self, hidden_states: torch.Tensor, return_debug: bool = False):
        shape = hidden_states.shape
        residual = hidden_states
        router_logits, topk_weights, topk_indices = self.gate(hidden_states)
        flat = hidden_states.view(-1, shape[-1])
        routed = torch.zeros_like(flat)
        for expert in torch.unique(topk_indices).tolist():
            slot_pos, token_idx = torch.where(topk_indices == expert)
            if token_idx.numel() == 0:
                continue
            contribution = self._expert(flat[slot_pos], expert)
            contribution = contribution * topk_weights[slot_pos, token_idx, None]
            routed.index_add_(0, slot_pos, contribution.to(routed.dtype))
        out = routed.view(*shape) + self.shared_experts(residual)
        if return_debug:
            return out, {
                "router_logits": router_logits,
                "topk_weights": topk_weights,
                "topk_indices": topk_indices,
                "routed_only": routed.view(*shape),
                "shared_only": self.shared_experts(residual),
            }
        return out


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

#: Every ``compare()`` call appends its metric dict here when
#: ``GLM53_COMPARE_EVIDENCE_JSON`` names an output path, labeled with the
#: pytest node that made the comparison (pytest exports PYTEST_CURRENT_TEST).
#: This exports the literal per-comparison ``max_abs``/``mean_abs``/``cosine``
#: values across *all* suites sharing this helper without touching any test
#: body. Dumped once at interpreter exit; the deliberately separate env var
#: keeps it from colliding with the suite-owned evidence fixtures that write
#: ``GLM53_REPLAY_EVIDENCE_JSON`` / ``GLM53_GRAPH_REPLAY_EVIDENCE_JSON``.
_COMPARE_EVIDENCE: List[Dict[str, Any]] = []
_COMPARE_EVIDENCE_REGISTERED = False


def _maybe_record_compare(metrics: Dict[str, Any]) -> None:
    out = os.environ.get("GLM53_COMPARE_EVIDENCE_JSON")
    if not out:
        return
    global _COMPARE_EVIDENCE_REGISTERED
    if not _COMPARE_EVIDENCE_REGISTERED:
        import atexit

        def _dump() -> None:
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            with open(out, "w") as fh:
                json.dump(_COMPARE_EVIDENCE, fh, indent=1, default=str)

        atexit.register(_dump)
        _COMPARE_EVIDENCE_REGISTERED = True
    entry = dict(metrics)
    entry["test"] = os.environ.get("PYTEST_CURRENT_TEST", "")
    entry["shape"] = list(entry.get("shape", ()))
    _COMPARE_EVIDENCE.append(entry)


def compare(actual: torch.Tensor, expected: torch.Tensor, name: str = "") -> Dict[str, Any]:
    """Report the metrics every activation-replay evidence item must carry."""
    a = actual.detach().float().flatten()
    e = expected.detach().float().flatten()
    if a.shape != e.shape:
        raise ValueError(f"{name}: shape mismatch {tuple(actual.shape)} vs {tuple(expected.shape)}")
    diff = (a - e).abs()
    denom = a.norm() * e.norm()
    cosine = float((a @ e) / denom) if float(denom) > 0 else float("nan")
    metrics = {
        "name": name,
        "max_abs": float(diff.max()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean()) if diff.numel() else 0.0,
        "cosine": cosine,
        "ref_max_abs": float(e.abs().max()) if e.numel() else 0.0,
        "all_finite": bool(torch.isfinite(a).all() and torch.isfinite(e).all()),
        "shape": tuple(actual.shape),
    }
    _maybe_record_compare(metrics)
    return metrics
