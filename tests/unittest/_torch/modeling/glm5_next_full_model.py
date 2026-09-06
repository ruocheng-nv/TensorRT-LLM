# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Whole-model driver for the GLM-5.3-Flash Stage-1 text path.

Stage 1 needs the complete 45-layer model to run against the real checkpoint
before any production backend exists, which raises two problems this module
solves and nothing else does:

*Placement.* The checkpoint is 328 GB in its published e4m3 form and a B200 has
183 GB, so no single device can hold it. The layers are laid out across the
visible GPUs by measured byte cost and the activation is handed forward across
devices, which is ordinary pipeline placement -- each device owns its own
layers' cache state exactly as a PP rank does in production.

*Cache.* Prefill and decode drive the model's own ``Glm5NextCacheManager``, one
instance per device covering that device's layers. The alternative -- flat
per-layer pools -- would have tested a mechanism the runtime does not use.

This is deliberately a Stage-1 diagnostic driver, not a runtime: it owns no
scheduler, no ``AttentionMetadata`` and no CUDA graph. Binding these modules to
the executor is the production-runtime goal's work, and when that lands this
driver is replaced rather than extended.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_glm5_next import (
    LINEAR_ATTENTION,
    SPARSE_ATTENTION,
    Glm5NextForCausalLM,
    build_glm5_next_quant_config,
    get_glm5_next_text_config,
    glm5_next_cache_manager_cls,
    resolve_glm5_next_schedule,
)

DEFAULT_CHECKPOINT = "/dev/shm/GLM-5.3-Flash"


class LazyCheckpoint:
    """Mapping view over a sharded safetensors checkpoint, without dequantizing.

    ``load_weights`` copies tensors exactly as published -- e4m3 payloads and
    their FP32 block scales -- so this deliberately does *not* wrap the
    dequantizing reader the module-level tests use. Materializing 76108 tensors
    as bf16 would be 656 GB.
    """

    def __init__(self, path: str = DEFAULT_CHECKPOINT):
        self.path = path
        with open(os.path.join(path, "model.safetensors.index.json")) as fh:
            self.weight_map: Dict[str, str] = json.load(fh)["weight_map"]
        self._handles: Dict[str, Any] = {}

    def keys(self):
        return self.weight_map.keys()

    def __contains__(self, key: str) -> bool:
        return key in self.weight_map

    def __getitem__(self, key: str) -> torch.Tensor:
        from safetensors import safe_open

        shard = self.weight_map[key]
        if shard not in self._handles:
            self._handles[shard] = safe_open(os.path.join(self.path, shard), "pt")
        return self._handles[shard].get_tensor(key)

    def nbytes(self, key: str) -> int:
        from safetensors import safe_open

        shard = self.weight_map[key]
        if shard not in self._handles:
            self._handles[shard] = safe_open(os.path.join(self.path, shard), "pt")
        sl = self._handles[shard].get_slice(key)
        elem = {"F8_E4M3": 1, "BF16": 2, "F16": 2, "F32": 4, "I64": 8}[sl.get_dtype()]
        size = elem
        for dim in sl.get_shape():
            size *= dim
        return size


def plan_device_map(
    weights: LazyCheckpoint,
    num_layers: int,
    devices: Sequence[torch.device],
) -> Dict[Any, torch.device]:
    """Assign owners to devices by measured byte cost, in layer order.

    Balancing by *measured* bytes rather than by layer count matters here: the
    three dense layers are 0.3 GB each while a routed layer is 7.4 GB, so an
    even split by count would overcommit whichever device owns the tail.
    """
    cost: Dict[Any, int] = {i: 0 for i in range(num_layers)}
    cost.update({"embed": 0, "norm": 0, "head": 0})
    for key in weights.keys():
        owner: Any
        if key.startswith("model.language_model.layers."):
            index = int(key.split(".")[3])
            if index >= num_layers:
                continue  # MTP, allowlisted out of scope
            owner = index
        elif key.startswith("model.language_model.embed_tokens."):
            owner = "embed"
        elif key.startswith("model.language_model.norm."):
            owner = "norm"
        elif key.startswith("lm_head."):
            owner = "head"
        else:
            continue
        cost[owner] += weights.nbytes(key)

    ordered: List[Any] = ["embed", *range(num_layers), "norm", "head"]
    total = sum(cost.values())
    budget = total / len(devices)
    device_map: Dict[Any, torch.device] = {}
    index, running = 0, 0
    for owner in ordered:
        # Keep the readout with the last layer: the head is only 0.6 GB and
        # moving the 154880-wide logits across a device boundary is pointless.
        if index < len(devices) - 1 and running + cost[owner] / 2 > budget * (index + 1):
            index += 1
        device_map[owner] = devices[index]
        running += cost[owner]
    return device_map


@dataclass
class DeviceStage:
    """The layers one device owns, plus the cache manager for exactly those."""

    device: torch.device
    layer_ids: List[int]
    manager: Any = None

    def close(self) -> None:
        if self.manager is not None:
            self.manager.shutdown()
            self.manager = None


@dataclass
class LoadedModel:
    model: Glm5NextForCausalLM
    config: Any
    text_config: Any
    schedule: Any
    device_map: Dict[Any, torch.device]
    stages: List[DeviceStage]
    load_report: Dict[str, Any]
    load_seconds: float = 0.0
    quant_placement: Dict[str, str] = field(default_factory=dict)
    #: Sequence budget the attached caches were sized for. Reported with every
    #: result so a score can never be silently a truncation artefact.
    cache_max_seq_len: int = 0

    @property
    def embed_device(self) -> torch.device:
        return self.device_map["embed"]

    @property
    def head_device(self) -> torch.device:
        return self.device_map["head"]


def load_full_model(
    checkpoint: str = DEFAULT_CHECKPOINT,
    devices: Optional[Sequence[Any]] = None,
    progress: bool = False,
) -> LoadedModel:
    """Materialize the whole text model across ``devices`` and fill it."""
    from transformers import AutoConfig

    if devices is None:
        devices = [torch.device("cuda", i) for i in range(torch.cuda.device_count())]
    devices = [torch.device(d) for d in devices]

    weights = LazyCheckpoint(checkpoint)
    config = AutoConfig.from_pretrained(checkpoint)
    text_config = get_glm5_next_text_config(config)
    schedule = resolve_glm5_next_schedule(config)
    # Quantization rides on the ModelConfig, exactly as the runtime's
    # ModelConfig.from_pretrained would set it, so the harness constructs the
    # same block-FP8 model AutoModelForCausalLM.from_config builds. The MoE
    # backend goes through the runtime's own AUTO resolution (FP8 block scales
    # on the SM100 family -> TRTLLM), so the harness runs the same production
    # routed-expert backend the serving path selects.
    quant_config = build_glm5_next_quant_config(config)
    model_config = ModelConfig(
        pretrained_config=config,
        quant_config=quant_config,
        moe_backend=ModelConfig.resolve_moe_backend(
            "AUTO",
            getattr(config, "architectures", ["Glm5NextForConditionalGeneration"])[0],
            quant_config,
        ),
    )

    with torch.device("meta"):
        model = Glm5NextForCausalLM(model_config)
    placement = model.apply_quant_plan(list(weights.keys()))

    device_map = plan_device_map(weights, schedule.num_layers, devices)
    started = time.time()
    report = model.load_weights(weights, device_map=device_map)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    load_seconds = time.time() - started
    if progress:
        print(f"[glm5] loaded in {load_seconds:.1f}s", flush=True)

    stages = []
    for device in devices:
        layer_ids = [i for i in range(schedule.num_layers) if device_map[i] == device]
        if layer_ids:
            stages.append(DeviceStage(device=device, layer_ids=layer_ids))
    return LoadedModel(
        model=model,
        config=config,
        text_config=text_config,
        schedule=schedule,
        device_map=device_map,
        stages=stages,
        load_report=report.summary(),
        load_seconds=load_seconds,
        quant_placement=placement,
    )


def attach_caches(
    loaded: LoadedModel,
    max_batch_size: int,
    max_seq_len: int,
    tokens_per_block: int = 64,
) -> None:
    """Give every device stage its own ``Glm5NextCacheManager``.

    One manager per device rather than one globally: a manager allocates its
    pools on the current device, and a layer must reach its own state without a
    cross-device index. This is the same ownership split a pipeline-parallel
    rank has, so it exercises the real class rather than a stand-in.
    """
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    text = loaded.text_config
    linear = dict(text.linear_attn_config)
    attention = list(text.layer_types)
    manager_cls = glm5_next_cache_manager_cls()
    loaded.cache_max_seq_len = max_seq_len

    pages_per_seq = (max_seq_len + tokens_per_block - 1) // tokens_per_block
    required_pages = pages_per_seq * max_batch_size

    def build(stage, mamba_mask, sparse_mask, sparse_ids, max_tokens):
        return manager_cls(
            mamba_d_state=int(linear["head_dim"]),
            mamba_d_conv=int(linear["short_conv_kernel_size"]),
            mamba_num_heads=int(linear["num_heads"]),
            mamba_n_groups=int(linear["num_heads"]),
            mamba_head_dim=int(linear["head_dim"]),
            mamba_num_layers=sum(mamba_mask),
            mamba_layer_mask=mamba_mask,
            mamba_cache_dtype=torch.bfloat16,
            mamba_ssm_cache_dtype=torch.float32,
            kv_cache_config=KvCacheConfig(max_tokens=max_tokens, enable_block_reuse=False),
            kv_cache_type=CacheTypeCpp.SELFKONLY,
            num_layers=max(sum(sparse_mask), 1),
            num_kv_heads=1,
            head_dim=int(text.kv_lora_rank),
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=Mapping(world_size=1, tp_size=1, pp_size=1),
            layer_mask=sparse_mask,
            dtype=DataType.BF16,
            conv_state_layout="q_k_v",
            sparse_layer_ids=sparse_ids,
            index_state_dim=2 * int(text.index_head_dim),
        )

    for stage in loaded.stages:
        owned = set(stage.layer_ids)
        # Masks span all 45 layers so global layer indices stay meaningful; only
        # this device's layers are marked, so only they get buffers here.
        mamba_mask = [i in owned and t == LINEAR_ATTENTION for i, t in enumerate(attention)]
        sparse_mask = [i in owned and t == SPARSE_ATTENTION for i, t in enumerate(attention)]
        sparse_ids = [i for i, on in enumerate(sparse_mask) if on]

        # `max_tokens` is a budget hint, not a slot count: V2 derives the pool
        # from it through its own accounting, and both the latent and INDEX_KEY
        # buffers are coalesced across this stage's sparse layers, so a stage
        # owning k of them gets 1/k of the raw pages as addressable slots.
        # Rather than reverse-engineer that arithmetic, ask for a budget and grow
        # it until the buffers V2 actually handed back are big enough. The pools
        # are megabytes, so overshooting is far cheaper than an out-of-range slot.
        max_tokens = 2 * max_batch_size * max_seq_len
        for _ in range(6):
            with torch.cuda.device(stage.device):
                manager = build(stage, mamba_mask, sparse_mask, sparse_ids, max_tokens)
            if not sparse_ids:
                stage.manager = manager
                break
            available = min(
                min(
                    manager.get_latent_state_buffer(layer_id).shape[0],
                    manager.get_index_state_buffer(layer_id).shape[0],
                )
                for layer_id in sparse_ids
            )
            if available >= required_pages:
                stage.manager = manager
                break
            manager.shutdown()
            max_tokens *= 2
        else:
            raise ValueError(
                f"{stage.device} could not allocate {required_pages} pages for "
                f"{max_batch_size} sequences of {max_seq_len} tokens"
            )


class Glm5NextGenerator:
    """Prefill plus greedy decode over the whole model, with real cache reuse.

    Sequences occupy fixed slots for their lifetime; every layer reaches its
    state through the slot id, never through a batch position, which is the
    aliasing bug the hybrid cache has to rule out.
    """

    def __init__(self, loaded: LoadedModel, tokens_per_block: int = 64):
        self.loaded = loaded
        self.model = loaded.model
        self.tokens_per_block = tokens_per_block
        self.lengths: List[int] = []
        self.block_tables: Dict[torch.device, torch.Tensor] = {}

    # -- cache plumbing ---------------------------------------------------

    def _reset(self, batch_size: int, max_seq_len: int) -> None:
        """Allocate page tables per device and zero all recurrent state."""
        pages = (max_seq_len + self.tokens_per_block - 1) // self.tokens_per_block
        self.lengths = [0] * batch_size
        self.block_tables = {}
        for stage in self.loaded.stages:
            sparse = [
                i for i in stage.layer_ids if self.loaded.schedule.attention[i] == SPARSE_ATTENTION
            ]
            if sparse:
                # Every sparse layer on the stage, not just the first: the
                # slot count of a coalesced pool can differ by one between
                # layers, and the smallest one is what bounds the block table.
                available = min(
                    min(pool.shape[0] for pool in self._pools(stage, layer_id))
                    for layer_id in sparse
                )
                if pages * batch_size > available:
                    raise ValueError(
                        f"glm5_next needs {pages * batch_size} slots for "
                        f"{batch_size} sequences of {max_seq_len} tokens but "
                        f"{stage.device} allocated {available}; "
                        "raise attach_caches(max_seq_len=...)"
                    )
            # Interleaved pages: request r owns pages r, r+batch, r+2*batch...
            # A layer that assumed a flat per-request buffer would silently read
            # another request's keys, and a contiguous table would hide it.
            table = torch.arange(pages * batch_size, device=stage.device, dtype=torch.long)
            self.block_tables[stage.device] = table.view(pages, batch_size).t().contiguous()
            manager = stage.manager
            for layer_id in stage.layer_ids:
                if self.loaded.schedule.attention[layer_id] == LINEAR_ATTENTION:
                    cache = manager.mamba_layer_cache(layer_id)
                    cache.conv.zero_()
                    cache.temporal.zero_()
                else:
                    # Clear through the slot views, not the page-indexed one:
                    # zeroing the latter would also wipe the neighbouring sparse
                    # layers' slots, which is only harmless while every layer is
                    # being reset in the same pass.
                    for pool in self._pools(stage, layer_id):
                        pool.zero_()

    @staticmethod
    def _pools(stage: DeviceStage, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """The layer's latent and indexer pools as ``[slots, tokens, dim]`` views.

        Both are zero-copy views over V2-managed memory, so writes propagate to
        the pool. The singleton axes are dropped by *indexing*, never by
        ``reshape``: both buffers are strided and reshaping them would copy,
        turning every cache write into a write to a temporary.

        Both accessors are **slot**-indexed. Using ``get_buffers`` directly here
        was a real defect, not a style choice: V2 coalesces every sparse layer's
        latent into one pool, so its page-indexed view overlaps the neighbouring
        layer's by all but one page, and every cached position older than
        ``tokens_per_block`` steps was being overwritten by another layer.
        """
        latent = stage.manager.get_latent_state_buffer(layer_id)  # [slots, tokens, heads, dim]
        index = stage.manager.get_index_state_buffer(layer_id)  # [slots, tokens, heads, dim]
        return latent[:, :, 0, :], index[:, :, 0, :]

    def _layer_kwargs(
        self, stage: DeviceStage, layer_id: int, phase: str, cu_seqlens, cached
    ) -> Dict[str, Any]:
        manager = stage.manager
        batch = len(cached)
        if self.loaded.schedule.attention[layer_id] == LINEAR_ATTENTION:
            cache = manager.mamba_layer_cache(layer_id)
            slots = torch.arange(batch, device=stage.device, dtype=torch.long)
            kwargs = {
                "slot_ids": slots,
                "conv_pool": cache.conv,
                "ssm_pool": cache.temporal,
            }
            if phase == "prefill":
                kwargs.update(cu_seqlens=cu_seqlens, cached_lens=cached)
            return kwargs
        # Sparse layers consume a prepared-metadata carrier: the backend
        # derives the slot-indexed pools from the stage's real manager and the
        # tables/lengths from the glm_* buffers, exactly the runtime contract.
        from types import SimpleNamespace

        kv_lens = torch.tensor([c + 1 for c in cached], device=stage.device, dtype=torch.long)
        metadata = SimpleNamespace(
            kv_cache_manager=manager,
            mamba_metadata=SimpleNamespace(
                glm_block_tables=self.block_tables[stage.device][:batch],
                glm_kv_lens=kv_lens,
            ),
            seq_lens=torch.ones(batch, dtype=torch.long),
            num_contexts=batch if phase == "prefill" else 0,
            is_cuda_graph=False,
        )
        kwargs = {"metadata": metadata}
        if phase == "prefill":
            kwargs.update(cu_seqlens=cu_seqlens, cached_lens=cached)
        else:
            # Batched decode consumes device kv_lens (cached + the one token
            # being decoded), matching the runtime's CUDA-graph-safe contract.
            kwargs["kv_lens"] = kv_lens
        return kwargs

    # -- execution --------------------------------------------------------

    def _run_stack(self, streams: torch.Tensor, phase: str, cu_seqlens, cached) -> torch.Tensor:
        for stage in self.loaded.stages:
            # The current device must follow the data. Plain torch ops infer it
            # from their arguments, but the block-FP8 custom ops launch on the
            # *current* device, so running a cuda:3 layer while cuda:0 is current
            # hands the kernel foreign pointers -- an illegal access that
            # surfaces later, at whatever unrelated call next synchronizes.
            with torch.cuda.device(stage.device):
                streams = streams.to(stage.device, non_blocking=True)
                for layer_id in stage.layer_ids:
                    layer = self.model.model.layers[layer_id]
                    kwargs = self._layer_kwargs(stage, layer_id, phase, cu_seqlens, cached)
                    streams = layer.forward_direct(streams, phase=phase, **kwargs)
        return streams

    def _logits(self, streams: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        head = self.loaded.head_device
        with torch.cuda.device(head):
            hidden = self.model.model.collapse_streams(streams.to(head))
            return self.model.lm_head(hidden[rows.to(head)]).float()

    @torch.inference_mode()
    def prefill(self, prompts: Sequence[Sequence[int]]) -> torch.Tensor:
        """Run context phase for a packed batch; return each prompt's last logits."""
        batch = len(prompts)
        lengths = [len(p) for p in prompts]
        budget = self.loaded.cache_max_seq_len
        if budget < max(lengths):
            raise ValueError(
                f"cache budget {budget} is shorter than the longest prompt "
                f"({max(lengths)}); every reported result would be a truncation"
            )
        self._reset(batch, budget)
        cu = [0]
        for length in lengths:
            cu.append(cu[-1] + length)
        flat = torch.tensor(
            [t for prompt in prompts for t in prompt],
            device=self.loaded.embed_device,
            dtype=torch.long,
        )
        with torch.cuda.device(self.loaded.embed_device):
            embeds = self.model.model.embed_tokens(flat)
            streams = self.model.model.expand_streams(embeds)
        streams = self._run_stack(streams, "prefill", cu, [0] * batch)
        self.lengths = list(lengths)
        last = torch.tensor([cu[i + 1] - 1 for i in range(batch)], dtype=torch.long)
        return self._logits(streams, last)

    @torch.inference_mode()
    def decode(self, tokens: Sequence[int]) -> torch.Tensor:
        """Advance every sequence by one token; return the next-token logits."""
        cached = list(self.lengths)
        ids = torch.tensor(list(tokens), device=self.loaded.embed_device, dtype=torch.long)
        with torch.cuda.device(self.loaded.embed_device):
            embeds = self.model.model.embed_tokens(ids)
            streams = self.model.model.expand_streams(embeds)
        streams = self._run_stack(streams, "decode", None, cached)
        self.lengths = [n + 1 for n in cached]
        rows = torch.arange(len(cached), dtype=torch.long)
        return self._logits(streams, rows)

    @torch.inference_mode()
    def generate(
        self,
        prompts: Sequence[Sequence[int]],
        max_new_tokens: int,
        forced: Optional[Sequence[Sequence[int]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Greedy generation.

        Returns ``(prefill_logits, step_logits, tokens)`` where ``step_logits``
        is ``[batch, max_new_tokens, vocab]``.

        ``forced`` teacher-forces the *input* of each step from a reference run
        while still reporting this model's own argmax, so a divergence at step
        k does not contaminate every later step through a different prefix.
        """
        prefill_logits = self.prefill(prompts)
        chosen = prefill_logits.argmax(-1)
        steps, tokens = [], [chosen]
        for step in range(max_new_tokens - 1):
            feed = (
                chosen
                if forced is None
                else torch.tensor(
                    [seq[step] for seq in forced], device=chosen.device, dtype=chosen.dtype
                )
            )
            logits = self.decode(feed.tolist())
            steps.append(logits)
            chosen = logits.argmax(-1)
            tokens.append(chosen)
        step_logits = (
            torch.stack(steps, dim=1)
            if steps
            else prefill_logits.new_zeros((len(prompts), 0, prefill_logits.shape[-1]))
        )
        return prefill_logits, step_logits, torch.stack(tokens, dim=1)

    @torch.inference_mode()
    def generate_until_eos(
        self,
        prompts: Sequence[Sequence[int]],
        max_new_tokens: int,
        eos_token_ids: Sequence[int],
        progress_every: int = 0,
    ) -> List[Dict[str, Any]]:
        """Free-running greedy generation that stops each sequence at EOS.

        A finished sequence keeps stepping -- the batch advances in lockstep and
        this driver has no scheduler to evict it -- but its later tokens are
        discarded, and whether it stopped on EOS or ran into ``max_new_tokens``
        is returned rather than inferred. That distinction is the difference
        between a score and a measurement of the decode budget.
        """
        eos = {int(t) for t in eos_token_ids}
        batch = len(prompts)
        logits = self.prefill(prompts)
        chosen = logits.argmax(-1)
        emitted: List[List[int]] = [[] for _ in range(batch)]
        finished = [False] * batch
        for index in range(batch):
            token = int(chosen[index])
            emitted[index].append(token)
            finished[index] = token in eos

        for step in range(1, max_new_tokens):
            if all(finished):
                break
            logits = self.decode(chosen.tolist())
            chosen = logits.argmax(-1)
            for index in range(batch):
                if finished[index]:
                    continue
                token = int(chosen[index])
                emitted[index].append(token)
                if token in eos:
                    finished[index] = True
            if progress_every and step % progress_every == 0:
                print(
                    f"[glm5] step {step}/{max_new_tokens}, {sum(finished)}/{batch} finished",
                    flush=True,
                )
        return [
            {
                "tokens": emitted[index],
                "num_generated": len(emitted[index]),
                "stopped_on_eos": finished[index],
            }
            for index in range(batch)
        ]

    def close(self) -> None:
        for stage in self.loaded.stages:
            stage.close()
