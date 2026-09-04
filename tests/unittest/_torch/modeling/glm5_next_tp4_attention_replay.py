# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Goal 5.2 four-rank TP4 production-attention replay driver (Stage 5).

Run under ``mpirun -n 4`` on four CUDA GPUs::

    mpirun -n 4 python glm5_next_tp4_attention_replay.py --json out.json

What one invocation proves, per the Stage-5 attention acceptance item:

1. **Ownership geometry** — the full TP4 model (built by the Goal-5.1
   swap + shard-aware loader) gives every KDA/sparse-MLA layer exactly 16 of
   64 local heads, the matching absorbed kv_b / q / output slices, eight of 32
   indexer scoring heads plus the one FP32 score all-reduce, and a
   full-width (replicated) latent/index path.
2. **Real cache manager at TP4** — a real ``Glm5NextCacheManager``
   (KVCacheManagerV2 hybrid) built with the four-rank Mapping over the full
   45-layer schedule allocates KDA pools ``[slots, 6144, 3]`` (bf16 conv) and
   ``[slots, 16, 128, 128]`` (fp32 recurrent), 512-wide latent pages and
   256-wide packed index state, and drives prefill / chunked continuation /
   decode / cache reuse through the prepared runtime metadata
   (``real_runtime`` at module scope).
3. **source_activation_replay (B)** — hooked native-HF hidden states from the
   real checkpoint (the Stage-1 fixture) run through every captured KDA layer
   (0/2/22/44) and sparse layer (3/23/43) at TP4; the all-reduced outputs are
   compared against the in-model hooked HF output (FP8 envelope) and the
   bf16-dequant single-rank module — the same reference class the accepted
   PP4 (Stage-3) replay used, i.e. the accepted PP4 single-rank-per-layer
   form. KDA recurrent/conv state slices must match the single-rank state's
   head slices.
4. **Identical selection on every rank** — for every sparse leg the
   pool/tail/-1 selection is recorded per rank and must be bitwise identical
   across all four ranks (the FP32 SUM all-reduce hands every rank the same
   reduced score); Jaccard vs the single-rank selection is reported.
5. **Length ladder + isolation** — selection and attention at 37 (below),
   512 (at pool capacity), and 2100 (above index_topk=2048) tokens; batched
   request isolation, mid-request cancellation, and slot reuse are bitwise
   per rank.
6. **B then E** — the eager legs run first; the E leg captures each family's
   decode step in a ``torch.cuda.CUDAGraph`` **with the collectives inside**
   (KDA row o_proj all-reduce; indexer FP32 score all-reduce + MLA row
   o_proj all-reduce) in lockstep on all four ranks and replays it with
   fresh inputs. Capture success is the no-fallback hard-path proof; replays
   must match eager within the predeclared envelope.

Module-scope conventions follow the accepted Stage-3/Goal-5.1 precedent:
``overlap_scheduler`` is a serving-level property (Goal 5.4); E here means
CUDA-graph capture/replay of the production module forwards with collectives
captured.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Dict, List, Optional, Sequence

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from glm5_next_tp4_dense_loader import CHECKPOINT, check_envelope, log, metrics  # noqa: E402
from glm5_next_tp4_dense_loader import Driver as DenseDriver

FIXTURE = os.environ.get(
    "GLM53_HF_FIXTURE",
    os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            )
        ),
        "agent-flow/workspace/glm-5.3-flash-bringup/reports/hf_reference_fixture.pt",
    ),
)
LAYER_PREFIX = "model.language_model.layers"

#: The fixture's captured attention layers, split by family (the literal
#: 45-entry schedule: linear at 0/2/22/44, sparse at 3/23/43 among captures).
KDA_LAYERS = (0, 2, 22, 44)
MLA_LAYERS = (3, 23, 43)

#: Envelopes. The TP4 layers run the production block-FP8 kernels while both
#: references are bf16 (the hooked in-model output came from HF's own FP8
#: path) — these are the Stage-1/3-accepted FP8-replay bounds.
FP8_MODEL_ENVELOPE = {"cosine": 0.995, "rel_max_abs": 8e-2}
#: TP4 (fp8 kernels) vs the bf16-dequant single-rank module — the accepted
#: PP4 single-rank-per-layer form (Stage-3's replay reference class).
PP4_FORM_ENVELOPE = {"cosine": 0.995, "rel_max_abs": 8e-2}
#: Same-path consistency (chunked vs one-shot on identical weights/kernels).
SAME_PATH_ENVELOPE = {"cosine": 0.9995, "rel_max_abs": 2e-2}
#: Per-head state slices vs the single-rank module state (bf16 working
#: precision; fp8-vs-bf16 projections feed the state, so wider than the
#: unit tier's identical-weight slice check).
STATE_ENVELOPE = {"cosine": 0.995, "rel_max_abs": 8e-2}
#: Graph replay vs eager on identical inputs; every family here contains at
#: least one collective inside the capture.
GRAPH_ENVELOPE = {"cosine": 0.9999, "rel_max_abs": 5e-3}

# below pool capacity / at pool capacity 128 / exactly at index_topk=2048 /
# above index_topk. 2048 is the exact boundary the criterion names: the pool
# budget fills to the top-k limit with no expansion headroom, so the tail and
# the compressed-pool selection meet at the width edge.
SYNTH_LENGTHS = (37, 512, 2048, 2100)


def _hidden(num_tokens: int, seed: int, device) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(num_tokens, 4096, generator=gen, dtype=torch.float32)
    return (x * 0.05).to(device=device, dtype=torch.bfloat16)


class _SelectRecorder:
    """Wrap an indexer's ``select`` to record every emitted index tensor."""

    def __init__(self, indexer):
        self.indexer = indexer
        self.rows: List[torch.Tensor] = []
        self._orig = indexer.select

    def __enter__(self):
        def wrapped(*args, **kwargs):
            out = self._orig(*args, **kwargs)
            self.rows.append(out.detach().clone())
            return out

        self.indexer.select = wrapped
        return self

    def __exit__(self, *exc):
        self.indexer.select = self._orig
        return False


def _jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    """Min per-row Jaccard similarity of two [T, W] index tensors (-1 ignored)."""
    worst = 1.0
    for row_a, row_b in zip(a.tolist(), b.tolist()):
        sa = {v for v in row_a if v >= 0}
        sb = {v for v in row_b if v >= 0}
        if not sa and not sb:
            continue
        worst = min(worst, len(sa & sb) / max(len(sa | sb), 1))
    return worst


class AttnDriver(DenseDriver):
    def __init__(self):
        super().__init__("tp4")
        self.result["driver"] = "glm5_next_tp4_attention_replay"
        self.result["fixture"] = FIXTURE

    # ------------------------------------------------------------------
    # references
    # ------------------------------------------------------------------

    def _load_reader(self):
        from glm5_next_ref import CheckpointReader

        if not hasattr(self, "_reader"):
            self._reader = CheckpointReader(CHECKPOINT)
        return self._reader

    def _text_config(self):
        if not hasattr(self, "_text_cfg"):
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(CHECKPOINT)
            self._text_cfg = cfg.text_config
        return self._text_cfg

    def _tp1_kda(self, layer_idx: int):
        """The accepted single-rank (PP4-form) KDA module: bf16 dequant weights.

        Identical construction to the Stage-3 replay reference
        (``test_glm5_next_attention._load_kda`` minus the HF twin).
        """
        from test_glm5_next_attention import _KDA_RENAMED, _KDA_SHARED

        from tensorrt_llm._torch.models.modeling_glm5_next import Glm5NextLinearAttention

        cache = self.__dict__.setdefault("_tp1_kda_cache", {})
        if layer_idx in cache:
            return cache[layer_idx]
        reader = self._load_reader()
        p = f"{LAYER_PREFIX}.{layer_idx}.self_attn"
        trt = Glm5NextLinearAttention(self._text_config(), layer_idx).to(self.device).eval()
        params = dict(trt.named_parameters())
        with torch.no_grad():
            for name in _KDA_SHARED:
                params[name].copy_(
                    reader.get(f"{p}.{name}").to(device=self.device, dtype=torch.bfloat16)
                )
            for trt_name, (_hf, suffix) in _KDA_RENAMED.items():
                value = reader.get(f"{p}.{suffix}").to(self.device)
                params[trt_name].copy_(value.to(params[trt_name].dtype))
            conv = torch.cat(
                [reader.get(f"{p}.{n}_conv1d.weight").to(self.device) for n in ("q", "k", "v")],
                dim=0,
            )
            params["conv1d.weight"].copy_(conv.view_as(params["conv1d.weight"]))
        cache[layer_idx] = trt
        return trt

    def _tp1_mla(self, layer_idx: int):
        """The accepted single-rank (PP4-form) sparse-MLA module (bf16 dequant)."""
        from test_glm5_next_attention import _MLA_SHARED

        from tensorrt_llm._torch.models.modeling_glm5_next import Glm5NextSparseAttention

        cache = self.__dict__.setdefault("_tp1_mla_cache", {})
        if layer_idx in cache:
            return cache[layer_idx]
        reader = self._load_reader()
        p = f"{LAYER_PREFIX}.{layer_idx}.self_attn"
        trt = Glm5NextSparseAttention(self._text_config(), layer_idx).to(self.device).eval()
        params = dict(trt.named_parameters())
        with torch.no_grad():
            for name in _MLA_SHARED:
                params[name].copy_(
                    reader.get(f"{p}.{name}").to(device=self.device, dtype=torch.bfloat16)
                )
        cache[layer_idx] = trt
        return trt

    def _fixture_prompt(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        if not hasattr(self, "_fixture"):
            self._fixture = torch.load(FIXTURE, map_location="cpu", weights_only=False)
        for prompt in self._fixture["prompts"]:
            acts = prompt.get("activations") or {}
            key_in, key_out = (
                f"layer{layer_idx}.self_attn.input",
                f"layer{layer_idx}.self_attn.output",
            )
            if key_in in acts and key_out in acts:
                return {
                    "prompt_index": prompt["index"],
                    "x": acts[key_in][0].to(device=self.device, dtype=torch.bfloat16),
                    "y": acts[key_out][0].to(device=self.device, dtype=torch.float32),
                }
        raise KeyError(f"fixture has no self_attn capture for layer {layer_idx}")

    # ------------------------------------------------------------------
    # cross-rank helpers
    # ------------------------------------------------------------------

    def _assert_identical_across_ranks(self, t: torch.Tensor, label: str) -> bool:
        rows = self.comm.allgather(t.detach().to("cpu"))
        same = all(torch.equal(rows[0], r) for r in rows[1:])
        if not same:
            self.problems.append(f"{label}: not bitwise identical across ranks")
        return same

    # ------------------------------------------------------------------
    # phases
    # ------------------------------------------------------------------

    def geometry(self) -> None:
        rows = []
        head_ranges, chan_ranges = [], []
        for layer_idx in KDA_LAYERS:
            kda = self.model.model.layers[layer_idx].self_attn
            row = {
                "layer": layer_idx,
                "family": "kda",
                "num_heads": kda.num_heads,
                "total_num_heads": kda.total_num_heads,
                "conv_weight": list(kda.conv1d.weight.shape),
                "A_log": list(kda.A_log.shape),
                "dt_bias": list(kda.dt_bias.shape),
                "head_range": list(kda.kda_head_range()),
                "channel_range": list(kda.kda_channel_range()),
                "q_proj_local": list(kda.q_proj.weight.shape),
                "o_proj_local": list(kda.o_proj.weight.shape),
                "prefill_dispatch": type(kda._kda_dispatch).__name__,
            }
            ok = (
                kda.num_heads == 16
                and kda.total_num_heads == 64
                and tuple(kda.conv1d.weight.shape) == (6144, 1, 4)
                and tuple(kda.A_log.shape) == (16,)
                and tuple(kda.dt_bias.shape) == (2048,)
                and tuple(kda.q_proj.weight.shape)[0] == 2048
            )
            row["pass"] = ok
            if not ok:
                self.problems.append(f"geometry kda layer {layer_idx}: {row}")
            rows.append(row)
        head_ranges = self.comm.allgather(
            list(self.model.model.layers[KDA_LAYERS[0]].self_attn.kda_head_range())
        )
        chan_ranges = self.comm.allgather(
            list(self.model.model.layers[KDA_LAYERS[0]].self_attn.kda_channel_range())
        )
        if sorted(map(tuple, head_ranges)) != [(0, 16), (16, 32), (32, 48), (48, 64)]:
            self.problems.append(f"geometry: kda head ranges do not tile 0..64: {head_ranges}")
        if sorted(map(tuple, chan_ranges)) != [
            (0, 2048),
            (2048, 4096),
            (4096, 6144),
            (6144, 8192),
        ]:
            self.problems.append(f"geometry: kda channel ranges do not tile: {chan_ranges}")

        for layer_idx in MLA_LAYERS:
            attn = self.model.model.layers[layer_idx].self_attn
            idx = attn.indexer
            row = {
                "layer": layer_idx,
                "family": "sparse_mla",
                "num_heads": attn.num_heads,
                "backend_class": type(attn.attn_backend).__name__,
                "backend_num_heads": attn.attn_backend.num_heads,
                "backend_head_dim": attn.attn_backend.head_dim,
                "backend_num_kv_heads": attn.attn_backend.num_kv_heads,
                "q_b_local": list(attn.q_b_proj.weight.shape),
                "kv_b_local": list(attn.kv_b_proj.weight.shape),
                "o_proj_local": list(attn.o_proj.weight.shape),
                "indexer_local_heads": idx.n_heads,
                "indexer_total_heads": idx.total_n_heads,
                "wq_b_local": list(idx.wq_b.weight.shape),
                "weights_proj_local": list(idx.weights_proj.weight.shape),
                "wk_replicated": list(idx.wk.weight.shape),
                "score_all_reduce": type(idx.score_all_reduce).__name__
                if idx.score_all_reduce is not None
                else None,
            }
            w_k, w_v_t = attn.absorbed_kv_b()
            row["absorbed_w_k"] = list(w_k.shape)
            row["absorbed_w_v_t"] = list(w_v_t.shape)
            ok = (
                attn.num_heads == 16
                and type(attn.attn_backend).__name__ == "GlmKpoolSparseAttention"
                and attn.attn_backend.num_heads == 16
                and attn.attn_backend.head_dim == 512
                and attn.attn_backend.num_kv_heads == 1
                and idx.n_heads == 8
                and idx.total_n_heads == 32
                and idx.score_all_reduce is not None
                and tuple(idx.wq_b.weight.shape) == (1024, 1536)
                and tuple(idx.weights_proj.weight.shape) == (8, 4096)
                and tuple(idx.wk.weight.shape) == (128, 4096)
                and tuple(w_k.shape) == (16, 256, 512)
                and tuple(w_v_t.shape) == (16, 512, 256)
            )
            row["pass"] = ok
            if not ok:
                self.problems.append(f"geometry mla layer {layer_idx}: {row}")
            rows.append(row)
        self.result["geometry"] = rows
        self.comm.Barrier()

    def manager_leg(self) -> None:
        """Real Glm5NextCacheManager at TP4: shapes + runtime-path execution."""
        from test_glm5_next_attention import _real_prepared_metadata

        from tensorrt_llm._torch.models.modeling_glm5_next import (
            build_glm5_next_runtime_context,
            glm5_next_cache_manager_cls,
        )
        from tensorrt_llm.bindings import DataType
        from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
        from tensorrt_llm.llmapi.llm_args import KvCacheConfig

        text_config = self._text_config()
        linear = dict(text_config.linear_attn_config)
        attention = list(text_config.layer_types)
        sparse_ids = [i for i, t in enumerate(attention) if t == "deepseek_sparse_attention"]
        linear_ids = [i for i, t in enumerate(attention) if t == "linear_attention"]
        max_seq_len = 4352

        manager = glm5_next_cache_manager_cls()(
            mamba_d_state=linear["head_dim"],
            mamba_d_conv=linear["short_conv_kernel_size"],
            mamba_num_heads=linear["num_heads"],
            mamba_n_groups=linear["num_heads"],
            mamba_head_dim=linear["head_dim"],
            mamba_num_layers=len(linear_ids),
            mamba_layer_mask=[t == "linear_attention" for t in attention],
            mamba_cache_dtype=torch.bfloat16,
            mamba_ssm_cache_dtype=torch.float32,
            kv_cache_config=KvCacheConfig(max_tokens=8 * max_seq_len * 4, enable_block_reuse=False),
            kv_cache_type=CacheTypeCpp.SELFKONLY,
            num_layers=len(sparse_ids),
            num_kv_heads=1,
            head_dim=int(text_config.kv_lora_rank),
            tokens_per_block=32,
            max_seq_len=max_seq_len,
            max_batch_size=4,
            mapping=self.mapping,
            layer_mask=[t == "deepseek_sparse_attention" for t in attention],
            dtype=DataType.BF16,
            conv_state_layout="q_k_v",
            sparse_layer_ids=sparse_ids,
            index_state_dim=2 * int(text_config.index_head_dim),
        )
        self._manager = manager
        kda_layer, mla_layer = KDA_LAYERS[0], MLA_LAYERS[0]
        conv = manager.get_conv_states(kda_layer)
        ssm = manager.get_ssm_states(kda_layer)
        latent = manager.get_latent_state_buffer(mla_layer)
        index = manager.get_index_state_buffer(mla_layer)
        row = {
            "manager_class": type(manager).__name__,
            "manager_bases": [b.__name__ for b in type(manager).__mro__[1:4]],
            "conv_pool": list(conv.shape),
            "ssm_pool": list(ssm.shape),
            "latent_page_shape": list(latent.shape),
            "index_page_shape": list(index.shape),
        }
        ok = (
            tuple(conv.shape[1:]) == (6144, 3)
            and tuple(ssm.shape[1:]) == (16, 128, 128)
            and ssm.dtype == torch.float32
            and latent.shape[-1] == 512
            and index.shape[-1] == 256
        )
        row["pass"] = ok
        if not ok:
            self.problems.append(f"manager shapes at TP4: {row}")

        # Runtime-path execution: add_dummy_requests → prepared TRTLLM
        # metadata → runtime-context-derived module forwards, i.e. exactly the
        # engine's argument path (real_runtime at module scope). One-shot
        # prefill for request A, chunked continuation for request B, then a
        # batched decode step for both — through the same manager.
        # 512 one-shot tokens = 8 chunks of 64: the CuTe ``trtllm::kda_prefill``
        # dispatch gate (>= 4 chunks on Blackwell) accepts the batch, so this
        # leg proves the production prefill kernel at 16 local heads.
        seq_a, chunk1, chunk2 = 512, 65, 64
        request_ids = [11, 12]
        # Horizon covers the prefill tokens plus the decode step below.
        manager.add_dummy_requests(request_ids, token_nums=[seq_a + 2, chunk1 + chunk2 + 2])
        kda = self.model.model.layers[kda_layer].self_attn
        mla = self.model.model.layers[mla_layer].self_attn
        x_a = _hidden(seq_a, 101, self.device)
        x_b = _hidden(chunk1 + chunk2, 102, self.device)

        with torch.no_grad():
            md = _real_prepared_metadata(
                self._manager,
                lens=[seq_a, chunk1],
                num_contexts=2,
                cached=[0, 0],
                request_ids=request_ids,
            )
            ctx = build_glm5_next_runtime_context(md)
            kda_out1 = kda.forward_prefill(
                torch.cat([x_a, x_b[:chunk1]]), **ctx.linear_kwargs(kda_layer, "prefill")
            )
            first_prefill_path = kda.last_prefill_path
            mla_out1 = mla.forward_prefill(
                torch.cat([x_a, x_b[:chunk1]]), **ctx.sparse_kwargs(mla_layer, "prefill")
            )
            md2 = _real_prepared_metadata(
                self._manager,
                lens=[chunk2],
                num_contexts=1,
                cached=[chunk1],
                request_ids=[request_ids[1]],
            )
            ctx2 = build_glm5_next_runtime_context(md2)
            kda_out2 = kda.forward_prefill(x_b[chunk1:], **ctx2.linear_kwargs(kda_layer, "prefill"))
            mla_out2 = mla.forward_prefill(x_b[chunk1:], **ctx2.sparse_kwargs(mla_layer, "prefill"))

            step = _hidden(2, 103, self.device)
            md3 = _real_prepared_metadata(
                self._manager,
                lens=[1, 1],
                num_contexts=0,
                cached=[seq_a, chunk1 + chunk2],
                request_ids=request_ids,
            )
            ctx3 = build_glm5_next_runtime_context(md3)
            kda_dec = kda.forward_decode(step, **ctx3.linear_kwargs(kda_layer, "decode"))
            mla_dec = mla.forward_decode(step, **ctx3.sparse_kwargs(mla_layer, "decode"))

        finite = all(
            bool(torch.isfinite(t).all())
            for t in (kda_out1, kda_out2, mla_out1, mla_out2, kda_dec, mla_dec)
        )
        row["runtime_path"] = {
            "kda_prefill_path": first_prefill_path,
            "kda_continuation_path": kda.last_prefill_path,
            "kda_decode_path": kda.decode_step_path,
            "requests": {"one_shot": seq_a, "chunked": [chunk1, chunk2]},
            "decode_rows": int(kda_dec.shape[0]),
            "outputs_finite": finite,
        }
        if not finite:
            self.problems.append("manager runtime path produced non-finite outputs")
        if first_prefill_path != "trtllm::kda_prefill":
            self.problems.append(
                "manager runtime path: the 512-token one-shot leg did not dispatch "
                f"trtllm::kda_prefill at 16 local heads (got {first_prefill_path})"
            )
        self.result["manager_leg"] = row
        self.comm.Barrier()

    # -- fixture replay (B) -------------------------------------------------

    def _run_kda_tp4(
        self, kda, x: torch.Tensor, chunks: Optional[Sequence[int]] = None, decode_steps: int = 2
    ):
        """Run one TP4 KDA layer on module-scope local pools; return outputs+state."""
        slots = 1
        conv = torch.zeros(slots, kda.conv_dim, 3, device=self.device, dtype=torch.bfloat16)
        ssm = torch.zeros(slots, kda.num_heads, 128, 128, device=self.device, dtype=torch.float32)
        slot = torch.tensor([0], device=self.device)
        seq_len = x.shape[0]
        with torch.no_grad():
            if chunks:
                pieces, cached = [], 0
                for size in chunks:
                    pieces.append(
                        kda.forward_prefill(
                            x[cached : cached + size],
                            [0, size],
                            slot,
                            conv,
                            ssm,
                            cached_lens=[cached],
                        )
                    )
                    cached += size
                out = torch.cat(pieces, dim=0)
            else:
                out = kda.forward_prefill(x, [0, seq_len], slot, conv, ssm, cached_lens=[0])
            decs = []
            for step in range(decode_steps):
                decs.append(kda.forward_decode(x[step : step + 1], slot, conv, ssm))
        return out, decs, conv, ssm

    def _run_mla_tp4(
        self, attn, x: torch.Tensor, chunks: Optional[Sequence[int]] = None, decode_steps: int = 2
    ):
        """Run one TP4 sparse layer on stand-in pools; record per-rank topk."""
        from test_glm5_next_attention import _kpool_metadata, _KpoolPools

        seq_len = x.shape[0]
        tokens_per_block = 32
        pages = seq_len // tokens_per_block + 2
        pools = _KpoolPools(attn, pages, tokens_per_block, self.device)
        table = torch.arange(pages, device=self.device, dtype=torch.long).unsqueeze(0)

        def ctx_md(cached):
            kv = torch.zeros(1, device=self.device, dtype=torch.long)
            return _kpool_metadata(pools, block_tables=table, kv_lens=kv, num_contexts=1)

        with torch.no_grad(), _SelectRecorder(attn.indexer) as rec:
            if chunks:
                pieces, cached = [], 0
                for size in chunks:
                    pieces.append(
                        attn.forward_prefill(
                            x[cached : cached + size], [0, size], [cached], ctx_md(cached)
                        )
                    )
                    cached += size
                out = torch.cat(pieces, dim=0)
            else:
                out = attn.forward_prefill(x, [0, seq_len], [0], ctx_md(0))
            decs = []
            for step in range(decode_steps):
                kv_lens = torch.tensor([seq_len + step + 1], device=self.device)
                decs.append(
                    attn.forward_decode(
                        x[step : step + 1],
                        kv_lens,
                        _kpool_metadata(pools, block_tables=table, kv_lens=kv_lens, num_contexts=0),
                    )
                )
        return out, decs, rec.rows

    def fixture_replay(self) -> None:
        """source_activation_replay: hooked HF activations through TP4 layers."""
        rows = []
        for layer_idx in KDA_LAYERS:
            fx = self._fixture_prompt(layer_idx)
            x = fx["x"]
            kda = self.model.model.layers[layer_idx].self_attn
            tp1 = self._tp1_kda(layer_idx)
            out4, decs4, conv4, ssm4 = self._run_kda_tp4(kda, x)
            chunk_split = [x.shape[0] // 2, x.shape[0] - x.shape[0] // 2]
            out4_chunked, _, _, _ = self._run_kda_tp4(kda, x, chunks=chunk_split, decode_steps=0)
            out1, decs1, conv1, ssm1 = self._run_kda_tp4(tp1, x)

            hd = kda.kda_head_range()
            ch = kda.kda_channel_range()
            m_hf = metrics(out4.float(), fx["y"].to(self.device))
            m_pp4 = metrics(out4.float(), out1.float())
            m_chunk = metrics(out4_chunked.float(), out4.float())
            m_ssm = metrics(ssm4[0], ssm1[0, hd[0] : hd[1]])
            conv_ref = torch.cat(
                [conv1[0, s * 8192 + ch[0] : s * 8192 + ch[1]] for s in range(3)], dim=0
            )
            m_conv = metrics(conv4[0].float(), conv_ref.float())
            m_dec = metrics(decs4[-1].float(), decs1[-1].float())
            row = {
                "layer": layer_idx,
                "family": "kda",
                "prompt_index": fx["prompt_index"],
                "tokens": int(x.shape[0]),
                "prefill_path": kda.last_prefill_path,
                "vs_hooked_hf": m_hf,
                "vs_pp4_form": m_pp4,
                "chunked_vs_one_shot": m_chunk,
                "ssm_slice_vs_pp4": m_ssm,
                "conv_slice_vs_pp4": m_conv,
                "decode_vs_pp4": m_dec,
                "local_pool_shapes": {
                    "conv": list(conv4.shape[1:]),
                    "ssm": list(ssm4.shape[1:]),
                },
            }
            checks = [
                ("vs_hooked_hf", m_hf, FP8_MODEL_ENVELOPE),
                ("vs_pp4_form", m_pp4, PP4_FORM_ENVELOPE),
                ("chunked_vs_one_shot", m_chunk, SAME_PATH_ENVELOPE),
                ("ssm_slice_vs_pp4", m_ssm, STATE_ENVELOPE),
                ("conv_slice_vs_pp4", m_conv, STATE_ENVELOPE),
                ("decode_vs_pp4", m_dec, PP4_FORM_ENVELOPE),
            ]
            row["pass"] = True
            for label, m, env in checks:
                bad = check_envelope(m, env)
                if bad:
                    row["pass"] = False
                    self.problems.append(f"B kda layer {layer_idx} {label}: {bad}")
            rows.append(row)

        for layer_idx in MLA_LAYERS:
            fx = self._fixture_prompt(layer_idx)
            x = fx["x"]
            attn = self.model.model.layers[layer_idx].self_attn
            tp1 = self._tp1_mla(layer_idx)
            out4, decs4, topk4 = self._run_mla_tp4(attn, x)
            chunk_split = [x.shape[0] // 2, x.shape[0] - x.shape[0] // 2]
            out4_chunked, _, _ = self._run_mla_tp4(attn, x, chunks=chunk_split, decode_steps=0)
            out1, decs1, topk1 = self._run_mla_tp4(tp1, x)

            m_hf = metrics(out4.float(), fx["y"].to(self.device))
            m_pp4 = metrics(out4.float(), out1.float())
            m_chunk = metrics(out4_chunked.float(), out4.float())
            m_dec = metrics(decs4[-1].float(), decs1[-1].float())
            sel_identical = all(
                self._assert_identical_across_ranks(t, f"B mla layer {layer_idx} topk[{i}]")
                for i, t in enumerate(topk4)
            )
            jac = min(_jaccard(a, b) for a, b in zip(topk4, topk1))
            row = {
                "layer": layer_idx,
                "family": "sparse_mla",
                "prompt_index": fx["prompt_index"],
                "tokens": int(x.shape[0]),
                "vs_hooked_hf": m_hf,
                "vs_pp4_form": m_pp4,
                "chunked_vs_one_shot": m_chunk,
                "decode_vs_pp4": m_dec,
                "selection_identical_across_ranks": sel_identical,
                "selection_jaccard_vs_pp4": jac,
                "select_calls": len(topk4),
            }
            checks = [
                ("vs_hooked_hf", m_hf, FP8_MODEL_ENVELOPE),
                ("vs_pp4_form", m_pp4, PP4_FORM_ENVELOPE),
                ("chunked_vs_one_shot", m_chunk, SAME_PATH_ENVELOPE),
                ("decode_vs_pp4", m_dec, PP4_FORM_ENVELOPE),
            ]
            row["pass"] = sel_identical and jac >= 0.99
            for label, m, env in checks:
                bad = check_envelope(m, env)
                if bad:
                    row["pass"] = False
                    self.problems.append(f"B mla layer {layer_idx} {label}: {bad}")
            if jac < 0.99:
                self.problems.append(
                    f"B mla layer {layer_idx}: selection Jaccard vs PP4 form {jac:.4f} < 0.99"
                )
            rows.append(row)
        self.result["fixture_replay_B"] = rows
        self.comm.Barrier()

    def kda_production_prefill(self) -> None:
        """CuTe ``trtllm::kda_prefill`` parity at 16 local heads vs the PP4 form.

        The hooked-fixture prompts are short (24 tokens), which takes the
        documented small-batch torch route; this leg runs 512 tokens (8
        chunks) so both the TP4 layer and the single-rank reference dispatch
        the production CuTe kernel, then compares outputs, decode steps, and
        per-head state slices.
        """
        rows = []
        for layer_idx in (KDA_LAYERS[0], KDA_LAYERS[-1]):
            kda = self.model.model.layers[layer_idx].self_attn
            tp1 = self._tp1_kda(layer_idx)
            x = _hidden(512, 500 + layer_idx, self.device)
            out4, decs4, conv4, ssm4 = self._run_kda_tp4(kda, x)
            path4 = kda.last_prefill_path
            out1, decs1, conv1, ssm1 = self._run_kda_tp4(tp1, x)
            path1 = tp1.last_prefill_path
            hd = kda.kda_head_range()
            m_out = metrics(out4.float(), out1.float())
            m_dec = metrics(decs4[-1].float(), decs1[-1].float())
            m_ssm = metrics(ssm4[0], ssm1[0, hd[0] : hd[1]])
            row = {
                "layer": layer_idx,
                "tokens": 512,
                "tp4_prefill_path": path4,
                "pp4_form_prefill_path": path1,
                "decode_path": kda.decode_step_path,
                "vs_pp4_form": m_out,
                "decode_vs_pp4": m_dec,
                "ssm_slice_vs_pp4": m_ssm,
            }
            row["pass"] = (
                path4 == "trtllm::kda_prefill"
                and path1 == "trtllm::kda_prefill"
                and check_envelope(m_out, PP4_FORM_ENVELOPE) is None
                and check_envelope(m_dec, PP4_FORM_ENVELOPE) is None
                and check_envelope(m_ssm, STATE_ENVELOPE) is None
            )
            if not row["pass"]:
                self.problems.append(
                    f"kda production prefill layer {layer_idx}: paths=({path4},{path1}) "
                    f"out={check_envelope(m_out, PP4_FORM_ENVELOPE)} "
                    f"dec={check_envelope(m_dec, PP4_FORM_ENVELOPE)} "
                    f"ssm={check_envelope(m_ssm, STATE_ENVELOPE)}"
                )
            rows.append(row)
        self.result["kda_production_prefill"] = rows
        self.comm.Barrier()

    def length_ladder(self) -> None:
        """Selection + attention at 37 / 512 / 2048 / 2100 tokens.

        Covers below pool capacity, at pool capacity, exactly at the
        index_topk=2048 boundary, and above it.
        """
        rows = []
        for layer_idx in (MLA_LAYERS[0], MLA_LAYERS[-1]):
            attn = self.model.model.layers[layer_idx].self_attn
            tp1 = self._tp1_mla(layer_idx)
            for seq_len in SYNTH_LENGTHS:
                x = _hidden(seq_len, 1000 + layer_idx * 7 + seq_len, self.device)
                out4, decs4, topk4 = self._run_mla_tp4(attn, x, decode_steps=1)
                out1, decs1, topk1 = self._run_mla_tp4(tp1, x, decode_steps=1)
                m_pp4 = metrics(out4.float(), out1.float())
                m_dec = metrics(decs4[-1].float(), decs1[-1].float())
                sel_identical = all(
                    self._assert_identical_across_ranks(
                        t, f"ladder mla layer {layer_idx} len {seq_len} topk[{i}]"
                    )
                    for i, t in enumerate(topk4)
                )
                jac = min(_jaccard(a, b) for a, b in zip(topk4, topk1))
                # Sentinel / width contract on the decode-step selection.
                last = topk4[-1]
                width_ok = last.shape[-1] == attn.indexer.output_width
                valid = last[last >= 0]
                bounds_ok = bool((valid < seq_len + 1).all()) if valid.numel() else False
                row = {
                    "layer": layer_idx,
                    "seq_len": seq_len,
                    "regime": "below_2048"
                    if seq_len < 2048
                    else ("at_or_above_2048" if seq_len >= 2048 else "at"),
                    "vs_pp4_form": m_pp4,
                    "decode_vs_pp4": m_dec,
                    "selection_identical_across_ranks": sel_identical,
                    "selection_jaccard_vs_pp4": jac,
                    "output_width": int(last.shape[-1]),
                    "sentinel_bounds_ok": bounds_ok,
                }
                row["pass"] = (
                    sel_identical
                    and width_ok
                    and bounds_ok
                    and jac >= 0.99
                    and check_envelope(m_pp4, PP4_FORM_ENVELOPE) is None
                    and check_envelope(m_dec, PP4_FORM_ENVELOPE) is None
                )
                if not row["pass"]:
                    self.problems.append(
                        f"ladder mla layer {layer_idx} len {seq_len}: "
                        f"identical={sel_identical} jaccard={jac:.4f} "
                        f"pp4={check_envelope(m_pp4, PP4_FORM_ENVELOPE)} "
                        f"dec={check_envelope(m_dec, PP4_FORM_ENVELOPE)} "
                        f"width_ok={width_ok} bounds_ok={bounds_ok}"
                    )
                rows.append(row)
        self.result["length_ladder"] = rows
        self.comm.Barrier()

    def isolation_and_reuse(self) -> None:
        """Request isolation, cancellation, slot reuse — bitwise per rank."""
        from test_glm5_next_attention import _kpool_metadata, _KpoolPools

        rows = []
        layer_idx = MLA_LAYERS[0]
        attn = self.model.model.layers[layer_idx].self_attn
        tokens_per_block, seq_len = 32, 256
        pages_per_req = seq_len // tokens_per_block + 1
        pools = _KpoolPools(attn, 2 * pages_per_req + 2, tokens_per_block, self.device)
        table = torch.tensor(
            [
                [2 * i for i in range(pages_per_req)],
                [2 * i + 1 for i in range(pages_per_req)],
            ],
            device=self.device,
            dtype=torch.long,
        )
        x0 = _hidden(seq_len, 11, self.device)
        x1 = _hidden(seq_len, 12, self.device)

        def ctx_md(rows_t, cached):
            kv = torch.zeros(len(cached), device=self.device, dtype=torch.long)
            return _kpool_metadata(
                pools, block_tables=rows_t, kv_lens=kv, num_contexts=rows_t.shape[0]
            )

        with torch.no_grad():
            both = attn.forward_prefill(
                torch.cat([x0, x1]), [0, seq_len, 2 * seq_len], [0, 0], ctx_md(table, [0, 0])
            )
            isolation = metrics(both[seq_len:].float(), both[:seq_len].float())
            # Cancellation mid-prefill, slot handed to a new request.
            pools.latent.zero_()
            pools.index.zero_()
            attn.forward_prefill(x1[:100], [0, 100], [0], ctx_md(table[1:], [0]))
            reused = attn.forward_prefill(x0, [0, seq_len], [0], ctx_md(table[1:], [0]))
            pools.latent.zero_()
            pools.index.zero_()
            fresh = attn.forward_prefill(x0, [0, seq_len], [0], ctx_md(table[1:], [0]))
        reuse = metrics(reused.float(), fresh.float())
        mla_row = {
            "layer": layer_idx,
            "family": "sparse_mla",
            "request_isolation_differs": isolation["max_abs"] > 0.1,
            "slot_reuse_after_cancel": reuse,
        }
        mla_row["pass"] = mla_row["request_isolation_differs"] and reuse["bitwise"]
        if not mla_row["pass"]:
            self.problems.append(f"isolation mla: {mla_row}")
        rows.append(mla_row)

        kda_layer = KDA_LAYERS[0]
        kda = self.model.model.layers[kda_layer].self_attn
        conv = torch.zeros(2, kda.conv_dim, 3, device=self.device, dtype=torch.bfloat16)
        ssm = torch.zeros(2, kda.num_heads, 128, 128, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            kda.forward_prefill(
                x1[:100],
                [0, 100],
                torch.tensor([1], device=self.device),
                conv,
                ssm,
                cached_lens=[0],
            )
            reused_k = kda.forward_prefill(
                x0, [0, seq_len], torch.tensor([1], device=self.device), conv, ssm, cached_lens=[0]
            )
            conv.zero_()
            ssm.zero_()
            fresh_k = kda.forward_prefill(
                x0, [0, seq_len], torch.tensor([1], device=self.device), conv, ssm, cached_lens=[0]
            )
        reuse_k = metrics(reused_k.float(), fresh_k.float())
        kda_row = {
            "layer": kda_layer,
            "family": "kda",
            "slot_reuse_after_cancel": reuse_k,
            "pass": reuse_k["bitwise"],
        }
        if not kda_row["pass"]:
            self.problems.append(f"isolation kda: {kda_row}")
        rows.append(kda_row)
        self.result["isolation"] = rows
        self.comm.Barrier()

    # -- CUDA graph (E) -------------------------------------------------------

    def replay_graph_attention(self) -> None:
        """E leg: capture each family's decode (collectives inside) and replay."""
        from test_glm5_next_attention import _kpool_metadata, _KpoolPools

        rows = []
        for layer_idx in KDA_LAYERS[:1] + KDA_LAYERS[-1:]:
            kda = self.model.model.layers[layer_idx].self_attn
            prefix = _hidden(160, 300 + layer_idx, self.device)
            conv = torch.zeros(1, kda.conv_dim, 3, device=self.device, dtype=torch.bfloat16)
            ssm = torch.zeros(1, kda.num_heads, 128, 128, device=self.device, dtype=torch.float32)
            slot = torch.tensor([0], device=self.device)
            with torch.no_grad():
                kda.forward_prefill(prefix, [0, 160], slot, conv, ssm, cached_lens=[0])
            static_x = _hidden(1, 301 + layer_idx, self.device)
            row = {"layer": layer_idx, "family": "kda", "collectives": ["o_proj all-reduce"]}
            try:
                self.comm.Barrier()
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                conv_w, ssm_w = conv.clone(), ssm.clone()
                with torch.cuda.stream(side), torch.no_grad():
                    for _ in range(3):
                        kda.forward_decode(static_x, slot, conv_w, ssm_w)
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                self.comm.Barrier()
                conv_g, ssm_g = conv.clone(), ssm.clone()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph), torch.no_grad():
                    static_y = kda.forward_decode(static_x, slot, conv_g, ssm_g)
                row["captured"] = True
                fresh = _hidden(1, 302 + layer_idx, self.device)
                static_x.copy_(fresh)
                conv_g.copy_(conv)
                ssm_g.copy_(ssm)
                self.comm.Barrier()
                graph.replay()
                torch.cuda.synchronize()
                replayed = static_y.detach().clone()
                conv_e, ssm_e = conv.clone(), ssm.clone()
                with torch.no_grad():
                    eager = kda.forward_decode(fresh, slot, conv_e, ssm_e)
                m = metrics(replayed.float(), eager.float())
                m_state = metrics(ssm_g[0], ssm_e[0])
                row.update({"replay_vs_eager": m, "state_replay_vs_eager": m_state})
                bad = check_envelope(m, GRAPH_ENVELOPE)
                row["pass"] = bad is None and m_state["finite"]
                if bad:
                    self.problems.append(f"E kda layer {layer_idx}: {bad}")
                del graph
            except Exception as exc:
                row.update(
                    {"captured": False, "error": f"{type(exc).__name__}: {exc}", "pass": False}
                )
                self.problems.append(f"E kda layer {layer_idx} capture/replay failed: {exc}")
            rows.append(row)

        for layer_idx in (MLA_LAYERS[0], MLA_LAYERS[-1]):
            attn = self.model.model.layers[layer_idx].self_attn
            tokens_per_block, prefix_len = 32, 200
            pages = prefix_len // tokens_per_block + 3
            pools = _KpoolPools(attn, pages, tokens_per_block, self.device)
            table = torch.arange(pages, device=self.device, dtype=torch.long).unsqueeze(0)
            prefix = _hidden(prefix_len, 400 + layer_idx, self.device)
            with torch.no_grad():
                kv0 = torch.zeros(1, device=self.device, dtype=torch.long)
                attn.forward_prefill(
                    prefix,
                    [0, prefix_len],
                    [0],
                    _kpool_metadata(pools, block_tables=table, kv_lens=kv0, num_contexts=1),
                )
            kv_buf = torch.tensor([prefix_len + 1], device=self.device)
            decode_md = _kpool_metadata(
                pools, block_tables=table, kv_lens=kv_buf, num_contexts=0, is_cuda_graph=True
            )
            static_x = _hidden(1, 401 + layer_idx, self.device)
            row = {
                "layer": layer_idx,
                "family": "sparse_mla",
                "collectives": ["indexer fp32 score all-reduce", "o_proj all-reduce"],
            }
            try:
                self.comm.Barrier()
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side), torch.no_grad():
                    for _ in range(3):
                        attn.forward_decode(static_x, kv_buf, decode_md)
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                self.comm.Barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph), torch.no_grad():
                    static_y = attn.forward_decode(static_x, kv_buf, decode_md)
                row["captured"] = True
                # Replay 1: fresh input at the same length; replay 2: length+1
                # (cache growth), proving replays track refreshed buffers.
                results = []
                for bump, seed in ((0, 402 + layer_idx), (1, 403 + layer_idx)):
                    fresh = _hidden(1, seed, self.device)
                    static_x.copy_(fresh)
                    kv_buf.fill_(prefix_len + 1 + bump)
                    self.comm.Barrier()
                    graph.replay()
                    torch.cuda.synchronize()
                    replayed = static_y.detach().clone()
                    with torch.no_grad(), _SelectRecorder(attn.indexer) as rec_e:
                        eager = attn.forward_decode(fresh, kv_buf.clone(), decode_md)
                    m = metrics(replayed.float(), eager.float())
                    results.append(m)
                    bad = check_envelope(m, GRAPH_ENVELOPE)
                    if bad:
                        self.problems.append(f"E mla layer {layer_idx} replay(bump={bump}): {bad}")
                    sel_ok = self._assert_identical_across_ranks(
                        rec_e.rows[-1], f"E mla layer {layer_idx} eager topk bump={bump}"
                    )
                    if not sel_ok:
                        self.problems.append(
                            f"E mla layer {layer_idx}: eager-side selection diverged across ranks"
                        )
                row["replays_vs_eager"] = results
                row["pass"] = all(check_envelope(m, GRAPH_ENVELOPE) is None for m in results)
                del graph
            except Exception as exc:
                row.update(
                    {"captured": False, "error": f"{type(exc).__name__}: {exc}", "pass": False}
                )
                self.problems.append(f"E mla layer {layer_idx} capture/replay failed: {exc}")
            rows.append(row)
        self.result["replay_E"] = rows
        self.comm.Barrier()

    # -- orchestration --------------------------------------------------------

    def run(self, json_path: str) -> int:  # noqa: D102 — see module docstring
        try:
            t0 = time.time()
            self.build()
            self.geometry()
            self.manager_leg()
            self.fixture_replay()
            self.kda_production_prefill()
            self.length_ladder()
            self.isolation_and_reuse()
            self.replay_graph_attention()
            self.result["driver_seconds"] = round(time.time() - t0, 1)
        except Exception:
            self.problems.append(f"driver exception: {traceback.format_exc(limit=8)}")
        all_problems = self.comm.gather(list(self.problems), root=0)
        gathered = self.comm.gather(self.result, root=0)
        code = 0
        if self.rank == 0:
            merged = [p for ps in all_problems for p in ps]
            ok = not merged
            out = {
                "driver": "glm5_next_tp4_attention_replay",
                "layout": "tp4",
                "ok": ok,
                "problems": merged,
                "backend": "GlmKpoolSparseAttention (TRTLLM sparse family)",
                "cache_manager": "Glm5NextCacheManager (KVCacheManagerV2 hybrid)",
                "conventions": [
                    "E = module-scope CUDA-graph capture/replay with collectives inside "
                    "(accepted Stage-3/Goal-5.1 partial_model convention)",
                    "overlap_scheduler is a serving-level property owned by Goal 5.4",
                    "PP4-form reference = the bf16-dequant single-rank modules the accepted "
                    "Stage-3 replay validated (same class, same weights, full 64 heads)",
                ],
                "envelopes": {
                    "fp8_model": FP8_MODEL_ENVELOPE,
                    "pp4_form": PP4_FORM_ENVELOPE,
                    "same_path": SAME_PATH_ENVELOPE,
                    "state": STATE_ENVELOPE,
                    "graph": GRAPH_ENVELOPE,
                },
                "ranks": gathered,
            }
            with open(json_path, "w") as f:
                json.dump(out, f, indent=1, default=str)
            log(0, f"attention replay ok={ok} problems={len(merged)}")
            code = 0 if ok else 1
        code = self.comm.bcast(code, root=0)
        return code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    return AttnDriver().run(args.json)


if __name__ == "__main__":
    rc = main()
    with open(os.environ.get("GLM5_EXIT_FILE", "/tmp/glm5_tp4_attn_exit.txt"), "w") as f:
        f.write(str(rc))
