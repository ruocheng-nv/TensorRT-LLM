# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""source_activation_replay under CUDA graph capture/replay (criterion-2 E leg).

The B leg replays hooked native-HF activations from the real checkpoint
through the TensorRT-LLM modules eagerly. This file is the E leg of the same
claim: the *same* real-checkpoint weights and the *same* hooked activations,
but with the decode step executed inside a captured ``torch.cuda.CUDAGraph``
and driven purely through replays — the module-level equivalent of
``CudaGraphConfig()`` at ``test_type=partial_model`` scope.

Each test proves three things per replayed position:

1. **Hard path** — the decode step captures without error and every replay
   reads fresh state from persistent device buffers (capture would raise on
   any H2D copy or host sync, so a successful capture *is* the no-fallback
   proof at this scope);
2. **Graph == eager** — the replayed output and every cache write are
   bitwise-equal to an eager call over identical state (``torch.equal``);
3. **Graph == source** — the replayed output row still sits inside the same
   predeclared FP8-replay envelope against the hooked in-model output that
   the eager B leg uses, with ``max_abs``/``mean_abs``/``cosine`` recorded.

Pool tensors use the geometry ``Glm5NextCacheManager`` (KVCacheManagerV2)
hands out; manager-owned addressing itself is proven by
``test_coalesced_latent_pool_is_addressed_by_slot`` and the paged prefill/
decode tests in ``test_glm5_next_attention.py``. Fixture prompts are short
(tens of tokens), so pool selection runs in the below-``index_topk`` regime;
checkpoint-scale graph capture of the integrated FP8 path is covered by the
end-to-end config-E LLM API evidence on the same tree.

Numeric evidence is exported when ``GLM53_GRAPH_REPLAY_EVIDENCE_JSON`` names
a path (deliberately a different variable from the eager suite's
``GLM53_REPLAY_EVIDENCE_JSON`` so one combined pytest invocation can export
both without the module teardowns overwriting each other).

Stage-4 (graph-safe serving) additions:

* :func:`test_sparse_mla_decode_graph_replay_crosses_index_topk` — ONE
  captured decode replayed across the ``index_topk=2048`` boundary, the
  regime where pool selection first becomes lossy. Every engine-scale graph
  run so far decoded well below 2048, so the fixed-capacity ``-1`` index
  buffers had no capture-scope coverage in the lossy regime.
* :func:`test_batched_decode_graph_padding_slot_isolation` — the CUDA-graph
  *padding* contract at module scope: when the runner pads a decode batch
  with a dummy request, the dummy's cache slots and input row must not be
  able to perturb any real row (its state is arbitrary by design). Proven by
  poisoning the dummy slot state (NaN) and dummy input between replays of
  one captured batched decode and requiring the real rows bitwise-unchanged.
* :func:`test_padding_dummy_allocation_on_real_manager` — the engine-side
  premise of padding: ``CUDAGraphRunner._get_or_create_padding_dummy`` calls
  ``add_dummy_requests(..., is_gen=True)`` on the cache manager; this pins
  that ``Glm5NextCacheManager`` (KVCacheManagerV2 hybrid) actually honors it
  from its reserved dummy slot while real requests hold every regular slot,
  and that the prepared metadata gives the dummy valid, non-colliding KDA
  state and sparse page rows.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List

import pytest
import torch
from glm5_next_ref import CheckpointReader, compare  # noqa: E402  (test-dir import)
from test_glm5_next_attention import (  # noqa: E402
    _build_cache_manager,
    _hidden,
    _kpool_metadata,
    _KpoolPools,
    _load_kda,
    _load_mla,
    _real_prepared_metadata,
)
from test_glm5_next_decoder import _load_decoder_layer  # noqa: E402

from tensorrt_llm._torch.models.modeling_glm5_next import (
    Glm5NextLinearAttention,
    Glm5NextMoE,
    Glm5NextSparseAttention,
    resolve_glm5_next_schedule,
)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 4))
CHECKPOINT = os.environ.get("GLM53_CHECKPOINT", "/dev/shm/GLM-5.3-Flash")
FIXTURE = os.environ.get(
    "GLM53_HF_FIXTURE",
    os.path.join(
        _REPO_ROOT, "agent-flow/workspace/glm-5.3-flash-bringup/reports/hf_reference_fixture.pt"
    ),
)

# The same predeclared bounds the eager B-leg suites assert: the module output
# was produced in-model by block-FP8 kernels while this rung runs on
# bf16-dequantized weights, so agreement with the in-model output is bounded
# by the FP8 envelope while graph-vs-eager agreement must be *bitwise*.
FP8_REPLAY_MODEL_COSINE = 0.995
FP8_REPLAY_MODEL_REL_MAX_ABS = 8e-2

TOKENS_PER_BLOCK = 32

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graphs")


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda")


@pytest.fixture(scope="module")
def full_config():
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(CHECKPOINT)
    config.text_config._attn_implementation = "eager"
    return config


@pytest.fixture(scope="module")
def text_config(full_config):
    return full_config.text_config


@pytest.fixture(scope="module")
def schedule(full_config):
    return resolve_glm5_next_schedule(full_config)


@pytest.fixture(scope="module")
def reader():
    r = CheckpointReader(CHECKPOINT)
    yield r
    r.close()


@pytest.fixture(scope="module")
def hooked():
    assert os.path.isfile(FIXTURE), (
        f"missing the native-HF activation fixture at {FIXTURE}; build it with "
        f"glm5_next_hf_reference.py first"
    )
    payload = torch.load(FIXTURE, map_location="cpu", weights_only=False)
    prompts = [p for p in payload["prompts"] if p.get("activations")]
    assert prompts, "fixture has no captured activations"
    return prompts


@pytest.fixture(scope="module")
def evidence():
    bucket: Dict[str, List[dict]] = {}
    yield bucket
    out = os.environ.get("GLM53_GRAPH_REPLAY_EVIDENCE_JSON")
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            json.dump(bucket, fh, indent=2, default=str)


def _record(evidence, key, payload):
    evidence.setdefault(key, []).append(payload)


def _first_prompt_with(hooked, layer_idx: int, kind: str) -> dict:
    key_in = f"layer{layer_idx}.{kind}.input" if kind else f"layer{layer_idx}.input"
    key_out = key_in.replace(".input", ".output")
    for prompt in hooked:
        if key_in in prompt["activations"] and key_out in prompt["activations"]:
            return prompt
    raise AssertionError(f"fixture has no prompt with {key_in}/{key_out}")


def _graph_capture(decode_fn):
    """Warm up on a side stream, then capture one decode call.

    The warmup runs execute for real, so callers must re-seed any cache state
    the warmup polluted *before* relying on replays. Capture success is itself
    the hard-path proof at module scope: any host->device copy, ``.tolist()``,
    or host sync inside the step raises ``operation not permitted when stream
    is capturing`` and fails the test.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), torch.inference_mode():
        for _ in range(2):
            decode_fn()
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        static_out = decode_fn()
    return graph, static_out


def _row_metrics(static_out: torch.Tensor, captured_row: torch.Tensor, label: str) -> dict:
    m = compare(static_out.float().reshape(-1), captured_row.float().reshape(-1), label)
    assert m["all_finite"], (label, m)
    assert m["cosine"] > FP8_REPLAY_MODEL_COSINE, (label, m)
    assert m["max_abs"] <= FP8_REPLAY_MODEL_REL_MAX_ABS * max(m["ref_max_abs"], 1e-3), (label, m)
    return m


@pytest.mark.parametrize(
    "layer_idx,variant",
    [(0, "linear_dense"), (3, "sparse_moe"), (22, "linear_moe")],
    ids=["linear_dense-0", "sparse_moe-3", "linear_moe-22"],
)
def test_decoder_layer_decode_cuda_graph_replay_matches_hooked_hf(
    reader, text_config, schedule, hooked, device, evidence, layer_idx, variant
):
    """Whole-decoder-layer decode under graph capture, on real weights.

    Every operator the layer invokes at decode — both hyper-connection
    pre/post mixes with all 20 Sinkhorn rounds, both layer norms, the
    attention module (KDA or sparse MLA + k-pool indexer), and the dense or
    288-expert routed clamped-SwiGLU MoE — runs inside one captured graph,
    and two successive replays must track growing cache state bitwise
    against eager while staying inside the FP8 envelope against the hooked
    in-model output rows.
    """
    experts: List[int] = (
        list(range(int(text_config.n_routed_experts)))
        if schedule.mlp[layer_idx] == "sparse"
        else []
    )
    _hf, trt = _load_decoder_layer(reader, text_config, schedule, layer_idx, device, experts)
    is_linear = schedule.attention[layer_idx] == "linear_attention"
    assert isinstance(
        trt.self_attn, Glm5NextLinearAttention if is_linear else Glm5NextSparseAttention
    )
    if not is_linear:
        # The captured decode must run through the configured production
        # backend: the module has no other sparse-attention route.
        from tensorrt_llm._torch.attention_backend.sparse.glm_kpool import GlmKpoolSparseAttention
        from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention

        assert type(trt.self_attn.attn_backend) is GlmKpoolSparseAttention
        assert isinstance(trt.self_attn.attn_backend, TrtllmAttention)
    if schedule.mlp[layer_idx] == "sparse":
        assert isinstance(trt.mlp, Glm5NextMoE)

    prompt = _first_prompt_with(hooked, layer_idx, "")
    streams_in = prompt["activations"][f"layer{layer_idx}.input"].to(
        device=device, dtype=torch.bfloat16
    )[0]
    captured = prompt["activations"][f"layer{layer_idx}.output"].to(
        device=device, dtype=torch.float32
    )[0]
    total = int(streams_in.shape[0])
    assert total >= 4, f"fixture prompt too short for two decode steps: {total} tokens"
    prefix = total - 2

    attn = trt.self_attn
    if is_linear:
        conv = torch.zeros(
            1, attn.conv_dim, attn.conv_kernel_size - 1, device=device, dtype=torch.bfloat16
        )
        ssm = torch.zeros(
            1, attn.num_heads, attn.head_dim, attn.head_dim, device=device, dtype=torch.float32
        )
        slot = torch.zeros(1, device=device, dtype=torch.long)
        decode_kwargs: Dict[str, Any] = {"slot_ids": slot, "conv_pool": conv, "ssm_pool": ssm}
        pools = {"conv_pool": conv, "ssm_pool": ssm}

        def seed_prefix():
            conv.zero_()
            ssm.zero_()
            with torch.inference_mode():
                trt.forward_direct(
                    streams_in[:prefix],
                    phase="prefill",
                    cu_seqlens=[0, prefix],
                    cached_lens=[0],
                    **decode_kwargs,
                )

        def make_eager_kwargs():
            cloned = {"conv_pool": conv.clone(), "ssm_pool": ssm.clone()}
            kwargs = dict(decode_kwargs)
            kwargs.update(cloned)
            return kwargs, cloned

    else:
        pages = (total + TOKENS_PER_BLOCK - 1) // TOKENS_PER_BLOCK + 1
        owner = _KpoolPools(attn, pages, TOKENS_PER_BLOCK, device)
        latent, index = owner.latent, owner.index
        table = torch.arange(pages, device=device, dtype=torch.long).unsqueeze(0)
        kv_buf = torch.ones(1, device=device, dtype=torch.long)
        # The persistent metadata carrier the captured kernels read through --
        # glm_block_tables/glm_kv_lens are the same buffers refreshed between
        # replays, exactly the runtime prepare() contract in miniature.
        decode_md = _kpool_metadata(owner, block_tables=table, kv_lens=kv_buf, num_contexts=0)
        ctx_md = _kpool_metadata(
            owner,
            block_tables=table,
            kv_lens=torch.zeros(1, dtype=torch.long, device=device),
            num_contexts=1,
        )
        decode_kwargs = {"kv_lens": kv_buf, "metadata": decode_md}
        pools = {"latent_pool": latent, "index_pool": index}

        def seed_prefix():
            latent.zero_()
            index.zero_()
            with torch.inference_mode():
                trt.forward_direct(
                    streams_in[:prefix],
                    phase="prefill",
                    cu_seqlens=[0, prefix],
                    cached_lens=[0],
                    metadata=ctx_md,
                )

        def make_eager_kwargs():
            clone_owner = _KpoolPools(attn, pages, TOKENS_PER_BLOCK, device)
            clone_owner.latent.copy_(latent)
            clone_owner.index.copy_(index)
            md = _kpool_metadata(
                clone_owner, block_tables=table.clone(), kv_lens=kv_buf.clone(), num_contexts=0
            )
            return (
                {"kv_lens": kv_buf.clone(), "metadata": md},
                {"latent_pool": clone_owner.latent, "index_pool": clone_owner.index},
            )

    x_buf = torch.zeros_like(streams_in[:1])

    def decode():
        return trt.forward_direct(x_buf, phase="decode", **decode_kwargs)

    seed_prefix()
    graph, static_out = _graph_capture(decode)
    seed_prefix()  # warmup executed for real; rebuild the prefix state

    rows = []
    for step, pos in enumerate((prefix, prefix + 1)):
        x_buf.copy_(streams_in[pos : pos + 1])
        if not is_linear:
            kv_buf.fill_(pos + 1)

        eager_kwargs, cloned = make_eager_kwargs()
        with torch.inference_mode():
            expected = trt.forward_direct(x_buf.clone(), phase="decode", **eager_kwargs)

        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(static_out, expected), f"{variant} pos {pos}: graph != eager"
        for name, clone in cloned.items():
            assert torch.equal(pools[name], clone), f"{variant} pos {pos}: {name} cache diverged"

        metrics = _row_metrics(
            static_out[0], captured[pos], f"decoder_layer[{variant}] pos {pos} graph_vs_in_model"
        )
        rows.append({"position": pos, "decode_step": step, "graph_vs_in_model": metrics})

    _record(
        evidence,
        "decoder_layer_decode_cuda_graph_replay",
        {
            "layer_idx": layer_idx,
            "variant": variant,
            "attention_type": schedule.attention[layer_idx],
            "mlp_type": schedule.mlp[layer_idx],
            "attn_backend": None if is_linear else type(trt.self_attn.attn_backend).__name__,
            "prompt_index": prompt["index"],
            "total_tokens": total,
            "prefix_tokens": prefix,
            "num_experts_materialized": len(experts),
            "captured_under": "torch.cuda.CUDAGraph",
            "graph_equals_eager_bitwise": True,
            "phase": "decode",
            "dtype": "bfloat16 (dequantized rung)",
            "replays": rows,
        },
    )


@pytest.mark.parametrize(
    "layer_idx", [3, 23, 43], ids=["first_sparse", "middle_sparse", "last_sparse"]
)
def test_sparse_mla_decode_cuda_graph_replay_matches_hooked_hf(
    reader, text_config, hooked, device, evidence, layer_idx
):
    """Sparse-MLA + k-pool-indexer decode under graph capture, on real weights.

    The attention-module-scoped E leg: hooked hidden states from the real
    model drive two replays of one captured batched decode, checking bitwise
    graph-vs-eager equality (outputs and latent/index cache writes) and the
    FP8 envelope against the in-model attention output rows.
    """
    if text_config.layer_types[layer_idx] != "deepseek_sparse_attention":
        pytest.skip(f"layer {layer_idx} is not sparse in this schedule")
    _hf, trt = _load_mla(reader, text_config, layer_idx, device)
    # The captured decode must run through the configured production backend:
    # the module has no other sparse-attention route.
    from tensorrt_llm._torch.attention_backend.sparse.glm_kpool import GlmKpoolSparseAttention
    from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention

    assert type(trt.attn_backend) is GlmKpoolSparseAttention
    assert isinstance(trt.attn_backend, TrtllmAttention)

    prompt = _first_prompt_with(hooked, layer_idx, "self_attn")
    x = prompt["activations"][f"layer{layer_idx}.self_attn.input"].to(
        device=device, dtype=torch.bfloat16
    )[0]
    captured = prompt["activations"][f"layer{layer_idx}.self_attn.output"].to(
        device=device, dtype=torch.float32
    )[0]
    total = int(x.shape[0])
    assert total >= 4, f"fixture prompt too short for two decode steps: {total} tokens"
    prefix = total - 2

    pages = (total + TOKENS_PER_BLOCK - 1) // TOKENS_PER_BLOCK + 1
    owner = _KpoolPools(trt, pages, TOKENS_PER_BLOCK, device)
    latent, index = owner.latent, owner.index
    table = torch.arange(pages, device=device, dtype=torch.long).unsqueeze(0)
    x_buf = torch.zeros(1, x.shape[-1], device=device, dtype=torch.bfloat16)
    kv_buf = torch.ones(1, device=device, dtype=torch.long)
    # Persistent metadata carrier: the captured kernels read the cache state
    # exclusively through it, and the buffers it names (table/kv_buf) are the
    # ones refreshed between replays -- the runtime prepare() contract in
    # miniature.
    decode_md = _kpool_metadata(owner, block_tables=table, kv_lens=kv_buf, num_contexts=0)
    ctx_md = _kpool_metadata(
        owner,
        block_tables=table,
        kv_lens=torch.zeros(1, dtype=torch.long, device=device),
        num_contexts=1,
    )

    def seed_prefix():
        latent.zero_()
        index.zero_()
        with torch.inference_mode():
            trt.forward_prefill(x[:prefix], [0, prefix], [0], ctx_md)

    def decode():
        return trt.forward_decode(x_buf, kv_buf, decode_md)

    seed_prefix()
    graph, static_out = _graph_capture(decode)
    seed_prefix()

    rows = []
    for step, pos in enumerate((prefix, prefix + 1)):
        x_buf.copy_(x[pos : pos + 1])
        kv_buf.fill_(pos + 1)

        clone_owner = _KpoolPools(trt, pages, TOKENS_PER_BLOCK, device)
        clone_owner.latent.copy_(latent)
        clone_owner.index.copy_(index)
        eager_md = _kpool_metadata(
            clone_owner, block_tables=table.clone(), kv_lens=kv_buf.clone(), num_contexts=0
        )
        with torch.inference_mode():
            expected = trt.forward_decode(x_buf.clone(), kv_buf.clone(), eager_md)

        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(static_out, expected), f"L{layer_idx} pos {pos}: graph != eager"
        assert torch.equal(latent, clone_owner.latent), (
            f"L{layer_idx} pos {pos}: latent cache diverged"
        )
        assert torch.equal(index, clone_owner.index), (
            f"L{layer_idx} pos {pos}: index cache diverged"
        )

        metrics = _row_metrics(
            static_out[0], captured[pos], f"sparse_mla L{layer_idx} pos {pos} graph_vs_in_model"
        )
        rows.append({"position": pos, "decode_step": step, "graph_vs_in_model": metrics})

    _record(
        evidence,
        "sparse_mla_decode_cuda_graph_replay",
        {
            "layer_idx": layer_idx,
            "layer_type": "deepseek_sparse_attention",
            "attn_backend": type(trt.attn_backend).__name__,
            "metadata_class": type(trt.attn_backend).Metadata.__name__,
            "metadata_carrier": "persistent glm_block_tables/glm_kv_lens buffers (prepare() in miniature)",
            "prompt_index": prompt["index"],
            "total_tokens": total,
            "prefix_tokens": prefix,
            "captured_under": "torch.cuda.CUDAGraph",
            "graph_equals_eager_bitwise": True,
            "phase": "decode",
            "index_topk_regime": "below (fixture-length prompts)",
            "dtype": "bfloat16 (dequantized rung)",
            "replays": rows,
        },
    )


def test_sparse_mla_decode_graph_replay_crosses_index_topk(reader, text_config, device, evidence):
    """ONE captured decode graph replayed across the ``index_topk`` boundary.

    Every engine-scale graph run so far decoded far below ``index_topk=2048``
    (manifest prompts plus 512 steps), and the module-scope graph tests above
    use fixture-length prompts, so the regime where pool selection first
    becomes LOSSY — more candidate pools than the fixed selection budget —
    had no capture-scope coverage at all. This is exactly where the
    fixed-capacity ``-1`` index buffers and packed-tail rebuild have to keep
    working from a graph whose shapes were fixed at capture time.

    One graph is captured once, then replayed at ``kv_len`` 2047, 2048, 2049
    and 2050 — below, at, and (twice) above the boundary — with only the
    prepare()-style buffer refresh between replays. Each replay must be
    bitwise-equal to an eager call over cloned identical state, including
    every latent/index cache write. The eager side's own correctness in this
    regime is pinned separately by the eager suites (HF Jaccard/parity at
    2100 and the 512/2048/2600 dispatch tests), so bitwise graph==eager here
    closes the capture-scope gap without needing a hooked 2050-token HF
    fixture.
    """
    layer_idx = 3
    _hf, trt = _load_mla(reader, text_config, layer_idx, device)
    index_topk = int(text_config.index_topk)
    kpool = int(text_config.index_kpool)
    prefix = index_topk - 2  # 2046: the replay ladder crosses the boundary
    steps = 4
    total = prefix + steps  # 2050

    pages = total // TOKENS_PER_BLOCK + 2
    owner = _KpoolPools(trt, pages, TOKENS_PER_BLOCK, device)
    latent, index = owner.latent, owner.index
    table = torch.arange(pages, device=device, dtype=torch.long).unsqueeze(0)
    x = _hidden(total, text_config.hidden_size, device, seed=4046)
    x_buf = torch.zeros(1, x.shape[-1], device=device, dtype=torch.bfloat16)
    kv_buf = torch.full((1,), prefix + 1, device=device, dtype=torch.long)
    decode_md = _kpool_metadata(owner, block_tables=table, kv_lens=kv_buf, num_contexts=0)
    ctx_md = _kpool_metadata(
        owner,
        block_tables=table,
        kv_lens=torch.zeros(1, dtype=torch.long, device=device),
        num_contexts=1,
    )

    def seed_prefix():
        latent.zero_()
        index.zero_()
        with torch.inference_mode():
            trt.forward_prefill(x[:prefix], [0, prefix], [0], ctx_md)

    def decode():
        return trt.forward_decode(x_buf, kv_buf, decode_md)

    seed_prefix()
    graph, static_out = _graph_capture(decode)
    seed_prefix()  # warmup/capture executed for real; rebuild the prefix state

    rows = []
    for pos in range(prefix, total):
        kv_len = pos + 1
        x_buf.copy_(x[pos : pos + 1])
        kv_buf.fill_(kv_len)

        clone_owner = _KpoolPools(trt, pages, TOKENS_PER_BLOCK, device)
        clone_owner.latent.copy_(latent)
        clone_owner.index.copy_(index)
        eager_md = _kpool_metadata(
            clone_owner, block_tables=table.clone(), kv_lens=kv_buf.clone(), num_contexts=0
        )
        with torch.inference_mode():
            expected = trt.forward_decode(x_buf.clone(), kv_buf.clone(), eager_md)

        graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(static_out).all(), f"kv_len {kv_len}: non-finite graph output"
        assert torch.equal(static_out, expected), f"kv_len {kv_len}: graph != eager"
        assert torch.equal(latent, clone_owner.latent), f"kv_len {kv_len}: latent cache diverged"
        assert torch.equal(index, clone_owner.index), f"kv_len {kv_len}: index cache diverged"

        rows.append(
            {
                "kv_len": kv_len,
                "regime": (
                    "below" if kv_len < index_topk else "at" if kv_len == index_topk else "above"
                ),
                "candidate_pools": math.ceil(kv_len / kpool),
                "graph_equals_eager_bitwise": True,
            }
        )

    # The ladder must actually cross into the lossy regime, or the test
    # silently degrades into another below-topk check.
    assert rows[0]["regime"] == "below" and rows[-1]["regime"] == "above"
    assert rows[-1]["candidate_pools"] > index_topk // kpool

    _record(
        evidence,
        "sparse_mla_graph_replay_crosses_index_topk",
        {
            "layer_idx": layer_idx,
            "attn_backend": type(trt.attn_backend).__name__,
            "index_topk": index_topk,
            "index_kpool": kpool,
            "selection_budget_pools": index_topk // kpool,
            "output_width_fixed": int(trt.indexer.output_width),
            "captured_once_replayed_across_lengths": True,
            "captured_under": "torch.cuda.CUDAGraph",
            "prefix_tokens": prefix,
            "dtype": "bfloat16 (dequantized rung)",
            "replays": rows,
        },
    )


@pytest.mark.parametrize("variant", ["sparse", "linear"], ids=["sparse_mla", "kda"])
def test_batched_decode_graph_padding_slot_isolation(
    reader, text_config, device, evidence, variant
):
    """A padded batch's dummy slot cannot perturb real rows under replay.

    ``CUDAGraphRunner._get_padded_batch`` extends a decode-only batch with a
    persistent dummy request whose cache state is arbitrary-by-design (it is
    written by warmups and every padded replay, never reset). The engine
    therefore relies on strict per-row/per-slot isolation: whatever the
    dummy's state and input contain, the real rows' outputs and cache writes
    must be exactly what they would have been with any other dummy content.

    Module-scope proof for both hybrid state families: capture ONE batched
    decode (rows 0-2 real, row 3 the padding slot), replay it from a fixed
    pre-state with a benign dummy, then restore the pre-state, poison ONLY
    the dummy slot's cache state (NaN — the strongest possible contaminant)
    and the dummy's input row, and replay again. The real rows' outputs and
    cache writes must be bitwise-identical between the two replays, while
    the dummy row itself must visibly change (the positive control that the
    poison actually flowed through the captured kernels).
    """
    batch = 4
    real_rows = 3
    if variant == "sparse":
        layer_idx = 3
        _hf, trt = _load_mla(reader, text_config, layer_idx, device)
        prefix = 93  # deliberately not a multiple of the pool or page size
        pages_per_req = prefix // TOKENS_PER_BLOCK + 2
        owner = _KpoolPools(trt, batch * pages_per_req, TOKENS_PER_BLOCK, device)
        latent, index = owner.latent, owner.index
        table = torch.arange(batch * pages_per_req, device=device, dtype=torch.long).view(
            batch, pages_per_req
        )
        dummy_pages = table[real_rows]
        x_buf = torch.zeros(batch, text_config.hidden_size, device=device, dtype=torch.bfloat16)
        # Real rows decode at kv 94; the dummy decodes at kv 2 (the engine's
        # padding dummy is a token_num=1 request), so the batch is
        # heterogeneous in length exactly like a padded live batch.
        kv_buf = torch.tensor([prefix + 1] * real_rows + [2], device=device, dtype=torch.long)
        decode_md = _kpool_metadata(owner, block_tables=table, kv_lens=kv_buf, num_contexts=0)

        def seed_state():
            latent.zero_()
            index.zero_()
            with torch.inference_mode():
                for i in range(real_rows):
                    xi = _hidden(prefix, text_config.hidden_size, device, seed=51 + i)
                    trt.forward_prefill(
                        xi,
                        [0, prefix],
                        [0],
                        _kpool_metadata(
                            owner,
                            block_tables=table[i : i + 1],
                            kv_lens=torch.zeros(1, dtype=torch.long, device=device),
                            num_contexts=1,
                        ),
                    )
                trt.forward_prefill(
                    _hidden(1, text_config.hidden_size, device, seed=54),
                    [0, 1],
                    [0],
                    _kpool_metadata(
                        owner,
                        block_tables=table[real_rows:],
                        kv_lens=torch.zeros(1, dtype=torch.long, device=device),
                        num_contexts=1,
                    ),
                )

        def decode():
            return trt.forward_decode(x_buf, kv_buf, decode_md)

        state_tensors = {"latent": latent, "index": index}

        def poison_dummy_state():
            latent[dummy_pages] = float("nan")
            index[dummy_pages] = float("nan")

        def real_state_views():
            real_pages = table[:real_rows].reshape(-1)
            return {"latent": latent[real_pages], "index": index[real_pages]}

    else:
        layer_idx = 0
        _hf, attn = _load_kda(reader, text_config, layer_idx, device)
        trt = attn
        gen = torch.Generator(device="cpu").manual_seed(77)
        conv_init = (
            torch.randn(batch, attn.conv_dim, attn.conv_kernel_size - 1, generator=gen).to(
                device=device, dtype=torch.bfloat16
            )
            * 0.05
        )
        ssm_init = (
            torch.randn(batch, attn.num_heads, attn.head_dim, attn.head_dim, generator=gen).to(
                device=device, dtype=torch.float32
            )
            * 0.05
        )
        conv = conv_init.clone()
        ssm = ssm_init.clone()
        slot_ids = torch.arange(batch, device=device, dtype=torch.long)
        x_buf = torch.zeros(batch, text_config.hidden_size, device=device, dtype=torch.bfloat16)

        def seed_state():
            conv.copy_(conv_init)
            ssm.copy_(ssm_init)

        def decode():
            return trt.forward_decode(x_buf, slot_ids, conv, ssm)

        state_tensors = {"conv": conv, "ssm": ssm}

        def poison_dummy_state():
            conv[real_rows:] = float("nan")
            ssm[real_rows:] = float("nan")

        def real_state_views():
            return {"conv": conv[:real_rows], "ssm": ssm[:real_rows]}

    seed_state()
    graph, static_out = _graph_capture(decode)
    seed_state()  # warmup/capture executed for real; rebuild the pre-state
    pre_state = {name: t.clone() for name, t in state_tensors.items()}

    x_next = _hidden(batch, text_config.hidden_size, device, seed=61)

    # Replay 1 — benign dummy content.
    x_buf.copy_(x_next)
    graph.replay()
    torch.cuda.synchronize()
    benign_out = static_out.clone()
    benign_real_state = {name: view.clone() for name, view in real_state_views().items()}
    assert torch.isfinite(benign_out).all()

    # Restore the exact pre-replay state, then poison ONLY the dummy slot.
    for name, t in state_tensors.items():
        t.copy_(pre_state[name])
    poison_dummy_state()
    x_buf.copy_(x_next)
    x_buf[real_rows:] = 1e4  # a different dummy input row as well

    # Replay 2 — poisoned dummy content, identical real inputs/state.
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(static_out[:real_rows], benign_out[:real_rows]), (
        f"{variant}: poisoning the padding slot changed a real row's output"
    )
    for name, view in real_state_views().items():
        assert torch.equal(view, benign_real_state[name]), (
            f"{variant}: poisoning the padding slot changed real {name} cache writes"
        )
    # Positive control: the poison must have flowed through the captured
    # kernels (NaN != NaN, so torch.equal is False for a NaN-carrying row).
    assert not torch.equal(static_out[real_rows:], benign_out[real_rows:]), (
        f"{variant}: dummy row unchanged — the poison never reached the captured kernels"
    )

    _record(
        evidence,
        "batched_decode_graph_padding_slot_isolation",
        {
            "variant": variant,
            "layer_idx": layer_idx,
            "batch": batch,
            "real_rows": real_rows,
            "padding_slot_poison": "NaN state + 1e4 input row",
            "captured_under": "torch.cuda.CUDAGraph",
            "real_rows_bitwise_invariant": True,
            "real_cache_writes_bitwise_invariant": True,
            "positive_control_dummy_row_changed": True,
            "dtype": "bfloat16 (dequantized rung)",
        },
    )


def test_padding_dummy_allocation_on_real_manager(text_config, device, evidence):
    """``Glm5NextCacheManager`` honors the runner's padding-dummy allocation.

    ``CUDAGraphRunner._get_or_create_padding_dummy`` pads a live decode batch
    by calling ``kv_cache_manager.add_dummy_requests([id], is_gen=True, ...)``
    and silently falls back to eager (with only a warning) when that returns
    ``None``. The hybrid V2 manager reserves dedicated dummy slots for this
    (``_num_reserved_dummy_slots``), so the allocation must succeed even when
    every regular slot is held by a live request — otherwise a padding-enabled
    serving config would quietly degrade to eager on every padded batch.

    This pins the whole engine-side premise on the real manager: allocation
    succeeds under full occupancy with the runner's exact call shape, the
    prepared runtime metadata gives the dummy a valid KDA state slot distinct
    from every real request's slot and valid sparse page rows, and the
    runner's release path (``free_resources``) works.
    """
    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID

    manager, _attention, _sparse_ids, _linear_ids, _heads, _dim = _build_cache_manager(
        text_config, tokens_per_block=64, max_seq_len=2688
    )
    try:
        # Hold every regular slot (the helper builds max_batch_size=4).
        manager.add_dummy_requests([0, 1, 2, 3], token_nums=[257, 129, 65, 33])

        # The runner's exact allocation shape (_get_or_create_padding_dummy
        # with no speculative decoding, no encoder-decoder, no mrope).
        dummy_list = manager.add_dummy_requests(
            [CUDA_GRAPH_DUMMY_REQUEST_ID],
            is_gen=True,
            max_num_draft_tokens=0,
            use_mrope=False,
            max_beam_width=1,
        )
        assert dummy_list, (
            "add_dummy_requests(is_gen=True) returned no request while every "
            "regular slot is occupied — the reserved padding-dummy slot is "
            "not working, so padded batches would silently run eager"
        )
        dummy = dummy_list[0]
        dummy.is_cuda_graph_dummy = True

        # A padded batch never exceeds max_batch_size (the runner refuses to
        # pad past the budget), so the padded-batch-shaped metadata schedules
        # 3 of the real gen rows + the dummy; request 3 stays live but
        # unscheduled, exactly like a request waiting out this step.
        metadata = _real_prepared_metadata(
            manager,
            lens=[1, 1, 1, 1],
            num_contexts=0,
            cached=[256, 128, 64, 0],
            request_ids=[0, 1, 2, CUDA_GRAPH_DUMMY_REQUEST_ID],
        )
        mamba_md = metadata.mamba_metadata
        slots = mamba_md.state_indices[:4].tolist()
        assert len(set(slots)) == 4 and all(s >= 0 for s in slots), (
            f"padded batch state slots must be distinct and valid, got {slots}"
        )
        tables = mamba_md.glm_block_tables
        assert tables is not None and int(tables.shape[0]) >= 4
        first_pages = tables[:4, 0].tolist()
        assert all(p >= 0 for p in first_pages), (
            f"padded batch sparse page rows must be valid, got {first_pages}"
        )
        kv_lens = mamba_md.glm_kv_lens[:4].tolist()
        assert kv_lens == [257, 129, 65, 1], f"unexpected prepared kv lens: {kv_lens}"

        # The runner's release path (CUDAGraphRunner.clear on shutdown).
        manager.free_resources(dummy)

        _record(
            evidence,
            "padding_dummy_allocation_on_real_manager",
            {
                "cache_manager_class": type(manager).__name__,
                "dummy_request_id": CUDA_GRAPH_DUMMY_REQUEST_ID,
                "allocation_call": (
                    "add_dummy_requests(is_gen=True, max_num_draft_tokens=0, "
                    "use_mrope=False, max_beam_width=1) under full slot occupancy"
                ),
                "reserved_dummy_slots": int(getattr(manager, "_num_reserved_dummy_slots", -1)),
                "padded_batch_state_slots": slots,
                "padded_batch_first_pages": first_pages,
                "padded_batch_kv_lens": kv_lens,
                "freed_cleanly": True,
            },
        )
    finally:
        manager.shutdown()
