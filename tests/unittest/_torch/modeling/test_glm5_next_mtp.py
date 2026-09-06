# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GLM-5.3-Flash one-model MTP speculative decoding: module-level contracts.

Three layers of evidence, cheapest first:

* **Checkpoint accounting** (no GPU): with MTP enabled the appended
  ``layers.45.*`` keys are real destinations rather than an allowlisted
  ignore, and the ``eh_proj`` fusion has a declared four-rank owner.
* **Construction** (meta device, real config): ``AutoModelForCausalLM`` with
  an MTP ``spec_config`` builds the ``Glm5NextMTP`` draft layer, aliases it
  onto ``model.layers[45]``, dispatches the MTP-Eagle worker, and the quant
  plan lands BF16/FP8 on the MTP layer exactly as the checkpoint's exclusion
  list says.
* **Verification semantics** (GPU, random weights): the multi-token
  ``forward_verify`` paths of both attention families equal ``T`` sequential
  single-token decodes -- outputs *and*, for KDA, the per-step recurrent
  states written to the hybrid manager's intermediate buffers -- while the
  live KDA pools are left untouched for the sampler's acceptance to decide.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.modeling_glm5_next import (
    Disposition,
    Glm5NextLinearAttention,
    Glm5NextSparseAttention,
    audit_glm5_next_checkpoint,
    resolve_glm5_next_projection_spec,
)

CHECKPOINT = os.environ.get("GLM53_FLASH_CHECKPOINT", "/dev/shm/GLM-5.3-Flash")
MTP_LAYER = 45

requires_checkpoint = pytest.mark.skipif(
    not os.path.isdir(CHECKPOINT), reason=f"requires the checkpoint at {CHECKPOINT}"
)
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(scope="module")
def hf_config():
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(CHECKPOINT)


@pytest.fixture(scope="module")
def checkpoint_keys():
    with open(os.path.join(CHECKPOINT, "model.safetensors.index.json")) as fh:
        return sorted(json.load(fh)["weight_map"])


# ---------------------------------------------------------------------------
# Checkpoint accounting
# ---------------------------------------------------------------------------


@requires_checkpoint
def test_audit_places_the_mtp_layer_only_when_enabled(hf_config, checkpoint_keys):
    mtp_keys = [k for k in checkpoint_keys if f".layers.{MTP_LAYER}." in k]
    assert len(mtp_keys) == 1760, "checkpoint inventory changed"

    off = audit_glm5_next_checkpoint(checkpoint_keys, hf_config)
    assert {off.disposition[k] for k in mtp_keys} == {Disposition.IGNORED}

    on = audit_glm5_next_checkpoint(checkpoint_keys, hf_config, num_mtp_layers=1)
    assert not on.unresolved
    assert {on.disposition[k] for k in mtp_keys} <= {Disposition.LOADED, Disposition.TRANSFORMED}
    # Every other disposition is unchanged: the MTP switch only moves layer 45.
    for key in checkpoint_keys:
        if key not in mtp_keys:
            assert on.disposition[key] == off.disposition[key], key
    # Destinations live under the aliased decoder index, in the runtime spelling.
    dests = {d for d, src in on.destinations.items() if f".layers.{MTP_LAYER}." in src}
    assert all(d.startswith(f"model.layers.{MTP_LAYER}.") for d in dests)
    for leaf in ("enorm.weight", "hnorm.weight", "eh_proj.weight", "shared_head.norm.weight"):
        assert f"model.layers.{MTP_LAYER}.{leaf}" in dests, leaf


@requires_checkpoint
def test_audit_rejects_more_mtp_layers_than_the_checkpoint_has(hf_config, checkpoint_keys):
    with pytest.raises(ValueError, match="num_nextn_predict_layers"):
        audit_glm5_next_checkpoint(checkpoint_keys, hf_config, num_mtp_layers=2)


def test_eh_proj_has_row_parallel_ownership():
    from tensorrt_llm.mapping import Mapping

    spec = resolve_glm5_next_projection_spec(
        f"model.layers.{MTP_LAYER}.eh_proj", Mapping(world_size=4, tp_size=4, rank=0)
    )
    assert spec.mode == "row" and spec.reduce_output is True


# ---------------------------------------------------------------------------
# Construction through the runtime's own entry point
# ---------------------------------------------------------------------------


@requires_checkpoint
@requires_cuda
def test_mtp_model_constructs_with_aliased_draft_layer(checkpoint_keys):
    """from_config + MTP spec_config: draft layer, alias, worker, quant plan."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM
    from tensorrt_llm._torch.models.modeling_glm5_next import Glm5NextMTP, Glm5NextMTPHead
    from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode
    from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

    spec = MTPDecodingConfig(num_nextn_predict_layers=1)
    model_config = ModelConfig.from_pretrained(CHECKPOINT, spec_config=spec)
    assert spec.spec_dec_mode == SpeculativeDecodingMode.MTP_EAGLE_ONE_MODEL
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(model_config)

    assert type(model).__name__ == "Glm5NextForCausalLM"
    assert len(model.model.layers) == MTP_LAYER + 1
    assert len(model.mtp_layers) == 1
    draft = model.model.layers[MTP_LAYER]
    assert isinstance(draft, Glm5NextMTP)
    assert draft is model.draft_model.mtp_layers[0], "layer 45 must alias the drafter's layer"
    assert isinstance(draft.shared_head, Glm5NextMTPHead)
    assert type(model.spec_worker).__name__ == "MTPEagleWorker"
    # The decoder forward runs the 45 main layers only; the draft layer runs
    # inside the speculative worker.
    assert model.model.schedule.num_layers == MTP_LAYER

    # Exact-placement contract: every MTP-layer parameter has a checkpoint
    # source under the aliased name, and every layer-45 checkpoint tensor has
    # a destination (routed experts reach the fused layer by rule).
    audit = model.audit_checkpoint(checkpoint_keys)
    dests = set(audit.destinations)
    for name, _ in draft.named_parameters():
        if name.startswith("mlp.experts."):
            continue
        assert f"model.layers.{MTP_LAYER}.{name}" in dests, name

    placement = model.apply_quant_plan(checkpoint_keys)
    mtp_placement = {
        name.split(f"layers.{MTP_LAYER}.")[1]: dtype
        for name, dtype in placement.items()
        if f"layers.{MTP_LAYER}." in name
    }
    # The checkpoint's exclusion list keeps these BF16 on layer 45.
    for bf16 in (
        "eh_proj",
        "self_attn.kv_b_proj",
        "self_attn.indexer.wq_b",
        "self_attn.indexer.wk",
    ):
        assert mtp_placement[bf16] == "bfloat16", (bf16, mtp_placement[bf16])
    for fp8 in ("self_attn.q_a_proj", "self_attn.q_b_proj", "self_attn.o_proj"):
        assert mtp_placement[fp8] == "float8_e4m3fn", (fp8, mtp_placement[fp8])
    eh_proj = draft.eh_proj
    assert getattr(eh_proj.tp_mode, "value", None) == "row"
    assert tuple(eh_proj.glm5_full_shape) == (4096, 8192)


# ---------------------------------------------------------------------------
# KDA: multi-token verification == T sequential decodes, state uncommitted
# ---------------------------------------------------------------------------


def _small_text_config(hf_config):
    import copy

    text = copy.deepcopy(hf_config.text_config)
    # Fewer heads keeps the random-weight module small; head_dim, the conv
    # width, and the gate bound stay at the checkpoint's values because the
    # kernels are shape-gated on them.
    linear = dict(text.linear_attn_config)
    linear["num_heads"] = 4
    text.linear_attn_config = linear
    text.hidden_size = 512
    text.q_lora_rank = 256
    text.index_n_heads = 4
    return text


def _randomize_(module: torch.nn.Module, seed: int) -> None:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name, param in module.named_parameters():
            if name.endswith("A_log"):
                param.copy_(torch.rand(param.shape, generator=gen).to(param) * 0.5)
            elif name.endswith("dt_bias"):
                param.copy_((torch.rand(param.shape, generator=gen) - 0.5).to(param))
            else:
                param.copy_((torch.randn(param.shape, generator=gen) * 0.05).to(param))


@requires_checkpoint
@requires_cuda
@pytest.mark.parametrize("tokens_per_request", [2, 3], ids=["draft1", "draft2"])
def test_kda_verify_matches_sequential_decode_and_leaves_pools_uncommitted(
    hf_config, tokens_per_request
):
    device = torch.device("cuda")
    text = _small_text_config(hf_config)
    attn = Glm5NextLinearAttention(text, layer_idx=0).to(device).eval()
    _randomize_(attn, seed=7)
    torch.manual_seed(11)

    batch, slots = 3, 5
    T = tokens_per_request
    slot_ids = torch.tensor([4, 1, 2], device=device)
    conv_pool = torch.randn(
        slots, attn.conv_dim, attn.conv_kernel_size - 1, device=device, dtype=torch.bfloat16
    )
    ssm_pool = torch.randn(
        slots, attn.num_heads, attn.head_dim, attn.head_dim, device=device, dtype=torch.float32
    )
    x = torch.randn(batch * T, text.hidden_size, device=device, dtype=torch.bfloat16) * 0.5

    # Sequential reference: T single-token decodes, committing pools in place.
    ref_conv, ref_ssm = conv_pool.clone(), ssm_pool.clone()
    ref_out, ref_ssm_steps, ref_conv_steps = [], [], []
    with torch.no_grad():
        for j in range(T):
            ref_out.append(
                attn.forward_decode(x.view(batch, T, -1)[:, j], slot_ids, ref_conv, ref_ssm)
            )
            ref_ssm_steps.append(ref_ssm[slot_ids].clone())
            ref_conv_steps.append(ref_conv[slot_ids].clone())

    # Verify path: one call, per-step states into the intermediate scratch.
    spec_rows = 6
    intermediate_conv = torch.zeros(
        spec_rows, T, attn.conv_dim, attn.conv_kernel_size - 1, device=device, dtype=torch.bfloat16
    )
    intermediate_ssm = torch.zeros(
        spec_rows,
        T,
        attn.num_heads,
        attn.head_dim,
        attn.head_dim,
        device=device,
        dtype=torch.float32,
    )
    rows = torch.tensor([3, 0, 5], device=device, dtype=torch.int32)
    live_conv, live_ssm = conv_pool.clone(), ssm_pool.clone()
    with torch.no_grad():
        out = attn.forward_verify(
            x, slot_ids, live_conv, live_ssm, T, intermediate_conv, intermediate_ssm, rows
        )

    ref = torch.stack(ref_out, dim=1).reshape(batch * T, -1)
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
    # The reference decode convolves with the CUDA single-token op and the
    # verify path with the Triton multi-token op; both round their bf16
    # output independently (each within ~8e-3 of an fp32 reference), so the
    # fp32 recurrent states carry that bf16 input noise, not a math gap.
    for j in range(T):
        torch.testing.assert_close(
            intermediate_ssm[rows.long(), j], ref_ssm_steps[j], atol=4e-3, rtol=2e-2
        )
        torch.testing.assert_close(
            intermediate_conv[rows.long(), j].float(), ref_conv_steps[j].float(), atol=0, rtol=0
        )
    # The recurrent pool is read-only on the verify path: acceptance decides
    # which intermediate step is promoted, so nothing may be committed here.
    assert torch.equal(live_ssm, ssm_pool)
    # Rows of untouched slots and intermediate rows are never written.
    untouched = [s for s in range(slots) if s not in slot_ids.tolist()]
    assert torch.equal(live_conv[untouched], conv_pool[untouched])
    unused_rows = [r for r in range(spec_rows) if r not in rows.tolist()]
    assert not intermediate_ssm[unused_rows].any()


# ---------------------------------------------------------------------------
# Sparse MLA: multi-token verification == T sequential decodes
# ---------------------------------------------------------------------------


class _Pools:
    """Miniature of the ``Glm5NextCacheManager`` accessors the backend reads."""

    def __init__(self, trt, num_pages, tokens_per_block, device):
        self.tokens_per_block = tokens_per_block
        self._latent = torch.zeros(
            num_pages, tokens_per_block, 1, trt.kv_lora_rank, device=device, dtype=torch.bfloat16
        )
        self._index = torch.zeros(
            num_pages,
            tokens_per_block,
            1,
            trt.indexer.packed_state_dim,
            device=device,
            dtype=torch.bfloat16,
        )

    def get_latent_state_buffer(self, layer_idx):
        return self._latent

    def get_index_state_buffer(self, layer_idx):
        return self._index


def _metadata(pools, block_tables, kv_lens, num_contexts):
    return SimpleNamespace(
        kv_cache_manager=pools,
        mamba_metadata=SimpleNamespace(glm_block_tables=block_tables, glm_kv_lens=kv_lens),
        seq_lens=torch.ones(block_tables.shape[0], dtype=torch.long),
        num_contexts=num_contexts,
        is_cuda_graph=False,
    )


@requires_checkpoint
@requires_cuda
@pytest.mark.parametrize("tokens_per_request", [2, 3], ids=["draft1", "draft2"])
def test_sparse_verify_matches_sequential_decode(hf_config, tokens_per_request):
    device = torch.device("cuda")
    text = _small_text_config(hf_config)
    trt = Glm5NextSparseAttention(text, layer_idx=3).to(device).eval()
    _randomize_(trt, seed=5)
    torch.manual_seed(3)

    T = tokens_per_request
    batch = 2
    tokens_per_block = 16
    prefix = [37, 21]  # different lengths so per-request positions matter
    pages_per_req = (max(prefix) + T) // tokens_per_block + 1
    pools = _Pools(trt, 2 * pages_per_req, tokens_per_block, device)
    # Interleaved page ownership: a request-crossing row bug shows up.
    table = torch.tensor(
        [[2 * i for i in range(pages_per_req)], [2 * i + 1 for i in range(pages_per_req)]],
        device=device,
        dtype=torch.long,
    )
    hidden = text.hidden_size
    ctx = [torch.randn(n, hidden, device=device, dtype=torch.bfloat16) * 0.5 for n in prefix]
    new = torch.randn(batch, T, hidden, device=device, dtype=torch.bfloat16) * 0.5

    def seed_prefix(md_pools):
        md = _metadata(md_pools, table, torch.zeros(batch, device=device, dtype=torch.long), batch)
        with torch.no_grad():
            trt.forward_prefill(torch.cat(ctx), [0, prefix[0], sum(prefix)], [0, 0], md)

    # Sequential reference: one token per request per step.
    seed_prefix(pools)
    ref = []
    with torch.no_grad():
        for j in range(T):
            kv_lens = torch.tensor([p + j + 1 for p in prefix], device=device, dtype=torch.long)
            ref.append(trt.forward_decode(new[:, j], kv_lens, _metadata(pools, table, kv_lens, 0)))
    ref = torch.stack(ref, dim=1).reshape(batch * T, -1)
    ref_latent, ref_index = pools._latent.clone(), pools._index.clone()

    # Verify path on fresh pools with the same prefix: all T tokens at once.
    pools2 = _Pools(trt, 2 * pages_per_req, tokens_per_block, device)
    seed_prefix(pools2)
    kv_lens = torch.tensor([p + T for p in prefix], device=device, dtype=torch.long)
    with torch.no_grad():
        out = trt.forward_verify(
            new.reshape(batch * T, hidden), kv_lens, _metadata(pools2, table, kv_lens, 0), T
        )

    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
    # Both paths leave identical latent/indexer state at identical positions.
    assert torch.equal(pools2._latent, ref_latent)
    assert torch.equal(pools2._index, ref_index)
