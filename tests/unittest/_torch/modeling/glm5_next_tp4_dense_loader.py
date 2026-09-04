# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Goal 5.1 four-rank dense-path driver (Stage 5, TP4 / TP4-EP4).

Run under ``mpirun -n 4`` on four CUDA GPUs::

    mpirun -n 4 python glm5_next_tp4_dense_loader.py --layout tp4    --json out.json
    mpirun -n 4 python glm5_next_tp4_dense_loader.py --layout tp4ep4 --json out.json

What one invocation proves, per the Stage-5 dense-loader acceptance item:

1. **Construction + loading** — the exact /dev/shm checkpoint meta-constructs
   with the four-rank Mapping and loads shard-aware through each destination
   Linear's own ``load_shard``/quant-method contract (weights stay lazy
   safetensors slices; only this rank's contiguous rows/columns and 128x128
   scale rows are materialized). Routed experts load through the fused layer's
   ``initial_local_expert_ids``; EP-remote experts are counted, never read.
2. **Per-rank accounting** — every checkpoint key resolves exactly once as
   replicated / locally sharded / transformed / EP-remote / intentionally
   ignored; the four ranks' shard ranges union to each full tensor with no
   overlap and no missing destination.
3. **Replicated-bitwise state** — hyper-connection packed weights, router
   weight + e_score_correction_bias, norms, embeddings, and the declared
   replicated projections hash identically on all four ranks.
4. **Reconstructed activations/logits** — every human-named projection family
   is executed on CUDA at TP4; column shards are gathered, row outputs are
   reduced by the Linear's own single all-reduce, and the reconstruction must
   satisfy the predeclared dtype-aware envelope against the full-weight
   reference math (identical to single-rank PP4 execution: same weights, same
   kernels' full-tensor form). The MoE composition (routed partial + shared,
   exactly one reduction) runs end to end against the Stage-1/2-verified
   block-FP8 reference expert math.
5. **B then E** — the eager (B) replay runs first; the E leg re-executes every
   family inside a captured ``torch.cuda.CUDAGraph`` (collectives included)
   and replays it with fresh inputs. A successful capture is the no-fallback
   proof (capture raises on any host sync), and the replayed outputs must
   match the eager outputs within the predeclared envelope.

Deliberately *not* exercised here (later Goals own them): the KDA/MLA
attention forward at TP4 (head-sharded state/kernels — Goal 5.2) and deep
routed-expert replay vs hooked HF (Goal 5.3).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CHECKPOINT = os.environ.get("GLM53_FLASH_CHECKPOINT", "/dev/shm/GLM-5.3-Flash")
SEED = 20260904
TOKENS = 32

#: Predeclared dtype-aware envelopes (see report §envelopes). The reference is
#: full-weight math on the same checkpoint bytes — the single-rank (PP4) form
#: of each projection — so the only admissible error is kernel tiling /
#: partial-sum rounding at bf16/fp8 working precision.
LINEAR_ENVELOPE = {"cosine": 0.9995, "rel_max_abs": 2e-2}
MLP_ENVELOPE = {"cosine": 0.999, "rel_max_abs": 5e-2}
#: Fused TRTLLMGen vs block-matmul reference expert math (goal3.3 precedent).
MOE_ENVELOPE = {"cosine": 0.998, "rel_max_abs": 8e-2}
#: Graph replay vs eager on identical inputs. Non-collective families must be
#: bitwise; families with a collective inside the capture may legitimately
#: pick a different reduction algorithm under graph mode.
GRAPH_ENVELOPE = {"cosine": 0.9999, "rel_max_abs": 5e-3}

KDA_LAYERS = (0, 44)
MLA_LAYERS = (3, 43)
DENSE_MLP_LAYER = 0
MOE_LAYER = 4


def log(rank: int, msg: str) -> None:
    print(f"[glm5-tp4-dense rank{rank}] {msg}", flush=True)


def tensor_digest(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def metrics(actual: torch.Tensor, reference: torch.Tensor) -> Dict[str, float]:
    a = actual.detach().float().flatten()
    r = reference.detach().float().flatten()
    diff = (a - r).abs()
    scale = max(1.0, float(r.abs().max()))
    cosine = torch.nn.functional.cosine_similarity(a, r, dim=0).item()
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "cosine": float(cosine),
        "ref_scale": scale,
        "rel_max_abs": float(diff.max()) / scale,
        "bitwise": bool(torch.equal(actual, reference)),
        "finite": bool(torch.isfinite(actual).all()),
    }


def check_envelope(m: Dict[str, float], env: Dict[str, float]) -> Optional[str]:
    if not m["finite"]:
        return "non-finite output"
    if m["cosine"] < env["cosine"]:
        return f"cosine {m['cosine']:.6f} < {env['cosine']}"
    if m["rel_max_abs"] > env["rel_max_abs"]:
        return f"rel_max_abs {m['rel_max_abs']:.6f} > {env['rel_max_abs']}"
    return None


class Driver:
    def __init__(self, layout: str):
        from mpi4py import MPI

        import tensorrt_llm

        # Package provenance is a hard gate: a run against an installed wheel
        # (python puts the *script's* directory, not the cwd, on sys.path, so
        # `mpirun python tests/...` silently resolves site-packages) would
        # validate stale code. Launch with PYTHONPATH=<repo root>.
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 4)
        )
        pkg = os.path.abspath(tensorrt_llm.__file__)
        if not pkg.startswith(repo_root + os.sep):
            raise RuntimeError(
                f"stale package: tensorrt_llm resolved to {pkg}, expected under "
                f"{repo_root}; launch with PYTHONPATH={repo_root}"
            )
        self.package_file = pkg

        self.comm = MPI.COMM_WORLD
        self.rank = tensorrt_llm.mpi_rank()
        self.world = self.comm.Get_size()
        assert self.world == 4, f"run under mpirun -n 4 (got {self.world})"
        torch.cuda.set_device(self.rank)
        self.device = torch.device("cuda", self.rank)
        self.layout = layout
        self.problems: List[str] = []
        self.result: Dict[str, Any] = {
            "layout": layout,
            "rank": self.rank,
            "device_name": torch.cuda.get_device_name(self.rank),
            "package_file": self.package_file,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "problems": self.problems,
        }

    # ---- construction + loading -----------------------------------------

    def build(self) -> None:
        from glm5_next_full_model import LazyCheckpoint
        from transformers import AutoConfig

        from tensorrt_llm._torch.model_config import ModelConfig
        from tensorrt_llm._torch.models.modeling_glm5_next import (
            Glm5NextForCausalLM,
            build_glm5_next_quant_config,
        )
        from tensorrt_llm.mapping import Mapping

        moe_tp, moe_ep = (4, 1) if self.layout == "tp4" else (1, 4)
        self.mapping = Mapping(
            world_size=4,
            tp_size=4,
            pp_size=1,
            rank=self.rank,
            gpus_per_node=4,
            moe_tp_size=moe_tp,
            moe_ep_size=moe_ep,
        )
        self.result["mapping"] = {
            "world_size": 4,
            "tp_size": 4,
            "pp_size": 1,
            "rank": self.rank,
            "moe_tp_size": moe_tp,
            "moe_ep_size": moe_ep,
        }
        config = AutoConfig.from_pretrained(CHECKPOINT)
        quant_config = build_glm5_next_quant_config(config)
        model_config = ModelConfig(
            pretrained_config=config,
            quant_config=quant_config,
            mapping=self.mapping,
            moe_backend=ModelConfig.resolve_moe_backend(
                "AUTO", "Glm5NextForConditionalGeneration", quant_config
            ),
        )
        self.weights = LazyCheckpoint(CHECKPOINT)
        self.keys = list(self.weights.keys())

        t0 = time.time()
        with torch.device("meta"):
            self.model = Glm5NextForCausalLM(model_config)
        self.placement = self.model.apply_quant_plan(self.keys)
        construct_s = time.time() - t0
        t0 = time.time()
        self.report = self.model.load_weights(self.weights)
        self.load_s = time.time() - t0
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        summary = self.report.summary()
        self.result["construct_seconds"] = round(construct_s, 1)
        self.result["load_seconds"] = round(self.load_s, 1)
        self.result["load_summary"] = {
            k: summary[k]
            for k in (
                "loaded",
                "transformed",
                "ignored",
                "skipped_remote",
                "remote_experts",
                "total",
            )
        }
        self.result["tp_shards_count"] = len(summary["tp_shards"])
        self.tp_shards = summary["tp_shards"]
        # The acceptance item wants every rank's local shapes/ranges/dtypes on
        # the record; ~580 small rows per rank stays well bounded.
        self.result["tp_shards"] = self.tp_shards
        log(
            self.rank,
            f"constructed {construct_s:.0f}s loaded {self.load_s:.0f}s "
            f"summary={self.result['load_summary']}",
        )

    # ---- accounting proofs ------------------------------------------------

    def accounting(self) -> None:
        summary = self.result["load_summary"]
        total = summary["total"]
        if total != len(self.keys):
            self.problems.append(
                f"key accounting: loaded+transformed+ignored+remote={total} != {len(self.keys)}"
            )
        expected_remote = 42 * 216 * 6 if self.layout == "tp4ep4" else 0
        if summary["remote_experts"] != expected_remote:
            self.problems.append(
                f"remote_experts {summary['remote_experts']} != expected {expected_remote}"
            )
        if summary["loaded"] < 0 or summary["transformed"] < 0:
            self.problems.append(
                f"negative accounting bucket: loaded={summary['loaded']} "
                f"transformed={summary['transformed']}"
            )
        if self.report.missing_destinations:
            self.problems.append(f"missing destinations: {self.report.missing_destinations[:5]}")

        # Local structural spot-asserts: the human-named projections own the
        # expected local geometry at TP4.
        checks = []
        l0 = self.model.model.layers[KDA_LAYERS[0]].self_attn
        l3 = self.model.model.layers[MLA_LAYERS[0]].self_attn
        mlp0 = self.model.model.layers[DENSE_MLP_LAYER].mlp
        moe = self.model.model.layers[MOE_LAYER].mlp
        checks.append(("kda.q_proj", tuple(l0.q_proj.weight.shape), (2048, 4096)))
        checks.append(("kda.b_proj", tuple(l0.b_proj.weight.shape), (16, 4096)))
        checks.append(("kda.f_a_proj", tuple(l0.f_a_proj.weight.shape), (128, 4096)))
        checks.append(("kda.o_proj", tuple(l0.o_proj.weight.shape), (4096, 2048)))
        # Goal 5.2 head-sharded KDA state parameters and the local conv filter.
        checks.append(("kda.A_log", tuple(l0.A_log.shape), (16,)))
        checks.append(("kda.dt_bias", tuple(l0.dt_bias.shape), (2048,)))
        checks.append(("kda.conv1d", tuple(l0.conv1d.weight.shape), (6144, 1, 4)))
        checks.append(("mla.q_b_proj", tuple(l3.q_b_proj.weight.shape), (4096, 1536)))
        checks.append(("mla.q_b_scale", tuple(l3.q_b_proj.weight_scale.shape), (32, 12)))
        checks.append(("mla.kv_b_proj", tuple(l3.kv_b_proj.weight.shape), (8192, 512)))
        checks.append(("mla.kv_a", tuple(l3.kv_a_proj_with_mqa.weight.shape), (512, 4096)))
        checks.append(("mla.o_proj", tuple(l3.o_proj.weight.shape), (4096, 4096)))
        checks.append(("idx.wq_b", tuple(l3.indexer.wq_b.weight.shape), (1024, 1536)))
        checks.append(("idx.wk", tuple(l3.indexer.wk.weight.shape), (128, 4096)))
        checks.append(("idx.weights_proj", tuple(l3.indexer.weights_proj.weight.shape), (8, 4096)))
        checks.append(("dense.gate", tuple(mlp0.gate_proj.weight.shape), (3072, 4096)))
        checks.append(("dense.down", tuple(mlp0.down_proj.weight.shape), (4096, 3072)))
        if self.layout == "tp4":
            checks.append(
                ("shared.gate", tuple(moe.shared_experts.gate_proj.weight.shape), (512, 4096))
            )
            checks.append(
                ("shared.down", tuple(moe.shared_experts.down_proj.weight.shape), (4096, 512))
            )
            if moe.shared_experts.down_proj.reduce_output:
                self.problems.append(
                    "TP4 shared down_proj must keep a partial (reduce_output=False)"
                )
            checks.append(
                ("experts.w3_w1", tuple(moe.experts.w3_w1_weight.shape), (288, 1024, 4096))
            )
        else:
            checks.append(
                ("shared.gate", tuple(moe.shared_experts.gate_proj.weight.shape), (2048, 4096))
            )
            checks.append(
                ("shared.down", tuple(moe.shared_experts.down_proj.weight.shape), (4096, 2048))
            )
            checks.append(
                ("experts.w3_w1", tuple(moe.experts.w3_w1_weight.shape), (72, 4096, 4096))
            )
            local_ids = list(moe.experts.initial_local_expert_ids)
            expect_ids = list(range(self.rank * 72, (self.rank + 1) * 72))
            if local_ids != expect_ids:
                self.problems.append(
                    f"EP local experts {local_ids[:3]}..{local_ids[-1]} != {expect_ids[:3]}.."
                )
        checks.append(("lm_head", tuple(self.model.lm_head.weight.shape), (38720, 4096)))
        for name, got, want in checks:
            if got != want:
                self.problems.append(f"local shape {name}: {got} != {want}")
        if moe.moe_all_reduce is None:
            self.problems.append("Glm5NextMoE.moe_all_reduce missing at tp_size=4")
        self.result["local_shape_checks"] = len(checks)

        # Cross-rank union proof from the per-rank shard records.
        gathered = self.comm.gather(self.tp_shards, root=0)
        if self.rank == 0:
            union_bad: List[str] = []
            names = set()
            for shards in gathered:
                names.update(shards.keys())
            for name in sorted(names):
                rows = [g[name] for g in gathered if name in g]
                if len(rows) != 4:
                    union_bad.append(f"{name}: only {len(rows)} ranks reported")
                    continue
                full = rows[0]["full_shape"]
                if any(r["full_shape"] != full for r in rows):
                    union_bad.append(f"{name}: full_shape disagreement")
                    continue
                mode = rows[0]["mode"]
                if mode == "replicated":
                    if any(r["local_shape"] != full for r in rows):
                        union_bad.append(f"{name}: replicated local != full")
                    continue
                dim = full[0] if mode == "column" else full[1]
                ranges = sorted(tuple(r["range"]) for r in rows)
                cursor = 0
                for start, end in ranges:
                    if start != cursor:
                        union_bad.append(f"{name}: gap/overlap at {start} (expected {cursor})")
                        break
                    cursor = end
                else:
                    if cursor != dim:
                        union_bad.append(f"{name}: union ends at {cursor} != {dim}")
            self.result["union_proof"] = {
                "modules": len(names),
                "failures": union_bad,
            }
            if union_bad:
                self.problems.append(f"shard union failures: {union_bad[:3]}")

    def replicated_bitwise(self) -> None:
        model = self.model.model
        targets = {
            "embed_tokens.weight": model.embed_tokens.weight,
            "final_norm.weight": model.norm.weight,
            "layer4.router.weight": model.layers[MOE_LAYER].mlp.gate.weight,
            "layer4.router.bias": model.layers[MOE_LAYER].mlp.gate.e_score_correction_bias,
            "layer0.input_layernorm": model.layers[0].input_layernorm.weight,
            "layer3.indexer.wk": model.layers[3].self_attn.indexer.wk.weight,
            "layer3.q_a_proj": model.layers[3].self_attn.q_a_proj.weight,
            "layer3.kv_a_proj": model.layers[3].self_attn.kv_a_proj_with_mqa.weight,
            "layer0.f_a_proj": model.layers[0].self_attn.f_a_proj.weight,
            "layer0.g_a_proj": model.layers[0].self_attn.g_a_proj.weight,
            # Goal 5.2 head-shards dt_bias / A_log / the conv filter (they
            # follow the column-sharded q/k/v head ranges); they moved from
            # this replicated-bitwise list into the tp_shards union proof.
            "layer0.o_norm": model.layers[0].self_attn.o_norm_weight,
            "layer3.indexer.ape": model.layers[3].self_attn.indexer.index_kpool_compress_ape,
            "layer3.indexer.gate": model.layers[3].self_attn.indexer.index_kpool_compress_gate,
        }
        for layer_idx in (0, 22, 44):
            for site in ("hc_attn", "hc_ffn"):
                hc = getattr(model.layers[layer_idx], site)
                for pname, p in hc.named_parameters():
                    targets[f"layer{layer_idx}.{site}.{pname}"] = p
        if self.layout == "tp4ep4":
            shared = model.layers[MOE_LAYER].mlp.shared_experts
            targets["layer4.shared.gate"] = shared.gate_proj.weight
            targets["layer4.shared.down"] = shared.down_proj.weight
        digests = {k: tensor_digest(v) for k, v in targets.items()}
        gathered = self.comm.gather(digests, root=0)
        if self.rank == 0:
            mismatched = [k for k in digests if len({g[k] for g in gathered}) != 1]
            self.result["replicated_bitwise"] = {
                "tensors": len(digests),
                "mismatched": mismatched,
            }
            if mismatched:
                self.problems.append(f"replicated state differs across ranks: {mismatched[:5]}")

    # ---- reconstruction replay -------------------------------------------

    def _bcast_input(self, shape: Tuple[int, ...], seed_bump: int) -> torch.Tensor:
        if self.rank == 0:
            g = torch.Generator(device="cpu").manual_seed(SEED + seed_bump)
            x = torch.randn(*shape, generator=g, dtype=torch.float32)
        else:
            x = torch.empty(*shape, dtype=torch.float32)
        self.comm.Bcast(x.numpy(), root=0)
        return x.to(self.device, dtype=torch.bfloat16)

    def _full_tensor(self, key: str) -> torch.Tensor:
        t = self.weights[key]
        t = t if torch.is_tensor(t) else t[:]
        return t.to(self.device)

    def _reference_linear(self, prefix: str, x: torch.Tensor) -> torch.Tensor:
        """Full-weight reference math (the single-rank PP4 form)."""
        from tensorrt_llm._torch.models.modeling_glm5_next import glm5_next_block_fp8_matmul

        w = self._full_tensor(f"{prefix}.weight")
        if w.dtype == torch.float8_e4m3fn:
            s = self._full_tensor(f"{prefix}.weight_scale_inv")
            return glm5_next_block_fp8_matmul(x, w, s)
        return torch.nn.functional.linear(x, w.to(torch.bfloat16))

    def _families(self) -> List[Dict[str, Any]]:
        """The replayed projection families: (name, module, ckpt prefix, kind)."""
        fams: List[Dict[str, Any]] = []
        base = "model.language_model.layers"
        for li in KDA_LAYERS:
            sa = self.model.model.layers[li].self_attn
            p = f"{base}.{li}.self_attn"
            fams += [
                dict(name=f"kda{li}.q_proj", mod=sa.q_proj, ckpt=f"{p}.q_proj", kind="column"),
                dict(
                    name=f"kda{li}.f_b_proj",
                    mod=sa.f_b_proj,
                    ckpt=f"{p}.f_b_proj",
                    kind="column",
                    in_dim=128,
                ),
                dict(name=f"kda{li}.b_proj", mod=sa.b_proj, ckpt=f"{p}.b_proj", kind="column"),
                dict(
                    name=f"kda{li}.f_a_proj",
                    mod=sa.f_a_proj,
                    ckpt=f"{p}.f_a_proj",
                    kind="replicated",
                ),
                dict(
                    name=f"kda{li}.o_proj",
                    mod=sa.o_proj,
                    ckpt=f"{p}.o_proj",
                    kind="row",
                    in_dim=8192,
                ),
            ]
        for li in MLA_LAYERS:
            sa = self.model.model.layers[li].self_attn
            p = f"{base}.{li}.self_attn"
            fams += [
                dict(
                    name=f"mla{li}.q_a_proj",
                    mod=sa.q_a_proj,
                    ckpt=f"{p}.q_a_proj",
                    kind="replicated",
                ),
                dict(
                    name=f"mla{li}.q_b_proj",
                    mod=sa.q_b_proj,
                    ckpt=f"{p}.q_b_proj",
                    kind="column",
                    in_dim=1536,
                ),
                dict(
                    name=f"mla{li}.kv_a_proj",
                    mod=sa.kv_a_proj_with_mqa,
                    ckpt=f"{p}.kv_a_proj_with_mqa",
                    kind="replicated",
                ),
                dict(
                    name=f"mla{li}.kv_b_proj",
                    mod=sa.kv_b_proj,
                    ckpt=f"{p}.kv_b_proj",
                    kind="column",
                    in_dim=512,
                ),
                dict(
                    name=f"mla{li}.o_proj",
                    mod=sa.o_proj,
                    ckpt=f"{p}.o_proj",
                    kind="row",
                    in_dim=16384,
                ),
                dict(
                    name=f"mla{li}.idx.wq_b",
                    mod=sa.indexer.wq_b,
                    ckpt=f"{p}.indexer.wq_b",
                    kind="column",
                    in_dim=1536,
                ),
                dict(
                    name=f"mla{li}.idx.wk",
                    mod=sa.indexer.wk,
                    ckpt=f"{p}.indexer.wk",
                    kind="replicated",
                ),
                dict(
                    name=f"mla{li}.idx.weights_proj",
                    mod=sa.indexer.weights_proj,
                    ckpt=f"{p}.indexer.weights_proj",
                    kind="column",
                ),
            ]
        return fams

    def _run_family(
        self, fam: Dict[str, Any], x_full: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one family eagerly; returns (local input, reconstructed output)."""
        mod = fam["mod"]
        if fam["kind"] == "row":
            start, end = mod.tp_sharding if mod.tp_sharding else (0, x_full.shape[-1])
            x_local = x_full[:, start:end].contiguous()
        else:
            x_local = x_full
        y_local = mod(x_local)
        if fam["kind"] == "column":
            gathered = self.comm.allgather(y_local.detach().float().cpu().numpy())
            y_full = torch.cat([torch.from_numpy(a) for a in gathered], dim=-1)
            return x_local, y_full.to(self.device, dtype=torch.bfloat16)
        return x_local, y_local

    def replay_eager(self) -> None:
        """The B leg: eager reconstruction of every named projection family."""
        rows: List[Dict[str, Any]] = []
        self._eager_cache: Dict[str, torch.Tensor] = {}
        for i, fam in enumerate(self._families()):
            in_dim = fam.get("in_dim", 4096)
            x_full = self._bcast_input((TOKENS, in_dim), seed_bump=i)
            x_local, y = self._run_family(fam, x_full)
            self._eager_cache[fam["name"]] = (fam, x_full)
            row = {"name": fam["name"], "kind": fam["kind"]}
            if fam["kind"] in ("row", "replicated"):
                digests = self.comm.gather(tensor_digest(y), root=0)
                row["cross_rank_bitwise"] = bool(digests and len(set(digests)) == 1)
            if self.rank == 0:
                ref = self._reference_linear(fam["ckpt"], x_full)
                m = metrics(y.float(), ref.float())
                bad = check_envelope(m, LINEAR_ENVELOPE)
                row.update(m)
                row["pass"] = bad is None
                if bad:
                    self.problems.append(f"B {fam['name']}: {bad}")
                if row.get("cross_rank_bitwise") is False:
                    self.problems.append(f"B {fam['name']}: rank outputs differ")
            rows.append(row)
            self.comm.Barrier()

        # Dense MLP: full module forward (gate/up column, down row + its own
        # single in-Linear reduction).
        mlp = self.model.model.layers[DENSE_MLP_LAYER].mlp
        x = self._bcast_input((TOKENS, 4096), seed_bump=101)
        y = mlp(x)
        row = {"name": f"dense_mlp{DENSE_MLP_LAYER}", "kind": "mlp"}
        digests = self.comm.gather(tensor_digest(y), root=0)
        row["cross_rank_bitwise"] = bool(digests and len(set(digests)) == 1)
        if self.rank == 0:
            ref = self._reference_mlp(
                f"model.language_model.layers.{DENSE_MLP_LAYER}.mlp", x, float(mlp.swiglu_limit)
            )
            m = metrics(y.float(), ref.float())
            bad = check_envelope(m, MLP_ENVELOPE)
            row.update(m)
            row["pass"] = bad is None
            if bad:
                self.problems.append(f"B dense_mlp: {bad}")
        self._eager_cache["dense_mlp"] = (mlp, x)
        rows.append(row)
        self.comm.Barrier()

        # MoE composition: routed partial + shared, exactly one reduction.
        moe = self.model.model.layers[MOE_LAYER].mlp
        x = self._bcast_input((16, 4096), seed_bump=202)
        y = moe(x)
        row = {
            "name": f"moe{MOE_LAYER}",
            "kind": "moe",
            "layout": self.layout,
            "backend": moe.moe_backend_name,
        }
        digests = self.comm.gather(tensor_digest(y), root=0)
        row["cross_rank_bitwise"] = bool(digests and len(set(digests)) == 1)
        if self.rank == 0:
            ref = self._reference_moe(moe, x)
            m = metrics(y.float(), ref.float())
            bad = check_envelope(m, MOE_ENVELOPE)
            row.update(m)
            row["pass"] = bad is None
            if bad:
                self.problems.append(f"B moe: {bad}")
        self._eager_cache["moe"] = (moe, x)
        rows.append(row)
        self.comm.Barrier()

        # lm_head logits: vocab-column shards allgathered by the LMHead itself.
        head = self.model.lm_head
        x = self._bcast_input((8, 4096), seed_bump=303)
        y = head(x)
        row = {"name": "lm_head", "kind": "column+gather", "logits_shape": list(y.shape)}
        digests = self.comm.gather(tensor_digest(y), root=0)
        row["cross_rank_bitwise"] = bool(digests and len(set(digests)) == 1)
        if y.shape[-1] != 154880:
            self.problems.append(f"lm_head gathered logits width {y.shape[-1]} != 154880")
        if self.rank == 0:
            ref = self._reference_linear("lm_head", x)
            m = metrics(y.float(), ref.float())
            bad = check_envelope(m, LINEAR_ENVELOPE)
            row.update(m)
            row["pass"] = bad is None
            if bad:
                self.problems.append(f"B lm_head: {bad}")
        self._eager_cache["lm_head"] = (head, x)
        rows.append(row)
        self.result["replay_B"] = rows
        self.comm.Barrier()

    def _reference_mlp(self, prefix: str, x: torch.Tensor, limit: float) -> torch.Tensor:
        from tensorrt_llm._torch.models.modeling_glm5_next import clamped_swiglu

        gate = self._reference_linear(f"{prefix}.gate_proj", x)
        up = self._reference_linear(f"{prefix}.up_proj", x)
        return self._reference_linear(f"{prefix}.down_proj", clamped_swiglu(gate, up, limit))

    def _reference_moe(self, moe: Any, x: torch.Tensor) -> torch.Tensor:
        """Independent routed+shared reference: model router (replicated,
        bitwise-checked) + per-selected-expert block-FP8 math + shared expert,
        all from the full checkpoint tensors."""
        from tensorrt_llm._torch.models.modeling_glm5_next import clamped_swiglu

        prefix = f"model.language_model.layers.{MOE_LAYER}.mlp"
        flat = x.reshape(-1, 4096)
        _, topk_w, topk_i = moe.gate(flat)
        routed = torch.zeros(flat.shape[0], 4096, dtype=torch.float32, device=self.device)
        expert_cache: Dict[int, Tuple[torch.Tensor, ...]] = {}
        for t in range(flat.shape[0]):
            xt = flat[t : t + 1]
            for k in range(topk_i.shape[1]):
                e = int(topk_i[t, k])
                if e not in expert_cache:
                    expert_cache[e] = tuple(
                        (
                            self._full_tensor(f"{prefix}.experts.{e}.{p}.weight"),
                            self._full_tensor(f"{prefix}.experts.{e}.{p}.weight_scale_inv"),
                        )
                        for p in ("gate_proj", "up_proj", "down_proj")
                    )
                from tensorrt_llm._torch.models.modeling_glm5_next import glm5_next_block_fp8_matmul

                (gw, gs), (uw, us), (dw, ds) = expert_cache[e]
                gate = glm5_next_block_fp8_matmul(xt, gw, gs)
                up = glm5_next_block_fp8_matmul(xt, uw, us)
                down = glm5_next_block_fp8_matmul(
                    clamped_swiglu(gate, up, float(moe.swiglu_limit)), dw, ds
                )
                routed[t] += float(topk_w[t, k]) * down[0].float()
        shared = self._reference_mlp(f"{prefix}.shared_experts", flat, float(moe.swiglu_limit))
        return (routed + shared.float()).to(torch.bfloat16).view_as(x)

    # ---- CUDA-graph (E) leg ----------------------------------------------

    def replay_graph(self) -> None:
        """The E leg: capture each family (collectives included) and replay.

        Capture success is the no-fallback proof — capture raises on any host
        sync or allocation the graph cannot own. Replayed outputs must match
        the eager outputs on the same fresh inputs within GRAPH_ENVELOPE
        (bitwise expected and recorded for non-collective families).
        """
        rows: List[Dict[str, Any]] = []
        entries: List[Tuple[str, Any, torch.Tensor, bool]] = []
        for name, (fam_or_mod, x_full) in self._eager_cache.items():
            if isinstance(fam_or_mod, dict):
                fam = fam_or_mod
                mod = fam["mod"]
                collective = fam["kind"] == "row" and self.mapping.tp_size > 1
                if fam["kind"] == "row":
                    start, end = mod.tp_sharding if mod.tp_sharding else (0, x_full.shape[-1])
                    x_local = x_full[:, start:end].contiguous()
                else:
                    x_local = x_full
                entries.append((name, mod, x_local, collective))
            else:
                collective = self.mapping.tp_size > 1
                entries.append((name, fam_or_mod, x_full, collective))

        for name, mod, x_eager, collective in entries:
            self.comm.Barrier()
            row: Dict[str, Any] = {"name": name, "collective_in_graph": collective}
            try:
                static_x = torch.empty_like(x_eager)
                static_x.copy_(x_eager)
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        _ = mod(static_x)
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                self.comm.Barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    static_y = mod(static_x)
                row["captured"] = True
                # Fresh input -> replay must recompute, in lockstep on all ranks.
                fresh = torch.roll(x_eager, shifts=1, dims=0)
                static_x.copy_(fresh)
                self.comm.Barrier()
                graph.replay()
                torch.cuda.synchronize()
                replayed = static_y.detach().clone()
                eager = mod(fresh)
                m = metrics(replayed.float(), eager.float())
                bad = check_envelope(m, GRAPH_ENVELOPE)
                row.update(m)
                row["pass"] = bad is None
                if bad:
                    self.problems.append(f"E {name}: replay-vs-eager {bad}")
                if not collective and not m["bitwise"]:
                    self.problems.append(f"E {name}: non-collective replay not bitwise")
                del graph
            except Exception as exc:  # capture failure = hard-path failure
                row["captured"] = False
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["pass"] = False
                self.problems.append(f"E {name}: capture/replay failed: {exc}")
            rows.append(row)
        self.result["replay_E"] = rows
        self.comm.Barrier()

    # ---- orchestration -----------------------------------------------------

    def run(self, json_path: str) -> int:
        try:
            self.build()
            self.accounting()
            self.replicated_bitwise()
            self.replay_eager()
            self.replay_graph()
        except Exception:
            self.problems.append(f"driver exception: {traceback.format_exc(limit=8)}")
        all_problems = self.comm.gather(list(self.problems), root=0)
        gathered_results = self.comm.gather(self.result, root=0)
        code = 0
        if self.rank == 0:
            merged_problems = [p for ps in all_problems for p in ps]
            ok = not merged_problems
            out = {
                "layout": self.layout,
                "ok": ok,
                "problems": merged_problems,
                "not_exercised": [
                    "kda/mla attention forward at TP4 (head-sharded state/kernels — Goal 5.2)",
                    "routed-expert deep replay vs hooked HF (Goal 5.3)",
                    "trtllm-serve overlap scheduling (Goal 5.4; E here = CUDA-graph capture/replay of module forwards)",
                ],
                "envelopes": {
                    "linear": LINEAR_ENVELOPE,
                    "mlp": MLP_ENVELOPE,
                    "moe": MOE_ENVELOPE,
                    "graph": GRAPH_ENVELOPE,
                },
                "ranks": gathered_results,
            }
            with open(json_path, "w") as f:
                json.dump(out, f, indent=1, default=str)
            log(0, f"layout={self.layout} ok={ok} problems={len(merged_problems)}")
            code = 0 if ok else 1
        code = self.comm.bcast(code, root=0)
        return code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", choices=("tp4", "tp4ep4"), required=True)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    driver = Driver(args.layout)
    return driver.run(args.json)


if __name__ == "__main__":
    rc = main()
    with open(os.environ.get("GLM5_EXIT_FILE", "/tmp/glm5_tp4_dense_exit.txt"), "w") as f:
        f.write(str(rc))
    sys.exit(rc)
