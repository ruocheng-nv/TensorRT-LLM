# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Goal 5.3 four-rank TP4/EP4 MoE source_activation_replay driver (Stage 5).

Run under ``mpirun -n 4`` on four CUDA GPUs::

    mpirun -n 4 python glm5_next_tp4ep4_moe_replay.py --json out.json

What one invocation proves, per the Stage-5 TP4/EP4 MoE acceptance item:

1. **EP ownership** — the routed path is built through
   ``create_moe``/``ConfigurableMoE`` with the resolved ``TRTLLMGenFusedMoE``
   FP8-block backend; ``Mapping(moe_tp_size=1, moe_ep_size=4)`` assigns exactly
   72 contiguous routed experts per rank with disjoint union 0..287; the FP32
   noaux_tc router weight, ``e_score_correction_bias``, and the one shared
   expert are replicated bitwise. The per-rank ``MoEResolutionReport`` (winner,
   eligibility, environment fingerprint) is recorded and must be identical on
   all four ranks, and every constructed collective is the pinned
   ``AllReduceStrategy.NCCL``.
2. **source_activation_replay (B)** — hooked native-HF ``mlp.input`` rows from
   the real checkpoint (the Stage-1 fixture) drive the production path at
   representative routed layers: FP32 router logits, exact top-8 IDs and
   source-normalized/scaled weights are compared against an independent
   in-driver noaux_tc recomputation from the raw checkpoint router tensors and
   must be bitwise identical across ranks; each rank's **local expert
   partial** (``reduce_results=False``) is compared against the reference sum
   of exactly its own experts' block-FP8 contributions; the combined routed
   output (one NCCL all-reduce — the EP combine), the replicated shared
   output, and the post-MoE result are compared against the from-checkpoint
   reference math and against the hooked in-model HF output. The reference
   rung itself is pinned in-run against the hooked HF rows. Every fused-path
   comparison is gated by the strict predeclared envelopes, with exactly one
   documented, quantitatively bounded, human-authorized exception
   (``C3_EXCEPTION``: the Option-A decision of 2026-09-04 for the
   TRTLLMGenFusedMoE cubin-internal intermediate re-quantization on layer 44
   expert 238). A failing check reports per-token offender diagnostics and
   fails unless *every* offender token qualifies under the exception's
   prerequisites and the remaining tokens still satisfy the strict envelope;
   qualification/application evidence is emitted either way.
3. **Empty-local-token case** — a token subset whose routed experts all live
   on other ranks: the starved rank's local partial must be exactly zero and
   the combined output must still satisfy the reference envelope.
4. **B then E** — the eager legs run first; the E leg captures the full MoE
   forward (router FP32 logits, fused FP8-block experts, the EP-combine NCCL
   all-reduce, shared expert) in a ``torch.cuda.CUDAGraph`` in lockstep on all
   four ranks — once on the fixture batch and once on the empty-local-token
   batch — and replays with fresh inputs. Capture success is the no-fallback
   hard-path proof; replays must match eager within the predeclared envelope.

Module-scope conventions follow the accepted Stage-3/Goal-5.1/Goal-5.2
precedent: ``overlap_scheduler`` is a serving-level property (Goal 5.4 pairs
it at the LLM/serve level); E here means CUDA-graph capture/replay of the
production module forward with the collective captured.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from glm5_next_tp4_dense_loader import (  # noqa: E402
    MLP_ENVELOPE,
    MOE_ENVELOPE,
    check_envelope,
    log,
    metrics,
    tensor_digest,
)
from glm5_next_tp4_dense_loader import Driver as DenseDriver  # noqa: E402

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

#: Routed layers with hooked-HF mlp captures (mlp_layer_types is sparse from
#: index 3). Full expert-math replay runs on first/last; the cheap router
#: contract additionally covers the middle capture.
ROUTER_LAYERS = (3, 23, 44)
REPLAY_LAYERS = (3, 44)

#: The production path runs block-FP8 fused kernels while the hooked in-model
#: output came from HF's own FP8 path — the Stage-1/3-accepted FP8 bounds.
FP8_MODEL_ENVELOPE = {"cosine": 0.995, "rel_max_abs": 8e-2}
#: Independent in-driver recompute of the FP32 router math from checkpoint
#: tensors: same ops on the same fp32 inputs, so agreement is essentially
#: exact; the envelope only absorbs kernel-algorithm nondeterminism.
ROUTER_ENVELOPE = {"cosine": 0.999999, "rel_max_abs": 1e-4}
#: Graph replay vs eager on identical inputs (one NCCL collective inside).
GRAPH_ENVELOPE = {"cosine": 0.9999, "rel_max_abs": 5e-3}
#: The bf16-dequant reference rung against the hooked in-model HF rows: both
#: are faithful source math (HF's own FP8 dynamic path lands within ~1 unit of
#: the dequant rung at |y|~190), so this stays tight and, when it holds, pins
#: the reference as sound at exactly the tokens where the fused kernel drifts.
REF_VS_HF_ENVELOPE = {"cosine": 0.9999, "rel_max_abs": 1e-2}

#: Fused-path gating: every comparison is gated by the strict predeclared
#: envelopes above. When a check fails, the driver emits per-token diagnostics
#: (offending tokens, their reference top-8 experts and intra-expert 128x128
#: scale spreads, and whether the bf16 reference agrees with the hooked-HF row
#: there — i.e. whether production is the outlier) so a failure localizes
#: itself; the diagnostics never convert a failure into a pass.
#:
#: THE ONE AUTHORIZED EXCEPTION (human Option-A decision, 2026-09-04 21:26
#: human_feedback; recorded in acceptance-criteria.md Stage-5 C3 / Stage-3 C2
#: and reports/stage5-tp4-ep4-moe-source-activation-replay.md §4–§5): the
#: resolved TRTLLMGenFusedMoE FP8-block kernel re-quantizes its
#: post-activation intermediate inside pre-compiled sm_100f cubins with an
#: implementation-defined representation (exact recipe Unknown from Python —
#: iteration-47 channel-corrected analysis), and on layer 44 the
#: ill-conditioned expert-238 down_proj row (cancellation factor 26.4)
#: amplifies the difference between two legitimate fp8 intermediate
#: representations beyond the 8% envelope. The frozen, jq-gated
#: characterization (probe_bisect-iter47.{json,log}; hashes in
#: goal5.2-logs/artifacts-sha256-iter47{,-archive}.txt): kernel intermediate
#: elementwise max_abs 3.68 vs the bf16 reference where the source's own
#: faithful 1x128 quantization noise is already 2.25; kernel gemm2 agrees with
#: reference gemm2 on the same intermediate (rel 0.013); the observed +72.1
#: fork reconciles linearly to +74.2; worst-case linear response at the
#: faithful-quant budget is ±562. The same kernel is behind the human-accepted
#: full-1319 GSM8K 96.89 vs HF 96.70. The exception is NARROW and per-token:
#: an offender token qualifies only when ALL ``c3_exception_split``
#: prerequisites hold (characterized layer, expert 238 in the token's
#: reference top-8, reference HF-sound at the token, deviation within the
#: measured ±562 bound); every other layer/expert/token keeps the strict
#: envelopes — MOE_ENVELOPE is NOT widened — and the driver records explicit
#: qualification/application evidence for the session gates.
C3_EXCEPTION = {
    "id": "trtllmgen-fp8-intermediate-l44-e238",
    "authorized_by": (
        "human Option-A decision 2026-09-04T21:26 (progress.yaml human_feedback); "
        "recorded in acceptance-criteria.md Stage-5 C3 / Stage-3 C2"
    ),
    "characterization": (
        "reports/stage5-tp4-ep4-moe-source-activation-replay.md §4–§5; frozen "
        "goal5.3-logs/probe_bisect-iter47.{json,log} "
        "(goal5.2-logs/artifacts-sha256-iter47{,-archive}.txt)"
    ),
    "layer": 44,
    "expert": 238,
    "abs_bound": 562.0,
    "bound_basis": (
        "measured worst-case linear cancellation response at the faithful-quant "
        "grid budget (down_proj cancellation factor 26.4 at channel 1448; "
        "observed fork +72.1 reconciles linearly to +74.2)"
    ),
}


def c3_exception_split(
    layer_idx: int,
    per_tok_abs: torch.Tensor,
    strict_abs: float,
    ref_i: torch.Tensor,
    ref_sound_per_token: torch.Tensor,
    exception: Dict[str, Any] = C3_EXCEPTION,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split strict-envelope offender tokens into exception-qualified and not.

    Pure function (CPU-testable without MPI/GPU). A token whose per-token
    ``max_abs`` exceeds ``strict_abs`` qualifies under the Option-A exception
    only when ALL prerequisites hold:

    * the layer is the characterized layer (44);
    * expert 238 is in the token's *reference* top-8 (the deviation routes
      through the characterized expert);
    * the independent reference agrees with the hooked-HF row at that token
      (reference-faithfulness: production is the outlier, the reference is
      within faithful-quant noise);
    * the absolute deviation is within the measured ±562 worst-case
      cancellation bound.

    Returns ``(qualified, unqualified)`` entry lists; each entry carries the
    token, its deviation, its reference top-8, and the per-check verdicts so
    the session can jq-gate the qualification evidence.
    """
    qualified: List[Dict[str, Any]] = []
    unqualified: List[Dict[str, Any]] = []
    for t in range(int(per_tok_abs.shape[0])):
        v = float(per_tok_abs[t])
        if v <= strict_abs:
            continue
        top8 = [int(e) for e in ref_i[t].tolist()]
        checks = {
            "layer_is_characterized": layer_idx == exception["layer"],
            "expert_in_reference_top8": exception["expert"] in top8,
            "reference_agrees_with_hooked_hf_at_token": bool(ref_sound_per_token[t]),
            "abs_deviation_within_measured_bound": v <= exception["abs_bound"],
        }
        entry = {
            "token": t,
            "token_max_abs": v,
            "strict_abs_threshold": strict_abs,
            "top8_experts": top8,
            "checks": checks,
        }
        (qualified if all(checks.values()) else unqualified).append(entry)
    return qualified, unqualified


class MoeDriver(DenseDriver):
    def __init__(self):
        super().__init__("tp4ep4")
        self.result["driver"] = "glm5_next_tp4ep4_moe_replay"
        self.result["fixture"] = FIXTURE

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _fixture_prompt(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        if not hasattr(self, "_fixture"):
            self._fixture = torch.load(FIXTURE, map_location="cpu", weights_only=False)
        for prompt in self._fixture["prompts"]:
            acts = prompt.get("activations") or {}
            key_in, key_out = f"layer{layer_idx}.mlp.input", f"layer{layer_idx}.mlp.output"
            if key_in in acts and key_out in acts:
                return {
                    "prompt_index": prompt["index"],
                    "x": acts[key_in][0].to(device=self.device, dtype=torch.bfloat16),
                    "y": acts[key_out][0].to(device=self.device, dtype=torch.float32),
                }
        raise KeyError(f"fixture has no mlp capture for layer {layer_idx}")

    def _assert_identical_across_ranks(self, t: torch.Tensor, label: str) -> bool:
        rows = self.comm.allgather(t.detach().to("cpu"))
        same = all(torch.equal(rows[0], r) for r in rows[1:])
        if not same:
            self.problems.append(f"{label}: not bitwise identical across ranks")
        return same

    def _router_reference(
        self, layer_idx: int, flat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Independent noaux_tc recompute from the raw checkpoint tensors.

        Mirrors the source contract, not the model class: FP32 linear ->
        sigmoid -> +e_score_correction_bias (selection only) -> top-8 ->
        weights gathered from the *uncorrected* sigmoid -> sum-normalize
        (+1e-20) -> * routed_scaling_factor=2.5.
        """
        p = f"{LAYER_PREFIX}.{layer_idx}.mlp.gate"
        w = self._full_tensor(f"{p}.weight").float()
        bias = self._full_tensor(f"{p}.e_score_correction_bias").float()
        logits = torch.nn.functional.linear(flat.float(), w)
        scores = logits.sigmoid()
        _, topk_idx = torch.topk(scores + bias, k=8, dim=-1, sorted=False)
        weights = scores.gather(-1, topk_idx)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * 2.5, topk_idx

    def _expert_scale_spread(self, layer_idx: int, expert: int) -> float:
        """Max intra-expert 128x128 scale spread (max/min) across the three mats."""
        cache = self.__dict__.setdefault("_spread_cache", {})
        key = (layer_idx, expert)
        if key not in cache:
            spreads = []
            for p in ("gate_proj", "up_proj", "down_proj"):
                s = self._full_tensor(
                    f"{LAYER_PREFIX}.{layer_idx}.mlp.experts.{expert}.{p}.weight_scale_inv"
                ).float()
                spreads.append(float(s.max() / s.min()))
            cache[key] = max(spreads)
        return cache[key]

    def _check_fused(
        self,
        layer_idx: int,
        label: str,
        prod: torch.Tensor,
        ref_t: torch.Tensor,
        env: Dict[str, float],
        ref_i: torch.Tensor,
        ref_sound_per_token,
        row: Dict[str, Any],
    ) -> bool:
        """Strict predeclared envelope on a fused-kernel output.

        Exactly one documented, human-authorized exception exists
        (``C3_EXCEPTION``, Option A 2026-09-04): when every strict-envelope
        offender token qualifies under ``c3_exception_split`` AND the
        remaining tokens still satisfy the strict envelope, the check passes
        with explicit application evidence recorded in the row. Any
        unqualified offender — wrong layer/expert, reference not HF-sound at
        the token, or deviation beyond the measured ±562 bound — fails the
        check with per-token diagnostics (reference top-8 experts,
        intra-expert 128x128 scale spreads, reference-vs-HF soundness, and
        the per-check qualification verdicts) so the failure localizes
        itself. Diagnostics never convert an unqualified failure into a pass.
        """
        m = metrics(prod.float(), ref_t.float())
        row[label] = m
        bad = check_envelope(m, env)
        if bad is None:
            return True
        a, r = prod.float(), ref_t.float()
        scale = max(1.0, float(r.abs().max()))
        per_tok = (a - r).abs().amax(dim=-1)
        strict = env["rel_max_abs"] * scale
        qualified, unqualified = c3_exception_split(
            layer_idx, per_tok, strict, ref_i, ref_sound_per_token
        )
        if qualified and not unqualified:
            keep = torch.ones(a.shape[0], dtype=torch.bool)
            for q in qualified:
                keep[q["token"]] = False
            remaining = int(keep.sum())
            m_rest = metrics(a[keep], r[keep]) if remaining else None
            bad_rest = check_envelope(m_rest, env) if remaining else (
                "no non-exception tokens remain"
            )
            if bad_rest is None:
                row[f"{label}_exception_applied"] = {
                    "exception_id": C3_EXCEPTION["id"],
                    "authorized_by": C3_EXCEPTION["authorized_by"],
                    "abs_bound": C3_EXCEPTION["abs_bound"],
                    "strict_failure_before_exception": bad,
                    "qualified_tokens": qualified,
                    "remaining_tokens": remaining,
                    "metrics_excluding_exception_tokens": m_rest,
                }
                return True
            self.problems.append(
                f"B moe layer {layer_idx} {label}: non-exception tokens still "
                f"out of envelope after excluding qualified tokens: {bad_rest}"
            )
        diagnostics = []
        for entry in qualified + unqualified:
            t = entry["token"]
            spreads = {
                e: round(self._expert_scale_spread(layer_idx, e), 1)
                for e in entry["top8_experts"]
            }
            diagnostics.append(
                {
                    **entry,
                    "top8_scale_spreads": spreads,
                    "reference_agrees_with_hooked_hf_at_token": bool(ref_sound_per_token[t]),
                    "exception_qualified": entry in qualified,
                }
            )
        row[f"{label}_offender_diagnostics"] = diagnostics
        self.problems.append(
            f"B moe layer {layer_idx} {label}: {bad}; "
            f"offending_tokens={[d['token'] for d in diagnostics]}; "
            f"unqualified_tokens={[e['token'] for e in unqualified]}"
        )
        return False

    def _expert_terms(
        self, layer_idx: int, flat: torch.Tensor, topk_w: torch.Tensor, topk_i: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reference routed output, total and split by EP owner rank.

        Returns ``(per_rank[4, T, H] fp32, total[T, H] fp32)`` computed from
        the raw checkpoint block-FP8 expert tensors with the clamped-SwiGLU
        reference math (the Stage-1/2-verified rung). Rank 0 does the
        checkpoint reads (up to ~5 GB of expert tensors per layer) and
        broadcasts the small result to the other ranks.
        """
        from tensorrt_llm._torch.models.modeling_glm5_next import (
            clamped_swiglu,
            glm5_next_block_fp8_matmul,
        )

        if self.rank != 0:
            per_rank = self.comm.bcast(None, root=0)
            per_rank = per_rank.to(self.device)
            return per_rank, per_rank.sum(dim=0)
        prefix = f"{LAYER_PREFIX}.{layer_idx}.mlp"
        tokens, hidden = flat.shape
        per_rank = torch.zeros(4, tokens, hidden, dtype=torch.float32, device=self.device)
        cache: Dict[int, Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = {}
        for t in range(tokens):
            xt = flat[t : t + 1]
            for k in range(topk_i.shape[1]):
                e = int(topk_i[t, k])
                if e not in cache:
                    cache[e] = tuple(
                        (
                            self._full_tensor(f"{prefix}.experts.{e}.{p}.weight"),
                            self._full_tensor(f"{prefix}.experts.{e}.{p}.weight_scale_inv"),
                        )
                        for p in ("gate_proj", "up_proj", "down_proj")
                    )
                (gw, gs), (uw, us), (dw, ds) = cache[e]
                gate = glm5_next_block_fp8_matmul(xt, gw, gs)
                up = glm5_next_block_fp8_matmul(xt, uw, us)
                down = glm5_next_block_fp8_matmul(clamped_swiglu(gate, up, 10.0), dw, ds)
                per_rank[e // 72, t] += float(topk_w[t, k]) * down[0].float()
        self.comm.bcast(per_rank.detach().cpu(), root=0)
        return per_rank, per_rank.sum(dim=0)

    # ------------------------------------------------------------------
    # phases
    # ------------------------------------------------------------------

    def ep_geometry(self) -> None:
        """EP ownership, backend resolution, scales, replication, NCCL pins."""
        from tensorrt_llm._torch.moe.fused_moe.moe_resolution import resolve_moe_impl

        moe = self.model.model.layers[REPLAY_LAYERS[0]].mlp
        backend = moe.experts.backend
        local_ids = list(moe.experts.initial_local_expert_ids)
        expect = list(range(self.rank * 72, (self.rank + 1) * 72))
        row: Dict[str, Any] = {
            "layer": REPLAY_LAYERS[0],
            "configurable_moe": type(moe.experts).__name__,
            "resolved_backend": type(backend).__name__,
            "local_expert_ids": [local_ids[0], local_ids[-1], len(local_ids)],
            "swiglu_limit_scalar": getattr(backend, "swiglu_limit_scalar", None),
            "op_path": "torch.ops.trtllm.fp8_block_scale_moe_runner",
            "activation": "clamped SwiGLU (gemm1_clamp_limit=10.0)",
            "w3_w1_weight": list(backend.w3_w1_weight.shape),
            "w3_w1_weight_dtype": str(backend.w3_w1_weight.dtype),
            "w3_w1_scale": list(backend.w3_w1_weight_scaling_factor.shape),
            "w2_weight": list(backend.w2_weight.shape),
            "w2_scale": list(backend.w2_weight_scaling_factor.shape),
            "moe_all_reduce_strategy": moe.moe_all_reduce.strategy.name
            if moe.moe_all_reduce is not None
            else None,
        }
        ok = (
            type(moe.experts).__name__ == "ConfigurableMoE"
            and type(backend).__name__ == "TRTLLMGenFusedMoE"
            and local_ids == expect
            and row["swiglu_limit_scalar"] == 10.0
            and tuple(backend.w3_w1_weight.shape)[0] == 72
            and tuple(backend.w2_weight.shape)[0] == 72
            and tuple(backend.w3_w1_weight_scaling_factor.shape)[0] == 72
            and row["moe_all_reduce_strategy"] == "NCCL"
        )
        row["pass"] = ok
        if not ok:
            self.problems.append(f"ep geometry: {row}")

        # Disjoint union 0..287 across the four ranks.
        gathered_ids = self.comm.allgather(local_ids)
        union = sorted(i for ids in gathered_ids for i in ids)
        if union != list(range(288)):
            self.problems.append(
                f"EP expert union is not exactly 0..287 (len {len(union)}, "
                f"head {union[:3]}, tail {union[-3:]})"
            )
        overlap = sum(len(ids) for ids in gathered_ids) - len(set(union))
        if overlap:
            self.problems.append(f"EP expert ranges overlap by {overlap} ids")

        # Replicated-bitwise router / correction bias / shared expert on the
        # replay layers themselves (the dense driver pinned layer 4; the
        # criterion wants it on the representative routed layers under test).
        digests = {}
        for layer_idx in REPLAY_LAYERS:
            lm = self.model.model.layers[layer_idx].mlp
            digests[f"layer{layer_idx}.router.weight"] = tensor_digest(lm.gate.weight)
            digests[f"layer{layer_idx}.router.bias"] = tensor_digest(
                lm.gate.e_score_correction_bias
            )
            digests[f"layer{layer_idx}.shared.gate"] = tensor_digest(
                lm.shared_experts.gate_proj.weight
            )
            digests[f"layer{layer_idx}.shared.down"] = tensor_digest(
                lm.shared_experts.down_proj.weight
            )
        gathered = self.comm.allgather(digests)
        mismatched = [k for k in digests if len({g[k] for g in gathered}) != 1]
        if mismatched:
            self.problems.append(f"replicated router/shared state differs: {mismatched}")
        row["replicated_bitwise_tensors"] = len(digests)
        row["replicated_bitwise_mismatched"] = mismatched

        # Per-rank MoE resolution report: winner + eligibility + environment
        # fingerprint must agree on every rank (backend degradation on any one
        # rank would otherwise hang or silently diverge the communication).
        from tensorrt_llm.models.modeling_utils import QuantConfig
        from tensorrt_llm.quantization.mode import QuantAlgo

        report = resolve_moe_impl(
            self.model.model_config,
            override_quant_config=QuantConfig(
                quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=128
            ),
            dtype=torch.bfloat16,
            num_experts=288,
            hidden_size=4096,
            intermediate_size=2048,
            routing=getattr(moe.experts, "routing_method", None),
            layer_idx=REPLAY_LAYERS[0],
        ).to_dict()
        resolution = {
            "winner": report["winner"],
            "selected_by": report["selected_by"],
            "eligible": report["eligible"],
            "env_fingerprint": report["env_fingerprint"],
        }
        row["moe_resolution"] = resolution
        gathered_res = self.comm.allgather(resolution)
        if any(g != gathered_res[0] for g in gathered_res[1:]):
            self.problems.append(f"MoEResolutionReport disagrees across ranks: {gathered_res}")
        if resolution["winner"] != "TRTLLMGenFusedMoE":
            self.problems.append(f"resolution winner {resolution['winner']!r} != TRTLLMGenFusedMoE")

        # Intra-expert 128x128 scale-spread census on the replay layers
        # (rank 0; checkpoint metadata, identical on all ranks) — diagnostic
        # context for any fused-path failure (kept from the iteration-43
        # investigation; spread turned out to be a correlate, not the trigger).
        if self.rank == 0:
            census = {}
            for layer_idx in REPLAY_LAYERS:
                spreads = []
                for e in range(288):
                    s = self._full_tensor(
                        f"{LAYER_PREFIX}.{layer_idx}.mlp.experts.{e}.down_proj.weight_scale_inv"
                    ).float()
                    spreads.append((float(s.max() / s.min()), e))
                spreads.sort(reverse=True)
                census[str(layer_idx)] = {
                    "top4_down_proj_spread": [[e, round(r, 1)] for r, e in spreads[:4]],
                }
            row["scale_spread_census"] = census
        self.result["ep_geometry"] = row
        self.comm.Barrier()

    def router_replay(self) -> None:
        """FP32 logits / exact top-8 IDs / scaled weights on hooked-HF inputs."""
        rows = []
        for layer_idx in ROUTER_LAYERS:
            fx = self._fixture_prompt(layer_idx)
            flat = fx["x"]
            moe = self.model.model.layers[layer_idx].mlp
            with torch.no_grad():
                logits = moe.gate.logits(flat)
                _, topk_w, topk_i = moe.gate(flat)
            ref_logits, ref_w, ref_i = self._router_reference(layer_idx, flat)
            ids_match = all(set(a) == set(b) for a, b in zip(topk_i.tolist(), ref_i.tolist()))
            # Weight comparison in matching order: sort both by expert id.
            order = topk_i.argsort(dim=-1)
            ref_order = ref_i.argsort(dim=-1)
            m_logits = metrics(logits, ref_logits)
            m_w = metrics(topk_w.gather(-1, order), ref_w.gather(-1, ref_order))
            same_logits = self._assert_identical_across_ranks(
                logits, f"router layer {layer_idx} logits"
            )
            same_ids = self._assert_identical_across_ranks(
                topk_i.gather(-1, order), f"router layer {layer_idx} topk ids"
            )
            row = {
                "layer": layer_idx,
                "prompt_index": fx["prompt_index"],
                "tokens": int(flat.shape[0]),
                "router_dtype": "float32",
                "logits_vs_reference": m_logits,
                "weights_vs_reference": m_w,
                "top8_ids_exact_match": ids_match,
                "weights_sum_over_scaling": float((topk_w.sum(dim=-1) / 2.5).mean()),
                "cross_rank_bitwise": {"logits": same_logits, "topk_ids": same_ids},
            }
            ok = (
                ids_match
                and same_logits
                and same_ids
                and check_envelope(m_logits, ROUTER_ENVELOPE) is None
                and check_envelope(m_w, ROUTER_ENVELOPE) is None
            )
            row["pass"] = ok
            if not ok:
                self.problems.append(
                    f"router layer {layer_idx}: ids={ids_match} "
                    f"logits={check_envelope(m_logits, ROUTER_ENVELOPE)} "
                    f"weights={check_envelope(m_w, ROUTER_ENVELOPE)}"
                )
            rows.append(row)
        self.result["router_replay"] = rows
        self.comm.Barrier()

    def fixture_moe_replay(self) -> None:
        """source_activation_replay (B): hooked-HF rows through the EP4 path."""
        rows = []
        self._graph_inputs: Dict[int, torch.Tensor] = {}
        for layer_idx in REPLAY_LAYERS:
            fx = self._fixture_prompt(layer_idx)
            flat = fx["x"]
            moe = self.model.model.layers[layer_idx].mlp
            self._graph_inputs[layer_idx] = flat
            with torch.no_grad():
                logits = moe.gate.logits(flat)
                _, topk_w, topk_i = moe.gate(flat)
                local_partial = moe.experts(flat, logits)
                combined = moe.moe_all_reduce(local_partial.contiguous())
                shared = moe.shared_experts(flat)
                post = moe(flat)
            # Decomposition identity: the module's own forward must be the
            # combine plus the replicated shared contribution added once.
            m_decomp = metrics(post.float(), (combined + shared).float())
            same_post = self._assert_identical_across_ranks(
                post, f"moe layer {layer_idx} post output"
            )
            same_shared = self._assert_identical_across_ranks(
                shared, f"moe layer {layer_idx} shared output"
            )
            # Per-rank local-token census for the record.
            owners = (topk_i // 72).tolist()
            local_tokens = sum(1 for o in owners if self.rank in set(o))

            # References from the raw checkpoint tensors, driven by the
            # *independent* router recompute (router_replay already proved the
            # model gate matches it): ref router -> ref block-FP8 experts is a
            # fully model-independent oracle for the fused path.
            _, ref_w, ref_i = self._router_reference(layer_idx, flat)
            per_rank_ref, routed_ref = self._expert_terms(layer_idx, flat, ref_w, ref_i)
            shared_ref = self._reference_mlp(
                f"{LAYER_PREFIX}.{layer_idx}.mlp.shared_experts", flat, 10.0
            )
            ref_post = routed_ref + shared_ref.float()
            # The reference rung pinned against the hooked in-model HF rows,
            # in-run: when this holds and production drifts, production is the
            # outlier at exactly the drifting tokens.
            m_ref_hf = metrics(ref_post, fx["y"])
            hf_scale = max(1.0, float(fx["y"].abs().max()))
            ref_sound = (ref_post - fx["y"]).abs().amax(dim=-1) <= REF_VS_HF_ENVELOPE[
                "rel_max_abs"
            ] * hf_scale
            row = {
                "layer": layer_idx,
                "prompt_index": fx["prompt_index"],
                "tokens": int(flat.shape[0]),
                "local_tokens_this_rank": local_tokens,
                "reference_vs_hooked_hf": m_ref_hf,
                "forward_decomposition": m_decomp,
                "cross_rank_bitwise": {"post": same_post, "shared": same_shared},
            }
            ok = same_post and same_shared
            bad_ref = check_envelope(m_ref_hf, REF_VS_HF_ENVELOPE)
            if bad_ref:
                ok = False
                self.problems.append(
                    f"B moe layer {layer_idx} reference_vs_hooked_hf: {bad_ref} "
                    "(the reference rung itself drifted; no outlier classification is valid)"
                )
            m_shared = metrics(shared.float(), shared_ref.float())
            row["shared_vs_reference"] = m_shared
            if check_envelope(m_shared, MLP_ENVELOPE):
                ok = False
                self.problems.append(
                    f"B moe layer {layer_idx} shared_vs_reference: "
                    f"{check_envelope(m_shared, MLP_ENVELOPE)}"
                )
            if check_envelope(m_decomp, GRAPH_ENVELOPE):
                ok = False
                self.problems.append(
                    f"B moe layer {layer_idx} forward_decomposition: "
                    f"{check_envelope(m_decomp, GRAPH_ENVELOPE)}"
                )
            ok &= self._check_fused(
                layer_idx,
                "local_partial_vs_reference",
                local_partial,
                per_rank_ref[self.rank],
                MOE_ENVELOPE,
                ref_i,
                ref_sound,
                row,
            )
            ok &= self._check_fused(
                layer_idx,
                "combined_routed_vs_reference",
                combined,
                routed_ref,
                MOE_ENVELOPE,
                ref_i,
                ref_sound,
                row,
            )
            ok &= self._check_fused(
                layer_idx,
                "post_moe_vs_reference",
                post,
                ref_post,
                MOE_ENVELOPE,
                ref_i,
                ref_sound,
                row,
            )
            ok &= self._check_fused(
                layer_idx,
                "post_moe_vs_hooked_hf",
                post,
                fx["y"],
                FP8_MODEL_ENVELOPE,
                ref_i,
                ref_sound,
                row,
            )
            row["pass"] = bool(ok)
            rows.append(row)
        self.result["fixture_replay_B"] = rows
        self.comm.Barrier()

    def empty_local_tokens(self) -> None:
        """A batch whose routed experts all avoid one rank's local range."""
        layer_idx = REPLAY_LAYERS[0]
        fx = self._fixture_prompt(layer_idx)
        moe = self.model.model.layers[layer_idx].mlp
        candidates = fx["x"]
        with torch.no_grad():
            _, cand_w, cand_i = moe.gate(candidates)
        owners = [set(o) for o in (cand_i // 72).tolist()]
        starved, subset_idx = None, None
        for r in range(4):
            avoid = [t for t, o in enumerate(owners) if r not in o]
            if len(avoid) >= 1 and (starved is None or len(avoid) > len(subset_idx)):
                starved, subset_idx = r, avoid
        row: Dict[str, Any] = {"layer": layer_idx, "source": "fixture tokens"}
        if starved is None:
            # Deterministic synthetic fallback: scan seeded tokens until one
            # rank's range is avoided by a whole batch.
            gen = torch.Generator(device="cpu").manual_seed(20260904)
            pool = (torch.randn(256, 4096, generator=gen) * 0.05).to(
                self.device, dtype=torch.bfloat16
            )
            with torch.no_grad():
                _, _, pi = moe.gate(pool)
            powners = [set(o) for o in (pi // 72).tolist()]
            for r in range(4):
                avoid = [t for t, o in enumerate(powners) if r not in o]
                if len(avoid) >= 4:
                    starved, subset_idx, candidates = r, avoid[:8], pool
                    row["source"] = "synthetic scan"
                    break
        if starved is None:
            self.problems.append("empty-local-token: no starving batch found")
            row["pass"] = False
            self.result["empty_local_tokens"] = row
            self.comm.Barrier()
            return
        subset_idx = subset_idx[:8]
        sub = candidates[subset_idx].contiguous()
        self._empty_local = (layer_idx, sub, starved)
        with torch.no_grad():
            logits = moe.gate.logits(sub)
            partial = moe.experts(sub, logits)
            combined = moe.moe_all_reduce(partial.contiguous())
            post = moe(sub)
        partial_abs = float(partial.detach().float().abs().max())
        _, ref_w, ref_i = self._router_reference(layer_idx, sub)
        per_rank_ref, routed_ref = self._expert_terms(layer_idx, sub, ref_w, ref_i)
        shared_ref = self._reference_mlp(
            f"{LAYER_PREFIX}.{layer_idx}.mlp.shared_experts", sub, 10.0
        )
        m_combined = metrics(combined.float(), routed_ref)
        m_post = metrics(post.float(), routed_ref + shared_ref.float())
        starved_ref_abs = float(per_rank_ref[starved].abs().max())
        row.update(
            {
                "starved_rank": starved,
                "subset_tokens": len(subset_idx),
                "local_partial_max_abs_on_starved_rank": partial_abs
                if self.rank == starved
                else None,
                "reference_starved_contribution_max_abs": starved_ref_abs,
                "combined_vs_reference": m_combined,
                "post_vs_reference": m_post,
            }
        )
        ok = (
            check_envelope(m_combined, MOE_ENVELOPE) is None
            and check_envelope(m_post, MOE_ENVELOPE) is None
            and starved_ref_abs == 0.0
        )
        if self.rank == starved and partial_abs != 0.0:
            ok = False
            self.problems.append(
                f"empty-local-token: starved rank {starved} produced a nonzero "
                f"local partial (max_abs {partial_abs})"
            )
        if not ok and check_envelope(m_combined, MOE_ENVELOPE):
            self.problems.append(
                f"empty-local-token combined: {check_envelope(m_combined, MOE_ENVELOPE)}"
            )
        if not ok and check_envelope(m_post, MOE_ENVELOPE):
            self.problems.append(f"empty-local-token post: {check_envelope(m_post, MOE_ENVELOPE)}")
        row["pass"] = ok
        self.result["empty_local_tokens"] = row
        self.comm.Barrier()

    def replay_graph_moe(self) -> None:
        """E leg: capture the full MoE forward (EP-combine NCCL inside)."""
        rows = []
        entries: List[Tuple[str, int, torch.Tensor]] = [
            (f"moe{layer_idx}.fixture", layer_idx, self._graph_inputs[layer_idx])
            for layer_idx in REPLAY_LAYERS
        ]
        if hasattr(self, "_empty_local"):
            layer_idx, sub, starved = self._empty_local
            entries.append((f"moe{layer_idx}.empty_local_rank{starved}", layer_idx, sub))
        for name, layer_idx, x_eager in entries:
            moe = self.model.model.layers[layer_idx].mlp
            row: Dict[str, Any] = {
                "name": name,
                "layer": layer_idx,
                "tokens": int(x_eager.shape[0]),
                "collectives": ["EP-combine NCCL all-reduce"],
            }
            try:
                self.comm.Barrier()
                static_x = torch.empty_like(x_eager)
                static_x.copy_(x_eager)
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side), torch.no_grad():
                    for _ in range(3):
                        moe(static_x)
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                self.comm.Barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph), torch.no_grad():
                    static_y = moe(static_x)
                row["captured"] = True
                fresh = torch.roll(x_eager, shifts=1, dims=0)
                static_x.copy_(fresh)
                self.comm.Barrier()
                graph.replay()
                torch.cuda.synchronize()
                replayed = static_y.detach().clone()
                with torch.no_grad():
                    eager = moe(fresh)
                m = metrics(replayed.float(), eager.float())
                bad = check_envelope(m, GRAPH_ENVELOPE)
                row["replay_vs_eager"] = m
                row["pass"] = bad is None
                if bad:
                    self.problems.append(f"E {name}: replay-vs-eager {bad}")
                del graph
            except Exception as exc:
                row.update(
                    {"captured": False, "error": f"{type(exc).__name__}: {exc}", "pass": False}
                )
                self.problems.append(f"E {name}: capture/replay failed: {exc}")
            rows.append(row)
        self.result["replay_E"] = rows
        self.comm.Barrier()

    # ------------------------------------------------------------------
    # orchestration
    # ------------------------------------------------------------------

    def run(self, json_path: str) -> int:  # noqa: D102 — see module docstring
        try:
            t0 = time.time()
            self.build()
            self.ep_geometry()
            self.router_replay()
            self.fixture_moe_replay()
            self.empty_local_tokens()
            self.replay_graph_moe()
            self.result["driver_seconds"] = round(time.time() - t0, 1)
        except Exception:
            self.problems.append(f"driver exception: {traceback.format_exc(limit=8)}")
        all_problems = self.comm.gather(list(self.problems), root=0)
        gathered = self.comm.gather(self.result, root=0)
        code = 0
        if self.rank == 0:
            merged = [p for ps in all_problems for p in ps]
            ok = not merged
            # Option-A exception accounting: collect every application of the
            # authorized layer-44 expert-238 exception (pass evidence the
            # session jq-gates) and every UNQUALIFIED offender (a real
            # failure). ``ok`` reflects the strict result: an applied
            # exception appends no problem; anything unqualified does.
            fused_labels = (
                "local_partial_vs_reference",
                "combined_routed_vs_reference",
                "post_moe_vs_reference",
                "post_moe_vs_hooked_hf",
            )
            offenders, applications, non_fused = [], [], []
            for rr in gathered:
                for rowset in rr.get("fixture_replay_B") or []:
                    for lbl in fused_labels:
                        for d in rowset.get(f"{lbl}_offender_diagnostics", []) or []:
                            offenders.append(
                                {
                                    "rank": rr["rank"],
                                    "layer": rowset["layer"],
                                    "label": lbl,
                                    **d,
                                }
                            )
                        app = rowset.get(f"{lbl}_exception_applied")
                        if app:
                            applications.append(
                                {
                                    "rank": rr["rank"],
                                    "layer": rowset["layer"],
                                    "label": lbl,
                                    **app,
                                }
                            )
            fused_problem_prefixes = tuple(f"B moe layer 44 {lbl}" for lbl in fused_labels)
            for p in merged:
                if not p.startswith(fused_problem_prefixes):
                    non_fused.append(p)
            self._c3_exception = {
                "c3_source_parity_passed": ok,
                "exception_id": C3_EXCEPTION["id"],
                "exception_authorized_by": C3_EXCEPTION["authorized_by"],
                "exception_abs_bound": C3_EXCEPTION["abs_bound"],
                "exception_applications": applications,
                "exception_layers": sorted({e["layer"] for e in applications}),
                "exception_labels": sorted({e["label"] for e in applications}),
                "exception_token_count": sum(
                    len(e["qualified_tokens"]) for e in applications
                ),
                "unqualified_offender_count": len(offenders),
                "non_fused_regressions": non_fused,
            }
            out = {
                "driver": "glm5_next_tp4ep4_moe_replay",
                "layout": "tp4ep4",
                "ok": ok,
                "problems": merged,
                "c3_exception": self._c3_exception,
                "moe_backend": "ConfigurableMoE -> TRTLLMGenFusedMoE (FP8 block scales)",
                "op_path": "torch.ops.trtllm.fp8_block_scale_moe_runner",
                "activation": "clamped SwiGLU, gemm1_clamp_limit=swiglu_limit_scalar=10.0",
                "conventions": [
                    "E = module-scope CUDA-graph capture/replay with the EP-combine "
                    "collective inside (accepted Stage-3/Goal-5.1/5.2 convention)",
                    "overlap_scheduler is a serving-level property owned by Goal 5.4",
                    "references: hooked native-HF mlp rows (FP8 envelope) and "
                    "from-checkpoint block-FP8 clamped-SwiGLU expert math (the "
                    "Stage-1/2-verified rung), pinned against each other in-run",
                    "strict envelopes with exactly ONE documented, quantitatively "
                    "bounded, human-authorized exception (Option A 2026-09-04): "
                    "layer-44 expert-238 routed-output effects of the "
                    "TRTLLMGenFusedMoE cubin-internal intermediate re-quantization, "
                    "per-token qualified (reference HF-sound, expert 238 in the "
                    "reference top-8, deviation within the measured +/-562 bound) "
                    "with application evidence in c3_exception; every other "
                    "layer/expert/token keeps the strict envelopes and MOE_ENVELOPE "
                    "is not widened (see "
                    "reports/stage5-tp4-ep4-moe-source-activation-replay.md)",
                ],
                "c3_exception_definition": C3_EXCEPTION,
                "envelopes": {
                    "fp8_model": FP8_MODEL_ENVELOPE,
                    "moe": MOE_ENVELOPE,
                    "mlp": MLP_ENVELOPE,
                    "router": ROUTER_ENVELOPE,
                    "graph": GRAPH_ENVELOPE,
                    "ref_vs_hf": REF_VS_HF_ENVELOPE,
                },
                "ranks": gathered,
            }
            with open(json_path, "w") as f:
                json.dump(out, f, indent=1, default=str)
            log(0, f"moe replay ok={ok} problems={len(merged)}")
            code = 0 if ok else 1
        code = self.comm.bcast(code, root=0)
        return code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    return MoeDriver().run(args.json)


if __name__ == "__main__":
    rc = main()
    with open(os.environ.get("GLM5_EXIT_FILE", "/tmp/glm5_tp4ep4_moe_exit.txt"), "w") as f:
        f.write(str(rc))
    sys.exit(rc)
