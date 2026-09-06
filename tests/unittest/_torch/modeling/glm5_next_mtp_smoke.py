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
"""GLM-5.3-Flash one-model MTP speculative decoding: LLM API smoke.

Brings the engine up with ``speculative_config=MTPDecodingConfig(
num_nextn_predict_layers=1)`` on the requested geometry (the deployment
target is ``--tp 4 --ep 4``), decodes a fixed prompt set greedily, and
records per-request draft acceptance from ``RequestPerfMetrics
.speculative_decoding``. With ``--baseline`` it then tears the engine down,
brings up the *same* configuration without MTP, and compares token ids: MTP
with greedy verification is lossless up to BF16 near-ties, so long common
prefixes and identical final answers are expected, exact equality is not a
gate (task.yaml: per-token equality is diagnostic only).

Run from the repo root with the in-repo package on PYTHONPATH, e.g.::

    CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=$PWD python3 \\
        tests/unittest/_torch/modeling/glm5_next_mtp_smoke.py \\
        --tp 4 --ep 4 --config B --baseline --summary out/mtp_smoke_b.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, *[".."] * 4))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from glm5_next_llm_api_logit_replay import resolve_moe_parallel  # noqa: E402  (script-dir import)

PROMPTS = [
    "The capital of France is",
    "2, 4, 6, 8,",
    "Water is composed of hydrogen and",
    "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as "
    "many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer:",
    "Count upward without stopping: " + " ".join(str(i) for i in range(1, 220)) + " and",
]


def _build_llm(args, *, mtp: bool, moe_llm_kwargs):
    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MTPDecodingConfig

    enabled = args.config == "E"
    kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=1,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
        max_num_tokens=args.max_num_tokens,
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False, free_gpu_memory_fraction=args.kv_fraction
        ),
        disable_overlap_scheduler=not enabled,
        cuda_graph_config=CudaGraphConfig() if enabled else None,
        **moe_llm_kwargs,
    )
    if mtp:
        kwargs["speculative_config"] = MTPDecodingConfig(
            num_nextn_predict_layers=args.num_nextn_predict_layers
        )
    return LLM(**kwargs)


def _rows(outputs):
    rows = []
    for out in outputs:
        seq = out.outputs[0]
        row = {
            "prompt": out.prompt[:80],
            "token_ids": list(seq.token_ids),
            "text": seq.text,
            "finish_reason": seq.finish_reason,
        }
        perf = getattr(seq, "request_perf_metrics", None)
        spec = getattr(perf, "speculative_decoding", None) if perf is not None else None
        if spec is not None and spec.total_draft_tokens > 0:
            row["spec_dec"] = {
                "accepted": int(spec.total_accepted_draft_tokens),
                "drafted": int(spec.total_draft_tokens),
                "acceptance_rate": float(spec.acceptance_rate),
            }
        rows.append(row)
    return rows


def _common_prefix(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/dev/shm/GLM-5.3-Flash")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--ep", type=int, default=4)
    parser.add_argument("--config", choices=["B", "E"], default="B")
    parser.add_argument("--num-nextn-predict-layers", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-num-tokens", type=int, default=4096)
    parser.add_argument("--kv-fraction", type=float, default=0.35)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--baseline", action="store_true", help="also run non-MTP and compare")
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    _moe_tp, _moe_ep, mapping_label, moe_llm_kwargs = resolve_moe_parallel(args.tp, 1, args.ep)
    os.makedirs(os.path.dirname(os.path.abspath(args.summary)), exist_ok=True)

    import tensorrt_llm
    from tensorrt_llm import SamplingParams

    assert tensorrt_llm.__file__.startswith(_REPO), tensorrt_llm.__file__
    summary = {
        "config": {
            "mapping": mapping_label,
            "configuration": args.config,
            "num_nextn_predict_layers": args.num_nextn_predict_layers,
            "max_new_tokens": args.max_new_tokens,
            "max_batch_size": args.max_batch_size,
        },
        "ok": False,
        "problems": ["run did not complete"],
        "phases": {},
    }
    sampling = SamplingParams(max_tokens=args.max_new_tokens, return_perf_metrics=True)
    problems = []

    def write():
        with open(args.summary, "w") as fh:
            json.dump(summary, fh, indent=2)

    llm = None
    try:
        t0 = time.time()
        llm = _build_llm(args, mtp=True, moe_llm_kwargs=moe_llm_kwargs)
        summary["phases"]["mtp_load_seconds"] = round(time.time() - t0, 1)
        print(f"[mtp-smoke] MTP engine up in {summary['phases']['mtp_load_seconds']}s", flush=True)

        t0 = time.time()
        mtp_out = llm.generate(PROMPTS, sampling)
        summary["phases"]["mtp_generate_seconds"] = round(time.time() - t0, 1)
        # Identical repeat: slot reuse + determinism under speculation.
        repeat = llm.generate(PROMPTS, sampling)
        summary["mtp"] = _rows(mtp_out)
        summary["mtp_repeat_identical"] = [
            a["token_ids"] == b["token_ids"] for a, b in zip(summary["mtp"], _rows(repeat))
        ]
        accepted = sum(r.get("spec_dec", {}).get("accepted", 0) for r in summary["mtp"])
        drafted = sum(r.get("spec_dec", {}).get("drafted", 0) for r in summary["mtp"])
        summary["acceptance"] = {
            "accepted": accepted,
            "drafted": drafted,
            "rate": (accepted / drafted) if drafted else None,
        }
        print(f"[mtp-smoke] acceptance {summary['acceptance']}", flush=True)
        for row in summary["mtp"]:
            print(
                f"[mtp-smoke] {row['prompt'][:40]!r} -> {row['text'][:80]!r} {row.get('spec_dec')}"
            )
        if drafted == 0:
            problems.append("no draft tokens were produced: MTP did not engage")
        elif accepted / drafted < 0.3:
            problems.append(f"draft acceptance rate {accepted / drafted:.3f} is implausibly low")
        if not all(summary["mtp_repeat_identical"]):
            problems.append("identical repeat batch produced different tokens under MTP")
        # Persist the MTP leg now: the baseline engine load that follows is
        # the memory-heaviest phase and an external kill there must not lose
        # the evidence already gathered.
        summary["problems"] = problems + (["baseline not yet run"] if args.baseline else [])
        write()
        llm.shutdown()
        llm = None

        if args.baseline:
            t0 = time.time()
            llm = _build_llm(args, mtp=False, moe_llm_kwargs=moe_llm_kwargs)
            summary["phases"]["baseline_load_seconds"] = round(time.time() - t0, 1)
            t0 = time.time()
            base_out = llm.generate(PROMPTS, sampling)
            summary["phases"]["baseline_generate_seconds"] = round(time.time() - t0, 1)
            summary["baseline"] = _rows(base_out)
            comparison = []
            for m, b in zip(summary["mtp"], summary["baseline"]):
                prefix = _common_prefix(m["token_ids"], b["token_ids"])
                comparison.append(
                    {
                        "prompt": m["prompt"][:40],
                        "identical": m["token_ids"] == b["token_ids"],
                        "common_prefix": prefix,
                        "mtp_len": len(m["token_ids"]),
                        "baseline_len": len(b["token_ids"]),
                    }
                )
            summary["comparison"] = comparison
            identical = sum(c["identical"] for c in comparison)
            print(f"[mtp-smoke] MTP vs baseline identical rows: {identical}/{len(comparison)}")
            for c in comparison:
                print(f"[mtp-smoke]   {c}")
            # Diagnostic only (BF16 near-ties may fork); a *short* common prefix
            # on every row would indicate a real defect.
            if all(c["common_prefix"] < 8 for c in comparison):
                problems.append("every MTP row diverges from the baseline within 8 tokens")
            llm.shutdown()
            llm = None

        summary["problems"] = problems
        summary["ok"] = not problems
        write()
        print(f"[mtp-smoke] {'OK' if not problems else 'PROBLEMS'}: {problems}", flush=True)
        return 0 if not problems else 1
    except Exception as exc:  # report, never swallow
        summary["problems"] = problems + [f"exception: {type(exc).__name__}: {exc}"]
        summary["traceback"] = traceback.format_exc()
        write()
        traceback.print_exc()
        return 2
    finally:
        if llm is not None:
            try:
                llm.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
