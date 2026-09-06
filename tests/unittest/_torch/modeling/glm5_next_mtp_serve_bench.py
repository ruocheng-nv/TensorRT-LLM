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
"""Paired ``trtllm-serve`` decode-throughput benchmark: MTP on vs off.

One session launches two servers *sequentially* on the same GPUs with the
same engine geometry and the same ``--extra_llm_api_options`` YAML except
for the ``speculative_config`` section, performs identical warmups, then
runs ``--repeats`` identical measured rounds of ``--concurrency`` concurrent
``/v1/completions`` requests (greedy, ``ignore_eos`` so every request decodes
exactly ``--max-tokens`` tokens). Output tokens per second per round are
computed from the servers' own ``usage.completion_tokens`` and wall time;
the paired medians are the comparison the MTP acceptance criterion asks for
(MTP output-token/s strictly higher than non-MTP at concurrency 8).

Example (deployment geometry, config E)::

    CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=$PWD python3 \\
        tests/unittest/_torch/modeling/glm5_next_mtp_serve_bench.py \\
        --tp 4 --ep 4 --config E --concurrency 8 --max-tokens 256 \\
        --summary out/mtp_bench_e.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from typing import List, Optional

import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, *[".."] * 4))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from glm5_next_trtllm_serve_smoke import (  # noqa: E402  (script-dir import)
    HOST,
    _post_completion,
    free_port,
    shutdown_server,
    wait_ready,
)

PROMPTS = [
    "Explain, step by step, how rain forms:",
    "Describe a lighthouse keeper's morning routine:",
    "Write a short story about a robot learning to paint:",
    "Summarize the causes of the French Revolution:",
    "Question: A train travels 60 miles per hour for 2.5 hours. How far does it go?\nAnswer:",
    "List ten uses for a paperclip and explain each briefly:",
    "Give a beginner's guide to sourdough baking:",
    "Explain how a hash table works to a new programmer:",
]


def write_yaml(path: str, *, enabled: bool, mtp: bool, kv_fraction: float) -> str:
    text = (
        ("cuda_graph_config: {}\n" if enabled else "cuda_graph_config: null\n")
        + f"disable_overlap_scheduler: {str(not enabled).lower()}\n"
        + "kv_cache_config:\n"
        + "  enable_block_reuse: false\n"
        + f"  free_gpu_memory_fraction: {kv_fraction}\n"
    )
    if mtp:
        text += "speculative_config:\n  decoding_type: MTP\n  num_nextn_predict_layers: 1\n"
    with open(path, "w") as fh:
        fh.write(text)
    return text


def serve_command(args, port: int, yaml_path: str) -> List[str]:
    return [
        sys.executable,
        "-m",
        "tensorrt_llm.commands.serve",
        args.model,
        "--backend",
        "pytorch",
        "--host",
        HOST,
        "--port",
        str(port),
        "--tensor_parallel_size",
        str(args.tp),
        "--moe_expert_parallel_size",
        str(args.ep),
        "--max_batch_size",
        str(args.max_batch_size),
        "--max_num_tokens",
        str(args.max_num_tokens),
        "--max_seq_len",
        str(args.max_seq_len),
        "--log_level",
        "info",
        "--extra_llm_api_options",
        yaml_path,
    ]


def measured_round(
    base_url: str, model_id: str, prompts: List[str], max_tokens: int, timeout: float
):
    rows: List[Optional[dict]] = [None] * len(prompts)

    def one(i: int, prompt: str) -> None:
        rows[i] = _post_completion(base_url, model_id, prompt, max_tokens, timeout)

    started = time.time()
    threads = [threading.Thread(target=one, args=(i, p)) for i, p in enumerate(prompts)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=timeout)
    elapsed = time.time() - started
    completed = [r for r in rows if r and r.get("status") == 200]
    tokens = sum(int(r.get("completion_tokens") or 0) for r in completed)
    return {
        "seconds": round(elapsed, 3),
        "requests_ok": len(completed),
        "requests": len(prompts),
        "output_tokens": tokens,
        "output_tokens_per_second": round(tokens / elapsed, 2) if elapsed > 0 else None,
        "errors": [r for r in rows if not r or r.get("status") != 200][:3],
    }


def run_mode(args, *, mtp: bool, out_dir: str, runlog: str) -> dict:
    label = "mtp" if mtp else "baseline"
    yaml_path = os.path.join(out_dir, f"serve_{label}_{args.config.lower()}.yaml")
    yaml_text = write_yaml(
        yaml_path, enabled=args.config == "E", mtp=mtp, kv_fraction=args.kv_fraction
    )
    port = free_port()
    argv = serve_command(args, port, yaml_path)
    base_url = f"http://{HOST}:{port}"
    result = {"label": label, "yaml": yaml_text, "command": argv, "base_url": base_url}
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", _REPO)
    print(f"[mtp-bench] launching {label}: {' '.join(argv)}", flush=True)
    with open(runlog, "ab") as logfh:
        proc = subprocess.Popen(argv, stdout=logfh, stderr=subprocess.STDOUT, env=env)
        try:
            readiness = wait_ready(proc, base_url, args.readiness_timeout)
            result["readiness"] = readiness
            print(f"[mtp-bench] {label} readiness: {readiness}", flush=True)
            if not readiness.get("ready"):
                result["problems"] = ["server never became ready"]
                return result
            models = requests.get(f"{base_url}/v1/models", timeout=60).json()["data"]
            model_id = models[0]["id"]
            prompts = (PROMPTS * ((args.concurrency + len(PROMPTS) - 1) // len(PROMPTS)))[
                : args.concurrency
            ]
            # Identical warmups on both servers (autotuner / graph capture settle).
            for _ in range(args.warmups):
                measured_round(base_url, model_id, prompts, args.max_tokens, args.request_timeout)
            rounds = []
            for i in range(args.repeats):
                r = measured_round(
                    base_url, model_id, prompts, args.max_tokens, args.request_timeout
                )
                print(f"[mtp-bench] {label} round {i}: {r}", flush=True)
                rounds.append(r)
            result["rounds"] = rounds
            rates = [r["output_tokens_per_second"] for r in rounds if r["output_tokens_per_second"]]
            result["median_output_tokens_per_second"] = statistics.median(rates) if rates else None
            result["problems"] = [
                f"round {i}: {r['requests_ok']}/{r['requests']} requests ok"
                for i, r in enumerate(rounds)
                if r["requests_ok"] != r["requests"]
            ]
            # Acceptance evidence for the MTP server: its log reports per-iteration
            # spec-dec stats when TLLM_LOG_LEVEL allows; the request-level rate is
            # exposed through the LLM API smoke instead, so here we keep the
            # server's own summary lines if present.
            return result
        finally:
            result["shutdown"] = shutdown_server(proc, args.shutdown_grace)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/dev/shm/GLM-5.3-Flash")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--ep", type=int, default=4)
    parser.add_argument("--config", choices=["B", "E"], default="E")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-num-tokens", type=int, default=8192)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--kv-fraction", type=float, default=0.8)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--readiness-timeout", type=float, default=5400.0)
    parser.add_argument("--shutdown-grace", type=float, default=300.0)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument(
        "--order", choices=["mtp-first", "baseline-first"], default="baseline-first"
    )
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.summary))
    os.makedirs(out_dir, exist_ok=True)
    runlog = args.summary + ".runlog.txt"
    summary = {
        "config": {
            "mapping": f"TP{args.tp}/EP{args.ep}",
            "configuration": args.config,
            "max_batch_size": args.max_batch_size,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "warmups": args.warmups,
            "repeats": args.repeats,
        },
        "ok": False,
        "problems": ["run did not complete"],
    }
    order = [False, True] if args.order == "baseline-first" else [True, False]
    try:
        for mtp in order:
            summary["mtp" if mtp else "baseline"] = run_mode(
                args, mtp=mtp, out_dir=out_dir, runlog=runlog
            )
        base = summary["baseline"].get("median_output_tokens_per_second")
        mtp_rate = summary["mtp"].get("median_output_tokens_per_second")
        problems = summary["baseline"].get("problems", []) + summary["mtp"].get("problems", [])
        summary["comparison"] = {
            "baseline_median_tok_s": base,
            "mtp_median_tok_s": mtp_rate,
            "speedup": (mtp_rate / base) if base and mtp_rate else None,
            "mtp_strictly_higher": bool(base and mtp_rate and mtp_rate > base),
        }
        if not summary["comparison"]["mtp_strictly_higher"]:
            problems.append("MTP median output tok/s is not strictly higher than baseline")
        summary["problems"] = problems
        summary["ok"] = not problems
        print(f"[mtp-bench] comparison: {summary['comparison']}", flush=True)
        return 0 if not problems else 1
    finally:
        with open(args.summary, "w") as fh:
            json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    sys.exit(main())
