# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit lm-eval ``samples_gsm8k.json`` artifacts (HF reference vs TensorRT-LLM).

Both sides run ``tensorrt_llm.evaluate.GSM8K`` — the exact class ``trtllm-eval``
uses — with the same ``(dataset_path, random_seed, num_samples)``, so their
``doc_id`` spaces align row-for-row. Two subcommands:

``diff``
    Proves the matched-config contract and extracts the discriminating set:
    asserts the rendered prompt strings (``arguments``) are IDENTICAL per
    doc_id, recomputes per-filter scores from the stored ``exact_match``
    values, and lists doc_ids where one side is correct and the other wrong.

``truncation``
    Tokenizes every stored response with the checkpoint tokenizer and reports
    rows that consumed the whole decode budget. A budget-length row counts as
    *truncated* only when its text carries neither a ``####`` answer marker
    nor a task ``until`` stop string — the same engine-equivalent stop
    semantics the HF reference driver applies: a row whose scored text was
    already complete (marker) or already cut at a stop string (until) was not
    shortened by the budget.

Usage::

  python glm5_next_lmeval_diff.py diff --hf <dir-or-file> --trt <dir-or-file> --out diff.json
  python glm5_next_lmeval_diff.py truncation --samples <dir-or-file> \
      --tokenizer /dev/shm/GLM-5.3-Flash --budget 3072 --out trunc.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ANSWER_MARKER = "####"
#: Re-encoding a decoded string can shift the count by a token or two, so a
#: row is treated as budget-length from ``budget - AT_BUDGET_SLACK`` tokens.
AT_BUDGET_SLACK = 8


def load_samples(path: str):
    """Load samples keyed by doc_id, merging per-filter entries.

    lm-eval 0.4.x emits ONE entry per (doc, filter) with the filter name in
    ``entry["filter"]`` and the score in ``entry["exact_match"]``; older
    layouts put ``exact_match,<filter>`` keys on one entry. Handle both.
    """
    if os.path.isdir(path):
        candidates = sorted(
            glob.glob(os.path.join(path, "**", "samples_gsm8k*.json"), recursive=True),
            key=os.path.getmtime,
        )
        assert candidates, f"no samples_gsm8k*.json under {path}"
        path = candidates[-1]
    with open(path) as fh:
        payload = json.load(fh)
    samples = (
        payload["samples"]["gsm8k"]
        if isinstance(payload, dict) and "samples" in payload
        else payload
    )
    if isinstance(samples, dict):
        samples = samples.get("gsm8k", samples)
    rows = {}
    for s in samples:
        doc_id = int(s["doc_id"])
        row = rows.setdefault(doc_id, dict(s))
        if "filter" in s and "exact_match" in s:
            row.setdefault("_filters", {})[s["filter"]] = float(s["exact_match"])
            row.setdefault("_filtered", {})[s["filter"]] = s.get("filtered_resps")
    return path, rows


def prompt_of(sample) -> str:
    args = sample.get("arguments")
    # lm-eval stores arguments either as [[context, gen_kwargs]] or a dict.
    if isinstance(args, list) and args:
        first = args[0]
        if isinstance(first, (list, tuple)) and first:
            return str(first[0])
        return str(first)
    if isinstance(args, dict):
        gen = args.get("gen_args_0", {})
        if isinstance(gen, dict):
            return str(gen.get("arg_0", ""))
    return ""


def response_of(sample) -> str:
    resps = sample.get("resps")
    while isinstance(resps, (list, tuple)) and resps:
        resps = resps[0]
    return str(resps) if resps is not None else ""


def filters_of(sample):
    if "_filters" in sample:
        return dict(sample["_filters"])
    out = {}
    for key, value in sample.items():
        if key.startswith("exact_match,"):
            out[key.split(",", 1)[1]] = float(value)
    return out


def run_diff(args) -> int:
    hf_path, hf = load_samples(args.hf)
    trt_path, trt = load_samples(args.trt)

    report = {
        "hf_samples": hf_path,
        "trt_samples": trt_path,
        "num_hf": len(hf),
        "num_trt": len(trt),
        "doc_id_aligned": sorted(hf.keys()) == sorted(trt.keys()),
        "prompt_mismatches": [],
        "scores": {},
        "discriminating": {},
    }

    shared = sorted(set(hf) & set(trt))
    for doc_id in shared:
        if prompt_of(hf[doc_id]) != prompt_of(trt[doc_id]):
            report["prompt_mismatches"].append(doc_id)

    filter_names = sorted(filters_of(hf[shared[0]])) if shared else []
    for name in filter_names:
        hf_score = sum(filters_of(hf[d]).get(name, 0.0) for d in shared)
        trt_score = sum(filters_of(trt[d]).get(name, 0.0) for d in shared)
        hf_right_trt_wrong = [
            d
            for d in shared
            if filters_of(hf[d]).get(name, 0.0) > 0.5 > filters_of(trt[d]).get(name, 0.0)
        ]
        trt_right_hf_wrong = [
            d
            for d in shared
            if filters_of(trt[d]).get(name, 0.0) > 0.5 > filters_of(hf[d]).get(name, 0.0)
        ]
        report["scores"][name] = {
            "hf": hf_score,
            "trt": trt_score,
            "n": len(shared),
            "hf_pct": round(100.0 * hf_score / max(len(shared), 1), 2),
            "trt_pct": round(100.0 * trt_score / max(len(shared), 1), 2),
        }
        report["discriminating"][name] = {
            "hf_correct_trt_wrong": [
                {
                    "doc_id": d,
                    "question": str(hf[d].get("doc", {}).get("question", ""))[:120],
                    "hf_filtered": hf[d]
                    .get("_filtered", {})
                    .get(name, hf[d].get("filtered_resps")),
                    "trt_filtered": trt[d]
                    .get("_filtered", {})
                    .get(name, trt[d].get("filtered_resps")),
                }
                for d in hf_right_trt_wrong
            ],
            "trt_correct_hf_wrong": [
                {
                    "doc_id": d,
                    "question": str(trt[d].get("doc", {}).get("question", ""))[:120],
                    "hf_filtered": hf[d]
                    .get("_filtered", {})
                    .get(name, hf[d].get("filtered_resps")),
                    "trt_filtered": trt[d]
                    .get("_filtered", {})
                    .get(name, trt[d].get("filtered_resps")),
                }
                for d in trt_right_hf_wrong
            ],
        }

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)

    print(
        f"[diff] aligned={report['doc_id_aligned']} "
        f"prompt_mismatches={len(report['prompt_mismatches'])}"
    )
    for name, s in report["scores"].items():
        disc = report["discriminating"][name]
        print(
            f"[diff] {name}: HF {s['hf']:.0f}/{s['n']} ({s['hf_pct']}%) vs TRT {s['trt']:.0f}/{s['n']} "
            f"({s['trt_pct']}%) | HF-right/TRT-wrong {len(disc['hf_correct_trt_wrong'])} "
            f"| TRT-right/HF-wrong {len(disc['trt_correct_hf_wrong'])}"
        )
    print(f"[diff] wrote {args.out}")
    ok = report["doc_id_aligned"] and not report["prompt_mismatches"]
    return 0 if ok else 1


def run_truncation(args) -> int:
    from transformers import AutoTokenizer

    path, rows = load_samples(args.samples)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    counts = {}
    at_budget = []
    truncated = []
    for doc_id in sorted(rows):
        text = response_of(rows[doc_id])
        n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        counts[doc_id] = n_tokens
        if n_tokens >= args.budget - AT_BUDGET_SLACK:
            has_marker = ANSWER_MARKER in text
            has_until = any(u in text for u in args.until)
            at_budget.append(
                {
                    "doc_id": doc_id,
                    "tokens": n_tokens,
                    "has_marker": has_marker,
                    "has_until": has_until,
                }
            )
            if not has_marker and not has_until:
                truncated.append(doc_id)

    report = {
        "samples": path,
        "num_rows": len(rows),
        "budget": args.budget,
        "at_budget_slack": AT_BUDGET_SLACK,
        "until": list(args.until),
        "max_generated_tokens": max(counts.values()) if counts else 0,
        "mean_generated_tokens": round(sum(counts.values()) / max(len(counts), 1), 1),
        "rows_at_budget": at_budget,
        "truncated_rows": truncated,
    }
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(
        f"[truncation] rows={report['num_rows']} max_tokens={report['max_generated_tokens']} "
        f"at_budget={len(at_budget)} truncated={truncated}"
    )
    print(f"[truncation] wrote {args.out}")
    return 0 if not truncated else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    diff = sub.add_parser("diff", help="matched-config proof + discriminating set")
    diff.add_argument("--hf", required=True)
    diff.add_argument("--trt", required=True)
    diff.add_argument("--out", required=True)

    trunc = sub.add_parser("truncation", help="budget-truncation audit")
    trunc.add_argument("--samples", required=True)
    trunc.add_argument("--tokenizer", required=True)
    trunc.add_argument("--budget", type=int, required=True)
    trunc.add_argument("--until", action="append", default=None)
    trunc.add_argument("--out", required=True)

    args = parser.parse_args()
    if args.mode == "diff":
        return run_diff(args)
    if args.until is None:
        args.until = ["Question:"]
    return run_truncation(args)


if __name__ == "__main__":
    sys.exit(main())
