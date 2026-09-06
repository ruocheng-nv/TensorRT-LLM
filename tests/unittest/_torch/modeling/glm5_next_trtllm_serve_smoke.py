# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage-4 Goal 4.2: the actual ``trtllm-serve`` B/E hard-path smoke.

Unlike ``glm5_next_llm_api_smoke.py`` (in-process LLM API), this driver starts
the REAL ``trtllm-serve`` process (the OpenAI-compatible server over the same
normal LLM API) as a subprocess, waits for readiness on ``/health`` (the server
binds uvicorn only AFTER ``PyTorchLLM`` finished constructing, so 200 means the
engine — and, for config E, its CUDA-graph capture — is fully up), then drives
text-only HTTP request legs and audits the server's own log for hard-path
evidence:

* **B/E configurations via ``--extra_llm_api_options``.** B writes
  ``cuda_graph_config: null`` + ``disable_overlap_scheduler: true`` (the LLM
  API default for graphs is ENABLED, so B must be explicit); E writes
  ``cuda_graph_config: {}`` — coerced by the LLM API to a default
  ``CudaGraphConfig()`` — plus ``disable_overlap_scheduler: false``. Engine
  geometry mirrors the approved LLM API smoke: PP=8 over the 328 GB FP8
  checkpoint, ``max_batch_size=4``, ``max_num_tokens=4096``, KV cache with
  ``enable_block_reuse=false`` (so repeated identical prompts recompute their
  prefill instead of short-circuiting through reused blocks).

* **Request legs** (all text-only; vision/MTP stay inactive — the harvested
  per-rank load report accounts the allowlisted ignored weights):
  1. ``solo_pass_1`` — each of the 4 smoke prompts (incl. the ~450-token
     counting prompt that forces the CuTe ``trtllm::kda_prefill`` >=4-chunk
     path) runs ALONE (bs=1 prefill + 32 decode/cache-reuse steps,
     ``ignore_eos`` for an exact budget).
  2. ``interleave`` — two concurrent STREAMING completions; per-chunk arrival
     timestamps must genuinely interleave (overlapping windows + alternating
     arrivals), proving the scheduler served both requests together.
  3. ``cancel`` — three concurrent requests; the middle one streams and its
     connection is closed mid-decode. The server's disconnect watcher must
     abort it (the runlog must contain the ``is disconnected, abort`` line)
     and both survivors must still complete their exact full budget.
  4. ``solo_pass_2`` — leg 1 repeated verbatim AFTER the concurrency and
     cancellation legs: greedy text + token counts must be identical per
     prompt. Solo runs are composition-controlled (bs=1 throughout), so any
     divergence is real cross-request state corruption (slot/cache leak), not
     the documented batch-shape last-ulp fork the iteration-21 override
     excludes from gating.
  5. ``chat`` — one ``/v1/chat/completions`` request through the checkpoint's
     own chat template.

* **Hard-path audit** reuses the approved helpers: per-rank
  ``audit_graph_ladder`` (every PP rank must log the full ``[1,2,3,4]``
  capture ladder; config B must log none), capture/fallback grep, and the
  production-stack provenance harvest (``GlmKpoolSparseAttention``,
  ``TRTLLMGenFusedMoE``, KDA dispatch, per-rank load report, hybrid-V2 dtype
  override). Because capture happens during engine construction and every
  decode-only batch size 1..4 is captured with zero fallback warnings, every
  decode step served after readiness is a graph REPLAY: the totals section
  quantifies repeated replay as ``completion_tokens_total`` decode steps
  across >= 14 requests against the fixed one-time capture.

* **Infrastructure honesty.** ``disk_preflight`` gates before launch; a
  readiness failure is classified ``infrastructure`` when the runlog shows an
  NCCL-init/disk marker (the iteration-36 transient) so the session policy
  may retry the same command once, preserving attempt-1 artifacts; anything
  else stays a model failure. The summary JSON and an ``<summary>.exit.txt``
  marker are written on EVERY exit path.

Geometry is parametric (``--tp``/``--pp``/``--ep``): the Stage-4 record used
``--pp 8``; Stage-5 single-node production serving uses ``--tp 4`` (TP4) and
``--tp 4 --ep 4`` (TP4/EP4). The per-rank graph-ladder audit keys on
``world_size = tp * pp`` so every worker rank must capture under config E.

Run (four GPUs free; Stage-5 TP4/EP4 config E):
    python tests/unittest/_torch/modeling/glm5_next_trtllm_serve_smoke.py \
        --tp 4 --ep 4 --config E \
        --summary reports/goal5.4-logs/serve-tp4ep4-e.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from typing import List, Optional, Tuple

import requests

# Same rationale as glm5_next_llm_api_smoke: script-style invocation must pin
# the repo root so the serve subprocess (and its MPI worker ranks) resolve the
# in-repo tensorrt_llm package instead of a stale installed wheel.
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 4))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from glm5_next_driver_preflight import (  # noqa: E402  (script-dir import)
    audit_graph_ladder,
    disk_preflight,
    expected_graph_batch_sizes,
)
from glm5_next_llm_api_smoke import (  # noqa: E402  (script-dir import)
    _start_memory_sampler,
    check_cuda_graph_hard_path,
    harvest_production_provenance,
)

HOST = "127.0.0.1"
# openai_server.py's await_disconnected logs this on the server-side abort of a
# disconnected request; it is the positive proof the cancellation reached the
# engine rather than just closing an HTTP socket.
ABORT_MARK = "is disconnected, abort"
# Failure-classification markers: these identify environment/transport
# failures (the iteration-36 NCCL-init transient, disk exhaustion) for which
# the session policy allows ONE verbatim same-parameter retry with attempt-1
# artifacts preserved. Anything unmatched stays classified as a model failure.
INFRA_MARKS = (
    "NcclCommunicator::createComm",
    "ncclCommunicator.cpp",
    "ncclInternalError",
    "ncclSystemError",
    "ncclUnhandledCudaError",
    "NCCL error",
    "Errno 28",
    "No space left on device",
)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested in test_glm5_next_trtllm_serve_smoke.py)
# ---------------------------------------------------------------------------


def parse_sse_line(raw: Optional[str]) -> Optional[str]:
    """Return the payload of an SSE ``data:`` line, or None for non-data."""
    if not raw:
        return None
    line = raw.strip()
    if not line.startswith("data:"):
        return None
    return line[len("data:") :].strip()


def interleave_evidence(times_a: List[float], times_b: List[float]) -> dict:
    """Prove two streams' chunk arrivals genuinely interleaved.

    ``interleaved`` requires a positive joint window, both streams delivering
    >=2 chunks inside it, and >=4 alternations in the merged arrival order —
    a serialized pair (all of A, then all of B) has exactly 1 alternation.
    """
    evidence = {
        "events": [len(times_a), len(times_b)],
        "overlap_seconds": 0.0,
        "alternations": 0,
        "events_in_overlap": [0, 0],
        "interleaved": False,
    }
    if not times_a or not times_b:
        return evidence
    start = max(times_a[0], times_b[0])
    end = min(times_a[-1], times_b[-1])
    evidence["overlap_seconds"] = round(max(0.0, end - start), 3)
    merged = sorted([(t, 0) for t in times_a] + [(t, 1) for t in times_b])
    evidence["alternations"] = sum(
        1 for i in range(1, len(merged)) if merged[i][1] != merged[i - 1][1]
    )
    in_overlap = [sum(1 for t in times if start <= t <= end) for times in (times_a, times_b)]
    evidence["events_in_overlap"] = in_overlap
    evidence["interleaved"] = (
        evidence["overlap_seconds"] > 0 and evidence["alternations"] >= 4 and min(in_overlap) >= 2
    )
    return evidence


def find_server_abort_lines(runlog_path: str, expect_at_least: int = 1) -> Tuple[dict, List[str]]:
    """Grep the runlog for the server-side disconnect-abort proof."""
    lines: List[str] = []
    try:
        with open(runlog_path, errors="replace") as fh:
            for line in fh:
                if ABORT_MARK in line:
                    lines.append(line.rstrip()[-240:])
    except OSError as exc:
        return (
            {"runlog": runlog_path, "error": str(exc)},
            [f"abort audit: runlog unreadable: {exc}"],
        )
    evidence = {"count": len(lines), "lines": lines[:8]}
    problems = []
    if len(lines) < expect_at_least:
        problems.append(
            f"cancellation: expected >={expect_at_least} server-side "
            f"'{ABORT_MARK}' line(s) in the runlog, found {len(lines)} — the "
            "disconnect never aborted the engine request"
        )
    return evidence, problems


def cancel_leg_problems(cancel: dict, budget: int, min_events: int) -> List[str]:
    """Validate the interleaved-cancellation leg from its collected evidence.

    Pure so it is unit-testable deterministically, independent of the live
    concurrent leg (whose survivor requests can transiently fail under a
    loaded host — a transport hiccup there must NOT be silently miscounted as
    the injected budget defect). Gates: the target actually streamed
    ``>=min_events`` chunks and then closed mid-decode; each survivor returned
    HTTP 200 with the full ``budget`` of completion tokens. A survivor that
    failed transport is reported distinctly from one that was short-changed.
    """
    problems: List[str] = []
    target = cancel.get("target", {})
    events = len(target.get("event_times", []))
    if not target.get("closed_early"):
        problems.append(
            f"cancel: the target stream was never closed mid-decode "
            f"(events={events}) — no cancellation exercised"
        )
    if events < min_events:
        problems.append(
            f"cancel: target streamed only {events} events before close "
            f"(need >={min_events} to prove mid-decode)"
        )
    for i, survivor in enumerate(cancel.get("survivors", [])):
        if survivor.get("status") != 200:
            problems.append(
                f"cancel: survivor {i} HTTP {survivor.get('status')} "
                f"(error={survivor.get('error')})"
            )
        elif survivor.get("completion_tokens") != budget:
            problems.append(
                f"cancel: survivor {i} produced {survivor.get('completion_tokens')} "
                f"tokens (expected the full {budget} budget)"
            )
    return problems


def solo_determinism_problems(rows_before: List[dict], rows_after: List[dict]) -> List[str]:
    """Compare two composition-controlled solo passes (greedy => identical)."""
    problems = []
    if len(rows_before) != len(rows_after):
        return [f"solo determinism: pass sizes differ ({len(rows_before)} vs {len(rows_after)})"]
    for i, (before, after) in enumerate(zip(rows_before, rows_after)):
        if before.get("text") != after.get("text"):
            problems.append(
                f"solo determinism: prompt {i} text diverged across identical "
                "solo runs (cross-request state corruption)"
            )
        if before.get("completion_tokens") != after.get("completion_tokens"):
            problems.append(
                f"solo determinism: prompt {i} token count diverged "
                f"({before.get('completion_tokens')} vs {after.get('completion_tokens')})"
            )
    return problems


def classify_failure(runlog_path: str) -> Tuple[str, List[str]]:
    """(failure_class, matched marker lines) for a failed launch/run."""
    matched: List[str] = []
    try:
        with open(runlog_path, errors="replace") as fh:
            for line in fh:
                if any(mark in line for mark in INFRA_MARKS):
                    matched.append(line.rstrip()[-240:])
    except OSError:
        return "model", []
    return ("infrastructure" if matched else "model"), matched[:8]


# The only clean ends of a deliberate SIGTERM shutdown: uvicorn's graceful
# path exits 0, or the default disposition reports death-by-SIGTERM (-15).
# Anything else (1, -11, None, ...) is an abnormal teardown and must FAIL —
# the Stage-4 criterion counts a nonzero server exit as a failure signal.
ACCEPTED_SHUTDOWN_RETURNCODES = (0, -int(signal.SIGTERM))


def shutdown_problems(shutdown: dict) -> List[str]:
    """Gate the server lifecycle: it must live until asked, then die cleanly."""
    problems = []
    if shutdown.get("exited_before_shutdown"):
        problems.append(
            f"server exited (rc={shutdown.get('returncode')}) before shutdown "
            "was requested — it crashed while serving"
        )
        return problems
    if shutdown.get("forced_kill"):
        problems.append("server did not exit within the shutdown grace period and required SIGKILL")
    elif shutdown.get("returncode") not in ACCEPTED_SHUTDOWN_RETURNCODES:
        problems.append(
            f"server exit status {shutdown.get('returncode')} after SIGTERM is outside "
            f"the accepted {set(ACCEPTED_SHUTDOWN_RETURNCODES)} — abnormal teardown "
            "(crash/segfault/nonzero exit)"
        )
    return problems


# ---------------------------------------------------------------------------
# HTTP request legs
# ---------------------------------------------------------------------------


def _post_completion(
    base_url: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> dict:
    started = time.time()
    row = {"prompt_head": prompt[:80], "status": None, "seconds": None}
    try:
        resp = requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": model_id,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "ignore_eos": True,
            },
            timeout=timeout,
        )
        row["status"] = resp.status_code
        if resp.status_code == 200:
            data = resp.json()
            choice = data["choices"][0]
            row["text"] = choice.get("text", "")
            row["finish_reason"] = choice.get("finish_reason")
            row["completion_tokens"] = data.get("usage", {}).get("completion_tokens")
        else:
            row["body"] = resp.text[:300]
    except requests.RequestException as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"[:300]
    row["seconds"] = round(time.time() - started, 1)
    return row


def _stream_completion(
    base_url: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    epoch: float,
    close_after_events: Optional[int] = None,
    close_after_seconds: Optional[float] = None,
) -> dict:
    """One streaming completion; optionally hang up mid-stream (cancellation)."""
    out = {
        "prompt_head": prompt[:80],
        "status": None,
        "event_times": [],
        "text_events": 0,
        "finish_reason": None,
        "closed_early": False,
    }
    session = requests.Session()
    try:
        with session.post(
            f"{base_url}/v1/completions",
            json={
                "model": model_id,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "ignore_eos": True,
                "stream": True,
            },
            stream=True,
            timeout=timeout,
        ) as resp:
            out["status"] = resp.status_code
            if resp.status_code != 200:
                out["body"] = resp.text[:300]
                return out
            started = time.time()
            for raw in resp.iter_lines(decode_unicode=True):
                payload = parse_sse_line(raw)
                if payload is None:
                    continue
                if payload == "[DONE]":
                    break
                out["event_times"].append(round(time.time() - epoch, 3))
                try:
                    choice = json.loads(payload).get("choices", [{}])[0]
                    if choice.get("text"):
                        out["text_events"] += 1
                    if choice.get("finish_reason"):
                        out["finish_reason"] = choice["finish_reason"]
                except ValueError:
                    pass
                elapsed = time.time() - started
                if (
                    close_after_events is not None
                    and len(out["event_times"]) >= close_after_events
                    and (close_after_seconds is None or elapsed >= close_after_seconds)
                ):
                    # Hanging up mid-stream is the cancellation: the server's
                    # disconnect watcher aborts the engine request within ~1s.
                    out["closed_early"] = True
                    break
    except requests.RequestException as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    finally:
        session.close()
    return out


def _in_thread(fn, box: dict, key: str):
    def run():
        try:
            box[key] = fn()
        except Exception as exc:  # noqa: BLE001 — surfaced through the box
            box[key] = {"error": f"{type(exc).__name__}: {exc}"[:300]}

    thread = threading.Thread(target=run, daemon=True, name=f"leg-{key}")
    thread.start()
    return thread


def run_request_legs(base_url: str, cfg: dict) -> Tuple[dict, List[str]]:
    """Drive the five text-only legs against a ready server."""
    problems: List[str] = []
    evidence: dict = {}
    timeout = cfg["request_timeout"]

    # Model discovery: use the id the server itself advertises.
    resp = requests.get(f"{base_url}/v1/models", timeout=60)
    if resp.status_code != 200:
        return evidence, [f"/v1/models returned {resp.status_code}: {resp.text[:200]}"]
    models = resp.json().get("data", [])
    if not models:
        return evidence, ["/v1/models returned no models"]
    model_id = models[0]["id"]
    evidence["models"] = {"status": resp.status_code, "served_model_id": model_id}

    def solo_pass(name: str) -> List[dict]:
        rows = []
        for prompt in cfg["solo_prompts"]:
            row = _post_completion(base_url, model_id, prompt, cfg["solo_tokens"], timeout)
            rows.append(row)
            if row.get("status") != 200:
                problems.append(f"{name}: HTTP {row.get('status')} for {row['prompt_head']!r}")
            elif row.get("completion_tokens") != cfg["solo_tokens"]:
                problems.append(
                    f"{name}: {row.get('completion_tokens')} completion tokens "
                    f"(expected {cfg['solo_tokens']}) for {row['prompt_head']!r}"
                )
            elif not row.get("text", "").strip():
                problems.append(f"{name}: empty text for {row['prompt_head']!r}")
        return rows

    evidence["solo_pass_1"] = solo_pass("solo_pass_1")

    # Interleave: two concurrent streams must genuinely share the scheduler.
    epoch = time.time()
    box: dict = {}
    threads = [
        _in_thread(
            lambda p=prompt: _stream_completion(
                base_url, model_id, p, cfg["stream_tokens"], timeout, epoch
            ),
            box,
            f"stream{i}",
        )
        for i, prompt in enumerate(cfg["stream_prompts"])
    ]
    for thread in threads:
        thread.join(timeout=timeout)
    streams = [box.get("stream0", {}), box.get("stream1", {})]
    inter = interleave_evidence(
        streams[0].get("event_times", []), streams[1].get("event_times", [])
    )
    evidence["interleave"] = {"streams": streams, **inter}
    for i, stream in enumerate(streams):
        if stream.get("status") != 200:
            problems.append(f"interleave: stream {i} HTTP {stream.get('status')}")
        elif stream.get("finish_reason") != "length":
            problems.append(
                f"interleave: stream {i} finish_reason={stream.get('finish_reason')!r} "
                "(expected 'length' at the full budget)"
            )
    if not inter["interleaved"]:
        problems.append(
            "interleave: the two concurrent streams did not interleave "
            f"(evidence: {inter}) — requests were served serially"
        )

    # Cancellation: 3 concurrent requests, the middle one hangs up mid-decode.
    epoch = time.time()
    box = {}
    survivor_threads = [
        _in_thread(
            lambda p=cfg["cancel_prompts"][i]: _post_completion(
                base_url, model_id, p, cfg["cancel_tokens"], timeout
            ),
            box,
            f"survivor{n}",
        )
        for n, i in enumerate((0, 2))
    ]
    target_thread = _in_thread(
        lambda: _stream_completion(
            base_url,
            model_id,
            cfg["cancel_prompts"][1],
            cfg["cancel_tokens"],
            timeout,
            epoch,
            close_after_events=cfg["cancel_min_events"],
            close_after_seconds=cfg["cancel_after_seconds"],
        ),
        box,
        "target",
    )
    for thread in survivor_threads + [target_thread]:
        thread.join(timeout=timeout)
    target = box.get("target", {})
    survivors = [box.get("survivor0", {}), box.get("survivor1", {})]
    evidence["cancel"] = {
        "budget": cfg["cancel_tokens"],
        "target": target,
        "survivors": survivors,
    }
    problems.extend(
        cancel_leg_problems(evidence["cancel"], cfg["cancel_tokens"], cfg["cancel_min_events"])
    )

    # Solo repeat AFTER concurrency+cancellation: slot reuse must be clean.
    evidence["solo_pass_2"] = solo_pass("solo_pass_2")
    problems.extend(solo_determinism_problems(evidence["solo_pass_1"], evidence["solo_pass_2"]))

    # One chat request through the checkpoint's own chat template.
    started = time.time()
    chat_row: dict = {"status": None}
    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": cfg["chat_content"]}],
                "max_tokens": cfg["chat_tokens"],
                "temperature": 0.0,
                "ignore_eos": True,
            },
            timeout=timeout,
        )
        chat_row["status"] = resp.status_code
        if resp.status_code == 200:
            data = resp.json()
            message = data["choices"][0].get("message", {})
            text = (message.get("content") or "") + (message.get("reasoning_content") or "")
            chat_row["text_head"] = text[:200]
            chat_row["text_nonempty"] = bool(text.strip())
            chat_row["completion_tokens"] = data.get("usage", {}).get("completion_tokens")
        else:
            chat_row["body"] = resp.text[:300]
    except requests.RequestException as exc:
        chat_row["error"] = f"{type(exc).__name__}: {exc}"[:300]
    chat_row["seconds"] = round(time.time() - started, 1)
    evidence["chat"] = chat_row
    if chat_row.get("status") != 200:
        problems.append(f"chat: HTTP {chat_row.get('status')}")
    elif not chat_row.get("text_nonempty"):
        problems.append("chat: empty completion text")

    completed = (
        evidence["solo_pass_1"]
        + evidence["solo_pass_2"]
        + survivors
        + ([chat_row] if chat_row.get("status") == 200 else [])
    )
    total_tokens = sum(r.get("completion_tokens") or 0 for r in completed)
    total_tokens += cfg["stream_tokens"] * sum(
        1 for s in streams if s.get("finish_reason") == "length"
    )
    evidence["totals"] = {
        "requests_submitted": len(cfg["solo_prompts"]) * 2 + len(streams) + 3 + 1,
        "completion_tokens_total": total_tokens,
        "repeated_replay_reasoning": (
            "capture happens once during engine construction (before /health "
            "turns 200); every decode-only step of every request after "
            "readiness is a graph replay because the captured ladder covers "
            "all decode batch sizes 1..4 and zero fallback warnings exist"
        ),
    }
    return evidence, problems


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return sock.getsockname()[1]


def probe_child_package(env: dict) -> Tuple[Optional[dict], Optional[str]]:
    """Resolve tensorrt_llm exactly as the serve subprocess will."""
    code = (
        "import json, tensorrt_llm\n"
        "from tensorrt_llm.llmapi import CudaGraphConfig\n"
        "print('PROBE:' + json.dumps({\n"
        "    'version': tensorrt_llm.__version__,\n"
        "    'file': tensorrt_llm.__file__,\n"
        "    'default_cuda_graph_batch_sizes': list(CudaGraphConfig().batch_sizes or []),\n"
        "}))\n"
    )
    try:
        out = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.SubprocessError as exc:
        return None, f"package probe failed: {exc}"
    for line in out.stdout.splitlines():
        if line.startswith("PROBE:"):
            return json.loads(line[len("PROBE:") :]), None
    return None, f"package probe failed: rc={out.returncode} stderr={out.stderr[-400:]}"


def build_serve_command(args, port: int, yaml_path: str) -> List[str]:
    exe = shutil.which("trtllm-serve")
    head = [exe] if exe else [sys.executable, "-m", "tensorrt_llm.commands.serve"]
    cmd = head + [
        args.model,
        "--backend",
        "pytorch",
        "--host",
        HOST,
        "--port",
        str(port),
        "--tensor_parallel_size",
        str(args.tp),
        "--pipeline_parallel_size",
        str(args.pp),
        "--max_batch_size",
        "4",
        "--max_num_tokens",
        "4096",
        "--max_seq_len",
        str(args.max_seq_len),
        "--log_level",
        "info",
    ]
    # Stage-5 TP4/EP4: only pass --moe_expert_parallel_size when an EP override
    # is requested. With just --tensor_parallel_size=4 the Mapping resolves the
    # MoE to moe_tp_size=4/moe_ep_size=1 (TP4); adding --moe_expert_parallel_size=4
    # over the same four ranks resolves it to moe_tp_size=1/moe_ep_size=4
    # (TP4/EP4). Absent (None) keeps the pre-Stage-5 single-geometry behaviour.
    ep = getattr(args, "ep", None)
    if ep is not None:
        cmd += ["--moe_expert_parallel_size", str(ep)]
    cmd += ["--extra_llm_api_options", yaml_path]
    return cmd


def write_extra_options_yaml(path: str, enabled: bool) -> str:
    """B: graphs OFF + overlap OFF. E: default CudaGraphConfig() + overlap ON."""
    text = (
        ("cuda_graph_config: {}\n" if enabled else "cuda_graph_config: null\n")
        + f"disable_overlap_scheduler: {str(not enabled).lower()}\n"
        + "kv_cache_config:\n"
        + "  enable_block_reuse: false\n"
        + "  max_tokens: 16384\n"
    )
    with open(path, "w") as fh:
        fh.write(text)
    return text


def wait_ready(proc: subprocess.Popen, base_url: str, timeout_s: float) -> dict:
    started = time.time()
    while time.time() - started < timeout_s:
        code = proc.poll()
        if code is not None:
            return {
                "ready": False,
                "server_exited": code,
                "seconds": round(time.time() - started, 1),
            }
        try:
            if requests.get(f"{base_url}/health", timeout=10).status_code == 200:
                return {"ready": True, "seconds": round(time.time() - started, 1)}
        except requests.RequestException:
            pass
        time.sleep(10)
    return {"ready": False, "timeout": True, "seconds": round(time.time() - started, 1)}


def shutdown_server(proc: subprocess.Popen, grace_s: float) -> dict:
    started = time.time()
    if proc.poll() is not None:
        return {
            "exited_before_shutdown": True,
            "returncode": proc.returncode,
            "seconds": 0.0,
        }
    result = {"signal": "SIGTERM", "forced_kill": False}
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        result["forced_kill"] = True
        proc.kill()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            pass
    result["returncode"] = proc.returncode
    result["seconds"] = round(time.time() - started, 1)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/dev/shm/GLM-5.3-Flash")
    parser.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    parser.add_argument("--pp", type=int, default=1, help="pipeline_parallel_size")
    parser.add_argument(
        "--ep",
        type=int,
        default=None,
        help="moe_expert_parallel_size; omit for TP-only MoE (moe_ep_size=1). "
        "For Stage-5 TP4/EP4 pass --tp 4 --ep 4 (moe_tp_size=1, moe_ep_size=4).",
    )
    parser.add_argument("--config", choices=["B", "E"], default="B")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--solo-tokens", type=int, default=32)
    parser.add_argument("--stream-tokens", type=int, default=128)
    parser.add_argument(
        "--cancel-tokens",
        type=int,
        default=256,
        help="decode budget for the cancellation leg; large enough that the "
        "mid-stream hangup lands mid-decode against --cancel-after-seconds",
    )
    parser.add_argument("--cancel-after-seconds", type=float, default=4.0)
    parser.add_argument("--cancel-min-events", type=int, default=4)
    parser.add_argument("--chat-tokens", type=int, default=48)
    parser.add_argument("--readiness-timeout", type=float, default=5400.0)
    parser.add_argument("--shutdown-grace", type=float, default=300.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.summary)), exist_ok=True)
    runlog = args.summary + ".runlog.txt"
    enabled = args.config == "E"
    started = time.time()

    # world_size = tp * pp is the number of MPI worker ranks; under pure TP every
    # rank runs every decode step and must capture the graph ladder, so the
    # per-rank ladder audit keys on world_size (not pp). MoE geometry: with an EP
    # override the four ranks split experts (moe_tp=world//ep, moe_ep=ep); without
    # it the MoE is tensor-parallel (moe_tp=tp, moe_ep=1).
    world_size = args.tp * args.pp
    if args.ep is not None:
        moe_tp_size, moe_ep_size = world_size // args.ep, args.ep
    else:
        moe_tp_size, moe_ep_size = args.tp, 1
    mapping_label = (
        f"TP{args.tp}/EP{args.ep}" if args.ep is not None else f"TP{args.tp}"
    )
    if args.pp > 1:
        mapping_label += f"/PP{args.pp}"

    summary: dict = {
        "config": {
            "model": args.model,
            "configuration": args.config,
            "mapping_label": mapping_label,
            "tensor_parallel_size": args.tp,
            "pipeline_parallel_size": args.pp,
            "moe_expert_parallel_size": moe_ep_size,
            "moe_tensor_parallel_size": moe_tp_size,
            "world_size": world_size,
            "max_batch_size": 4,
            "max_num_tokens": 4096,
            "max_seq_len": args.max_seq_len,
            "cuda_graph": enabled,
            "overlap_scheduler": enabled,
            "kv_cache": {"enable_block_reuse": False, "max_tokens": 16384},
            "attention_backend": "GlmKpoolSparseAttention (TRTLLM sparse MLA) + KDA",
            "kv_cache_manager": "Glm5NextCacheManager (KVCacheManagerV2)",
        },
        "ok": False,
        "problems": ["run did not complete"],
    }

    def finish(code: int) -> int:
        summary.setdefault("phases", {})["total_seconds"] = round(time.time() - started, 1)
        if "memory_samples" in summary:
            # Snapshot: the sampler thread may still append while we dump.
            summary["memory_samples"] = list(summary["memory_samples"])
        with open(args.summary, "w") as fh:
            json.dump(summary, fh, indent=2)
        with open(args.summary + ".exit.txt", "w") as fh:
            fh.write(f"{code}\n")
        print(f"[serve-smoke] wrote {args.summary} (exit {code})", flush=True)
        return code

    # Infrastructure gate BEFORE anything expensive.
    preflight, preflight_problems = disk_preflight(args.summary)
    summary["disk_preflight"] = preflight
    print(f"[serve-smoke] disk preflight: {json.dumps(preflight['measured'])}", flush=True)
    if preflight_problems:
        summary["failure_class"] = "infrastructure"
        summary["problems"] = preflight_problems
        return finish(1)

    env = os.environ.copy()
    env["TLLM_LOG_LEVEL"] = "INFO"
    env["PYTHONPATH"] = _REPO + os.pathsep + env.get("PYTHONPATH", "")

    probe, probe_error = probe_child_package(env)
    if probe_error:
        summary["failure_class"] = "environment"
        summary["problems"] = [probe_error]
        return finish(1)
    summary["package"] = {
        "tensorrt_llm_version": probe["version"],
        "tensorrt_llm_file": probe["file"],
    }
    if not probe["file"].startswith(_REPO):
        summary["failure_class"] = "environment"
        summary["problems"] = [
            f"stale tensorrt_llm package resolved by the serve environment: "
            f"{probe['file']} (expected the in-repo tree under {_REPO})"
        ]
        return finish(1)
    expected_sizes = (
        expected_graph_batch_sizes(
            probe["default_cuda_graph_batch_sizes"], engine_max_batch_size=4, max_num_tokens=4096
        )
        if enabled
        else []
    )

    yaml_path = os.path.join(
        os.path.dirname(os.path.abspath(args.summary)),
        f"serve_extra_options_{args.config.lower()}.yaml",
    )
    summary["config"]["extra_llm_api_options_yaml"] = write_extra_options_yaml(yaml_path, enabled)

    port = free_port()
    argv = build_serve_command(args, port, yaml_path)
    summary["config"]["serve_command"] = argv
    base_url = f"http://{HOST}:{port}"
    summary["config"]["base_url"] = base_url

    samples: list = []
    sampler_stop, sampler_thread = _start_memory_sampler(samples, started)
    summary["memory_samples"] = samples

    proc: Optional[subprocess.Popen] = None
    logfh = None
    try:
        print(f"[serve-smoke] launching: {' '.join(argv)}", flush=True)
        logfh = open(runlog, "ab")
        proc = subprocess.Popen(argv, stdout=logfh, stderr=subprocess.STDOUT, env=env)
        summary["server_pid"] = proc.pid

        readiness = wait_ready(proc, base_url, args.readiness_timeout)
        summary["readiness"] = readiness
        print(f"[serve-smoke] readiness: {readiness}", flush=True)
        if not readiness.get("ready"):
            failure_class, marks = classify_failure(runlog)
            summary["failure_class"] = failure_class
            summary["failure_marker_lines"] = marks
            summary["problems"] = [
                f"server never became ready ({readiness}); failure_class={failure_class}"
            ]
            return finish(1)

        leg_cfg = {
            "solo_prompts": [
                "The capital of France is",
                "2, 4, 6, 8,",
                "Water is composed of hydrogen and",
                "Count upward without stopping: "
                + " ".join(str(i) for i in range(1, 220))
                + " and",
            ],
            "solo_tokens": args.solo_tokens,
            "stream_prompts": [
                "Explain, step by step, how rain forms:",
                "Describe a lighthouse keeper's morning routine:",
            ],
            "stream_tokens": args.stream_tokens,
            "cancel_prompts": [
                "Explain, step by step, how rain forms:",
                "List the powers of two in order: 2, 4, 8,",
                "Describe a lighthouse keeper's morning routine:",
            ],
            "cancel_tokens": args.cancel_tokens,
            "cancel_after_seconds": args.cancel_after_seconds,
            "cancel_min_events": args.cancel_min_events,
            "chat_content": "Name three primary colors, briefly.",
            "chat_tokens": args.chat_tokens,
            "request_timeout": args.request_timeout,
        }
        legs, problems = run_request_legs(base_url, leg_cfg)
        summary["legs"] = legs

        # The server must still be alive after every leg (no crash), then be
        # shut down deliberately so the runlog is complete for the audits.
        shutdown = shutdown_server(proc, args.shutdown_grace)
        summary["shutdown"] = shutdown
        problems.extend(shutdown_problems(shutdown))
        print(f"[serve-smoke] shutdown: {shutdown}", flush=True)

        hard_path, hard_problems = check_cuda_graph_hard_path(runlog, enabled)
        summary["cuda_graph_hard_path"] = hard_path
        problems.extend(hard_problems)
        ladder, ladder_problems = audit_graph_ladder(
            runlog,
            enabled=enabled,
            expected_sizes=expected_sizes,
            engine_max_batch_size=4,
            # audit_graph_ladder's rank set is "every worker rank that must
            # capture" — world_size (tp*pp), not just pp. Under TP4 that is
            # ranks 0..3, each tagged [RANK k] by the MPI logger.
            pp_size=world_size,
        )
        summary["graph_ladder"] = ladder
        problems.extend(ladder_problems)
        provenance, provenance_problems = harvest_production_provenance(runlog)
        summary["production_provenance"] = provenance
        problems.extend(provenance_problems)
        abort_evidence, abort_problems = find_server_abort_lines(runlog)
        summary["server_abort_evidence"] = abort_evidence
        problems.extend(abort_problems)

        summary["problems"] = problems
        summary["ok"] = not problems
        print(f"[serve-smoke] ok={summary['ok']} problems={problems}", flush=True)
    except BaseException as exc:  # noqa: BLE001 — recorded, then exit code
        summary["ok"] = False
        summary["error"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[
            -4000:
        ]
        summary["problems"] = [f"exception: {type(exc).__name__}: {exc}"]
        print(f"[serve-smoke] FAILED: {type(exc).__name__}: {exc}", flush=True)
    finally:
        sampler_stop.set()
        sampler_thread.join(timeout=5)
        if proc is not None and proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pass
        if logfh is not None:
            logfh.close()

    return finish(0 if summary.get("ok") else 1)


if __name__ == "__main__":
    raise SystemExit(main())
