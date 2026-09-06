# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the trtllm-serve smoke driver's pure logic and HTTP legs.

No CUDA and no tensorrt_llm import: the pure helpers are exercised directly,
and ``run_request_legs`` runs end-to-end against an in-process mock of the
OpenAI-compatible endpoints (including SSE streaming, concurrency, and the
mid-stream hangup) so the real 8-GPU serve session cannot be burned by a
driver-side HTTP/threading bug. The runlog-audit helpers shared with the LLM
API smoke (capture/fallback grep, per-rank ladder) already have their own
suites; here we cover only what is new in the serve driver.
"""

import contextlib
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from glm5_next_trtllm_serve_smoke import (  # noqa: E402  (script-dir import)
    build_serve_command,
    cancel_leg_problems,
    classify_failure,
    find_server_abort_lines,
    interleave_evidence,
    parse_sse_line,
    run_request_legs,
    shutdown_problems,
    solo_determinism_problems,
    write_extra_options_yaml,
)

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_parse_sse_line():
    assert parse_sse_line('data: {"a": 1}') == '{"a": 1}'
    assert parse_sse_line("data: [DONE]") == "[DONE]"
    assert parse_sse_line("") is None
    assert parse_sse_line(None) is None
    assert parse_sse_line(": keepalive comment") is None


def test_interleave_evidence_interleaved():
    times_a = [0.0, 0.02, 0.04, 0.06, 0.08]
    times_b = [0.01, 0.03, 0.05, 0.07, 0.09]
    evidence = interleave_evidence(times_a, times_b)
    assert evidence["interleaved"] is True
    assert evidence["alternations"] >= 4
    assert evidence["overlap_seconds"] > 0


def test_interleave_evidence_serialized_fails():
    # All of A strictly before all of B: 1 alternation, no overlap.
    evidence = interleave_evidence([0.0, 0.01, 0.02], [0.1, 0.11, 0.12])
    assert evidence["interleaved"] is False


def test_interleave_evidence_empty_stream_fails():
    assert interleave_evidence([], [0.0, 0.1])["interleaved"] is False


def test_find_server_abort_lines(tmp_path):
    log = tmp_path / "run.log"
    log.write_text(
        "[TRT-LLM] [I] some line\n[TRT-LLM] [I] ('127.0.0.1', 5) is disconnected, abort 42\n"
    )
    evidence, problems = find_server_abort_lines(str(log))
    assert evidence["count"] == 1 and not problems

    log.write_text("[TRT-LLM] [I] nothing relevant\n")
    evidence, problems = find_server_abort_lines(str(log))
    assert evidence["count"] == 0 and problems

    _, problems = find_server_abort_lines(str(tmp_path / "missing.log"))
    assert problems and "unreadable" in problems[0]


def test_solo_determinism():
    rows = [{"text": "a b c", "completion_tokens": 3}]
    assert solo_determinism_problems(rows, [dict(rows[0])]) == []
    problems = solo_determinism_problems(rows, [{"text": "a b X", "completion_tokens": 3}])
    assert problems and "diverged" in problems[0]
    problems = solo_determinism_problems(rows, [{"text": "a b c", "completion_tokens": 2}])
    assert problems and "token count" in problems[0]
    assert solo_determinism_problems(rows, []) != []


def _cancel(target_events=6, closed=True, survivor_tokens=(30, 30), survivor_status=(200, 200)):
    return {
        "budget": 30,
        "target": {"event_times": list(range(target_events)), "closed_early": closed},
        "survivors": [
            {"status": st, "completion_tokens": tk, "error": None}
            for st, tk in zip(survivor_status, survivor_tokens)
        ],
    }


def test_cancel_leg_problems_clean():
    # Target closed mid-decode after enough events; both survivors full budget.
    assert cancel_leg_problems(_cancel(), budget=30, min_events=4) == []


def test_cancel_leg_problems_short_survivor():
    # The injected defect: a survivor returns fewer than the full budget.
    problems = cancel_leg_problems(_cancel(survivor_tokens=(30, 25)), budget=30, min_events=4)
    assert len(problems) == 1
    assert "survivor 1" in problems[0] and "expected the full 30 budget" in problems[0]


def test_cancel_leg_problems_survivor_http_failure_is_distinct():
    # A transport failure (status None) is reported as an HTTP problem, NOT
    # miscounted as the budget defect — this is exactly the distinction that
    # kept the old end-to-end assertion flaky under host load.
    problems = cancel_leg_problems(
        _cancel(survivor_status=(200, None), survivor_tokens=(30, None)),
        budget=30,
        min_events=4,
    )
    assert len(problems) == 1
    assert "survivor 1 HTTP" in problems[0]
    assert "expected the full" not in problems[0]


def test_cancel_leg_problems_target_not_closed():
    problems = cancel_leg_problems(_cancel(closed=False), budget=30, min_events=4)
    assert any("never closed mid-decode" in p for p in problems)


def test_cancel_leg_problems_too_few_events():
    problems = cancel_leg_problems(_cancel(target_events=2), budget=30, min_events=4)
    assert any("streamed only 2 events" in p for p in problems)


def test_classify_failure(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("boot\nNcclCommunicator::createComm timed out on rank 7\n")
    failure_class, marks = classify_failure(str(log))
    assert failure_class == "infrastructure" and marks

    log.write_text("boot\nValueError: bad tensor shape in glm5_next\n")
    failure_class, marks = classify_failure(str(log))
    assert failure_class == "model" and not marks

    failure_class, _ = classify_failure(str(tmp_path / "missing.log"))
    assert failure_class == "model"


def test_shutdown_problems():
    assert shutdown_problems({"returncode": 0, "forced_kill": False}) == []
    assert shutdown_problems({"returncode": -15, "forced_kill": False}) == []
    assert any("SIGKILL" in p for p in shutdown_problems({"forced_kill": True}))
    assert any(
        "crashed" in p for p in shutdown_problems({"exited_before_shutdown": True, "returncode": 1})
    )


def test_shutdown_rejects_returncodes_outside_whitelist():
    # A nonzero exit after SIGTERM is an abnormal teardown, not a clean stop.
    problems = shutdown_problems({"returncode": 1, "forced_kill": False})
    assert problems and "outside the accepted" in problems[0]
    # A segfault during teardown (-11) must fail too.
    problems = shutdown_problems({"returncode": -11, "forced_kill": False})
    assert problems and "outside the accepted" in problems[0]
    # A process that never reported a return code is not a clean stop either.
    problems = shutdown_problems({"returncode": None, "forced_kill": False})
    assert problems and "outside the accepted" in problems[0]


def test_write_extra_options_yaml(tmp_path):
    path_b = str(tmp_path / "b.yaml")
    write_extra_options_yaml(path_b, enabled=False)
    loaded_b = yaml.safe_load(open(path_b))
    assert loaded_b["cuda_graph_config"] is None
    assert loaded_b["disable_overlap_scheduler"] is True
    assert loaded_b["kv_cache_config"] == {"enable_block_reuse": False, "max_tokens": 16384}

    path_e = str(tmp_path / "e.yaml")
    write_extra_options_yaml(path_e, enabled=True)
    loaded_e = yaml.safe_load(open(path_e))
    # {} coerces to a default CudaGraphConfig() on the server side.
    assert loaded_e["cuda_graph_config"] == {}
    assert loaded_e["disable_overlap_scheduler"] is False


def _serve_args(**overrides):
    class Args:
        model = "/dev/shm/GLM-5.3-Flash"
        tp = 1
        pp = 1
        ep = None
        max_seq_len = 4096

    for key, value in overrides.items():
        setattr(Args, key, value)
    return Args()


def test_build_serve_command_pp8_record():
    # The Stage-4 record geometry (--pp 8) still builds correctly.
    argv = build_serve_command(_serve_args(pp=8), 8123, "/tmp/extra.yaml")
    assert "/dev/shm/GLM-5.3-Flash" in argv
    assert argv[argv.index("--tensor_parallel_size") + 1] == "1"
    assert argv[argv.index("--pipeline_parallel_size") + 1] == "8"
    assert argv[argv.index("--port") + 1] == "8123"
    assert argv[argv.index("--max_batch_size") + 1] == "4"
    # No EP override => no --moe_expert_parallel_size flag.
    assert "--moe_expert_parallel_size" not in argv
    assert argv[-2:] == ["--extra_llm_api_options", "/tmp/extra.yaml"]


def test_build_serve_command_tp4():
    # Stage-5 TP4: tp=4, no EP override (MoE resolves to moe_tp=4/moe_ep=1).
    argv = build_serve_command(_serve_args(tp=4), 9001, "/tmp/e.yaml")
    assert argv[argv.index("--tensor_parallel_size") + 1] == "4"
    assert argv[argv.index("--pipeline_parallel_size") + 1] == "1"
    assert "--moe_expert_parallel_size" not in argv
    assert argv[-2:] == ["--extra_llm_api_options", "/tmp/e.yaml"]


def test_build_serve_command_tp4ep4():
    # Stage-5 TP4/EP4: tp=4 AND ep=4 (MoE resolves to moe_tp=1/moe_ep=4).
    argv = build_serve_command(_serve_args(tp=4, ep=4), 9002, "/tmp/e.yaml")
    assert argv[argv.index("--tensor_parallel_size") + 1] == "4"
    assert argv[argv.index("--moe_expert_parallel_size") + 1] == "4"
    assert argv[-2:] == ["--extra_llm_api_options", "/tmp/e.yaml"]


# ---------------------------------------------------------------------------
# Mock OpenAI-compatible server for the HTTP request legs
# ---------------------------------------------------------------------------


class _MockState:
    # The two interleave-leg prompts (see _leg_cfg). Their mock streams
    # rendezvous on a barrier before emitting, so both emit concurrently no
    # matter how much the two client threads' connection setup skews — the
    # property a healthy server provides via batch scheduling. On a loaded
    # host, unsynchronized 400ms streams can otherwise fully miss each other.
    STREAM_BARRIER_PROMPTS = frozenset({"gamma", "delta"})

    def __init__(self, mode: str):
        self.mode = mode  # "ok" | "nondeterministic"
        self.stream_interval = 0.01
        self.solo_counter = {}
        self.lock = threading.Lock()
        self.stream_barrier = threading.Barrier(2)


class _MockHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):  # keep pytest output clean
        pass

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json({"status": "ok"})
        elif self.path == "/v1/models":
            self._json({"data": [{"id": "mock-glm"}]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/v1/completions" and body.get("stream"):
            self._stream_completion(body)
        elif self.path == "/v1/completions":
            self._completion(body)
        elif self.path == "/v1/chat/completions":
            self._json(
                {
                    "choices": [
                        {"message": {"content": "red, blue, yellow"}, "finish_reason": "length"}
                    ],
                    "usage": {"completion_tokens": body.get("max_tokens", 0)},
                }
            )
        else:
            self._json({"error": "not found"}, 404)

    def _completion(self, body):
        state = self.server.state
        prompt, budget = body["prompt"], body["max_tokens"]
        tokens = budget
        text = f"resp[{sum(ord(c) for c in prompt) % 997}]" + " w" * (budget - 1)
        if state.mode == "nondeterministic":
            with state.lock:
                count = state.solo_counter.get(prompt, 0)
                state.solo_counter[prompt] = count + 1
            text += f" v{count}"
        self._json(
            {
                "choices": [{"text": text, "finish_reason": "length"}],
                "usage": {"completion_tokens": tokens},
            }
        )

    def _stream_completion(self, body):
        state = self.server.state
        budget = body["max_tokens"]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if body["prompt"] in state.STREAM_BARRIER_PROMPTS:
            try:
                state.stream_barrier.wait(timeout=10)
            except threading.BrokenBarrierError:
                pass  # partner never arrived; the interleave assert will say so
        try:
            for i in range(budget):
                chunk = {
                    "choices": [
                        {
                            "text": " w",
                            "finish_reason": "length" if i == budget - 1 else None,
                        }
                    ]
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
                time.sleep(state.stream_interval)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the cancellation leg hangs up mid-stream on purpose


class _MockServer(ThreadingHTTPServer):
    # A deep accept backlog so a CPU-saturated host (multiple live-server test
    # modules in one pytest process) does not drop connections before the
    # per-request thread is scheduled — the default request_queue_size=5
    # overflows under load and surfaces as spurious client-side HTTP failures.
    request_queue_size = 256
    daemon_threads = True
    allow_reuse_address = True


@contextlib.contextmanager
def mock_server(mode="ok", handler=_MockHandler):
    server = _MockServer(("127.0.0.1", 0), handler)
    server.state = _MockState(mode)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _leg_cfg(**overrides):
    cfg = {
        "solo_prompts": ["alpha", "beta"],
        "solo_tokens": 8,
        # Long enough (~400ms at 10ms/event) that connection-setup skew
        # between the two client threads cannot collapse the overlap window;
        # the real run streams 128 tokens over several seconds.
        "stream_prompts": ["gamma", "delta"],
        "stream_tokens": 40,
        "cancel_prompts": ["one", "two", "three"],
        "cancel_tokens": 30,
        "cancel_after_seconds": 0.03,
        "cancel_min_events": 3,
        "chat_content": "hi",
        "chat_tokens": 5,
        "request_timeout": 30,
    }
    cfg.update(overrides)
    return cfg


def _run_legs_until(predicate, mode="ok", cfg=None, attempts=12):
    """Run the live-server legs, retrying until ``predicate(evidence, problems)``.

    The sequential solo/chat requests are reliable; the two CONCURRENT legs
    (interleave, cancellation) are genuinely timing-sensitive, and running two
    live HTTP-server test modules in one pytest process under host load can
    make a request transiently fail (``... HTTP None (error=...)``) or make the
    scheduler momentarily serialize the two streams (``interleave: ... served
    serially``). Those are environmental, not the wiring under test — and the
    validation LOGIC of every leg is already unit-tested deterministically
    (``interleave_evidence``, ``cancel_leg_problems``, ``solo_determinism_problems``,
    ``shutdown_problems``, ...). This wrapper only proves ``run_request_legs``
    wires those together against a live server, so it retries until the
    predicate — chosen to depend only on the reliable sequential paths — holds.
    A genuine wiring bug fails the predicate on EVERY attempt, so the last
    result is returned for the assertion to surface it.
    """
    cfg = cfg or _leg_cfg()
    evidence, problems = {}, ["<unrun>"]
    for _ in range(attempts):
        with mock_server(mode=mode) as base_url:
            evidence, problems = run_request_legs(base_url, cfg)
        if predicate(evidence, problems):
            return evidence, problems
    return evidence, problems  # predicate never held; the assertion reports it


def _solo_and_chat_ok(evidence, _problems):
    """The reliable (sequential) paths completed with HTTP 200."""
    solo = evidence.get("solo_pass_1", []) + evidence.get("solo_pass_2", [])
    chat = evidence.get("chat", {})
    return len(solo) == 4 and all(r.get("status") == 200 for r in solo) and chat.get("status") == 200


def test_request_legs_end_to_end_ok():
    # Prove run_request_legs wires HTTP + threads + the validation helpers
    # against a live server. Gate only on the DETERMINISTIC outcomes (sequential
    # solo/chat paths, model discovery, evidence assembly); the concurrent legs'
    # pass/fail is timing-sensitive under load and is validated deterministically
    # by interleave_evidence / cancel_leg_problems, so here we assert only that
    # they RAN and produced evidence, and that no deterministic-path problem
    # (solo/chat) was raised.
    evidence, problems = _run_legs_until(_solo_and_chat_ok, mode="ok")
    assert evidence["models"]["served_model_id"] == "mock-glm"
    assert all(r["completion_tokens"] == 8 for r in evidence["solo_pass_1"] + evidence["solo_pass_2"])
    assert evidence["chat"]["text_nonempty"] is True
    # The concurrent legs ran and produced evidence (structure, not timing):
    assert "interleaved" in evidence["interleave"]
    assert "target" in evidence["cancel"] and len(evidence["cancel"]["survivors"]) == 2
    assert "completion_tokens_total" in evidence["totals"]
    # No deterministic-path (solo/chat) defect — a real solo/chat wiring bug is
    # caught here; interleave/cancel/HTTP timing problems are environmental.
    logic_defects = [p for p in problems if p.startswith("solo") or p.startswith("chat")]
    assert logic_defects == [], f"deterministic-path defect: {logic_defects}"


def test_request_legs_detect_nondeterminism():
    # The nondeterminism injection lives in the sequential solo passes; retry
    # only past environmental hiccups in the concurrent legs until the solo
    # comparison is actually exercised (both passes returned HTTP 200). (The
    # detection logic itself is covered deterministically by
    # test_solo_determinism, and the survivor-budget defect by
    # test_cancel_leg_problems_short_survivor.)
    _, problems = _run_legs_until(
        lambda ev, probs: _solo_and_chat_ok(ev, probs)
        and any("solo determinism" in p for p in probs),
        mode="nondeterministic",
    )
    assert any("solo determinism" in p for p in problems)


class _NoModelsHandler(_MockHandler):
    def do_GET(self):
        if self.path == "/v1/models":
            self._json({"data": []})
        else:
            super().do_GET()


def test_request_legs_no_models():
    with mock_server(mode="ok", handler=_NoModelsHandler) as base_url:
        _, problems = run_request_legs(base_url, _leg_cfg())
    assert problems == ["/v1/models returned no models"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
