# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the glm5_next driver preflight/graph-ladder helpers.

Pure-Python (no CUDA, no checkpoint): they pin the *analysis* contracts the
LLM API drivers rely on — the disk-space gate, the engine's filtered-ladder
math, and the per-PP-rank silent-eager audit — against synthetic runlogs and
statvfs values. The runtime that produces those runlogs is exercised by the
sanctioned smoke/replay sessions; here we prove the helpers turn a given log
into the right pass/fail verdict.

The load-bearing case is :func:`test_missing_pp_rank_fails`: at PP>1 a single
rank that skipped CUDA-graph capture decodes eagerly for the whole pipeline
while the aggregate log still shows capture lines, so the audit must reject a
runlog missing any expected rank — not merely check that some line has the
right count.
"""

from __future__ import annotations

import os

import pytest
from glm5_next_driver_preflight import (
    audit_graph_ladder,
    disk_preflight,
    expected_graph_batch_sizes,
)

# One real capture line, verbatim shape from a PP=8 E runlog:
# "[09/04/2026-00:08:52] [TRT-LLM] [I] [_torch][RANK 0] Running CUDA graph
#  capture for 4 batch sizes."
_LINE = "[09/04/2026-00:08:52] [TRT-LLM] [I] [_torch][RANK {rank}] Running CUDA graph capture for {n} batch sizes."


def _write_runlog(tmp_path, *, ranks, n, two_pass=True, extra_lines=()):
    """Synthesize a tee'd runlog with one capture line per rank (per pass)."""
    lines = list(extra_lines)
    passes = 2 if two_pass else 1  # the engine logs a warmup pass + a capture pass
    for _ in range(passes):
        for rank in ranks:
            lines.append(_LINE.format(rank=rank, n=n))
    path = os.path.join(str(tmp_path), "run.log")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------
# expected_graph_batch_sizes — the engine's filtered ladder, recomputed.
# --------------------------------------------------------------------------


def test_expected_ladder_matches_engine_filter():
    from tensorrt_llm.llmapi import CudaGraphConfig

    default = CudaGraphConfig().batch_sizes
    # smoke uses max_batch_size=4, replay max_batch_size=2; token budget is ample.
    assert expected_graph_batch_sizes(default, 4, 4096) == [1, 2, 3, 4]
    assert expected_graph_batch_sizes(default, 2, 4096) == [1, 2]
    # The token budget caps the ladder when it bites first.
    assert expected_graph_batch_sizes(default, 8, 5) == [1, 2, 3, 4, 5]
    # A custom (sparser) config is filtered but not densified.
    assert expected_graph_batch_sizes([1, 4, 8], 4, 4096) == [1, 4]


# --------------------------------------------------------------------------
# audit_graph_ladder — per-rank silent-eager audit (the REJECT fix).
# --------------------------------------------------------------------------


def test_all_pp_ranks_present_passes(tmp_path):
    runlog = _write_runlog(tmp_path, ranks=range(8), n=4)
    evidence, problems = audit_graph_ladder(
        runlog, enabled=True, expected_sizes=[1, 2, 3, 4], engine_max_batch_size=4, pp_size=8
    )
    assert problems == [], problems
    assert evidence["all_pp_ranks_covered"] is True
    assert evidence["ranks_covering_expected_ladder"] == list(range(8))
    assert evidence["missing_ranks"] == []
    assert evidence["covers_every_decode_batch_size"] is True


def test_missing_pp_rank_fails(tmp_path):
    # Rank 5 never logged a capture — it would decode eagerly for the whole
    # pipeline. The aggregate log still has 14 capture lines from the others.
    ranks = [r for r in range(8) if r != 5]
    runlog = _write_runlog(tmp_path, ranks=ranks, n=4)
    evidence, problems = audit_graph_ladder(
        runlog, enabled=True, expected_sizes=[1, 2, 3, 4], engine_max_batch_size=4, pp_size=8
    )
    assert evidence["missing_ranks"] == [5]
    assert evidence["all_pp_ranks_covered"] is False
    assert any("PP ranks [5]" in p for p in problems), problems
    # The old count-only check would have passed this (all counts == 4).
    assert evidence["capture_line_total"] == 14


def test_only_rank0_present_fails_at_pp8(tmp_path):
    # The exact hole the reviewer named: a log with only rank 0's line.
    runlog = _write_runlog(tmp_path, ranks=[0], n=4)
    evidence, problems = audit_graph_ladder(
        runlog, enabled=True, expected_sizes=[1, 2, 3, 4], engine_max_batch_size=4, pp_size=8
    )
    assert evidence["missing_ranks"] == [1, 2, 3, 4, 5, 6, 7]
    assert any("did not log a capture" in p for p in problems), problems


def test_multiple_missing_ranks_reported(tmp_path):
    runlog = _write_runlog(tmp_path, ranks=[0, 1, 2, 3], n=4)
    evidence, problems = audit_graph_ladder(
        runlog, enabled=True, expected_sizes=[1, 2, 3, 4], engine_max_batch_size=4, pp_size=8
    )
    assert evidence["missing_ranks"] == [4, 5, 6, 7]
    assert any("PP ranks [4, 5, 6, 7]" in p for p in problems), problems


def test_wrong_count_on_one_rank_fails(tmp_path):
    # Rank 3 captured a shorter ladder (e.g. the token budget bit only there);
    # that rank does not count as covering the expected ladder.
    good = [_LINE.format(rank=r, n=4) for r in range(8) if r != 3]
    bad = [_LINE.format(rank=3, n=2)]
    path = os.path.join(str(tmp_path), "run.log")
    with open(path, "w") as fh:
        fh.write("\n".join(good + bad) + "\n")
    evidence, problems = audit_graph_ladder(
        path, enabled=True, expected_sizes=[1, 2, 3, 4], engine_max_batch_size=4, pp_size=8
    )
    assert 2 in [c for counts in evidence["per_rank_capture_counts"].values() for c in counts]
    assert 3 in evidence["missing_ranks"]
    assert any("captured [2] batch sizes" in p for p in problems), problems


def test_baseline_with_captures_fails(tmp_path):
    runlog = _write_runlog(tmp_path, ranks=range(8), n=4)
    evidence, problems = audit_graph_ladder(
        runlog, enabled=False, expected_sizes=[1, 2, 3, 4], engine_max_batch_size=4, pp_size=8
    )
    assert any("must not capture" in p for p in problems), problems


def test_baseline_clean_passes(tmp_path):
    path = os.path.join(str(tmp_path), "run.log")
    with open(path, "w") as fh:
        fh.write("[TRT-LLM] [I] engine up, no capture in baseline B\n")
    evidence, problems = audit_graph_ladder(
        path, enabled=False, expected_sizes=[1, 2, 3, 4], engine_max_batch_size=4, pp_size=8
    )
    assert problems == [], problems
    assert evidence["capture_line_total"] == 0


def test_enabled_no_capture_lines_fails(tmp_path):
    path = os.path.join(str(tmp_path), "run.log")
    with open(path, "w") as fh:
        fh.write("[TRT-LLM] [I] engine up but capture never logged\n")
    evidence, problems = audit_graph_ladder(
        path, enabled=True, expected_sizes=[1, 2], engine_max_batch_size=2, pp_size=8
    )
    assert any("did not run" in p for p in problems) or any(
        "did not log a capture" in p for p in problems
    ), problems


def test_ladder_not_covering_all_sizes_fails(tmp_path):
    # A gapped ladder [1, 2, 4] at max_batch_size=4: size 3 has no graph and
    # would decode eagerly with no warning.
    lines = []
    for _ in range(2):
        for rank in range(8):
            lines.append(_LINE.format(rank=rank, n=3))
    path = os.path.join(str(tmp_path), "run.log")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    evidence, problems = audit_graph_ladder(
        path, enabled=True, expected_sizes=[1, 2, 4], engine_max_batch_size=4, pp_size=8
    )
    assert evidence["covers_every_decode_batch_size"] is False
    assert any("does not cover every" in p for p in problems), problems


def test_single_process_untagged_line_is_rank0(tmp_path):
    # pp=1 runs have no [RANK k] tag; an untagged capture line is rank 0.
    path = os.path.join(str(tmp_path), "run.log")
    with open(path, "w") as fh:
        fh.write("[TRT-LLM] [I] Running CUDA graph capture for 4 batch sizes.\n")
    evidence, problems = audit_graph_ladder(
        path, enabled=True, expected_sizes=[1, 2, 3, 4], engine_max_batch_size=4, pp_size=1
    )
    assert problems == [], problems
    assert evidence["ranks_seen"] == [0]
    assert evidence["all_pp_ranks_covered"] is True


def test_unreadable_runlog_reports_problem():
    evidence, problems = audit_graph_ladder(
        "/nonexistent/run.log",
        enabled=True,
        expected_sizes=[1, 2],
        engine_max_batch_size=2,
        pp_size=8,
    )
    assert any("unreadable" in p for p in problems), problems


# --------------------------------------------------------------------------
# disk_preflight — the fail-fast overlay/tmp gate.
# --------------------------------------------------------------------------


def test_disk_preflight_measures_and_passes(tmp_path):
    summary = os.path.join(str(tmp_path), "out.json")
    # Floors of 0 GiB can never fail: this proves the measurement path and the
    # recorded fields without depending on the host's free space.
    evidence, problems = disk_preflight(summary, min_overlay_gib=0, min_tmp_gib=0)
    assert problems == []
    assert evidence["ok"] is True
    for key in ("overlay_root", "tmpdir", "dev_shm", "summary_dir"):
        assert key in evidence["measured"]
        assert "free_gib" in evidence["measured"][key]
        assert "fs_type" in evidence["measured"][key]


def test_disk_preflight_gate_fires_on_high_floor(tmp_path):
    summary = os.path.join(str(tmp_path), "out.json")
    evidence, problems = disk_preflight(summary, min_overlay_gib=10**9, min_tmp_gib=0)
    assert evidence["ok"] is False
    assert any("overlay" in p and "Errno 28" in p for p in problems), problems


def test_disk_preflight_env_thresholds(tmp_path, monkeypatch):
    summary = os.path.join(str(tmp_path), "out.json")
    monkeypatch.setenv("GLM53_MIN_FREE_GIB_OVERLAY", "999999999")
    monkeypatch.setenv("GLM53_MIN_FREE_GIB_TMP", "0")
    evidence, problems = disk_preflight(summary)
    assert evidence["ok"] is False
    assert evidence["thresholds_gib"]["overlay_root"] == pytest.approx(999999999.0)
    assert any("overlay" in p for p in problems), problems
