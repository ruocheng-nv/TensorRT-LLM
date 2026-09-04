# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared pre-run infrastructure checks for the glm5_next LLM API drivers.

Two concerns, both Stage-4 hardening (graph-safe serving):

* :func:`disk_preflight` — measure and GATE free space on the filesystems a
  long engine run writes to, BEFORE the engine is constructed. The container
  overlay filling mid-run kills a multi-hour session with ``Errno 28`` at a
  point where all evidence is lost (observed: QA iteration 34, engine init
  died with ``OSError: [Errno 28] No space left on device`` at 100% overlay).
  A failed gate is an *infrastructure* condition, not model evidence — the
  caller records it with ``failure_class: infrastructure`` so a
  same-parameter retry after cleanup is classified as infra recovery.

  This complements (not replaces) the session-level convention of routing
  ``TMPDIR/TEMP/TMP`` to a SHORT scratch-backed path
  (``/home/scratch.ruochengj_sw/tmp`` in the sanctioned sessions): the path
  must stay short because TensorRT-LLM's IPC endpoints are ``ipc://``
  sockets under the temporary directory with the kernel's 107-byte
  ``sun_path`` limit — a reviewer run failed with ``ZMQError: ipc path ...
  is longer than 107 characters`` from a long TMPDIR. The gate measures
  whatever tmpdir is in effect and the overlay, and refuses to start a run
  that would die of Errno 28 hours in.

* :func:`expected_graph_batch_sizes` / :func:`audit_graph_ladder` — turn "no
  partial/silent eager fallback" from an assumption into a checked property.
  A decode-only batch whose size has no captured graph runs EAGER SILENTLY
  (``CUDAGraphRunner.maybe_get_cuda_graph`` returns no graph for a size not
  in ``supported_batch_sizes`` — no warning line), so counting fallback
  *warnings* alone cannot reject silent eager. The audit recomputes the
  engine's filtered ladder (the ``_filter_cuda_graph_batch_sizes`` rule) from
  the driver's own config, parses the ``[RANK k]`` tag on every "Running CUDA
  graph capture for N batch sizes." line, and requires that **every** PP rank
  ``0..pp_size-1`` logged the capture with ``N == len(expected_sizes)`` and
  that the ladder covers EVERY decode-only batch size the engine can schedule
  (``1..max_batch_size``). The per-rank requirement is the load-bearing part
  at PP>1: one rank that silently skipped capture decodes eagerly for the
  whole pipeline while the aggregate log still shows capture lines, so a
  count-only check (or "some line has the right count") would pass a partial
  capture. With the full rank set covered plus zero fallback warnings, the
  silent-eager hole is closed for generation steps; context/mixed-phase steps
  are eager by design in this runner and are disclosed as such.
"""

from __future__ import annotations

import os
import re
import tempfile
from typing import Dict, List, Optional, Tuple

# Matches model_engine._capture_generation_cuda_graphs's INFO line. The count
# is the length of the engine's *filtered* ladder, logged once per pass per
# worker rank.
_CAPTURE_COUNT_RE = re.compile(r"Running CUDA graph capture for (\d+) batch sizes")
# The MPI worker-rank tag the logger prepends under PP/TP, e.g.
# "[TRT-LLM] [I] [_torch][RANK 3] Running CUDA graph capture for 4 batch
# sizes." A single-process (pp=tp=1) run has no such tag; those lines are
# attributed to rank 0.
_RANK_RE = re.compile(r"\[RANK (\d+)\]")


def _fs_type(path: str) -> Optional[str]:
    """Filesystem type name via /proc/mounts longest-prefix match."""
    best, best_type = "", None
    try:
        real = os.path.realpath(path)
        with open("/proc/mounts") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mount, fstype = parts[1], parts[2]
                if (real == mount or real.startswith(mount.rstrip("/") + "/")) and len(mount) > len(
                    best
                ):
                    best, best_type = mount, fstype
    except OSError:
        return None
    return best_type


def _free_gib(path: str) -> Optional[float]:
    try:
        st = os.statvfs(path)
        return round(st.f_bavail * st.f_frsize / 1024**3, 1)
    except OSError:
        return None


def disk_preflight(
    summary_path: str,
    *,
    min_overlay_gib: Optional[float] = None,
    min_tmp_gib: Optional[float] = None,
) -> Tuple[dict, List[str]]:
    """Measure the run's writable filesystems and gate on free headroom.

    Returns ``(evidence, problems)``. A non-empty ``problems`` list means the
    caller must NOT construct the engine: record the evidence with
    ``failure_class: infrastructure`` and exit nonzero. Thresholds come from
    the arguments or the ``GLM53_MIN_FREE_GIB_OVERLAY`` /
    ``GLM53_MIN_FREE_GIB_TMP`` environment variables (defaults 15 / 5 GiB —
    the observed overlay incident needed ~3.5 GiB freed, so 15 leaves real
    margin for a multi-hour run's caches and logs).
    """
    if min_overlay_gib is None:
        min_overlay_gib = float(os.environ.get("GLM53_MIN_FREE_GIB_OVERLAY", "15"))
    if min_tmp_gib is None:
        min_tmp_gib = float(os.environ.get("GLM53_MIN_FREE_GIB_TMP", "5"))

    tmpdir = tempfile.gettempdir()
    summary_dir = os.path.dirname(os.path.abspath(summary_path)) or "."
    mounts: Dict[str, str] = {
        "overlay_root": "/",
        "tmpdir": tmpdir,
        "dev_shm": "/dev/shm",
        "summary_dir": summary_dir,
    }
    measured = {
        name: {"path": path, "free_gib": _free_gib(path), "fs_type": _fs_type(path)}
        for name, path in mounts.items()
    }
    evidence = {
        "measured": measured,
        "thresholds_gib": {"overlay_root": min_overlay_gib, "tmpdir": min_tmp_gib},
        "effective_tmpdir": tmpdir,
        "tmpdir_note": (
            "sessions route TMPDIR/TEMP/TMP to a SHORT scratch-backed path "
            "(ipc:// sun_path is limited to 107 bytes — a long TMPDIR "
            "already produced ZMQError); this gate measures whatever tmpdir "
            "is in effect plus the overlay and fails fast instead of dying "
            "of Errno 28 mid-run"
        ),
    }

    problems: List[str] = []
    overlay_free = measured["overlay_root"]["free_gib"]
    if overlay_free is not None and overlay_free < min_overlay_gib:
        problems.append(
            f"disk_preflight: overlay '/' has {overlay_free} GiB free "
            f"(< {min_overlay_gib} GiB floor) — a long run risks Errno 28 "
            "mid-session; free space before retrying (infrastructure, not "
            "model evidence)"
        )
    tmp_free = measured["tmpdir"]["free_gib"]
    if tmp_free is not None and tmp_free < min_tmp_gib:
        problems.append(
            f"disk_preflight: tmpdir '{tmpdir}' has {tmp_free} GiB free "
            f"(< {min_tmp_gib} GiB floor) — ipc sockets and temp files "
            "would fail mid-run (infrastructure, not model evidence)"
        )
    evidence["ok"] = not problems
    return evidence, problems


def expected_graph_batch_sizes(
    config_batch_sizes: List[int],
    engine_max_batch_size: int,
    max_num_tokens: int,
    tokens_per_request: int = 1,
) -> List[int]:
    """The engine's filtered capture ladder for a non-padding, non-spec run.

    Mirrors ``model_engine._filter_cuda_graph_batch_sizes`` for the
    configurations these drivers use (``enable_padding=False``, no
    speculative decoding, so ``tokens_per_request == 1``).
    """
    cap = min(engine_max_batch_size, max_num_tokens // tokens_per_request)
    return [size for size in sorted(config_batch_sizes) if size <= cap]


def audit_graph_ladder(
    runlog_path: str,
    *,
    enabled: bool,
    expected_sizes: List[int],
    engine_max_batch_size: int,
    pp_size: int,
) -> Tuple[dict, List[str]]:
    """Assert EVERY PP rank captured a ladder covering all decode-only sizes.

    Reads the worker ranks' "[RANK k] ... Running CUDA graph capture for N
    batch sizes." lines back from the tee'd runlog and requires, for config
    E, that **every** rank in ``0..pp_size-1`` logged the capture with
    ``N == len(expected_sizes)`` and that the ladder covers every decode-only
    size ``1..max_batch_size``. Combined with the caller's capture/no-fallback
    grep this rejects the SILENT eager path — a decode-only batch whose size
    has no captured graph replays nothing and runs eager with no warning.

    Per-rank completeness matters at PP>1: a single rank that failed to
    capture (OOM, a shape the filter dropped on that rank, a crash after the
    others logged) would decode eagerly for the whole model while the
    aggregate log still shows capture lines. Counting lines — or checking
    only that *some* line has the right count — cannot see that; requiring
    the full rank set can. A single-process run (``pp_size==1``) has no
    ``[RANK k]`` tag, so an untagged capture line is attributed to rank 0.
    """
    # (rank, count) for every capture line; rank defaults to 0 when untagged.
    entries: List[Tuple[int, int]] = []
    try:
        with open(runlog_path, errors="replace") as fh:
            for line in fh:
                match = _CAPTURE_COUNT_RE.search(line)
                if not match:
                    continue
                rank_match = _RANK_RE.search(line)
                rank = int(rank_match.group(1)) if rank_match else 0
                entries.append((rank, int(match.group(1))))
    except OSError as exc:
        return (
            {"runlog": runlog_path, "error": str(exc)},
            [f"graph_ladder: runlog unreadable: {exc}"],
        )

    expected_count = len(expected_sizes)
    covers_all = expected_sizes == list(range(1, engine_max_batch_size + 1))
    ranks_seen = sorted({rank for rank, _ in entries})
    # A rank "covered" the ladder only if it logged the capture with the
    # exact expected count — a wrong count on a rank does not count as cover.
    ranks_covered = sorted({rank for rank, count in entries if count == expected_count})
    expected_ranks = list(range(pp_size))
    missing_ranks = sorted(set(expected_ranks) - set(ranks_covered))
    per_rank_counts: Dict[int, List[int]] = {}
    for rank, count in entries:
        per_rank_counts.setdefault(rank, []).append(count)

    evidence = {
        "expected_sizes": expected_sizes,
        "engine_max_batch_size": engine_max_batch_size,
        "pp_size": pp_size,
        "expected_ranks": expected_ranks,
        "ranks_seen": ranks_seen,
        "ranks_covering_expected_ladder": ranks_covered,
        "missing_ranks": missing_ranks,
        "per_rank_capture_counts": {str(k): v for k, v in sorted(per_rank_counts.items())},
        "capture_line_total": len(entries),
        "covers_every_decode_batch_size": covers_all,
        "all_pp_ranks_covered": not missing_ranks,
        "eager_by_design": "context/mixed-phase steps (prefill is never captured)",
    }
    problems: List[str] = []
    if not enabled:
        if entries:
            problems.append(
                f"graph_ladder: baseline config B logged {len(entries)} "
                f"capture line(s) on ranks {ranks_seen} — B must not capture"
            )
        return evidence, problems

    if not entries:
        problems.append(
            "graph_ladder: no 'Running CUDA graph capture for N batch sizes' "
            "line in the run log — the warmup capture pass did not run"
        )
    bad_counts = sorted({count for _, count in entries if count != expected_count})
    if bad_counts:
        problems.append(
            f"graph_ladder: worker(s) captured {bad_counts} batch sizes, expected "
            f"{expected_count} ({expected_sizes}) — the engine's ladder does not "
            "match the driver's config"
        )
    if missing_ranks:
        problems.append(
            f"graph_ladder: PP ranks {missing_ranks} did not log a capture with "
            f"the expected {expected_count}-size ladder (ranks seen: {ranks_seen}); "
            "a rank that skipped capture decodes EAGERLY for the whole model with "
            "no fallback warning"
        )
    if not covers_all:
        problems.append(
            f"graph_ladder: ladder {expected_sizes} does not cover every "
            f"decode-only batch size in 1..{engine_max_batch_size} — those "
            "sizes would run eager SILENTLY (no warning line exists for this)"
        )
    return evidence, problems
