# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the LLM-API logit-replay driver's cross-geometry pure logic.

No CUDA and no tensorrt_llm import: covers the geometry resolution shared with
the serve driver (``resolve_moe_parallel``), the CLI surface (``--ep``,
``--cross-geometry-*``), and the cross-geometry comparator on synthetic
fixtures — so the real 4-GPU replay session cannot be burned by a driver-side
comparison bug and cannot mask an acceptance failure as INFO.

Stage-6 C1/C3 boundary matrix (plan.md Decision G): the
``qualified-tp4-pp4`` gate mode must accept exact identity and ONLY the two
sealed <=0.5-separation near-ties (canary0 step 35 {11,53059}, short9 step 8
{13,11}), while rejecting a third/new fork, a changed candidate pair, a
one-sided or >0.5 separation, a candidate logit outside [16,32), and
absent/non-finite evidence; the ``diagnostic`` mode must fully report
TP4/EP4-vs-PP4 forks without gating them; the default ``exact`` mode keeps
the Stage-5 any-fork-fails contract; and structural mismatches are problems
in every mode.
"""

import json
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from glm5_next_llm_api_logit_replay import (  # noqa: E402  (script-dir import)
    DECISION_G_QUALIFIED_FORKS,
    build_parser,
    classify_fork_decision_g,
    compare_cross_geometry,
    resolve_moe_parallel,
)

# ---------------------------------------------------------------------------
# resolve_moe_parallel — must mirror the serve driver / Mapping semantics
# ---------------------------------------------------------------------------


def test_resolve_moe_parallel_tp4_defaults_to_moe_tp():
    moe_tp, moe_ep, label, kwargs = resolve_moe_parallel(4, 1, None)
    assert (moe_tp, moe_ep) == (4, 1)
    assert label == "TP4"
    # TP-only must NOT pass moe_expert_parallel_size, keeping the LLM() call
    # byte-identical to the pre-EP driver behavior.
    assert kwargs == {}


def test_resolve_moe_parallel_tp4_ep4_splits_experts():
    moe_tp, moe_ep, label, kwargs = resolve_moe_parallel(4, 1, 4)
    assert (moe_tp, moe_ep) == (1, 4)
    assert label == "TP4/EP4"
    assert kwargs == {"moe_expert_parallel_size": 4}


def test_resolve_moe_parallel_pp_suffix():
    moe_tp, moe_ep, label, kwargs = resolve_moe_parallel(1, 4, None)
    assert (moe_tp, moe_ep) == (1, 1)
    assert label == "TP1/PP4"
    assert kwargs == {}


def test_resolve_moe_parallel_rejects_indivisible_world():
    with pytest.raises(ValueError, match="not divisible"):
        resolve_moe_parallel(4, 1, 3)


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_parser_defaults_and_c5_flags():
    parser = build_parser()
    args = parser.parse_args(["--summary", "s.json", "--metrics-out", "m.pt"])
    assert args.tp == 1 and args.ep is None
    assert args.cross_geometry_ref is None and args.cross_geometry_metrics is None

    args = parser.parse_args(
        [
            "--tp", "4", "--ep", "4", "--config", "E",
            "--summary", "s.json", "--metrics-out", "m.pt",
            "--cross-geometry-ref", "ref.json",
            "--cross-geometry-metrics", "ref.pt",
            "--cross-geometry-label", "TP4-vs-PP4-E",
        ]
    )  # fmt: skip
    assert (args.tp, args.ep, args.config) == (4, 4, "E")
    assert args.cross_geometry_ref == "ref.json"
    assert args.cross_geometry_metrics == "ref.pt"
    assert args.cross_geometry_label == "TP4-vs-PP4-E"
    # Fork gating defaults to the Stage-5 exact contract; the bounded and
    # diagnostic modes must be explicit opt-ins.
    assert args.cross_geometry_gate == "exact"


def test_parser_cross_geometry_gate_modes():
    parser = build_parser()
    for mode in ("exact", "qualified-tp4-pp4", "diagnostic"):
        args = parser.parse_args(
            ["--summary", "s.json", "--metrics-out", "m.pt", "--cross-geometry-gate", mode]
        )
        assert args.cross_geometry_gate == mode
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--summary", "s.json", "--metrics-out", "m.pt",
             "--cross-geometry-gate", "tolerance"]
        )  # fmt: skip


# ---------------------------------------------------------------------------
# compare_cross_geometry — synthetic fixtures
# ---------------------------------------------------------------------------


def _rec(token, top8_ids, top8_values):
    """Minimal per-step record with the fields the comparator consumes."""
    return {
        "trt_token": int(token),
        "trt_top8_ids": list(top8_ids),
        "trt_top8_values": [float(v) for v in top8_values],
    }


def _row_recs(tokens, sep=2.0):
    """One record per token: argmax=token with a fixed top-1/top-2 margin."""
    return [_rec(t, [t, t + 1, t + 2], [10.0, 10.0 - sep, 5.0]) for t in tokens]


def _write_ref(tmp_path, rows_tokens, ok=True, metrics_override=None):
    summary = {
        "ok": ok,
        "config": {"tensor_parallel_size": None, "pipeline_parallel_size": 4, "configuration": "B"},
        "rows": [{"name": n, "trt_tokens": list(t)} for n, t in rows_tokens.items()],
    }
    spath = os.path.join(tmp_path, "ref.json")
    with open(spath, "w") as fh:
        json.dump(summary, fh)
    metrics = (
        metrics_override
        if metrics_override is not None
        else {n: _row_recs(t) for n, t in rows_tokens.items()}
    )
    mpath = os.path.join(tmp_path, "ref.pt")
    torch.save(metrics, mpath)
    return spath, mpath


def _cur(rows_tokens, sep=1.0):
    summary = {"rows": [{"name": n, "trt_tokens": list(t)} for n, t in rows_tokens.items()]}
    store = {n: _row_recs(t, sep=sep) for n, t in rows_tokens.items()}
    return summary, store


def test_cross_geometry_identical_trajectories(tmp_path):
    rows = {"canary0": [11, 12, 13, 14], "short9": [21, 22]}
    spath, mpath = _write_ref(str(tmp_path), rows)
    cur_summary, store = _cur(rows)
    ev, problems = compare_cross_geometry(spath, mpath, cur_summary, store, "TP4-vs-PP4-B")
    assert problems == []
    assert ev["evaluated"] is True
    assert ev["token_identical"] is True
    assert ev["rows_compared"] == 2 and ev["forked_rows"] == 0
    assert ev["steps_compared"] == 6
    assert ev["reference_mapping"]["pipeline_parallel_size"] == 4
    assert ev["gate_mode"] == "exact" and ev["gate_result"] == "pass"
    assert ev["qualified_forks"] == 0 and ev["failed_forks"] == 0
    assert all(r["classification"] == "exact" for r in ev["row_results"])
    r0 = next(r for r in ev["row_results"] if r["name"] == "canary0")
    assert r0["identical_prefix_steps"] == 4
    # Margin trail present on both sides (ref sep=2.0, cur sep=1.0).
    assert r0["min_ref_top1_top2_separation"]["separation"] == pytest.approx(2.0)
    assert r0["min_cur_top1_top2_separation"]["separation"] == pytest.approx(1.0)


def test_cross_geometry_fork_is_failure_with_margins(tmp_path):
    ref_rows = {"canary0": [11, 12, 13, 14]}
    spath, mpath = _write_ref(str(tmp_path), ref_rows)
    cur_summary, store = _cur({"canary0": [11, 12, 99, 14]})
    ev, problems = compare_cross_geometry(spath, mpath, cur_summary, store, "TP4-vs-PP4-B")
    # Acceptance C5: ANY cross-geometry token mismatch fails the run — the
    # fork must surface as a problem (which drives summary.ok=false and a
    # nonzero driver exit), never as INFO.
    assert len(problems) == 1
    assert "token fork at step 2" in problems[0]
    assert "reference token 13 vs current token 99" in problems[0]
    assert ev["token_identical"] is False
    assert ev["identical_rows"] == 0 and ev["forked_rows"] == 1
    assert ev["gate_mode"] == "exact" and ev["gate_result"] == "fail"
    assert ev["failed_forks"] == 1 and ev["qualified_forks"] == 0
    # The replan evidence must be fully retained alongside the failure.
    row = ev["row_results"][0]
    assert row["token_identical"] is False
    assert row["identical_prefix_steps"] == 2
    fork = row["fork"]
    assert fork["step"] == 2
    assert fork["ref_token"] == 13 and fork["cur_token"] == 99
    # Each side's own argmax logit is always available (top-1 of its top-8).
    assert fork["ref_logit_at_ref_token"] == pytest.approx(10.0)
    assert fork["cur_logit_at_cur_token"] == pytest.approx(10.0)
    # The other side's token is absent from these synthetic top-8s -> None,
    # reported honestly rather than fabricated.
    assert fork["ref_logit_at_cur_token"] is None
    assert fork["cur_logit_at_ref_token"] is None
    assert fork["ref_top8_values"][0] == pytest.approx(10.0)
    assert fork["cur_top8_ids"][0] == 99


def test_cross_geometry_missing_row_is_problem(tmp_path):
    spath, mpath = _write_ref(str(tmp_path), {"canary0": [11, 12]})
    cur_summary, store = _cur({"canary0": [11, 12], "short9": [21, 22]})
    ev, problems = compare_cross_geometry(spath, mpath, cur_summary, store, "x")
    assert any("short9 missing" in p for p in problems)
    assert ev["token_identical"] is False  # structural problem poisons the gate


def test_cross_geometry_length_mismatch_is_problem(tmp_path):
    spath, mpath = _write_ref(str(tmp_path), {"canary0": [11, 12, 13]})
    cur_summary, store = _cur({"canary0": [11, 12]})
    ev, problems = compare_cross_geometry(spath, mpath, cur_summary, store, "x")
    assert any("retains 3 reference vs 2 current tokens" in p for p in problems)
    assert ev["rows_compared"] == 0


def test_cross_geometry_metrics_mismatch_is_problem(tmp_path):
    rows = {"canary0": [11, 12, 13]}
    spath, mpath = _write_ref(str(tmp_path), rows, metrics_override={"canary0": _row_recs([11])})
    cur_summary, store = _cur(rows)
    ev, problems = compare_cross_geometry(spath, mpath, cur_summary, store, "x")
    assert any("per-step records mismatch" in p for p in problems)


def test_cross_geometry_ref_not_ok_is_problem(tmp_path):
    rows = {"canary0": [11, 12]}
    spath, mpath = _write_ref(str(tmp_path), rows, ok=False)
    cur_summary, store = _cur(rows)
    ev, problems = compare_cross_geometry(spath, mpath, cur_summary, store, "x")
    assert any("has ok=false" in p for p in problems)
    assert ev["token_identical"] is False


# ---------------------------------------------------------------------------
# Decision G — Stage-6 C1 bounded TP4/PP4 qualification boundary matrix
# ---------------------------------------------------------------------------
#
# The sealed iteration-55 forks, verbatim from the sealed replay JSONs
# (goal5.4-logs/replay-tp4-{b,e}-iter55.json): the ONLY qualifiable cases.
SEALED_CANARY0 = {
    "ref_tok": 11, "cur_tok": 53059,
    "ref_top8_ids": [11, 53059, 22093], "ref_top8_vals": [21.5, 21.25, 19.125],
    "cur_top8_ids": [53059, 11, 22093], "cur_top8_vals": [21.75, 21.625, 19.25],
}  # fmt: skip
SEALED_SHORT9 = {
    "ref_tok": 13, "cur_tok": 11,
    "ref_top8_ids": [13, 11, 18630], "ref_top8_vals": [25.0, 24.75, 24.5],
    "cur_top8_ids": [11, 13, 18630], "cur_top8_vals": [25.125, 24.875, 24.5],
}  # fmt: skip


def _forked_row(prefix_len, prefix_base, fork):
    """Build (ref_tokens, cur_tokens, ref_recs, cur_recs) forking at step
    ``prefix_len`` with the given fork-step top-8 records."""
    prefix = [prefix_base + i for i in range(prefix_len)]
    ref_toks = prefix + [fork["ref_tok"]]
    cur_toks = prefix + [fork["cur_tok"]]
    ref_recs = _row_recs(prefix) + [
        _rec(fork["ref_tok"], fork["ref_top8_ids"], fork["ref_top8_vals"])
    ]
    cur_recs = _row_recs(prefix) + [
        _rec(fork["cur_tok"], fork["cur_top8_ids"], fork["cur_top8_vals"])
    ]
    return ref_toks, cur_toks, ref_recs, cur_recs


def _sealed_fixture(tmp_path, canary0=SEALED_CANARY0, short9=SEALED_SHORT9, extra_rows=None):
    """Ref/cur fixture with canary0 forking at step 35 and short9 at step 8
    (the sealed Decision-G locations), plus optional extra rows."""
    c_ref, c_cur, c_ref_recs, c_cur_recs = _forked_row(35, 1000, canary0)
    s_ref, s_cur, s_ref_recs, s_cur_recs = _forked_row(8, 2000, short9)
    ref_rows = {"canary0": c_ref, "short9": s_ref}
    cur_rows = {"canary0": c_cur, "short9": s_cur}
    ref_metrics = {"canary0": c_ref_recs, "short9": s_ref_recs}
    cur_store = {"canary0": c_cur_recs, "short9": s_cur_recs}
    for name, (rt, ct, rrec, crec) in (extra_rows or {}).items():
        ref_rows[name], cur_rows[name] = rt, ct
        ref_metrics[name], cur_store[name] = rrec, crec
    spath, mpath = _write_ref(str(tmp_path), ref_rows, metrics_override=ref_metrics)
    cur_summary = {"rows": [{"name": n, "trt_tokens": list(t)} for n, t in cur_rows.items()]}
    return spath, mpath, cur_summary, cur_store


def test_decision_g_qualifies_only_the_two_sealed_forks(tmp_path):
    spath, mpath, cur_summary, store = _sealed_fixture(tmp_path)
    ev, problems = compare_cross_geometry(
        spath, mpath, cur_summary, store, "TP4-vs-PP4-B", gate_mode="qualified-tp4-pp4"
    )
    # The two sealed near-ties qualify: NO problems, gate passes, and the
    # raw fact that tokens are not identical stays honestly recorded.
    assert problems == []
    assert ev["gate_mode"] == "qualified-tp4-pp4" and ev["gate_result"] == "pass"
    assert ev["forked_rows"] == 2 and ev["qualified_forks"] == 2 and ev["failed_forks"] == 0
    assert ev["token_identical"] is False
    for name, sealed, step in (("canary0", SEALED_CANARY0, 35), ("short9", SEALED_SHORT9, 8)):
        row = next(r for r in ev["row_results"] if r["name"] == name)
        assert row["classification"] == "qualified_near_tie"
        fork = row["fork"]
        assert fork["step"] == step
        assert (fork["ref_token"], fork["cur_token"]) == (sealed["ref_tok"], sealed["cur_tok"])
        qual = fork["qualification"]
        assert qual["qualified"] is True and qual["failed_conditions"] == []
        assert qual["ref_separation"] <= 0.5 and qual["cur_separation"] <= 0.5
        assert qual["allowed_location"] in list(DECISION_G_QUALIFIED_FORKS)
    # Sealed separations, exactly: canary0 ref 0.25 / cur 0.125; short9 0.25 / 0.25.
    c_qual = next(r for r in ev["row_results"] if r["name"] == "canary0")["fork"]["qualification"]
    assert c_qual["ref_separation"] == pytest.approx(0.25)
    assert c_qual["cur_separation"] == pytest.approx(0.125)


def test_decision_g_exact_identity_passes(tmp_path):
    rows = {"canary0": [11, 12, 13], "short9": [21, 22]}
    spath, mpath = _write_ref(str(tmp_path), rows)
    cur_summary, store = _cur(rows)
    ev, problems = compare_cross_geometry(
        spath, mpath, cur_summary, store, "TP4-vs-PP4-B", gate_mode="qualified-tp4-pp4"
    )
    assert problems == []
    assert ev["gate_result"] == "pass" and ev["token_identical"] is True
    assert ev["qualified_forks"] == 0 and ev["failed_forks"] == 0
    assert all(r["classification"] == "exact" for r in ev["row_results"])


def test_decision_g_third_fork_fails(tmp_path):
    # A fork in a row that is NOT a sealed location (short7-style) is
    # unqualified even with tiny margins — count/location cap, not tolerance.
    third = {
        "ref_tok": 45, "cur_tok": 27950,
        "ref_top8_ids": [45, 27950], "ref_top8_vals": [20.0, 19.875],
        "cur_top8_ids": [27950, 45], "cur_top8_vals": [20.125, 20.0],
    }  # fmt: skip
    t_ref, t_cur, t_ref_recs, t_cur_recs = _forked_row(8, 3000, third)
    spath, mpath, cur_summary, store = _sealed_fixture(
        tmp_path, extra_rows={"short7": (t_ref, t_cur, t_ref_recs, t_cur_recs)}
    )
    ev, problems = compare_cross_geometry(
        spath, mpath, cur_summary, store, "TP4-vs-PP4-B", gate_mode="qualified-tp4-pp4"
    )
    assert len(problems) == 1 and "UNQUALIFIED under Decision G" in problems[0]
    assert "short7" in problems[0] and "not one of the two sealed" in problems[0]
    assert ev["gate_result"] == "fail"
    assert ev["forked_rows"] == 3 and ev["qualified_forks"] == 2 and ev["failed_forks"] == 1


def test_decision_g_wrong_step_fails(tmp_path):
    # canary0 forking one step EARLIER than the sealed location must fail.
    c_ref, c_cur, c_ref_recs, c_cur_recs = _forked_row(34, 1000, SEALED_CANARY0)
    s_ref, s_cur, s_ref_recs, s_cur_recs = _forked_row(8, 2000, SEALED_SHORT9)
    ref_metrics = {"canary0": c_ref_recs, "short9": s_ref_recs}
    spath, mpath = _write_ref(
        str(tmp_path), {"canary0": c_ref, "short9": s_ref}, metrics_override=ref_metrics
    )
    cur_summary = {
        "rows": [
            {"name": "canary0", "trt_tokens": c_cur},
            {"name": "short9", "trt_tokens": s_cur},
        ]
    }
    store = {"canary0": c_cur_recs, "short9": s_cur_recs}
    ev, problems = compare_cross_geometry(
        spath, mpath, cur_summary, store, "TP4-vs-PP4-B", gate_mode="qualified-tp4-pp4"
    )
    assert len(problems) == 1
    assert "canary0" in problems[0] and "step 34" in problems[0]
    assert ev["failed_forks"] == 1 and ev["qualified_forks"] == 1


def test_decision_g_changed_candidate_pair_fails(tmp_path):
    changed = dict(
        SEALED_CANARY0,
        cur_tok=481,
        cur_top8_ids=[481, 11, 22093],
        cur_top8_vals=[21.75, 21.625, 19.25],
    )
    spath, mpath, cur_summary, store = _sealed_fixture(tmp_path, canary0=changed)
    ev, problems = compare_cross_geometry(
        spath, mpath, cur_summary, store, "TP4-vs-PP4-B", gate_mode="qualified-tp4-pp4"
    )
    assert len(problems) == 1 and "candidate pair" in problems[0]
    assert "(ref 11, cur 481)" in problems[0] and "(ref 11, cur 53059)" in problems[0]
    assert ev["failed_forks"] == 1 and ev["qualified_forks"] == 1


@pytest.mark.parametrize(
    "side_vals, expect",
    [
        # cur side separation 0.625 > 0.5 (ref side stays sealed): one-sided fail.
        ({"cur_top8_vals": [22.25, 21.625, 19.25]}, "current-side candidate separation"),
        # ref side separation 0.75 > 0.5.
        ({"ref_top8_vals": [21.5, 20.75, 19.125]}, "reference-side candidate separation"),
    ],
)
def test_decision_g_separation_above_half_fails(tmp_path, side_vals, expect):
    spath, mpath, cur_summary, store = _sealed_fixture(
        tmp_path, canary0=dict(SEALED_CANARY0, **side_vals)
    )
    ev, problems = compare_cross_geometry(
        spath, mpath, cur_summary, store, "TP4-vs-PP4-B", gate_mode="qualified-tp4-pp4"
    )
    assert len(problems) == 1 and expect in problems[0]
    assert ev["failed_forks"] == 1 and ev["qualified_forks"] == 1
    assert ev["gate_result"] == "fail"


@pytest.mark.parametrize(
    "vals",
    [
        [15.5, 15.25, 12.0],  # below the [16,32) band
        [33.0, 32.75, 19.0],  # at/above the band's exclusive upper edge
    ],
)
def test_decision_g_logit_outside_band_fails(tmp_path, vals):
    spath, mpath, cur_summary, store = _sealed_fixture(
        tmp_path, canary0=dict(SEALED_CANARY0, cur_top8_vals=vals)
    )
    ev, problems = compare_cross_geometry(
        spath, mpath, cur_summary, store, "TP4-vs-PP4-B", gate_mode="qualified-tp4-pp4"
    )
    assert len(problems) == 1 and "outside [16.0,32.0)" in problems[0]
    assert ev["failed_forks"] == 1


def test_decision_g_missing_or_nonfinite_logit_fails(tmp_path):
    # Missing: the other side's token absent from cur top-8 -> None logit.
    missing = dict(SEALED_CANARY0, cur_top8_ids=[53059, 22093], cur_top8_vals=[21.75, 19.25])
    spath, mpath, cur_summary, store = _sealed_fixture(tmp_path, canary0=missing)
    ev, problems = compare_cross_geometry(
        spath, mpath, cur_summary, store, "TP4-vs-PP4-B", gate_mode="qualified-tp4-pp4"
    )
    assert len(problems) == 1 and "absent or non-finite" in problems[0]
    assert "cur_logit_at_ref_token" in problems[0]
    assert ev["failed_forks"] == 1

    # Non-finite: an inf candidate logit is unqualified (fail-closed).
    nonfinite = dict(SEALED_CANARY0, cur_top8_vals=[math.inf, 21.625, 19.25])
    spath, mpath, cur_summary, store = _sealed_fixture(tmp_path, canary0=nonfinite)
    ev, problems = compare_cross_geometry(
        spath, mpath, cur_summary, store, "TP4-vs-PP4-B", gate_mode="qualified-tp4-pp4"
    )
    assert len(problems) == 1 and "absent or non-finite" in problems[0]
    assert ev["failed_forks"] == 1


def test_decision_g_structural_mismatch_still_fails(tmp_path):
    spath, mpath, cur_summary, store = _sealed_fixture(tmp_path)
    with open(spath) as fh:
        ref = json.load(fh)
    ref["ok"] = False
    with open(spath, "w") as fh:
        json.dump(ref, fh)
    ev, problems = compare_cross_geometry(
        spath, mpath, cur_summary, store, "TP4-vs-PP4-B", gate_mode="qualified-tp4-pp4"
    )
    assert any("has ok=false" in p for p in problems)
    assert ev["gate_result"] == "fail"


def test_diagnostic_mode_reports_forks_without_gating(tmp_path):
    # TP4/EP4-vs-PP4 scope: four forks at arbitrary locations are fully
    # reported but never problems ("not a cross-geometry equality or
    # qualification gate").
    forks = {}
    for i, name in enumerate(("canary0", "short6", "short7", "short9")):
        f = {
            "ref_tok": 100 + i, "cur_tok": 200 + i,
            "ref_top8_ids": [100 + i, 200 + i], "ref_top8_vals": [20.0, 19.0],
            "cur_top8_ids": [200 + i, 100 + i], "cur_top8_vals": [20.5, 19.5],
        }  # fmt: skip
        forks[name] = _forked_row(3 + i, 1000 * (i + 1), f)
    ref_rows = {n: v[0] for n, v in forks.items()}
    cur_rows = {n: v[1] for n, v in forks.items()}
    spath, mpath = _write_ref(
        str(tmp_path), ref_rows, metrics_override={n: v[2] for n, v in forks.items()}
    )
    cur_summary = {"rows": [{"name": n, "trt_tokens": list(t)} for n, t in cur_rows.items()]}
    store = {n: v[3] for n, v in forks.items()}
    ev, problems = compare_cross_geometry(
        spath, mpath, cur_summary, store, "TP4EP4-vs-PP4-B", gate_mode="diagnostic"
    )
    assert problems == []
    assert ev["gate_mode"] == "diagnostic" and ev["gate_result"] == "diagnostic"
    assert ev["forked_rows"] == 4 and ev["failed_forks"] == 0 and ev["qualified_forks"] == 0
    assert ev["token_identical"] is False  # the raw fact stays honest
    for r in ev["row_results"]:
        assert r["classification"] == "diagnostic_fork"
        assert r["fork"]["ref_token"] is not None  # full margin evidence retained


def test_diagnostic_mode_structural_mismatch_is_still_problem(tmp_path):
    rows = {"canary0": [11, 12]}
    spath, mpath = _write_ref(str(tmp_path), rows, ok=False)
    cur_summary, store = _cur(rows)
    ev, problems = compare_cross_geometry(
        spath, mpath, cur_summary, store, "TP4EP4-vs-PP4-B", gate_mode="diagnostic"
    )
    assert any("has ok=false" in p for p in problems)
    assert ev["gate_result"] == "fail"


def test_unknown_gate_mode_rejected(tmp_path):
    rows = {"canary0": [11]}
    spath, mpath = _write_ref(str(tmp_path), rows)
    cur_summary, store = _cur(rows)
    with pytest.raises(ValueError, match="unknown cross-geometry gate mode"):
        compare_cross_geometry(spath, mpath, cur_summary, store, "x", gate_mode="tolerance")


def test_classify_fork_decision_g_direct_qualified():
    # Direct classifier check on the sealed short9 values.
    fork = {
        "step": 8, "ref_token": 13, "cur_token": 11,
        "ref_logit_at_ref_token": 25.0, "ref_logit_at_cur_token": 24.75,
        "cur_logit_at_ref_token": 24.875, "cur_logit_at_cur_token": 25.125,
    }  # fmt: skip
    classification, qual = classify_fork_decision_g("short9", fork)
    assert classification == "qualified_near_tie"
    assert qual["qualified"] is True
    assert qual["ref_separation"] == pytest.approx(0.25)
    assert qual["cur_separation"] == pytest.approx(0.25)
    # Same numbers at an unsealed row name: location cap applies.
    classification, qual = classify_fork_decision_g("short8", fork)
    assert classification == "failed"
    assert any("not one of the two sealed" in c for c in qual["failed_conditions"])
