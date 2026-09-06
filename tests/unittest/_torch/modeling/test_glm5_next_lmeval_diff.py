# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial unit tests for the fail-closed lm-eval score diff (Stage-6 C2).

The Stage-6 C2 gate requires "all samples scored" to be PROVEN, not assumed:
a missing or non-finite per-row filter value must fail the diff rather than
silently count as 0.0, the expected filter set must be the union across ALL
rows of BOTH sides (not just the first HF row), and an empty filter map must
fail. These tests drive ``run_diff`` over synthetic lm-eval sample payloads
covering each of those failure modes plus the alignment/prompt-identity
contracts the sanctioned sessions gate on.
"""

import json
import math
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from glm5_next_lmeval_diff import load_samples, run_diff  # noqa: E402  (script-dir import)

STRICT = "strict-match"
FLEX = "flexible-extract"


def _entry(doc_id, filt, score, prompt=None, resp="resp"):
    """One lm-eval 0.4.x-shaped (doc, filter) sample entry."""
    return {
        "doc_id": doc_id,
        "filter": filt,
        "exact_match": score,
        "arguments": [[prompt if prompt is not None else f"prompt-{doc_id}", {}]],
        "resps": [[resp]],
        "doc": {"question": f"q{doc_id}"},
    }


def _write(tmp_path, name, entries):
    path = os.path.join(str(tmp_path), name)
    with open(path, "w") as fh:
        json.dump(entries, fh)
    return path


def _complete_side(tmp_path, name, docs=(0, 1, 2), scores=None):
    entries = []
    for d in docs:
        for filt in (STRICT, FLEX):
            entries.append(_entry(d, filt, (scores or {}).get((d, filt), 1.0)))
    return _write(tmp_path, name, entries)


def _run(tmp_path, hf_path, trt_path):
    out = os.path.join(str(tmp_path), "diff.json")
    rc = run_diff(SimpleNamespace(hf=hf_path, trt=trt_path, out=out))
    with open(out) as fh:
        return rc, json.load(fh)


def test_complete_scoring_passes(tmp_path):
    hf = _complete_side(tmp_path, "hf.json", scores={(1, STRICT): 0.0})
    trt = _complete_side(tmp_path, "trt.json", scores={(2, FLEX): 0.0})
    rc, rep = _run(tmp_path, hf, trt)
    assert rc == 0
    assert rep["doc_id_aligned"] is True and rep["prompt_mismatches"] == []
    assert rep["complete_scoring"] is True and rep["scoring_problem_count"] == 0
    assert rep["expected_filters"] == sorted([STRICT, FLEX])
    for name in (STRICT, FLEX):
        s = rep["scores"][name]
        assert s["n"] == 3 and s["expected_n"] == 3
    assert rep["scores"][STRICT]["hf"] == 2.0 and rep["scores"][STRICT]["trt"] == 3.0
    # Discriminating sets derive from valid pairs only.
    assert [e["doc_id"] for e in rep["discriminating"][FLEX]["hf_correct_trt_wrong"]] == [2]
    assert [e["doc_id"] for e in rep["discriminating"][STRICT]["trt_correct_hf_wrong"]] == [1]


def test_missing_trt_filter_fails_closed(tmp_path):
    hf = _complete_side(tmp_path, "hf.json")
    trt_entries = [
        _entry(d, filt, 1.0) for d in (0, 1, 2) for filt in (STRICT, FLEX)
        if not (d == 1 and filt == FLEX)
    ]  # fmt: skip
    trt = _write(tmp_path, "trt.json", trt_entries)
    rc, rep = _run(tmp_path, hf, trt)
    # The missing row-filter must be a named problem and a nonzero exit —
    # never a silent 0.0 that shows up only as a lower score.
    assert rc == 1
    assert rep["complete_scoring"] is False
    assert any("doc 1: missing trt filter" in p and FLEX in p for p in rep["scoring_problems"])
    assert rep["scores"][FLEX]["n"] == 2 and rep["scores"][FLEX]["expected_n"] == 3
    assert rep["scores"][STRICT]["n"] == 3


def test_filter_absent_from_first_hf_row_still_expected(tmp_path):
    # The old defect derived filter names from the FIRST HF row only. A
    # filter present everywhere except HF row 0 must still be expected and
    # its absence there must fail.
    hf_entries = [_entry(0, STRICT, 1.0)] + [
        _entry(d, filt, 1.0) for d in (1, 2) for filt in (STRICT, FLEX)
    ]
    hf = _write(tmp_path, "hf.json", hf_entries)
    trt = _complete_side(tmp_path, "trt.json")
    rc, rep = _run(tmp_path, hf, trt)
    assert rc == 1
    assert FLEX in rep["expected_filters"]
    assert any("doc 0: missing hf filter" in p and FLEX in p for p in rep["scoring_problems"])


def test_nonfinite_score_fails_closed(tmp_path):
    hf = _complete_side(tmp_path, "hf.json")
    trt = _complete_side(tmp_path, "trt.json", scores={(2, STRICT): math.nan})
    rc, rep = _run(tmp_path, hf, trt)
    assert rc == 1
    assert rep["complete_scoring"] is False
    assert any("non-finite trt filter" in p and STRICT in p for p in rep["scoring_problems"])
    assert rep["scores"][STRICT]["n"] == 2


def test_empty_filter_map_fails(tmp_path):
    # Entries with no filter/exact_match at all: an empty expected-filter set
    # must fail rather than yield a vacuous scores={} pass.
    entries = [
        {"doc_id": d, "arguments": [[f"prompt-{d}", {}]], "resps": [["r"]], "doc": {}}
        for d in (0, 1)
    ]
    hf = _write(tmp_path, "hf.json", entries)
    trt = _write(tmp_path, "trt.json", entries)
    rc, rep = _run(tmp_path, hf, trt)
    assert rc == 1
    assert rep["expected_filters"] == [] and rep["scores"] == {}
    assert rep["complete_scoring"] is False
    assert any("empty filter set" in p for p in rep["scoring_problems"])


def test_doc_misalignment_still_fails(tmp_path):
    hf = _complete_side(tmp_path, "hf.json", docs=(0, 1, 2))
    trt = _complete_side(tmp_path, "trt.json", docs=(0, 1, 3))
    rc, rep = _run(tmp_path, hf, trt)
    assert rc == 1
    assert rep["doc_id_aligned"] is False


def test_prompt_mismatch_still_fails(tmp_path):
    hf = _complete_side(tmp_path, "hf.json")
    trt_entries = [
        _entry(d, filt, 1.0, prompt=("DIFFERENT" if d == 1 else None))
        for d in (0, 1, 2)
        for filt in (STRICT, FLEX)
    ]
    trt = _write(tmp_path, "trt.json", trt_entries)
    rc, rep = _run(tmp_path, hf, trt)
    assert rc == 1
    assert rep["prompt_mismatches"] == [1]
    # Scoring itself is complete; only prompt identity failed.
    assert rep["complete_scoring"] is True


def test_load_samples_merges_per_filter_entries(tmp_path):
    path = _complete_side(tmp_path, "side.json", docs=(7,))
    _, rows = load_samples(path)
    assert sorted(rows) == [7]
    assert sorted(rows[7]["_filters"]) == sorted([STRICT, FLEX])
    assert rows[7]["_filters"][STRICT] == 1.0
