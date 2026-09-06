# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed unit coverage for the budget-raise cached-replay HF reference.

The replay path exists so a matched HF reference at a LARGER decode budget
can be issued without regenerating rows whose sealed scored text provably
did not consume the old budget (greedy prefix property). These tests pin the
qualification boundary, the fail-closed cache contract, the lazy model
build, and the runaway-disclosure accounting — all on CPU with fake
tokenizer/model objects, no checkpoint access.
"""

import glm5_next_lm_eval_hf as mod
import pytest
from glm5_next_lmeval_diff import AT_BUDGET_SLACK


class FakeTokenizer:
    """Token count == whitespace word count; deterministic and cheap."""

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(range(len(text.split())))


class FakeSamplingParams:
    def __init__(self, max_tokens, stop=("Question:",)):
        self.max_tokens = max_tokens
        self.stop = list(stop)


def words(n):
    return " ".join(f"w{i}" for i in range(n))


QUALIFY = 3072
NEW_BUDGET = 4096


def make_runner(cache, builder=None):
    def failing_builder():
        raise AssertionError("model builder must not be called for pure replay")

    return mod._CachedReplayRunner(
        model_builder=builder or failing_builder,
        tokenizer=FakeTokenizer(),
        batch_size=8,
        max_prompt_tokens=4096,
        cache_by_prompt=cache,
        qualify_budget=QUALIFY,
        slack=AT_BUDGET_SLACK,
    )


def entry(doc_id, n_tokens):
    return {"doc_id": doc_id, "text": words(n_tokens), "resp_tokens": n_tokens}


def test_qualified_rows_replay_byte_identical_without_model():
    cache = {"p0": entry(0, 100), "p1": entry(1, QUALIFY - AT_BUDGET_SLACK - 1)}
    runner = make_runner(cache)
    sp = FakeSamplingParams(NEW_BUDGET)
    out0 = runner.generate_async("p0", sp).result()
    out1 = runner.generate_async("p1", sp).result()
    assert out0.outputs[0].text == cache["p0"]["text"]
    assert out1.outputs[0].text == cache["p1"]["text"]
    assert all(o.outputs[0].finish_reason == "stop" for o in (out0, out1))
    assert runner.replayed_doc_ids == [0, 1]
    assert runner.regen_doc_ids == []
    assert runner.unserved_cache_rows() == 0
    assert runner.regen_truncated_doc_ids() == []


def test_at_cap_row_regenerates_and_model_builds_lazily_once(monkeypatch):
    calls = {"built": 0}

    def builder():
        calls["built"] += 1
        return object()

    def fake_flush(self):
        for p in self.pending:
            if not p.done:
                c = p.outputs[0]
                c.text = "regenerated answer #### 7"
                c.finish_reason = "stop"
                p.done = True
                self.completed_rows.append(
                    {
                        "generated_tokens": 42,
                        "budget": NEW_BUDGET,
                        "truncated": False,
                        "stop_string_hit": None,
                    }
                )

    monkeypatch.setattr(mod._BatchedHfRunner, "flush", fake_flush)
    cache = {"short": entry(3, 50), "capped": entry(786, QUALIFY)}
    runner = make_runner(cache, builder=builder)
    sp = FakeSamplingParams(NEW_BUDGET)
    runner.generate_async("short", sp)
    capped = runner.generate_async("capped", sp)
    assert calls["built"] == 0, "model must not build before a flush needs it"
    got = capped.result()
    assert calls["built"] == 1
    assert got.outputs[0].text == "regenerated answer #### 7"
    assert runner.regen_doc_ids == [786]
    assert runner.replayed_doc_ids == [3]
    assert runner.regen_truncated_doc_ids() == []


def test_boundary_row_exactly_at_slack_threshold_regenerates(monkeypatch):
    # resp_tokens == qualify - slack is NOT strictly below the threshold and
    # must regenerate: budget-invariance is only proven for rows strictly
    # under the retokenization-slack band.
    monkeypatch.setattr(mod._BatchedHfRunner, "flush", lambda self: None)
    cache = {"edge": entry(9, QUALIFY - AT_BUDGET_SLACK)}
    runner = make_runner(cache, builder=lambda: object())
    runner.generate_async("edge", FakeSamplingParams(NEW_BUDGET))
    assert runner.regen_doc_ids == [9]
    assert runner.replayed_doc_ids == []


def test_unknown_prompt_fails_closed():
    runner = make_runner({"known": entry(0, 10)})
    with pytest.raises(AssertionError, match="matched-config drift"):
        runner.generate_async("unknown prompt", FakeSamplingParams(NEW_BUDGET))


def test_double_service_of_same_prompt_fails_closed():
    runner = make_runner({"p": entry(0, 10)})
    sp = FakeSamplingParams(NEW_BUDGET)
    runner.generate_async("p", sp)
    with pytest.raises(AssertionError, match="matched-config drift"):
        runner.generate_async("p", sp)


def test_non_raised_budget_fails_closed():
    runner = make_runner({"p": entry(0, 10)})
    with pytest.raises(AssertionError, match="budget RAISE|must be >"):
        runner.generate_async("p", FakeSamplingParams(QUALIFY))


def test_regen_runaway_is_reported_by_doc_id(monkeypatch):
    def fake_flush(self):
        for p in self.pending:
            if not p.done:
                c = p.outputs[0]
                c.text = words(NEW_BUDGET)
                c.finish_reason = "length"
                p.done = True
                self.completed_rows.append(
                    {
                        "generated_tokens": NEW_BUDGET,
                        "budget": NEW_BUDGET,
                        "truncated": True,
                        "stop_string_hit": None,
                    }
                )

    monkeypatch.setattr(mod._BatchedHfRunner, "flush", fake_flush)
    cache = {"ok": entry(1, 20), "runaway": entry(786, QUALIFY)}
    runner = make_runner(cache, builder=lambda: object())
    sp = FakeSamplingParams(NEW_BUDGET)
    runner.generate_async("ok", sp)
    runner.generate_async("runaway", sp).result()
    assert runner.regen_truncated_doc_ids() == [786]
    # replayed rows never contribute to the runaway set
    assert 1 not in runner.regen_truncated_doc_ids()


def test_slack_matches_truncation_audit_constant():
    # One source of truth: the replay qualification band and the truncation
    # audit's at-budget band must be the same constant.
    assert AT_BUDGET_SLACK == 8


# --- native (batch-1, non-replay) disclosed-runaway truncation contract -----
# The matched-batch HF reference (iteration-62) is generated natively at
# batch_size=1, so its ok flag can no longer come from the replay path. These
# tests pin native_truncation_report to the SAME doc_id-keyed, disclosed-vs-
# undisclosed semantics as the replay path and the session truncation audit.

BUDGET = 20  # AT_BUDGET_SLACK == 8 -> a row is "at budget" from 12 tokens


def sample(doc_id, text):
    """A load_samples-shaped row: response_of() unwraps resps to ``text``."""
    return {"doc_id": doc_id, "resps": [[text]]}


def test_native_truncation_splits_disclosed_from_undisclosed():
    tok = FakeTokenizer()
    rows = {
        1: sample(1, words(30) + " #### 5"),  # long but ####-marked -> complete
        2: sample(2, words(30)),  # long, no marker/until -> truncated (undisclosed)
        3: sample(3, "short #### 7"),  # below budget -> complete
        4: sample(4, words(30) + " Question:"),  # until hit -> complete
        786: sample(786, words(40)),  # long, no marker -> truncated (disclosed)
    }
    rep = mod.native_truncation_report(rows, tok, BUDGET, [786])
    assert rep["truncated_doc_ids"] == [2, 786]
    assert rep["disclosed_runaway_doc_ids"] == [786]
    assert rep["undisclosed_truncated_doc_ids"] == [2]


def test_native_truncation_all_disclosed_is_clean():
    tok = FakeTokenizer()
    rows = {786: sample(786, words(40))}
    rep = mod.native_truncation_report(rows, tok, BUDGET, [786])
    assert rep["truncated_doc_ids"] == [786]
    assert rep["undisclosed_truncated_doc_ids"] == []


def test_native_truncation_empty_when_every_row_completes():
    tok = FakeTokenizer()
    rows = {
        1: sample(1, "a b c #### 1"),
        2: sample(2, words(30) + " #### 2"),  # long but ####-marked
    }
    rep = mod.native_truncation_report(rows, tok, BUDGET, [])
    assert rep["truncated_doc_ids"] == []
    assert rep["undisclosed_truncated_doc_ids"] == []


def test_native_truncation_no_disclosure_flags_every_at_budget_row():
    # With no disclosed ids, an at-budget unmarked row is undisclosed (the
    # fixed-100 canary expects zero truncation and must fail closed otherwise).
    tok = FakeTokenizer()
    rows = {9: sample(9, words(50))}
    rep = mod.native_truncation_report(rows, tok, BUDGET, [])
    assert rep["undisclosed_truncated_doc_ids"] == [9]
