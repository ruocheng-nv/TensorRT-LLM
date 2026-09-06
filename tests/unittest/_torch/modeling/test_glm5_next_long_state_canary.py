# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-helper tests for the Stage-4 long-horizon state canary driver.

No CUDA, no tensorrt_llm import, no torch import: these tests validate the
driver's audit logic — probe-record validation, slot-reuse isolation, and
B/E equivalence — against synthetic fixtures, including fault injections
that MUST be detected. The GPU session then runs the driver itself.
"""

import json

import glm5_next_long_state_canary as canary
import pytest


def make_hc_record(step, layer, **overrides):
    rec = {
        "kind": "hc",
        "probe_step": step,
        "layer": layer,
        "tokens": 1,
        "streams_finite": True,
        "stream_l2_mean": [10.0, 11.0, 12.0, 13.0],
        "stream_l2_max": [20.0, 21.0, 22.0, 23.0],
        "stream_sum_mean": [1.0, -2.0, 3.0, -4.0],
        "comb_finite": True,
        "comb_row_sum_min": 0.9999,
        "comb_row_sum_max": 1.0001,
        "comb_col_sum_min": 0.9999,
        "comb_col_sum_max": 1.0001,
        "post_finite": True,
        "post_min": 0.3,
        "post_max": 1.9,
    }
    rec.update(overrides)
    return rec


def make_pool_record(step, layer, **overrides):
    rec = {
        "kind": "pool",
        "probe_step": step,
        "layer": layer,
        "width": 2051,
        "kv_lens": [900],
        "sentinel_counts": [1200],
        "nonsentinel_min": [0],
        "nonsentinel_max": [899],
        "all_in_range": True,
        "rows_covered": [True],
    }
    rec.update(overrides)
    return rec


def healthy_records(min_steps=50):
    """Rank files jointly covering every layer with plenty of probed steps."""
    steps = [1 + 128 * i for i in range(min_steps)]
    # Split the 45 decoder layers across two synthetic ranks like PP would.
    rank_layers = {"0": range(0, 23), "1": range(23, canary.NUM_LAYERS)}
    records = {}
    for rank, layers in rank_layers.items():
        recs = []
        for step in steps:
            for layer in layers:
                recs.append(make_hc_record(step, layer))
                if layer in canary.SPARSE_LAYER_IDS:
                    recs.append(make_pool_record(step, layer))
        records[rank] = recs
    return records


class TestScheduleConstants:
    def test_literal_layer_schedule(self):
        assert canary.NUM_LAYERS == 45
        assert canary.SPARSE_LAYER_IDS == tuple(range(3, 45, 4))
        assert len(canary.SPARSE_LAYER_IDS) == 11

    def test_expected_min_probe_steps(self):
        # 3 rows x 2047 decode forwards / 128 = 47 full periods, minus slack 2.
        assert canary.expected_min_probe_steps(3, 2048, 128) == 45
        assert canary.expected_min_probe_steps(1, 2, 1000) == 1


class TestValidateProbeRecords:
    def test_healthy_records_pass(self):
        evidence, problems = canary.validate_probe_records(healthy_records(), min_probe_steps=45)
        assert problems == []
        assert evidence["missing_hc_layers"] == []
        assert evidence["missing_pool_layers"] == []
        assert evidence["distinct_probed_steps_hc"] >= 45
        assert evidence["comb_sum_max_abs_dev_from_1"] <= canary.COMB_SUM_TOL

    def test_non_finite_streams_detected(self):
        records = healthy_records()
        records["0"][0] = make_hc_record(1, 0, streams_finite=False)
        _, problems = canary.validate_probe_records(records, min_probe_steps=45)
        assert any("non-finite streams" in p for p in problems)

    def test_non_finite_post_detected(self):
        records = healthy_records()
        records["0"][0] = make_hc_record(1, 0, post_finite=False)
        _, problems = canary.validate_probe_records(records, min_probe_steps=45)
        assert any("non-finite post" in p for p in problems)

    def test_sinkhorn_sum_drift_detected(self):
        records = healthy_records()
        records["0"][0] = make_hc_record(1, 0, comb_row_sum_max=2.5)
        evidence, problems = canary.validate_probe_records(records, min_probe_steps=45)
        assert any("Sinkhorn comb" in p for p in problems)
        assert evidence["comb_sum_max_abs_dev_from_1"] == pytest.approx(1.5)

    def test_real_checkpoint_sinkhorn_envelope_tolerated(self):
        # The measured real-checkpoint envelope (max dev 0.084, iteration-39
        # wiring run) must NOT be flagged: finiteness is the hard gate.
        records = healthy_records()
        records["0"][0] = make_hc_record(1, 0, comb_row_sum_max=1.084, comb_col_sum_min=0.916)
        _, problems = canary.validate_probe_records(records, min_probe_steps=45)
        assert problems == []

    def test_pool_index_out_of_range_detected(self):
        records = healthy_records()
        records["0"].append(make_pool_record(1, 3, all_in_range=False))
        _, problems = canary.validate_probe_records(records, min_probe_steps=45)
        assert any("outside [0, kv_len)" in p for p in problems)

    def test_uncovered_request_row_detected(self):
        records = healthy_records()
        records["0"].append(make_pool_record(1, 3, rows_covered=[False], kv_lens=[900]))
        _, problems = canary.validate_probe_records(records, min_probe_steps=45)
        assert any("selected no visible position" in p for p in problems)

    def test_zero_kv_len_row_not_flagged(self):
        # A warmup row with kv_len 0 carries no coverage obligation.
        records = healthy_records()
        records["0"].append(make_pool_record(1, 3, rows_covered=[False], kv_lens=[0]))
        _, problems = canary.validate_probe_records(records, min_probe_steps=45)
        assert problems == []

    def test_missing_layer_coverage_detected(self):
        records = healthy_records()
        for rank in records:
            records[rank] = [
                r for r in records[rank] if not (r["kind"] == "hc" and r["layer"] == 44)
            ]
        _, problems = canary.validate_probe_records(records, min_probe_steps=45)
        assert any("never probed decoder layer(s) [44]" in p for p in problems)

    def test_missing_sparse_layer_detected(self):
        records = healthy_records()
        for rank in records:
            records[rank] = [
                r for r in records[rank] if not (r["kind"] == "pool" and r["layer"] == 43)
            ]
        _, problems = canary.validate_probe_records(records, min_probe_steps=45)
        assert any("never probed sparse layer(s) [43]" in p for p in problems)

    def test_insufficient_probed_steps_detected(self):
        records = healthy_records(min_steps=10)
        _, problems = canary.validate_probe_records(records, min_probe_steps=45)
        assert any("periodicity" in p for p in problems)


class TestReadProbeRecords(object):
    def test_reads_rank_jsonl_files(self, tmp_path):
        recs = [make_hc_record(1, 0), make_pool_record(1, 3)]
        with open(tmp_path / "rank0.jsonl", "w") as fh:
            for rec in recs:
                fh.write(json.dumps(rec) + "\n")
        (tmp_path / "not-a-rank.txt").write_text("ignored")
        loaded = canary.read_probe_records(str(tmp_path))
        assert list(loaded) == ["0"]
        assert loaded["0"] == recs


def make_row(name, tokens, state, finish="length"):
    return {
        "name": name,
        "prompt_len": 50,
        "trt_tokens": list(tokens),
        "trt_generated": len(tokens),
        "finish_reason": finish,
        "state_sha256_steps": list(state),
    }


class TestSlotReuseIsolation:
    def test_identical_rerun_passes(self):
        rows = [
            make_row("canary0", [1, 2, 3], ["a", "b"]),
            make_row("canary1", [9, 9, 9], ["x", "y"]),
            make_row("canary0_rerun", [1, 2, 3], ["a", "b"]),
        ]
        evidence, problems = canary.slot_reuse_isolation(rows)
        assert problems == []
        assert evidence["tokens_bitwise_equal"] and evidence["state_bitwise_equal"]

    def test_token_fork_detected_with_step(self):
        rows = [
            make_row("canary0", [1, 2, 3], ["a", "b"]),
            make_row("canary0_rerun", [1, 7, 3], ["a", "b"]),
        ]
        _, problems = canary.slot_reuse_isolation(rows)
        assert any("fork from canary0 at step 1" in p for p in problems)

    def test_state_mismatch_detected(self):
        rows = [
            make_row("canary0", [1, 2, 3], ["a", "b"]),
            make_row("canary0_rerun", [1, 2, 3], ["a", "z"]),
        ]
        _, problems = canary.slot_reuse_isolation(rows)
        assert any("state digests differ" in p for p in problems)

    def test_missing_rerun_detected(self):
        _, problems = canary.slot_reuse_isolation([make_row("canary0", [1], ["a"])])
        assert any("rows missing" in p for p in problems)


def make_summary(rows, ok=True, configuration="B"):
    return {"ok": ok, "config": {"configuration": configuration}, "rows": rows}


class TestCompareCanaryRuns:
    def base_rows(self):
        return [
            make_row("canary0", [1, 2, 3], ["a", "b"]),
            make_row("canary1", [4, 5, 6], ["c", "d"]),
            make_row("canary0_rerun", [1, 2, 3], ["a", "b"]),
        ]

    def test_identical_runs_bitwise_equal(self):
        b = make_summary(self.base_rows())
        e = make_summary(self.base_rows(), configuration="E")
        evidence, problems = canary.compare_canary_runs(b, e)
        assert problems == []
        assert evidence["bitwise_identical"]
        assert evidence["rows_compared"] == 3
        assert evidence["token_steps_compared"] == 9
        assert evidence["state_steps_compared"] == 6

    def test_token_fork_detected(self):
        e_rows = self.base_rows()
        e_rows[1]["trt_tokens"] = [4, 8, 6]
        _, problems = canary.compare_canary_runs(
            make_summary(self.base_rows()), make_summary(e_rows)
        )
        assert any("B/E mismatches" in p and "'trt_tokens'" in p for p in problems)

    def test_state_fork_detected(self):
        e_rows = self.base_rows()
        e_rows[0]["state_sha256_steps"] = ["a", "Z"]
        _, problems = canary.compare_canary_runs(
            make_summary(self.base_rows()), make_summary(e_rows)
        )
        assert any("'state_sha256'" in p for p in problems)

    def test_finish_reason_mismatch_detected(self):
        e_rows = self.base_rows()
        e_rows[2]["finish_reason"] = "stop"
        _, problems = canary.compare_canary_runs(
            make_summary(self.base_rows()), make_summary(e_rows)
        )
        assert any("'finish_reason'" in p for p in problems)

    def test_state_length_mismatch_detected(self):
        e_rows = self.base_rows()
        e_rows[0]["state_sha256_steps"] = ["a"]
        _, problems = canary.compare_canary_runs(
            make_summary(self.base_rows()), make_summary(e_rows)
        )
        assert any("baseline vs 1 current state steps" in p for p in problems)

    def test_bad_baseline_detected(self):
        _, problems = canary.compare_canary_runs(
            make_summary(self.base_rows(), ok=False), make_summary(self.base_rows())
        )
        assert any("ok=false" in p for p in problems)

    def test_missing_row_detected(self):
        _, problems = canary.compare_canary_runs(
            make_summary(self.base_rows()[:2]),
            make_summary(self.base_rows()),
        )
        assert any("missing from baseline" in p for p in problems)
