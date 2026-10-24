"""RSI 控制器的单元测试（不需要模型/显卡，只需 numpy）。
Unit tests for the RSI controller (no model / GPU; numpy only).

测的是"尺子和账本"本身是否可信：种子是否真不重叠、探针哈希是否随配置
变化、显著性判定是否会被噪声骗过、账本是否可回溯。
These test whether the ruler and the ledger themselves are trustworthy:
are the seeds really disjoint, does the probe hash track the config,
can significance be fooled by noise, is the ledger traceable.
"""
import json

import numpy as np
import pytest

from homeo_lm.rsi import (DEFAULT_PROBE_CFG, GUARDRAILS, Ledger,
                          bootstrap_ci, check_guardrails, probe_hash,
                          seed_set, seeds_are_disjoint, significant)


class TestSeedSplit:
    """种子三分：不许对着同一套题反复练。
    Three-way seed split: no drilling on the same exam."""

    def test_sets_are_disjoint(self):
        tr = seed_set("train", 20)
        se = seed_set("select", 20)
        ho = seed_set("holdout", 20)
        assert seeds_are_disjoint(tr, se, ho)

    def test_deterministic(self):
        assert seed_set("holdout", 5) == seed_set("holdout", 5)

    def test_wrong_name_raises(self):
        with pytest.raises(ValueError):
            seed_set("exam", 5)

    def test_overlap_is_detected(self):
        a = seed_set("train", 10)
        assert not seeds_are_disjoint(a, a[:3])


class TestProbeLock:
    """探针锁：考试变了，成绩就必须作废。
    Probe lock: if the exam changes, scores must be voided."""

    def test_hash_is_stable(self):
        assert probe_hash(DEFAULT_PROBE_CFG) == probe_hash(DEFAULT_PROBE_CFG)

    def test_hash_tracks_config(self):
        cfg = dict(DEFAULT_PROBE_CFG)
        cfg["modulation_trials"] = 999
        assert probe_hash(cfg) != probe_hash(DEFAULT_PROBE_CFG)

    def test_hash_ignores_key_order(self):
        a = {"x": 1, "y": 2}
        b = {"y": 2, "x": 1}
        assert probe_hash(a) == probe_hash(b)


class TestSignificance:
    """显著性：不许把噪声当进步。Significance: noise is not progress."""

    def test_clear_improvement_is_significant(self):
        rng = np.random.default_rng(0)
        before = rng.normal(0.50, 0.02, 40)
        after = rng.normal(0.70, 0.02, 40)
        r = significant(before, after, n_boot=500, seed=0)
        assert r["significant"] is True
        assert r["delta"] > 0.15

    def test_noise_is_not_significant(self):
        rng = np.random.default_rng(1)
        before = rng.normal(0.50, 0.05, 30)
        after = rng.normal(0.51, 0.05, 30)
        r = significant(before, after, n_boot=500, seed=0)
        assert r["significant"] is False, "噪声被判成了进步 / noise called progress"

    def test_regression_is_flagged(self):
        rng = np.random.default_rng(2)
        before = rng.normal(0.70, 0.02, 40)
        after = rng.normal(0.50, 0.02, 40)
        r = significant(before, after, n_boot=500, seed=0)
        assert r["regressed"] is True and r["significant"] is False

    def test_empty_sample_is_safe(self):
        r = significant([], [1.0, 2.0], n_boot=10)
        assert r["significant"] is False

    def test_bootstrap_ci_contains_mean(self):
        xs = list(np.random.default_rng(3).normal(0.5, 0.1, 100))
        lo, hi = bootstrap_ci(xs, n_boot=500, seed=0)
        assert lo <= np.mean(xs) <= hi


class TestLedger:
    """账本：可回溯、可回滚。Ledger: traceable, rollback-able."""

    def test_append_and_read(self, tmp_path):
        led = Ledger(tmp_path / "ledger.jsonl")
        e1 = led.append({"tag": "baseline-A", "metrics": {"gap": -1.0}})
        e2 = led.append({"tag": "candidate-1", "parent": e1["id"],
                         "metrics": {"gap": 0.4}})
        rows = led.read()
        assert len(rows) == 2
        assert rows[1]["parent"] == e1["id"]
        assert led.last()["tag"] == "candidate-1"
        assert e2["id"] != e1["id"]

    def test_entries_are_valid_json_lines(self, tmp_path):
        led = Ledger(tmp_path / "l.jsonl")
        led.append({"tag": "a"})
        led.append({"tag": "b"})
        for line in (tmp_path / "l.jsonl").read_text(encoding="utf-8").splitlines():
            assert json.loads(line)

    def test_missing_file_reads_empty(self, tmp_path):
        assert Ledger(tmp_path / "nope.jsonl").read() == []


class TestGuardrails:
    """守卫：v1 坍缩与 v2 反向调制直接编码成报警条件。
    Guardrails: v1 collapse and v2 inversion encoded as alarms."""

    def test_healthy_report_passes(self):
        rep = {"agreement": {"illegal_rate": 0.0},
               "survival": {"illegal_rate": 0.0},
               "modulation": {"modulation_gap": 0.8}}
        assert check_guardrails(rep)["ok"] is True

    def test_illegal_rate_trips(self):
        rep = {"agreement": {"illegal_rate": GUARDRAILS["illegal_rate_max"] + 0.1},
               "survival": {"illegal_rate": 0.0}}
        out = check_guardrails(rep)
        assert out["ok"] is False and out["violations"]

    def test_inverted_modulation_is_named(self):
        rep = {"modulation": {"modulation_gap": -1.0}}
        out = check_guardrails(rep)
        assert any("inverted" in v for v in out["violations"])
