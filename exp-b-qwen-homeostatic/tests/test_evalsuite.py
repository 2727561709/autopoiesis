"""评测 Harness 的单元测试（不需要模型/显卡）。
Unit tests for the eval harness (no model / GPU).

测的是**判定口径会不会漂**：门槛操作符是否按写的执行、方向是否按
"越大越好"统一、缺失指标是否算不通过、对比表的 delta 符号是否一致。
These test whether the VERDICT CRITERION can drift: do the operators do
what they say, are metrics oriented consistently, does a missing metric
FAIL, and is the delta sign consistent across the table.
"""
import json

import pytest

from homeo_lm.evalsuite import (HIGHER_IS_BETTER, PROBES, Case,
                                check_thresholds, compare, default_suite,
                                evaluate, load_suite)

BASE_REPORT = {
    "agreement": {"agreement": 0.60, "illegal_rate": 0.02},
    "survival": {"survival_rate": 0.40, "mean_steps": 120.0,
                 "illegal_rate": 0.01},
    "modulation": {"modulation_gap": 0.20},
    "modulation-sweep": {"slope": 0.05, "monotone_increasing": True},
}
CAND_REPORT = {
    "agreement": {"agreement": 0.66, "illegal_rate": 0.01},
    "survival": {"survival_rate": 0.55, "mean_steps": 150.0,
                 "illegal_rate": 0.00},
    "modulation": {"modulation_gap": 0.70},
    "modulation-sweep": {"slope": 0.30, "monotone_increasing": True},
}


class TestCase:
    def test_unknown_probe_is_rejected(self):
        with pytest.raises(ValueError):
            Case(name="bad", probe="nope")

    def test_every_known_probe_is_accepted(self):
        for p in PROBES:
            assert Case(name=p, probe=p).probe == p

    def test_roundtrip_to_dict(self):
        c = Case(name="x", probe="survival", samples=5,
                 thresholds={"illegal_rate": ("<=", 0.2)})
        d = c.as_dict()
        assert d["thresholds"]["illegal_rate"] == ["<=", 0.2]
        assert json.loads(json.dumps(d))["name"] == "x"


class TestThresholds:
    def test_ge_passes_at_the_boundary(self):
        r = check_thresholds({"gap": 0.5}, {"gap": (">=", 0.5)})
        assert r["ok"] is True

    def test_gt_fails_at_the_boundary(self):
        r = check_thresholds({"gap": 0.5}, {"gap": (">", 0.5)})
        assert r["ok"] is False

    def test_illegal_rate_is_inverted(self):
        """非法率越低越好：0.3 应当**不过**（阈值 ≤0.2）。
        Illegal rate is lower-is-better: 0.3 must FAIL (threshold <=0.2)."""
        assert HIGHER_IS_BETTER["illegal_rate"] is False
        r = check_thresholds({"illegal_rate": 0.3},
                             {"illegal_rate": ("<=", 0.2)})
        assert r["ok"] is False

    def test_missing_metric_fails(self):
        """缺失指标 = 不知道 = **不过**（不许默认通过）。
        A missing metric is UNKNOWN and FAILS (never a default pass)."""
        r = check_thresholds({}, {"modulation_gap": (">=", 0.5)})
        assert r["ok"] is False
        assert "missing" in r["reasons"][0]

    def test_unknown_operator_fails(self):
        r = check_thresholds({"gap": 1.0}, {"gap": ("~", 0.5)})
        assert r["ok"] is False

    def test_no_thresholds_passes_vacuously(self):
        assert check_thresholds({"gap": 0.1}, {})["ok"] is True


class TestCompare:
    def test_delta_is_positive_when_the_candidate_is_better(self):
        cmp = compare(BASE_REPORT, CAND_REPORT)
        assert cmp["delta"]["modulation_gap"] > 0
        assert cmp["delta"]["survival_rate"] > 0

    def test_illegal_rate_delta_is_oriented(self):
        """非法率下降 → delta 为正（"变好"统一成正号）。
        A falling illegal rate must give a POSITIVE delta ("better" is
        always positive)."""
        cmp = compare(BASE_REPORT, CAND_REPORT)
        assert cmp["delta"]["illegal_rate"] > 0

    def test_missing_metric_is_none_not_zero(self):
        cmp = compare(BASE_REPORT, {"modulation": {"modulation_gap": 0.9}})
        assert cmp["delta"]["survival_rate"] is None

    def test_composite_tracks_the_metrics(self):
        cmp = compare(BASE_REPORT, CAND_REPORT)
        assert cmp["composite"]["candidate"] > cmp["composite"]["base"]

    def test_thresholds_are_applied_to_the_candidate(self):
        cmp = compare(BASE_REPORT, CAND_REPORT,
                      thresholds={"modulation_gap": (">=", 0.5)})
        assert cmp["thresholds"]["ok"] is True

    def test_failing_threshold_is_reported(self):
        cmp = compare(BASE_REPORT, BASE_REPORT,
                      thresholds={"modulation_gap": (">=", 0.5)})
        assert cmp["thresholds"]["ok"] is False


class TestSuite:
    def test_default_suite_covers_all_probes(self):
        probes = {c.probe for c in default_suite()}
        assert probes == set(PROBES)

    def test_cases_have_unique_names(self):
        names = [c.name for c in default_suite()]
        assert len(names) == len(set(names))

    def test_load_suite_from_file(self, tmp_path):
        p = tmp_path / "s.json"
        p.write_text(json.dumps([
            {"name": "m", "probe": "modulation", "seed": 1, "samples": 2,
             "thresholds": {"modulation_gap": [">=", 0.5]}}]),
            encoding="utf-8")
        s = load_suite(str(p))
        assert s[0].name == "m" and s[0].thresholds["modulation_gap"] == \
            (">=", 0.5)

    def test_load_suite_default(self):
        assert len(load_suite(None)) == len(default_suite())

    def test_evaluate_passes_when_thresholds_are_met(self):
        out = evaluate({"base": BASE_REPORT, "candidate": CAND_REPORT},
                       [Case(name="m", probe="modulation",
                             thresholds={"modulation_gap": (">=", 0.5)})])
        assert out["overall_pass"] is True

    def test_evaluate_fails_when_a_threshold_is_missed(self):
        out = evaluate({"base": BASE_REPORT, "candidate": BASE_REPORT},
                       [Case(name="m", probe="modulation",
                             thresholds={"modulation_gap": (">=", 0.5)})])
        assert out["overall_pass"] is False

    def test_evaluate_reports_the_probe_hash(self):
        out = evaluate({"base": BASE_REPORT}, default_suite())
        assert out["probe_hash"]
