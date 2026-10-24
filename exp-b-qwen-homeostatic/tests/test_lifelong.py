"""P4 的单元测试（不需要模型/显卡）。
Unit tests for P4 (no model / GPU).

测的是"合一之后的护栏"：线上策略会不会被坏更新换掉、长期会不会悄悄
退化、以及 RSI 有没有对照组的证据。
These test the guardrails that unification requires: can a bad update
replace the live policy, does the run degrade quietly over time, and is
there control-group evidence that RSI pays.
"""
import pytest

from homeo_lm.lifelong import (MONITORED, PROMOTION, drift, health,
                               promote, throttle)
from homeo_lm.metaexp import GROUPS, equal_compute, group_plan, verdict


LIVE = {"modulation_gap": 0.5, "survival_rate": 0.4, "agreement": 0.6,
        "illegal_rate": 0.0}


class TestPromotion:
    """晋升：任一指标回退即不许上线。Any regression blocks promotion."""

    def test_strictly_better_with_holdout_promotes(self):
        cand = {"modulation_gap": 0.7, "survival_rate": 0.5,
                "agreement": 0.65, "illegal_rate": 0.0}
        p = promote(cand, LIVE, {"candidate": 0.7, "incumbent": 0.5})
        assert p["ok"] is True and p["holdout_delta"] > 0

    def test_holdout_not_measured_blocks(self):
        """线上涨但留出没测 = 不许晋升（规则 3）。
        Live up but holdout unmeasured -> no promotion (rule 3)."""
        cand = dict(LIVE, modulation_gap=0.9)
        p = promote(cand, LIVE, None)
        assert p["ok"] is False
        assert any("holdout" in r for r in p["reasons"])

    def test_holdout_flat_blocks(self):
        """留出不动 = 只是在背现在的题。Holdout flat = memorising."""
        cand = dict(LIVE, modulation_gap=0.9)
        p = promote(cand, LIVE, {"candidate": 0.50, "incumbent": 0.50})
        assert p["ok"] is False

    def test_one_metric_regressing_blocks(self):
        """总分高但某项塌了——不许用一项换另一项。
        Total up but one metric collapsed: no trading one for another."""
        cand = {"modulation_gap": 0.9, "survival_rate": 0.2,
                "agreement": 0.6, "illegal_rate": 0.0}
        p = promote(cand, LIVE, {"candidate": 0.9, "incumbent": 0.5})
        assert p["ok"] is False
        assert any("survival_rate" in r for r in p["reasons"])

    def test_illegal_rate_ceiling_blocks(self):
        cand = dict(LIVE, illegal_rate=PROMOTION["illegal_max"] + 0.1)
        p = promote(cand, LIVE, {"candidate": 0.6, "incumbent": 0.5})
        assert p["ok"] is False

    def test_missing_metric_blocks(self):
        cand = {"survival_rate": 0.9, "agreement": 0.6}
        p = promote(cand, LIVE, {"candidate": 0.9, "incumbent": 0.5})
        assert p["ok"] is False
        assert any("缺失" in r for r in p["reasons"])


class TestDrift:
    def test_no_drift_when_identical(self):
        assert drift(LIVE, LIVE) == []

    def test_big_change_is_flagged(self):
        out = drift({"modulation_gap": 0.0, "survival_rate": 0.4,
                     "agreement": 0.6}, LIVE)
        assert any("modulation_gap" in s for s in out)

    def test_missing_metrics_are_skipped(self):
        assert drift({}, LIVE) == []


class TestThrottle:
    """改进不能霸占线上算力。Improvement must not hog live compute."""

    def test_too_early(self):
        assert throttle(100, 10.0)["run"] is False

    def test_scheduled_run(self):
        assert throttle(10_000, 3600.0)["run"] is True

    def test_drift_can_start_early(self):
        assert throttle(100, 400.0, drifted=True)["run"] is True

    def test_drift_still_respects_a_floor(self):
        assert throttle(100, 0.0, drifted=True)["run"] is False


class TestHealth:
    """长期：无退化、无振荡、平台要报警。
    Long run: no degradation, no oscillation, plateau alarms."""

    def _rows(self, scores):
        return [{"score": s} for s in scores]

    def test_improving_run_is_healthy(self):
        h = health(self._rows([0.1, 0.2, 0.3, 0.4]), window=4)
        assert h["ok"] is True and h["degraded"] is False

    def test_degradation_is_detected(self):
        rows = self._rows([0.9, 0.9, 0.9, 0.9, 0.2, 0.2, 0.2, 0.2])
        h = health(rows, window=4)
        assert h["degraded"] is True and h["ok"] is False

    def test_plateau_is_flagged(self):
        rows = self._rows([0.9, 0.5, 0.5, 0.5, 0.5, 0.5])
        assert health(rows, window=10, plateau_k=5)["plateau"] is True

    def test_oscillating_knob_is_flagged(self):
        rows = [{"score": 0.3, "change": {"shape_w": 0.0}},
                {"score": 0.4, "change": {"shape_w": 0.1}},
                {"score": 0.3, "change": {"shape_w": 0.0}},
                {"score": 0.4, "change": {"shape_w": 0.1}}]
        h = health(rows, window=4)
        assert h["oscillating_knobs"] == ["shape_w"]
        assert h["ok"] is False

    def test_empty_history_is_safe(self):
        h = health([])
        assert h["cycles"] == 0 and h["best_ever"] is None
        assert h["ok"] is True

    def test_monitored_metrics_are_defined(self):
        assert set(MONITORED) == {"modulation_gap", "survival_rate",
                                  "agreement"}


class TestMetaExperiment:
    """RSI 是否真有用，只能靠对照组回答。
    Only a control group can answer whether RSI pays."""

    def test_all_five_groups_exist(self):
        assert set(GROUPS) == set("ABCDE")
        assert GROUPS["A"]["levels"] == []        # 对照组不开任何 RSI
        assert GROUPS["D"]["levels"] == ["L0", "L1"]

    def test_compute_is_equal_across_groups(self):
        plans = [group_plan(g, (0, 1, 2)) for g in "ABCDE"]
        assert equal_compute(plans)["equal"] is True

    def test_more_seeds_cost_more_but_stay_equal(self):
        plans = [group_plan(g, (0, 1, 2, 3)) for g in "ABCDE"]
        eq = equal_compute(plans)
        assert eq["equal"] is True
        assert eq["totals"]["A"] > equal_compute(
            [group_plan(g, (0,)) for g in "ABCDE"])["totals"]["A"]

    def test_superadditive_verdict(self):
        r = {"A": [0.30, 0.30, 0.30], "B": [0.35, 0.35, 0.35],
             "C": [0.36, 0.36, 0.36], "D": [0.50, 0.50, 0.50],
             "E": [0.52, 0.52, 0.52]}
        v = verdict(r)
        assert v["verdict"] == "rsi_pays"
        assert v["superadditive"] is True

    def test_equal_to_control_is_no_evidence(self):
        """D ≈ A：RSI 没买到东西——这是负结果，不是失败要藏起来。
        D ≈ A: RSI bought nothing — a negative result, not something to
        hide."""
        r = {"A": [0.40] * 3, "B": [0.40] * 3, "C": [0.40] * 3,
             "D": [0.41] * 3}
        assert verdict(r)["verdict"] == "no_evidence"

    def test_beating_control_but_not_single_levels_is_not_enough(self):
        r = {"A": [0.30] * 3, "B": [0.45] * 3, "C": [0.45] * 3,
             "D": [0.40] * 3}
        v = verdict(r)
        assert v["beats"]["A"] is True
        assert v["superadditive"] is False

    def test_noise_margin_is_required(self):
        r = {"A": [0.30] * 3, "B": [0.30] * 3, "C": [0.30] * 3,
             "D": [0.31] * 3}
        assert verdict(r, noise=0.05)["verdict"] == "no_evidence"

    def test_incomplete_results_are_not_a_verdict(self):
        assert verdict({"A": [0.3], "B": [0.3]})["verdict"] == "incomplete"

    def test_l2_increment_is_reported(self):
        r = {"A": [0.30] * 3, "B": [0.35] * 3, "C": [0.36] * 3,
             "D": [0.50] * 3, "E": [0.58] * 3}
        assert verdict(r)["l2_increment"] == pytest.approx(0.08, abs=1e-6)

    def test_plans_are_serialisable(self):
        import json
        assert json.loads(json.dumps(group_plan("D", (0,))))["group"] == "D"
