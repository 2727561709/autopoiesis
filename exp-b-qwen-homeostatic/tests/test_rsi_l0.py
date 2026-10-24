"""RSI L0 的单元测试（不需要模型/显卡，只需 numpy）。
Unit tests for RSI L0 (no model / GPU; numpy only).

测的是"自我改进的权限边界"：哪些改动连被考虑的资格都没有，改进到底是
真的还是在噪声里打转，变差了会不会自动回滚。
These test the PERMISSION BOUNDARY of self-improvement: which changes
are not even eligible, whether an improvement is real or noise, and
whether a regression rolls back.
"""
import numpy as np
import pytest

from homeo_lm.rsi import significant
from homeo_lm.rsi_l0 import (FORBIDDEN_KEYS, KNOBS, Budget, Knob,
                             cycle_plan, decide, forbidden_keys_in, gate,
                             knob_to_args, oscillating, ucb_propose,
                             validate_knobs)


class TestNeverChangeList:
    """不可改清单：结构性拒绝，不看指标。
    Never-change list: structural rejection, metrics never consulted."""

    def test_objective_keys_are_forbidden(self):
        assert "energy_setpoint" in FORBIDDEN_KEYS
        assert "deficit_fn" in FORBIDDEN_KEYS

    def test_forbidden_key_is_detected(self):
        assert forbidden_keys_in({"energy_setpoint": 0.9}) == ["energy_setpoint"]

    def test_gate_rejects_objective_change(self):
        g = gate({"energy_setpoint": 0.99}, "abc", "abc")
        assert g["ok"] is False
        assert any("forbidden" in r for r in g["reasons"])

    def test_gate_rejects_probe_change_even_with_good_metrics(self):
        """改探针 = 改考卷，成绩再好也作废。
        Changing the probe = rewriting the exam; void even if high."""
        g = gate({"shape_w": 0.05}, "newhash", "oldhash")
        assert g["ok"] is False
        assert any("probe changed" in r for r in g["reasons"])

    def test_empty_change_rejected(self):
        assert gate({}, "h", "h")["ok"] is False


class TestKnobSpace:
    """旋钮必须有界。Knobs must be bounded."""

    def test_every_knob_has_values(self):
        for name, k in KNOBS.items():
            assert k.values and isinstance(k, Knob)

    def test_in_range_is_valid(self):
        assert validate_knobs({"shape_w": 0.05}) == []

    def test_out_of_range_is_rejected(self):
        bad = validate_knobs({"shape_w": 99.0})
        assert bad and "out of range" in bad[0]

    def test_unknown_knob_is_rejected(self):
        assert any("unknown knob" in r for r in
                   validate_knobs({"make_it_smarter": True}))

    def test_knob_to_args_maps_stage_and_flag(self):
        args = knob_to_args({"shape_w": 0.05})
        assert args["dpo_pairs"] == ["--shape-w", "0.05"]

    def test_forbidden_keys_never_reach_args(self):
        assert knob_to_args({"energy_setpoint": 0.9}) == {}

    def test_empty_knob_raises(self):
        with pytest.raises(ValueError):
            Knob("empty", [], "env")


class TestUCB:
    """UCB 提议器：先试没试过的，再按均值+探索项挑。
    UCB: try untried arms first, then mean + exploration bonus."""

    def test_first_proposal_is_an_untried_arm(self):
        p = ucb_propose([], ["shape_w"])
        assert p["untried"] is True and p["change"]["shape_w"] in \
            KNOBS["shape_w"].values

    def test_second_proposal_differs_from_the_first(self):
        h = [{"change": ucb_propose([], ["shape_w"])["change"], "score": 0.1}]
        p2 = ucb_propose(h, ["shape_w"])
        assert p2["arm"] != h[0]["change"] and p2["untried"] is True

    def test_best_arm_is_exploited_once_all_are_tried(self):
        hist = []
        for v in KNOBS["shape_w"].values:
            hist.append({"change": {"shape_w": v},
                         "score": 1.0 if v == 0.1 else 0.0})
        p = ucb_propose(hist, ["shape_w"])
        assert p["change"]["shape_w"] == 0.1
        assert p["untried"] is False

    def test_unknown_knob_is_ignored(self):
        assert ucb_propose([], ["nope"])["change"] == {}


class TestBudget:
    """预算与熔断：不许算力失控，连续失败要交给人类。
    Budget + circuit breaker: no runaway compute; repeated failure
    escalates to a human."""

    def test_within_budget(self):
        assert Budget(max_trials=3).can_try(1, 0)[0] is True

    def test_budget_exhausted(self):
        ok, why = Budget(max_trials=3).can_try(3, 0)
        assert ok is False and "budget" in why

    def test_circuit_breaker_freezes(self):
        ok, why = Budget(max_consecutive_fail=2).can_try(0, 2)
        assert ok is False and "circuit breaker" in why


class TestOscillation:
    """反振荡：在噪声里打转要停下来。Spin in noise -> stop."""

    def test_zigzag_detected(self):
        h = [{"change": {"shape_w": 0.0}}, {"change": {"shape_w": 0.1}},
             {"change": {"shape_w": 0.0}}, {"change": {"shape_w": 0.1}}]
        assert oscillating(h, "shape_w") is True

    def test_monotone_is_not_oscillation(self):
        h = [{"change": {"shape_w": 0.0}}, {"change": {"shape_w": 0.025}},
             {"change": {"shape_w": 0.05}}, {"change": {"shape_w": 0.075}}]
        assert oscillating(h, "shape_w") is False

    def test_too_short_to_judge(self):
        assert oscillating([{"change": {"shape_w": 0.0}}], "shape_w") is False


class TestDecide:
    """采纳/拒绝/回滚。Adopt / reject / roll back."""

    def _samples(self, mu_a, mu_b, n=30, sd=0.05):
        # sd=0.05 / n=30：0.01 的差值确实落在噪声里（第一次写测试时用了
        # sd=0.02/n=40，那个尺度下 0.01 是**真**差异，被判显著是对的）
        # sd=0.05 / n=30: a 0.01 gap really is within noise (the first
        # version used sd=0.02/n=40, where 0.01 IS a real difference and
        # calling it significant is correct)
        rng = np.random.default_rng(7)
        return (list(rng.normal(mu_a, sd, n)),
                list(rng.normal(mu_b, sd, n)))

    def test_clear_gain_is_adopted(self):
        a, b = self._samples(0.40, 0.70)
        assert decide(a, b, n_boot=500)["verdict"] == "adopt"

    def test_noise_is_rejected_not_adopted(self):
        # sd=0.25 / n=20：0.01 的差值彻底淹没在噪声里（与种子无关）
        # sd=0.25 / n=20: a 0.01 gap is buried in noise (seed-independent)
        a, b = self._samples(0.50, 0.51, n=20, sd=0.25)
        assert decide(a, b, n_boot=500)["verdict"] == "reject"

    def test_regression_is_rolled_back(self):
        a, b = self._samples(0.70, 0.40)
        d = decide(a, b, n_boot=500)
        assert d["verdict"] == "rollback"

    def test_guardrail_violation_rolls_back_regardless_of_metrics(self):
        """成绩再好，守卫失败也必须回滚。
        No matter how good the score, a guardrail trips -> roll back."""
        a, b = self._samples(0.40, 0.90)
        d = decide(a, b, guardrails_ok=False, n_boot=200)
        assert d["verdict"] == "rollback"
        assert "guardrail" in d["reason"]


class TestCyclePlan:
    """整周期计划：门控不过就什么都不许跑。
    Full cycle plan: if the gate fails, nothing may run."""

    def test_plan_runs_when_clean(self):
        plan = cycle_plan([], ["shape_w"], "h", None)
        assert plan["gate"]["ok"] is True
        assert plan["args"]

    def test_oscillating_knob_blocks_the_cycle(self):
        h = [{"change": {"shape_w": 0.0}, "score": 0.2},
             {"change": {"shape_w": 0.1}, "score": 0.9},
             {"change": {"shape_w": 0.0}, "score": 0.2},
             {"change": {"shape_w": 0.1}, "score": 0.9}]
        plan = cycle_plan(h, ["shape_w"], "h", "h")
        assert plan["gate"]["ok"] is False
        assert plan["args"] == {}

    def test_frozen_after_repeated_failure(self):
        plan = cycle_plan([], ["shape_w"], "h", "h",
                          Budget(max_consecutive_fail=1),
                          cycle_trials=0, consecutive_fail=3)
        assert plan["gate"]["ok"] is False
        assert plan["args"] == {}


def test_significance_and_decide_agree():
    """显著性判定与采纳判定不能自相矛盾。
    The significance test and the adopt decision must not disagree."""
    rng = np.random.default_rng(11)
    a = list(rng.normal(0.5, 0.05, 30))
    b = list(rng.normal(0.5, 0.05, 30))
    assert significant(a, b, n_boot=300)["significant"] is False
    assert decide(a, b, n_boot=300)["verdict"] in ("reject", "rollback")
