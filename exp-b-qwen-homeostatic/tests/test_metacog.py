"""元认知的单元测试（不需要模型/显卡）。
Unit tests for metacognition (no model / GPU).

测的是"假设是否真的可证伪、证据是否真的够、历史是否真的会改变选择"——
防止 L2 退化成超参数乱撞。
These test whether hypotheses are genuinely falsifiable, whether the
evidence really suffices, and whether history actually changes the
choice — so L2 cannot decay into random hyperparameter search.
"""
import pytest

from homeo_lm.metacog import (L2_KNOBS, T, Hypothesis, diagnose, hit_rates,
                              l2_gate, rank)


def _hyp(**kw) -> Hypothesis:
    base = dict(id="x", claim="c", prediction="p", knob="lora_rank",
                values=[8, 16, 32], level="L2")
    base.update(kw)
    return Hypothesis(**base)


class TestFalsifiability:
    """没有 prediction 的假设不许动手。No prediction -> no action."""

    def test_complete_hypothesis_is_falsifiable(self):
        assert _hyp().is_falsifiable() is True

    def test_missing_prediction_is_not_falsifiable(self):
        assert _hyp(prediction="").is_falsifiable() is False

    def test_missing_claim_is_not_falsifiable(self):
        assert _hyp(claim="").is_falsifiable() is False

    def test_unknown_knob_is_not_falsifiable(self):
        assert _hyp(knob="make_it_better").is_falsifiable() is False

    def test_l1_knob_counts_as_real(self):
        assert _hyp(knob="shape_w", level="L1").is_falsifiable() is True


class TestDiagnosis:
    """规则要求多条件同时成立，且缺失 = 不知道。
    Rules need several conditions; missing means unknown, never evidence."""

    def test_capacity_fires_when_trained_but_not_usable(self):
        m = {"train_acc": 0.97, "modulation_gap": -1.0, "sweep_slope": -0.2}
        ids = [h.id for h in diagnose(m)]
        assert "capacity_deficit" in ids

    def test_single_condition_is_not_enough(self):
        """单指标不能点火——否则噪声就能触发改学习规则。
        One condition must not fire: otherwise noise could trigger a
        change of the learning rule."""
        m = {"train_acc": 0.97}
        assert "capacity_deficit" not in [h.id for h in diagnose(m)]

    def test_missing_metrics_are_unknown_not_evidence(self):
        """缺失指标不许被当成"满足条件"。
        A missing metric must not count as satisfying a condition."""
        assert diagnose({}) == []
        assert diagnose({"modulation_gap": None}) == []

    def test_healthy_metrics_fire_nothing(self):
        m = {"train_acc": 0.6, "modulation_gap": 0.9, "sweep_slope": 0.5,
             "survival_rate": 0.9, "agreement": 0.5, "illegal_rate": 0.0,
             "top_action_share": 0.3}
        assert diagnose(m) == []

    def test_collapse_fires_on_top_share_and_low_entropy(self):
        m = {"top_action_share": 0.95, "entropy": 0.05}
        assert "collapse" in [h.id for h in diagnose(m)]

    def test_collapse_has_the_v1_antidote_knob(self):
        h = [x for x in diagnose({"top_action_share": 0.95,
                                  "entropy": 0.05}) if x.id == "collapse"][0]
        assert h.knob == "entropy_coef"

    def test_distribution_mismatch_fires(self):
        m = {"agreement": 0.8, "survival_rate": 0.0}
        assert "distribution_mismatch" in [h.id for h in diagnose(m)]

    def test_every_hypothesis_carries_all_three_parts(self):
        m = {"train_acc": 0.97, "modulation_gap": -1.0, "sweep_slope": -0.2,
             "agreement": 0.8, "survival_rate": 0.0,
             "top_action_share": 0.9, "entropy": 0.1}
        for h in diagnose(m):
            assert h.is_falsifiable()

    def test_sorted_by_evidence_strength(self):
        m = {"train_acc": 0.97, "modulation_gap": -1.0, "sweep_slope": -0.2,
             "agreement": 0.8, "survival_rate": 0.0}
        hs = diagnose(m)
        assert hs[0].strength >= hs[-1].strength


class TestMetacognition:
    """历史命中率必须真的改变选择。History must actually change picks."""

    def test_hit_rates_count_confirmed(self):
        rows = [{"change": {"hypothesis": "capacity_deficit"},
                 "verdict": "adopt"},
                {"change": {"hypothesis": "capacity_deficit"},
                 "verdict": "reject"},
                {"change": {"hypothesis": "collapse"}, "verdict": "reject"}]
        s = hit_rates(rows)
        assert s["capacity_deficit"] == {"tries": 2, "confirmed": 1}
        assert s["collapse"] == {"tries": 1, "confirmed": 0}

    def test_unguessed_changes_are_not_counted(self):
        assert hit_rates([{"change": {"shape_w": 0.05}}]) == {}

    def test_untried_hypotheses_come_first(self):
        """没试过的假设优先（UCB = inf）。Untried first (UCB = inf)."""
        hs = diagnose({"train_acc": 0.97, "modulation_gap": -1.0,
                       "sweep_slope": -0.2})
        scored = rank(hs, [])
        assert scored[0]["prior_tries"] == 0
        assert scored[0]["ucb"] is None

    def test_tried_and_confirmed_outranks_tried_and_refuted(self):
        hs = [_hyp(id="a", values=[8, 16],
                   evidence=["e1", "e2"], prediction="p"),
              _hyp(id="b", values=[8, 16], evidence=["e1", "e2"],
                   prediction="p")]
        rows = ([{"change": {"hypothesis": "a"}, "verdict": "adopt"}] * 3 +
                [{"change": {"hypothesis": "b"}, "verdict": "reject"}] * 3)
        scored = {d["id"]: d for d in rank(hs, rows)}
        assert scored["a"]["score"] > scored["b"]["score"]


class TestL2Gate:
    """改学习规则是手术，门槛必须更高。
    Changing the learning rule is surgery: the bar must be higher."""

    def test_passes_with_holdout_and_two_seeds(self):
        h = _hyp()
        assert l2_gate(h, True, 2, {"lora_rank": 16})["ok"] is True

    def test_holdout_required(self):
        g = l2_gate(_hyp(), False, 3, {"lora_rank": 16})
        assert g["ok"] is False
        assert any("holdout" in r for r in g["reasons"])

    def test_single_seed_rejected(self):
        g = l2_gate(_hyp(), True, 1, {"lora_rank": 16})
        assert g["ok"] is False and any("seed" in r for r in g["reasons"])

    def test_two_knobs_at_once_rejected(self):
        """同时改两个旋钮就无法归因。Two knobs -> no attribution."""
        g = l2_gate(_hyp(), True, 3, {"lora_rank": 16, "lr": 1e-5})
        assert g["ok"] is False
        assert any("single variable" in r for r in g["reasons"])

    def test_non_l2_knob_rejected_at_l2(self):
        g = l2_gate(_hyp(knob="shape_w"), True, 3, {"shape_w": 0.05})
        assert g["ok"] is False

    def test_unfalsifiable_hypothesis_rejected(self):
        g = l2_gate(_hyp(prediction=""), True, 3, {"lora_rank": 16})
        assert g["ok"] is False


class TestL2Knobs:
    def test_l2_knobs_are_bounded(self):
        assert L2_KNOBS["lora_rank"].values == [8, 16, 32]
        assert set(L2_KNOBS) == {"lora_rank", "lr", "epochs", "replay_frac"}

    def test_thresholds_are_auditable(self):
        assert T["top_share_collapse"] == 0.60
        assert T["train_acc_high"] == 0.90


@pytest.mark.parametrize("metrics", [
    {}, {"train_acc": None}, {"modulation_gap": 0.9},
])
def test_no_evidence_no_hypothesis(metrics):
    """没有证据就不许有假设——不许凭"感觉"改学习规则。
    No evidence -> no hypothesis: never change the learning rule on a
    hunch."""
    assert diagnose(metrics) == []
