"""L0 真跑接线的单元测试（不需要模型/显卡）。
Unit tests for the real-run L0 wiring (no model / GPU).

测的是"真跑起来会不会把结果搞成不可归因"：试验目录是否可回溯、是否所有
试验都从**同一个起点**训练、守卫与指标提取是否只看环境侧信号。
These test whether a real run would produce results that cannot be
attributed: is the trial dir traceable, do all trials train from the
SAME starting point, do the guardrails and metrics look only at
environment-side signals.
"""
import pytest

from homeo_lm.rsi import (guardrails_from_report, metrics_from_report_safe,
                          trial_commands, trial_dir)


class TestTrialDir:
    def test_change_is_in_the_name(self):
        d = trial_dir("runs", {"shape_w": 0.08}, "t1")
        assert "shape_w0.08" in str(d) and "t1" in str(d)

    def test_empty_change_gets_a_stable_name(self):
        assert str(trial_dir("runs", {}, "t")).endswith("t-base")

    def test_unsafe_characters_are_sanitised(self):
        d = trial_dir("runs", {"curriculum": "hard/mixed"}, "t")
        assert "/" not in str(d)

    def test_sorted_so_the_same_change_maps_to_one_dir(self):
        a = trial_dir("runs", {"shape_w": 0.05, "gate_margin": 0.1}, "t")
        b = trial_dir("runs", {"gate_margin": 0.1, "shape_w": 0.05}, "t")
        assert str(a) == str(b)


class TestTrialCommands:
    """归因的前提：同起点、同数据量、旋钮真的传下去。
    Attribution requires: same start, same data size, knob actually
    passed through."""

    def _tc(self, change):
        return trial_commands(change, "runs", "teacher.pt",
                              ref_ckpt="runs/sft-lora-v2", tag="t")

    def test_has_generate_and_train_stages(self):
        tc = self._tc({"shape_w": 0.08})
        assert set(tc["commands"]) == {"gen", "train"}
        assert "gen_dpo_data" in tc["commands"]["gen"][2]
        assert "dpo" in tc["commands"]["train"][2]

    def test_knob_reaches_the_data_generation(self):
        tc = self._tc({"shape_w": 0.08})
        gen = tc["commands"]["gen"]
        assert "--shape-w" in gen and "0.08" in gen

    def test_every_trial_trains_from_the_same_start(self):
        """不同旋钮的试验必须同起点，否则"旋钮好坏"与"起点不同"混淆。
        （数据路径各试验不同是对的；要比的是训练起点 `--ref`。）
        Trials of different knobs must share a start, or 'is the knob
        good' is confounded with 'the start differed'. (Each trial has
        its own data path — that is correct; what must match is the
        training start, `--ref`.)"""
        def ref_of(tc):
            cmd = tc["commands"]["train"]
            return cmd[cmd.index("--ref") + 1]

        a = self._tc({"shape_w": 0.05})
        b = self._tc({"gate_margin": 0.2})
        assert ref_of(a) == ref_of(b) == "runs/sft-lora-v2"

    def test_data_and_output_stay_inside_the_trial_dir(self):
        tc = self._tc({"shape_w": 0.08})
        d = tc["dir"]
        assert all(d in x for x in tc["commands"]["gen"] if x.endswith(
            "pairs.jsonl"))
        assert tc["ckpt"].endswith("adapter")

    def test_forbidden_change_is_not_mapped(self):
        """不可改清单里的键不会变成命令——结构性拒绝，不看指标。
        A forbidden key never becomes a command: structural rejection,
        metrics never consulted."""
        tc = self._tc({"energy_setpoint": 0.9})
        assert tc["unmapped"] == {"energy_setpoint": 0.9}
        assert "--energy-setpoint" not in tc["commands"]["gen"]

    def test_unknown_knob_is_reported_not_silently_dropped(self):
        tc = self._tc({"magic_knob": 1})
        assert tc["unmapped"] == {"magic_knob": 1}


class TestGuardrails:
    def test_clean_report_passes(self):
        rep = {"survival": {"illegal_rate": 0.0},
               "agreement": {"illegal_rate": 0.01}}
        assert guardrails_from_report(rep) is True

    def test_high_illegal_rate_fails(self):
        rep = {"survival": {"illegal_rate": 0.5}, "agreement": {}}
        assert guardrails_from_report(rep) is False

    def test_either_source_can_trip_it(self):
        assert guardrails_from_report(
            {"agreement": {"illegal_rate": 0.3}}) is False

    def test_missing_report_is_not_a_violation(self):
        assert guardrails_from_report({}) is True


class TestMetricsExtraction:
    def test_extracts_nested_values(self):
        rep = {"modulation": {"modulation_gap": 0.7},
               "survival": {"survival_rate": 0.4}}
        m = metrics_from_report_safe(rep)
        assert m["modulation_gap"] == 0.7
        assert m["survival_rate"] == 0.4

    def test_missing_report_is_empty_not_an_error(self):
        assert metrics_from_report_safe({}) == {}


@pytest.mark.parametrize("change", [
    {"shape_w": 0.05}, {"gate_margin": 0.2}, {"teacher_mix": 0.5},
])
def test_every_knob_produces_a_usable_plan(change):
    tc = trial_commands(change, "runs", "teacher.pt", tag="t")
    assert tc["commands"]["gen"] and tc["commands"]["train"]
