"""智能体 Harness 的单元测试（假模型，CPU 毫秒级）。
Unit tests for the agent harness (fake model, milliseconds on CPU).

测的是**接线**而非模型质量：批量形状对不对、动作是否落在 5 个里、
已结束的局是否不再进批、轨迹长度是否与步数一致。
These test the WIRING, not model quality: batch shapes, actions landing
inside the 5, finished episodes leaving the batch, trace length matching
the step count.
"""
import pytest
import torch

from homeo_lm.runner import BatchRunner, Step, Trace
from tests.fakes import ACTION_FIRST_TOKENS, FakeTokenizer, TinyCausalModel


class TestStepAndTrace:
    def test_trace_derives_length_and_illegal_rate(self):
        t = Trace(seed=1)
        t.steps = [Step(t=0, action=1, legal=True),
                   Step(t=1, action=2, legal=False)]
        assert t.length == 2 and t.illegal == 1
        assert t.illegal_rate == pytest.approx(0.5)

    def test_empty_trace_is_safe(self):
        t = Trace(seed=1)
        assert t.length == 0 and t.illegal_rate == 0.0
        assert t.died is False and t.survived is True

    def test_died_comes_from_the_last_step(self):
        t = Trace(seed=1)
        t.steps = [Step(t=0, action=1, legal=True),
                   Step(t=1, action=1, legal=True, done=True, died=True)]
        assert t.died is True and t.survived is False


class TestBatchRunnerDecide:
    """批量决策：一次前向出一批动作，且必须落在 5 个动作里。
    Batched decisions: one forward pass per batch, and every action must
    land inside the 5."""

    def _runner(self, bias=0, greedy=True, batch_size=4):
        torch.manual_seed(0)
        model, tok = TinyCausalModel(bias_action=bias), FakeTokenizer()
        return BatchRunner(model, tok, "cpu", greedy=greedy,
                           batch_size=batch_size)

    def test_actions_are_in_range(self):
        r = self._runner()
        acts = r.decide([[2, 3, 4]] * 8)
        assert len(acts) == 8
        assert all(0 <= int(a) < len(ACTION_FIRST_TOKENS) for a in acts)

    def test_greedy_picks_the_biased_action(self):
        r = self._runner(bias=3, greedy=True)
        acts = r.decide([[2, 3, 4]] * 6)
        assert all(int(a) == 3 for a in acts)

    def test_batch_size_is_respected(self):
        r = self._runner(batch_size=3)
        acts = r.decide([[2, 3, 4]] * 7)
        assert len(acts) == 7

    def test_custom_driver_overrides_the_model(self):
        r = self._runner()
        r.driver = lambda prompts: [4] * len(prompts)
        assert all(int(a) == 4 for a in r.decide([[2, 3, 4]] * 5))


class TestRollout:
    """轨迹推进：跑完的局留在轨迹里，且不再接受前向。
    Rollout: finished episodes stay in the trace and receive no further
    forward passes."""

    def test_returns_one_trace_per_episode(self, env_cfg):
        torch.manual_seed(0)
        runner = BatchRunner(TinyCausalModel(), FakeTokenizer(), "cpu")
        traces = runner.rollout(env_cfg, episodes=6, seed=9_000)
        assert len(traces) == 6

    def test_trace_length_matches_the_steps_taken(self, env_cfg):
        torch.manual_seed(0)
        runner = BatchRunner(TinyCausalModel(), FakeTokenizer(), "cpu")
        traces = runner.rollout(env_cfg, episodes=4, seed=9_000)
        for tr in traces:
            assert [s.t for s in tr.steps] == list(range(tr.length))
            assert tr.length <= env_cfg.max_steps

    def test_reported_seeds_are_distinct(self, env_cfg):
        torch.manual_seed(0)
        runner = BatchRunner(TinyCausalModel(), FakeTokenizer(), "cpu")
        traces = runner.rollout(env_cfg, episodes=5, seed=9_000)
        assert len({t.seed for t in traces}) == 5

    def test_summary_aggregates(self, env_cfg):
        torch.manual_seed(0)
        runner = BatchRunner(TinyCausalModel(), FakeTokenizer(), "cpu")
        traces = runner.rollout(env_cfg, episodes=6, seed=9_000)
        s = BatchRunner.summary(traces)
        assert s["episodes"] == 6
        assert 0.0 <= s["survival_rate"] <= 1.0
        assert 0.0 <= s["illegal_rate"] <= 1.0

    def test_summary_of_nothing_is_safe(self):
        assert BatchRunner.summary([]) == {"episodes": 0}

    def test_driver_can_be_a_random_walk(self, env_cfg):
        """驱动可以是随机的——L1 的"随机轨迹"来源就靠这个口子。
        The driver may be random: that is the hook L1's random-rollout
        source uses."""
        torch.manual_seed(0)
        runner = BatchRunner(TinyCausalModel(), FakeTokenizer(), "cpu")
        runner.decide = lambda prompts: [i % 5 for i in range(len(prompts))]
        traces = runner.rollout(env_cfg, episodes=3, seed=1)
        assert len(traces) == 3
