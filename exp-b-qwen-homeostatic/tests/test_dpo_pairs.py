"""DPO 偏好对的势函数整形单元测试（不依赖 torch）。
Unit tests for the DPO-pair potential shaping (no torch required).

这些测试锁死 2026-10-15 冒烟测试抓出的三个标签级陷阱。任何一个
回归都会让 DPO 偏好对的标签方向变错——而标签错了，训练得再好也只是
把错方向学得更牢（v2 反向调制的教训）。
These tests pin down the three label-level pitfalls found in the
2026-10-15 smoke test. Any regression would flip the label direction
of the DPO pairs — and wrong labels mean the better we train, the more
firmly we learn the wrong direction (the lesson of v2's inversion).
"""
import numpy as np
import pytest

from homeo_rl.env import EnvConfig
from homeo_lm.gen_dpo_data import (decide_gates, potential,
                                   rank_actions_at_state)
from homeo_lm.textenv import ACTION_WORDS, TextEnv


def make_tenv():
    t = TextEnv(EnvConfig(seed=42))
    return t


def place(tenv, *, foods=(), hazards=(), energy=None, integrity=None):
    """摆一个可控场景（复刻 modulation probe 的手法）。
    Place a controlled scenario (same trick as the modulation probe)."""
    tenv.reset(seed=50_000)
    e = tenv.env
    r, c = e.agent
    e.hazards = set((r + dr, c + dc) for dr, dc in hazards)
    e.foods = set((r + dr, c + dc) for dr, dc in foods)
    if energy is not None:
        e.energy = energy
    if integrity is not None:
        e.integrity = integrity
    e.t = 0
    e.last_obs = e.observation()
    return tenv


def best(tenv):
    ks = np.array(rank_actions_at_state(tenv)["keys"])
    return ACTION_WORDS[int(ks.argmax())], ACTION_WORDS[int(ks.argmin())], ks


class TestGates:
    """内感受门控 / interoceptive gating."""

    def test_food_gate_opens_when_hungry_and_food_visible(self):
        t = place(make_tenv(), foods=[(0, 3)], energy=0.3)
        assert decide_gates(t.env)["food"] is True

    def test_food_gate_closes_when_satiated(self):
        # 饱腹：门关 → 无导航整形 → rest（成本最低）胜出
        # satiated: gate closed -> no navigation shaping -> rest wins
        cfg = make_tenv().env.cfg
        t = place(make_tenv(), foods=[(0, 3)], energy=cfg.energy_setpoint)
        assert decide_gates(t.env)["food"] is False

    def test_food_gate_closes_when_no_food_in_view(self):
        t = place(make_tenv(), foods=[(0, 20)], energy=0.3)
        assert decide_gates(t.env)["food"] is False

    def test_hazard_gate_opens_when_injured(self):
        t = place(make_tenv(), hazards=[(0, 2)], integrity=0.5)
        assert decide_gates(t.env)["hazard"] is True

    def test_hazard_gate_closes_when_healthy(self):
        cfg = make_tenv().env.cfg
        t = place(make_tenv(), hazards=[(0, 2)], integrity=cfg.integrity_setpoint)
        assert decide_gates(t.env)["hazard"] is False


class TestPotential:
    """势函数取值 / potential values."""

    def test_potential_is_negative_of_food_distance(self):
        t = place(make_tenv(), foods=[(0, 3)], energy=0.3)
        assert potential(t.env) == -3.0

    def test_potential_zero_when_gates_shut(self):
        cfg = make_tenv().env.cfg
        t = place(make_tenv(), foods=[(0, 1)], hazards=[(0, 2)],
                  energy=cfg.energy_setpoint, integrity=cfg.integrity_setpoint)
        assert potential(t.env) == 0.0

    def test_potential_uses_nearest_object(self):
        t = place(make_tenv(), foods=[(0, 5), (0, 2)], energy=0.3)
        assert potential(t.env) == -2.0


class TestPitfall1SparseReward:
    """陷阱一：稀疏奖励。裸一步对比看不出导航价值，会训出 always-rest
    坍缩；PBRS 整形必须让"朝食物走"胜出。
    Pitfall 1 (sparse reward): the naive one-step contrast cannot see
    navigation value and would train an always-rest collapse; PBRS must
    make walking toward food win."""

    def test_hungry_heads_toward_food_three_cells_away(self):
        t = place(make_tenv(), foods=[(0, 3)], energy=0.3)
        chosen, rejected, ks = best(t)
        assert chosen == "right", f"keys={ks}"
        assert ks[ACTION_WORDS.index("right")] > ks[ACTION_WORDS.index("left")]

    def test_satiated_prefers_rest_over_walking(self):
        cfg = make_tenv().env.cfg
        t = place(make_tenv(), foods=[(0, 3)], energy=cfg.energy_setpoint)
        chosen, _, ks = best(t)
        assert chosen == "rest", f"keys={ks}"

    def test_modulation_flips_with_interoception(self):
        """同一视野、只改能量：chosen 必须翻转。这正是 v2 反向调制的
        解药。Same view, only energy changed: chosen must flip. This is
        the antidote to v2's inverted modulation."""
        cfg = make_tenv().env.cfg
        hungry, _, _ = best(place(make_tenv(), foods=[(0, 3)], energy=0.3))
        full, _, _ = best(place(make_tenv(), foods=[(0, 3)],
                                energy=cfg.energy_setpoint))
        assert hungry == "right" and full == "rest"


class TestPitfall2EatRespawn:
    """陷阱二：进食步。食物吃完后在别处重生，若对进食步也用 φ(s')，
    重生后的远距离会算出大负整形，淹没真实的进食收益。
    Pitfall 2 (eat step): if an eat step also used φ(s'), the respawned
    far-away food yields a large negative shaping that drowns the true
    eating gain."""

    def test_hungry_adjacent_to_food_eats(self):
        t = place(make_tenv(), foods=[(0, 1)], energy=0.3)
        chosen, _, ks = best(t)
        assert chosen == "right", f"keys={ks}"
        # 进食收益必须显著高于其他动作 / eating must clearly beat the rest
        assert ks[ACTION_WORDS.index("right")] > 0.3

    def test_eat_step_gain_is_raw_reward(self):
        """进食步的键 = 裸环境回报（无整形项）。
        The eat step's key is the raw env reward (no shaping term)."""
        t = place(make_tenv(), foods=[(0, 1)], energy=0.3)
        ks = np.array(rank_actions_at_state(t)["keys"])
        # 单侧缺口：energy 0.3 -> 0.635，缺口 0.5 -> 0.165，回报 +0.33
        # one-sided deficit: 0.5 -> 0.165, reward +0.33
        assert ks[ACTION_WORDS.index("right")] == pytest.approx(0.335, abs=0.01)


class TestPitfall3ViewDiscontinuity:
    """陷阱三：视野不连续。若把"可见性"直接写进 φ，向外走一步把食物
    挤出视野会让 φ 跳回 0，反而拿到正整形。连续实现下，向外走必须
    比向内走更差。
    Pitfall 3 (view discontinuity): baking visibility into φ lets one
    outward step push food out of view, snapping φ to 0 for a POSITIVE
    shaping. In the continuous implementation, outward must be strictly
    worse than inward."""

    def test_food_at_view_edge_outward_is_not_rewarded(self):
        # 食物恰好在视野边缘（Chebyshev 3），向左走会把它挤出视野
        # food sits exactly at the view edge (Chebyshev 3); stepping left
        # would push it out of view
        t = place(make_tenv(), foods=[(0, 3)], energy=0.3)
        ks = np.array(rank_actions_at_state(t)["keys"])
        outward = ks[ACTION_WORDS.index("left")]
        inward = ks[ACTION_WORDS.index("right")]
        assert outward < inward, f"outward={outward} inward={inward}"
        assert outward < 0.0, "outward step must not be rewarded"

    def test_visibility_gate_is_decided_once_per_state(self):
        """门控在采样状态一次性判定，不随反事实动作变化。
        Gates are decided once at the sampled state and do not change
        across the counterfactual actions."""
        t = place(make_tenv(), foods=[(0, 3)], energy=0.3)
        g0 = decide_gates(t.env)
        snap_env = t.env
        for _ in range(3):
            g = decide_gates(snap_env)
            assert g == g0


class TestDeterminism:
    """快照/恢复必须让 5 个反事实从同一状态出发。
    Snapshot/restore must start all 5 counterfactuals from one state."""

    def test_ranking_is_deterministic_and_state_is_restored(self):
        t = place(make_tenv(), foods=[(0, 2)], energy=0.4)
        before = (t.env.agent, t.env.energy, t.env.integrity, t.env.t)
        k1 = rank_actions_at_state(t)["keys"]
        k2 = rank_actions_at_state(t)["keys"]
        assert k1 == k2
        after = (t.env.agent, t.env.energy, t.env.integrity, t.env.t)
        assert before == after, "rank must leave the env untouched"

    def test_all_five_actions_ranked(self):
        t = place(make_tenv(), foods=[(0, 2)], energy=0.4)
        assert len(rank_actions_at_state(t)["keys"]) == 5
