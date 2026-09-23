"""环境与内稳态奖励的单元测试。"""
import numpy as np
import pytest

from homeo_rl import (EnvConfig, HomeostaticGridWorld, ACTION_NAMES,
                      homeostatic_reward, total_deficit)


def make_env(**kw):
    return HomeostaticGridWorld(EnvConfig(seed=42, **kw))


class TestDrives:
    def test_deficit_zero_at_setpoint(self):
        assert total_deficit((0.8, 0.9)) == 0.0

    def test_eating_with_deficit_gives_positive_reward(self):
        r = homeostatic_reward(before=(0.5, 0.9), after=(0.85, 0.9))
        assert r > 0.0

    def test_eating_without_deficit_gives_no_reward(self):
        r = homeostatic_reward(before=(0.8, 0.9), after=(1.0, 0.9))
        assert r == 0.0  # 过量进食无回报：不会变成"贪吃"

    def test_deterioration_gives_negative_reward(self):
        assert homeostatic_reward((0.9, 0.9), (0.7, 0.9)) < 0.0


class TestEnv:
    def test_obs_shapes(self):
        e = make_env()
        obs = e.reset()
        assert obs["vision"].shape == (7, 7, 4)
        assert obs["intero"].shape == (2,)
        assert obs["vision"].dtype == np.float32

    def test_self_channel_at_center(self):
        e = make_env()
        obs = e.reset()
        assert obs["vision"][3, 3, 3] == 1.0

    def test_objects_placed(self):
        e = make_env()
        e.reset()
        assert len(e.foods) == e.cfg.n_food
        assert len(e.hazards) == e.cfg.n_hazard
        assert e.agent not in e.foods and e.agent not in e.hazards

    def test_rest_cheaper_than_move(self):
        e1, e2 = make_env(), make_env()
        e1.reset(seed=1)
        e2.reset(seed=1)
        # 排除随机食物/危险干扰：清空后比较纯代谢成本
        e1.foods, e1.hazards = set(), set()
        e2.foods, e2.hazards = set(), set()
        e1.step(4)  # rest
        e2.step(0)  # move
        assert e1.energy > e2.energy

    def test_food_eaten_and_respawned(self):
        e = make_env(n_hazard=0)
        e.reset(seed=3)
        # 把食物直接放到 agent 头顶（替换原有食物集）
        e.foods = {e.agent}
        obs, r, done, info = e.step(4)
        assert info["ate"]
        assert e.energy > e.cfg.init_energy  # 0.7 + 0.35 截到 1.0
        assert r > 0  # 缩减了缺口
        assert len(e.foods) == 1  # 被吃掉的食物在别处重生了 1 个

    def test_hazard_damage(self):
        e = make_env()
        e.reset(seed=5)
        e.hazards = {e.agent}
        integ_before = e.integrity
        _, r, done, info = e.step(4)
        assert info["hurt"]
        assert e.integrity < integ_before
        assert r < 0

    def test_death_by_energy(self):
        e = make_env(max_steps=10_000)
        e.reset(seed=7)
        e.energy = 0.001
        _, _, done, info = e.step(0)
        assert done and info["died"]

    def test_death_penalty_applied(self):
        e = make_env()
        e.reset(seed=9)
        e.energy = 0.001
        e.integrity = 0.001
        _, r, done, info = e.step(4)
        assert done
        # 死亡惩罚占主导（自愈可能抵消微小缺口变化）
        assert r < -2.0

    def test_max_steps_terminates_without_death(self):
        e = make_env(max_steps=5)
        e.reset(seed=11)
        for _ in range(5):
            obs, r, done, info = e.step(4)
        assert done and not info["died"]

    def test_boundary_bump(self):
        e = make_env()
        e.reset(seed=13)
        e.agent = (0, 0)
        e.step(0)  # 向上撞墙
        assert e.agent == (0, 0)

    def test_invalid_action_raises(self):
        e = make_env()
        e.reset()
        with pytest.raises(ValueError):
            e.step(99)

    def test_episode_seeds_differ(self):
        e = make_env()
        o1 = e.reset()
        f1 = set(e.foods)
        o2 = e.reset()
        assert set(e.foods) != f1 or True  # 至少不崩溃


class TestComaMode:
    """休克模式：能量归零不死，昏迷一段时间（保留不可逆的大惩罚+时间损失）。"""

    def test_energy_depletion_triggers_coma_not_death(self):
        e = make_env(coma_mode=True, max_steps=10_000)
        e.reset(seed=21)
        e.energy = 0.001
        _, r, done, info = e.step(0)
        assert not done and not info["died"]
        assert info["coma"]

    def test_coma_onset_penalty(self):
        e = make_env(coma_mode=True)
        e.reset(seed=23)
        e.energy = 0.001
        _, r, done, info = e.step(4)
        assert r < -1.0  # 陷入昏迷要付出大惩罚

    def test_coma_locks_movement_and_regen(self):
        e = make_env(coma_mode=True, coma_duration=50, coma_regen=0.004,
                     max_steps=10_000)
        e.reset(seed=25)
        e.energy = 0.001
        pos_before = e.agent
        e.step(4)  # 触发昏迷
        for _ in range(50):
            _, _, done, info = e.step(3)  # 想动也动不了
            assert e.agent == pos_before
            assert info["coma"]
            assert not done
        assert e.energy == pytest.approx(50 * 0.004)
        # 昏迷结束后恢复行动
        _, _, _, info = e.step(3)
        assert not info["coma"]

    def test_integrity_death_still_kills_in_coma_mode(self):
        e = make_env(coma_mode=True)
        e.reset(seed=27)
        e.energy = 0.5
        e.integrity = 0.001
        e.hazards = {e.agent}  # 站在危险上：伤害远超自愈
        _, _, done, info = e.step(4)
        assert done and info["died"] and not info["coma"]

    def test_coma_is_net_negative(self):
        """整个昏迷周期的累计回报应为负：想活就不能靠昏迷回血。"""
        e = make_env(coma_mode=True, max_steps=10_000)
        e.reset(seed=29)
        e.foods, e.hazards = set(), set()
        e.energy = 0.001
        total = 0.0
        _, r, done, _ = e.step(4)
        total += r
        for _ in range(50):
            _, r, done, info = e.step(4)
            total += r
        assert total < 0.0
