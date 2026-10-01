"""文本环境的单元测试（不依赖 torch / transformers）。"""
import numpy as np
import pytest

from homeo_rl.env import EnvConfig
from homeo_lm.textenv import (ACTION_WORDS, WORD_TO_ACTION, TextEnv,
                              render_grid, render_obs)


def make_tenv(**kw):
    return TextEnv(EnvConfig(seed=42, **kw))


class TestRender:
    def test_grid_is_square_rows_of_view_size(self):
        t = make_tenv()
        t.reset()
        rows = render_grid(t.env)
        assert len(rows) == 7
        assert all(len(r) == 7 for r in rows)

    def test_self_at_center(self):
        t = make_tenv()
        t.reset()
        rows = render_grid(t.env)
        assert rows[3][3] == "O"

    def test_obs_contains_interoception(self):
        t = make_tenv()
        obs = t.reset()
        assert "energy:" in obs and "integrity:" in obs
        assert "previous action: none" in obs

    def test_prev_action_appears(self):
        t = make_tenv()
        t.reset()
        obs, _, _, _ = t.step("up")
        assert "previous action: up" in obs


class TestStep:
    def test_legal_word_moves(self):
        t = make_tenv()
        t.reset()
        r0, c0 = t.env.agent
        _, _, _, _ = t.step("down")
        assert t.env.agent == (r0 + 1, c0)

    def test_illegal_word_counts_as_rest(self):
        t = make_tenv()
        t.reset()
        e0 = t.env.energy
        obs, r, done, info = t.step("banana")
        # 原地不动但付休息成本 / stays in place but pays the rest cost
        assert t.env.energy < e0

    def test_all_words_accepted(self):
        t = make_tenv()
        t.reset()
        for w in ACTION_WORDS:
            _, _, done, _ = t.step(w)
            assert not done

    def test_word_lookup_covers_actions(self):
        assert len(WORD_TO_ACTION) == 5
        assert set(ACTION_WORDS) == set(WORD_TO_ACTION)
