"""模型单元测试：前向形状、分布合法性、世界模型损失可反传。"""
import numpy as np
import pytest
import torch

from homeo_rl import HomeoActorCritic, N_ACTIONS


def make_batch(model, bs=3):
    v = torch.rand(bs, 7, 7, 4)
    i = torch.rand(bs, 2)
    pa = torch.randint(0, N_ACTIONS + 1, (bs,))
    return v, i, pa


def make_model(**kw):
    kw.setdefault("d_model", 32)
    kw.setdefault("nhead", 2)
    kw.setdefault("n_layers", 1)
    kw.setdefault("ff_dim", 64)
    return HomeoActorCritic(**kw)


class TestModel:
    def test_forward_shapes(self):
        m = make_model()
        v, i, pa = make_batch(m)
        out = m(v, i, pa)
        assert out["logits"].shape == (3, N_ACTIONS)
        assert out["value"].shape == (3,)
        assert out["wm_vision"].shape == (3, 32)
        assert out["wm_intero"].shape == (3, 2)

    def test_probs_normalized(self):
        m = make_model()
        v, i, pa = make_batch(m)
        out = m(v, i, pa)
        p = torch.softmax(out["logits"], dim=-1)
        assert torch.allclose(p.sum(-1), torch.ones(3))
        assert (p > 0).all()

    def test_wm_targets_detached(self):
        m = make_model()
        v, i, _ = make_batch(m)
        tv, ti = m.wm_targets(v, i)
        assert not tv.requires_grad
        assert not ti.requires_grad

    def test_backward_finite(self):
        m = make_model()
        v, i, pa = make_batch(m)
        nv, ni, _ = make_batch(m)
        out = m(v, i, pa)
        tv, ti = m.wm_targets(nv, ni)
        loss = ((out["wm_vision"] - tv) ** 2).mean() \
            + ((out["wm_intero"] - ti) ** 2).mean()
        loss.backward()
        for p in m.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()
        assert torch.isfinite(loss)

    def test_act_no_grad(self):
        m = make_model()
        v, i, pa = make_batch(m)
        a, logp, val = m.act(v, i, pa)
        assert a.shape == (3,)
        assert not a.requires_grad

    def test_param_count_small(self):
        m = HomeoActorCritic()  # 默认尺寸
        n = sum(p.numel() for p in m.parameters())
        assert 100_000 < n < 5_000_000  # 阶段A定位：单卡可训的小模型
