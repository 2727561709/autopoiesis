"""训练循环冒烟测试：微型配置跑 2 个 update，产物与损失均合法。"""
import json
from pathlib import Path

import pytest

from homeo_rl.env import EnvConfig
from homeo_rl.train import PPOConfig, train


def test_train_smoke(tmp_path: Path):
    cfg = PPOConfig(updates=2, num_envs=2, rollout=8, minibatch=8,
                    epochs=1, d_model=32, nhead=2, n_layers=1, ff_dim=64,
                    seed=0)
    env_cfg = EnvConfig(max_steps=20, seed=0)
    metrics_path = train(cfg, env_cfg=env_cfg, out_dir=tmp_path, device="cpu")
    assert metrics_path.exists()
    rows = [json.loads(l) for l in metrics_path.read_text().splitlines()]
    assert len(rows) == 2
    for r in rows:
        assert r["update"] in (1, 2)
        assert r["steps"] > 0
        for k in ("pg_loss", "v_loss", "wm_loss", "entropy"):
            assert r[k] == r[k]  # 非 NaN
        assert r["pe_ema"] > 0
    ckpt = tmp_path / "checkpoint.pt"
    assert ckpt.exists()
