"""pytest 脚手架：把假件与常用固定物挂成 fixture。
pytest scaffolding: expose the fakes and common fixtures.

新模块只要注入 `student` / `tok` 就能测**真走前向**的接线，不必载 0.5B
模型——这是"测试 Harness"存在的全部理由。
A new module can inject `student` / `tok` and test wiring that really
runs a forward pass, without loading a 0.5B model — the whole reason
this test harness exists.

预置算例 / built-in cases:
    env_cfg    —— 与探针一致的环境配置
    student    —— (model, tok) 假学生（动作偏好可指定）
    ledger     —— 指向 tmp_path 的空账本
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def env_cfg():
    from homeo_rl.env import EnvConfig
    return EnvConfig()


@pytest.fixture
def tok():
    from tests.fakes import FakeTokenizer
    return FakeTokenizer()


@pytest.fixture
def student():
    """默认假学生（偏向动作 0）。Default fake student (biased to action 0)."""
    from tests.fakes import fake_student
    model, tok = fake_student(bias_action=0)
    return model, tok


@pytest.fixture
def ledger(tmp_path):
    from homeo_lm.rsi import Ledger
    return Ledger(tmp_path / "rsi_ledger.jsonl")


@pytest.fixture
def healthy_metrics():
    """一份"看起来健康"的指标（用于诊断不该点火的用例）。
    A healthy-looking metric set (for cases where nothing should fire)."""
    return {"train_acc": 0.95, "modulation_gap": 0.8, "sweep_slope": 0.4,
            "agreement": 0.9, "survival_rate": 0.9, "loss": 0.05,
            "margin": 0.3, "top_share": 0.3, "entropy": 1.1}


@pytest.fixture
def inverted_metrics():
    """v2 的真实失败形态：反向调制 + 活不下来 / v2's real failure:
    inverted modulation and no survival."""
    return {"train_acc": 0.97, "modulation_gap": -1.0, "sweep_slope": -0.3,
            "agreement": 0.49, "survival_rate": 0.0}
