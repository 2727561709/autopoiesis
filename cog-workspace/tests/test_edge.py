"""M14 边界条件测试。"""
import numpy as np
import pytest

from cog_workspace import Workspace, WorkspaceError


def test_no_active_sources_raises():
    ws = Workspace({"perc": 8}, broadcast_dim=16, seed=0)
    with pytest.raises(WorkspaceError, match="no active sources"):
        ws()


def test_unconfigured_source_raises():
    ws = Workspace({"perc": 8}, broadcast_dim=16, seed=0)
    with pytest.raises(WorkspaceError, match="not configured"):
        ws(perc=np.ones(8), mem=np.ones(8))


def test_wrong_source_dim_raises():
    ws = Workspace({"perc": 8, "emo": 4}, broadcast_dim=16, seed=0)
    with pytest.raises(WorkspaceError, match="expected dim"):
        ws(perc=np.ones(9))
    with pytest.raises(WorkspaceError, match="expected dim"):
        ws(emo=np.ones(3))


def test_nan_source_raises():
    ws = Workspace({"perc": 8}, broadcast_dim=16, seed=0)
    with pytest.raises(WorkspaceError, match="NaN"):
        ws(perc=np.full(8, np.nan))


def test_invalid_constructor():
    with pytest.raises(WorkspaceError):
        Workspace({}, broadcast_dim=16)
    with pytest.raises(WorkspaceError):
        Workspace({"perc": 8}, broadcast_dim=0)
    with pytest.raises(WorkspaceError, match="unknown source"):
        Workspace({"hunger": 8})
    with pytest.raises(WorkspaceError, match="positive"):
        Workspace({"perc": 0})


def test_invalid_temperature():
    ws = Workspace({"perc": 8}, broadcast_dim=16, seed=0)
    with pytest.raises(WorkspaceError, match="temperature"):
        ws(perc=np.ones(8), temperature=0.0)
    with pytest.raises(WorkspaceError, match="temperature"):
        ws(perc=np.ones(8), temperature=-1.0)


def test_extreme_inputs_stable():
    ws = Workspace({"perc": 8, "goal": 8}, broadcast_dim=16, seed=0)
    b, w = ws(perc=np.full(8, 1e8), goal=np.full(8, -1e8))
    assert np.all(np.isfinite(b))
    assert sum(w.values()) == pytest.approx(1.0)


def test_two_sources_only():
    ws = Workspace({"mem": 8, "goal": 8}, broadcast_dim=16, seed=0)
    b, w = ws(mem=np.ones(8), goal=np.zeros(8))
    assert set(w) == {"mem", "goal"}
    assert b.shape == (16,)
