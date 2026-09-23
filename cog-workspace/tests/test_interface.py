"""M14 接口契约测试。"""
import inspect

import numpy as np

from cog_workspace import SOURCE_NAMES, Workspace


def test_api_surface():
    for name in ("forward", "__call__"):
        assert hasattr(Workspace, name)


def test_forward_signature_matches_spec():
    sig = inspect.signature(Workspace.forward)
    params = list(sig.parameters)
    assert params[1:5] == ["perc", "mem", "emo", "goal"]


def test_source_names_contract():
    assert SOURCE_NAMES == ("perc", "mem", "emo", "goal")


def test_broadcast_contract():
    ws = Workspace({"perc": 8}, broadcast_dim=16, seed=0)
    b, w = ws(perc=np.ones(8))
    assert isinstance(b, np.ndarray) and b.shape == (16,)
    assert isinstance(w, dict) and set(w) <= set(SOURCE_NAMES)
