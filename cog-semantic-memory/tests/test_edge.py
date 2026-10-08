"""M08 边界与防御测试。"""
import pytest

from cog_semantic import SemanticMemory, SemanticError


def test_bad_constructor():
    with pytest.raises(SemanticError):
        SemanticMemory(decay=0.0)
    with pytest.raises(SemanticError):
        SemanticMemory(decay=1.5)
    with pytest.raises(SemanticError):
        SemanticMemory(threshold=-0.1)


def test_bad_concept_id():
    mem = SemanticMemory()
    with pytest.raises(SemanticError):
        mem.add_concept("")
    with pytest.raises(SemanticError):
        mem.add_concept(123)


def test_link_unknown_concepts():
    mem = SemanticMemory()
    mem.add_concept("a")
    with pytest.raises(SemanticError):
        mem.link("a", "ghost")
    with pytest.raises(SemanticError):
        mem.link("ghost", "a")


def test_self_loop_rejected():
    mem = SemanticMemory()
    mem.add_concept("a")
    with pytest.raises(SemanticError):
        mem.link("a", "a")


def test_bad_weight():
    mem = SemanticMemory()
    mem.add_concept("a"); mem.add_concept("b")
    with pytest.raises(SemanticError):
        mem.link("a", "b", weight=0.0)
    with pytest.raises(SemanticError):
        mem.link("a", "b", weight=1.5)


def test_get_unknown():
    mem = SemanticMemory()
    with pytest.raises(SemanticError):
        mem.get("nope")
    with pytest.raises(SemanticError):
        mem.neighbors("nope")


def test_spread_bad_inputs():
    mem = SemanticMemory()
    mem.add_concept("a")
    with pytest.raises(SemanticError):
        mem.spread(["ghost"])
    with pytest.raises(SemanticError):
        mem.spread([])
    with pytest.raises(SemanticError):
        mem.spread({"a": -1.0})                # 负激活
    with pytest.raises(SemanticError):
        mem.spread({"a": float("nan")})
    with pytest.raises(SemanticError):
        mem.spread(["a"], depth=-1)


def test_add_concept_update_merges_attrs():
    mem = SemanticMemory()
    mem.add_concept("x", color="red")
    mem.add_concept("x", weight_kg=2)          # 幂等：更新而非报错
    assert mem.get("x").get("color") == "red"
    assert mem.get("x").get("weight_kg") == 2
    assert len(mem) == 1
