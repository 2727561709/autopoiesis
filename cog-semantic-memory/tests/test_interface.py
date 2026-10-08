"""M08 接口契约测试。"""
import inspect

from cog_semantic import Concept, Link, SemanticError, SemanticMemory


def test_api_surface():
    for name in ("add_concept", "link", "get", "neighbors", "spread",
                 "state_dict", "load_state_dict", "__len__",
                 "__contains__"):
        assert hasattr(SemanticMemory, name)


def test_signatures():
    assert list(inspect.signature(SemanticMemory.add_concept).parameters
                )[:2] == ["self", "cid"]
    assert list(inspect.signature(SemanticMemory.link).parameters
                )[:3] == ["self", "src", "dst"]
    assert list(inspect.signature(SemanticMemory.spread).parameters
                )[:3] == ["self", "seeds", "depth"]


def test_concept_contract():
    mem = SemanticMemory()
    c = mem.add_concept("food", kind="resource", energy=0.8)
    assert isinstance(c, Concept)
    assert c.get("energy") == 0.8
    assert c.get("missing", "dft") == "dft"
    assert "food" in mem and len(mem) == 1


def test_link_contract():
    mem = SemanticMemory()
    mem.add_concept("a"); mem.add_concept("b")
    lk = mem.link("a", "b", "near", 0.5)
    assert isinstance(lk, Link)
    assert (lk.src, lk.dst, lk.relation, lk.weight) == ("a", "b", "near", 0.5)


def test_spread_returns_activation():
    mem = SemanticMemory()
    mem.add_concept("a"); mem.add_concept("b")
    mem.link("a", "b", weight=1.0)
    act = mem.spread(["a"])
    assert isinstance(act, dict)                # Activation 即 Dict[str, float]
    assert act["a"] == 1.0 and act["b"] > 0.0


def test_error_type():
    assert issubclass(SemanticError, ValueError)
