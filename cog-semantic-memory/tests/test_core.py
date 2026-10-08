"""M08 核心行为测试：扩散激活传播 / 衰减 / 多种子叠加 / 存取。"""
import numpy as np

from cog_semantic import SemanticMemory


def _food_graph():
    """食物-地点小图谱。A small food-place graph."""
    mem = SemanticMemory(decay=0.5, threshold=0.05)
    for cid in ["hungry", "food", "berry", "bush", "water", "pond",
                "nest"]:
        mem.add_concept(cid)
    mem.link("hungry", "food", "motivates", 1.0)
    mem.link("food", "berry", "is-a", 0.8)
    mem.link("berry", "bush", "grows-on", 0.9)
    mem.link("water", "pond", "is-a", 0.9)
    mem.link("pond", "bush", "near", 0.6, bidirectional=True)
    mem.link("nest", "bush", "near", 0.7)
    return mem


def test_spreading_reaches_neighbors():
    act = _food_graph().spread(["hungry"], depth=3)
    assert act["hungry"] == 1.0
    assert act["food"] > act["berry"] > act["bush"] > 0.0   # 逐跳衰减


def test_spreading_respects_weight():
    mem = SemanticMemory(decay=0.9, threshold=0.01)
    mem.add_concept("s"); mem.add_concept("weak"); mem.add_concept("strong")
    mem.link("s", "weak", weight=0.2)
    mem.link("s", "strong", weight=1.0)
    act = mem.spread(["s"], depth=1)
    assert act["strong"] > act["weak"]       # 权重调制传播强度


def test_multiple_seeds_sum():
    mem = SemanticMemory(decay=1.0, threshold=0.0)
    mem.add_concept("a"); mem.add_concept("b"); mem.add_concept("x")
    mem.link("a", "x", weight=0.5)
    mem.link("b", "x", weight=0.5)
    act = mem.spread({"a": 1.0, "b": 1.0}, depth=1)
    assert act["x"] >= 0.99                  # 双种子叠加 ≈ 1.0


def test_bidirectional_link():
    mem = SemanticMemory(decay=0.5)
    mem.add_concept("u"); mem.add_concept("v")
    mem.link("u", "v", "near", 0.8, bidirectional=True)
    assert set(mem.neighbors("u")) == {"v"}
    assert set(mem.neighbors("v")) == {"u"}   # 反向边生效


def test_threshold_pruning():
    mem = SemanticMemory(decay=0.2, threshold=0.15)
    mem.add_concept("s"); mem.add_concept("far1"); mem.add_concept("far2")
    mem.link("s", "far1", weight=0.9)        # 0.2*0.9=0.18 过阈值
    mem.link("far1", "far2", weight=0.9)     # 再一跳被剪
    act = mem.spread(["s"], depth=3)
    assert "far1" in act and "far2" not in act


def test_depth_zero():
    mem = _food_graph()
    act = mem.spread(["hungry"], depth=0)
    assert set(act) == {"hungry"}             # 不传播


def test_activation_sorted_descending():
    act = _food_graph().spread(["hungry"], depth=4)
    vals = list(act.values())
    assert vals == sorted(vals, reverse=True)


def test_pure_recency_not_needed_decay_one_chain():
    mem = SemanticMemory(decay=1.0, threshold=0.0)
    for c in "abcde":
        mem.add_concept(c)
    for a, b in zip("abcd", "bcde"):
        mem.link(a, b, weight=1.0)
    act = mem.spread(["a"], depth=10)
    assert all(act[c] == 1.0 for c in "abcde")   # 不衰减 => 全链等激活


def test_state_dict_roundtrip():
    mem = _food_graph()
    sd = mem.state_dict()
    mem2 = SemanticMemory()
    mem2.load_state_dict(sd)
    assert len(mem2) == len(mem)
    a1 = mem.spread(["hungry"], depth=3)
    a2 = mem2.spread(["hungry"], depth=3)
    assert a1 == a2                            # 行为完全一致
    assert mem2.neighbors("pond") == ["bush"]
