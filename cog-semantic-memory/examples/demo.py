"""M08 最小可运行示例：python examples/demo.py

"饥饿的智能体想起什么？"—— 扩散激活沿知识图谱传播。
"What does a hungry agent recall?" — spreading activation over the
knowledge graph.
"""
from cog_semantic import SemanticMemory

if __name__ == "__main__":
    mem = SemanticMemory(decay=0.5, threshold=0.05)

    for cid in ["hungry", "thirsty", "food", "berry", "bush",
                "water", "pond", "nest", "fox", "hazard"]:
        mem.add_concept(cid)
    mem.link("hungry", "food", "motivates", 1.0)
    mem.link("thirsty", "water", "motivates", 1.0)
    mem.link("food", "berry", "is-a", 0.8)
    mem.link("berry", "bush", "grows-on", 0.9)
    mem.link("water", "pond", "is-a", 0.9)
    mem.link("pond", "bush", "near", 0.6, bidirectional=True)
    mem.link("nest", "bush", "near", 0.7)
    mem.link("fox", "hazard", "is-a", 0.9)
    mem.link("fox", "bush", "lurks-near", 0.8)

    # 1) 只饿：想起食物链，顺带觅食地点
    print("seeds=[hungry]")
    for cid, a in mem.spread(["hungry"], depth=3).items():
        print(f"  {cid:8s} {a:.3f}")

    # 2) 又饿又怕：两条动机流叠加（bush 被双重唤起，风险浮现）
    print("seeds={hungry:1.0, fox:0.8}")
    for cid, a in mem.spread({"hungry": 1.0, "fox": 0.8}, depth=3).items():
        print(f"  {cid:8s} {a:.3f}")
