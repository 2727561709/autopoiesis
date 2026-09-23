"""M25 最小可运行示例：python examples/demo.py"""
from cog_body import Body

if __name__ == "__main__":
    body = Body(position=(1.0, 1.0), arena_size=10.0)

    for t in range(5):
        effects = body.step({"move": (1.0, 0.5)})
        print(f"t={t} pos={body.position} energy={body.energy:.4f} effects={effects}")

    # 受伤 + 休息
    print(body.step({"damage": 0.3, "rest": True}))
    print("state:", body.state())
