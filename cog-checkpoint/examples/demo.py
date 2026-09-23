"""M03 最小可运行示例：python examples/demo.py"""
from cog_checkpoint import Checkpoint

if __name__ == "__main__":
    state = {"step": 100, "memory": ["ep1", "ep2"], "drives": [0.8, 0.5]}
    meta = Checkpoint.save(state, "checkpoints/demo.ckpt",
                           module_versions={"cog-agent": "0.1.0"}, note="demo")
    print("saved:", meta.created_at)

    restored = Checkpoint.load("checkpoints/demo.ckpt")
    assert restored == state
    print("restored:", restored)

    print("latest:", Checkpoint.latest("checkpoints"))
