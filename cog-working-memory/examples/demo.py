"""M06 最小可运行示例：python examples/demo.py"""
import numpy as np

from cog_working_memory import WorkingMemory

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    wm = WorkingMemory(capacity=4, dim=8)

    # 连续写入 6 条（容量 4，最早的被驱逐）
    for i in range(6):
        z = np.eye(8)[i]
        wm.write(z)
        print(f"write[{i}] len={len(wm)}")

    # 查询与第 5 条相同 => 应精确读出
    query = np.eye(8)[5]
    w = wm.attention(query)
    print("attention:", w.round(3))
    out = wm.read(query)
    print("read:", out.round(3))
    print("state:", wm.state())
