"""M12 最小可运行示例：python examples/demo.py"""
from cog_drives import DRIVE_NAMES, DriveSystem

if __name__ == "__main__":
    drives = DriveSystem(decay=0.99)

    for t in range(200):
        effects = {}
        if t % 50 == 0:
            effects["energy"] = +0.4   # 周期性进食
        drives.step(effects)
        if drives.in_crisis():
            print(f"t={t} 危机! {drives.in_crisis()}")

    print("最终驱动值:", {k: round(v, 3) for k, v in drives.values.items()})
    print("缺口      :", {k: round(v, 3) for k, v in drives.deficit().items()})
    print("to_tensor :", drives.to_tensor().round(3), "顺序:", DRIVE_NAMES)
