"""M15 最小可运行示例：python examples/demo.py"""
from cog_goal_manager import Goal, GoalManager

if __name__ == "__main__":
    gm = GoalManager(max_goals=5)

    # 从驱动缺口生成目标（对接 M12 DriveSystem.deficit() 的输出格式）
    deficits = {"energy": 0.45, "integrity": 0.02, "social": 0.7, "curiosity": 0.3}
    generated = gm.update_from_drives(deficits)
    print("生成/更新:", [(g.name, round(g.priority, 2)) for g in generated])

    # 手动添加长期目标
    gm.add(Goal(name="explore_world", priority=0.35, deadline=1000))

    # 注意焦点
    for step in [0, 3, 6]:
        goal = gm.current(step=step)
        print(f"step={step} 当前目标: {goal.name if goal else None} "
              f"(priority={goal.priority:.2f})" if goal else f"step={step} 无目标")
        if step == 3 and goal:
            gm.complete(goal.name)
            print(f"  -> 完成 {goal.name}")
