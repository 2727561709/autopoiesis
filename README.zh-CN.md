# Autopoiesis（自创生）

**[English](README.md)** | 中文

*Autopoiesis*（希腊语"自创生"，Maturana & Varela）——能自己生产自己的系统。
本仓库的工程哲学：在学习模型中放置生命性的最小条件，然后让它自己生长。

按 24+3 模块方案实施。每个模块 = 一个独立项目（独立目录、独立 pyproject、独立测试、独立版本、独立文档），可随时拆分为独立 Git 仓库。模块之间只通过明确定义的接口通信。

## 方向演进：从 Agent 到"新物种大模型"

本仓库的定位已升级：认知模块（M01-M27）不再是最终交付物（运行时 agent 脚手架），
而是**新物种大模型的评测探针与课程生成器**。核心范式分三阶段：

- **阶段 A（进行中）** `exp-a-homeostatic-rl`：从零训练小 Transformer，
  唯一回报 = 内稳态缺口缩减（M12 思想）+ 预测误差内在奖励（世界模型）。
  验证"不给人工任务奖励，生命性是否涌现"。**依赖 torch，用 `py -3.12` 运行。**
- **阶段 B（规划）**：把内稳态目标 + 世界模型目标注入开源底座（后训练）。
- **阶段 C（规划）**：在线终身学习，训练/部署合一。

```powershell
cd exp-a-homeostatic-rl
py -3.12 -m pip install -e ".[dev]"
py -3.12 -m pytest
py -3.12 -m homeo_rl.train --updates 500 --out runs/base
py -3.12 -m homeo_rl.evaluate --ckpt runs/base/checkpoint.pt
```


## 当前进度

| 阶段 | 模块 | 状态 |
|---|---|---|
| 1 | M01 cog-config, M02 cog-logger, M03 cog-checkpoint | **已完成** |
| 2 | M04 cog-perception, M05 cog-policy, M12 cog-drives, M25 cog-body, M26 cog-world | **已完成（172 tests 全部通过）** |
| 3 | M06 cog-working-memory, M14 cog-workspace, M15 cog-goal-manager | **已完成（264 tests 全部通过）** |
| 4 | M10 世界模型, M16 规划器 | 待开发 |
| 5 | M07/M08/M09 三种记忆 | 待开发 |
| 6 | M13 情绪, M17 元认知, M18 自我叙事 | 待开发 |
| 7 | M19 扎根, M20 内在言语, M21 社会言语 | 待开发 |
| 8 | M11 因果图, M22 伦理, M23 可视化 | 待开发 |
| 9 | M27 多智能体社会 | 待开发 |
| 10 | M24 集成（cog-agent） | 持续 |

## 目录结构（每个模块相同）

```
cog-xxx/
├── README.md / CHANGELOG.md / docs/api.md
├── pyproject.toml
├── src/cog_xxx/{__init__.py, core.py, types.py}
├── tests/{test_core.py, test_interface.py, test_edge.py}
└── examples/demo.py
```

## 运行测试

```powershell
.\run_tests.ps1    # 逐模块独立运行（保持"独立测试"原则）
```

## 技术决策

- **NumPy 后端**：当前 Python 3.14 无稳定 torch，M04/M05 使用自实现微型网络
  （`nn.py`：Linear/Tanh/Sequential + 手写反向传播），接口与 torch 命名对齐
  （`forward/parameters/zero_grad/backward/sgd_step`），后续统一切换 torch 时上层零改动。
- **模块依赖**：M26 依赖 cog-body（智能体即 Body）；M12 直接消费 M25 的效果字典。

## 已完成模块速览

| 模块 | 版本 | 接口 | 测试 |
|---|---|---|---|
| cog-config | 0.1.0 | `Config.load(path) -> Config` | 24 |
| cog-logger | 0.1.0 | `log(event)` / `metric(name, value, step)` | 14 |
| cog-checkpoint | 0.1.0 | `Checkpoint.save/load` | 21 |
| cog-perception | 0.1.0 | `Encoder(obs) -> z`（tanh 有界，多模态） | 22 |
| cog-policy | 0.1.0 | `Policy(z) -> (DiagGaussian, value)` | 23 |
| cog-drives | 0.1.0 | `step(effects)` / `deficit()` / `to_tensor()` | 24 |
| cog-body | 0.1.0 | `step(action) -> effects` | 22 |
| cog-world | 0.1.0 | `observe(id)` / `step(id, action) -> (obs, reward)` | 22 |
| cog-working-memory | 0.1.0 | `write(z)` / `read(query) -> z`（FIFO + 余弦注意力） | 24 |
| cog-workspace | 0.1.0 | `forward(perc, mem, emo, goal) -> (broadcast, weights)` | 20 |
| cog-goal-manager | 0.1.0 | `add(goal)` / `update_from_drives()` / `current()`（零依赖） | 28 |

## 典型数据流（阶段 3 已可跑通）

```
World.observe -> Encoder -> z ──┬─► WorkingMemory.write(z) ─► read(query)
                                ├─► Workspace(perc=z, mem=..., emo=..., goal=...)
                                │     └► (broadcast, weights)  # 注意焦点
Drives.deficit ──► GoalManager.update_from_drives ─► current()  # 当前目标
```

## 典型数据流（阶段 2 已可跑通）

```
World.observe(id) -> obs (6,) -> Encoder -> z (128,) -> Policy -> action
World.step(id, action) -> effects -> DriveSystem.step(effects) -> deficit
```

## 下一步（阶段 4，预计 3 周）

1. M10 cog-world-model：`forward(z, a) -> (z', r, d)` / `imagine(z, policy, H)`
   （潜空间转移模型 + rollout，NumPy 微型网络后端）。
2. M16 cog-planner：`plan(z, goal) -> actions`（基于 M10 的 CEM/MPC 规划）。

## 下一步（阶段 5，预告）

M07 情景记忆（相似检索）、M08 语义记忆（知识图谱 + 扩散激活）、M09 程序记忆（技能库）。
