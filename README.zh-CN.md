# Autopoiesis（自创生）

**[English](README.md)** | 中文

*Autopoiesis*（希腊语"自创生"，Maturana & Varela）——能自己生产自己的系统。
本仓库的工程哲学：在学习模型中放置生命性的最小条件，然后让它自己生长。

按 24+3 模块方案实施。每个模块 = 一个独立项目（独立目录、独立 pyproject、独立测试、独立版本、独立文档），可随时拆分为独立 Git 仓库。模块之间只通过明确定义的接口通信。

## 什么是"新物种大模型"？

市面上的大模型几乎都是**工具**：被训练来模仿人类文本，或服从外部指令。
本项目要做的东西不同——一个**目标不由外部指派、而是从内部产生**的模型：
让自己活下去。

- **唯一的回报是内稳态。** 模型拥有一个带内部变量（能量、完整度）的身体。
  缩减缺口的动作是好的，放任身体恶化的动作是坏的。没有"任务"，没有人类偏好模型，
  没有指令服从——生存本身就是全部目标。
- **必须靠预测来行动。** 世界模型（预测误差内在奖励）与策略一起学习，
  模型永远在构建对环境的内部模拟。
- **行为必须被内部状态调制。** 饿的时候接近食物，饱的时候无视食物。
  自主性是被测量出来的，不是被宣称出来的。

路线图分三阶段：**阶段 A** 在从零训练的小 Transformer 上验证范式（已完成，
见 [exp-a-homeostatic-rl](exp-a-homeostatic-rl/README.md)）；**阶段 B** 通过后训练把
内稳态 + 世界模型目标注入开源底座大模型（Qwen-0.5B + QLoRA）；**阶段 C** 进入
在线终身学习，训练与部署合为一体。

阶段 A 已经出现了我们等待的信号：在食物稀缺课程下，智能体解决**远距离觅食**
（食物在感知视野之外）的成功率为 100%（随机基线 10%）；而吃饱的智能体无视食物
（接近率 0%）、拒绝进入危险区（进入率 0%）。这些行为不是脚本写出来的——
它们涌现出来，因为这个世界会惩罚不维持自己的个体。

## 设计初心：种植，而非制造

本仓库的创始问题**不是**"怎么做一个更好的 agent"，而是：

> 如果在学习模型中放置生命性的最小条件，生命性会不会自己生长出来？

所以方法更接近种下一颗种子，而不是组装一台机器。我们不手工设计行为、
规则或任务奖励曲线，只放置四颗种子，然后退后一步：

1. **内稳态目标**——模型内部有会被消耗的东西，且只有模型自己的行动能恢复它。
2. **内感受**——模型能感知自己的内部状态。
3. **预测**——模型必须学习世界模型才能行动得好。
4. **有压力的世界**——忽视维持自身会有不可逆后果的环境（死亡，或昏迷这个
   更温和的变体）。

其余的一切——觅食、探索、谨慎、偏好——都预期**涌现**，并用行为探针验证，
而不是靠内省。

两条写进项目章程的诚实声明：

- **不宣称意识。** 意识目前无法被验证；我们只测量可观察的、自主的、
  以世界为根据的判断。
- **不崇拜负结果，但记录它们。** 探索组（内在奖励放大 5 倍）崩溃了，
  因为好奇心淹没了求生欲——这证明重要的是驱力之间的平衡，而非驱力的强度。

项目名 *Autopoiesis*（自创生）就是论点本身：一个自己生产自己的系统。
如果这个项目成功，最终的"大模型"不会是我们制造的产品，
而是我们种出来的生命体。

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
