# RSI 方案：递归自我改进（Recursive Self-Improvement）

> 状态：设计稿（2026-10-25）。Stage B 判读尚未完成，Phase 0 是硬前置。
> Status: design draft. Stage B read-out is still pending; Phase 0 is a
> hard prerequisite.

---

## 0. 一句话原则

**改进的目标由环境定义，改进的成绩由环境打分，系统永远不给自己打分。**
The improvement target is defined by the environment, the improvement is
graded by the environment, and the system never grades itself.

这一条是整个方案的承重墙。它把 Autopoiesis 的 RSI 与"用一个模型当
裁判、另一个模型当考生"的常见做法彻底分开：后者是可被说服的裁判，前者
是不可被说服的物理后果（饿死就是饿死）。
This is the load-bearing wall of the whole design. It separates
Autopoiesis's RSI from the common "one model judges another" setup: a
model judge can be talked into anything; starvation cannot.

---

## 1. 现状盘点：我们已有什么，还缺什么

阶段 B 已经把 RSI 的**大部分零件**造出来了，只是没装成闭环：

| 零件 | 现状 | 对应 RSI 角色 |
|---|---|---|
| `homeo_lm.ppo` | v4 已实现，未训练 | **RSI 的胚胎**：策略自己在环境里 rollout → 回报直入策略 |
| `homeo_lm.gen_dpo_data` | v3 已跑通 | 偏好对来自**环境真实后果**，不来自教师标签 |
| `homeo_lm.dpo` | v3 已训练（margin +4.23） | 从环境后果反推的对比学习 |
| `homeo_lm.probe` | agreement/survival/modulation + sweep | **打分器（环境侧）** |
| `cog-checkpoint` (M03) | 已实现，原子写 + 版本 + 元数据 | **回滚基座** |
| `cog-goal-manager` (M15) | 已实现 | 候选改动的来源之一（课程/目标参数） |
| M17 元认知 / M18 自传 | **未开发** | 缺失：诊断能力 + 演进账本 |

**缺口只有三个**：① 测量（探针还跑不动、也太慢）② 门控（谁批准一次改动）
③ 账本（改了什么、为什么、结果如何）。其余都可以复用。
Only three gaps: (1) measurement, (2) the gate that approves a change,
(3) the ledger. Everything else is reusable.

---

## 2. 三层分级：从便宜安全到昂贵危险

不做"一次性让模型改自己"，而是分三级，**每升一级都要先证明下一级有用**。
No single leap to "the model edits itself" — three levels, and each level
must prove itself before the next is unlocked.

| 级别 | 改什么 | 是否动权重 | 风险 | 状态 |
|---|---|---|---|---|
| **L0 自适应调参** | 课程难度、数据混合比、探索率、整形权重 `w`、门控 margin | 否 | 极低（最坏只是浪费算力） | 可直接做 |
| **L1 自产训练信号** | 用**自己当前的策略**在环境里 rollout，产出偏好对/轨迹，再更新自己 | 是（目标由环境固定） | 中 | PPO 已是此层；需补 on-policy 偏好对 |
| **L2 改学习规则本身** | LoRA rank / target modules / lr schedule / 记忆巩固策略 / 目标管理策略 | 是 | 高 | 需 M17 + 硬门控 |

**不可改清单（永远拒绝，结构性拒绝，不看指标）**：
Never-change list (structurally rejected, regardless of metrics):

1. 内稳态目标本身（设定点、缺口函数 `deficit = max(0, setpoint - x)`）
2. 环境的物理规则（移动成本、伤害、死亡条件）
3. 探针的定义与打分函数
4. 门控自身与不可改清单本身

---

## 3. 主循环：七步闭环

```
   ① measure   固定留出种子上跑探针 → 得到指标向量
   ② diagnose  归因：瓶颈在哪一层？（调制 / 搜索 / 世界模型 / 容量 / 坍缩）
   ③ propose   产出候选改动（本级别允许的维度）
   ④ gate      硬约束检查：不可改清单 / 探针哈希 / 预算 / 熵地板
   ⑤ trial     在训练种子上试跑（沙盒，不碰打分种子）
   ⑥ decide    留出种子复测：显著变好 → 采纳；否则 → 回滚
   ⑦ record    写入演进账本（改了什么、为什么、结果、回滚指针）
```

关键在第 ⑤⑥ 步的**种子三分**：
The crux is the three-way seed split in steps 5-6:

| 种子集 | 用途 | 是否可见给改进过程 |
|---|---|---|
| `train` | 试跑候选 | 可见 |
| `select` | 决定采纳哪个候选 | 可见 |
| `holdout`（冻结） | **最终判定"是否真的变好了"** | 不可见 |

不能让系统对着同一套题反复练——那不是改进，是背答案。这是 v2 的教训
（70k 条数据里把"饿+left"的少数模式背下来 → 反向调制）。
Letting the system drill on the same exam is memorisation, not
improvement — that is exactly what produced v2's inversion.

---

## 4. 各层设计

### L0：自适应调参（Phase 1）

可自动调的旋钮（每个都**有界**）：

| 旋钮 | 范围 | 观测指标 |
|---|---|---|
| 课程难度（食物距离/危险密度/初始能量） | 离散 5 档 | survival、steps_to_food |
| `shape_w`（势函数整形权重） | [0, 0.15] | modulation_gap、坍缩 top |
| 教师/随机轨迹混合比 | [0, 1] | agreement、泛化 |
| `gate_margin`（门控阈值） | [0.02, 0.15] | modulation_gap |
| 探索温度 / PPO entropy_coef | [0, 0.05] | 动作熵、collapse top |

搜索策略：**Bandit（UCB）起步，不上贝叶斯优化**——旋钮少、噪声大，
先用最笨但最可解释的方法，且每次试验的结果都进账本，可复现。
UCB bandit first, not Bayesian optimisation: few knobs, high noise,
and every trial's result lands in the ledger for reproducibility.

**P1 已实现**（`homeo_lm.rsi_l0`）：`KNOBS` 五旋钮各有离散取值域；
`FORBIDDEN_KEYS` 结构性拒绝（不看指标）；`ucb_propose` 先试未试过的臂、
再按"均值 + c·√(ln N / n)"挑；`Budget` 封顶试验数并对连续失败熔断；
`oscillating` 检测锯齿（在噪声里打转就停下交给人）；`decide` 三态判定
（守卫失败/显著变差 → 回滚；显著变好 → 采纳；否则拒绝）。
`rsi.py --cycle` 驱动一个周期，**默认 dry-run**——只打印"下一步试什么"
与训练命令，不执行（一次试验要真跑训练 + 全套探针，时机由人决定）。
P1 implemented (`homeo_lm.rsi_l0`): five bounded knobs; structural
rejection of `FORBIDDEN_KEYS` (metrics never consulted); UCB1 with
untried arms first; trial budget + circuit breaker; zig-zag detection;
three-way verdict (guardrail / regression -> rollback, significant gain
-> adopt, else reject). `rsi.py --cycle` drives one cycle and is
DRY-RUN by default.

### L1：自产训练信号（Phase 2）

两条并行轨道，目标都是"数据分布跟着自己当前的错误走"：
Two parallel tracks; both make the data distribution track the model's
own current mistakes:

1. **On-policy 偏好对**：`gen_dpo_data --policy <当前适配器>`（新增参数）
   —— 用当前学生自己 rollout 的状态做同状态穷举对比，而不是只用教师/
   随机轨迹。学生现在错在哪，数据就长在哪。
   Exhaustive contrast on states the *current student* actually visits,
   not only teacher/random ones.
2. **PPO 自博弈**：`homeo_lm.ppo --ckpt <当前适配器>`，回报直入策略。
   这层已经写完，缺的是跑起来 + 每次 update 后的轻量探针。

每次更新后跑 **fast probe**（见 Phase 0），形成"改进—验证"的密集闭环。
Iterated DPO 的每一轮都要重新生成偏好对（否则退化为普通 DPO）。

**P2 已实现**：
- `gen_dpo_data --policy <适配器> --policy-frac 0.34`：新增第三条数据来源
  ——**学生自己的 on-policy 轨迹**（`PolicyDriver` 用 acthead 采样驱动，
  不做贪心：贪心会把轨迹压到几条确定性路线上、覆盖反而变窄）。三段采集
  逻辑合并成一个 `_collect`，只有驱动不同。
- `homeo_lm.selftrain`：L1 编排器，**每步都用子进程**跑（生成要载教师、
  训练要载策略+参考、测量要载学生——同一进程叠两份 0.5B 模型在这台机器上
  必炸）。一轮 = 生成 → 训练 → 多次独立测量 → `decide` → 记账；采纳才
  推进 `current`，否则保留 parent。默认 **dry-run**。
- `probe --seed-offset`：显著性判定需要多个独立取样（同一 offset 可复现）。
- `ppo --probe-every N`：训练中定期插轻量调制探针（模型已在内存里，几乎
  免费），轨迹写入 `probe_trace.json`——RL 的"存活率/调制随训练变化"曲线
  因此可见。
P2 implemented: `gen_dpo_data --policy/--policy-frac` adds a third data
source — the student's OWN on-policy rollouts (`PolicyDriver` samples
via acthead; not greedy, because greedy collapses trajectories onto a
few deterministic routes). `homeo_lm.selftrain` orchestrates L1 in
subprocesses (generation loads the teacher, training loads policy+ref,
measurement loads the student — two 0.5B models in one process would
blow up here). `probe --seed-offset` gives independent samples for
significance. `ppo --probe-every N` runs a light in-training probe.

### L2：改学习规则（Phase 3，需 M17）

到这一层才允许改动"怎么学"，且必须：
- 由 M17 元认知产出**假设**（例如"调制错了，可能是容量不足：把 LoRA rank
  从 16 提到 32"），而不是盲搜
- 每个假设附带**可证伪的预期**（"若容量是瓶颈，rank↑ 后调制 gap 应上升"）
- 假设被证伪时，账本记录**负结果**，并降低该类假设的采样权重

**P3 已实现**（`homeo_lm.metacog` + `rsi.py --level 2`）：六条诊断规则
各自要求**多个证据同时成立**才点火，缺失指标视为"不知道"、**不触发**
（不许拿猜测当证据）：

| 假设 | 触发证据 | 证伪用旋钮 |
|---|---|---|
| `capacity_deficit` | train_acc 高 + gap≤0 + 曲线不升 | `lora_rank` |
| `optimization_underfit` | margin 低 + loss 高 + gap<0.5 | `lr` |
| `distribution_mismatch` | 一致率高 + 存活低 / 留出远差于训练 | `teacher_mix` |
| `shaping_misspecified` | gap≈0 + 非法率低 + 存活低 | `shape_w` |
| `catastrophic_forgetting` | 先升后降 + 训练损失仍在降 | `replay_frac` |
| `collapse` | top share≥0.6 + 熵<0.2 | `entropy_coef` |

每条假设强制三件套：`claim`（可证伪断言）+ `prediction`（若为真哪个数
往哪动）+ `knob`（用什么证伪它）——`is_falsifiable()` 会检查，缺
prediction 不许动手。**历史命中率进排序**（`hit_rates` + UCB）：哪类假设
过去常被环境证实就更优先试，这才是"元"。
`l2_gate` 比 L0 严：留出集必须验过、≥2 seed、**单变量**（一次只改一个
旋钮，否则无法归因）、旋钮必须属 L2。
P3 implemented (`homeo_lm.metacog` + `rsi.py --level 2`): six diagnosis
rules, each requiring several conditions at once; a missing metric is
UNKNOWN and never fires a rule. Every hypothesis must carry
claim + prediction + knob (`is_falsifiable()` enforces it). Historical
hit rates feed ranking (`hit_rates` + UCB) — that is what makes it
"meta". `l2_gate` is stricter than L0: holdout verified, >= 2 seeds,
single variable, L2-only knobs.

---

## 5. 安全设计：七条硬约束

| # | 约束 | 机制 |
|---|---|---|
| 1 | **目标锁** | 设定点/缺口函数存在冻结的 spec 文件，改动即结构性拒绝（不比较指标） |
| 2 | **探针锁** | 探针配置 + 种子集哈希写入账本；哈希变了，本轮所有采纳作废 |
| 3 | **环境打分，不自评** | 打分端只读环境后果（存活/缺口/死亡），模型输出**永不**进入打分函数 |
| 4 | **统计显著性** | 采纳需 Δ > 2σ（bootstrap CI），且留出种子上复现 |
| 5 | **熵地板 + 坍缩熔断** | 动作分布 top > 0.6 或熵 < 阈值 → 立即回滚（v1 的教训直接编码进系统） |
| 6 | **回滚 + 账本** | 每次采纳前落 checkpoint（M03），账本记 `(parent, diff, metrics, verdict)`；下一周期指标回退 → 自动回滚到 parent |
| 7 | **预算与熔断** | 每周期最多 K 次试验、封顶算力/时长；连续 N 次熔断 → 冻结并报警（交给人） |

额外一条**反振荡**：同一旋钮在 M 个周期内不得来回翻转（账本检测锯齿模式）。
Anti-oscillation: the same knob may not flip back and forth within M
cycles (the ledger detects zig-zag patterns).

---

## 6. 元探针：怎么证明"RSI 真的有用"

这是整个方案最容易被自欺的地方，必须用**对照组**回答。
The easiest place to fool ourselves; it must be answered with a control.

> **问题不是"最终指标高不高"，而是"同样算力下，RSI 是否比'闷头多训一会'更强"。**
> The question is not "are the final metrics high?" but "at equal compute,
> does RSI beat just training longer on a fixed curriculum?"

实验矩阵（每格固定算力预算，多seed）：

| 组 | 设置 | 用于回答 |
|---|---|---|
| A 基线 | 固定课程，一直训 | 算力本身能买多少性能 |
| B | L0 only | 自适应调参值不值 |
| C | L1 only（固定超参） | 自产信号值不值 |
| D | L0 + L1 | 是否互补 |
| E | L0+L1+L2 | 改学习规则还有多少增量 |

**判据**：`D > A` 且 `D > B、D > C`（超可加性）。若 `D ≈ A`，说明
RSI 在这个环境里买不到任何东西——**这也是有价值的结果，要如实记录**
（项目章程：不崇拜负结果，但必须记录）。

---

## 7. 分阶段路线图

| Phase | 内容 | 产物 | 验收标准 |
|---|---|---|---|
| **P0 测量** | 快速探针（批量生成、多种子、CI）、探针套扩展（steps_to_food、世界模型精度、驱动平衡）、种子三分 | `probe` 加速 + `probe_suite.py` + 种子管理 | 全套探针 **< 3 分钟**、可重复（同 seed 同结果） |
| **P1 L0** ✅ 代码已就绪（2026-10-25） | 旋钮空间定义 + UCB bandit + 门控 + 账本 + 回滚 | `rsi.py`、`rsi_l0.py` | 在固定预算下 `B ≥ A`；无熔断触发 |
| **P2 L1** ✅ 代码已就绪（2026-10-25） | on-policy 偏好对 + PPO 密集闭环 | `gen_dpo_data --policy`、`selftrain.py`、PPO `--probe-every` | 调制的**改进速度**快于基线（而非最终值） |
| **P3 L2** ✅ 代码已就绪（2026-10-25） | M17 元认知假设生成 + L2 旋钮 + 反振荡 | `metacog.py`、`rsi.py --level 2` | 至少一条假设被**证伪**并留下负结果记录 |
| **P4** | 并入 Stage C：训练与部署合一，在线终身改进 | 在线循环 | 长期（>10 周期）无退化、无振荡 |

**P0 的硬前置**：当前探针被系统内存卡住（`OSError 1455`），需先释放内存
或扩大页面文件。测量不跑起来，后面全是无根据的自我修改。
Hard prerequisite for P0: the probe is currently blocked by system memory.
Measurement must run before any self-modification is meaningful.

---

## 8. 风险与证伪条件

| 风险 | 表现 | 处置 |
|---|---|---|
| 背答案（过拟合种子） | select 上变好，holdout 不变 | 种子三分 + holdout 冻结（已在设计中） |
| 指标漂移 | 探针分数涨，行为更脆 | 多指标联合门槛（不许单项换全局） |
| Goodhart（改进者成为目标） | 系统优化"让自己看起来在改进" | 打分端不读模型输出（约束 #3） |
| 算力失控 | 试验数爆炸 | 预算封顶 + 熔断（约束 #7） |
| 振荡 | 旋钮来回翻 | 反振荡检测 |
| 能力上限幻觉 | 调参到头就以为到顶 | P4 才允许 L2，且要求先证伪一条假设 |

**证伪本方案的条件**（写下来，将来不许赖账）：
- 在 3 个独立 seed 上，`D ≤ A`（RSI 组不优于固定课程组）
- 或：所有 L0 旋钮在 holdout 上的效应量都落在噪声内

---

## 9. 为什么"内稳态 RSI"比通用 RSI 更安全

1. **目标在外，不在内**：目标是"别饿死、别受伤"，由物理环境给出；系统
   无法重新定义"活得好"来取巧。
   The objective lives in physics, not in the model — the system cannot
   redefine "doing well" to cheat.
2. **多维，不是单一标量**：能量 + 完整性两个缺口同时存在，压一个不能补
   另一个，天然抵抗 Goodhart（v2 就是把两维压成一维动作分布的后果）。
   Two simultaneous deficits; squeezing one cannot pay for the other.
3. **有不可撤销的失败**：死亡是外部事件，不可协商——这给了"改进"一个
   不需要对齐讨论的硬底。
   Death is external and non-negotiable: a hard floor for "improvement".
4. **裁判不可被说服**：打分只读环境后果，模型输出不进入评分函数，因此
   不存在"说服裁判"这条攻击面。
   No attack surface of "persuading the judge" exists, because the judge
   only reads consequences.

---

## 10. 立刻可做的最小第一步（MVI）

不碰权重、不碰门控，只做**"能测量 + 能回滚"**：

1. 让 `probe` 跑起来并加速到 <3 分钟（P0）
2. 写 `rsi.py` 的骨架：只实现 `measure → record`（暂时不自动改任何东西）
3. 手动跑 A 基线（固定课程，固定预算），把指标与 seed 记进账本

这三步做完，RSI 就有了**可信的起点和对照组**——之后每一次"自我改进"
都有东西可以对比。在此之前，任何自动修改都是在没有尺子的情况下动刀。
These three steps give RSI a trustworthy baseline. Any automatic
modification before that is surgery without a ruler.
