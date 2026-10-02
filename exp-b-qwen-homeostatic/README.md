# exp-b-qwen-homeostatic — Stage B: injecting homeostasis into a language model

**[English](README.en.md)** | 中文

## 目标

阶段 A 验证了范式（小 Transformer、唯一回报 = 内稳态缺口缩减，六组对照成立）。
阶段 B 的问题更进一步：

> **一个语言模型能否"具身"一个内稳态策略？它的行为是否被内感受调制？**

如果 Qwen-0.5B 在学会"活下去"的同时，行为随身体状态（能量/完整度）改变
——饿时觅食、饱时无视——那么"新物种大模型"的范式就跨过了第二级台阶：
不是背下策略，而是把内稳态判断装进了语言表征里。

## 方法（v1，可证伪）

```
exp-a 课程组策略（教师，12×12，satiated=0 / far=1.00）
        │ 轨迹回放
        ▼
文本观测 → 动作    SFT 数据集（约 2 万条）
        │ QLoRA（4bit，RTX 3050 4GB）
        ▼
Qwen2.5-0.5B-Instruct（学生）
        │ 行为探针
        ▼
1. 教师一致率 —— 学没学会
2. 存活率 —— 能不能活
3. 内感受调制 ★ —— 同样视野、不同能量状态，动作是否改变
   （这是核心探针：拟合策略 vs 装入内稳态判断的分水岭）
```

诚实边界：SFT 只证明"可注入"，不证明"涌现"。若内感受调制成立，
后续用 DPO（内稳态偏好对）与 RL（环境回报直接进策略）检验更深层的注入。

## 结构

- `homeo_lm.textenv` —— 文本渲染的内稳态网格世界（复用 exp-a 环境）
- `homeo_lm.gen_data` —— 教师轨迹 → SFT 数据集
- `homeo_lm.sft` —— QLoRA 微调（快照 + `--resume` 断点续训）
- `homeo_lm.probe` —— 行为探针（一致率 / 存活 / 内感受调制）

## 进度（v1，2026-10-01 → 10-02）

- [x] 环境：CUDA torch（RTX 3050 4GB），Qwen2.5-0.5B-Instruct 本地缓存
- [x] 教师数据：40 局 / **13,729 条**（教师死亡率 0.325，平均局长 343.2）
- [x] LoRA SFT：**完成**（1 epoch，batch 2 × accum 16；bitsandbytes DLL 被
      Smart App Control 拦截 WinError 4551 → 自动回退 bf16 LoRA；
      loss 0.65 → 0.19）
- [x] 三探针：**完成**，结果见下（`runs/sft-lora/probe_report.json`）

## 结果（v1）——负结果，诚实记录

| 探针 | 结果 | 标准 | 判定 |
|---|---|---|---|
| agreement 一致率 | **0.485**（随机 ≈ 0.2，非法率 0） | > 0.7 | ✗ |
| survival 存活率 | **0.00**（平均 77.7 步/局，非法率 0） | 显著高于随机 | ✗ |
| modulation_gap ★ | **0.00**（饥饿/饱腹朝食物均为 0%） | > 0.5 | ✗ |

**诊断**：贪心解码在 60 个构造状态（食物右侧 3 格、饥饿/饱腹交替）下
100% 输出 `down`——策略坍缩为单一动作。而数据集中教师动作分布为
rest 50% / right 15% / left 14% / up 11% / down 10%：坍缩到的甚至不是
多数类，说明 1 epoch 只完成了"输出一个合法动作词"的**格式学习**，
没有完成观测→动作的**策略绑定**。

**仍然有意义的信号**：非法输出率 0（格式学会了）；一致率 0.485 显著
高于随机 0.2（部分策略信号存在，但不足以支撑存活）。

## v2 计划

1. 数据扩到 200 局（约 7 万条），训练 2–3 epochs
2. 动作分层平衡采样（rest 占 50% 的类不平衡需要处理，或对 rest 下采样）
3. 训练中监控逐动作的预测分布，检测坍缩并早停
4. 扩大 LoRA 覆盖（加入 MLP 层）或增大 r
5. 判读逻辑不变：若 v2 达成一致率 + 存活但 modulation_gap 仍为 0，
   则恰好印证"拟合策略 ≠ 装入内稳态判断"的分水岭假设，升级到
   DPO（内稳态偏好对）

## 用法

```powershell
py -3.12 -m pip install -e ../exp-a-homeostatic-rl   # 教师环境
py -3.12 -m pip install -e ".[dev]"
py -3.12 -m homeo_lm.gen_data --ckpt ../exp-a-homeostatic-rl/runs/curriculum/checkpoint.pt --episodes 200
py -3.12 -m homeo_lm.sft --data data/sft.jsonl
py -3.12 -m homeo_lm.probe --ckpt runs/sft-lora --teacher ../exp-a-homeostatic-rl/runs/curriculum/checkpoint.pt
```

模型下载（国内镜像）：`$env:HF_ENDPOINT="https://hf-mirror.com"`

## 判读标准

1. `agreement` > 0.7 —— 动作分布与教师基本一致（学到了策略）
2. `survival` 显著高于随机基线 —— 能活
3. **`modulation_gap` > 0.5** —— 低能量与饱腹状态下动作分布显著不同
   （内稳态判断被装入语言模型，而非只背了视觉→动作映射）
