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
- `homeo_lm.sft` —— QLoRA 微调
- `homeo_lm.probe` —— 行为探针（一致率 / 存活 / 内感受调制）

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
