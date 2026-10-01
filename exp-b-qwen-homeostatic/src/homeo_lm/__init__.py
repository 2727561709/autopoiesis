"""homeo_lm —— 阶段 B：把内稳态目标注入语言模型。
homeo_lm — Stage B: injecting homeostatic objectives into a language model.

exp-a 验证了范式（小 Transformer + 内稳态回报）；本包检验语言模型
能否具身同一策略：SFT(QLoRA) 教师轨迹，然后用行为探针测"内感受
调制"——同样视野、不同身体状态，动作是否改变。
Stage A validated the paradigm (small Transformer + homeostatic reward);
this package tests whether a language model can embody the same policy:
SFT (QLoRA) on teacher trajectories, then behavioral probes measuring
"interoceptive modulation" — same view, different body state, do the
actions differ?
"""
__all__ = ["textenv", "gen_data", "sft", "probe"]
