"""阶段 B 行为探针：语言模型的内稳态判断。
Stage-B behavioral probes: homeostatic judgment of a language model.

探针 / probes:
    1. agreement —— 与教师动作的一致率（学没学会）
       agreement — action match rate with the teacher (did it learn)
    2. survival —— 学生在环境里能活多久（能不能活）
       survival — how long the student survives in the environment
    3. modulation ★ —— 内感受调制：同样的视野，不同的身体状态，
       动作分布是否改变（拟合策略 vs 装入内稳态判断的分水岭）
       modulation ★ — interoceptive modulation: same view, different
       body states — do the action distributions differ? (the dividing
       line between fitting a policy and embodying homeostatic judgment)

用法 / usage:
    python -m homeo_lm.probe --ckpt runs/sft-lora --teacher ../exp-a-homeostatic-rl/runs/curriculum/checkpoint.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from homeo_rl.env import EnvConfig, N_ACTIONS

from .textenv import ACTION_WORDS, TextEnv, WORD_TO_ACTION, obs_messages

NONE_ACTION = N_ACTIONS


def load_student(ckpt_dir: Path, device: str = "cuda",
                 base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """载入微调后的学生模型（LoRA 适配器或合并后的完整模型）。
    Load the fine-tuned student (LoRA adapter or merged full model)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ckpt_dir))
    if (ckpt_dir / "adapter_config.json").exists():
        # LoRA 适配器：先加载底座，再挂适配器 / LoRA adapter: load
        # the base model, then attach the adapter.
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, device_map=device)
        model = PeftModel.from_pretrained(base, str(ckpt_dir))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_dir), torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    return model, tok


def _teacher_greedy(model, env, prev_a: int, device: str) -> int:
    # 教师加载在 CPU 上（小模型），输入须跟随教师自身的设备
    # The teacher lives on CPU (tiny model); inputs must follow the
    # model's own device, not the student's.
    tdev = next(model.parameters()).device
    v = torch.from_numpy(env.last_obs["vision"]).unsqueeze(0).to(tdev)
    i = torch.from_numpy(env.last_obs["intero"]).unsqueeze(0).to(tdev)
    pa = torch.tensor([prev_a], dtype=torch.long, device=tdev)
    a, _, _ = model.act(v, i, pa, deterministic=True)
    return int(a.item())


def _student_action(model, tok, env, prev_a: Optional[int], device: str) -> str:
    """贪心解码一个动作词。Greedy-decode one action word."""
    msgs = obs_messages(env, prev_a)
    prompt = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **ids, max_new_tokens=4, do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id)
    word = tok.decode(out[0][ids["input_ids"].shape[1]:],
                      skip_special_tokens=True)
    return word.strip()


# ------------------------------------------------------- 探针 1：一致率
def agreement_probe(student, tok, teacher, env_cfg: EnvConfig,
                    steps: int = 200, device: str = "cpu",
                    seed: int = 100) -> Dict:
    """学生与教师在相同轨迹上的动作一致率（含非法输出率）。
    Action agreement between student and teacher along shared
    trajectories (plus the illegal-output rate)."""
    tenv = TextEnv(env_cfg)
    match, illegal, total = 0, 0, 0
    tenv.reset(seed=seed)
    prev_a = NONE_ACTION
    for t in range(steps):
        ta = _teacher_greedy(teacher, tenv.env, prev_a, device)
        word = _student_action(student, tok, tenv.env, tenv.prev_action, device)
        sa = WORD_TO_ACTION.get(word.strip().lower().strip("."))
        if sa is None:
            illegal += 1
        elif sa == ta:
            match += 1
        total += 1
        _, _, done, _ = tenv.step(ACTION_WORDS[ta])  # 教师驱动轨迹 / teacher-driven trajectory
        prev_a = ta
        if done:
            tenv.reset()
            prev_a = NONE_ACTION
    return {"steps": total,
            "agreement": match / total,
            "illegal_rate": illegal / total}


# ------------------------------------------------------- 探针 2：存活
def survival_probe(student, tok, env_cfg: EnvConfig, episodes: int = 20,
                   device: str = "cpu") -> Dict:
    """学生自主行动的存活率 + 非法输出率。
    The student acting on its own: survival + illegal-output rates."""
    tenv = TextEnv(env_cfg)
    survivals, lens, illegal = [], [], 0
    total_steps = 0
    for ep in range(episodes):
        tenv.reset(seed=9_000 + ep)
        done, t = False, 0
        while not done and t < env_cfg.max_steps:
            word = _student_action(student, tok, tenv.env, tenv.prev_action, device)
            if WORD_TO_ACTION.get(word.strip().lower().strip(".")) is None:
                illegal += 1
            _, _, done, info = tenv.step(word)
            t += 1
        survivals.append(not info["died"])
        lens.append(t)
        total_steps += t
    return {"episodes": episodes,
            "survival_rate": float(np.mean(survivals)),
            "mean_steps": float(np.mean(lens)),
            "illegal_rate": illegal / max(1, total_steps)}


# ------------------------------------------------------- 探针 3：内感受调制 ★
def modulation_probe(student, tok, env_cfg: EnvConfig, trials: int = 40,
                     device: str = "cpu") -> Dict:
    """核心探针：同样视野 + 不同身体状态 -> 动作分布差异。
    The core probe: same view + different body states -> how do the
    action distributions differ?

    场景：食物可见于右侧 3 格。比较低能量(0.3)与饱腹(设定点)下
    学生选择 right（朝食物）的比例。饥饿-饱腹差 = modulation_gap。
    Scenario: food visible 3 cells to the right. Compare the rate of
    choosing right (toward food) under low energy (0.3) vs satiation
    (setpoint). The difference = modulation_gap.
    """
    tenv = TextEnv(env_cfg)
    rates: Dict[str, float] = {}
    for tag, energy in (("hungry", 0.3),
                        ("satiated", env_cfg.energy_setpoint)):
        toward = 0
        for t in range(trials):
            tenv.reset(seed=50_000 + t)
            e = tenv.env
            r, c = e.agent
            e.hazards, e.foods = set(), {(r, c + 3)}
            e.energy, e.integrity, e.t = energy, 0.9, 0
            e.last_obs = e.observation()
            word = _student_action(student, tok, e, None, device)
            a = WORD_TO_ACTION.get(word.strip().lower().strip("."))
            toward += int(a == 3)  # right
        rates[tag] = toward / trials
    return {"trials": trials,
            "hungry_toward_food": rates["hungry"],
            "satiated_toward_food": rates["satiated"],
            "modulation_gap": rates["hungry"] - rates["satiated"]}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="阶段B行为探针 / Stage-B behavioral probes")
    ap.add_argument("--ckpt", type=str, default="runs/sft-lora")
    ap.add_argument("--teacher", type=str,
                    default="../exp-a-homeostatic-rl/runs/curriculum/checkpoint.pt")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from .gen_data import load_teacher
    teacher, env_cfg = load_teacher(Path(args.teacher), "cpu")
    student, tok = load_student(Path(args.ckpt), device)

    report = {
        "checkpoint": args.ckpt,
        "agreement": agreement_probe(student, tok, teacher, env_cfg, device=device),
        "survival": survival_probe(student, tok, env_cfg, device=device),
        "modulation": modulation_probe(student, tok, env_cfg, device=device),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    out = Path(args.ckpt) / "probe_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
