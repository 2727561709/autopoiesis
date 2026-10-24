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
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from homeo_rl.env import EnvConfig, N_ACTIONS

from .acthead import action_token_ids, greedy_with_legality
from .textenv import ACTION_WORDS, TextEnv, WORD_TO_ACTION, obs_messages

NONE_ACTION = N_ACTIONS


def load_student(ckpt_dir: Path, device: str = "cuda",
                 base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """载入微调后的学生模型（LoRA 适配器或合并后的完整模型）。
    Load the fine-tuned student (LoRA adapter or merged full model).

    显存策略（2026-10-21 实测）：`device_map=` 会把整份权重（约 944MB）
    申请成一个**连续**显存块；在 Windows WDDM 下，同时开着几十个共享
    GPU 的桌面应用（Chrome / Edge / 飞书 / LM Studio …）时，这块连续
    显存会申请失败——即使 nvidia-smi 显示还有 3.4GB 空闲。改为先在 CPU
    上加载、再逐张量搬到 GPU（每块都很小），并在 OOM 时自动降级 CPU，
    保证探针在任何内存条件下都能跑出结果。
    VRAM strategy (measured 2026-10-21): `device_map=` requests the whole
    weight set (~944MB) as one CONTIGUOUS block; under Windows WDDM with
    dozens of GPU-sharing desktop apps (Chrome / Edge / Feishu /
    LM Studio ...) that contiguous request fails even while nvidia-smi
    reports 3.4GB free. Load on CPU first, then move tensor by tensor
    (each block is tiny), and fall back to CPU on OOM so the probe can
    always produce a result.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ckpt_dir))
    if (ckpt_dir / "adapter_config.json").exists():
        # LoRA 适配器：先加载底座，再挂适配器 / LoRA adapter: load
        # the base model, then attach the adapter.
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, str(ckpt_dir))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_dir), torch_dtype=torch.bfloat16)
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            model.to(device)          # 逐张量搬移 / tensor-by-tensor
        except torch.OutOfMemoryError:
            print("WARNING: CUDA OOM while loading the student — "
                  "falling back to CPU (slower but it will finish).")
            device = "cpu"
    else:
        device = "cpu"
    model.eval()
    return model, tok, device


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
    """贪心解码一个动作词（逐条，慢——仅作对照与校验用）。
    Greedy-decode one action word (slow path; kept for the batching
    cross-check and for single-state use)."""
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


def prompt_ids(tok, env, prev_a: Optional[int]) -> List[int]:
    """观测 → prompt token ids（批量路径的第一步）。
    Observation -> prompt token ids (first step of the batched path)."""
    msgs = obs_messages(env, prev_a)
    prompt = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)
    return tok(prompt, add_special_tokens=False)["input_ids"]


def batched_actions(model, tok, prompts: Sequence[Sequence[int]],
                    act_ids: Sequence[int], device: str,
                    batch_size: int = 32) -> List[int]:
    """批量贪心决策：一次前向给一批动作（探针加速的核心）。
    Batched greedy decisions: one forward pass per batch of actions.
    This is the core of the probe speed-up.

    与逐条 `generate` 的等价性由 `--verify-batching` 实测校验。
    Equivalence with per-sample `generate` is checked empirically by
    `--verify-batching`.
    """
    acts, _ = greedy_with_legality(model, tok, prompts, act_ids, device,
                                   batch_size)
    return acts


# ------------------------------------------------------- 探针 1：一致率
def agreement_probe(student, tok, teacher, env_cfg: EnvConfig,
                    steps: int = 200, device: str = "cpu",
                    seed: int = 100, trajs: int = 4,
                    batch_size: int = 32) -> Dict:
    """学生与教师在相同轨迹上的动作一致率（含非法输出率）。
    Action agreement between student and teacher along shared
    trajectories (plus the illegal-output rate).

    加速：并行跑 `trajs` 条教师驱动轨迹，每步把 `trajs` 个观测合成
    一个 batch 前向（原来是逐条 generate，一步 4 次前向）。`steps`
    仍是**总**采样步数，指标定义不变。
    Speed-up: `trajs` teacher-driven trajectories run in parallel, one
    batched forward per step (instead of one 4-token `generate` per
    sample). `steps` is still the TOTAL number of sampled steps; the
    metric definition is unchanged.
    """
    act_ids = action_token_ids(tok)
    envs = [TextEnv(env_cfg) for _ in range(trajs)]
    prevs = [NONE_ACTION] * trajs
    for i, e in enumerate(envs):
        e.reset(seed=seed + i)
    match, illegal, total = 0, 0, 0
    per_traj = steps // trajs
    for _ in range(per_traj):
        tas = [_teacher_greedy(teacher, e.env, p, device)
               for e, p in zip(envs, prevs)]
        prompts = [prompt_ids(tok, e.env, e.prev_action) for e in envs]
        acts, legals = greedy_with_legality(student, tok, prompts, act_ids,
                                            device, batch_size)
        for i, (e, ta, a, lg) in enumerate(zip(envs, tas, acts, legals)):
            if not lg:
                illegal += 1      # 模型本来想说别的 / meant to say something else
            elif a == ta:
                match += 1
            total += 1
            _, _, done, _ = e.step(ACTION_WORDS[ta])  # 教师驱动 / teacher-driven
            prevs[i] = ta
            if done:
                e.reset(seed=seed + 1000 + i)
                prevs[i] = NONE_ACTION
    return {"steps": total, "trajs": trajs,
            "agreement": match / max(1, total),
            "illegal_rate": illegal / max(1, total)}


# ------------------------------------------------------- 探针 2：存活
def survival_probe(student, tok, env_cfg: EnvConfig, episodes: int = 20,
                   device: str = "cpu", seed: int = 9_000,
                   batch_size: int = 32) -> Dict:
    """学生自主行动的存活率 + 非法输出率。
    The student acting on its own: survival + illegal-output rates.

    加速：所有回合**并行**推进——每一步把所有还活着的观测合成一个
    batch，一次前向决定所有动作（原来是逐局逐条 generate）。
    Speed-up: all episodes advance IN PARALLEL — each step batches the
    observations of every still-alive episode into one forward pass
    (instead of one `generate` per episode per step).
    """
    act_ids = action_token_ids(tok)
    envs = [TextEnv(env_cfg) for _ in range(episodes)]
    for i, e in enumerate(envs):
        e.reset(seed=seed + i)
    alive = [True] * episodes
    lens = [0] * episodes
    died = [False] * episodes
    illegal = 0
    total_steps = 0
    while any(alive):
        idx = [i for i, a in enumerate(alive) if a]
        prompts = [prompt_ids(tok, envs[i].env, envs[i].prev_action)
                   for i in idx]
        acts, legals = greedy_with_legality(student, tok, prompts, act_ids,
                                            device, batch_size)
        for i, a, lg in zip(idx, acts, legals):
            if not lg:
                illegal += 1
            _, _, done, info = envs[i].step(ACTION_WORDS[a])
            lens[i] += 1
            total_steps += 1
            if done:
                alive[i] = False
                died[i] = bool(info.get("died"))
            elif lens[i] >= env_cfg.max_steps:
                alive[i] = False
    return {"episodes": episodes,
            "survival_rate": float(np.mean([not d for d in died])),
            "mean_steps": float(np.mean(lens)),
            "illegal_rate": illegal / max(1, total_steps)}


# ------------------------------------------------------- 探针 3：内感受调制 ★
def _energy_curve(student, tok, env_cfg: EnvConfig, device: str, trials: int,
                  energies, batch_size: int = 32, seed: int = 50_000,
                  integrity: float = 0.9) -> Dict[str, float]:
    """给定能量水平下"朝食物（right）"的比例（批量一次算完）。
    Rate of choosing 'right' (toward food) at given energy levels,
    computed in one batched pass."""
    act_ids = action_token_ids(tok)
    tenv = TextEnv(env_cfg)
    prompts: List[List[int]] = []
    tags: List[str] = []
    for energy in energies:
        for t in range(trials):
            tenv.reset(seed=seed + t)
            e = tenv.env
            r, c = e.agent
            e.hazards, e.foods = set(), {(r, c + 3)}
            e.energy, e.integrity, e.t = energy, integrity, 0
            e.last_obs = e.observation()
            prompts.append(prompt_ids(tok, e, None))
            tags.append(energy)
    acts, _ = greedy_with_legality(student, tok, prompts, act_ids, device,
                                   batch_size)
    totals: Dict[float, int] = {e: 0 for e in energies}
    toward: Dict[float, int] = {e: 0 for e in energies}
    for e_, a in zip(tags, acts):
        totals[e_] += 1
        toward[e_] += int(a == 3)  # right
    return {e: toward[e] / max(1, totals[e]) for e in energies}


def modulation_probe(student, tok, env_cfg: EnvConfig, trials: int = 40,
                     device: str = "cpu", batch_size: int = 32) -> Dict:
    """核心探针：同样视野 + 不同身体状态 -> 动作分布差异。
    The core probe: same view + different body states -> how do the
    action distributions differ?

    场景：食物可见于右侧 3 格。比较低能量(0.3)与饱腹(设定点)下
    学生选择 right（朝食物）的比例。饥饿-饱腹差 = modulation_gap。
    Scenario: food visible 3 cells to the right. Compare the rate of
    choosing right (toward food) under low energy (0.3) vs satiation
    (setpoint). The difference = modulation_gap.

    批量实现：2×trials 个状态一次前向算完（原来逐条 generate）。
    Batched: 2x trials states in one pass (was one generate per state).
    """
    rates = _energy_curve(student, tok, env_cfg, device, trials,
                          (0.3, env_cfg.energy_setpoint), batch_size)
    hungry, satiated = rates[0.3], rates[env_cfg.energy_setpoint]
    return {"trials": trials,
            "hungry_toward_food": hungry,
            "satiated_toward_food": satiated,
            "modulation_gap": hungry - satiated}


# ------------------------------------------- 探针 3b：内感受剂量-反应曲线
def modulation_sweep(student, tok, env_cfg: EnvConfig, trials: int = 20,
                     device: str = "cpu", batch_size: int = 32,
                     energies=(0.2, 0.4, 0.6, 0.8, 1.0)) -> Dict:
    """调制探针的剂量-反应版：扫一条能量梯度，看"朝食物"比例如何随
    内感受状态变化。
    Dose-response version of the modulation probe: sweep an energy
    gradient and watch how the toward-food rate tracks the
    interoceptive state.

    单个 modulation_gap 只给一个差值；曲线能区分三种失败模式：
    - 全程平（无调制）
    - 单调下降（反向调制，v2 就是这样）
    - 单调上升（正确调制——越饿越想找食物）
    A single gap gives one difference; the curve distinguishes three
    failure modes: flat (no modulation), monotonically decreasing
    (inverted — v2), monotonically increasing (correct — the hungrier,
    the more it heads for food).
    """
    rates_by_e = _energy_curve(student, tok, env_cfg, device, trials,
                               energies, batch_size)
    curve = {f"{e:.2f}": rates_by_e[e] for e in energies}
    rates = list(curve.values())
    # 能量升序下"朝食物"比例的斜率（>0 = 正确调制）
    # slope of the toward-food rate over ascending energy
    slope = float(np.polyfit(list(energies), rates, 1)[0]) if len(rates) > 1 else 0.0
    return {"trials": trials, "energies": list(energies),
            "toward_food_by_energy": curve,
            "slope": slope,
            "monotone_increasing": bool(all(
                rates[i] <= rates[i + 1] for i in range(len(rates) - 1)))}


def _verify_batching(student, tok, env_cfg: EnvConfig, device: str,
                     n: int, batch_size: int) -> None:
    """等价性校验：批量首 token argmax vs 逐条 generate 的决策一致率。
    Equivalence check: batched first-token argmax vs per-sample
    generate. Reports the agreement rate; a low rate means the batched
    speed-up is NOT measuring the same policy and must not be used."""
    act_ids = action_token_ids(tok)
    tenv = TextEnv(env_cfg)
    prompts, envs = [], []
    for t in range(n):
        tenv.reset(seed=50_000 + t)
        e = tenv.env
        r, c = e.agent
        e.hazards, e.foods = set(), {(r, c + 3)}
        e.energy, e.integrity, e.t = 0.3 + 0.1 * (t % 5), 0.9, 0
        e.last_obs = e.observation()
        prompts.append(prompt_ids(tok, e, None))
        envs.append(e)
    fast, _ = greedy_with_legality(student, tok, prompts, act_ids, device,
                                   batch_size)
    slow = []
    for e in envs:
        w = _student_action(student, tok, e, None, device)
        a = WORD_TO_ACTION.get(w.strip().lower().strip("."))
        slow.append(-1 if a is None else a)
    both = [(f, s) for f, s in zip(fast, slow) if s >= 0]
    agree = sum(int(f == s) for f, s in both) / max(1, len(both))
    print(json.dumps({"n": n, "compared": len(both), "agreement": agree,
                      "fast": list(fast), "slow": list(slow)},
                     ensure_ascii=False))
    print("BATCHING OK" if agree >= 0.99 else
          "BATCHING MISMATCH — do not use the fast path")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="阶段B行为探针 / Stage-B behavioral probes")
    ap.add_argument("--ckpt", type=str, default="runs/sft-lora")
    ap.add_argument("--teacher", type=str,
                    default="../exp-a-homeostatic-rl/runs/curriculum/checkpoint.pt")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--only", type=str, default=None,
                    choices=["agreement", "survival", "modulation",
                             "modulation-sweep"],
                    help="只跑一个探针（主机内存紧张时分批跑；默认跑全部）"
                         " / run a single probe (for tight-memory machines; "
                         "default: run all)")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="批量决策的批大小 / batched decision batch size")
    ap.add_argument("--verify-batching", type=int, default=0, metavar="N",
                    help="在 N 个状态上对比\"批量首 token\"与\"逐条 generate\""
                         " 的决策一致率（批量加速的等价性校验）"
                         " / compare batched first-token vs per-sample "
                         "generate on N states (equivalence check)")
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from .gen_data import load_teacher
    teacher, env_cfg = load_teacher(Path(args.teacher), "cpu")
    student, tok, device = load_student(Path(args.ckpt), device)
    print(f"student device: {device}", flush=True)
    bs = args.batch_size

    if args.verify_batching:
        _verify_batching(student, tok, env_cfg, device, args.verify_batching,
                         bs)
        return

    probes = {
        "agreement": lambda: agreement_probe(
            student, tok, teacher, env_cfg, device=device, batch_size=bs),
        "survival": lambda: survival_probe(
            student, tok, env_cfg, device=device, batch_size=bs),
        "modulation": lambda: modulation_probe(
            student, tok, env_cfg, device=device, batch_size=bs),
        "modulation-sweep": lambda: modulation_sweep(
            student, tok, env_cfg, device=device, batch_size=bs),
    }
    keys = [args.only] if args.only else list(probes)
    report = {"checkpoint": args.ckpt}
    out = Path(args.ckpt) / "probe_report.json"
    if out.exists():      # 分批跑时保留已有结果 / keep earlier batches
        try:
            report.update(json.loads(out.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass
    for k in keys:
        report[k] = probes[k]()
        print(f"[{k}] {json.dumps(report[k], ensure_ascii=False)}", flush=True)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
