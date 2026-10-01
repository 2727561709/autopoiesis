"""教师轨迹 → SFT 数据集。
Teacher trajectories -> SFT dataset.

教师 = exp-a 训练好的内稳态策略（推荐课程组 checkpoint）。
The teacher = a trained exp-a homeostatic policy (the curriculum
checkpoint is recommended).

每条样本 = (观测文本, 教师动作词)。内感受读数在观测里，动作分布
由教师按身体状态决定——"饿时觅食、饱时无视"全部由数据携带。
Each example = (observation text, teacher action word). Interoception
is inside the observation; the action distribution follows the
teacher's body state — "forage when hungry, ignore when full" is
carried entirely by the data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from homeo_rl.env import EnvConfig, N_ACTIONS
from homeo_rl.model import HomeoActorCritic

from .textenv import ACTION_WORDS, TextEnv, obs_messages

NONE_ACTION = N_ACTIONS


def load_teacher(ckpt_path: Path, device: str = "cpu"):
    """载入 exp-a 教师 checkpoint。Load an exp-a teacher checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    env_cfg = EnvConfig(**ckpt["env_cfg"])
    model = HomeoActorCritic(view=env_cfg.view, in_ch=env_cfg.n_channels).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, env_cfg


def _teacher_action(model, env, prev_a: int, device: str) -> int:
    """教师贪心动作。The teacher's greedy action."""
    v = torch.from_numpy(env.last_obs["vision"]).unsqueeze(0).to(device)
    i = torch.from_numpy(env.last_obs["intero"]).unsqueeze(0).to(device)
    pa = torch.tensor([prev_a], dtype=torch.long, device=device)
    a, _, _ = model.act(v, i, pa, deterministic=True)
    return int(a.item())


def generate(ckpt_path: Path, out_path: Path, episodes: int = 200,
             max_steps: int = 400, device: str = "cpu",
             seed: int = 0) -> dict:
    """回放教师轨迹并写出 SFT jsonl。Roll out the teacher and write the
    SFT jsonl."""
    model, env_cfg = load_teacher(ckpt_path, device)
    tenv = TextEnv(env_cfg)
    rows = []
    deaths = 0
    for ep in range(episodes):
        tenv.reset(seed=seed * 10_000 + ep)
        prev_a = NONE_ACTION
        for t in range(max_steps):
            obs = obs_messages(tenv.env, tenv.prev_action)
            a = _teacher_action(model, tenv.env, prev_a, device)
            rows.append({
                "messages": [
                    {"role": "system", "content": obs[0]["content"]},
                    {"role": "user", "content": obs[1]["content"]},
                    {"role": "assistant", "content": ACTION_WORDS[a]},
                ],
                "meta": {"ep": ep, "t": t},
            })
            _, r, done, info = tenv.step(ACTION_WORDS[a])
            prev_a = a
            if done:
                deaths += int(info["died"])
                break
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats = {
        "episodes": episodes, "examples": len(rows),
        "teacher_death_rate": deaths / max(1, episodes),
        "mean_ep_len": len(rows) / max(1, episodes),
    }
    print(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description="教师轨迹 -> SFT 数据集 / teacher trajectories -> SFT dataset")
    ap.add_argument("--ckpt", type=str,
                    default="../exp-a-homeostatic-rl/runs/curriculum/checkpoint.pt")
    ap.add_argument("--out", type=str, default="data/sft.jsonl")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    generate(Path(args.ckpt), Path(args.out), args.episodes,
             args.max_steps, seed=args.seed)


if __name__ == "__main__":
    main()
