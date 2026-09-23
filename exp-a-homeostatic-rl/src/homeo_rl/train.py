"""阶段 A 训练入口：PPO + 内稳态奖励 + 预测误差内在奖励。

总奖励 r = r_homeostatic + intrinsic_coef * clip(pe / EMA(pe), 0, pe_clip)

    r_homeostatic —— 缺口缩减（唯一外部回报，来自环境）
    pe            —— 世界模型预测误差（好奇式内在驱动，自我归一化）

用法：
    python -m homeo_rl.train --updates 500 --out runs/base
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .env import EnvConfig, HomeostaticGridWorld, N_ACTIONS
from .model import HomeoActorCritic

NONE_ACTION = N_ACTIONS  # 上一步动作 embedding 的"空"索引


@dataclass
class PPOConfig:
    updates: int = 500
    num_envs: int = 8
    rollout: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 256
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    wm_coef: float = 0.5          # 世界模型辅助损失权重
    intrinsic_coef: float = 0.02  # 预测误差内在奖励权重
    pe_clip: float = 5.0          # 归一化预测误差上限
    pe_ema: float = 0.99          # 预测误差水平的指数滑动平均
    grad_norm: float = 0.5
    seed: int = 0
    # 课程：食物密度退火（充裕 → 稀缺）
    food_anneal: bool = False
    food_start: int = 8
    food_end: int = 2
    # 模型尺寸
    d_model: int = 128
    nhead: int = 4
    n_layers: int = 4
    ff_dim: int = 256


class _Rollout:
    """单个 update 周期的轨迹缓存。"""

    def __init__(self) -> None:
        self.vision: List[np.ndarray] = []
        self.intero: List[np.ndarray] = []
        self.prev_a: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.logp: List[np.ndarray] = []
        self.values: List[np.ndarray] = []
        self.rewards: List[np.ndarray] = []   # 总奖励（内稳态 + 内在）
        self.dones: List[np.ndarray] = []
        self.next_vision: List[np.ndarray] = []
        self.next_intero: List[np.ndarray] = []

    def add(self, **kw: np.ndarray) -> None:
        for k, v in kw.items():
            getattr(self, k).append(v)

    def stack(self, key: str) -> np.ndarray:
        return np.stack(getattr(self, key))  # (T, N, ...)

    def flat(self, key: str) -> np.ndarray:
        arr = np.asarray(getattr(self, key))  # (T, N, ...)
        return arr.reshape(-1, *arr.shape[2:])


def _compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
                 last_value: np.ndarray, gamma: float, lam: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """GAE，逐环境列计算。rewards/values/dones: (T, N)；返回 (adv, returns)。"""
    T, N = rewards.shape
    adv = np.zeros((T, N), dtype=np.float32)
    lastgaelam = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t].astype(np.float32)
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
    return adv, adv + values


def train(ppo_cfg: PPOConfig,
          env_cfg: Optional[EnvConfig] = None,
          out_dir: Path = Path("runs/exp-a"),
          device: Optional[str] = None) -> Path:
    """运行完整训练，返回 metrics.jsonl 路径。"""
    cfg = ppo_cfg
    env_cfg = env_cfg or EnvConfig()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    envs = [HomeostaticGridWorld(
        EnvConfig(**{**asdict(env_cfg), "seed": cfg.seed * 1000 + i}))
        for i in range(cfg.num_envs)]

    model = HomeoActorCritic(view=env_cfg.view, d_model=cfg.d_model,
                             nhead=cfg.nhead, n_layers=cfg.n_layers,
                             ff_dim=cfg.ff_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    metrics_path = out_dir / "metrics.jsonl"
    ckpt_path = out_dir / "checkpoint.pt"
    save_every = max(1, cfg.updates // 10)

    ema_pe: Optional[float] = None
    episodes: List[Dict] = []
    pending_ret = [0.0] * cfg.num_envs  # 各环境的回合累计内稳态回报
    total_steps = 0
    t0 = time.time()

    cur_n_food = env_cfg.n_food
    for update in range(1, cfg.updates + 1):
        if cfg.food_anneal:
            frac = (update - 1) / max(1, cfg.updates - 1)
            cur_n_food = int(round(cfg.food_start
                                   + (cfg.food_end - cfg.food_start) * frac))
            for e in envs:  # 在下次 reset 时生效
                e.cfg.n_food = cur_n_food
        ro = _Rollout()
        prev_a = np.full(cfg.num_envs, NONE_ACTION, dtype=np.int64)

        for _ in range(cfg.rollout):
            v = np.stack([e.last_obs["vision"] for e in envs])
            i = np.stack([e.last_obs["intero"] for e in envs])
            tv = torch.from_numpy(v).to(device)
            ti = torch.from_numpy(i).to(device)
            tpa = torch.from_numpy(prev_a).to(device)
            pa_used = prev_a.copy()  # 本步采样所用的上一步动作

            out = model(tv, ti, tpa)
            dist = Categorical(logits=out["logits"])
            a = dist.sample()
            logp = dist.log_prob(a)

            a_np = a.cpu().numpy()
            r_homeo = np.zeros(cfg.num_envs, dtype=np.float32)
            done = np.zeros(cfg.num_envs, dtype=bool)
            for k, e in enumerate(envs):
                _, r, d, info = e.step(int(a_np[k]))
                pending_ret[k] += r
                if d:
                    episodes.append({
                        "ret": pending_ret[k],
                        "len": e.t,
                        "energy": info["energy"],
                        "integrity": info["integrity"],
                        "died": info["died"],
                    })
                    e.reset()
                    prev_a[k] = NONE_ACTION
                    pending_ret[k] = 0.0
                else:
                    prev_a[k] = a_np[k]
                r_homeo[k] = r
                done[k] = d

            nv = np.stack([e.last_obs["vision"] for e in envs])
            ni = np.stack([e.last_obs["intero"] for e in envs])
            tnv = torch.from_numpy(nv).to(device)
            tni = torch.from_numpy(ni).to(device)

            with torch.no_grad():
                pe = model.prediction_error(out, tnv, tni).cpu().numpy()
            if ema_pe is None:
                ema_pe = float(pe.mean())
            intrinsic = cfg.intrinsic_coef * np.clip(
                pe / (ema_pe + 1e-8), 0.0, cfg.pe_clip)
            ema_pe = cfg.pe_ema * ema_pe + (1 - cfg.pe_ema) * float(pe.mean())

            ro.add(vision=v, intero=i, prev_a=pa_used,
                   actions=a_np, logp=logp.detach().cpu().numpy(),
                   values=out["value"].detach().cpu().numpy(),
                   rewards=(r_homeo + intrinsic).astype(np.float32),
                   dones=done, next_vision=nv, next_intero=ni)
            total_steps += cfg.num_envs

        # bootstrap 终值
        v = np.stack([e.last_obs["vision"] for e in envs])
        i = np.stack([e.last_obs["intero"] for e in envs])
        with torch.no_grad():
            last_value = model(torch.from_numpy(v).to(device),
                               torch.from_numpy(i).to(device),
                               torch.from_numpy(prev_a).to(device))["value"]
            last_value = last_value.cpu().numpy()

        values = ro.stack("values")
        adv, returns = _compute_gae(ro.stack("rewards"), values, ro.stack("dones"),
                                    last_value, cfg.gamma, cfg.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # -------------------------------------------------- PPO 更新
        n = cfg.num_envs * cfg.rollout
        idx = np.arange(n)
        losses = {"pg": 0.0, "v": 0.0, "wm": 0.0, "ent": 0.0}
        for _ in range(cfg.epochs):
            np.random.shuffle(idx)
            for s in range(0, n, cfg.minibatch):
                mb = idx[s:s + cfg.minibatch]
                out = model(torch.from_numpy(ro.flat("vision")[mb]).to(device),
                            torch.from_numpy(ro.flat("intero")[mb]).to(device),
                            torch.from_numpy(ro.flat("prev_a")[mb]).to(device))
                dist = Categorical(logits=out["logits"])
                logp_new = dist.log_prob(torch.from_numpy(
                    ro.flat("actions")[mb]).to(device))
                ratio = torch.exp(logp_new - torch.from_numpy(
                    ro.flat("logp")[mb]).to(device))
                mb_adv = torch.from_numpy(adv.reshape(-1)[mb]).to(device)
                pg = -torch.min(ratio * mb_adv,
                                ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * mb_adv).mean()
                v_loss = F.mse_loss(out["value"], torch.from_numpy(
                    returns.reshape(-1)[mb]).to(device))
                ent = dist.entropy().mean()
                tv, tni_t = model.wm_targets(
                    torch.from_numpy(ro.flat("next_vision")[mb]).to(device),
                    torch.from_numpy(ro.flat("next_intero")[mb]).to(device))
                wm = F.mse_loss(out["wm_vision"], tv) \
                    + F.mse_loss(out["wm_intero"], tni_t)
                loss = pg + cfg.value_coef * v_loss + cfg.wm_coef * wm \
                    - cfg.entropy_coef * ent
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm)
                opt.step()
                _denom = cfg.epochs * max(1, (n - 1) // cfg.minibatch + 1)
                losses["pg"] += pg.item() / _denom
                losses["v"] += v_loss.item() / _denom
                losses["wm"] += wm.item() / _denom
                losses["ent"] += ent.item() / _denom

        # -------------------------------------------------- 日志
        recent = episodes[-50:]
        if recent:
            ep_ret = float(np.mean([e["ret"] for e in recent]))
            ep_len = float(np.mean([e["len"] for e in recent]))
            survival = float(np.mean([not e["died"] for e in recent]))
            energy = float(np.mean([e["energy"] for e in recent]))
            integrity = float(np.mean([e["integrity"] for e in recent]))
        else:
            ep_ret = ep_len = survival = energy = integrity = float("nan")
        row = {
            "update": update, "steps": total_steps,
            "ep_ret": ep_ret, "ep_len": ep_len, "survival": survival,
            "energy": energy, "integrity": integrity,
            "pe_ema": ema_pe,
            "pg_loss": losses["pg"], "v_loss": losses["v"],
            "wm_loss": losses["wm"], "entropy": losses["ent"],
            "elapsed_s": round(time.time() - t0, 1),
        }
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if update % 10 == 0 or update == cfg.updates:
            print(f"[{update:>4}/{cfg.updates}] steps={total_steps} "
                  f"ret={ep_ret:+.2f} len={ep_len:.0f} surv={survival:.2f} "
                  f"pe={ema_pe:.4f} ({row['elapsed_s']}s)", flush=True)
        if update % save_every == 0 or update == cfg.updates:
            torch.save({
                "model": model.state_dict(),
                "ppo_cfg": asdict(cfg),
                "env_cfg": {**asdict(env_cfg), "n_food": cur_n_food},
                "update": update,
            }, ckpt_path)

    return metrics_path


def main() -> None:
    ap = argparse.ArgumentParser(description="阶段A：内稳态RL + 预测误差辅助")
    ap.add_argument("--updates", type=int, default=500)
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--intrinsic-coef", type=float, default=0.02)
    ap.add_argument("--wm-coef", type=float, default=0.5)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--n-food", type=int, default=8)
    ap.add_argument("--n-hazard", type=int, default=6)
    ap.add_argument("--food-anneal", action="store_true",
                    help="课程学习：食物密度从 --food-start 线性退火到 --food-end")
    ap.add_argument("--food-start", type=int, default=8)
    ap.add_argument("--food-end", type=int, default=2)
    ap.add_argument("--coma", action="store_true",
                    help="休克模式：能量归零不死而昏迷（对照实验）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="runs/exp-a")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    args = ap.parse_args()

    cfg = PPOConfig(updates=args.updates, num_envs=args.num_envs,
                    rollout=args.rollout, lr=args.lr,
                    intrinsic_coef=args.intrinsic_coef, wm_coef=args.wm_coef,
                    entropy_coef=args.entropy_coef,
                    food_anneal=args.food_anneal, food_start=args.food_start,
                    food_end=args.food_end,
                    seed=args.seed, d_model=args.d_model, n_layers=args.n_layers)
    env_cfg = EnvConfig(n_food=args.n_food, n_hazard=args.n_hazard,
                        coma_mode=args.coma)
    path = train(cfg, env_cfg=env_cfg, out_dir=Path(args.out), device=args.device)
    print(f"metrics: {path}")


if __name__ == "__main__":
    main()
