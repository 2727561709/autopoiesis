"""PPO：让内稳态回报（活/死）直接进策略。
PPO: put the homeostatic reward (alive / dead) straight into the policy.

这是 v2 诊断之后的第二条路。v2 证明了 SFT 的天花板：行为克隆能拟合
动作分布，却学不到"内感受→行动"的**因果方向**（调制 gap = -1.0）。
v3 用 DPO 的偏好对绕开行为克隆；这条路线更彻底——**梯度直接来自环境**：

    rollout：学生自己在 TextEnv 里玩 → 拿到内稳态回报
    update ：PPO-clip 更新 LoRA，让高回报动作的 token 概率上升

This is the second route after the v2 diagnosis. v2 exposed the SFT
ceiling: behavior cloning fits the action distribution but cannot learn
the CAUSAL DIRECTION of interoception -> action (modulation gap -1.0).
v3 (DPO) sidesteps cloning with preference pairs; this route goes
further — the gradient comes straight from the environment:

    rollout: the student plays TextEnv itself -> homeostatic reward
    update : PPO-clip on the LoRA, raising the token probability of
             high-reward actions

设计要点 / design notes
----------------------
1. **动作空间 = 5 个动作词的首 token**。对每个观测前向一次，取末位
   logits，只在 5 个动作 token 上做 softmax（重归一化）。这与完整词表
   分布略有差别——被排除的概率质量不计入——但对"离散动作集上的 RL"
   是正确的做法，也让熵/采样可控。
   The action space is the FIRST TOKEN of each of the 5 action words.
   One forward pass per observation; take the last-position logits and
   softmax over the 5 action tokens only (renormalised). This differs
   slightly from the full-vocab distribution (the excluded mass is not
   counted) but is the right thing for RL over a discrete action set
   and keeps entropy and sampling well-behaved.

2. **没有 critic**：4GB 显存下不再塞一个价值网络。优势用"整批回报
   标准化"（returns - mean，除以 std），即 REINFORCE-with-baseline +
   PPO-clip。代价：方差比带 critic 的 GAE 高；好处：单模型即可训练，
   显存与 DPO 相当（~1.2GB）。
   No critic: 4GB VRAM leaves no room for a value net. Advantages are
   batch-normalised returns (returns - mean, / std) — REINFORCE-with-
   baseline plus the PPO clip. Cost: higher variance than GAE with a
   critic. Benefit: a single model trains, VRAM comparable to DPO.

3. **复用 DPO 的势函数整形**：内稳态回报本身是稀疏的（吃到/受伤才有
   差异），直接用会让早期学习极慢。`gen_dpo_data` 的 PBRS 势函数
   （饥饿门控的 -dist(食物) / 受伤门控的 +dist(危险)）在此复用——
   它理论不改变最优策略，只把信号变稠密。
   Reuses DPO's potential shaping: the homeostatic reward is sparse
   (only eating / injury differ), which makes early learning very slow
   if used raw. The PBRS potential from `gen_dpo_data` (hunger-gated
   -dist(food), injury-gated +dist(hazard)) is reused here — provably
   preserves the optimal policy while densifying the signal.

4. **不做 KL 惩罚**（省一份参考模型），靠 entropy bonus + 小 lr 稳定。
   No KL penalty (that would need a second reference model); stability
   comes from the entropy bonus and a small lr.

用法 / usage:
    python -m homeo_lm.ppo --ckpt runs/dpo-v3 --out runs/ppo-v4
    python -m homeo_lm.ppo --ckpt runs/sft-lora-v2 --out runs/ppo-v4 --no-shaping
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from homeo_rl.env import EnvConfig

from .acthead import act_batch, action_token_ids
from .gen_dpo_data import decide_gates, potential
from .sft import DEFAULT_MODEL
from .textenv import ACTION_WORDS, TextEnv, obs_messages


# ------------------------------------------------------------------ rollout
class EnvBatch:
    """N 个并行文本环境（各自独立播种）。
    N parallel text environments, each seeded independently."""

    def __init__(self, env_cfg: EnvConfig, n: int, seed: int = 0) -> None:
        self.envs = [TextEnv(env_cfg) for _ in range(n)]
        self.n = n
        for i, e in enumerate(self.envs):
            e.reset(seed=seed * 1000 + i)

    def reset_one(self, i: int, seed: int) -> None:
        self.envs[i].reset(seed=seed)

    def prompts(self, tok) -> List[List[int]]:
        out = []
        for e in self.envs:
            msgs = obs_messages(e.env, e.prev_action)
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
            out.append(tok(text, add_special_tokens=False)["input_ids"])
        return out


# 行动头（首 token 分类）与探针共用同一份实现，见 acthead.py
# The action head (first-token classification) is shared with the
# probes — see acthead.py


def compute_returns(rewards: Sequence[float], dones: Sequence[bool],
                    gamma: float) -> List[float]:
    """按回合边界截断的折扣回报。Discounted returns, cut at episode ends."""
    out = [0.0] * len(rewards)
    running = 0.0
    for i in reversed(range(len(rewards))):
        running = rewards[i] + (0.0 if dones[i] else gamma * running)
        out[i] = running
    return out


# ------------------------------------------------------------------- update
def ppo_update(model, tok, batch: Dict[str, list], act_ids: Sequence[int],
               opt, device: str, clip: float = 0.2, entropy_coef: float = 0.01,
               minibatch: int = 32, epochs: int = 2) -> Dict[str, float]:
    """PPO-clip 更新（末位 token 的 5 动作分布）。
    PPO-clip update over the last-position 5-action distribution."""
    prompts = batch["prompts"]
    actions = torch.tensor(batch["actions"], device=device)
    old_lp = torch.tensor(batch["logps"], device=device)
    adv = torch.tensor(batch["adv"], dtype=torch.float32, device=device)
    n = len(prompts)
    idx = list(range(n))
    random.shuffle(idx)
    stats = {"loss": 0.0, "clip_frac": 0.0, "entropy": 0.0, "n": 0}
    model.train()
    for _ in range(epochs):
        for s in range(0, n, minibatch):
            sel = idx[s: s + minibatch]
            sub_prompts = [prompts[i] for i in sel]
            maxlen = max(len(p) for p in sub_prompts)
            ids, attn = [], []
            for p in sub_prompts:
                pad = maxlen - len(p)
                ids.append(list(p) + [tok.pad_token_id] * pad)
                attn.append([1] * len(p) + [0] * pad)
            ids_t = torch.tensor(ids, device=device)
            attn_t = torch.tensor(attn, device=device)
            out = model(input_ids=ids_t, attention_mask=attn_t)
            last = torch.tensor([len(p) - 1 for p in sub_prompts],
                                device=device)
            logits = out.logits[torch.arange(len(sub_prompts), device=device),
                                last]
            logp = F.log_softmax(logits[:, list(act_ids)].float(), dim=-1)
            a = actions[sel]
            new_lp = logp.gather(1, a.unsqueeze(1)).squeeze(1)
            entropy = -(logp.exp() * logp).sum(dim=1).mean()

            ratio = (new_lp - old_lp[sel]).exp()
            a_adv = adv[sel]
            unclipped = ratio * a_adv
            clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * a_adv
            loss = -torch.min(unclipped, clipped).mean() - entropy_coef * entropy

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), 1.0)
            opt.step()

            stats["loss"] += loss.item()
            stats["clip_frac"] += (ratio - 1.0).abs().gt(clip).float().mean().item()
            stats["entropy"] += entropy.item()
            stats["n"] += 1
    k = max(1, stats["n"])
    return {k_: v / k for k_, v in stats.items() if k_ != "n"}


# --------------------------------------------------------------------- main
def train(ckpt: Optional[Path], out_dir: Path, model_name: str = DEFAULT_MODEL,
          updates: int = 300, envs: int = 4, steps_per_update: int = 64,
          lr: float = 2e-5, gamma: float = 0.99, clip: float = 0.2,
          entropy_coef: float = 0.01, shape_w: float = 0.05,
          shaping: bool = True, max_len: int = 512, device: str = "cuda",
          resume: bool = False, snapshot_every: int = 25,
          seed: int = 0, probe_every: int = 0, probe_trials: int = 20) -> Path:
    """PPO 主循环。The PPO main loop."""
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "train_state.json"
    start_update = 0

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    act_ids = action_token_ids(tok)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    if ckpt is not None and Path(ckpt).exists():
        # 从已有适配器出发（DPO 或 SFT 的产物）/ start from an existing
        # adapter (the output of DPO or SFT)
        model = PeftModel.from_pretrained(model, str(ckpt), is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()
    model.to(device)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr)

    if resume and state_path.exists():
        st = json.loads(state_path.read_text(encoding="utf-8"))
        start_update = st.get("update", 0)
        print(f"resume from update {start_update}")

    # 环境：教师配置决定网格/成本/设定点（与探针同一套规则）
    # Env: the teacher config fixes grid / costs / setpoints (same rules
    # the probes use).
    env_cfg = EnvConfig(seed=seed)
    batch_env = EnvBatch(env_cfg, envs, seed=seed)
    rng = random.Random(seed)
    ep_seed = 10_000 + seed * 100
    hist: List[Dict] = []          # 训练中的探针轨迹 / in-training probe trace

    for u in range(start_update, updates):
        prompts_all: List[List[int]] = []
        actions_all: List[int] = []
        logps_all: List[float] = []
        rewards_all: List[float] = []
        dones_all: List[bool] = []
        ep_lens: List[int] = []
        ep_deaths = 0
        ep_count = 0
        ep_len_now = [0] * envs      # 每环境当前回合已走步数

        model.eval()
        for _ in range(steps_per_update):
            prompts = batch_env.prompts(tok)
            acts, logps, _ = act_batch(model, tok, prompts, act_ids, device)
            step_dones: List[bool] = []
            step_deaths: List[bool] = []
            for i in range(envs):
                tenv = batch_env.envs[i]
                gates = decide_gates(tenv.env) if shaping else None
                pre_phi = potential(tenv.env, gates) if shaping else 0.0
                word = ACTION_WORDS[int(acts[i])]
                _, r, done, info = tenv.step(word)
                if shaping and not info.get("ate"):
                    # 与 DPO 偏好对同一套 PBRS（进食步不整形）
                    # same PBRS as the DPO pairs (eat steps skip shaping)
                    post = potential(tenv.env, gates)
                    r = r + shape_w * (post - pre_phi)
                prompts_all.append(prompts[i])
                actions_all.append(int(acts[i]))
                logps_all.append(float(logps[i]))
                rewards_all.append(float(r))
                dones_all.append(bool(done))
                step_dones.append(bool(done))
                step_deaths.append(bool(info.get("died")))
                ep_len_now[i] += 1
            # 结束的回合：记账后重开 / finished episodes: record, then restart
            for i, d in enumerate(step_dones):
                if d:
                    ep_lens.append(ep_len_now[i])
                    ep_deaths += int(step_deaths[i])
                    ep_count += 1
                    ep_len_now[i] = 0
                    ep_seed += 1
                    batch_env.reset_one(i, ep_seed)

        # 回报 -> 优势（整批标准化，无 critic）
        # returns -> advantages (batch-normalised, no critic)
        returns = compute_returns(rewards_all, dones_all, gamma)
        arr = np.array(returns, dtype=np.float64)
        adv = (arr - arr.mean()) / (arr.std() + 1e-8)
        batch = {"prompts": prompts_all, "actions": actions_all,
                 "logps": logps_all, "adv": adv.tolist()}
        stats = ppo_update(model, tok, batch, act_ids, opt, device,
                           clip=clip, entropy_coef=entropy_coef)

        if (u + 1) % 5 == 0 or u == 0:
            dist = Counter(ACTION_WORDS[a] for a in actions_all)
            share = {k: round(v / len(actions_all), 3) for k, v in dist.items()}
            top = max(dist.values()) / len(actions_all)
            flag = " <<< COLLAPSING" if top > 0.9 else ""
            surv = (1.0 - ep_deaths / ep_count) if ep_count else float("nan")
            mean_len = float(np.mean(ep_lens)) if ep_lens else float("nan")
            print(f"update {u+1}/{updates} "
                  f"ret {arr.mean():+.3f} "
                  f"loss {stats['loss']:+.4f} "
                  f"ent {stats['entropy']:.3f} "
                  f"clip {stats['clip_frac']:.2f} "
                  f"eps {ep_count} surv {surv:.2f} len {mean_len:.0f}"
                  f"{flag}", flush=True)
            print(f"   actions {share}", flush=True)

        # 密集闭环：训练中定期插一次轻量探针（模型已在内存里，几乎免费）
        # dense loop: a light probe during training (the model is already
        # in memory, so it is nearly free)
        if probe_every and ((u + 1) % probe_every == 0 or u == 0):
            gap = _inline_probe(model, tok, env_cfg, device, probe_trials)
            print(f"   probe modulation_gap {gap:+.3f}", flush=True)
            hist.append({"update": u + 1, "modulation_gap": gap})

        if snapshot_every and (u + 1) % snapshot_every == 0:
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
            state_path.write_text(json.dumps({"update": u + 1}),
                                  encoding="utf-8")
            print(f"snapshot saved at update {u+1}", flush=True)

    model.eval()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    if hist:
        (out_dir / "probe_trace.json").write_text(
            json.dumps(hist, indent=2), encoding="utf-8")
    print(f"ppo adapter saved: {out_dir}")
    return out_dir


@torch.no_grad()
def _inline_probe(model, tok, env_cfg, device: str, trials: int) -> float:
    """训练中的轻量调制探针（复用探针实现，模型已在内存里）。
    A light in-training modulation probe (reuses the probe
    implementation; the model is already in memory)."""
    from .probe import modulation_probe
    res = modulation_probe(model, tok, env_cfg, device=device,
                           trials=trials)
    return float(res.get("modulation_gap", float("nan")))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="PPO：内稳态回报直入策略 / PPO: homeostatic reward "
                    "straight into the policy")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="起点适配器（DPO/SFT 产物）；空=新建 LoRA / "
                         "starting adapter (DPO/SFT output); empty = new LoRA")
    ap.add_argument("--out", type=str, default="runs/ppo-v4")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--steps-per-update", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--shape-w", type=float, default=0.05)
    ap.add_argument("--no-shaping", action="store_true",
                    help="关闭势函数整形（用裸内稳态回报）/ disable "
                         "potential shaping (use the raw reward)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--snapshot-every", type=int, default=25)
    ap.add_argument("--probe-every", type=int, default=0,
                    help="每 N 次 update 插一次轻量调制探针（0=关闭）"
                         " / run a light modulation probe every N updates "
                         "(0 = off)")
    ap.add_argument("--probe-trials", type=int, default=20,
                    help="训练中探针的试验数（比正式探针小）"
                         " / trials for the in-training probe")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: PPO on CPU is extremely slow.")
    train(Path(args.ckpt) if args.ckpt else None, Path(args.out), args.model,
          args.updates, args.envs, args.steps_per_update, args.lr, args.gamma,
          args.clip, args.entropy_coef, args.shape_w, not args.no_shaping,
          device=device, resume=args.resume,
          snapshot_every=args.snapshot_every, seed=args.seed,
          probe_every=args.probe_every, probe_trials=args.probe_trials)


if __name__ == "__main__":
    main()
