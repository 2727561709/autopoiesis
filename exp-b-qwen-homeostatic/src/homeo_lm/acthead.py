"""5 动作首 token 的行动头（探针与 PPO 共用）。
The 5-action first-token head (shared by the probes and PPO).

为什么不用 `model.generate`：动作只有一个词，而 5 个动作词的**首 token
互不相同**（`action_token_ids` 会断言这一点），所以"贪心解码一个动作词"
等价于"在 5 个动作首 token 上取 argmax"。差别是把一次决策从
4 次前向（`max_new_tokens=4`，逐条串行）压缩成**批内一次前向**——
存活探针 20 局 × 上千步的开销由此降到原来的约 1/200，这是 RSI Phase 0
（测量必须够快）的关键改造。
Why not `model.generate`: the action is a single word and the 5 action
words have distinct FIRST TOKENS (asserted by `action_token_ids`), so
"greedy-decode one action word" is equivalent to "argmax over the 5
action first tokens". The difference is that a decision drops from 4
forward passes (`max_new_tokens=4`, one sample at a time) to ONE forward
pass per batch — roughly a 200x cut for the survival probe (20 episodes
x up to 1000 steps). This is the key change for RSI Phase 0, where
measurement has to be fast.

分布是在 5 个动作 token 上**重归一化**的（词表其余概率质量不计入）。
对"离散动作集上的决策与 RL"这是正确做法，也让熵与采样可控。
The distribution is RENORMALISED over the 5 action tokens (the rest of
the vocab's mass is excluded) — the right thing for decisions and RL
over a discrete action set, and it keeps entropy and sampling sane.

等价性由 `probe.py --verify-batching` 实测校验：对同一批状态，比较
"首 token argmax" 与 "generate 后解码" 的决策一致率。
Equivalence is verified empirically by `probe.py --verify-batching`,
which compares first-token argmax against generate-and-decode on the
same states.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F


def action_token_ids(tok) -> List[int]:
    """5 个动作词的首 token id。First-token id of each action word."""
    from .textenv import ACTION_WORDS
    ids = [tok.encode(w, add_special_tokens=False)[0] for w in ACTION_WORDS]
    if len(set(ids)) != len(ids):
        raise ValueError(
            f"action words do not have distinct first tokens: "
            f"{dict(zip(ACTION_WORDS, ids))} — the 5-action distribution "
            f"would be ill-defined")
    return ids


def _right_pad(tok, prompts: Sequence[Sequence[int]], device: str):
    """右填充成方阵，返回 (ids, attn, last_pos)。
    Right-pad into a square batch; returns (ids, attn, last_pos)."""
    maxlen = max(len(p) for p in prompts)
    ids, attn = [], []
    for p in prompts:
        pad = maxlen - len(p)
        ids.append(list(p) + [tok.pad_token_id] * pad)
        attn.append([1] * len(p) + [0] * pad)
    last = torch.tensor([len(p) - 1 for p in prompts], device=device)
    return (torch.tensor(ids, device=device),
            torch.tensor(attn, device=device), last)


@torch.no_grad()
def act_batch(model, tok, prompts: Sequence[Sequence[int]],
              act_ids: Sequence[int], device: str,
              greedy: bool = False) -> Tuple[torch.Tensor, torch.Tensor,
                                             torch.Tensor]:
    """对一批观测给出动作。Returns (actions, log_probs, entropy).
    greedy=False 时按分布采样，True 时取 argmax。"""
    ids_t, attn_t, last = _right_pad(tok, prompts, device)
    out = model(input_ids=ids_t, attention_mask=attn_t)
    logits = out.logits[torch.arange(len(prompts), device=device), last]
    logp = F.log_softmax(logits[:, list(act_ids)].float(), dim=-1)
    if greedy:
        probs = torch.zeros_like(logp)
        probs.scatter_(1, logp.argmax(dim=1, keepdim=True), 1.0)
    else:
        probs = logp.exp()
    dist = torch.distributions.Categorical(probs=probs)
    actions = dist.sample()
    return actions, dist.log_prob(actions), dist.entropy()


@torch.no_grad()
def greedy_with_legality(model, tok, prompts: Sequence[Sequence[int]],
                         act_ids: Sequence[int], device: str,
                         batch_size: int = 32) -> Tuple[List[int], List[bool]]:
    """批量贪心决策 + 合法性判定。Batched greedy actions plus legality.

    为什么需要显式判合法性：把决策限制在 5 个动作 token 上后，模型"永远
    输出合法动作"，非法率这个信号就被抹掉了——而 v1/v2 的非法率是重要
    诊断量。这里同时看**全词表 argmax**：若它落在 5 个动作 token 之外，
    说明模型本来想说别的（非法）。
    Why legality is explicit: restricting the decision to the 5 action
    tokens makes the model "always legal" and would erase the illegal
    rate — an important diagnostic in v1/v2. We therefore also look at
    the FULL-VOCAB argmax: if it falls outside the 5 action tokens, the
    model meant to say something else (illegal).
    """
    acts: List[int] = []
    legal: List[bool] = []
    ids_tensor = torch.tensor(list(act_ids), device=device)
    for s in range(0, len(prompts), batch_size):
        chunk = prompts[s: s + batch_size]
        ids_t, attn_t, last = _right_pad(tok, chunk, device)
        out = model(input_ids=ids_t, attention_mask=attn_t)
        rows = torch.arange(len(chunk), device=device)
        logits = out.logits[rows, last]
        full_argmax = logits.argmax(dim=-1)
        legal.extend((full_argmax.unsqueeze(1) == ids_tensor).any(1).tolist())
        # 5 动作内的 argmax / argmax within the 5 actions
        sub = logits[:, list(act_ids)].float()
        acts.extend(int(i) for i in sub.argmax(dim=-1))
    return acts, legal


@torch.no_grad()
def batch_logprobs(model, tok, prompts: Sequence[Sequence[int]],
                   act_ids: Sequence[int], device: str) -> torch.Tensor:
    """一批观测上的 5 动作 logprob（用于 RL 更新）。
    The 5-action logprobs for a batch (used by the RL update)."""
    ids_t, attn_t, last = _right_pad(tok, prompts, device)
    out = model(input_ids=ids_t, attention_mask=attn_t)
    logits = out.logits[torch.arange(len(prompts), device=device), last]
    return F.log_softmax(logits[:, list(act_ids)].float(), dim=-1)
