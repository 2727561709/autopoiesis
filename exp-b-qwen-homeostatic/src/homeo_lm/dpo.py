"""手写 DPO：在 v2 LoRA 之上做内稳态偏好对学习。
Hand-written DPO: homeostatic preference-pair learning on top of v2.

设计决策（2026-10-15 与用户确认）：
Design decisions (confirmed with the user, 2026-10-15):
    - 起点：v2 LoRA 同时作为策略 π_θ 与参考 π_ref（frozen）
      start: the v2 adapter is both the policy π_θ and the ref π_ref
    - β = 0.1（TRL 默认量级；先跑通验证）
      β = 0.1 (TRL default scale; verify first)
    - 损失只算 assistant 段（动作词 + eos）的对数概率
      the loss uses only assistant-token (action word + eos) logprobs

DPO 损失 / DPO loss:
    L = -log σ( β · [ (logπθ(a_c|s) - logπref(a_c|s))
                    - (logπθ(a_r|s) - logπref(a_r|s)) ] )

4GB 显存策略 / 4GB VRAM strategy:
    - 底座 bf16 (~1GB) × 2（policy + ref）≈ 2GB，LoRA 梯度 + AdamW
      状态 ~0.1GB，激活走梯度检查点 —— RTX 3050 4GB 可行。
    - 若超限：ref 也可 4bit 量化（--ref-4bit）。
    - policy 和 ref 是同一个底座的两个 LoRA 挂载——不做双份底座，
      而是 ref 用"禁用适配器"的基座 forward（PEFT disable_adapter）。
      显存省一半。

用法 / usage:
    python -m homeo_lm.dpo --data data/dpo_pairs.jsonl \
        --ref runs/sft-lora-v2 --out runs/dpo-v3
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .sft import DEFAULT_MODEL, collapse_check

import random


class DPOData(Dataset):
    """DPO 偏好对：prompt（chat 前缀）+ chosen / rejected 回复。
    DPO pairs: chat prompt + chosen / rejected replies."""

    def __init__(self, path: Path, tokenizer, max_len: int = 512) -> None:
        self.rows: List[dict] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                self.rows.append(json.loads(line))
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def _encode(self, row: dict, reply: str) -> Tuple[List[int], List[int]]:
        """编码 prompt+reply，返回 (input_ids, assistant_mask)。
        Encode prompt+reply; returns (ids, assistant_token_mask)."""
        prompt = self.tok.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=True)
        full = prompt + reply + self.tok.eos_token
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = self.tok(full, add_special_tokens=False)["input_ids"]
        f_ids = f_ids[: self.max_len]
        # assistant 段 = 全长减去 prompt 前缀的尾部 / assistant segment
        mask = [0] * min(len(p_ids), len(f_ids)) + [1] * max(0, len(f_ids) - len(p_ids))
        mask = mask[: len(f_ids)]
        return f_ids, mask

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        c_ids, c_mask = self._encode(row, row["chosen"])
        r_ids, r_mask = self._encode(row, row["rejected"])
        return {"c_ids": c_ids, "c_mask": c_mask,
                "r_ids": r_ids, "r_mask": r_mask}


def _collate(batch, pad_id: int):
    """把 chosen/rejected 各拼一个子批。Stack chosen/rejected sub-batches."""
    out = {}
    for tag in ("c", "r"):
        ids_tag = [b[f"{tag}_ids"] for b in batch]
        masks_tag = [b[f"{tag}_mask"] for b in batch]
        maxlen = max(len(x) for x in ids_tag)
        ids, attn, amask = [], [], []
        for i, m in zip(ids_tag, masks_tag):
            pad = maxlen - len(i)
            ids.append(i + [pad_id] * pad)
            attn.append([1] * len(i) + [0] * pad)
            amask.append(m + [0] * pad)
        out[f"{tag}_ids"] = torch.tensor(ids)
        out[f"{tag}_attn"] = torch.tensor(attn)
        out[f"{tag}_amask"] = torch.tensor(amask)
    return out


def _assistant_logps(model, ids, attn, amask) -> torch.Tensor:
    """assistant 段的序列对数概率（每样本一个标量）。
    Sequence log-probability over assistant tokens (one scalar per
    example)."""
    out = model(input_ids=ids, attention_mask=attn)
    logits = out.logits[:, :-1]               # 预测下一个 token
    tgt = ids[:, 1:]                          # 被预测的 token
    m = amask[:, 1:].float()                  # assistant 段掩码
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_lp = torch.gather(logp, 2, tgt.unsqueeze(-1)).squeeze(-1)
    return (tok_lp * m).sum(dim=1)


def dpo_loss(pol_c, pol_r, ref_c, ref_r, beta: float):
    """标准 DPO 损失 + 监控指标。Standard DPO loss + monitors.

    返回 / returns (loss, {rewards margins}):
        rewards_margin = β·[(Δc) - (Δr)]，>0 表示策略偏向 chosen
        rewards_margin = β·[(Δc) - (Δr)]; >0 means the policy prefers
        the chosen reply.

    acc 的口径（写死在这里，免得各人各解）：只有 **margin > 0 严格成立**
    才算命中。margin == 0 是"分不出"，算不中——因为偏好本身没有证据，
    不算半票。
    Definition of acc (pinned down here so it is not reinterpreted): a
    pair counts as a hit only if **margin > 0 strictly**. margin == 0
    means indifference and is NOT a hit — there is no evidence of
    preference, so it is not half a vote either.
    """
    delta_c = pol_c - ref_c
    delta_r = pol_r - ref_r
    logits = beta * (delta_c - delta_r)
    loss = -torch.nn.functional.logsigmoid(logits).mean()
    return loss, {"margin": logits.detach().mean().item(),
                  "acc": (logits > 0).float().mean().item()}


def train(data_path: Path, ref_dir: Path, out_dir: Path,
          model_name: str = DEFAULT_MODEL, epochs: int = 1,
          batch_size: int = 2, accum: int = 8, lr: float = 5e-5,
          beta: float = 0.1, max_len: int = 512, device: str = "cuda",
          resume: bool = False, snapshot_every: int = 50) -> Path:
    """DPO 训练：v2 LoRA 起点，同底座双 LoRA 视角（ref 用禁用适配器的
    基座 forward）。DPO training from the v2 adapter; the reference is
    the same base with the adapter disabled (PEFT context manager)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "train_state.json"
    start_epoch, skip_batches, step = 0, 0, 0

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device)
    model.config.use_cache = False
    # 策略：v2 适配器，可训练 / policy: the v2 adapter, trainable
    model = PeftModel.from_pretrained(model, str(ref_dir), is_trainable=True)
    model.print_trainable_parameters()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    ds = DPOData(data_path, tok, max_len)
    g = torch.Generator()
    g.manual_seed(1234)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g,
                    collate_fn=lambda b: _collate(b, tok.pad_token_id))
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr)

    if resume and state_path.exists():
        st = json.loads(state_path.read_text(encoding="utf-8"))
        start_epoch, skip_batches, step = st["epoch"], st["batch_idx"], st.get("step", 0)
        print(f"resume from epoch {start_epoch} batch {skip_batches}")

    model.train()
    for epoch in range(start_epoch, epochs):
        opt.zero_grad()
        run = {"loss": 0.0, "margin": 0.0, "acc": 0.0, "n": 0}
        for i, batch in enumerate(dl):
            if i < skip_batches:
                continue
            ids_c = batch["c_ids"].to(device); att_c = batch["c_attn"].to(device)
            ids_r = batch["r_ids"].to(device); att_r = batch["r_attn"].to(device)
            m_c = batch["c_amask"].to(device); m_r = batch["r_amask"].to(device)

            # ---- 参考模型（禁用适配器 = 基座，no grad）----
            # reference (adapter disabled = base model, no grad)
            with torch.no_grad(), model.disable_adapter():
                ref_c = _assistant_logps(model, ids_c, att_c, m_c)
                ref_r = _assistant_logps(model, ids_r, att_r, m_r)

            # ---- 策略模型（适配器开）----
            # policy (adapter on)
            pol_c = _assistant_logps(model, ids_c, att_c, m_c)
            pol_r = _assistant_logps(model, ids_r, att_r, m_r)

            loss, m = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)
            (loss / accum).backward()
            run["loss"] += loss.item(); run["margin"] += m["margin"]
            run["acc"] += m["acc"]; run["n"] += 1

            if (i + 1) % accum == 0 or (i + 1) == len(dl):
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
                if step % 10 == 0 and run["n"]:
                    print(f"epoch {epoch} step {step} "
                          f"loss {run['loss']/run['n']:.4f} "
                          f"margin {run['margin']/run['n']:+.3f} "
                          f"acc {run['acc']/run['n']:.2f}", flush=True)
                    run = {"loss": 0.0, "margin": 0.0, "acc": 0.0, "n": 0}
                if snapshot_every and step % snapshot_every == 0:
                    model.save_pretrained(out_dir)
                    tok.save_pretrained(out_dir)
                    state_path.write_text(json.dumps(
                        {"epoch": epoch, "batch_idx": i + 1, "step": step}),
                        encoding="utf-8")
                    # 坍缩监控沿用 SFT 的固定样本检查 / reuse SFT's
                    # collapse monitor on fixed prompts
                    sft_rows = [{"messages": r["messages"]} for r in ds.rows]
                    dist, top = collapse_check(model, tok, sft_rows, device)
                    flag = " <<< COLLAPSING" if top > 0.9 else ""
                    print(f"collapse-check step {step}: {dist} "
                          f"top={top:.2f}{flag}", flush=True)
        skip_batches = 0

    model.eval()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"dpo adapter saved: {out_dir}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(
        description="手写 DPO：内稳态偏好对学习 / hand-written DPO on "
                    "homeostatic preference pairs")
    ap.add_argument("--data", type=str, default="data/dpo_pairs.jsonl")
    ap.add_argument("--ref", type=str, default="runs/sft-lora-v2",
                    help="起点 LoRA（同时作策略初始化与参考）/ the "
                         "starting adapter (policy init AND reference)")
    ap.add_argument("--out", type=str, default="runs/dpo-v3")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--snapshot-every", type=int, default=50)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: DPO on CPU is extremely slow.")
    train(Path(args.data), Path(args.ref), Path(args.out), args.model,
          args.epochs, args.batch_size, args.accum, args.lr, args.beta,
          device=device, resume=args.resume,
          snapshot_every=args.snapshot_every)


if __name__ == "__main__":
    main()
