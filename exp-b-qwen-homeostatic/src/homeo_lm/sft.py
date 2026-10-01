"""QLoRA SFT：把内稳态策略注入 Qwen-0.5B。
QLoRA SFT: inject the homeostatic policy into Qwen-0.5B.

4bit 量化 + LoRA，目标 RTX 3050 4GB。只训练 LoRA 权重。
4-bit quantization + LoRA, targeting the RTX 3050 4GB. Only the LoRA
weights are trained.

用法 / usage:
    python -m homeo_lm.sft --data data/sft.jsonl --out runs/sft-lora
    python -m homeo_lm.sft --data data/sft.jsonl --merge --out runs/sft-merged
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class SFTData(Dataset):
    """chat 模板化的 SFT 数据，只对 assistant 段计算损失。
    Chat-templated SFT data; the loss is computed on the assistant
    segment only."""

    def __init__(self, path: Path, tokenizer, max_len: int = 512) -> None:
        self.rows: List[dict] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                self.rows.append(json.loads(line))
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        prompt = self.tok.apply_chat_template(
            row["messages"][:2], tokenize=False, add_generation_prompt=True)
        full = prompt + row["messages"][2]["content"] + self.tok.eos_token
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = self.tok(full, add_special_tokens=False)["input_ids"]
        f_ids = f_ids[: self.max_len]
        labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[len(p_ids):]
        labels = labels[: len(f_ids)]
        return {"input_ids": f_ids, "labels": labels}


def _collate(batch, pad_id: int):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * len(b["input_ids"]) + [0] * pad)
    return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn))


def train(data_path: Path, out_dir: Path, model_name: str = DEFAULT_MODEL,
          epochs: int = 2, batch_size: int = 4, accum: int = 8,
          lr: float = 1e-4, max_len: int = 512, merge: bool = False,
          device: str = "cuda", resume: bool = False,
          snapshot_every: int = 100) -> Path:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 断点续训状态：已跑到哪个 epoch、epoch 内第几个 batch
    # Resume state: which epoch and batch-in-epoch we reached.
    state_path = out_dir / "train_state.json"
    start_epoch, skip_batches, step = 0, 0, 0
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = None
    try:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    except ImportError:
        pass
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, device_map=device)
    except Exception as e:
        # bitsandbytes 原生 DLL 可能被系统应用控制策略拦截（WinError 4551）；
        # 回退到 bf16 LoRA——Qwen-0.5B 仅约 1GB，4GB 显存装得下。
        # The bitsandbytes native DLL may be blocked by an OS application
        # control policy (WinError 4551); fall back to bf16 LoRA — at
        # ~1GB, Qwen-0.5B fits in 4GB VRAM without quantization.
        print(f"4-bit load failed ({type(e).__name__}); falling back to bf16 LoRA")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device)
    model.config.use_cache = False

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    if resume and (out_dir / "adapter_config.json").exists():
        # 断点续训：载入已有适配器，恢复到上次快照处
        # Resume: load the existing adapter and continue from last snapshot.
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(out_dir), is_trainable=True)
        if state_path.exists():
            st = json.loads(state_path.read_text(encoding="utf-8"))
            start_epoch = st["epoch"]
            skip_batches = st["batch_idx"]
            step = st.get("step", 0)
            print(f"resume from epoch {start_epoch} batch {skip_batches}")
    else:
        model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    # 4GB 显存保险：梯度检查点 + 小批量 / 4GB VRAM safety: gradient
    # checkpointing + small batches
    if hasattr(model, "gradient_checkpointing_enable"):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    ds = SFTData(data_path, tok, max_len)
    # 固定种子的 shuffle：保证续训时 batch 顺序与首跑一致
    # Seeded shuffle: batch order is reproducible across resumed runs.
    g = torch.Generator()
    g.manual_seed(1234)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g,
                    collate_fn=lambda b: _collate(b, tok.pad_token_id))
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr)

    model.train()
    for epoch in range(start_epoch, epochs):
        total, n = 0.0, 0
        opt.zero_grad()
        for i, (ids, labels, attn) in enumerate(dl):
            if i < skip_batches:
                continue  # 跳过已训练的 batch / skip trained batches
            ids, labels, attn = ids.to(device), labels.to(device), attn.to(device)
            out = model(input_ids=ids, attention_mask=attn, labels=labels)
            (out.loss / accum).backward()
            total += out.loss.item()
            n += 1
            if (i + 1) % accum == 0 or (i + 1) == len(dl):
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
                if step % 20 == 0:
                    print(f"epoch {epoch} step {step} loss {total / n:.4f}",
                          flush=True)
                    total, n = 0.0, 0
                # 定期快照：中断后可 --resume 续训 / periodic snapshot
                if snapshot_every and step % snapshot_every == 0:
                    model.save_pretrained(out_dir)
                    tok.save_pretrained(out_dir)
                    state_path.write_text(
                        json.dumps({"epoch": epoch, "batch_idx": i + 1,
                                    "step": step}), encoding="utf-8")
        skip_batches = 0  # 恢复的 epoch 结束后不再跳过 / done skipping
    model.eval()

    if merge:
        # 合并 LoRA 权重保存（推理无需 peft）/ merge LoRA weights and save
        # (no peft needed for inference)
        merged_dir = out_dir.parent / (out_dir.name + "-merged")
        model = model.merge_and_unload()
        try:
            model = model.dequantize()  # 仅 4bit 量化时需要 / only when 4-bit
        except AttributeError:
            pass
        model.save_pretrained(merged_dir)
        tok.save_pretrained(merged_dir)
        print(f"merged model saved: {merged_dir}")
        return merged_dir
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"lora adapter saved: {out_dir}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(
        description="QLoRA SFT: 注入内稳态策略 / inject the homeostatic policy")
    ap.add_argument("--data", type=str, default="data/sft.jsonl")
    ap.add_argument("--out", type=str, default="runs/sft-lora")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--merge", action="store_true",
                    help="训练后合并 LoRA 并保存完整模型 / merge LoRA "
                         "after training and save the full model")
    ap.add_argument("--resume", action="store_true",
                    help="从上次快照续训 / resume from the last snapshot")
    ap.add_argument("--snapshot-every", type=int, default=100,
                    help="每 N 个优化步保存一次快照 / snapshot every N steps")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA unavailable — 4-bit QLoRA on CPU is very slow. "
              "bitsandbytes may be unavailable; falling back to fp32 LoRA.")
    train(Path(args.data), Path(args.out), args.model, args.epochs,
          args.batch_size, args.accum, args.lr, merge=args.merge,
          device=device, resume=args.resume,
          snapshot_every=args.snapshot_every)


if __name__ == "__main__":
    main()
