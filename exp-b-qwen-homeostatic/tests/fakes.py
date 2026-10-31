"""测试 Harness：不依赖 GPU / 模型的假件。
Test harness: fakes that need no GPU and no model.

为什么需要它：之前新模块的测试只能测"纯逻辑"（门控、账本、计划器）——
一旦想测**真的走一遍前向**的代码（批量决策、轨迹推进），就得载 0.5B
模型，而那正是这台机器最卡的地方。结果就是：最容易出错的那部分代码
反而测不到。
Why it exists: tests of new modules could previously cover pure logic
only (gates, ledgers, planners). Anything that actually runs a forward
pass (batched decisions, rollouts) needed a 0.5B model — exactly what
this machine struggles with. The consequence: the code most likely to
break was the code least likely to be tested.

这里给的是**最小可用假件**：一个几十参数的因果模型 + 一个假分词器，
满足 `acthead` 用的接口（`.logits`、`pad_token_id`、`encode`），前向在
CPU 上毫秒级完成。
These are MINIMAL usable fakes: a few-dozen-parameter causal model plus a
fake tokenizer, matching the interfaces `acthead` relies on (`.logits`,
`pad_token_id`, `encode`). Forward passes finish in milliseconds on CPU.

它测的是**接线**：批量形状对不对、动作是否落在 5 个里、结束的局是否不再
进批。它不测模型质量——那是真模型的活。
They test the WIRING: batch shapes, actions landing inside the 5, finished
episodes leaving the batch. They do not test model quality — that is the
real model's job.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from homeo_lm.textenv import ACTION_WORDS

VOCAB = 64
HIDDEN = 16
PAD_ID = 0
ACTION_FIRST_TOKENS = [10 + i for i in range(len(ACTION_WORDS))]


class TinyCausalModel(nn.Module):
    """最小因果模型：embedding → 单层线性 → logits。
    The smallest causal model: embedding -> one linear layer -> logits.

    `bias_action` 抬高该动作词首 token 的 logit，用来测"批量路径真的会
    改变动作分布"——否则测试只是在看常数输出。
    `bias_action` raises the logit of that action word's first token, so
    tests can show the batched path really shifts the action distribution
    — otherwise the test is just staring at a constant output.
    """

    def __init__(self, vocab: int = VOCAB, hidden: int = HIDDEN,
                 bias_action: Optional[int] = 0) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab)
        self.bias_action = bias_action

    def forward(self, input_ids, attention_mask=None):
        h = self.emb(input_ids)
        logits = self.head(h)
        if self.bias_action is not None:
            logits[..., ACTION_FIRST_TOKENS[self.bias_action]] += 8.0
        out = type("Out", (), {})()
        out.logits = logits
        return out


class FakeTokenizer:
    """够 `acthead` 用的假分词器。A tokenizer sufficient for `acthead`."""

    def __init__(self) -> None:
        self.pad_token_id = PAD_ID
        self.eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        try:
            idx = ACTION_WORDS.index(text)
        except ValueError:
            return [2, 3]
        return [ACTION_FIRST_TOKENS[idx], 20 + idx]

    def __call__(self, text: str, add_special_tokens: bool = True) -> Dict:
        return {"input_ids": [2, 3, 4]}

    def apply_chat_template(self, messages, tokenize: bool = True,
                            add_generation_prompt: bool = False):
        return "fake prompt"

    def save_pretrained(self, path) -> None:  # pragma: no cover
        from pathlib import Path
        Path(path).mkdir(parents=True, exist_ok=True)


def fake_student(bias_action: Optional[int] = 0):
    """配好的 (model, tok) 一对。A ready (model, tok) pair."""
    torch.manual_seed(0)
    return TinyCausalModel(bias_action=bias_action), FakeTokenizer()
