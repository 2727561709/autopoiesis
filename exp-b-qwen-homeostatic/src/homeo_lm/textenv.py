"""文本渲染的内稳态网格世界（复用 exp-a 环境）。
Text-rendered homeostatic grid world (reuses the exp-a environment).

语言模型看到的世界 = 文本：
The world a language model sees = text:
    - 视野网格（7×7 字符画：O=自己 F=食物 X=危险 .=空 #=边界）
      the view grid (7x7 ASCII: O=self F=food X=hazard .=empty #=boundary)
    - 内感受读数（energy / integrity）
      interoception readings (energy / integrity)
    - 上一步动作
      the previous action

动作以单词输出：up / down / left / right / rest。
Actions are single words: up / down / left / right / rest.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from homeo_rl.env import ACTION_NAMES, EnvConfig, HomeostaticGridWorld, N_ACTIONS

__all__ = ["ACTION_WORDS", "WORD_TO_ACTION", "TextEnv", "render_obs"]

ACTION_WORDS = ("up", "down", "left", "right", "rest")
WORD_TO_ACTION = {w: i for i, w in enumerate(ACTION_WORDS)}

# 视野通道 → 字符 / vision channel -> character
_CH = {1: "F", 2: "X"}


def render_grid(env: HomeostaticGridWorld) -> List[str]:
    """把当前视野渲染成字符画行。Render the current view as ASCII rows."""
    v = env.observation()["vision"]
    k = v.shape[0]
    rows: List[str] = []
    for r in range(k):
        line = ""
        for c in range(k):
            if v[r, c, 3] > 0:
                line += "O"
            elif v[r, c, 1] > 0:
                line += "F"
            elif v[r, c, 2] > 0:
                line += "X"
            elif v[r, c, 0] > 0:
                line += "#"
            else:
                line += "."
        rows.append(line)
    return rows


def render_obs(env: HomeostaticGridWorld, prev_action: Optional[int]) -> str:
    """完整文本观测：视野 + 内感受 + 上一步动作。
    Full textual observation: view + interoception + previous action."""
    rows = "\n".join(render_grid(env))
    e = env.energy
    i = env.integrity
    pa = "none" if prev_action is None else ACTION_WORDS[prev_action]
    return (
        f"energy: {e:.2f} | integrity: {i:.2f}\n"
        f"view (O=self, F=food, X=hazard, .=empty, #=wall):\n"
        f"{rows}\n"
        f"previous action: {pa}"
    )


SYSTEM_PROMPT = (
    "You are a living organism in a survival grid world. "
    "Your energy drains with every step and your integrity is damaged by "
    "hazards. Food restores energy. Stay alive: maintain energy and "
    "integrity near their setpoints (0.8 / 0.9). "
    "Reply with exactly one word: up, down, left, right, or rest."
)


def obs_messages(env: HomeostaticGridWorld, prev_action: Optional[int]) -> List[Dict[str, str]]:
    """构造 chat 格式的观测消息。Build the chat-format observation messages."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_obs(env, prev_action)},
    ]


class TextEnv:
    """语言模型接口的内稳态环境：文本进出，动作是单词。
    Homeostatic environment with a language-model interface: text in,
    actions as words."""

    def __init__(self, config: Optional[EnvConfig] = None) -> None:
        self.env = HomeostaticGridWorld(config)
        self.prev_action: Optional[int] = None

    def reset(self, seed: Optional[int] = None) -> str:
        self.env.reset(seed)
        self.prev_action = None
        return render_obs(self.env, None)

    def step(self, word: str):
        """执行一个动作词，返回 (观测, 回报, 结束, info)。
        Apply an action word; returns (obs, reward, done, info)."""
        word = word.strip().lower().strip(".")
        a = WORD_TO_ACTION.get(word)
        if a is None:
            # 非法输出视为 rest（要付出代价的是它，不是世界）
            # an illegal output counts as rest (the model, not the
            # world, pays for it)
            a = WORD_TO_ACTION["rest"]
        obs, r, done, info = self.env.step(a)
        self.prev_action = a
        return render_obs(self.env, a), r, done, info
