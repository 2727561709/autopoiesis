"""阶段 A 模型：Transformer 主干 + 三个头。
Stage-A model: Transformer trunk + three heads.

    - 视觉编码器：卷积编码自我中心视野 (k, k, 4)
      Vision encoder: conv encoder for the ego-centric view (k, k, 4)
    - 内感受编码器：MLP 编码 (能量, 完整性)
      Interoception encoder: MLP over (energy, integrity)
    - 主干：3 个 token（视觉 / 内感受 / 上一步动作）过 Transformer
      Trunk: 3 tokens (vision / interoception / previous action) through
      a Transformer encoder
    - actor 头：动作分布（生存行为）
      Actor head: action distribution (survival behavior)
    - critic 头：状态价值（预测未来内稳态回报）
      Critic head: state value (expected future homeostatic return)
    - 世界模型头：预测下一步的视觉表征与内感受 —— 预测误差
      同时作为辅助损失与好奇式内在奖励
      World-model head: predicts the next visual representation and
      interoception — the prediction error serves both as an auxiliary
      loss and as curiosity-style intrinsic reward
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

__all__ = ["VisionEncoder", "HomeoActorCritic"]


class VisionEncoder(nn.Module):
    """卷积视觉编码器：(B, 4, k, k) -> (B, out_dim)。默认 k=7 时 7->3->1。
    Conv vision encoder: (B, 4, k, k) -> (B, out_dim); 7->3->1 for k=7."""

    def __init__(self, in_ch: int = 4, out_dim: int = 128, view: int = 7) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2), nn.ReLU(),
        )
        with torch.no_grad():
            # 推导卷积输出的空间边长 / derive the conv output side length
            side = view
            for m in self.conv:
                if isinstance(m, nn.Conv2d):
                    side = (side - 3) // 2 + 1
            flat = 64 * side * side
        self.proj = nn.Linear(flat, out_dim)

    def forward(self, vision: torch.Tensor) -> torch.Tensor:
        h = self.conv(vision).flatten(1)
        return self.proj(h)


class HomeoActorCritic(nn.Module):
    """内稳态 actor-critic + 世界模型。
    Homeostatic actor-critic with a world model."""

    def __init__(self,
                 n_actions: int = 5,
                 view: int = 7,
                 in_ch: int = 4,
                 d_model: int = 128,
                 nhead: int = 4,
                 n_layers: int = 4,
                 ff_dim: int = 256) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.vision = VisionEncoder(in_ch, d_model, view)
        self.intero = nn.Sequential(
            nn.Linear(2, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        # 末位 = "无上一步动作" / last index = "no previous action"
        self.act_emb = nn.Embedding(n_actions + 1, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, ff_dim, dropout=0.0,
            batch_first=True, norm_first=True, activation="gelu")
        self.trunk = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.actor = nn.Linear(d_model, n_actions)
        self.critic = nn.Linear(d_model, 1)
        self.wm_vision_head = nn.Linear(d_model, d_model)
        self.wm_intero_head = nn.Linear(d_model, 2)

    # ------------------------------------------------------------------ 前向 / forward
    def features(self, vision: torch.Tensor, intero: torch.Tensor,
                 prev_action: torch.Tensor) -> torch.Tensor:
        """(B, k, k, 4) + (B, 2) + (B,) -> 池化后的主干表征 (B, d)。
        (B, k, k, 4) + (B, 2) + (B,) -> pooled trunk representation (B, d)."""
        v = vision.permute(0, 3, 1, 2)  # (B, 4, k, k)
        tokens = torch.stack([
            self.vision(v),
            self.intero(intero),
            self.act_emb(prev_action),
        ], dim=1)  # (B, 3, d)
        h = self.trunk(tokens).mean(dim=1)
        return h

    def forward(self, vision: torch.Tensor, intero: torch.Tensor,
                prev_action: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.features(vision, intero, prev_action)
        return {
            "logits": self.actor(h),
            "value": self.critic(h).squeeze(-1),
            "wm_vision": self.wm_vision_head(h),
            "wm_intero": self.wm_intero_head(h),
        }

    # ------------------------------------------------------- 世界模型 / world model
    @torch.no_grad()
    def wm_targets(self, next_vision: torch.Tensor,
                   next_intero: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """世界模型的学习目标：下一步的视觉表征（detach）与内感受。
        World-model learning targets: next-step visual representation
        (detached) and next interoception."""
        v = next_vision.permute(0, 3, 1, 2)
        return self.vision(v).detach(), next_intero.detach()

    def prediction_error(self, out: Dict[str, torch.Tensor],
                         next_vision: torch.Tensor,
                         next_intero: torch.Tensor) -> torch.Tensor:
        """逐步预测误差（标量均值版本的逐样本形式），形状 (B,)。
        Per-sample prediction error (per-sample form of the scalar mean
        version), shape (B,)."""
        tv, ti = self.wm_targets(next_vision, next_intero)
        return ((out["wm_vision"] - tv) ** 2).mean(-1) \
            + ((out["wm_intero"] - ti) ** 2).mean(-1)

    # ------------------------------------------------------------------ 采样 / acting
    def act(self, vision: torch.Tensor, intero: torch.Tensor,
            prev_action: torch.Tensor,
            deterministic: bool = False
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样动作。返回 (action, log_prob, value)，均不带梯度。
        Sample an action. Returns (action, log_prob, value), all
        gradient-free."""
        with torch.no_grad():
            out = self.forward(vision, intero, prev_action)
            dist = Categorical(logits=out["logits"])
            if deterministic:
                a = dist.probs.argmax(-1)
            else:
                a = dist.sample()
            return a, dist.log_prob(a), out["value"]
