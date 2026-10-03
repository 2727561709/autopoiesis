"""M10 认知世界模型模块主实现。
M10 cognitive world model — main implementation.

职责：在潜空间学习转移模型 —— 给定 (z, a) 预测 (z', r, done)，
并提供 imagine() 想象 rollout 供 M16 planner 在"脑内"试行动作。
Responsibilities: learn a latent transition model — predict (z', r,
done) from (z, a) — and provide imagine() rollouts so the M16 planner
can try actions "in its head".

三个独立头（不共享主干，保持手写反向传播简单）：
Three independent heads (no shared trunk, keeping hand-written
backprop simple):
- trans_net: (z ⊕ a) -> z'   tanh 有界
- reward_net: (z ⊕ a) -> r   tanh 有界
- done_net:   (z ⊕ a) -> d   (tanh + 1) / 2 ∈ (0, 1)

预测误差 prediction_error() 同时充当 exp-a 实验中"预测误差内在奖励"
的接口：世界越陌生，误差越大，好奇越强。
prediction_error() doubles as the "prediction-error intrinsic reward"
interface from exp-a: the more unfamiliar the world, the larger the
error, the stronger the curiosity.
"""
from __future__ import annotations

from typing import Any, List, Mapping

import numpy as np

from .nn import Linear, Sequential, Tanh
from .types import Action, ImagineResult, WorldModelError

__all__ = ["WorldModel"]


class WorldModel:
    """潜空间世界模型。Latent-space world model.

    Args:
        latent_dim: 潜状态维度（须与 M04 Encoder 的输出一致）。
            Latent-state dimension (must match M04 Encoder output).
        action_dim: 动作维度（离散动作数，或连续动作向量维度）。
            Action dimension (number of discrete actions, or the
            continuous action-vector dimension).
        hidden_dim: 各头隐层宽度。Hidden width of each head.
        seed: 随机种子（同种子 => 同初始化 => 可复现）。
            Random seed (same seed => same init => reproducible).
    """

    def __init__(self, latent_dim: int, action_dim: int,
                 hidden_dim: int = 128, seed: int = 0) -> None:
        if latent_dim <= 0 or action_dim <= 0 or hidden_dim <= 0:
            raise WorldModelError(
                "latent_dim/action_dim/hidden_dim must be positive")
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        in_dim = self.latent_dim + self.action_dim
        rng = np.random.default_rng(seed)
        self.trans_net = Sequential(
            Linear(in_dim, hidden_dim, rng), Tanh(),
            Linear(hidden_dim, self.latent_dim, rng), Tanh(),
        )
        self.reward_net = Sequential(
            Linear(in_dim, hidden_dim, rng), Tanh(),
            Linear(hidden_dim, 1, rng), Tanh(),
        )
        self.done_net = Sequential(
            Linear(in_dim, hidden_dim, rng), Tanh(),
            Linear(hidden_dim, 1, rng), Tanh(),
        )
        # 先验偏置：绝大多数转移并非终止 —— 让未训练的 done 头输出
        # tanh(-2)/2+0.5 ≈ 0.018，避免想象 rollout 被随机噪声提前打断。
        # Prior bias: most transitions are non-terminal — an untrained
        # done head outputs tanh(-2)/2+0.5 ≈ 0.018, so noise cannot cut
        # imagine() rollouts short.
        self.done_net.layers[2].b -= 2.0
        # 经验回放缓冲 / replay buffer of observed transitions
        self._buf: List[dict] = []
        self._rng = np.random.default_rng(seed + 1)

    # ------------------------------------------------------------------ API
    def forward(self, z: np.ndarray, a: Action):
        """(z, a) -> (z', r, done)。

        Args:
            z: (latent_dim,) 或 (B, latent_dim)。
            a: int（one-hot）/ (action_dim,) / (B, action_dim) /
                int 列表（批量离散动作）。
        Returns:
            单个输入 -> (z', r, done) 其中 r、done 为 float；
            批量输入 -> ((B, latent_dim), (B,), (B,))。
            Single input -> (z', r, done) with scalar r, done;
            batch input -> ((B, latent_dim), (B,), (B,)).
        """
        x = self._join(z, a)
        zp = self.trans_net.forward(x)
        r = self.reward_net.forward(x).squeeze(-1)
        d = (self.done_net.forward(x).squeeze(-1) + 1.0) / 2.0
        single = x.ndim == 1
        if single:
            return zp, float(r), float(d)
        return zp, r, d

    __call__ = forward

    def observe(self, z: np.ndarray, a: Action, z_next: np.ndarray,
                reward: float = 0.0, done: float = 0.0) -> None:
        """记录一条真实转移到回放缓冲（训练数据来源）。
        Store one real transition into the replay buffer (training
        data source)."""
        z = self._coerce_z(z)
        z_next = self._coerce_z(z_next, name="z_next")
        if z.shape != z_next.shape:
            raise WorldModelError(
                f"z {z.shape} and z_next {z_next.shape} shape mismatch")
        self._buf.append({
            "z": z, "a": self._coerce_action(a, batch=False),
            "z_next": z_next,
            "r": float(np.clip(reward, -1.0, 1.0)),
            "done": float(np.clip(done, 0.0, 1.0)),
        })

    def train_step(self, lr: float = 0.05,
                   batch_size: int = 32) -> Mapping[str, float]:
        """从回放缓冲采样一步 SGD，返回各头损失。
        One SGD step sampled from the replay buffer; returns per-head
        losses."""
        if not self._buf:
            raise WorldModelError("train_step on empty buffer — call "
                                  "observe() first")
        n = min(batch_size, len(self._buf))
        idx = self._rng.integers(0, len(self._buf), size=n)
        z = np.stack([self._buf[i]["z"] for i in idx])
        a = np.stack([self._buf[i]["a"] for i in idx])
        z_next = np.stack([self._buf[i]["z_next"] for i in idx])
        r = np.array([self._buf[i]["r"] for i in idx]).reshape(-1, 1)
        d = np.array([self._buf[i]["done"] for i in idx]).reshape(-1, 1)

        x = np.concatenate([z, a], axis=-1)
        # MSE 损失 + 手写反向传播 / MSE losses + hand-written backprop
        for net in (self.trans_net, self.reward_net, self.done_net):
            net.zero_grad()
        zp = self.trans_net.forward(x)
        rp = self.reward_net.forward(x)
        dp = (self.done_net.forward(x) + 1.0) / 2.0
        err_z, err_r, err_d = zp - z_next, rp - r, dp - d
        loss = {
            "trans": float((err_z ** 2).mean()),
            "reward": float((err_r ** 2).mean()),
            "done": float((err_d ** 2).mean()),
        }
        self.trans_net.backward(2.0 * err_z / err_z.size)
        self.reward_net.backward(2.0 * err_r / err_r.size)
        # 链式：d L / d (raw done) = err_d * 0.5
        self.done_net.backward(err_d)
        for net in (self.trans_net, self.reward_net, self.done_net):
            net.sgd_step(lr)
        return loss

    def prediction_error(self, z: np.ndarray, a: Action,
                         z_next: np.ndarray) -> float:
        """预测误差（exp-a 的"好奇"内在奖励接口）。
        Prediction error (the "curiosity" intrinsic-reward interface
        from exp-a)."""
        z = self._coerce_z(z)
        z_next = self._coerce_z(z_next, name="z_next")
        if z.shape != z_next.shape:
            raise WorldModelError("z / z_next shape mismatch")
        zp, _, _ = self.forward(z, a)
        return float(((zp - z_next) ** 2).mean())

    def imagine(self, z: np.ndarray, policy, horizon: int) -> ImagineResult:
        """在"脑内"试行动作序列。Rollout actions "in the head".

        Args:
            z: 初始潜状态 (latent_dim,)。
            policy: callable z -> Action（int 或动作向量）。
            horizon: 想象步数 H。
        Returns:
            ImagineResult（zs 长度 H+1；预测 done > 0.5 提前终止）。
            ImagineResult (zs of length H+1; early stop on predicted
            done > 0.5).
        """
        if horizon < 0:
            raise WorldModelError(f"horizon must be >= 0, got {horizon}")
        z = self._coerce_z(z)
        out = ImagineResult(zs=[z.copy()])
        for _ in range(horizon):
            a = policy(z)
            a_vec = self._coerce_action(a, batch=False)
            z2, r, d = self.forward(z, a)
            out.actions.append(a_vec)
            out.rewards.append(r)
            out.dones.append(d)
            out.zs.append(z2.copy())
            z = z2
            if d > 0.5:
                break
        return out

    # ------------------------------------------------------------- plumbing
    def parameters(self) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for net in (self.trans_net, self.reward_net, self.done_net):
            out.extend(net.parameters())
        return out

    def zero_grad(self) -> None:
        for net in (self.trans_net, self.reward_net, self.done_net):
            net.zero_grad()

    # ------------------------------------------------------- save / restore
    def state_dict(self) -> Mapping[str, Any]:
        """权重快照（纯 NumPy 数组，可 pickle / 交给 M03 存档）。
        Weight snapshot (pure NumPy arrays, picklable / hand to M03)."""
        sd = {"latent_dim": self.latent_dim, "action_dim": self.action_dim,
              "buf": list(self._buf)}
        for name, net in (("trans", self.trans_net), ("reward", self.reward_net),
                          ("done", self.done_net)):
            sd[f"{name}.W1"] = net.layers[0].W
            sd[f"{name}.b1"] = net.layers[0].b
            sd[f"{name}.W2"] = net.layers[2].W
            sd[f"{name}.b2"] = net.layers[2].b
        return sd

    def load_state_dict(self, sd: Mapping[str, Any]) -> None:
        if (sd["latent_dim"] != self.latent_dim
                or sd["action_dim"] != self.action_dim):
            raise WorldModelError(
                f"state_dict shape mismatch: "
                f"({sd['latent_dim']},{sd['action_dim']}) vs "
                f"({self.latent_dim},{self.action_dim})")
        for name, net in (("trans", self.trans_net), ("reward", self.reward_net),
                          ("done", self.done_net)):
            net.layers[0].W = np.array(sd[f"{name}.W1"], dtype=np.float64)
            net.layers[0].b = np.array(sd[f"{name}.b1"], dtype=np.float64)
            net.layers[2].W = np.array(sd[f"{name}.W2"], dtype=np.float64)
            net.layers[2].b = np.array(sd[f"{name}.b2"], dtype=np.float64)
        self._buf = list(sd.get("buf", []))

    # -------------------------------------------------------------- internals
    def _coerce_z(self, z: np.ndarray, name: str = "z") -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        if z.ndim not in (1, 2):
            raise WorldModelError(f"{name} must be 1-D or 2-D, got {z.ndim}-D")
        if not np.all(np.isfinite(z)):
            raise WorldModelError(f"{name} contains NaN/Inf")
        if z.shape[-1] != self.latent_dim:
            raise WorldModelError(
                f"{name} last dim must be {self.latent_dim}, got {z.shape[-1]}")
        return z

    def _coerce_action(self, a: Action, batch: bool = False) -> np.ndarray:
        """动作 -> (action_dim,) 或 (B, action_dim)；int 自动 one-hot。
        Action -> (action_dim,) or (B, action_dim); int is one-hot
        encoded automatically."""
        if isinstance(a, (int, np.integer)):
            if not 0 <= int(a) < self.action_dim:
                raise WorldModelError(
                    f"action id {a} out of range [0, {self.action_dim})")
            onehot = np.zeros(self.action_dim)
            onehot[int(a)] = 1.0
            return onehot
        raw = np.asarray(a)
        arr = np.asarray(a, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise WorldModelError("action contains NaN/Inf")
        if arr.ndim == 1:
            if arr.shape[0] == self.action_dim:
                return arr
            # 原生整型序列且长度 != action_dim => 批量离散动作
            # natively-int sequence and length != action_dim => batch of ids
            if raw.dtype.kind in "iu" and arr.size > 0:
                if arr.min() < 0 or arr.max() >= self.action_dim:
                    raise WorldModelError("action id out of range")
                out = np.zeros((arr.size, self.action_dim))
                out[np.arange(arr.size), arr.astype(np.int64)] = 1.0
                return out
            raise WorldModelError(
                f"action vector must have {self.action_dim} dims, got "
                f"{arr.shape[0]}")
        if arr.ndim == 2 and arr.shape[-1] == self.action_dim:
            return arr
        raise WorldModelError(f"unsupported action shape {arr.shape}")

    def _join(self, z: np.ndarray, a: Action) -> np.ndarray:
        """把 z 与 a 对齐成同批并拼接。Align z and a batches, then
        concatenate."""
        z = self._coerce_z(z)
        a_arr = self._coerce_action(a)
        single = z.ndim == 1
        if single != (a_arr.ndim == 1):
            raise WorldModelError("z and a must both be single or both "
                                  "be batched")
        if not single and z.shape[0] != a_arr.shape[0]:
            raise WorldModelError(
                f"batch mismatch: z {z.shape[0]} vs a {a_arr.shape[0]}")
        if single:
            return np.concatenate([z, a_arr])
        return np.concatenate([z, a_arr], axis=-1)
