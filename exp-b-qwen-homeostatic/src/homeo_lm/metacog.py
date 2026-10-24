"""RSI Level 2 的前置件：元认知——系统自己产出**可证伪的假设**。
The prerequisite for RSI Level 2: metacognition — the system produces
its own FALSIFIABLE hypotheses.

L0 调旋钮、L1 调数据，**L2 调学习规则本身**（LoRA rank / 学习率 / 巩固
策略）。这一层必须有假设，否则就是在超参数空间里乱撞——乱撞的结果无法
归因，也就无法积累。
L0 tunes knobs, L1 tunes data, **L2 changes the learning rule itself**
(LoRA rank / lr / consolidation). This level REQUIRES hypotheses,
otherwise it is random search: random search yields no attribution and
therefore no accumulation.

一条假设必须三件套齐全，缺一不可：
A hypothesis needs all three parts:
1. **claim**——可证伪的断言（"容量不足以表达调制映射"）
   a falsifiable claim
2. **prediction**——若它为真，**什么数会往哪个方向动**
   if it is true, WHICH number moves in WHICH direction
3. **test**——用哪个旋钮的哪几个取值去证伪它
   which knob values would refute it

承重原则不变：**假设由系统提，判定由环境做**。模型永远不自评——它只
负责说"我怀疑是 X"，然后由探针（只读环境后果）决定 X 成立与否。
Same load-bearing principle: the SYSTEM proposes, the ENVIRONMENT
decides. The model never grades itself.

设计要点 / design notes:
- 每条规则都要求**多个证据同时成立**才触发，避免单指标噪声点火
  every rule requires several conditions at once, so noise cannot fire it
- 指标缺失（None）视为"不知道"，**不触发**规则——不许用猜测当证据
  a missing metric means UNKNOWN and never triggers a rule
- 历史命中率进选择：哪类假设过去常被证实，就更优先试（这才是"元"）
  historical hit rates feed selection: hypothesis classes that used to
  pay off get priority (that is the "meta")
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .rsi_l0 import Knob

# ------------------------------------------------------------ L2 旋钮空间
# 这些会改变**学习规则本身**，因此比 L0 旋钮门槛更高（见 l2_gate）。
# These change the LEARNING RULE, so the bar is higher (see l2_gate).
L2_KNOBS: Dict[str, Knob] = {
    "lora_rank": Knob("lora_rank", [8, 16, 32], "sft", "--rank"),
    "lr": Knob("lr", [5e-6, 1e-5, 2e-5], "train", "--lr"),
    "epochs": Knob("epochs", [1, 2, 3], "train", "--epochs"),
    "replay_frac": Knob("replay_frac", [0.0, 0.2, 0.4], "data",
                        "--replay-frac"),
}

# 触发阈值（集中放，便于审计与调整）
# Thresholds (kept together so they can be audited and tuned)
T = {
    "train_acc_high": 0.90,     # 训练集几乎全对 / near-perfect on train
    "gap_bad": 0.0,             # 调制没有转正 / modulation not positive
    "slope_bad": 0.0,           # 剂量-反应曲线不上升 / curve not rising
    "survival_low": 0.20,
    "agree_high": 0.60,
    "dpo_margin_low": 1.0,
    "dpo_loss_high": 0.40,
    "top_share_collapse": 0.60,
    "illegal_high": 0.20,
}


@dataclass
class Hypothesis:
    """一条可证伪假设。A falsifiable hypothesis."""

    id: str
    claim: str            # 可证伪断言 / falsifiable claim
    prediction: str       # 若为真，什么数往哪动 / what moves, and how
    knob: str             # 用哪个旋钮证伪它 / the knob that tests it
    values: List          # 候选取值 / candidate values
    level: str            # L0 / L1 / L2
    evidence: List[str] = field(default_factory=list)

    @property
    def strength(self) -> float:
        """证据强度：命中条件越多越可信（不是分数，只用于排序）。
        Evidence strength: more matched conditions = more believable
        (a ranking aid only, never a score of the model)."""
        return float(len(self.evidence))

    def is_falsifiable(self) -> bool:
        """三件套是否齐全：claim + prediction + 一个真实存在的旋钮。
        Are all three parts present: claim + prediction + a real knob."""
        from .rsi_l0 import KNOBS
        return bool(self.claim) and bool(self.prediction) and \
            self.knob in set(L2_KNOBS) | set(KNOBS)

    def as_dict(self) -> Dict:
        return {"id": self.id, "claim": self.claim,
                "prediction": self.prediction, "knob": self.knob,
                "values": list(self.values), "level": self.level,
                "evidence": list(self.evidence),
                "strength": self.strength}


def _get(m: Dict, k: str) -> Optional[float]:
    """取值；缺失或 None = 不知道。Get a value; missing = unknown."""
    v = m.get(k)
    return None if v is None else float(v)


# ---------------------------------------------------------------- 诊断规则
def diagnose(metrics: Dict) -> List[Hypothesis]:
    """从当前指标产出候选假设（按证据强度降序）。
    Produce candidate hypotheses from the current metrics (sorted by
    evidence strength).

    每条规则都要**多个条件同时成立**才点火，且缺失指标视为不知道。
    Every rule needs several conditions at once, and a missing metric
    counts as unknown, never as evidence.
    """
    from .rsi_l0 import KNOBS

    out: List[Hypothesis] = []
    gap = _get(metrics, "modulation_gap")
    slope = _get(metrics, "sweep_slope")
    surv = _get(metrics, "survival_rate")
    agree = _get(metrics, "agreement")
    illegal = _get(metrics, "illegal_rate")
    top = _get(metrics, "top_action_share")
    train_acc = _get(metrics, "train_acc")
    holdout = _get(metrics, "holdout_gap")
    margin = _get(metrics, "dpo_margin")
    loss = _get(metrics, "dpo_loss")

    # 1) 容量不足：训得会、用得不会
    # capacity deficit: knows it on paper, cannot use it
    ev = []
    if train_acc is not None and train_acc >= T["train_acc_high"]:
        ev.append(f"train_acc {train_acc:.2f} >= {T['train_acc_high']}")
    if gap is not None and gap <= T["gap_bad"]:
        ev.append(f"modulation_gap {gap:+.2f} <= {T['gap_bad']}")
    if slope is not None and slope <= T["slope_bad"]:
        ev.append(f"sweep_slope {slope:+.2f} <= {T['slope_bad']}")
    if len(ev) >= 2:
        out.append(Hypothesis(
            "capacity_deficit",
            "适配器容量不足以表达'内感受 → 动作'的调制映射"
            " / the adapter cannot represent the interoception->action "
            "modulation mapping",
            "若为真：提高 lora_rank 应使 modulation_gap 单调上升"
            " / if true: raising lora_rank should raise modulation_gap "
            "monotonically",
            "lora_rank", L2_KNOBS["lora_rank"].values, "L2", ev))

    # 2) 优化不足：还没走到就停了
    # under-optimised: stopped before it got there
    ev = []
    if margin is not None and margin < T["dpo_margin_low"]:
        ev.append(f"dpo_margin {margin:.2f} < {T['dpo_margin_low']}")
    if loss is not None and loss > T["dpo_loss_high"]:
        ev.append(f"dpo_loss {loss:.2f} > {T['dpo_loss_high']}")
    if gap is not None and gap < 0.5:
        ev.append(f"modulation_gap {gap:+.2f} < 0.5")
    if len(ev) >= 2:
        out.append(Hypothesis(
            "optimization_underfit",
            "更新量/步数不足，策略还没收敛到偏好信号上"
            " / updates are too small or too few; the policy has not "
            "converged onto the preference signal",
            "若为真：提高 lr 或 epochs 应同时降低 loss 并抬高 gap"
            " / if true: raising lr or epochs should lower the loss AND "
            "raise the gap together",
            "lr", L2_KNOBS["lr"].values, "L2", ev))

    # 3) 分布错配：练的地方 ≠ 错的地方
    # distribution mismatch: training where it is not wrong
    ev = []
    if agree is not None and agree >= T["agree_high"]:
        ev.append(f"agreement {agree:.2f} >= {T['agree_high']}")
    if surv is not None and surv <= T["survival_low"]:
        ev.append(f"survival {surv:.2f} <= {T['survival_low']}")
    if holdout is not None and gap is not None and holdout < gap - 0.2:
        ev.append(f"holdout_gap {holdout:+.2f} << train gap {gap:+.2f}")
    if len(ev) >= 2:
        out.append(Hypothesis(
            "distribution_mismatch",
            "训练状态分布不覆盖它会犯错的状态（教师/随机轨迹 ≠ 自己的轨迹）"
            " / the training state distribution does not cover the states "
            "where it fails",
            "若为真：提高 on-policy 占比（policy_frac）应抬高存活率"
            " / if true: raising the on-policy share should raise survival",
            "teacher_mix", KNOBS["teacher_mix"].values, "L1", ev))

    # 4) 奖励整形设定错误：信号本身就是歪的
    # misspecified shaping: the signal itself is skewed
    ev = []
    if gap is not None and abs(gap) < 0.1:
        ev.append(f"modulation_gap {gap:+.2f} ≈ 0（几乎无调制）"
                  f" / ≈ 0 (almost no modulation)")
    if illegal is not None and illegal <= T["illegal_high"]:
        ev.append(f"illegal_rate {illegal:.2f} 低，不是输出格式问题"
                  f" / low, so not an output-format problem")
    if surv is not None and surv <= T["survival_low"]:
        ev.append(f"survival {surv:.2f} <= {T['survival_low']}")
    if len(ev) >= 2:
        out.append(Hypothesis(
            "shaping_misspecified",
            "PBRS 整形权重/门控阈值设定不当，偏好信号本身方向有问题"
            " / the PBRS weight or gate threshold is mis-set, so the "
            "preference signal itself points the wrong way",
            "若为真：改 shape_w 或 gate_margin 应改变曲线斜率符号"
            " / if true: changing shape_w or gate_margin should flip the "
            "sign of the sweep slope",
            "shape_w", KNOBS["shape_w"].values, "L1", ev))

    # 5) 灾难性遗忘：新的学会、旧的丢掉
    # catastrophic forgetting: new learned, old lost
    ev = []
    if _get(metrics, "non_monotone") == 1.0:
        ev.append("指标先升后降 / the metric rose then fell")
    if _get(metrics, "train_loss_falling") == 1.0:
        ev.append("训练损失持续下降 / training loss keeps falling")
    if len(ev) >= 2:
        out.append(Hypothesis(
            "catastrophic_forgetting",
            "新更新覆盖了旧行为，不是没学会而是学丢了"
            " / new updates overwrite old behaviour: not unlearned, "
            "but overwritten",
            "若为真：加入重放（replay_frac > 0）应恢复旧指标且不损新指标"
            " / if true: adding replay should restore the old metric "
            "without hurting the new one",
            "replay_frac", L2_KNOBS["replay_frac"].values, "L2", ev))

    # 6) 坍缩：退回到单一动作
    # collapse: retreated to a single action
    ev = []
    if top is not None and top >= T["top_share_collapse"]:
        ev.append(f"top action share {top:.2f} >= {T['top_share_collapse']}")
    if _get(metrics, "entropy") is not None and \
            _get(metrics, "entropy") < 0.2:
        ev.append("entropy < 0.2")
    if len(ev) >= 2:
        out.append(Hypothesis(
            "collapse",
            "策略坍缩到少数动作（v1 的失败模式）"
            " / the policy collapsed onto a few actions (v1's failure)",
            "若为真：提高 entropy_coef 应降低 top share 并抬高存活率"
            " / if true: raising entropy_coef should lower the top share "
            "and raise survival",
            "entropy_coef", KNOBS["entropy_coef"].values, "L0", ev))

    out.sort(key=lambda h: h.strength, reverse=True)
    return out


# ------------------------------------------------------------ 历史命中率
def hit_rates(rows: Sequence[Dict]) -> Dict[str, Dict[str, int]]:
    """每类假设的历史战绩（被证实几次/试过几次）。
    Historical record per hypothesis class (confirmed / tried).

    这是"元"的真正含义：**系统对自己哪类猜测更靠谱有统计**。
    This is what "meta" actually means: the system keeps statistics on
    which kinds of guesses tend to be right.
    """
    stats: Dict[str, Dict[str, int]] = {}
    for e in rows:
        hid = (e.get("change") or {}).get("hypothesis")
        if not hid:
            continue
        s = stats.setdefault(hid, {"tries": 0, "confirmed": 0})
        s["tries"] += 1
        if e.get("verdict") == "adopt":
            s["confirmed"] += 1
    return stats


def _ucb(confirmed: int, tries: int, total: int, c: float = 1.0) -> float:
    """假设臂的 UCB：命中率 + 探索项。
    UCB for a hypothesis arm: hit rate + exploration bonus."""
    if tries == 0:
        return float("inf")
    import math
    return confirmed / tries + c * math.sqrt(math.log(max(2, total)) / tries)


def rank(hypotheses: Sequence[Hypothesis],
         rows: Sequence[Dict]) -> List[Dict]:
    """按"证据强度 × 历史命中率(UCB)"排序。
    Rank by evidence strength x historical hit rate (UCB)."""
    stats = hit_rates(rows)
    total = sum(s["tries"] for s in stats.values()) or 1
    scored = []
    for h in hypotheses:
        s = stats.get(h.id, {"tries": 0, "confirmed": 0})
        u = _ucb(s["confirmed"], s["tries"], total)
        scored.append({**h.as_dict(), "prior_tries": s["tries"],
                       "prior_confirmed": s["confirmed"],
                       "ucb": None if u == float("inf") else round(u, 4),
                       "score": h.strength * (1.0 if u == float("inf")
                                              else 1.0 + u)})
    scored.sort(key=lambda d: d["score"], reverse=True)
    return scored


# ------------------------------------------------------------------ L2 门控
def l2_gate(hypothesis: Hypothesis, holdout_checked: bool, seeds: int,
            change: Dict, max_l2_per_cycle: int = 1) -> Dict:
    """L2 的门控：比 L0 严得多。The L2 gate: much stricter than L0.

    改学习规则属于"动手术"，要求：
    Changing the learning rule is surgery; it requires:
    - 留出集**必须**验过（否则就是对着训练集调参）
    - ≥2 个 seed（单 seed 无法区分噪声）
    - **单变量**：一次只许改一个旋钮，否则无法归因
    - 假设三件套齐全（没有 prediction 的假设不许动手）
    """
    reasons: List[str] = []
    if not holdout_checked:
        reasons.append("holdout not checked — tuning on the training set")
    if seeds < 2:
        reasons.append(f"only {seeds} seed(s): cannot separate signal from "
                       f"noise, need >= 2")
    if len(change) > max_l2_per_cycle:
        reasons.append(f"{len(change)} knobs at once > {max_l2_per_cycle}: "
                       f"no attribution possible — single variable only")
    if not hypothesis.prediction:
        reasons.append("hypothesis has no prediction — not falsifiable")
    if hypothesis.knob not in L2_KNOBS:
        reasons.append(f"knob {hypothesis.knob} is not an L2 knob")
    return {"ok": not reasons, "reasons": reasons}


def main() -> None:  # pragma: no cover - 手动查看用 / manual inspection
    import argparse
    ap = argparse.ArgumentParser(
        description="元认知诊断：从指标产出可证伪假设"
                    " / metacognition: falsifiable hypotheses from metrics")
    ap.add_argument("--metrics", type=str, default="{}",
                    help='JSON，例如 {"modulation_gap": -1.0, ...}')
    ap.add_argument("--ledger", type=str, default="runs/rsi_ledger.jsonl")
    args = ap.parse_args()
    from pathlib import Path
    from .rsi import Ledger
    m = json.loads(args.metrics)
    hyps = diagnose(m)
    print(json.dumps({"hypotheses": rank(hyps,
                                         Ledger(Path(args.ledger)).read())},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
