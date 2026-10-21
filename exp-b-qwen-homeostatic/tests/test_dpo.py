"""DPO 损失与批处理的单元测试（只需 torch，不需要模型/显卡）。
Unit tests for the DPO loss and batching (torch only, no model / GPU).

标签方向错了，训练得越好越糟（v2 反向调制就是这个下场），所以损失
本身的方向必须被测住：chosen 优势越大，损失越低。
If the loss pointed the wrong way, better training would only make
things worse (that was v2's inversion), so the direction of the loss
itself must be pinned: the bigger the chosen advantage, the lower the
loss.
"""
import torch

from homeo_lm.dpo import _collate, dpo_loss


def test_neutral_pairs_give_log_two():
    """策略与参考完全一致时，DPO 损失 = log 2（随机猜的水平）。
    When the policy equals the reference, the DPO loss is log 2 (the
    chance level)."""
    pol_c = torch.tensor([0.0, 0.0])
    pol_r = torch.tensor([0.0, 0.0])
    ref_c = torch.tensor([0.0, 0.0])
    ref_r = torch.tensor([0.0, 0.0])
    loss, m = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.1)
    assert torch.allclose(loss, torch.log(torch.tensor(2.0)), atol=1e-6)
    assert m["margin"] == 0.0
    assert m["acc"] == 0.5   # margin > 0 的比例（等于 0 时算不中）


def test_preferring_chosen_lowers_loss():
    """策略偏向 chosen（Δc > Δr）→ 损失下降、margin 为正。
    Preferring chosen (Δc > Δr) lowers the loss and gives a positive
    margin."""
    loss_pos, m_pos = dpo_loss(torch.tensor([1.0]), torch.tensor([0.0]),
                               torch.tensor([0.0]), torch.tensor([0.0]),
                               beta=0.1)
    loss_neg, m_neg = dpo_loss(torch.tensor([0.0]), torch.tensor([1.0]),
                               torch.tensor([0.0]), torch.tensor([0.0]),
                               beta=0.1)
    assert loss_pos < loss_neg
    assert m_pos["margin"] > 0 and m_neg["margin"] < 0
    assert m_pos["acc"] == 1.0 and m_neg["acc"] == 0.0


def test_beta_scales_the_margin():
    """β 只缩放 margin（进而缩放梯度的偏好强度），不改方向。
    β only scales the margin (and thus the preference strength of the
    gradient); it never flips the direction."""
    _, m1 = dpo_loss(torch.tensor([1.0]), torch.tensor([0.0]),
                     torch.tensor([0.0]), torch.tensor([0.0]), beta=0.1)
    _, m2 = dpo_loss(torch.tensor([1.0]), torch.tensor([0.0]),
                     torch.tensor([0.0]), torch.tensor([0.0]), beta=0.5)
    assert abs(m2["margin"] - 5 * m1["margin"]) < 1e-5


def test_reference_shifts_cancel():
    """参考模型的绝对大小无关，只有"策略相对参考的偏移"起作用。
    Only the policy's offset from the reference matters; the absolute
    scale of the reference cancels out."""
    base = torch.tensor([3.0])
    l1, _ = dpo_loss(base + 1.0, base, base, base, beta=0.1)
    l2, _ = dpo_loss(torch.tensor([1.0]), torch.tensor([0.0]),
                     torch.tensor([0.0]), torch.tensor([0.0]), beta=0.1)
    assert torch.allclose(l1, l2, atol=1e-6)


def test_collate_pads_and_masks():
    """chosen / rejected 各自填充到同长，attention 与 assistant 掩码
    对齐。chosen / rejected are padded to a common length, with the
    attention and assistant masks aligned."""
    batch = [
        {"c_ids": [1, 2, 3], "c_mask": [0, 0, 1],
         "r_ids": [4, 5], "r_mask": [0, 1]},
        {"c_ids": [6, 7, 8, 9], "c_mask": [0, 0, 1, 1],
         "r_ids": [1, 1, 1], "r_mask": [0, 0, 1]},
    ]
    out = _collate(batch, pad_id=0)
    assert out["c_ids"].shape == (2, 4)
    assert out["r_ids"].shape == (2, 3)
    # 第二条 c 长 4，无需填充 / the second c is already length 4
    assert out["c_attn"][1].tolist() == [1, 1, 1, 1]
    assert out["c_attn"][0].tolist() == [1, 1, 1, 0]
    # assistant 掩码只在真实 token 上为 1，填充位为 0
    assert out["c_amask"][0].tolist() == [0, 0, 1, 0]
    assert out["r_amask"][1].tolist() == [0, 0, 1]
