"""
Masks 工具模块
==============
Transformer 中两种关键的 Mask：

1. Padding Mask（填充掩码）
   - 用途：屏蔽填充位置（<pad> token），让注意力不关注无意义的填充
   - 使用场景：Encoder 自注意力、Decoder 交叉注意力
   - 形状：[batch, 1, 1, seq_len]

2. Causal Mask（因果掩码 / 未来掩码）
   - 用途：Decoder 自注意力，位置 i 只能看到位置 0..i（防止信息泄露）
   - 使用场景：Decoder 自注意力
   - 形状：[1, 1, seq_len, seq_len]，下三角矩阵

3. 组合 Mask（Decoder 自注意力）
   - = causal_mask & padding_mask
   - 同时屏蔽未来位置和 padding 位置
"""

import torch


def make_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    生成 padding mask

    Args:
        seq:     [batch, seq_len]  — token ids
        pad_idx: padding token 的 id（通常为 0）

    Returns:
        mask: [batch, 1, 1, seq_len]
              1 = 有效 token（可以关注），0 = padding（需屏蔽）

    Example:
        tokens = [[5, 3, 7, 0, 0],
                  [1, 4, 0, 0, 0]]
        mask   = [[[[1, 1, 1, 0, 0]],
                  [[1, 1, 0, 0, 0]]]]
    """
    return (seq != pad_idx).bool().unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]


def make_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    生成下三角因果 mask（Decoder 自注意力用）

    Args:
        seq_len: 序列长度
        device:  目标设备

    Returns:
        mask: [1, 1, seq_len, seq_len]
              下三角为 1（可关注），上三角为 0（屏蔽未来）

    Example (seq_len=4):
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool().bool()
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]


def make_decoder_self_attn_mask(
    tgt: torch.Tensor, pad_idx: int = 0
) -> torch.Tensor:
    """
    Decoder 自注意力的组合 mask = causal_mask & padding_mask
    同时屏蔽：(1) 未来位置，(2) padding 位置

    Args:
        tgt:     [batch, tgt_len]  — Decoder 输入 token ids
        pad_idx: padding id

    Returns:
        mask: [batch, 1, tgt_len, tgt_len]
    """
    tgt_len = tgt.size(1)
    device = tgt.device

    # causal mask:   [1, 1, tgt_len, tgt_len]
    causal = make_causal_mask(tgt_len, device=device)

    # padding mask:  [batch, 1, 1, tgt_len]
    pad_mask = make_padding_mask(tgt, pad_idx)

    # 组合：两者都为 1 的位置才允许关注
    return causal & pad_mask   # [batch, 1, tgt_len, tgt_len]


def make_cross_attn_mask(src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Encoder-Decoder 交叉注意力的 mask
    屏蔽 Encoder 输出中的 padding 位置

    Args:
        src:     [batch, src_len]  — Encoder 输入 token ids
        pad_idx: padding id

    Returns:
        mask: [batch, 1, 1, src_len]
    """
    return make_padding_mask(src, pad_idx)


# ─────────────────────────────────────────────
# 验证
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Masks 工具验证")
    print("=" * 50)

    batch_size = 2
    src_len = 6
    tgt_len = 5

    # 模拟有 padding 的 token ids
    src = torch.tensor([[5, 3, 7, 2, 0, 0],
                        [1, 4, 6, 9, 3, 0]])  # [2, 6]
    tgt = torch.tensor([[8, 2, 4, 0, 0],
                        [3, 7, 1, 5, 0]])     # [2, 5]

    # Padding mask
    pad_mask = make_padding_mask(src)
    print(f"\nPadding mask shape: {pad_mask.shape}")
    print("src[0] padding mask:", pad_mask[0, 0, 0].int().tolist())  # [1,1,1,1,0,0]

    # Causal mask
    causal_mask = make_causal_mask(tgt_len)
    print(f"\nCausal mask shape: {causal_mask.shape}")
    print("Causal mask:")
    print(causal_mask[0, 0].int())

    # Decoder 自注意力 mask
    dec_self_mask = make_decoder_self_attn_mask(tgt)
    print(f"\nDecoder self-attn mask shape: {dec_self_mask.shape}")
    print("样本0（有 padding）:")
    print(dec_self_mask[0, 0].int())

    # 交叉注意力 mask
    cross_mask = make_cross_attn_mask(src)
    print(f"\nCross-attn mask shape: {cross_mask.shape}")
    print("src[1] cross mask:", cross_mask[1, 0, 0].int().tolist())  # [1,1,1,1,1,0]

    print("\n✅ Masks 验证通过！")

