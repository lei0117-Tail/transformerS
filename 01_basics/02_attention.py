"""
Step 2: 缩放点积注意力 (Scaled Dot-Product Attention)
======================================================
注意力机制是 Transformer 的核心，让每个 token 都能"关注"序列中的其他 token。

核心公式：
  Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V

直觉理解：
  - Q (Query):  当前 token 想"查询"什么信息
  - K (Key):    每个 token 提供的"索引标签"
  - V (Value):  每个 token 实际携带的"内容信息"
  - Q·Kᵀ:       计算 Query 和每个 Key 的相似度（注意力分数）
  - / √d_k:     缩放，防止点积过大导致 softmax 进入饱和区（梯度消失）
  - softmax:    归一化为概率分布（注意力权重）
  - · V:        用权重对 Value 做加权求和，得到输出

Mask 的作用：
  - padding_mask: 忽略 padding token（填充位）
  - causal_mask:  让 decoder 只能看到当前位置之前的 token（防止未来信息泄露）
"""

import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 缩放点积注意力（函数式）
# ─────────────────────────────────────────────
def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None,
    dropout: nn.Dropout = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    缩放点积注意力

    Args:
        Q:    [batch, heads, seq_q, d_k]
        K:    [batch, heads, seq_k, d_k]
        V:    [batch, heads, seq_k, d_v]
        mask: [batch, 1, seq_q, seq_k] 或 [batch, 1, 1, seq_k]
              mask=True 的位置表示"不应该被关注"（会被填充为 -inf）
        dropout: 可选的 dropout 层

    Returns:
        output:  [batch, heads, seq_q, d_v]  — 注意力加权后的值
        weights: [batch, heads, seq_q, seq_k] — 注意力权重（用于可视化）
    """
    d_k = Q.size(-1)

    # Step 1: 计算注意力分数 Q·Kᵀ / √d_k
    # Q: [batch, heads, seq_q, d_k]
    # K.transpose(-2, -1): [batch, heads, d_k, seq_k]
    # scores: [batch, heads, seq_q, seq_k]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Step 2: 应用 mask
    # 将 mask=True 的位置填充为极小值，softmax 后接近 0
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # Step 3: Softmax 归一化
    # [batch, heads, seq_q, seq_k]
    weights = F.softmax(scores, dim=-1)

    # 处理全 -inf 的行（整行被 mask）：softmax 后是 nan，改回 0
    weights = torch.nan_to_num(weights, nan=0.0)

    if dropout is not None:
        weights = dropout(weights)

    # Step 4: 加权求和
    # weights: [batch, heads, seq_q, seq_k]
    # V:       [batch, heads, seq_k, d_v]
    # output:  [batch, heads, seq_q, d_v]
    output = torch.matmul(weights, V)

    return output, weights


# ─────────────────────────────────────────────
# Mask 工具函数
# ─────────────────────────────────────────────
def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    创建 padding mask，屏蔽掉填充位置。

    Args:
        seq: [batch_size, seq_len]  — token ids
        pad_idx: padding token 的 id（通常为 0）

    Returns:
        mask: [batch_size, 1, 1, seq_len]
              值为 1 表示"有效 token"，值为 0 表示"padding，需屏蔽"
    """
    # seq != pad_idx: True 表示有效位置
    # unsqueeze: 增加 heads 和 query 维度，方便广播
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, seq_len]


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    创建因果（下三角）mask，确保位置 i 只能看到 j <= i 的位置。
    这是 Decoder 自注意力的关键，防止"偷看"未来信息。

    Args:
        seq_len: 序列长度

    Returns:
        mask: [1, 1, seq_len, seq_len]
              下三角矩阵，1 表示可以关注，0 表示不可关注
    """
    # tril: 下三角矩阵（包含对角线）
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]


# ─────────────────────────────────────────────
# 可视化工具
# ─────────────────────────────────────────────
def visualize_attention(weights: torch.Tensor, tokens: list = None, title: str = "Attention Weights"):
    """
    可视化注意力权重热力图

    Args:
        weights: [seq_q, seq_k]
        tokens:  token 标签列表
        title:   图标题
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(weights.detach().cpu().numpy(), cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im)

    if tokens:
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45)
        ax.set_yticklabels(tokens)

    ax.set_xlabel("Key positions")
    ax.set_ylabel("Query positions")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig("attention_weights.png", dpi=150)
    plt.show()
    print("✅ 注意力权重可视化已保存到 attention_weights.png")


# ─────────────────────────────────────────────
# 单元测试 / 快速验证
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Step 2: Scaled Dot-Product Attention")
    print("=" * 60)

    batch_size = 2
    num_heads = 1
    seq_len = 6
    d_k = 32
    d_v = 32

    # 随机生成 Q, K, V
    Q = torch.randn(batch_size, num_heads, seq_len, d_k)
    K = torch.randn(batch_size, num_heads, seq_len, d_k)
    V = torch.randn(batch_size, num_heads, seq_len, d_v)

    print(f"\n输入形状:")
    print(f"  Q: {Q.shape}  K: {K.shape}  V: {V.shape}")

    # ── 1. 无 mask ──
    output, weights = scaled_dot_product_attention(Q, K, V)
    print(f"\n[无 mask] 输出 shape: {output.shape}")   # [2, 1, 6, 32]
    print(f"[无 mask] 权重 shape: {weights.shape}")   # [2, 1, 6, 6]
    print(f"[无 mask] 权重行和（应为1）: {weights[0, 0].sum(dim=-1).detach().numpy().round(3)}")

    # ── 2. Causal mask（Decoder 自注意力）──
    causal_mask = create_causal_mask(seq_len)
    print(f"\nCausal mask (seq_len={seq_len}):\n{causal_mask[0, 0].int()}")

    output_causal, weights_causal = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
    print(f"\n[Causal mask] 注意力权重（第一个样本）:")
    print(weights_causal[0, 0].detach().numpy().round(3))
    print("（上三角应全为0，因为未来位置被屏蔽）")

    # ── 3. Padding mask ──
    # 假设 token ids，最后两个是 padding (id=0)
    token_ids = torch.tensor([[5, 3, 7, 2, 0, 0],
                               [1, 4, 6, 0, 0, 0]])
    padding_mask = create_padding_mask(token_ids, pad_idx=0)
    print(f"\nPadding mask shape: {padding_mask.shape}")  # [2, 1, 1, 6]
    print(f"Padding mask (样本0): {padding_mask[0, 0, 0].int()}")  # [1,1,1,1,0,0]

    output_pad, weights_pad = scaled_dot_product_attention(Q, K, V, mask=padding_mask)
    print(f"\n[Padding mask] 注意力权重（第一个样本，pad 列应全为0）:")
    print(weights_pad[0, 0].detach().numpy().round(3))

    print("\n✅ Step 2 验证通过！")

    # 可选：可视化
    # tokens = ["I", "love", "NLP", "!", "<pad>", "<pad>"]
    # visualize_attention(weights_causal[0, 0], tokens, "Causal Attention")

