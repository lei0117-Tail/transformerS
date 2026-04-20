"""
Step 6: 完整 Encoder 层 (Encoder Layer)
=======================================
把前面实现的所有模块组合成一个完整的 Transformer Encoder 层。

结构（原论文 Post-LN 版本）：
  ┌─────────────────────────────────────┐
  │  输入 x                             │
  │  ↓                                  │
  │  Multi-Head Self-Attention          │
  │  ↓                                  │
  │  Add & Norm  (残差 + LayerNorm)     │
  │  ↓                                  │
  │  Feed-Forward Network               │
  │  ↓                                  │
  │  Add & Norm  (残差 + LayerNorm)     │
  │  ↓                                  │
  │  输出                               │
  └─────────────────────────────────────┘

重要：Encoder 的注意力是双向的（每个 token 可以看到所有其他 token），
      这与 Decoder 的单向 causal mask 不同。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 复用前面步骤的组件（内联，避免跨模块导入问题）──

def scaled_dot_product_attention(Q, K, V, mask=None, dropout=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    if dropout is not None:
        weights = dropout(weights)
    return torch.matmul(weights, V), weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    def _split(self, x):
        B, S, _ = x.shape
        return x.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

    def _merge(self, x):
        B, _, S, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, S, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q, K, V = self.W_Q(Q), self.W_K(K), self.W_V(V)
        Q, K, V = self._split(Q), self._split(K), self._split(V)
        out, self.attention_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout
        )
        return self.W_O(self._merge(out))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ─────────────────────────────────────────────
# Encoder 层（核心）
# ─────────────────────────────────────────────
class EncoderLayer(nn.Module):
    """
    单个 Transformer Encoder 层

    包含：
        1. 多头自注意力 (Multi-Head Self-Attention)
        2. Add & Norm（残差 + 层归一化）
        3. 前馈网络 (Feed-Forward Network)
        4. Add & Norm（残差 + 层归一化）

    Args:
        d_model:   模型维度
        num_heads: 注意力头数
        d_ff:      FFN 中间层维度（默认 4×d_model）
        dropout:   dropout 概率
        pre_norm:  True = Pre-LN（现代），False = Post-LN（原论文）
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = None,
        dropout: float = 0.1,
        pre_norm: bool = True,
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.pre_norm = pre_norm

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:        [batch, seq_len, d_model]  — 输入序列
            src_mask: [batch, 1, 1, seq_len]      — padding mask（可选）

        Returns:
            [batch, seq_len, d_model]

        数据流（Pre-LN）：
            x → norm1 → self_attn → dropout → + x → norm2 → ffn → dropout → + x
        """
        if self.pre_norm:
            # ─── Pre-LN（训练更稳定，推荐）───
            # 子层 1：自注意力
            residual = x
            x = self.norm1(x)
            x = self.self_attn(Q=x, K=x, V=x, mask=src_mask)
            x = self.dropout(x)
            x = residual + x  # 残差连接

            # 子层 2：前馈网络
            residual = x
            x = self.norm2(x)
            x = self.ffn(x)
            x = self.dropout(x)
            x = residual + x  # 残差连接
        else:
            # ─── Post-LN（原论文）───
            # 子层 1：自注意力
            attn_out = self.self_attn(Q=x, K=x, V=x, mask=src_mask)
            x = self.norm1(x + self.dropout(attn_out))  # Add & Norm

            # 子层 2：前馈网络
            ffn_out = self.ffn(x)
            x = self.norm2(x + self.dropout(ffn_out))   # Add & Norm

        return x

    @property
    def attention_weights(self):
        """访问当前层的注意力权重（用于可视化）"""
        return self.self_attn.attention_weights


# ─────────────────────────────────────────────
# 单元测试 / 快速验证
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Step 6: Encoder Layer (完整)")
    print("=" * 60)

    batch_size = 2
    seq_len = 12
    d_model = 128
    num_heads = 8
    d_ff = 512

    x = torch.randn(batch_size, seq_len, d_model)
    print(f"\n输入 shape: {x.shape}")

    # ── Pre-LN Encoder 层 ──
    encoder_layer_pre = EncoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.0,
        pre_norm=True,
    )
    out_pre = encoder_layer_pre(x)
    print(f"\n[Pre-LN] 输出 shape: {out_pre.shape}")    # [2, 12, 128]
    print(f"[Pre-LN] 注意力权重 shape: {encoder_layer_pre.attention_weights.shape}")

    # ── Post-LN Encoder 层（原论文）──
    encoder_layer_post = EncoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.0,
        pre_norm=False,
    )
    out_post = encoder_layer_post(x)
    print(f"\n[Post-LN] 输出 shape: {out_post.shape}")  # [2, 12, 128]

    # ── 带 padding mask ──
    # 假设序列后 2 个 token 是 padding
    pad_mask = torch.ones(batch_size, 1, 1, seq_len)
    pad_mask[:, :, :, -2:] = 0  # 最后 2 个位置被屏蔽
    out_masked = encoder_layer_pre(x, src_mask=pad_mask)
    print(f"\n[带 Mask] 输出 shape: {out_masked.shape}")

    # ── 参数量统计 ──
    total_params = sum(p.numel() for p in encoder_layer_pre.parameters())
    print(f"\nEncoder 层参数量: {total_params:,}")
    # Self-Attn: 4 × d²  = 4 × 128² = 65536
    # FFN:  W1: 128×512 + W2: 512×128 = 131072
    # Norm: 2 × (2 × d_model) = 512
    # 总计 ≈ 197,120

    print("\n✅ Step 6 验证通过！")

    # ── 理解每个组件的作用 ──
    print("\n─── 组件拆解 ───")
    print("1. MultiHeadAttention: 让每个 token 聚合全局信息")
    print("2. FeedForward:        对每个 token 做非线性变换")
    print("3. LayerNorm:          归一化，稳定训练")
    print("4. Residual:           梯度高速公路，防止梯度消失")

