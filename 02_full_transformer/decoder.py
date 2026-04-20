"""
Transformer Decoder
====================
Decoder 比 Encoder 多了一个交叉注意力子层，用于接收 Encoder 的上下文信息。

单个 Decoder 层结构：
  ┌─────────────────────────────────────────────┐
  │  输入 (目标序列)                             │
  │  ↓                                           │
  │  Masked Multi-Head Self-Attention            │  ← Causal mask（只看过去）
  │  ↓                                           │
  │  Add & Norm                                  │
  │  ↓                                           │
  │  Multi-Head Cross-Attention                  │  ← Q=Decoder, K/V=Encoder
  │  ↓                                           │
  │  Add & Norm                                  │
  │  ↓                                           │
  │  Feed-Forward Network                        │
  │  ↓                                           │
  │  Add & Norm                                  │
  │  ↓                                           │
  │  输出                                        │
  └─────────────────────────────────────────────┘

关键区别 Encoder vs Decoder：
  - Encoder 自注意力：双向（可看全局），无 causal mask
  - Decoder 自注意力：单向（只看过去），有 causal mask
  - Decoder 交叉注意力：Q 来自 Decoder，K/V 来自 Encoder
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 基础组件（自包含）
# ─────────────────────────────────────────────

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
    def __init__(self, d_model, num_heads, dropout=0.1):
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
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────
# 单个 Decoder 层
# ─────────────────────────────────────────────
class DecoderLayer(nn.Module):
    """
    单个 Transformer Decoder 层（Pre-LN）

    包含三个子层：
        1. Masked Multi-Head Self-Attention（+ causal mask）
        2. Multi-Head Cross-Attention（查询 Encoder 输出）
        3. Feed-Forward Network

    Args:
        d_model:   模型维度
        num_heads: 注意力头数
        d_ff:      FFN 中间层维度
        dropout:   dropout 概率
    """

    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
        super().__init__()

        # 子层 1：Masked 自注意力（Decoder 看自身）
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # 子层 2：交叉注意力（Decoder 看 Encoder）
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # 子层 3：前馈网络
        self.ffn = FeedForward(d_model, d_ff, dropout)

        # 三个归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        enc_output: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        src_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt:        [batch, tgt_len, d_model]  — Decoder 当前输入
            enc_output: [batch, src_len, d_model]  — Encoder 输出
            tgt_mask:   [batch, 1, tgt_len, tgt_len] — Decoder causal mask
            src_mask:   [batch, 1, 1, src_len]       — Encoder padding mask

        Returns:
            [batch, tgt_len, d_model]

        数据流（Pre-LN）：
            tgt
             ↓ norm1
             → Masked Self-Attn → + residual
             ↓ norm2
             → Cross-Attn(Q=self, K/V=enc) → + residual
             ↓ norm3
             → FFN → + residual
        """
        # ── 子层1：Masked 自注意力 ──
        residual = tgt
        tgt = self.norm1(tgt)
        tgt = self.self_attn(Q=tgt, K=tgt, V=tgt, mask=tgt_mask)  # 注意：使用 tgt_mask
        tgt = residual + self.dropout(tgt)

        # ── 子层2：交叉注意力 ──
        # Q 来自 Decoder（当前 tgt）
        # K, V 来自 Encoder（enc_output）
        residual = tgt
        tgt = self.norm2(tgt)
        tgt = self.cross_attn(Q=tgt, K=enc_output, V=enc_output, mask=src_mask)
        tgt = residual + self.dropout(tgt)

        # ── 子层3：前馈网络 ──
        residual = tgt
        tgt = self.norm3(tgt)
        tgt = self.ffn(tgt)
        tgt = residual + self.dropout(tgt)

        return tgt

    @property
    def self_attn_weights(self):
        return self.self_attn.attention_weights

    @property
    def cross_attn_weights(self):
        return self.cross_attn.attention_weights


# ─────────────────────────────────────────────
# 完整 Decoder
# ─────────────────────────────────────────────
class Decoder(nn.Module):
    """
    Transformer Decoder

    Args:
        vocab_size: 词表大小
        d_model:    模型维度
        num_heads:  注意力头数
        num_layers: Decoder 层数
        d_ff:       FFN 中间层维度
        max_len:    最大序列长度
        dropout:    dropout 概率
        pad_idx:    padding token id

    Forward:
        tgt:        [batch, tgt_len]          — 目标序列 token ids
        enc_output: [batch, src_len, d_model] — Encoder 输出
        tgt_mask:   Decoder 自注意力 mask
        src_mask:   Encoder 输出的 padding mask

    Returns:
        [batch, tgt_len, d_model]
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = None,
        max_len: int = 5000,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=self.d_model ** -0.5)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Decoder 自注意力 mask：causal_mask & padding_mask
        [batch, 1, tgt_len, tgt_len]
        """
        tgt_len = tgt.size(1)
        device = tgt.device

        # 下三角 causal mask
        causal = torch.tril(torch.ones(tgt_len, tgt_len, device=device)).bool()
        causal = causal.unsqueeze(0).unsqueeze(0)  # [1, 1, tgt_len, tgt_len]

        # padding mask
        pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, tgt_len]

        return causal & pad_mask  # [B, 1, tgt_len, tgt_len]

    def forward(
        self,
        tgt: torch.Tensor,
        enc_output: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        src_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt:        [batch, tgt_len]
            enc_output: [batch, src_len, d_model]
            tgt_mask:   自动生成（若 None）
            src_mask:   由外部传入（来自 Encoder）
        """
        if tgt_mask is None:
            tgt_mask = self.make_tgt_mask(tgt)

        # 词嵌入 + 位置编码
        x = self.token_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # N 个 Decoder 层
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask=tgt_mask, src_mask=src_mask)

        x = self.final_norm(x)
        return x


# ─────────────────────────────────────────────
# 验证
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Transformer Decoder")
    print("=" * 60)

    vocab_size = 5000
    d_model = 128
    num_heads = 4
    num_layers = 2
    batch_size = 2
    src_len = 15
    tgt_len = 10

    decoder = Decoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=0.0,
    )
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"\n参数总量: {total_params:,}")

    # 模拟输入
    tgt = torch.randint(1, vocab_size, (batch_size, tgt_len))
    tgt[0, -2:] = 0  # padding
    enc_output = torch.randn(batch_size, src_len, d_model)

    # 模拟 src_mask（Encoder 输出的 padding mask）
    src = torch.randint(1, vocab_size, (batch_size, src_len))
    src[1, -3:] = 0
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

    print(f"\ntgt shape: {tgt.shape}")
    print(f"enc_output shape: {enc_output.shape}")

    output = decoder(tgt, enc_output, src_mask=src_mask)
    print(f"Decoder 输出 shape: {output.shape}")  # [2, 10, 128]

    # 验证 causal mask（位置 i 只能看到 j<=i）
    tgt_mask = decoder.make_tgt_mask(tgt)
    print(f"\nDecoder tgt_mask shape: {tgt_mask.shape}")  # [2, 1, 10, 10]
    print("样本0的 tgt_mask（下三角）:")
    print(tgt_mask[0, 0].int())

    # 查看交叉注意力权重
    cross_weights = decoder.layers[-1].cross_attn_weights
    print(f"\n交叉注意力权重 shape: {cross_weights.shape}")  # [2, 4, 10, 15]

    print("\n✅ Decoder 验证通过！")

