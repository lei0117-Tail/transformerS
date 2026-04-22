"""
Transformer Encoder
====================
将 N 个 EncoderLayer 堆叠起来，加上嵌入层，构成完整的 Encoder。

结构：
  Token Embedding
       +
  Positional Encoding
       ↓
  EncoderLayer × N
       ↓
  Final LayerNorm（Pre-LN 方式需要）
       ↓
  输出：[batch, src_len, d_model]

Encoder 的输出会被传给 Decoder（作为交叉注意力的 K 和 V），
或直接用于下游任务（BERT 风格 Only Encoder）。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 基础组件（自包含，避免跨路径导入问题）
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


class EncoderLayer(nn.Module):
    """单个 Encoder 层（Pre-LN 风格）"""

    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # 子层 1：多头自注意力
        residual = x
        x = self.norm1(x)
        x = self.self_attn(Q=x, K=x, V=x, mask=src_mask)
        x = residual + self.dropout(x)

        # 子层 2：前馈网络
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)

        return x

    @property
    def attention_weights(self):
        return self.self_attn.attention_weights


# ─────────────────────────────────────────────
# Positional Encoding
# ─────────────────────────────────────────────
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
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────
# 完整 Encoder
# ─────────────────────────────────────────────
class Encoder(nn.Module):
    """
    Transformer Encoder

    Args:
        vocab_size: 词表大小
        d_model:    模型维度
        num_heads:  注意力头数
        num_layers: Encoder 层数
        d_ff:       FFN 中间层维度（默认 4×d_model）
        max_len:    最大序列长度
        dropout:    dropout 概率
        pad_idx:    padding token id

    Forward:
        src:      [batch, src_len]     — 输入 token ids
        src_mask: [batch, 1, 1, src_len] — 可选的 padding mask

    Returns:
        [batch, src_len, d_model]  — 上下文化的 token 表示
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

        # 嵌入层
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # N 个 Encoder 层（深拷贝保证独立参数）
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Pre-LN 最后需要一个额外的 LayerNorm
        self.final_norm = nn.LayerNorm(d_model)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """Xavier 均匀初始化线性层权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=self.d_model ** -0.5)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """自动生成 padding mask"""
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, src_len]

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            src:      [batch, src_len]
            src_mask: [batch, 1, 1, src_len]（若为 None，自动生成）

        Returns:
            [batch, src_len, d_model]
        """
        if src_mask is None:
            src_mask = self.make_src_mask(src)

        # Step 1: 词嵌入 + 位置编码
        # 原论文将 embedding 乘以 √d_model 来缩放
        x = self.token_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Step 2: N 个 Encoder 层
        for layer in self.layers:
            x = layer(x, src_mask=src_mask)

        # Step 3: 最终归一化（Pre-LN 需要）
        x = self.final_norm(x)

        return x


# ─────────────────────────────────────────────
# 验证
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Transformer Encoder")
    print("=" * 60)

    # 原论文 base 配置
    config = dict(
        vocab_size=10000,   # 词表大小：模型认识多少个不同的 token
        d_model=512,        # 模型维度：每个 token 的向量长度
        num_heads=8,        # 注意力头数：把 d_model 分成 8 个子空间
        num_layers=6,       # Encoder 层数：堆叠 6 个 EncoderLayer
        d_ff=2048,          # FFN 中间层维度：4 × d_model（瓶颈-扩张结构）
        dropout=0.0,
    )

    encoder = Encoder(**config)
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\n模型配置: {config}")
    print(f"参数总量: {total_params:,}")

    # 打印关键组件的权重形状
    print(f"\n{'='*50}")
    print("各组件权重形状:")
    print(f"{'='*50}")
    print(f"  Token Embedding: {encoder.token_embedding.weight.shape}  [vocab={config['vocab_size']}, d_model={config['d_model']}]")
    print(f"  Positional Encoding pe: {encoder.pos_encoding.pe.shape}     [1, max_len=5000, d_model={config['d_model']}]")
    layer0 = encoder.layers[0]
    print(f"  Layer0 Self-Attn W_Q: {layer0.self_attn.W_Q.weight.shape}   [d_model, d_model] = [{config['d_model']}, {config['d_model']}]")
    print(f"  Layer0 Self-Attn W_O: {layer0.self_attn.W_O.weight.shape}   [d_model, d_model]")
    print(f"  Layer0 FFN linear1:   {layer0.ffn.net[0].weight.shape}         [d_ff={config['d_ff']}, d_model={config['d_model']}]")
    print(f"  Layer0 FFN linear2:   {layer0.ffn.net[3].weight.shape}         [d_model={config['d_model']}, d_ff={config['d_ff']}]")

    # 前向传播测试
    batch_size = 2
    src_len = 20
    src = torch.randint(1, config["vocab_size"], (batch_size, src_len))
    src[0, -3:] = 0  # 添加 padding

    print(f"\n{'='*50}")
    print("逐步跟踪 Encoder 前向传播:")
    print(f"{'='*50}")
    print(f"  输入 src (token ids): {src.shape}              [batch={batch_size}, src_len={src_len}]")

    # Step 1: Embedding
    emb_out = encoder.token_embedding(src) * math.sqrt(encoder.d_model)
    print(f"  Token Embedding 后:   {emb_out.shape}            [batch, src_len, d_model={config['d_model']}]")

    pos_out = encoder.pos_encoding(emb_out)
    print(f"  + Position Encoding:  {pos_out.shape}            [batch, src_len, d_model] ← 形状不变，融合了位置信息")

    # Step 2: Encoder Layers（只跟踪第一层）
    x = pos_out
    src_mask = encoder.make_src_mask(src)
    print(f"  src_mask:             {src_mask.shape}           [batch, 1, 1, src_len] ← padding mask")

    x_after_attn = encoder.layers[0](x, src_mask=src_mask)
    print(f"  经过 EncoderLayer 0:  {x_after_attn.shape}      [batch, src_len, d_model] ← 形状不变！")

    # Step 3: 完整前向传播
    output = encoder(src)
    print(f"\n  最终输出 (6层后):     {output.shape}            [batch, src_len, d_model]")

    # 验证 padding 位置的输出（有 mask 但 LM 本身不强制为0）
    print(f"输出[0,0,:4]: {output[0, 0, :4].detach().numpy().round(3)}")

    # 小模型快速测试
    small_encoder = Encoder(vocab_size=1000, d_model=64, num_heads=4, num_layers=2, dropout=0.0)
    src_small = torch.randint(1, 1000, (2, 8))
    out_small = small_encoder(src_small)
    print(f"\n[小模型] 输出 shape: {out_small.shape}")  # [2, 8, 64]

    print("\n✅ Encoder 验证通过！")

