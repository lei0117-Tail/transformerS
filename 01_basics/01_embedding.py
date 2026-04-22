"""
Step 1: 词嵌入 (Token Embedding) + 位置编码 (Positional Encoding)
=================================================================
Transformer 没有 RNN/CNN 的时序结构，因此需要显式地给每个 token 注入位置信息。

核心思想：映射到连续的向量空间
  - Token Embedding: 把离散的词 ID
  - Positional Encoding: 用 sin/cos 函数编码位置，让模型感知 token 的相对/绝对位置

公式：
  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
"""

import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# 1. Token Embedding
# ─────────────────────────────────────────────
class TokenEmbedding(nn.Module):
    """
    把 token id (整数) 转换为稠密向量。
    本质就是一个查找表: vocab_size × d_model
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len]  —— token ids
        Returns:
            [batch_size, seq_len, d_model]
        """
        # 原论文将 embedding 权重乘以 sqrt(d_model) 来缩放，
        # 防止 positional encoding 的信号被淹没
        return self.embedding(x) * math.sqrt(self.d_model)


# ─────────────────────────────────────────────
# 2. Positional Encoding
# ─────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    """
    固定式位置编码（Sinusoidal）。
    不需要学习，直接用公式生成，好处是可以外推到训练时没见过的序列长度。

    Tips:
      - 偶数维度用 sin，奇数维度用 cos
      - dropout 防止模型过度依赖位置信息
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 预计算位置编码矩阵，shape: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # position: [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # div_term: [d_model/2]
        # 等价于 1 / 10000^(2i/d_model)，用 exp(log) 形式计算更稳定
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维

        # 增加 batch 维度，变成 [1, max_len, d_model]，方便广播
        pe = pe.unsqueeze(0)

        # register_buffer: 不是模型参数（不会被 optimizer 更新），
        # 但会随模型一起 save/load，并随模型移动到 GPU
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        # self.pe[:, :x.size(1)] 取前 seq_len 个位置
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────
# 3. 组合：TransformerEmbedding
# ─────────────────────────────────────────────
class TransformerEmbedding(nn.Module):
    """
    完整的 Transformer 嵌入层 = Token Embedding + Positional Encoding
    """

    def __init__(self, vocab_size: int, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.token_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len]
        Returns:
            [batch_size, seq_len, d_model]
        """
        token_emb = self.token_emb(x)   # 词向量
        return self.pos_enc(token_emb)  # + 位置编码


# ─────────────────────────────────────────────
# 4. 可视化验证
# ─────────────────────────────────────────────
def visualize_positional_encoding(d_model: int = 64, max_len: int = 100):
    """可视化位置编码矩阵，验证 sin/cos 的周期性"""
    pe_module = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.0)
    # pe shape: [1, max_len, d_model]
    pe_matrix = pe_module.pe.squeeze(0).numpy()  # [max_len, d_model]

    plt.figure(figsize=(12, 6))
    plt.pcolormesh(pe_matrix, cmap="RdBu")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Position")
    plt.title(f"Positional Encoding  (d_model={d_model}, max_len={max_len})")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("pe_visualization.png", dpi=150)
    plt.show()
    print("✅ 位置编码可视化已保存到 pe_visualization.png")


# ─────────────────────────────────────────────
# 5. 单元测试 / 快速验证
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Step 1: Token Embedding + Positional Encoding")
    print("=" * 60)

    # 超参数
    batch_size = 2   # 每一批次可以同时处理多少个句子（并行计算）
    seq_len = 10     # 每个句子的固定长度（token 数量）
    vocab_size = 1000  # 词表大小：模型认识多少个不同的词（或字）
    # 词表大小 = 模型认识多少个不同的词（或字）。
    # token id 的范围是 [0, vocab_size-1]。
    # 真实模型：BERT 约 30000，GPT-2 约 50000。
    # 这里用 1000 只是演示。
    '''
    vocab = {
    "<pad>": 0,    # 特殊token：填充
    "<unk>": 1,    # 特殊token：未知词
    "我":    10,
    "你":    11,
    "他":    12,
    "喜欢":  100,
    "讨厌":  101,
    "吃饭":  200,
    ...
}
    '''
    d_model = 64      # 嵌入维度：每个 token 映射成多长的向量（原论文用 512，这里演示用 64）

    # 构造随机 token ids：模拟一个 batch 的输入，每个值代表一个 token 的 id
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"\n输入 token ids shape: {x.shape}")  # [2, 10] → 2个句子，每个10个token

    # ── Token Embedding ──
    token_emb = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)
    emb_out = token_emb(x)  # 查表：把每个 token id 转换成 d_model 维向量，并乘以 √d_model 缩放
    print(f"Token Embedding 输出 shape: {emb_out.shape}")  # [2, 10, 64] → 每个token变成64维向量

    # ── Positional Encoding ──
    pos_enc = PositionalEncoding(d_model=d_model, dropout=0.0)
    pos_out = pos_enc(emb_out)  # 加上位置信息：token向量 + sin/cos位置编码
    print(f"+ Positional Encoding 输出 shape: {pos_out.shape}")  # [2, 10, 64] → 形状不变，融合了位置信息

    # ── 完整 TransformerEmbedding（一步到位）──
    transformer_emb = TransformerEmbedding(vocab_size=vocab_size, d_model=d_model)
    full_out = transformer_emb(x)  # 等价于先 Token Embedding 再加 Positional Encoding
    print(f"TransformerEmbedding 输出 shape: {full_out.shape}")  # [2, 10, 64]

    # 验证位置编码的不同位置值不同（用全零输入排除语义干扰，只看位置编码的效果）
    pe_module = PositionalEncoding(d_model=d_model, dropout=0.0)
    dummy = torch.zeros(1, 5, d_model)  # 全零张量，shape: [1, 5, 64]，模拟5个位置的空输入
    pe_out = pe_module(dummy)  # 输出 = 0 + 位置编码，所以输出就是纯位置编码
    print(f"\n位置 0 的编码 (前8维): {pe_out[0, 0, :8].detach().numpy().round(3)}")
    print(f"位置 1 的编码 (前8维): {pe_out[0, 1, :8].detach().numpy().round(3)}")
    print(f"位置 2 的编码 (前8维): {pe_out[0, 2, :8].detach().numpy().round(3)}")

    print("\n✅ Step 1 验证通过！")

    # 可选：可视化
    # visualize_positional_encoding()

