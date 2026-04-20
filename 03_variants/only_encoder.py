"""
变体 1: Only Encoder（BERT 风格）
==================================
只使用 Transformer 的 Encoder 部分，用于理解任务（判别式）。

代表模型：
  - BERT (Bidirectional Encoder Representations from Transformers)
  - RoBERTa, ALBERT, ELECTRA, DeBERTa...

核心特点：
  - 双向注意力：每个 token 可以同时看到左边和右边的上下文
  - 适合：文本分类、序列标注、问答、文本相似度等"理解"任务
  - 不适合：文本生成（因为训练时使用双向上下文，生成时无法做到）

训练范式：
  - Masked Language Modeling (MLM)：随机 mask 15% token，预测被 mask 的词
  - Next Sentence Prediction (NSP)：预测两个句子是否相邻（BERT 原版）

本文件实现：
  1. BertLikeEncoder：核心 Only Encoder 模块
  2. BertForClassification：文本分类头
  3. BertForTokenClassification：序列标注头
  4. BertForMLM：预训练用的 MLM 头
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 基础组件
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
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
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
    def __init__(self, d_model, d_ff=None, dropout=0.1, activation="gelu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            act,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout, activation="gelu")  # BERT 用 GELU
        self.norm1 = nn.LayerNorm(d_model, eps=1e-12)  # BERT 用较小的 eps
        self.norm2 = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Post-LN（BERT 原版风格）
        attn_out = self.self_attn(Q=x, K=x, V=x, mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


# ─────────────────────────────────────────────
# BERT 风格位置编码（可学习，而非固定 sin/cos）
# ─────────────────────────────────────────────
class BertEmbeddings(nn.Module):
    """
    BERT 的嵌入层（与原始 Transformer 不同：使用可学习的位置编码）

    BERT Embedding = Token Embedding + Position Embedding + Segment Embedding

    Segment Embedding：区分句子 A 和句子 B（用于 NSP 任务）
    Position Embedding：可学习的，而非固定的 sin/cos（BERT 的特点）
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int = 512,
        num_segments: int = 2,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.position_embedding = nn.Embedding(max_len, d_model)  # 可学习！
        self.segment_embedding = nn.Embedding(num_segments, d_model)  # 句子A/B
        self.norm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:   [batch, seq_len]
            segment_ids: [batch, seq_len]  0=句子A, 1=句子B（默认全0）
        """
        seq_len = input_ids.size(1)
        device = input_ids.device

        # 位置 ids：0, 1, 2, ..., seq_len-1
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0)  # [1, seq_len]

        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)

        # 三种 embedding 相加
        embeddings = (
            self.token_embedding(input_ids)
            + self.position_embedding(position_ids)
            + self.segment_embedding(segment_ids)
        )

        return self.dropout(self.norm(embeddings))


# ─────────────────────────────────────────────
# Only Encoder 主体
# ─────────────────────────────────────────────
class BertLikeEncoder(nn.Module):
    """
    BERT 风格的 Only Encoder

    Args:
        vocab_size:   词表大小
        d_model:      模型维度（BERT-base: 768）
        num_heads:    注意力头数（BERT-base: 12）
        num_layers:   层数（BERT-base: 12）
        d_ff:         FFN 维度（BERT-base: 3072）
        max_len:      最大序列长度（BERT: 512）
        dropout:      dropout 概率
        pad_idx:      padding token id

    Returns（forward）:
        last_hidden_state:  [batch, seq_len, d_model]  — 每个 token 的表示
        cls_representation: [batch, d_model]           — [CLS] token 的表示（用于分类）
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        d_ff: int = 3072,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.embeddings = BertEmbeddings(vocab_size, d_model, max_len, dropout=dropout, pad_idx=pad_idx)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Pooler：BERT 用来提取 [CLS] token 表示，接一个 tanh 激活的线性层
        self.pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )

    def make_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """生成 padding mask [batch, 1, 1, seq_len]"""
        return (input_ids != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ) -> tuple:
        """
        Args:
            input_ids:        [batch, seq_len]
            segment_ids:      [batch, seq_len]（默认全0）
            attention_mask:   [batch, seq_len]（1=有效，0=pad；若None则自动生成）

        Returns:
            (last_hidden_state, cls_output)
            - last_hidden_state: [batch, seq_len, d_model]
            - cls_output:        [batch, d_model]  — [CLS] 的 pooled 表示
        """
        # 构建 mask
        if attention_mask is None:
            mask = self.make_mask(input_ids)
        else:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # 嵌入
        x = self.embeddings(input_ids, segment_ids)

        # N 个 Encoder 层
        for layer in self.layers:
            x = layer(x, mask=mask)

        # [CLS] token（位置 0）的表示，经过 pooler 用于分类
        cls_output = self.pooler(x[:, 0, :])

        return x, cls_output  # (last_hidden_state, cls_output)


# ─────────────────────────────────────────────
# 下游任务头
# ─────────────────────────────────────────────
class BertForClassification(nn.Module):
    """
    文本分类（单句或句对）

    使用 [CLS] token 的 pooled 表示，接一个分类头。
    适用：情感分析、文本分类、句子关系判断等
    """

    def __init__(self, bert: BertLikeEncoder, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.bert = bert
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(bert.d_model, num_classes),
        )

    def forward(self, input_ids, segment_ids=None, attention_mask=None):
        """
        Returns:
            logits: [batch, num_classes]
        """
        _, cls_output = self.bert(input_ids, segment_ids, attention_mask)
        return self.classifier(cls_output)


class BertForTokenClassification(nn.Module):
    """
    序列标注（对每个 token 做分类）

    适用：命名实体识别（NER）、词性标注（POS）等
    """

    def __init__(self, bert: BertLikeEncoder, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.bert = bert
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(bert.d_model, num_labels),
        )

    def forward(self, input_ids, segment_ids=None, attention_mask=None):
        """
        Returns:
            logits: [batch, seq_len, num_labels]
        """
        last_hidden_state, _ = self.bert(input_ids, segment_ids, attention_mask)
        return self.classifier(last_hidden_state)


class BertForMLM(nn.Module):
    """
    Masked Language Modeling（预训练任务）

    对所有 token 位置预测词表中的词，
    训练时只计算被 [MASK] 位置的 loss。
    """

    def __init__(self, bert: BertLikeEncoder):
        super().__init__()
        self.bert = bert
        self.mlm_head = nn.Sequential(
            nn.Linear(bert.d_model, bert.d_model),
            nn.GELU(),
            nn.LayerNorm(bert.d_model, eps=1e-12),
        )
        # 输出投影共享 embedding 权重
        self.output_projection = nn.Linear(bert.d_model, bert.embeddings.token_embedding.num_embeddings)
        self.output_projection.weight = bert.embeddings.token_embedding.weight

    def forward(self, input_ids, segment_ids=None, attention_mask=None):
        """
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        last_hidden_state, _ = self.bert(input_ids, segment_ids, attention_mask)
        hidden = self.mlm_head(last_hidden_state)
        return self.output_projection(hidden)


# ─────────────────────────────────────────────
# 验证
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("变体1: Only Encoder (BERT 风格)")
    print("=" * 60)

    # 小模型配置（BERT-tiny）
    bert = BertLikeEncoder(
        vocab_size=1000,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=512,
        max_len=64,
        dropout=0.0,
    )
    total_params = sum(p.numel() for p in bert.parameters())
    print(f"\nBERT-tiny 参数量: {total_params:,}")

    batch_size = 2
    seq_len = 16

    # 模拟输入（[CLS]=1, [SEP]=2, [MASK]=3, [PAD]=0）
    input_ids = torch.randint(4, 1000, (batch_size, seq_len))
    input_ids[:, 0] = 1   # [CLS] token
    input_ids[0, -3:] = 0  # padding

    segment_ids = torch.zeros_like(input_ids)
    segment_ids[1, 8:] = 1  # 第二个句子用 segment 1

    print(f"\n输入 input_ids shape: {input_ids.shape}")
    print(f"segment_ids[1]: {segment_ids[1].tolist()}")

    last_hidden, cls_out = bert(input_ids, segment_ids)
    print(f"\n[Encoder] last_hidden shape: {last_hidden.shape}")  # [2, 16, 128]
    print(f"[Encoder] cls_output shape: {cls_out.shape}")          # [2, 128]

    # ── 文本分类 ──
    classifier = BertForClassification(bert, num_classes=3)
    logits_cls = classifier(input_ids, segment_ids)
    print(f"\n[分类] logits shape: {logits_cls.shape}")             # [2, 3]
    print(f"[分类] 预测类别: {logits_cls.argmax(dim=-1).tolist()}")

    # ── 序列标注 ──
    ner_model = BertForTokenClassification(bert, num_labels=5)
    logits_ner = ner_model(input_ids, segment_ids)
    print(f"\n[NER] logits shape: {logits_ner.shape}")              # [2, 16, 5]

    # ── MLM ──
    mlm_model = BertForMLM(bert)
    logits_mlm = mlm_model(input_ids, segment_ids)
    print(f"\n[MLM] logits shape: {logits_mlm.shape}")              # [2, 16, 1000]

    print("\n✅ Only Encoder 验证通过！")
    print("\n核心总结：")
    print("  - Only Encoder = 双向注意力 = 擅长理解")
    print("  - [CLS] token 聚合全局信息 → 用于分类")
    print("  - 每个 token 表示 → 用于序列标注")
    print("  - BERT 用 GELU 激活，可学习位置编码，Post-LN")

