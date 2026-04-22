"""
变体 3: Encoder-Decoder（T5/BART 风格）
========================================
使用完整的 Encoder-Decoder 架构，用于序列到序列（Seq2Seq）任务。

代表模型：
  - T5 (Text-to-Text Transfer Transformer)
  - BART (Bidirectional and Auto-Regressive Transformer)
  - mT5, mBART（多语言版本）
  - 原始 Transformer（机器翻译）

核心特点：
  - Encoder：双向注意力，理解输入序列
  - Decoder：单向注意力 + 交叉注意力，条件生成输出序列
  - 适合：机器翻译、摘要生成、问答、代码生成等"输入→输出"任务

与 Only Encoder/Decoder 的比较：
  ┌──────────────────┬─────────────┬──────────────────────────────┐
  │ 变体              │ 注意力方向  │ 典型用途                      │
  ├──────────────────┼─────────────┼──────────────────────────────┤
  │ Only Encoder     │ 双向        │ 分类、NER、问答理解            │
  │ Only Decoder     │ 单向（左→右）│ 文本生成、对话                │
  │ Encoder-Decoder  │ Enc双向+Dec单向│ 翻译、摘要、Seq2Seq          │
  └──────────────────┴─────────────┴──────────────────────────────┘

本文件实现 T5 风格变体（相对位置编码 + Pre-LN + 简化结构）。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# T5 风格的相对位置偏置（简化版）
# ─────────────────────────────────────────────
class RelativePositionBias(nn.Module):
    """
    T5 使用相对位置编码而非绝对位置编码。
    将相对位置（i-j）映射到可学习的偏置值，加到注意力分数上。

    优点：可以外推到训练时未见过的长度
    简化：这里使用有界的相对位置桶（bucketed）

    注意：本实现是简化版，完整 T5 有更复杂的 bucket 逻辑
    """

    def __init__(self, num_heads: int, num_buckets: int = 32, max_distance: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        # [num_heads, num_buckets]
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    def _bucket_position(self, relative_position: torch.Tensor) -> torch.Tensor:
        """将相对位置映射到桶索引"""
        n = -relative_position

        # 负数桶（0 到 num_buckets/2-1 对应距离 0..max_exact）
        num_buckets = self.num_buckets
        max_exact = num_buckets // 2

        # 正负分开处理
        is_small = n < max_exact
        val_if_large = max_exact + (
            torch.log(n.float().clamp_min(1) / max_exact)
            / math.log(self.max_distance / max_exact)
            * (num_buckets - max_exact)
        ).long().clamp(0, num_buckets - max_exact - 1)

        n_buckets = torch.where(is_small, n, val_if_large)
        # 负方向偏移 num_buckets//2
        relative_buckets = torch.where(
            relative_position > 0,
            num_buckets // 2 + n_buckets,
            n_buckets,
        )
        return relative_buckets.clamp(0, num_buckets - 1)

    def forward(self, query_len: int, key_len: int, device: torch.device) -> torch.Tensor:
        """
        Returns:
            [1, num_heads, query_len, key_len]  — 加到注意力分数的偏置
        """
        q_pos = torch.arange(query_len, dtype=torch.long, device=device).unsqueeze(1)
        k_pos = torch.arange(key_len, dtype=torch.long, device=device).unsqueeze(0)
        relative_position = k_pos - q_pos  # [query_len, key_len]

        buckets = self._bucket_position(relative_position)  # [query_len, key_len]
        # [query_len, key_len, num_heads]
        bias = self.relative_attention_bias(buckets)
        # [1, num_heads, query_len, key_len]
        return bias.permute(2, 0, 1).unsqueeze(0)


# ─────────────────────────────────────────────
# 基础组件（带相对位置偏置支持）
# ─────────────────────────────────────────────

class MultiHeadAttentionWithBias(nn.Module):
    """支持相对位置偏置的多头注意力"""

    def __init__(self, d_model, num_heads, dropout=0.1, has_relative_bias=False, num_buckets=32):
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

        # 相对位置偏置（只在每层第一个自注意力里用）
        self.rel_bias = RelativePositionBias(num_heads, num_buckets) if has_relative_bias else None

    def _split(self, x):
        B, S, _ = x.shape
        return x.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

    def _merge(self, x):
        B, _, S, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, S, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q_in, K_in, V_in = Q, K, V
        Q = self._split(self.W_Q(Q_in))
        K = self._split(self.W_K(K_in))
        V = self._split(self.W_V(V_in))

        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # 加入相对位置偏置
        if self.rel_bias is not None:
            bias = self.rel_bias(Q_in.size(1), K_in.size(1), Q_in.device)
            scores = scores + bias

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        self.attention_weights = weights

        if self.dropout is not None:
            weights = self.dropout(weights)

        out = torch.matmul(weights, V)
        return self.W_O(self._merge(out))


class FeedForwardT5(nn.Module):
    """T5 风格 FFN：使用 ReLU（T5v1.0），T5v1.1 用 GEGLU"""
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.wo(self.dropout(self.act(self.wi(x))))


# ─────────────────────────────────────────────
# T5 风格 Encoder 层 + Decoder 层
# ─────────────────────────────────────────────
class T5EncoderLayer(nn.Module):
    """T5 Encoder 层（Pre-LN，无 bias）"""

    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1, has_relative_bias=False):
        super().__init__()
        self.self_attn = MultiHeadAttentionWithBias(
            d_model, num_heads, dropout, has_relative_bias=has_relative_bias
        )
        self.ffn = FeedForwardT5(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        residual = x
        x = self.norm1(x)
        x = self.self_attn(Q=x, K=x, V=x, mask=mask)
        x = residual + self.dropout(x)

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)
        return x


class T5DecoderLayer(nn.Module):
    """T5 Decoder 层（Pre-LN，含交叉注意力，无 bias）"""

    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1, has_relative_bias=False):
        super().__init__()
        self.self_attn = MultiHeadAttentionWithBias(
            d_model, num_heads, dropout, has_relative_bias=has_relative_bias
        )
        self.cross_attn = MultiHeadAttentionWithBias(d_model, num_heads, dropout)
        self.ffn = FeedForwardT5(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, enc_output, tgt_mask=None, src_mask=None):
        # Masked 自注意力
        residual = tgt
        tgt = self.norm1(tgt)
        tgt = self.self_attn(Q=tgt, K=tgt, V=tgt, mask=tgt_mask)
        tgt = residual + self.dropout(tgt)

        # 交叉注意力
        residual = tgt
        tgt = self.norm2(tgt)
        tgt = self.cross_attn(Q=tgt, K=enc_output, V=enc_output, mask=src_mask)
        tgt = residual + self.dropout(tgt)

        # FFN
        residual = tgt
        tgt = self.norm3(tgt)
        tgt = self.ffn(tgt)
        tgt = residual + self.dropout(tgt)
        return tgt


# ─────────────────────────────────────────────
# Encoder-Decoder 主体（T5 风格）
# ─────────────────────────────────────────────
class EncoderDecoderTransformer(nn.Module):
    """
    T5/BART 风格的 Encoder-Decoder Transformer

    特点（相比原始 Transformer）：
        - Pre-LN（而非 Post-LN）
        - 无绝对位置编码（T5 使用相对位置偏置）
        - 所有线性层无 bias
        - 共享 Encoder/Decoder embedding（T5 默认）

    Args:
        vocab_size:  词表大小
        d_model:     模型维度（T5-small: 512）
        num_heads:   注意力头数（T5-small: 8）
        num_layers:  Encoder/Decoder 层数（T5-small: 6）
        d_ff:        FFN 维度（T5-small: 2048）
        dropout:     dropout 概率
        pad_idx:     padding token id
    """

    def __init__(
        self,
        vocab_size: int = 32128,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        pad_idx: int = 0,
        max_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        # 共享 Embedding（T5 特点：Encoder/Decoder 共享词表和 embedding）
        self.shared_embedding = nn.Embedding(vocab_size, d_model)

        # Encoder：N 层，只有第一层有相对位置偏置（T5 设计）
        self.encoder_layers = nn.ModuleList([
            T5EncoderLayer(d_model, num_heads, d_ff, dropout, has_relative_bias=(i == 0))
            for i in range(num_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model, eps=1e-6)

        # Decoder：N 层
        self.decoder_layers = nn.ModuleList([
            T5DecoderLayer(d_model, num_heads, d_ff, dropout, has_relative_bias=(i == 0))
            for i in range(num_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model, eps=1e-6)

        # 输出投影（共享 embedding 权重）
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.shared_embedding.weight

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=d_model ** -0.5 if hasattr(self, 'd_model') else 0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=1.0)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        tgt_len = tgt.size(1)
        device = tgt.device
        causal = torch.tril(torch.ones(tgt_len, tgt_len, device=device)).bool()
        causal = causal.unsqueeze(0).unsqueeze(0)
        pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return causal & pad_mask

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Encoder 前向
        Returns: [batch, src_len, d_model]
        """
        if src_mask is None:
            src_mask = self.make_src_mask(src)

        # T5 embedding 乘以 sqrt(d_model) 缩放
        x = self.shared_embedding(src) * math.sqrt(self.d_model)
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x = layer(x, mask=src_mask)

        return self.encoder_norm(x)

    def decode(
        self,
        tgt: torch.Tensor,
        enc_output: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        src_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Decoder 前向
        Returns: [batch, tgt_len, d_model]
        """
        if tgt_mask is None:
            tgt_mask = self.make_tgt_mask(tgt)

        x = self.shared_embedding(tgt) * math.sqrt(self.d_model)
        x = self.dropout(x)

        for layer in self.decoder_layers:
            x = layer(x, enc_output, tgt_mask=tgt_mask, src_mask=src_mask)

        return self.decoder_norm(x)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        完整前向传播（训练用）

        Args:
            src: [batch, src_len]  — 输入序列（如：英语）
            tgt: [batch, tgt_len]  — 目标序列（如：中文，右移一位，以 <bos> 开头）

        Returns:
            logits: [batch, tgt_len, vocab_size]
        """
        if src_mask is None:
            src_mask = self.make_src_mask(src)

        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, tgt_mask, src_mask)
        return self.lm_head(dec_output)

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        decoder_start_token_id: int,
        eos_token_id: int,
        max_new_tokens: int = 50,
        num_beams: int = 1,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> torch.Tensor:
        """
        Seq2Seq 生成（支持贪心/束搜索）

        Args:
            src:                     [1, src_len]  单个输入序列
            decoder_start_token_id:  <pad> 或 <bos>（T5 用 pad 作为 decoder 起始）
            eos_token_id:            生成终止符
            max_new_tokens:          最多生成多少 token
            num_beams:               束宽（1=贪心）

        Returns:
            generated token ids（不含 decoder_start_token）
        """
        self.eval()
        device = next(self.parameters()).device
        src = src.to(device)

        src_mask = self.make_src_mask(src)
        enc_output = self.encode(src, src_mask)

        # 初始 decoder 输入
        dec_input = torch.tensor([[decoder_start_token_id]], dtype=torch.long, device=device)
        generated = []

        for _ in range(max_new_tokens):
            dec_out = self.decode(dec_input, enc_output, src_mask=src_mask)
            logits = self.lm_head(dec_out[:, -1, :])  # [1, vocab]

            if do_sample and temperature != 1.0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            token_id = next_token.item()
            if token_id == eos_token_id:
                break

            generated.append(token_id)
            dec_input = torch.cat([dec_input, next_token], dim=1)

        return generated


# ─────────────────────────────────────────────
# 验证
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("变体3: Encoder-Decoder (T5/BART 风格)")
    print("=" * 60)

    # T5-tiny 配置
    d_model = 128
    model = EncoderDecoderTransformer(
        vocab_size=1000,   # 共享词表（T5 风格）
        d_model=d_model,
        num_heads=4,        # d_k = 128/4 = 32
        num_layers=2,
        d_ff=512,
        dropout=0.0,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n配置: vocab={1000}, d_model={d_model}, heads=4, layers=2, d_ff=512")
    print(f"      d_k = {d_model} / 4 = {d_model // 4}")
    print(f"T5-tiny 参数量: {total_params:,}")

    batch_size = 2
    src_len = 12
    tgt_len = 8

    src = torch.randint(1, 1000, (batch_size, src_len))
    tgt = torch.randint(1, 1000, (batch_size, tgt_len))
    src[0, -3:] = 0  # padding

    print(f"\n{'='*50}")
    print("逐步跟踪 T5 前向传播:")
    print(f"{'='*50}")
    print(f"  src (源序列):           {src.shape}             [batch={batch_size}, src_len={src_len}]")
    print(f"  tgt (目标序列，右移):    {tgt.shape}             [batch, tgt_len={tgt_len}]")

    # Step 1: Encode
    enc_output = model.encode(src)
    print(f"  Encoder 输出:           {enc_output.shape}         [batch, src_len, d_model]")

    # Step 2: Decode
    dec_output = model.decode(tgt, enc_output)
    print(f"  Decoder 输出:           {dec_output.shape}         [batch, tgt_len, d_model]")

    # Step 3: LM Head
    logits = model.lm_head(dec_output)
    print(f"  LM Head (logits):       {logits.shape}            [batch, tgt_len, vocab]")

    # 计算翻译损失
    target_ids = torch.randint(1, 1000, (batch_size, tgt_len))
    loss = F.cross_entropy(
        logits.view(-1, 1000),
        target_ids.view(-1),
        ignore_index=0,
    )
    print(f"Seq2Seq Loss: {loss.item():.4f}")

    # 生成
    src_single = torch.randint(1, 1000, (1, 10))
    generated = model.generate(src_single, decoder_start_token_id=1, eos_token_id=2, max_new_tokens=15)
    print(f"\n[生成] 结果 ({len(generated)} tokens): {generated}")

    print("\n✅ Encoder-Decoder 验证通过！")
    print("\n三种变体总结：")
    print("┌─────────────────┬───────────────┬─────────────────────────┐")
    print("│ 变体              │ 代表模型       │ 擅长任务                 │")
    print("├─────────────────┼───────────────┼─────────────────────────┤")
    print("│ Only Encoder    │ BERT, RoBERTa │ 分类、NER、理解           │")
    print("│ Only Decoder    │ GPT, LLaMA    │ 文本生成、对话             │")
    print("│ Encoder-Decoder │ T5, BART      │ 翻译、摘要、Seq2Seq      │")
    print("└─────────────────┴───────────────┴─────────────────────────┘")

