"""
变体 2: Only Decoder（GPT 风格）
==================================
只使用 Transformer 的 Decoder 部分（去掉交叉注意力），用于生成任务（生成式）。

代表模型：
  - GPT-1/2/3/4、ChatGPT、LLaMA、Mistral、Gemma、Qwen、DeepSeek...

核心特点：
  - 单向注意力（Causal/Autoregressive）：每个 token 只能看到之前的 token
  - 适合：文本生成、语言模型、对话系统、代码生成等"生成"任务
  - 不包含交叉注意力（没有独立的 Encoder）

与完整 Decoder 的区别：
  - 完整 Decoder（Seq2Seq）有 3 个子层（自注意力 + 交叉注意力 + FFN）
  - Only Decoder（GPT）只有 2 个子层（自注意力 + FFN，去掉交叉注意力）

训练范式：
  - Causal Language Modeling (CLM)：预测下一个 token
  - 给定 "The cat sat on the"，预测 "mat"

本文件实现：
  1. GPTDecoderLayer：去掉交叉注意力的 Decoder 层
  2. GPTLikeDecoder：Only Decoder 主体
  3. GPTForLM：语言模型（生成）
  4. 自回归文本生成（包含 temperature 采样、top-k/top-p 采样）
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
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),         # GPT-2 开始用 GELU
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# GPT 风格 Decoder 层（去掉交叉注意力）
# ─────────────────────────────────────────────
class GPTDecoderLayer(nn.Module):
    """
    GPT 风格的 Decoder 层（Only Decoder）

    与完整 DecoderLayer 的区别：
        ❌ 没有 Cross-Attention 子层（不需要 Encoder）
        ✅ 只有 Masked Self-Attention + FFN

    结构（Pre-LN，GPT-2 风格）：
        x → norm1 → Masked Self-Attn → + residual
          → norm2 → FFN → + residual
    """

    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:    [batch, seq_len, d_model]
            mask: [1, 1, seq_len, seq_len]  — causal mask

        Returns:
            [batch, seq_len, d_model]
        """
        # 子层1：Masked 自注意力（Pre-LN）
        residual = x
        x = self.norm1(x)
        x = self.self_attn(Q=x, K=x, V=x, mask=mask)
        x = residual + self.dropout(x)

        # 子层2：FFN（Pre-LN）
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)

        return x

    @property
    def attention_weights(self):
        return self.self_attn.attention_weights


# ─────────────────────────────────────────────
# Only Decoder 主体
# ─────────────────────────────────────────────
class GPTLikeDecoder(nn.Module):
    """
    GPT 风格的 Only Decoder

    Args:
        vocab_size: 词表大小
        d_model:    模型维度（GPT-2 small: 768）
        num_heads:  注意力头数（GPT-2 small: 12）
        num_layers: 层数（GPT-2 small: 12）
        d_ff:       FFN 维度（GPT-2 small: 3072）
        max_len:    最大上下文长度（GPT-2: 1024）
        dropout:    dropout 概率

    Forward:
        input_ids: [batch, seq_len]
        past_len:  用于推理时 KV cache 的位置偏移（可选）

    Returns:
        [batch, seq_len, d_model]
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        d_ff: int = 3072,
        max_len: int = 1024,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # GPT 使用可学习的位置编码（与 BERT 相同）
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            GPTDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # GPT-2 在最后加一个 LayerNorm（Pre-LN 架构需要）
        self.final_norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """生成下三角 causal mask [1, 1, seq_len, seq_len]"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:    [batch, seq_len]
            position_ids: [batch, seq_len]（若 None，自动生成 0..seq_len-1）

        Returns:
            [batch, seq_len, d_model]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        assert seq_len <= self.max_len, (
            f"序列长度 {seq_len} 超过最大上下文长度 {self.max_len}"
        )

        # 位置 ids
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)  # [1, seq_len]

        # 嵌入
        x = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        x = self.emb_dropout(x)

        # Causal mask
        causal_mask = self.make_causal_mask(seq_len, device)

        # N 个 Decoder 层
        for layer in self.layers:
            x = layer(x, mask=causal_mask)

        x = self.final_norm(x)
        return x


# ─────────────────────────────────────────────
# 语言模型头
# ─────────────────────────────────────────────
class GPTForLM(nn.Module):
    """
    GPT 语言模型（用于文本生成）

    训练目标：预测下一个 token（Causal LM）
    """

    def __init__(self, gpt: GPTLikeDecoder):
        super().__init__()
        self.gpt = gpt
        # 输出投影，共享 token embedding 权重
        self.lm_head = nn.Linear(gpt.d_model, gpt.token_embedding.num_embeddings, bias=False)
        self.lm_head.weight = gpt.token_embedding.weight  # 权重共享

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len]
        Returns:
            logits:    [batch, seq_len, vocab_size]
        """
        hidden = self.gpt(input_ids)
        return self.lm_head(hidden)

    def compute_loss(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        计算 CLM 训练损失
        输入：[batch, seq_len]
        目标：input_ids 右移一位（预测下一个 token）

        Example：
            输入:  [the, cat, sat, on]
            目标:  [cat, sat, on, mat]  (右移)
        """
        # logits: [batch, seq_len, vocab_size]
        logits = self.forward(input_ids)

        # 目标：当前序列右移（用 input_ids 的 1.. 作目标，预测 ..seq_len-1）
        # 输入取前 seq_len-1 个，目标取后 seq_len-1 个
        shift_logits = logits[:, :-1, :].contiguous()       # [B, seq_len-1, vocab]
        shift_labels = input_ids[:, 1:].contiguous()        # [B, seq_len-1]

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=0,  # 忽略 padding
        )
        return loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        do_sample: bool = True,
        eos_token_id: int = None,
    ) -> torch.Tensor:
        """
        自回归文本生成

        Args:
            input_ids:      [batch, prompt_len]  — 提示词
            max_new_tokens: 最多生成多少个新 token
            temperature:    温度（越高越随机，越低越确定）
            top_k:          只从概率最高的 k 个词中采样（0=不限制）
            top_p:          Nucleus Sampling：只从累积概率 ≤ p 的词中采样
            do_sample:      True=采样，False=贪心
            eos_token_id:   遇到此 token 停止生成

        Returns:
            [batch, prompt_len + generated_len]
        """
        self.eval()
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)

        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # 截断到最大上下文长度
            cur_ids = generated[:, -self.gpt.max_len:]

            logits = self.forward(cur_ids)           # [B, seq, vocab]
            next_logits = logits[:, -1, :]            # [B, vocab] 只取最后位置

            # ── 温度缩放 ──
            if temperature != 1.0:
                next_logits = next_logits / temperature

            # ── Top-K 过滤 ──
            if top_k > 0:
                top_k = min(top_k, next_logits.size(-1))
                values, _ = torch.topk(next_logits, top_k)
                min_val = values[:, -1].unsqueeze(-1)
                next_logits = next_logits.masked_fill(next_logits < min_val, float("-inf"))

            # ── Top-P (Nucleus) 过滤 ──
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # 去掉累积概率超过 p 的 token（保留刚刚超过的那个）
                sorted_idx_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_idx_to_remove] = float("-inf")
                # 还原排序
                next_logits = torch.scatter(next_logits, 1, sorted_idx, sorted_logits)

            # ── 采样或贪心 ──
            if do_sample:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)  # [B, 1]

            generated = torch.cat([generated, next_token], dim=1)

            # 遇到 EOS 停止
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated


# ─────────────────────────────────────────────
# 验证
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("变体2: Only Decoder (GPT 风格)")
    print("=" * 60)

    # GPT-tiny 配置
    gpt = GPTLikeDecoder(
        vocab_size=1000,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=512,
        max_len=128,
        dropout=0.0,
    )
    lm = GPTForLM(gpt)
    total_params = sum(p.numel() for p in lm.parameters())
    print(f"\nGPT-tiny 参数量: {total_params:,}")

    batch_size = 2
    seq_len = 20

    input_ids = torch.randint(1, 1000, (batch_size, seq_len))

    # ── 前向传播 ──
    logits = lm(input_ids)
    print(f"\n[前向] input_ids shape: {input_ids.shape}")
    print(f"[前向] logits shape: {logits.shape}")  # [2, 20, 1000]

    # ── 计算训练损失 ──
    loss = lm.compute_loss(input_ids)
    print(f"[训练] CLM Loss: {loss.item():.4f}")
    print(f"       初始 perplexity ≈ {loss.exp().item():.2f}（未训练，约等于 vocab_size）")

    # ── 文本生成（贪心）──
    prompt = torch.randint(1, 1000, (1, 5))
    print(f"\n[生成] 提示词 ({prompt.shape[1]} tokens): {prompt[0].tolist()}")

    # 贪心
    greedy_out = lm.generate(prompt, max_new_tokens=10, do_sample=False)
    print(f"[贪心] 生成结果 ({greedy_out.shape[1]} tokens): {greedy_out[0].tolist()}")

    # 采样（temperature=0.8, top-k=50）
    sampled_out = lm.generate(
        prompt, max_new_tokens=10, temperature=0.8, top_k=50, do_sample=True
    )
    print(f"[采样] 生成结果 ({sampled_out.shape[1]} tokens): {sampled_out[0].tolist()}")

    print("\n✅ Only Decoder 验证通过！")
    print("\n核心总结：")
    print("  - Only Decoder = 单向注意力 = 擅长生成")
    print("  - 训练目标：预测下一个 token（CLM）")
    print("  - 推理：自回归，一次生成一个 token")
    print("  - GPT 用 GELU 激活，可学习位置编码，Pre-LN（GPT-2+）")
    print("  - 多种采样策略：贪心、温度采样、top-k、top-p (nucleus)")

