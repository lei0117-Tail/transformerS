"""
Step 3: 多头注意力 (Multi-Head Attention)
==========================================
单头注意力只能学到一种"关注模式"，多头注意力让模型同时从多个子空间捕获不同类型的依赖关系。

核心思想：
  1. 把 Q/K/V 分别用 h 组不同的线性变换投影到低维子空间
  2. 在每个子空间独立做 Scaled Dot-Product Attention
  3. 把 h 个头的输出拼接，再经过一个线性变换

公式：
  MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O
  head_i = Attention(Q·W_Q_i, K·W_K_i, V·W_V_i)

维度关系：
  d_model = h × d_k  （通常 d_k = d_v = d_model / h）

三种使用场景（后续会用到）：
  1. Encoder 自注意力:    Q=K=V=Encoder输入（每个位置关注序列所有位置）
  2. Decoder 自注意力:    Q=K=V=Decoder输入，+ causal_mask（只能看之前的位置）
  3. Encoder-Decoder 交叉注意力: Q=Decoder输出, K=V=Encoder输出
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 辅助：缩放点积注意力（Step 2 核心逻辑内联）
# ─────────────────────────────────────────────
def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None,
    dropout: nn.Dropout = None,
) -> tuple:
    """
    Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V

    Args:
        Q, K, V: [batch, heads, seq, d_k]
        mask:    [batch, 1, seq_q, seq_k]  0 表示屏蔽
    Returns:
        output:  [batch, heads, seq_q, d_k]
        weights: [batch, heads, seq_q, seq_k]
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)

    if dropout is not None:
        weights = dropout(weights)

    output = torch.matmul(weights, V)
    return output, weights


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """下三角 causal mask，shape: [1, 1, seq_len, seq_len]"""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return mask.unsqueeze(0).unsqueeze(0)


# ─────────────────────────────────────────────
# 多头注意力
# ─────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    """
    多头注意力机制

    参数：
        d_model:   模型维度（embedding 维度）
        num_heads: 注意力头数
        dropout:   dropout 概率

    关键实现技巧：
        不用 h 个独立的线性层，而是用一个大矩阵统一投影，
        然后通过 reshape + transpose 切分成多个头，效率更高。
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model({d_model}) 必须能被 num_heads({num_heads}) 整除"
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # 四个线性变换：W_Q, W_K, W_V, W_O
        # 注意：W_Q/W_K/W_V 的输出维度是 d_model，
        # 之后通过 reshape 切分成 num_heads 个 d_k 维的头
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None  # 用于可视化，保存最近一次的注意力权重

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将最后一维切分成 (num_heads, d_k)，并转置为 (batch, num_heads, seq, d_k)

        Args:
            x: [batch, seq_len, d_model]
        Returns:
            [batch, num_heads, seq_len, d_k]
        """
        batch_size, seq_len, _ = x.shape
        # reshape: [batch, seq, d_model] → [batch, seq, num_heads, d_k]
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        # transpose: [batch, seq, num_heads, d_k] → [batch, num_heads, seq, d_k]
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        split_heads 的逆操作：把多个头的输出拼接回来

        Args:
            x: [batch, num_heads, seq_len, d_k]
        Returns:
            [batch, seq_len, d_model]
        """
        batch_size, _, seq_len, _ = x.shape
        # 先 transpose 回 [batch, seq, num_heads, d_k]
        x = x.transpose(1, 2)
        # 再 view 合并 num_heads 和 d_k 维度
        # contiguous() 确保内存连续，view 才能正常工作
        return x.contiguous().view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            Q:    [batch, seq_q, d_model]
            K:    [batch, seq_k, d_model]
            V:    [batch, seq_k, d_model]
            mask: [batch, 1, seq_q, seq_k] 或 [batch, 1, 1, seq_k]

        Returns:
            output: [batch, seq_q, d_model]

        三种调用方式（Encoder/Decoder 不同场景）：
            自注意力:    mha(x, x, x, mask)
            交叉注意力:  mha(decoder_out, encoder_out, encoder_out, cross_mask)
        """
        # Step 1: 线性变换
        Q = self.W_Q(Q)  # [batch, seq_q, d_model]
        K = self.W_K(K)  # [batch, seq_k, d_model]
        V = self.W_V(V)  # [batch, seq_k, d_model]

        # Step 2: 切分多头
        Q = self.split_heads(Q)  # [batch, num_heads, seq_q, d_k]
        K = self.split_heads(K)  # [batch, num_heads, seq_k, d_k]
        V = self.split_heads(V)  # [batch, num_heads, seq_k, d_k]

        # Step 3: 缩放点积注意力（各头独立计算）
        # attn_out: [batch, num_heads, seq_q, d_k]
        attn_out, self.attention_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout
        )

        # Step 4: 拼接多头输出
        # [batch, num_heads, seq_q, d_k] → [batch, seq_q, d_model]
        attn_out = self.combine_heads(attn_out)

        # Step 5: 输出线性变换
        output = self.W_O(attn_out)  # [batch, seq_q, d_model]

        return output


# ─────────────────────────────────────────────
# 单元测试 / 快速验证
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Step 3: Multi-Head Attention")
    print("=" * 60)

    batch_size = 2     # 批次大小：同时处理多少个样本
    seq_len = 10       # 序列长度：每个样本包含多少个 token
    d_model = 128      # 模型维度：每个 token 的向量长度（原论文用 512）
    num_heads = 8      # 注意力头数：把 d_model 分成 8 个子空间分别做注意力

    # d_k = 128 / 8 = 16
    print(f"\n配置: d_model={d_model}, num_heads={num_heads}, d_k={d_model // num_heads}")

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    x = torch.randn(batch_size, seq_len, d_model)  # 输入：模拟 Embedding 层的输出
    print(f"\n输入 x shape: {x.shape}")  # [2, 10, 128] → batch=2, seq_len=10, d_model=128

    # ── 打印权重矩阵形状 ──
    print(f"\n{'='*50}")
    print("各权重矩阵的形状（理解 d_k 从哪来）:")
    print(f"{'='*50}")
    print(f"  W_Q 权重 shape: {mha.W_Q.weight.shape}   ← nn.Linear(128, 128) 的 [out, in]")
    print(f"  W_K 权重 shape: {mha.W_K.weight.shape}   ← 同样是 [128, 128]")
    print(f"  W_V 权重 shape: {mha.W_V.weight.shape}   ← 同样是 [128, 128]")
    print(f"  W_O 权重 shape: {mha.W_O.weight.shape}   ← 同样是 [128, 128]")
    print(f"\n  注意：W_Q/W_K/W_V 都是 [{d_model}, {d_model}] 方阵！")
    print(f"        行={d_model}（和输入向量匹配），列={d_model}（输出还是 d_model，不是 d_k）")

    # ── 手动拆解每一步的形状变化 ──
    print(f"\n{'='*50}")
    print("逐步跟踪形状变化:")
    print(f"{'='*50}")

    # Step 1: 线性投影
    Q_raw = mha.W_Q(x)  # 模拟 forward 中的第一步
    print(f"\n  Step1 - 线性投影 W_Q(x):")
    print(f"    输入 x:        {x.shape}       [batch, seq, d_model]")
    print(f"    W_Q.weight:   {mha.W_Q.weight.shape}  [d_model, d_model]")
    print(f"    输出 Q_raw:    {Q_raw.shape}      [batch, seq, d_model]  ← 维度没变！还是 {d_model}")
    print(f"    ↑ 此时还没有 d_k，输出依然是 d_model={d_model} 维")

    # Step 2: 切分多头
    Q_split = Q_raw.view(batch_size, seq_len, num_heads, d_model // num_heads)
    print(f"\n  Step2 - reshape 切分多头:")
    print(f"    Q_raw:         {Q_raw.shape}              [batch, seq, d_model]")
    print(f"    view 切分后:    {Q_split.shape}            [batch, seq, num_heads, d_k]")
    print(f"                   ↑ 这里才出现 d_k = {d_model // num_heads}")

    Q_heads = Q_split.transpose(1, 2)
    print(f"    transpose 后:   {Q_heads.shape}            [batch, heads, seq, d_k]")
    print(f"                   ↑ 这就是送进 attention 的最终 Q 形状")

    # Step 3: attention 内部
    scores = torch.matmul(Q_heads, Q_heads.transpose(-2, -1)) / math.sqrt(d_model // num_heads)
    print(f"\n  Step3 - Attention 内部 (Q·Kᵀ / √d_k):")
    print(f"    Q:             {Q_heads.shape}             [batch, heads, seq, d_k]")
    print(f"    Kᵀ:           {Q_heads.transpose(-2,-1).shape}  [batch, heads, d_k, seq]")
    print(f"    分数 scores:   {scores.shape}              [batch, heads, seq, seq]")
    print(f"    ↑ Q({d_model//num_heads}维) · Kᵀ({d_model//num_heads}维) → 标量分数矩阵")

    # Step 4: combine_heads (拼接回来)
    combined = Q_heads.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    print(f"\n  Step4 - combine_heads 拼接回去:")
    print(f"    多头输出:       {Q_heads.shape}             [batch, heads, seq, d_k]")
    print(f"    transpose:      {Q_heads.transpose(1,2).shape}   [batch, seq, heads, d_k]")
    print(f"    view 合并:      {combined.shape}             [batch, seq, d_model]  ← 拼回 {d_model} 维")

    # Step 5: W_O 输出投影
    final = mha.W_O(combined)
    print(f"\n  Step5 - W_O 输出投影:")
    print(f"    拼接后:         {combined.shape}             [batch, seq, d_model]")
    print(f"    W_O.weight:     {mha.W_O.weight.shape}       [d_model, d_model]")
    print(f"    最终输出:       {final.shape}               [batch, seq, d_model]")

    # ── 场景1: Self-Attention（Q=K=V=x）──
    out = mha(Q=x, K=x, V=x)  # 自注意力：Q=K=V=输入 x（每个位置关注所有位置，包括自己）
    print(f"\n[Self-Attention] 最终输出 shape: {out.shape}")           # [2, 10, 128] → 形状与输入相同
    print(f"[Self-Attention] 注意力权重 shape: {mha.attention_weights.shape}")  # [2, 8, 10, 10] → 8个头，每个头一个 10×10 的权重矩阵

    # ── 场景2: Cross-Attention（Decoder Q，Encoder K/V）──
    encoder_out = torch.randn(batch_size, 15, d_model)  # Encoder 输出（长度可以和 Decoder 不同）
    decoder_q = torch.randn(batch_size, seq_len, d_model)   # Decoder 的 Query（来自 Decoder 自身）
    cross_out = mha(Q=decoder_q, K=encoder_out, V=encoder_out)
    # 交叉注意力：Decoder 查询 Encoder 的信息（用于翻译等 Seq2Seq 任务）
    print(f"\n[Cross-Attention] Q.shape={decoder_q.shape}, K.shape={encoder_out.shape}")
    print(f"[Cross-Attention] 输出 shape: {cross_out.shape}")      # [2, 10, 128]

    # ── 场景3: 带 Causal Mask 的自注意力（Decoder）──
    causal_mask = create_causal_mask(seq_len)  # 因果 mask：防止看到未来位置（GPT 类模型必须用）
    causal_out = mha(Q=x, K=x, V=x, mask=causal_mask)
    # 带 causal mask 的自注意力：位置 i 只能看到 j<=i 的位置
    print(f"\n[Causal Self-Attention] 输出 shape: {causal_out.shape}")  # [2, 10, 128]

    # 验证参数数量
    total_params = sum(p.numel() for p in mha.parameters())
    print(f"\n多头注意力参数量: {total_params:,}")
    # 4 个线性层，每个 d_model × d_model，共 4 × 128×128 = 65536

    print("\n✅ Step 3 验证通过！")

