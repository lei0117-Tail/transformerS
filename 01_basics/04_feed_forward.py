"""
Step 4: 位置前馈网络 (Position-wise Feed-Forward Network, FFN)
================================================================
每个 Encoder/Decoder 层都包含一个 FFN，对序列中每个位置独立地做两层全连接变换。

核心公式：
  FFN(x) = ReLU(x·W_1 + b_1)·W_2 + b_2

特点：
  - "Position-wise"：每个位置独立、共享参数（相当于对 seq 维度应用同一个 MLP）
  - 内层维度 d_ff 通常是 d_model 的 4 倍（原论文：512→2048）
  - 先升维（d_model→d_ff）再降维（d_ff→d_model），形成"瓶颈-扩张"结构
  - ReLU 引入非线性，让模型能学习更复杂的特征变换

激活函数对比（现代变体）：
  - 原始论文: ReLU
  - GPT-2/3:  GELU（更平滑，梯度更稳定）
  - LLaMA:    SwiGLU（引入门控机制，效果更好）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 标准 FFN（原论文版本）
# ─────────────────────────────────────────────
class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network

    结构：Linear → Activation → Dropout → Linear → Dropout
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int = None,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        """
        Args:
            d_model:    输入/输出维度
            d_ff:       中间层维度（默认 4 × d_model）
            dropout:    dropout 概率
            activation: 激活函数，支持 "relu" | "gelu"
        """
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model  # 原论文默认值

        self.linear1 = nn.Linear(d_model, d_ff)   # 升维
        self.linear2 = nn.Linear(d_ff, d_model)   # 降维
        self.dropout = nn.Dropout(dropout)

        # 激活函数
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"不支持的激活函数: {activation}，请选择 'relu' 或 'gelu'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]

        注意：虽然输入是三维的，但 nn.Linear 会对最后一维做变换，
        相当于对每个位置独立地应用同一个 MLP，这就是"position-wise"的含义。
        """
        # x: [B, S, d_model]
        # linear1: [B, S, d_model] → [B, S, d_ff]
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)

        # linear2: [B, S, d_ff] → [B, S, d_model]
        x = self.linear2(x)
        x = self.dropout(x)

        return x


# ─────────────────────────────────────────────
# 扩展：带 SwiGLU 的 FFN（LLaMA 风格）
# ─────────────────────────────────────────────
class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU FFN（Llama/PaLM 等现代 LLM 使用）

    公式：FFN_SwiGLU(x) = (Swish(x·W_1) ⊙ (x·W_2)) · W_3
    其中 Swish(x) = x·sigmoid(x)，也叫 SiLU

    核心思想：用"门控"机制过滤信息，让网络更有选择性地保留特征。
    维度设计：通常将 d_ff 设为 2/3 × 4 × d_model，保持总参数量不变。
    """

    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.0):
        super().__init__()
        if d_ff is None:
            # 保持参数量和标准 FFN 近似：2/3 × 4 × d_model ≈ 2.67 × d_model
            d_ff = int(2 / 3 * 4 * d_model)

        # 两个并行的升维层（一个用于激活，一个用于门控）
        self.w1 = nn.Linear(d_model, d_ff, bias=False)   # 主路
        self.w2 = nn.Linear(d_model, d_ff, bias=False)   # 门控路
        self.w3 = nn.Linear(d_ff, d_model, bias=False)   # 降维
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU(W1·x) ⊙ (W2·x)：门控激活
        gate = F.silu(self.w1(x))   # [B, S, d_ff]
        up = self.w2(x)              # [B, S, d_ff]
        x = gate * up                # element-wise 乘法（门控）
        x = self.dropout(x)
        x = self.w3(x)               # [B, S, d_model]
        return x


# ─────────────────────────────────────────────
# 单元测试 / 快速验证
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Step 4: Position-wise Feed-Forward Network")
    print("=" * 60)

    batch_size = 2
    seq_len = 10
    d_model = 128
    d_ff = 512  # 4 × d_model

    x = torch.randn(batch_size, seq_len, d_model)
    print(f"\n输入 shape: {x.shape}")

    # ── 标准 FFN (ReLU) ──
    ffn_relu = FeedForward(d_model=d_model, d_ff=d_ff, activation="relu")
    out_relu = ffn_relu(x)
    print(f"\n[ReLU FFN] 输出 shape: {out_relu.shape}")  # [2, 10, 128]

    # ── 标准 FFN (GELU) ──
    ffn_gelu = FeedForward(d_model=d_model, d_ff=d_ff, activation="gelu")
    out_gelu = ffn_gelu(x)
    print(f"[GELU FFN] 输出 shape: {out_gelu.shape}")    # [2, 10, 128]

    # ── SwiGLU FFN ──
    ffn_swiglu = SwiGLUFeedForward(d_model=d_model)
    out_swiglu = ffn_swiglu(x)
    print(f"[SwiGLU FFN] 输出 shape: {out_swiglu.shape}")  # [2, 10, 128]

    # ── 参数量对比 ──
    print(f"\n参数量对比 (d_model={d_model}, d_ff={d_ff}):")
    print(f"  标准 FFN:    {sum(p.numel() for p in ffn_relu.parameters()):,}")
    # 2 × (d_model × d_ff) = 2 × 128×512 = 131072
    print(f"  SwiGLU FFN:  {sum(p.numel() for p in ffn_swiglu.parameters()):,}")
    # 3 × (d_model × d_ff')，d_ff' ≈ 2/3 × 512 ≈ 341

    # ── 验证"position-wise"：对每个位置独立变换 ──
    ffn_test = FeedForward(d_model=d_model, dropout=0.0)
    # 只改变位置 0 的输入
    x_modified = x.clone()
    x_modified[:, 0, :] += 1.0
    out_orig = ffn_test(x)
    out_mod = ffn_test(x_modified)
    # 位置 1 的输出不受影响
    print(f"\n[Position-wise 验证] 位置1输出不受位置0影响: "
          f"{torch.allclose(out_orig[:, 1, :], out_mod[:, 1, :])}")

    print("\n✅ Step 4 验证通过！")

