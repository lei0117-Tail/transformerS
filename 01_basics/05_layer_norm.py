"""
Step 5: 层归一化 (Layer Normalization) + 残差连接 (Residual Connection)
=======================================================================
这是 Transformer 稳定训练的关键结构，通常组合成 "Add & Norm" 子层。

层归一化 (LayerNorm)：
  - 对每个样本的特征维度做归一化（而非 BatchNorm 跨样本归一化）
  - 公式：y = (x - mean) / std × γ + β
  - 好处：不依赖 batch size，适合变长序列

残差连接 (Residual Connection)：
  - 思想来自 ResNet：output = x + F(x)
  - 缓解梯度消失/爆炸，让梯度更容易回流
  - 让深层网络的学习目标变为"残差"，更容易优化

两种组合方式：
  1. Post-LN（原论文）: output = LayerNorm(x + Sublayer(x))
  2. Pre-LN（现代常用）: output = x + Sublayer(LayerNorm(x))
     Pre-LN 训练更稳定，不需要 warm-up，大模型常用此方式
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# Post-LN（原始 Transformer 论文方式）
# ─────────────────────────────────────────────
class PostLNSublayer(nn.Module):
    """
    Post-Layer Norm 子层包装器
    output = LayerNorm(x + Sublayer(x))

    原论文使用此方式，但训练时通常需要 warm-up 调度。
    """

    def __init__(self, d_model: int, sublayer: nn.Module, dropout: float = 0.1):
        """
        Args:
            d_model:   模型维度
            sublayer:  子层模块（Multi-Head Attention 或 FFN）
            dropout:   应用在 sublayer 输出上的 dropout
        """
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.sublayer = sublayer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            *args, **kwargs: 传给 sublayer 的额外参数（如 mask）
        """
        # 先经过子层，再残差连接，最后归一化
        sublayer_out = self.sublayer(x, *args, **kwargs)
        return self.norm(x + self.dropout(sublayer_out))


# ─────────────────────────────────────────────
# Pre-LN（现代大模型常用方式）
# ─────────────────────────────────────────────
class PreLNSublayer(nn.Module):
    """
    Pre-Layer Norm 子层包装器
    output = x + Sublayer(LayerNorm(x))

    训练更稳定，不需要 warm-up，GPT-2、GPT-3、LLaMA 等均使用此方式。
    """

    def __init__(self, d_model: int, sublayer: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.sublayer = sublayer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """先归一化，再经过子层，最后残差连接"""
        normed = self.norm(x)
        sublayer_out = self.sublayer(normed, *args, **kwargs)
        return x + self.dropout(sublayer_out)


# ─────────────────────────────────────────────
# 手动实现 LayerNorm，理解原理
# ─────────────────────────────────────────────
class ManualLayerNorm(nn.Module):
    """
    手动实现 LayerNorm，用于理解内部机制
    公式：y = (x - μ) / √(σ² + ε) × γ + β

    其中：
        μ = mean(x)，对最后一维（特征维）求均值
        σ² = var(x)，方差
        ε = 防止除零的小常数
        γ, β = 可学习的缩放和偏移参数（形状 [d_model]）
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 可学习参数
        self.gamma = nn.Parameter(torch.ones(d_model))   # 初始化为 1
        self.beta = nn.Parameter(torch.zeros(d_model))   # 初始化为 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., d_model]  最后一维是特征维
        """
        mean = x.mean(dim=-1, keepdim=True)                    # 均值
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)    # 方差
        x_norm = (x - mean) / torch.sqrt(var + self.eps)      # 归一化
        return self.gamma * x_norm + self.beta                 # 缩放 + 偏移


# ─────────────────────────────────────────────
# 单元测试 / 快速验证
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Step 5: Layer Normalization + Residual Connection")
    print("=" * 60)

    batch_size = 2
    seq_len = 10
    d_model = 64

    x = torch.randn(batch_size, seq_len, d_model)

    # ── 1. 验证 PyTorch 内置 LayerNorm ──
    layer_norm = nn.LayerNorm(d_model)
    out = layer_norm(x)
    print(f"\n[LayerNorm] 输出 shape: {out.shape}")
    print(f"[LayerNorm] 输出均值（约0）: {out.mean().item():.6f}")
    print(f"[LayerNorm] 输出标准差（约1）: {out.std().item():.6f}")

    # ── 2. 验证手动实现的 LayerNorm 与 PyTorch 一致 ──
    manual_ln = ManualLayerNorm(d_model)
    # 把权重设成与 PyTorch 内置相同（都是 γ=1, β=0）
    manual_out = manual_ln(x)
    print(f"\n[ManualLayerNorm] 输出均值（约0）: {manual_out.mean().item():.6f}")
    print(f"[ManualLayerNorm] 与 PyTorch 结果是否一致: "
          f"{torch.allclose(out, manual_out, atol=1e-5)}")

    # ── 3. 验证残差连接 ──
    # 使用一个简单的线性层作为 sublayer
    linear = nn.Linear(d_model, d_model)
    post_ln = PostLNSublayer(d_model=d_model, sublayer=linear, dropout=0.0)
    pre_ln = PreLNSublayer(d_model=d_model, sublayer=linear, dropout=0.0)

    post_out = post_ln(x)
    pre_out = pre_ln(x)
    print(f"\n[Post-LN] 输出 shape: {post_out.shape}")
    print(f"[Pre-LN]  输出 shape: {pre_out.shape}")

    # ── 4. 直观理解残差连接的作用 ──
    # 如果子层输出全为 0（什么都没学），残差连接保证 x 原样通过
    class ZeroLayer(nn.Module):
        def forward(self, x):
            return torch.zeros_like(x)

    zero_sublayer = ZeroLayer()
    residual_test = PreLNSublayer(d_model=d_model, sublayer=zero_sublayer, dropout=0.0)
    residual_out = residual_test(x)
    print(f"\n[残差验证] 子层输出全0时，残差连接保留原始输入:")
    print(f"  原始输入 x[0,0,:4]:    {x[0, 0, :4].detach().numpy().round(3)}")
    print(f"  残差输出 out[0,0,:4]:  {residual_out[0, 0, :4].detach().numpy().round(3)}")
    print(f"  两者相等: {torch.allclose(x, residual_out)}")

    print("\n✅ Step 5 验证通过！")

