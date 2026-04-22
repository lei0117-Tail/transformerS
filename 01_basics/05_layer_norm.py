"""
Step 5: 层归一化 (Layer Normalization) + 残差连接 (Residual Connection)
=======================================================================
这是 Transformer 稳定训练的关键结构，通常组合成 "Add & Norm" 子层。

════════════════════════════════════════════════════════════════════
                    📖 核心概念总览
════════════════════════════════════════════════════════════════════

一个 Encoder Layer 内部有两个"子层"(Sublayer)，每个子层都被一组 Add & Norm 包裹：

  子层1 = Multi-Head Attention（多头注意力）→ "收集上下文信息"
  子层2 = Feed Forward Network    （前馈网络）   → "非线性加工信息"

Add & Norm 的两个组成部分：

  ┌─ Norm（层归一化 LayerNorm）：
  │   公式: y = (x - mean) / √(var + ε) × γ + β
  │   作用: 把数值拉到 均值≈0, 标准差≈1，稳定训练
  │   特点: 对每个样本的每个 token 独立归一化（不跨样本，区别于 BatchNorm）
  │
  └─ Add（残差连接 Residual Connection）：
      公式: output = x + F(x)  （原始输入 + 子层输出）
      作用:
        ① 梯度高速公路：反向传播时梯度通过 "+" 直接流向浅层，防止梯度消失
        ② 保证最低性能：子层没学到东西时，原始信息原样保留
        ③ 信息融合：输出 = 原始信息 + 新信息，不需要单独传递原始向量

════════════════════════════════════════════════════════════════════
                    🔀 两种组合方式：Post-LN vs Pre-LN
════════════════════════════════════════════════════════════════════

  Post-LN（原论文 2017）— 层后归一化:
    x → [Sublayer] → [Dropout] → [+ x] → [LayerNorm] → 输出
         子层处理              ↑ Add       ↑ 最后才Norm
    公式: output = LayerNorm(x + Sublayer(x))
    ⚠️ 问题: 梯度必须穿过 LayerNorm，深层时容易不稳定，需要 warm-up

  Pre-LN（现代主流 2018+）— 层前归一化（本项目默认使用）:
    x → [LayerNorm] → [Sublayer] → [Dropout] → [+ x] → 输出
         ↑ 先Norm       子层                ↑ 后Add（不再Norm）
    公式: output = x + Sublayer(LayerNorm(x))
    ✅ 优势: 梯度直接通过 Add 流过（恒有 +1 项），训练更稳定，无需 warm-up
    代表模型: GPT-2, GPT-3, BERT, LLaMA, 所有现代大模型

  核心区别就一个: Norm 在 Add 之前还是之后？

════════════════════════════════════════════════════════════════════
                    🗺️ Encoder Layer 完整路线图（Pre-LN 版本）
════════════════════════════════════════════════════════════════════

  输入: ① 原始向量 x  [batch, seq_len, d_model]
        来自 Embedding 层 或 上一层 Encoder Layer
        │
        ├── 第一套 Add & Norm（包裹 子层1: MHA）──┐
        │                                             │
        │  residual = x        # 保存 ① 原始向量      │
        │  x = Norm1(x)        # 先归一化            │
        │  x = MHA(Q=x,K=x,V=x)# 多头自注意力         │
        │  x = Dropout(x)                           │
        │  x = residual + x    # Add: ①+②注意力向量   │
        │  ↓ 中间结果（包含原始信息 + 注意力信息）     │
        ├── 第二套 Add & Norm（包裹 子层2: FFN）──┐  │
        │                                             │  │
        │  residual = x        # 保存中间结果         │  │
        │  x = Norm2(x)        # 先归一化            │  │
        │  x = FFN(x)           # 前馈网络            │  │
        │  x = Dropout(x)                           │  │
        │  x = residual + x    # Add: 中间+③FFN向量   │  │
        │                                             │  │
        ├─────────────────────────────────────────────┘  │
        ▼                                                │
  输出: [batch, seq_len, d_model]  shape 不变！          │
  （包含: 原始信息 + 注意力信息 + FFN 加工信息）          │
        │                                                │
        └────→ 送入下一层 Encoder Layer（或 Decoder）──────┘

  ⚠️ 重要: Add 的输出已经融合了原始信息，不需要单独传递！
     就像记账: 第1层 本金100+利息5=105, 第2层 本金105+利息8=113
     每一层都在前面积累的基础上叠加新信息。

════════════════════════════════════════════════════════════════════
                    📐 三种关键向量的含义
════════════════════════════════════════════════════════════════════

  ① 原始向量 = Token Embedding + Positional Encoding
     来自 Embedding 层，是 Encoder Layer 的输入

  ② 注意力向量 = Multi-Head Attention 的输出
     "收集了上下文信息"后的向量（每个 token 看了所有 token）

  ③ FFN 向量 = Feed Forward Network 的输出
     "经过非线性加工"后的向量（升维→激活→降维）

  经过 Add 后: 输出 = 之前的向量 + 当前子层的向量
  所以最终输出融合了全部三种信息！
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# Post-LN（原始 Transformer 论文方式）
# ─────────────────────────────────────────────
class PostLNSublayer(nn.Module):
    """
    Post-Layer Norm 子层包装器（原论文方式 / Post-LN / 层后归一化）

    数据流: x → [Sublayer] → [Dropout] → [+ x] → [LayerNorm] → 输出
    公式:   output = LayerNorm(x + Sublayer(x))

    这是 2017 年原始 Transformer 论文使用的方式。
    ⚠️ 缺点: 梯度必须穿过最后的 LayerNorm，深层时容易不稳定，需要 warm-up。

    Args:
        d_model:   模型维度
        sublayer:  子层模块（Multi-Head Attention 或 FFN）
        dropout:   应用在 sublayer 输出上的 dropout
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
    Pre-Layer Norm 子层包装器（现代主流方式 / Pre-LN / 层前归一化）

    数据流: x → [LayerNorm] → [Sublayer] → [Dropout] → [+ x] → 输出
    公式:   output = x + Sublayer(LayerNorm(x))

    ✅ 这是 2018 年后所有现代大模型采用的方式（GPT-2/BERT/LLaMA 等）。
    ✅ 优势: 梯度可以直接通过残差连接流过（恒有 +1 项），不被 LayerNorm 压缩。
    ✅ 训练非常稳定，即使 80 层以上的深度网络也不需要 warm-up。

    Args:
        d_model:   模型维度
        sublayer:  子层模块（Multi-Head Attention 或 FFN）
        dropout:   应用在 sublayer 输出上的 dropout
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

    batch_size = 2     # 批次大小
    seq_len = 10       # 序列长度
    d_model = 64       # 模型维度（特征维度）

    x = torch.randn(batch_size, seq_len, d_model)  # 输入：模拟上一层的输出

    # ── 1. 验证 PyTorch 内置 LayerNorm ──
    layer_norm = nn.LayerNorm(d_model)  # PyTorch 内置层归一化
    out = layer_norm(x)  # 对每个 token 的特征维度做归一化（均值≈0，标准差≈1）
    print(f"\n[LayerNorm] 输出 shape: {out.shape}")
    print(f"[LayerNorm] 输出均值（约0）: {out.mean().item():.6f}")
    print(f"[LayerNorm] 输出标准差（约1）: {out.std().item():.6f}")

    # ── 2. 验证手动实现的 LayerNorm 与 PyTorch 一致 ──
    manual_ln = ManualLayerNorm(d_model)  # 手动实现的 LayerNorm，用于理解原理
    # 把权重设成与 PyTorch 内置相同（都是 γ=1, β=0）
    manual_out = manual_ln(x)
    print(f"\n[ManualLayerNorm] 输出均值（约0）: {manual_out.mean().item():.6f}")
    print(f"[ManualLayerNorm] 与 PyTorch 结果是否一致: "
          f"{torch.allclose(out, manual_out, atol=1e-5)}")

    # ═══════════════════════════════════════════════════════════════
    #  3. 验证残差连接 — 核心代码实现解析
    # ═══════════════════════════════════════════════════════════════
    #
    #  残差连接的本质就一行代码: output = original + sublayer_output
    #  但它解决了深度网络的梯度消失问题！
    #
    #  Pre-LN 方式的完整流程（5 行代码）:
    #    residual = x              # 第1行: 保存原始向量（"拍照留底"）
    #    x = self.norm1(x)         # 第2行: 归一化（准备阶段，让子层输入稳定）
    #    x = self.self_attn(...)   # 第3行: 子层处理 ← 核心计算，产生新信息
    #    x = self.dropout(x)       # 第4行: 随机置零（防过拟合）
    #    x = residual + x          # 第5行: 🎯 残差！原始 + 子层输出
    #
    #  ⚠️ 重要: 加的是"子层的输出"，不是"归一化的值"！
    #     Norm 和 Dropout 都是辅助操作，真正产生新信息的是子层(MHA或FFN)。
    #     残差把子层的产出叠加到原始上: 最终 = 原始信息 + 新信息
    #
    #  Post-LN 方式更简洁（1 行搞定）:
    #    x = self.norm(x + self.dropout(sublayer(x)))  # Add 和 Norm 在一起
    #
    # ═══════════════════════════════════════════════════════════════

    linear = nn.Linear(d_model, d_model)  # 用一个简单的线性层作为子层来测试残差连接
    
    post_ln = PostLNSublayer(d_model=d_model, sublayer=linear, dropout=0.0)   # Post-LN: 先子层再归一化（原论文方式）
    
    pre_ln = PreLNSublayer(d_model=d_model, sublayer=linear, dropout=0.0)     # Pre-LN: 先归一化再子层（现代大模型方式）

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
    # 原理: output = x + 0 = x （原始信息完整保留）

    # ── 5. 理解 Add & Norm 的完整数据流 ──
    print("\n─── Add & Norm 完整总结 ───")
    print("  一个 Encoder Layer 有两组 Add & Norm:")
    print("    第1组 包裹: Multi-Head Attention（收集上下文信息）")
    print("    第2组 包裹: Feed Forward Network    （非线性加工信息）")
    print("  每组包含: Norm(稳定数值) → Sublayer(核心计算) → Add(融合原始+新信息)")
    print("  最终输出融合了三种向量: 原始 + 注意力 + FFN")
    print("  不需要单独传递原始向量 — Add 已经把它编码进输出了（像记账一样累积）")

    print("\n✅ Step 5 验证通过！")

