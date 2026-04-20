# 01_basics — Transformer 基础构建块

> 本目录是整个课程的地基，按照「从零搭砖」的顺序，逐一实现 Transformer 的六个核心模块。
> 每个文件都可以 **独立运行**，文件末尾的 `if __name__ == "__main__"` 块会打印验证结果。

---

## 文件说明

### `01_embedding.py` — 词嵌入 + 位置编码

Transformer 没有 RNN 的时序结构，必须显式地告诉模型每个 token 在序列中的位置。

| 类 | 作用 |
|----|------|
| `TokenEmbedding` | 把整数 token id 映射到 d_model 维连续向量（查找表） |
| `SinusoidalPositionalEncoding` | 用 sin/cos 函数注入绝对位置信息，**无需训练** |
| `LearnablePositionalEncoding` | 可学习的位置嵌入，BERT/GPT 风格 |
| `TransformerEmbedding` | 将 Token Embedding 与 Positional Encoding 相加，作为模型输入 |

**核心公式**：
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

---

### `02_attention.py` — 缩放点积注意力

Transformer 的核心计算单元。本文件实现最基础的单头注意力和两种 Mask。

| 函数/类 | 作用 |
|---------|------|
| `scaled_dot_product_attention` | 实现 `softmax(QKᵀ/√d_k)·V`，支持 mask 和 dropout |
| `make_padding_mask` | 生成填充掩码，屏蔽 `<pad>` 位置 |
| `make_causal_mask` | 生成下三角因果掩码，防止 Decoder 看到未来信息 |

**为什么要除以 √d_k**：防止向量维度大时点积值过大，导致 softmax 进入梯度饱和区。

---

### `03_multi_head_attention.py` — 多头注意力

将单头注意力扩展为多头，让模型同时从不同子空间学习不同类型的依赖关系。

| 类 | 作用 |
|----|------|
| `MultiHeadAttention` | 核心多头注意力模块，包含 W_Q/W_K/W_V/W_O 四个线性变换 |

**三种使用场景**：
- **Encoder 自注意力**：`Q=K=V=x`，双向，无 Causal Mask
- **Decoder 自注意力**：`Q=K=V=x`，单向，带 Causal Mask
- **Decoder 交叉注意力**：`Q=decoder_out, K=V=encoder_out`

> 本文件内联了 `scaled_dot_product_attention` 和 `create_causal_mask`，无需依赖 `02_attention.py`，可独立运行。

---

### `04_feed_forward.py` — 位置前馈网络（FFN）

Encoder/Decoder 层的第二个子层。对序列中的**每个位置独立**做非线性变换。

| 类 | 激活函数 | 说明 |
|----|---------|------|
| `FeedForward` | ReLU 或 GELU | 标准 FFN，`d_model → d_ff → d_model` |
| `SwiGLUFeedForward` | SwiGLU | LLaMA 系列使用，门控机制，性能更好 |

**为什么需要 FFN**：注意力只做线性加权聚合，FFN 负责引入非线性，增强模型表达能力。

---

### `05_layer_norm.py` — 层归一化 + 残差连接

让深层 Transformer 训练稳定的两个关键技术。

| 类 | 作用 |
|----|------|
| `LayerNorm` | 对每个 token 的特征维度做归一化（区别于 BatchNorm） |
| `ManualLayerNorm` | 手动实现 LayerNorm，帮助理解原理 |
| `PostLNResidual` | Post-LN：`LayerNorm(x + Sublayer(x))`，原版 Transformer |
| `PreLNResidual` | Pre-LN：`x + Sublayer(LayerNorm(x))`，GPT-2 以后的标准 |

**Pre-LN vs Post-LN**：Pre-LN 训练更稳定（梯度不容易消失），是现代大模型的主流选择。

---

### `06_encoder_layer.py` — 完整 Encoder 层

将前5步的所有模块组合成一个完整的 Transformer Encoder Layer。

| 类 | 作用 |
|----|------|
| `EncoderLayer` | 完整 Encoder 层：MHA → Add&Norm → FFN → Add&Norm |

**数据流**：
```
输入 x
  → [多头自注意力] → 残差+归一化
  → [前馈网络]     → 残差+归一化
→ 输出（形状不变，仍为 [batch, seq_len, d_model]）
```

---

### `__init__.py` — 包初始化

将 `01_basics` 作为 Python 包，方便其他模块导入。

---

## 运行方式

```bash
# 逐步运行，观察每一步的输出
python 01_basics/01_embedding.py
python 01_basics/02_attention.py
python 01_basics/03_multi_head_attention.py
python 01_basics/04_feed_forward.py
python 01_basics/05_layer_norm.py
python 01_basics/06_encoder_layer.py

# 或通过主入口统一运行
python run_all.py --step 1
```

---

## 学习建议

1. **按顺序学习**：01 → 02 → 03 → 04 → 05 → 06，每个文件的注释都包含完整的理论说明
2. **动手修改**：改变 `d_model`、`num_heads`、`seq_len` 等参数，观察 shape 变化
3. **理解 Mask**：Mask 是 Transformer 最容易出错的地方，重点理解 `02_attention.py` 中的两种 Mask

