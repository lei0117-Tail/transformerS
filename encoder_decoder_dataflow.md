# Transformer Encoder-Decoder 完整数据流

> 本文档从宏观视角完整追踪数据在 Transformer 中的流动路径，
> 涵盖 Embedding → Encoder → Decoder → 输出的全过程。

---

## 🎯 一句话概括整个数据流

```
源句子 "I love you"
    ↓
[Encoder: 理解源句子的完整语义]
    ↓ (编码后的记忆)
[Decoder: 边看编码记忆，边逐词生成目标语言]
    ↓
目标句子 "我喜欢你"
```

---

## 🏗️ 完整架构总览图

```
═══════════════════════════════════════════════════════════════════════════
                         完整 Transformer (Encoder-Decoder)
═══════════════════════════════════════════════════════════════════════════

【输入】
  源句子 (Source): "I love you"          目标句子 (Target): "我喜欢你"
         ↓                                      ↓
  Token IDs: [12, 25, 38]               Token IDs: [55, 10, 100, 11]
                                                            ↑
                                              训练时偏移一位（见下方说明）
                                                         │
╔═════════════════════════════════════════╧════════════════════════════════╗
║                                                                         ║
║   ┌─────────────────────┐         ┌─────────────────────────┐           ║
║   │   INPUT EMBEDDING   │         │   OUTPUT EMBEDDING      │           ║
║   │                     │         │                         │           ║
║   │  TokenEmb + PosEnc  │         │  TokenEmb + PosEnc      │           ║
║   │  [1, 3, 512]        │         │  [1, 4, 512]            │           ║
║   └──────────┬──────────┘         └───────────┬─────────────┘           ║
║              ↓                                ↓                          ║
║   ╔═════════════════════════════╗                            ║
║   ║       ENCODER (×N 层)       ║                            ║
║   ║                             ║                            ║
║   ║  ┌───────────────────────┐  ║                            ║
║   ║  │ Layer 1:              │  ║     ╔═══════════════════╗  ║
║   ║  │  Self-Attn → Add&Norm │  ║     ║   DECODER (×M 层)  ║  ║
║   ║  │  FFN     → Add&Norm   │  ║     ║                   ║  ║
║   ║  └───────────────────────┘  ║     ║  ┌─────────────┐  ║  ║
║   ║  ┌───────────────────────┐  ║     ║  │ Layer 1:    │  ║  ║
║   ║  │ Layer 2:              │  ║     ║  │ MaskedSelf  │  ║  ║
║   ║  │  Self-Attn → Add&Norm │  ║     ║  │ Cross-Attn  │  ║  ║
║   ║  │  FFN     → Add&Norm   │  ║     ║  │ FFN→Add&Norm│  ║  ║
║   ║  └───────────────────────┘  ║     ║  └─────────────┘  ║  ║
║   ║  ...                        ║     ║  ...              ║  ║
║   ║  ┌───────────────────────┐  ║     ║  ┌─────────────┐  ║  ║
║   ║  │ Layer N:              │  ║     ║  │ Layer M:    │  ║  ║
║   ║  │  Self-Attn → Add&Norm │  ║     ║  │ MaskedSelf  │  ║  ║
║   ║  │  FFN     → Add&Norm   │  ║     ║  │ Cross-Attn  │  ║  ║
║   ║  └───────────┬───────────┘  ║     ║  │ FFN→Add&Norm│  ║  ║
║   ╚═════════════╪═══════════════╝     ║  └──────┬──────┘  ║  ║
║                  ↓                      ╚════════╧═════════╝  ║
║            Encoder Output                       ↓             ║
║            [1, 3, 512]                    Decoder Output          ║
║            (源句子的完整编码)               [1, 4, 512]           ║
║                    │                              │             ║
║                    └──────────┬───────────────────┘             ║
║                               ↓                                  ║
║                    ┌──────────────────┐                         ║
║                    │ Linear + Softmax │                         ║
║                    │ [512] → [vocab]   │                         ║
║                    └────────┬─────────┘                         ║
║                             ↓                                   ║
║                    输出概率分布 P(下一个词)                       ║
║                    shape: [1, 4, vocab_size]                     ║
║                                                                         ║
╚═══════════════════════════════════════════════════════════════════════════
```

---

## 📊 逐步 Shape 追踪（以翻译任务为例）

### 场景设定

```
源句子 (英文): "I love you"
目标句子 (中文): "我喜欢你"

超参数:
  d_model = 512
  d_ff = 2048
  num_heads = 8
  num_encoder_layers = 6
  num_decoder_layers = 6
  src_vocab_size = 10000   (英文词表)
  tgt_vocab_size = 8000    (中文词表)
  max_len = 100
  batch_size = 1
```

### Step 0：输入准备

```python
# 源句子（Encoder 输入）
src = [12, 25, 38]                    # "I", "love", "you"
# shape: [batch_size=1, src_len=3]

# 目标句子（Decoder 输入）— 注意：训练时偏移一位！
tgt = [55, 10, 100, 11]               # "<start>", "我", "喜欢", "你"
# shape: [batch_size=1, tgt_len=4]

# 期望输出（用于计算 Loss）
tgt_expected = [10, 100, 11, 1]        # "我", "喜欢", "你", "</end>"
```

> ⚠️ **为什么 Target 要偏移一位？**
> Decoder 的任务是"给定前面的词，预测下一个词"。所以：
> - 输入 `["<start>", "我", "喜欢", "你"]` → 应该预测 `["我", "喜欢", "你", "</end>"]`
> - 每个位置的输入 = 前一个位置的真实答案（Teacher Forcing）

---

### Step 1：Input Embedding（源句子嵌入）

```
src: [1, 3] (Token IDs)

  ↓ nn.Embedding(src_vocab_size, d_model) — 查表
token_emb: [1, 3, 512]   每个 token → 512 维向量

  ↓ × √d_model (= √512 ≈ 22.6) — 缩放
scaled: [1, 3, 512]

  ↓ + SinusoidalPositionalEncoding(max_len, d_model) — 加位置
pos_enc: [1, 3, 512]   预计算的 sin/cos 矩阵

encoder_input: [1, 3, 512]  ✅ 最终送入 Encoder
```

**每个位置的向量含义**：

```
位置 0 "I":   [语义"I"] + [位置0编码] = 512维混合向量
位置 1 "love": [语义"love"] + [位置1编码] = 512维混合向量
位置 2 "you":  [语义"you"] + [位置2编码] = 512维混合向量
```

---

### Step 2：Output Embedding（目标句子嵌入）

```
tgt: [1, 4] (Token IDs)

  ↓ nn.Embedding(tgt_vocab_size, d_model) — 查表
token_emb: [1, 4, 512]

  ↓ × √d_model — 缩放
  ↓ + Positional Encoding — 加位置

decoder_input: [1, 4, 512]  ✅ 最终送入 Decoder
```

> 注意：Decoder 的 Embedding 和 Encoder 使用**不同的词表**（不同语言），
> 但 d_model 必须相同（因为后面要做 Cross-Attention）。

---

### Step 3：Encoder 处理（×6 层）

每一层 Encoder Layer 的内部流程：

```
输入: [1, 3, 512]

  ╔══ Encoder Layer 1 ════════════════════════════╗
  ║                                               ║
  ║  residual = x  ([1, 3, 512])                 ║
  ║  x = Norm1(x)    ([1, 3, 512])  ← 归一化     ║
  ║  x = MHA(x,x,x)  ([1, 3, 512])  ← 自注意力    ║
  ║    ├── Q = x · W_Q: [1, 3, 512]              ║
  ║    ├── K = x · W_K: [1, 3, 512]              ║
  ║    ├── V = x · W_V: [1, 3, 512]              ║
  ║    ├── 分头: [1, 8, 3, 64]  (8头, 每头64维)   ║
  ║    ├── Scaled Dot-Product Attention           ║
  ║    ├── 合头: [1, 8, 3, 64] → [1, 3, 512]     ║
  ║    └── x · W_O: [1, 3, 512]                  ║
  ║  x = Dropout(x)                               ║
  ║  x = residual + x  ([1, 3, 512])  ← 残差①    ║
  ║                                               ║
  ║  residual = x  ([1, 3, 512])                 ║
  ║  x = Norm2(x)    ([1, 3, 512])               ║
  ║  x = FFN(x)      ([1, 3, 512])               ║
  ║    ├── Linear(512→2048): [1, 3, 2048]  升维   ║
  ║    ├── ReLU/GELU:    [1, 3, 2048]  非线性    ║
  ║    └── Linear(2048→512): [1, 3, 512]  降维   ║
  ║  x = Dropout(x)                               ║
  ║  x = residual + x  ([1, 3, 512])  ← 残差②    ║
  ╚═══════════════════╪════════════════════════════╝
                      ↓
              Layer 1 输出: [1, 3, 512]
                      （shape 不变，可继续堆叠）

  ... 重复 5 次 (Layer 2 ~ Layer 6) ...

                      ↓
              encoder_output: [1, 3, 512]
              （包含 "I love you" 的深层语义编码）
```

**关键观察**：
- ✅ 每层输入输出 shape 都是 `[1, 3, 512]`——**完全不变**
- ✅ 这就是残差连接的意义：shape 兼容才能相加
- ✅ Encoder 内部 **没有 Causal Mask**（双向注意力，每个 token 可以看所有 token）

---

### Step 4：Decoder 处理（×6 层）

每一层 Decoder Layer 的内部流程：

```
decoder_input: [1, 4, 512]
encoder_output: [1, 3, 512]  ← 来自 Step 3！

  ╔══ Decoder Layer 1 ════════════════════════════════════╗
  ║                                                       ║
  ║  ══ 子层一：Masked Self-Attention ══                   ║
  ║                                                       ║
  ║  residual = x  ([1, 4, 512])                         ║
  ║  x = Norm1(x)    ([1, 4, 512])                       ║
  ║  x = MHA(x, x, x, causal_mask)  ← 有 Mask!           ║
  ║    ├── Q,K,V 都来自 decoder_input                    ║
  ║    ├── attention shape: [1, 8, 4, 4]                 ║
  ║    └── Causal Mask 把上三角填 -∞                     ║
  ║  x = residual + x  ([1, 4, 512])  ← 残差①           ║
  ║                                                       ║
  ║  📌 此时: 每个位置只知道自己和之前的信息                ║
  ║                                                       ║
  ║  ══ 子层二：Cross-Attention ══                         ║
  ║                                                       ║
  ║  residual = x  ([1, 4, 512])                         ║
  ║  x = Norm2(x)    ([1, 4, 512])                       ║
  ║  x = CrossAttn(Q=x, K=enc_out, V=enc_out)            ║
  ║    ├── Q: [1, 4, 512]  ← 来自 Decoder!               ║
  ║    ├── K: [1, 3, 512]  ← 来自 Encoder!               ║
  ║    ├── V: [1, 3, 512]  ← 来自 Encoder!               ║
  ║    ├── attention shape: [1, 8, 4, 3]  👈 注意!       ║
  ║    │   (seq_len=4 查询 seq_len=3 的 key)             ║
  ║    └── 无需 Causal Mask（可以看 Encoder 全部位置）    ║
  ║  x = residual + x  ([1, 4, 512])  ← 残差②           ║
  ║                                                       ║
  ║  📌 此时: 每个位置融合了 Encoder 对源句子的理解         ║
  ║                                                       ║
  ║  ══ 子层三：Feed Forward Network ══                    ║
  ║                                                       ║
  ║  residual = x  ([1, 4, 512])                         ║
  ║  x = Norm3(x)    ([1, 4, 512])                       ║
  ║  x = FFN(x)      ([1, 4, 512])                       ║
  ║    ├── Linear(512→2048): [1, 4, 2048]                ║
  ║    ├── ReLU/GELU:    [1, 4, 2048]                    ║
  ║    └── Linear(2048→512): [1, 4, 512]                 ║
  ║  x = residual + x  ([1, 4, 512])  ← 残差③           ║
  ║                                                       ║
  ╚═══════════════════╪═══════════════════════════════════╝
                      ↓
              Layer 1 输出: [1, 4, 512]

  ... 重复 5 次 (Layer 2 ~ Layer 6) ...

                      ↓
              decoder_output: [1, 4, 512]
```

**Cross-Attention 的 Shape 细节**（这是最容易混淆的地方！）：

```
Q 来自 Decoder:  [batch=1, tgt_len=4,  d_model=512]
K 来自 Encoder:  [batch=1, src_len=3,  d_model=512]
V 来自 Encoder:  [batch=1, src_len=3,  d_model=512]

Attention 分数: Q · Kᵀ / √d_k
  shape: [1, 4, 512] @ [1, 512, 3] → [1, 4, 3]
         ↑         ↑              ↑
      tgt_len   d_model        tgt_len × src_len!

Softmax 后的权重: [1, num_heads=8, 4, 3]
  每行(共4行)对应 Decoder 的一个位置
  每列(共3列)对应 Encoder 的一个位置

最终输出: weights @ V
  [1, 8, 4, 3] @ [1, 8, 3, 64] → [1, 8, 4, 64] → [1, 4, 512]
                                        ↑
                                   回到 tgt_len 维度!
```

---

### Step 5：Linear + Softmax（输出层）

```
decoder_output: [1, 4, 512]
                      │
                      ▼
              nn.Linear(512, tgt_vocab_size)
              (将 512 维映射到词表大小)
                      │
                      ▼
              logits: [1, 4, 8000]
                      │
                      │  位置0 ("<start>"处): [0.001, 0.0005, ..., 0.8, ...]
                      │  位置1 ("我"处):      [..., 0.9, ..., 0.05, ...]  ← "我"的概率最高
                      │  位置2 ("喜欢"处):    [..., 0.03, ..., 0.85, ...] ← "喜欢"概率高
                      │  位置3 ("你"处):      [..., 0.02, ..., 0.88, ...]  ← "你"概率高
                      │
                      ▼
              softmax(logits): [1, 4, 8000]
              (每行的和为 1，就是概率分布)

📌 每个位置输出一个完整的词表概率分布
   位置 t 的输出 = P(第 t+1 个词 | 源句子, 已生成的 t 个词)
```

---

### Step 6：Loss 计算（训练时）

```
模型输出:     [1, 4, 8000]  (softmax 后的概率)
期望输出:     [1, 4]        (Token IDs: ["我","喜欢","你","</end>"])

  ↓ CrossEntropyLoss (逐位置计算，再取平均)

loss = -log(P("我" | "<start>"))
     - log(P("喜欢" | "<start> 我"))
     - log(P("你" | "<start> 我 喜欢"))
     - log(P("</end>" | "<start> 我 喜欢 你"))
     ─────────────────────────────────────
                           4

📌 Loss 越小 → 模型在每个位置都把更高概率分配给了正确答案
```

---

## 🔗 关键连接点：Encoder 输出如何流入 Decoder

这是整个数据流中最关键的"桥梁"：

```
Encoder 最后一个 Layer 的输出:
  encoder_output: [1, src_len=3, d_model=512]
                  │
                  │ 这个张量会被复制给每一个 Decoder Layer！
                  │
                  ├─→ Decoder Layer 1 的 Cross-Attention (作为 K, V)
                  ├─→ Decoder Layer 2 的 Cross-Attention (作为 K, V)
                  ├─→ Decoder Layer 3 的 Cross-Attention (作为 K, V)
                  │   ...
                  └─→ Decoder Layer 6 的 Cross-Attention (作为 K, V)

⚠️ 注意: 所有 Decoder Layer 共享同一个 encoder_output!
   但每层的 Cross-Attention 学到的关注模式不同（参数不同）
```

**为什么每层 Decoder 都需要 Cross-Attention？**

```
Layer 1 的 Cross-Attention:
  "我刚生成开头，应该重点参考原文的哪些词？"  → 关注主要实体

Layer 3 的 Cross-Attention:
  "我已经有了一些上下文，现在该细化哪些信息？"  → 关注语法关系

Layer 6 的 Cross-Attention:
  "我要做最终决策了，原文的关键证据是什么？"  → 关注全局一致性

每一层都可以独立地、不同粒度地去"查询" Encoder 的记忆。
```

---

## 🔄 训练 vs 推理的完整数据流对比

### 训练时的数据流（Teacher Forcing）

```
src: "I love you"
tgt_input:  "<start> 我 喜欢 你"
tgt_expected: "我 喜欢 你 </end>"

  ↓ 并行处理（GPU 加速）

Encoder: 一次性处理全部 src → encoder_output [1, 3, 512]
Decoder: 一次处理全部 tgt_input → logits [1, 4, 8000]
Loss: CrossEntropy(logits, tgt_expected)
Backprop: 反向传播更新所有参数

⏱️ 时间: O(1) 并行时间（一个 forward pass）
```

### 推理时的数据流（自回归 / Autoregressive）

```
src: "I love you"

Step 1:
  Encoder: 处理 src → encoder_output [1, 3, 512]  （只做一次！）
  Decoder 输入: ["<start>"]
  Decoder 输出: logits → argmax → "我"

Step 2:
  Encoder 输出: 复用！（不重新计算）
  Decoder 输入: ["<start>", "我"]
  Decoder 输出: logits → argmax → "喜欢"

Step 3:
  Decoder 输入: ["<start>", "我", "喜欢"]
  Decoder 输出: logits → argmax → "你"

Step 4:
  Decoder 输入: ["<start>", "我", "喜欢", "你"]
  Decoder 输出: logits → argmax → "</end>"  → 停止！

最终输出: "我喜欢你"

⏱️ 时间: O(tgt_len) 串行时间（tgt_len 个 forward pass）
```

> **关键优化**：Encoder 的输出只需要计算一次，然后在推理的每一步复用。
> 这也是为什么 inference 时 Encoder 的开销可以忽略不计——瓶颈全在 Decoder 的循环上。

---

## 📋 完整 Shape 变化一览表

| 阶段 | 操作 | Input Shape | Output Shape | 说明 |
|------|------|------------|-------------|------|
| **Embedding** | Token Embedding | `[1, 3]` | `[1, 3, 512]` | 查表 |
| | ×√d_model | `[1, 3, 512]` | `[1, 3, 512]` | 缩放 |
| | + Positional Enc | `[1, 3, 512]` | `[1, 3, 512]` | 加位置 |
| **Encoder L1** | LayerNorm | `[1, 3, 512]` | `[1, 3, 512]` | 归一化 |
| | MHA (Q·Kᵀ/√d) | `[1, 3, 512]` | `[1, 3, 512]` | 自注意力 |
| | Residual Add | `[1, 3, 512]` | `[1, 3, 512]` | 残差 |
| | LayerNorm | `[1, 3, 512]` | `[1, 3, 512]` | 归一化 |
| | FFN (512→2048→512) | `[1, 3, 512]` | `[1, 3, 512]` | 前馈网络 |
| | Residual Add | `[1, 3, 512]` | `[1, 3, 512]` | 残差 |
| **Encoder L2~L6** | (同上，重复5次) | `[1, 3, 512]` | `[1, 3, 512]` | 堆叠 |
| **Decoder L1** | Masked Self-Attn | `[1, 4, 512]` | `[1, 4, 512]` | 因果注意力 |
| | **Cross-Attn** | `Q:[1,4,512] K,V:[1,3,512]` | `[1, 4, 512]` | **跨模态!** |
| | FFN | `[1, 4, 512]` | `[1, 4, 512]` | 前馈网络 |
| **Decoder L2~L6** | (同上，重复5次) | `[1, 4, 512]` | `[1, 4, 512]` | 堆叠 |
| **Output** | Linear | `[1, 4, 512]` | `[1, 4, 8000]` | 映射到词表 |
| | Softmax | `[1, 4, 8000]` | `[1, 4, 8000]` | 概率分布 |

---

## 🎯 三种 Mask 在完整数据流中的位置

```
                    使用位置                    形状           作用
                  ──────────────────────────────────────────────────────

Padding Mask     Encoder Self-Attention     [1, 1, 3, 3]    屏蔽源句子中的 <pad>
(屏蔽填充)       Decoder Cross-Attention     [1, 1, 4, 3]    屏蔽 Encoder 输出中的 <pad>

Causal Mask      Decoder Self-Attention      [1, 1, 4, 4]    屏蔽"未来"的位置
(防止偷看)       (仅 Decoder 第一层子层使用)


📌 Padding Mask 同时出现在两个地方:
   1. Encoder 内部: 确保 Encoder 不关注 <pad> 位置
   2. Decoder Cross-Attention: 确保 Decoder 不关注 Encoder 输出中的 <pad> 位置

📌 Causal Mask 只出现在一个地方:
   Decoder 的 Masked Self-Attention（第一层子层）
```

---

## 💡 常见误区澄清

### ❌ 误区 1："Encoder 和 Decoder 是串行执行的"

**✅ 正确**：Encoder 先执行完，得到 `encoder_output`，然后 Decoder 才开始。但 Decoder 内部是并行的（训练时）。

### ❌ 误区 2："Cross-Attention 只在最后一层 Decoder 做"

**✅ 正确**：**每一层** Decoder 都有自己的 Cross-Attention，它们共享同一个 `encoder_output`，但每层学到的关注模式不同。

### ❌ 误区 3："推理时 Encoder 也要重新计算"

**✅ 正确**：Encoder 输出只算一次，然后缓存起来供 Decoder 的每一步复用。这是 KV Cache 优化的基础。

### ❌ 误区 4："Decoder 的输入和输出长度必须相同"

**✅ 正确**：训练时确实相同（Teacher Forcing）。但推理时 Decoder 输入从长度 1 逐步增长到最终长度。输出的 logits 长度始终等于当前输入长度。

### ❌ 误区 5："Linear 输出层只在最后有一个"

**✅ 正确**：是的，整个模型只有一个输出 Linear 层（`nn.Linear(d_model, vocab_size)`），它共享所有位置的参数。不管序列多长，每个位置都用同一组权重映射到词表。

