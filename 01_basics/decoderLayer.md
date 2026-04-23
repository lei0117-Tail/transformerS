# Decoder Layer — 解码器三层架构详解

## 🎯 Decoder 解决什么问题？

**Encoder 负责"理解输入"，Decoder 负责"生成输出"。**

如果说 Encoder 是一个**阅读者**（读入整句话，全面理解），那么 Decoder 就是一个**写作者**（一个词一个词地写出答案）。Decoder 的三层架构正是为了支持这种 **"边看边写、逐步生成"** 的模式。

### 🔑 为什么 Decoder 需要 3 层，而 Encoder 只有 2 层？

| 需求 | Encoder 的做法 | Decoder 的做法 | 为什么 Decoder 多一层 |
|------|---------------|---------------|---------------------|
| 理解自身 | Self-Attention（双向） | Self-Attention（**单向/因果**） | Decoder 生成时不能看"未来" |
| 获取来源信息 | 不需要（自己就是源头） | **Cross-Attention**（看向 Encoder） | Decoder 需要参考输入来生成输出 |
| 加工信息 | FFN | FFN | 相同 |

> **一句话：Decoder 比 Encoder 多了一个 Cross-Attention 层，用来"参考 Encoder 的理解结果"。**

---

## Decoder Layer 完整结构

```
┌─────────────────────────────────────────────────────┐
│              Decoder Layer（一层）                    │
│                                                     │
│   ┌───────────────────────────────────┐             │
│   │  第一层: Masked Self-Attention     │  ← 子层1    │
│   │  (带 Causal Mask 的自注意力)        │             │
│   └─────────────────┬─────────────────┘             │
│                     ↓ Add & Norm                      │
│   ┌───────────────────────────────────┐             │
│   │  第二层: Cross-Attention           │  ← 子层2 👈  │
│   │  (交叉注意力，看向 Encoder)         │  Decoder特有!│
│   └─────────────────┬─────────────────┘             │
│                     ↓ Add & Norm                      │
│   ┌───────────────────────────────────┐             │
│   │  第三层: Feed Forward Network      │  ← 子层3    │
│   │  (前馈网络，与 Encoder 相同)        │             │
│   └─────────────────┬─────────────────┘             │
│                     ↓ Add & Norm                      │
└─────────────────────────────────────────────────────┘
```

**对比 Encoder Layer**：

```
Encoder Layer (2 个子层):          Decoder Layer (3 个子子层):
┌──────────────────────┐          ┌──────────────────────────┐
│ ① Self-Attention     │          │① Masked Self-Attention  │
│   (双向，无 Mask)     │          │   (单向，有 Causal Mask) │
├──────────────────────┤          ├──────────────────────────┤
│② FFN                 │          │② Cross-Attention  👈多! │
└──────────────────────┘          ├──────────────────────────┤
                                   │③ FFN                     │
                                   └──────────────────────────┘
```

---

## 三层逐一详解

### 第一层：Masked Self-Attention（掩码自注意力）

#### 🎯 这一层解决什么问题？

**防止 Decoder "偷看未来"。**

在训练时，我们一次性把整个目标序列（如翻译答案）喂给 Decoder。但模型在**实际推理时**是逐词生成的——生成第 3 个词时，它只能看到第 1、2 个词，不能看到第 4、5 个词。

Causal Mask 就是强制让模型在训练时就养成"不看未来"的习惯。

#### 🔑 为什么要用 Causal Mask？

```
没有 Mask（Encoder 的做法）:
  生成 "我" 时，能看到 [我, 喜欢, 你, 吗, ？] 全部 → 作弊！❌

有 Causal Mask（Decoder 的做法）:
  生成 "我"   时，只能看到 [我]                          ✅
  生成 "喜欢" 时，只能看到 [我, 喜欢]                    ✅
  生成 "你"   时，只能看到 [我, 喜欢, 你]                ✅
  ...
```

#### 📊 Causal Mask 原理

Causal Mask 是一个**下三角矩阵**，把"未来"位置填成 `-∞`：

```
       位置0   位置1   位置2   位置3   位置4
       "我"   "喜欢"   "你"    "吗"    "?"
位置0 "我"   [  0    -∞     -∞     -∞     -∞  ]   ← 只看自己
位置1 "喜欢" [  ✓     0     -∞     -∞     -∞  ]   ← 看 0,1
位置2 "你"   [  ✓     ✓      0     -∞     -∞  ]   ← 看 0,1,2
位置3 "吗"   [  ✓     ✓      ✓      0     -∞  ]   ← 看 0,1,2,3
位置4 "?"    [  ✓     ✓      ✓      ✓      0  ]   ← 看全部（最后一个）
```

经过 Softmax 后，`-∞` 位置变成 `0`（完全忽略）：

```
Softmax 后的注意力权重（每行和为 1）:

位置0 "我":   [1.00,  0.00,  0.00,  0.00,  0.00]   100% 关注自己
位置1 "喜欢": [0.47,  0.53,  0.00,  0.00,  0.00]   关注 "我" 和 "喜欢"
位置2 "你":   [0.31,  0.28,  0.41,  0.00,  0.00]   关注 "我"、"喜欢"、"你"
位置3 "吗":   [0.22,  0.20,  0.18,  0.40,  0.00]   关注前4个
位置4 "?":    [0.18,  0.16,  0.15,  0.14,  0.37]   关注全部
```

#### 💡 直观类比：考试答题

```
Causal Mask = 考试时不能翻到后面看答案

Encoder = 开卷考试（可以看到整篇文章的所有内容）
Decoder = 闭卷写作（写第一句时不能看第二句怎么写）
```

#### 代码对应

```python
# Decoder 的 Self-Attention
# Q = K = V = x（来自 Decoder 的输入）
# 但多了 causal_mask 参数！
attn_output = self.self_attn(x, x, x,
                              mask=causal_mask)   # 👈 关键区别！
```

---

### 第二层：Cross-Attention（交叉注意力）

#### 🎯 这一层解决什么问题？

**让 Decoder 能够"参考" Encoder 对源句子的理解结果来生成输出。**

翻译任务中，Decoder 在生成目标语言的每个词时，需要知道源句子中的哪些词与当前要生成的词相关。Cross-Attention 就是这座"桥梁"。

#### 🔑 Cross-Attention 的核心：Q 来自 Decoder，K/V 来自 Encoder

这是 Cross-Attention 和 Self-Attention 的**本质区别**：

```
Self-Attention（第一层）:           Cross-Attention（第二层）:
  Q = Decoder 自己的向量              Q = Decoder 的向量      ← 查询方
  K = Decoder 自己的向量              K = Encoder 的输出向量   ← 被查询方
  V = Decoder 自己的向量              V = Encoder 的输出向量   ← 被查询方

  自己问自己："我和自己的哪些部分相关？"   问 Encoder："你的哪些部分和我相关？"
```

#### 📊 具体例子：翻译 "I love you" → "我喜欢你"

假设 Encoder 已经处理完源句子 "I love you"，现在 Decoder 要生成第一个字 "我"：

```
Encoder 输出（K, V 来源）:
  "I":    [0.3, 1.2, -0.5, ...]   ← 编码后的 "I"
  "love": [0.8, 0.1,  2.3, ...]   ← 编码后的 "love"
  "you":  [-0.2, 0.9,  0.4, ...]   ← 编码后的 "you"

Decoder 当前状态（Q 来源）:
  正准备生成 "我": [0.5, -0.3, 1.1, ...]

Cross-Attention 计算:
  Q(Decoder) · K(Encoder)ᵀ / √d_k → 注意力分数
  ↓
  Softmax → 权重: [0.15, 0.10, 0.75]
                         ↑    ↑    ↑
                       "I" "love" "you"  → 最关注 "you"！
  ↓
  加权求和 V → 输出: 主要包含 "you" 的编码信息
  ↓
  结论: Decoder 知道要生成 "我"，应该重点参考源句中的 "you"
```

这就是为什么叫 **"交叉"** 注意力 —— 它跨越了 Encoder 和 Decoder 的边界！

#### 🔑 为什么 Cross-Attention 不需要 Causal Mask？

```
Cross-Attention 中:
  - Q 来自 Decoder（当前正在生成的位置）
  - K, V 来自 Encoder（已经完整编码的源句子）

Encoder 的输出是"完整的"——它已经看到了源句子的所有 token，
所以 Decoder 可以自由地关注 Encoder 输出的任何位置。

类比:
  你在写一封回信（Decoder），
  可以随时翻看对方的原信（Encoder 输出）的任何部分。
  限制只在于你不能看你还没写出来的回信内容（由第一层的 Causal Mask 保证）。
```

#### 代码对应

```python
# Cross-Attention
# Q 来自 Decoder，K 和 V 来自 Encoder 的输出！
attn_output = self.cross_attn(query=x,      # Decoder 的当前状态
                               key=enc_out,   # 👈 Encoder 的输出
                               value=enc_out, # 👈 Encoder 的输出
                               mask=pad_mask) # 只需要 Padding Mask
```

---

### 第三层：Feed Forward Network（前馈网络）

#### 🎯 这一层的作用？

**和 Encoder 中的 FFN 完全相同。**

每个 token 独立地做非线性变换：升维 → ReLU/GELU → 降维。

```python
# Decoder 的 FFN — 与 Encoder 一模一样
ffn_output = self.ffn(x)
# 内部: Linear(d_model→d_ff) → Activation → Linear(d_ff→d_model)
```

#### 为什么 Decoder 也需要 FFN？

```
第一层 Masked Self-Attention: "我已经写了什么？上下文是什么？"   （内部关系）
第二层 Cross-Attention:         "原文说了什么？和我有什么关系？"   （外部参考）
第三层 FFN:                     "基于以上信息，我应该输出什么？"   （独立加工）

FFN 给 Decoder 提供了独立思考的能力，
而不是只会"照搬" Attention 的加权结果。
```

---

## 完整数据流图（单个 Decoder Layer）

以翻译任务为例，Pre-LN 模式：

```
输入 x: Decoder 的嵌入表示（已加 Positional Encoding）
      例如: ["<start>", "我", "喜欢"]
      shape: [batch=1, seq_len=3, d_model=512]

enc_out: Encoder 的最终输出（来自源句子 "I love you"）
         shape: [batch=1, src_len=3, d_model=512]
│
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  ══ 子层一：Masked Self-Attention ══                          ║
║                                                              ║
║  residual = x                                                ║
║  normed = LayerNorm(x)                                       ║
║  attn = Softmax(Q·Kᵀ/√d + causal_mask) · V                   ║
║       ↑ Causal Mask 防止偷看未来                              ║
║  x = residual + Dropout(attn)                                 ║
║                                                              ║
║  📌 此时 x 融合了: Decoder 自身的上下文信息                    ║
║     ("我"知道了前面有 "<start>"）                             ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  ══ 子层二：Cross-Attention ══                                ║
║                                                              ║
║  residual = x                                                ║
║  normed = LayerNorm(x)                                       ║
║  attn = Softmax(Q_dec·K_encᵀ/√d) · V_enc                     ║
║       ↑ Q 来自 Decoder, K/V 来自 Encoder!                    ║
║  x = residual + Dropout(attn)                                 ║
║                                                              ║
║  📌 此时 x 融合了: Encoder 对源句子的理解                     ║
║     ("我"知道了对应英文中的 "you"）                           ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  ══ 子层三：Feed Forward Network ══                           ║
║                                                              ║
║  residual = x                                                ║
║  normed = LayerNorm(x)                                       ║
║  ffn = ReLU(x·W₁ + b₁) · W₂ + b₂                            ║
║       ↑ 升维→非线性→降维                                     ║
║  x = residual + Dropout(ffn)                                  ║
║                                                              ║
║  📌 此时 x 是最终输出:                                        ║
║     自身上下文 + Encoder 信息 + 非线性加工                    ║
║                                                              ║
╚══════════════════════╪═══════════════════════════════════════╝
                        ▼
              最终输出 → 送入下一层 Decoder Layer
                         （或送入 Linear + Softmax 生成下一个词的概率）
```

---

## 训练 vs 推理：Decoder 的两种工作模式

### 训练阶段：Teacher Forcing（教师强制）

```
目标翻译: "我喜欢你"

输入 Decoder: ["<start>", "我", "喜欢", "你"]
期望输出:   [ "我",     "喜欢", "你",  "</end>"]

特点:
✅ 一次性输入整个序列（并行计算，效率高）
✅ 使用 Causal Mask 防止每个位置看后面
✅ 输入的是**真实答案**（Ground Truth），不是模型自己的预测
⚠️ 这就是 Teacher Forcing：老师把正确答案直接告诉你
```

Teacher Forcing 的数据流：

```
时间步 t=1: 输入="<start>" → 应该输出 "我"    ← 用真实 "我" 当下一步输入
时间步 t=2: 输入="我"      → 应该输出 "喜欢"  ← 用真实 "喜欢" 当下一步输入
时间步 t=3: 输入="喜欢"    → 应该输出 "你"    ← 用真实 "你" 当下一步输入
时间步 t=4: 输入="你"      → 应该输出 "</end>"
```

### 推理阶段：自回归生成（Autoregressive）

```
生成过程（逐词生成）:

Step 1: 输入 ["<start>"]     → 输出概率 → 选择 "我"   (argmax)
Step 2: 输入 ["<start>", "我"] → 输出概率 → 选择 "喜欢" (argmax)
Step 3: 输入 ["<start>", "我", "喜欢"] → 输出 → 选择 "你" (argmax)
Step 4: 输入 ["<start>", "我", "喜欢", "你"] → 输出 → "</end]" (停止)

特点:
❌ 不能并行（必须等上一个词生成完）
✅ 每一步的输入都是**上一步模型自己预测的**
⚠️ 如果中间预测错了，错误会累积（误差传播）
```

### 对比总结

| | 训练（Teacher Forcing） | 推理（自回归） |
|--|------------------------|---------------|
| **输入** | 整个目标序列（真实答案） | 逐步拼接已生成的词 |
| **并行性** | ✅ 高度并行 | ❌ 串行逐词生成 |
| **输入来源** | Ground Truth | 模型自己的预测 |
| **Causal Mask** | ✅ 需要 | ✅ 需要 |
| **效率** | 快（GPU 并行加速） | 慢（需等待每步完成） |
| **风险** | 训练推理 gap（训练时看到的永远是对的） | 错误累积 |

---

## Encoder vs Decoder 完整对比

| 维度 | Encoder | Decoder |
|------|---------|---------|
| **角色** | 理解输入（阅读者） | 生成输出（写作者） |
| **子层数** | 2 个 | **3 个** |
| **Self-Attention** | 双向（可看所有位置） | 单向（Causal Mask，只能看过去） |
| **Cross-Attention** | ❌ 无 | ✅ 有（Q=Dec, K/V=Enc） |
| **输入** | 源句子 embedding | 目标句子 embedding（偏移一位） |
| **输出** | 传给 Decoder 的 Cross-Attention | 经过 Linear → Softmax 生成词概率 |
| **类比** | 阅读+理解全文 | 边看笔记边写作文 |

### 视觉对比图

```
═══════════════════════════════════════════════════════════════
                    Encoder（编码器）
═══════════════════════════════════════════════════════════════

  "I" "love" "you"
    ↓    ↓     ↓
  [Embedding + Positional Encoding]
    ↓
  ┌─────────────────┐
  │ Self-Attention  │  ← 我可以看 "I","love","you" 所有词
  │   (双向)         │
  ├─────────────────┤
  │       FFN       │
  └────────┬────────┘
           ↓
      Encoder Output  ──────────────────┐
      (包含了 "I love you" 的完整理解)   │
                                         ↓ Cross-Attention


═══════════════════════════════════════════════════════════════
                    Decoder（解码器）
═══════════════════════════════════════════════════════════════

  "<s>" "我" "喜欢"
    ↓     ↓    ↓
  [Embedding + Positional Encoding]
    ↓
  ┌─────────────────────┐
  │ Masked Self-Attn    │  ← "我"只能看"<s>"，不能看"喜欢"
  │   (单向 + Causal)    │
  ├─────────────────────┤
  │  Cross-Attention  ◄──┼── 来自 Encoder 的输出
  │   (Q=Dec, K/V=Enc)  │     "我"去查 Encoder:"谁跟我相关?"
  ├─────────────────────┤
  │        FFN          │
  └────────┬────────────┘
           ↓
      Linear + Softmax → 下一个词的概率分布
```

---

## 三层各自的责任总结

```
Decoder Layer 的三个子层，就像一个翻译官的工作流程：

  第一层 Masked Self-Attention
  📖 "我已经翻译了什么？上下文连贯吗？"
     → 确保生成的目标语言 internally consistent

  第二层 Cross-Attention
  📖 "原文说了什么？哪个词对应我现在要翻的？"
     → 从 Encoder 的理解中提取相关信息

  第三层 Feed Forward Network
  📖 "综合以上信息，最终怎么表达？"
     → 独立进行非线性特征变换，做出最终决策

三者缺一不可：
  缺第一层 → 生成的目标语言不连贯（不知道前面说了什么）
  缺第二层 → 无法利用源句子信息（翻译质量差）
  缺第三层 → 表达能力不足（无法做复杂变换）

