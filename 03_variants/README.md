# 03_variants — 三种主流 Transformer 变体

> 同一套 Transformer 机制，因训练目标和使用场景的不同，演化出了三种主流架构。
> 本目录用精简代码逐一实现，帮助理解它们的设计差异和使用场景。

---

## 三种架构对比

| 维度 | Only Encoder | Only Decoder | Encoder-Decoder |
|------|-------------|-------------|----------------|
| **代表模型** | BERT, RoBERTa, ALBERT | GPT, LLaMA, Qwen, DeepSeek | T5, BART, 原始 Transformer |
| **注意力方向** | 双向（全局） | 单向（左→右） | Enc 双向 + Dec 单向 |
| **训练目标** | MLM（掩码语言模型） | CLM（因果语言模型） | Seq2Seq |
| **适合任务** | 分类、NER、问答理解 | 文本生成、对话、代码 | 翻译、摘要、条件生成 |
| **参数效率** | 高（理解任务） | 高（生成任务） | 中（两者兼顾） |

---

## 文件说明

### `only_encoder.py` — BERT 风格（Only Encoder）

去掉 Decoder，只保留 Encoder，利用双向注意力对输入序列做深度理解。

| 类 | 说明 |
|----|------|
| `BertLikeEncoder` | 核心模块：可学习位置编码 + GELU + Post-LN，BERT 配置 |
| `BertForClassification` | 文本分类头：取 `[CLS]` token 表示 → 线性层 → 类别概率 |
| `BertForTokenClassification` | 序列标注头（NER）：每个 token 表示 → 线性层 → 标签 |
| `BertForMLM` | 预训练 MLM 头：每个 token 表示 → 预测原始词 |

**与原版 BERT 的差异**（教学简化）：
- 使用可学习位置编码（与 BERT 相同）
- 使用 GELU 激活（与 BERT 相同）
- Post-LN 归一化（与 BERT 相同）
- 省略了 NSP（Next Sentence Prediction）任务头
- 省略了 Token Type Embedding（segment embedding）

**为什么 Only Encoder 不适合生成**：
双向注意力意味着每个 token 的表示依赖整个序列（包括未来 token），生成时无法做到。

---

### `only_decoder.py` — GPT 风格（Only Decoder）

去掉 Encoder 和交叉注意力，只保留带 Causal Mask 的自注意力层。

| 类 | 说明 |
|----|------|
| `GPTDecoderLayer` | 单层：MaskedSelfAttention → Add&Norm → FFN → Add&Norm（Pre-LN） |
| `GPTLikeDecoder` | 核心模块：可学习位置编码 + GELU + Pre-LN，GPT 配置 |
| `GPTForLM` | 语言模型头：每个位置的隐状态 → 预测下一个 token |

**生成方法（`GPTForLM.generate`）**：
- **贪心（Greedy）**：每步选概率最大的 token，确定性但单调
- **温度采样（Temperature）**：调节 `softmax(logits/T)`，T>1 更随机，T<1 更确定
- **Top-K 采样**：只在概率最高的 K 个 token 中采样
- **Top-P 采样（Nucleus）**：只在累积概率超过 P 的最小 token 集合中采样

**Pre-LN 的优势**：梯度在残差路径上直接回传，不经过 LayerNorm，训练更稳定，是 GPT-2 以后的标准。

---

### `encoder_decoder.py` — T5/BART 风格（Encoder-Decoder）

完整的 Encoder-Decoder 架构，T5 风格（Pre-LN + 相对位置编码 + 共享词嵌入）。

| 类 | 说明 |
|----|------|
| `RelativePositionBias` | 简化版相对位置偏置（T5 风格），替代绝对位置编码 |
| `T5EncoderLayer` | T5 风格 Encoder 层：Pre-LN，无位置编码（由 Bias 提供） |
| `T5DecoderLayer` | T5 风格 Decoder 层：Pre-LN，含交叉注意力 |
| `T5LikeTransformer` | 完整模型：Encoder + Decoder + 共享词嵌入 |

**T5 与原始 Transformer 的主要差异**：
1. **相对位置编码**：不用 sin/cos，改用可学习的相对位置偏置，更灵活
2. **Pre-LN**：层归一化前置，训练更稳定
3. **共享词嵌入**：Encoder 和 Decoder 共用同一份 Embedding，减少参数量
4. **无 Bias 的线性层**：所有 Linear 层去掉偏置项

---

### `__init__.py` — 包初始化

导出三个变体的主要类。

---

## 运行方式

```bash
# 逐一验证三种变体
python 03_variants/only_encoder.py
python 03_variants/only_decoder.py
python 03_variants/encoder_decoder.py

# 或通过主入口
python run_all.py --step 3
```

---

## 选择架构的决策树

```
你的任务是什么？
│
├── 需要"理解"输入（分类/NER/问答）
│     └── → Only Encoder（BERT 系列）
│
├── 需要"生成"文本（续写/对话/代码）
│     └── → Only Decoder（GPT 系列）
│
└── 需要"输入→输出"（翻译/摘要/改写）
      └── → Encoder-Decoder（T5/BART 系列）

