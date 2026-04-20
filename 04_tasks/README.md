# 04_tasks — 三种实际训练任务

> 理论最终要落地于实践。本目录通过三个端到端的训练任务，验证三种 Transformer 变体在真实场景中的效果。
> 每个任务均使用**合成数据**，无需下载外部数据集，可直接运行。

---

## 任务概览

| 任务 | 架构 | 训练目标 | 最终效果 |
|------|------|---------|---------|
| 文本分类 | Only Encoder（BERT 风格） | 交叉熵分类损失 | **99.2% 准确率**（10 epochs） |
| 字符级语言模型 | Only Decoder（GPT 风格） | Causal LM 损失 | **Epoch 3 完美续写莎士比亚** |
| 数字→单词翻译 | Encoder-Decoder（T5 风格） | Seq2Seq 损失 | **100% 翻译准确率**（Epoch 15） |

---

## 文件说明

### `classification_task.py` — 文本分类（情感分析）

使用 **Only Encoder** 完成情感分类任务（正面 / 负面 / 中性三分类）。

**合成数据设计**：
- 正面文本：以高频词 ID 开头（模拟正面情感词汇），长度 6～12
- 负面文本：以低频词 ID 开头（模拟负面情感词汇），长度 6～12
- 中性文本：随机词 ID 混合，不明显偏向

**模型结构**：
```
token ids [B, L]
   → TokenEmbedding + LearnablePositionalEncoding
   → N × EncoderLayer（双向自注意力 + FFN）
   → [CLS] token 表示 [B, d_model]
   → 线性分类头
   → logits [B, num_classes]
```

**训练技巧**：
- 批归一化 dropout（0.1）防止过拟合
- AdamW 优化器 + 学习率 Warmup
- 动态 padding（batch 内补齐到最长序列）

**关键代码路径**：`SentimentDataset` → `SentimentClassifier` → `train_epoch` → `evaluate`

---

### `lm_task.py` — 字符级语言模型

使用 **Only Decoder** 训练字符级语言模型，学习莎士比亚风格的英文文本。

**数据**：内嵌的莎士比亚《哈姆雷特》片段（约 12,000 字符），无需下载。

**字符级分词**：
- 词表 = 文本中出现的所有不重复字符（约 39 个）
- 每个字符对应一个 token id
- 优点：词表极小，适合演示；缺点：需要更长序列

**训练目标**：给定前 N 个字符，预测第 N+1 个字符（CLM）。

**生成效果演示**（训练完成后）：
```
提示词 → "To be, or "
生成结果 → "To be, or not to be, that is the question:
            Whether 'tis nobler in the mind to suffer..."
```

**采样策略**：实现了贪心、温度采样、top-k、top-p 四种生成方式。

**关键代码路径**：`CharDataset` → `GPTStyleModel` → `train` → `generate`

---

### `translation_task.py` — 数字→单词翻译（Seq2Seq）

使用完整 **Encoder-Decoder** 架构训练一个迷你翻译系统。

**任务设计**（教学向）：
- 输入（"源语言"）：数字序列，如 `[3, 1, 4]`
- 输出（"目标语言"）：对应英文单词，如 `"three one four"`
- 简化了真实翻译，但完整演示了 Seq2Seq 的训练和推理流程

**特殊 Token**：
```
<pad> = 0   # 填充
<bos> = 1   # 序列开始（Begin of Sequence）
<eos> = 2   # 序列结束（End of Sequence）
<unk> = 3   # 未知词
```

**Teacher Forcing（训练）**：
```
src:        [3,   1,    4   ]
tgt_input:  [<bos>, three, one  ]  ← Decoder 输入（右移一位）
tgt_target: [three, one,   four ]  ← 损失计算目标
```

**自回归解码（推理）**：
```
src: [3, 1, 4]
Step 1: Decoder 输入 [<bos>]         → 预测 "three"
Step 2: Decoder 输入 [<bos>, three]  → 预测 "one"
Step 3: Decoder 输入 [<bos>, ..., one] → 预测 "four"
Step 4: Decoder 输入 [...]           → 预测 <eos>，停止
```

**优化器配置**：AdamW（lr=1e-3）+ CosineAnnealingLR，比原论文 Noam Scheduler 在小数据集上收敛更快。

**关键代码路径**：`NumberToWordDataset` → `Seq2SeqTransformer` → `train_epoch` → `evaluate_accuracy`

**演示类**：
| 类/函数 | 说明 |
|---------|------|
| `NoamScheduler` | 原论文学习率调度（保留作展示，实际训练改用 AdamW+Cosine） |
| `LabelSmoothingLoss` | 手动实现标签平滑（保留作展示，实际训练用 PyTorch 内置） |

---

### `__init__.py` — 包初始化

---

## 运行方式

```bash
# 分别运行各个任务（各需 1～5 分钟，取决于机器性能）
python 04_tasks/classification_task.py   # ~1 分钟
python 04_tasks/lm_task.py              # ~3 分钟
python 04_tasks/translation_task.py     # ~5 分钟

# 或通过主入口
python run_all.py --step 4
```

---

## 关键训练技巧汇总

| 技巧 | 用于 | 作用 |
|------|------|------|
| Teacher Forcing | 翻译（Seq2Seq） | 训练时用真实目标序列输入 Decoder，避免错误积累 |
| Label Smoothing | 翻译 | 将 one-hot 目标软化，防止模型过度自信 |
| Gradient Clipping | 全部任务 | `clip_grad_norm_(params, 1.0)`，防止梯度爆炸 |
| Noam Scheduler | 原论文展示 | `lr = d_model^(-0.5) × min(step^(-0.5), step×warmup^(-1.5))` |
| CosineAnnealingLR | 翻译实际训练 | 余弦退火，对小数据集更友好 |
| Dynamic Padding | 分类、翻译 | batch 内只填充到最长序列，而非全局最大长度 |

