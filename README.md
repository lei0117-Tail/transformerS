# Transformer 从 0 到 1 学习课程

> 一份带完整源码的 Transformer 学习课程，从最基础的矩阵乘法出发，逐步搭建出 BERT、GPT、T5 三种主流架构，并通过真实训练任务加以验证。

---

## 目录结构

```
transformerS/
├── 01_basics/            # 第一阶段：基础构建块（Step 1～6）
├── 02_full_transformer/  # 第二阶段：完整 Transformer
├── 03_variants/          # 第三阶段：三种主流架构变体
├── 04_tasks/             # 第四阶段：三种实际训练任务
├── utils/                # 工具模块（可视化 / 分词器）
├── run_all.py            # 一键运行入口
└── requirements.txt      # Python 依赖
```

---

## 快速开始

```bash
# 1. 激活虚拟环境（项目根目录已包含 .venv）
source .venv/bin/activate

# 2. 一键运行所有基础验证
python run_all.py

# 3. 分步运行
python run_all.py --step 1   # 基础构建块
python run_all.py --step 2   # 完整 Transformer
python run_all.py --step 3   # 三种变体
python run_all.py --step 4   # 三种训练任务

# 4. 单独运行任一文件
python 01_basics/01_embedding.py
python 04_tasks/translation_task.py
```

---

## 学习路径

### 第一阶段：基础构建块（`01_basics/`）

| 步骤 | 文件 | 核心概念 |
|------|------|---------|
| Step 1 | `01_embedding.py` | Token Embedding + 正弦位置编码 |
| Step 2 | `02_attention.py` | 缩放点积注意力 + Mask |
| Step 3 | `03_multi_head_attention.py` | 多头注意力机制 |
| Step 4 | `04_feed_forward.py` | 位置前馈网络（ReLU/GELU/SwiGLU） |
| Step 5 | `05_layer_norm.py` | 层归一化 + 残差连接（Pre/Post-LN） |
| Step 6 | `06_encoder_layer.py` | 完整 Encoder 层 |

### 第二阶段：完整 Transformer（`02_full_transformer/`）

| 文件 | 说明 |
|------|------|
| `masks.py` | 所有 Mask 工具函数（Padding / Causal / Decoder） |
| `encoder.py` | N 层 Encoder 堆叠 |
| `decoder.py` | N 层 Decoder（含交叉注意力） |
| `transformer.py` | 完整 Encoder-Decoder，含贪心解码与 Beam Search |

### 第三阶段：三种架构变体（`03_variants/`）

| 变体 | 代表模型 | 适用任务 |
|------|---------|---------|
| Only Encoder | BERT / RoBERTa | 分类、NER、问答理解 |
| Only Decoder | GPT / LLaMA | 文本生成、对话 |
| Encoder-Decoder | T5 / BART | 翻译、摘要、Seq2Seq |

### 第四阶段：训练任务验证（`04_tasks/`）

| 任务 | 使用架构 | 训练结果 |
|------|---------|---------|
| 文本分类 | Only Encoder | **99.2% 准确率**（10 epochs） |
| 字符级语言模型 | Only Decoder | **Epoch 3 完美生成莎士比亚** |
| 数字→单词翻译 | Encoder-Decoder | **100% 翻译准确率**（Epoch 15） |

---

## 环境依赖

```
torch >= 2.0
numpy >= 1.18
matplotlib >= 3.2
```

项目附带 `.venv/` 虚拟环境，内含 PyTorch 2.8.0，可直接激活使用。

---

## 自我调适记录

本课程在调试过程中经历了以下关键优化，记录于此供学习参考：

### 1. Mask 类型统一：Float → Bool

**问题**：`torch.tril(torch.ones(...))` 返回 Float 类型，与 `(seq != pad_idx)` 返回的 Bool 类型做 `&` 操作时抛出 `NotImplementedError`。

**根因**：PyTorch 的按位与运算 `&` 不支持 Float 张量，需统一为 Bool。

**修复**：
```python
# 修复前（报错）
mask = torch.tril(torch.ones(seq_len, seq_len))

# 修复后
mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
```

**影响文件**：`02_full_transformer/masks.py`、`01_basics/02_attention.py`、`01_basics/03_multi_head_attention.py`、`02_full_transformer/decoder.py`、`03_variants/only_decoder.py`、`03_variants/encoder_decoder.py`、`04_tasks/lm_task.py`、`04_tasks/translation_task.py`

---

### 2. 翻译任务 optimizer.step() 缺失

**问题**：翻译任务训练 30 个 epoch，Loss 始终卡在 2.77（约等于 `-log(1/14)` 即随机水平）。

**排查过程**：
1. 单样本过拟合测试 → Loss 可以快速降到 0.0001，说明模型本身没问题
2. 检查 `train_epoch` 函数 → 发现 `optimizer.step()` 从未被调用！
3. 根因：引入 `_NoamWrapper` 类时，其 `.step()` 方法只是 `pass`，而 `train_epoch` 只调用了 `scheduler.step()`，没有单独调用 `optimizer.step()`

**修复**：
```python
# 修复前（参数从未更新！）
scheduler.step()

# 修复后
scheduler.optimizer.step()   # 先更新参数
scheduler.step()             # 再更新 lr
```

**效果**：修复后 Epoch 1 即开始收敛，Epoch 15 达到 100% 翻译准确率。

---

### 3. Noam Scheduler 峰值学习率过低

**问题**：原代码用 `d_model=64`、`warmup_steps=200` 的 Noam Scheduler，峰值 lr ≈ `64^(-0.5) / sqrt(200) ≈ 0.00088`，对小数据集来说过低。

**优化**：改用 `AdamW + CosineAnnealingLR`，初始 lr=1e-3，更适合小数据集快速收敛。

```python
# 优化前
optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
scheduler = NoamScheduler(optimizer, d_model=64, warmup_steps=200)

# 优化后
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=5e-5)
```

---

### 4. 导入路径规范化（Linter 修复）

**问题**：早期版本通过 `sys.path.insert + from module import func` 动态导入，Linter 会报 Unresolved reference 错误。

**优化**：改用 `importlib.util.spec_from_file_location` 进行完全动态加载，消除静态分析误报。同时将各 Basics 文件中交叉依赖改为内联（避免跨文件相对引用）。

---

### 5. import 顺序规范化

按照 PEP 8 规范，将 `import` 顺序调整为：**标准库 → 第三方库 → 本地库**，并在各分组之间加空行。例如：

```python
# 规范后
import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

