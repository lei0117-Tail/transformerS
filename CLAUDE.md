# CLAUDE.md — AI 记忆文件

> 这是我（CatPaw / Claude）在参与本项目过程中的工作记忆。
> 记录了项目结构、关键决策、踩过的坑和我自己的调适过程。
> 下次继续开发本项目时，请先阅读此文件。

---

## 项目概况

- **项目路径**：`/Users/liulei/aiS/transformerS`
- **目标**：Transformer 从 0 到 1 学习课程，含源码，支持三种架构变体
- **虚拟环境**：`.venv/`，Python 3.9.6，PyTorch 2.8.0，已完整安装所有依赖
- **激活方式**：`source .venv/bin/activate`（或直接用 `.venv/bin/python`）
- **一键运行**：`python run_all.py`

---

## 目录结构与职责

```
01_basics/           Step 1-6，每个文件独立可运行，末尾有验证代码
02_full_transformer/ 完整 Enc-Dec Transformer，自包含（不 import 01_basics）
03_variants/         三种变体，自包含
04_tasks/            三种训练任务，自包含
utils/               可视化 + 分词器，辅助工具
run_all.py           总入口，--step 1/2/3/4 分步运行
```

---

## 已验证的运行结果（2026-04-20）

| 模块/任务 | 运行命令 | 结果 |
|---------|---------|------|
| Step 1: Embedding | `python 01_basics/01_embedding.py` | ✅ 通过 |
| Step 2: Attention | `python 01_basics/02_attention.py` | ✅ 通过 |
| Step 3: MHA | `python 01_basics/03_multi_head_attention.py` | ✅ 通过 |
| Step 4: FFN | `python 01_basics/04_feed_forward.py` | ✅ 通过 |
| Step 5: LayerNorm | `python 01_basics/05_layer_norm.py` | ✅ 通过 |
| Step 6: EncoderLayer | `python 01_basics/06_encoder_layer.py` | ✅ 通过 |
| Masks | `python 02_full_transformer/masks.py` | ✅ 通过 |
| Encoder | `python 02_full_transformer/encoder.py` | ✅ 通过 |
| Decoder | `python 02_full_transformer/decoder.py` | ✅ 通过 |
| Transformer | `python 02_full_transformer/transformer.py` | ✅ 通过 |
| Only Encoder | `python 03_variants/only_encoder.py` | ✅ 通过 |
| Only Decoder | `python 03_variants/only_decoder.py` | ✅ 通过 |
| Enc-Dec | `python 03_variants/encoder_decoder.py` | ✅ 通过 |
| 分类任务 | `python 04_tasks/classification_task.py` | ✅ **99.2% 准确率** |
| 语言模型 | `python 04_tasks/lm_task.py` | ✅ **Epoch 3 完美续写** |
| 翻译任务 | `python 04_tasks/translation_task.py` | ✅ **100% 准确（Epoch 15）** |

---

## 关键设计决策

### 自包含原则
每个阶段的文件不 import 上一阶段，而是内联（inline）必要的基础函数。原因：
1. 每个文件可以独立运行，无需关心 `sys.path`
2. 避免 Linter 的 Unresolved Reference 误报
3. 阅读单个文件时不需要跳转查找依赖

### Mask 统一用 Bool
所有 Mask 张量统一使用 `bool` 类型：
- `torch.tril(torch.ones(...)).bool()` — Causal Mask
- `(seq != pad_idx).bool().unsqueeze(...)` — Padding Mask（`!= ` 已返回 bool，`.bool()` 是防御性写法）

原因：PyTorch 的 `&` 运算不支持 Float，统一 Bool 避免 `NotImplementedError`。

### 任务文件不依赖 03_variants
`04_tasks/` 下的每个任务文件自己实现了精简版模型（`MHA`、`FFN`、`EncLayer` 等），而不是导入 `03_variants`。原因：保持任务文件的可读性和独立性，读者无需来回跳转文件。

---

## 我自己的调适过程

### 阶段一：环境踩坑（已解决）

最初遇到两个环境问题：
1. 系统 Python（x86_64）与 numpy（arm64）架构不匹配
2. pip install 退出码 137（内存不足被 OOM Kill）

**解决方案**：项目根目录有现成的 `.venv/`，内含正确架构的 PyTorch 2.8.0，直接用 `.venv/bin/python` 绕过系统 Python 即可。

### 阶段二：Mask 类型 Bug

**现象**：运行 `masks.py` 时抛 `NotImplementedError: "bitwise_and_cpu" not implemented for 'Float'`

**定位**：`make_causal_mask` 中 `torch.tril(torch.ones(...))` 返回 Float，与 Bool 的 `padding_mask` 做 `&` 时类型冲突。

**修复**：全局搜索所有 `torch.tril(torch.ones(` 出现处，统一加 `.bool()`。

**反思**：这是一个在写代码时容易忽略的细节——Python 的 `bool & bool` 很自然，但 PyTorch 的按位运算要求类型严格匹配。

### 阶段三：翻译任务 Loss 不降

**现象**：翻译任务训练 30 个 epoch，Loss 始终 ≈ 2.77，等于随机猜测水平（`-log(1/14) ≈ 2.64`）。

**排查步骤**：
1. 先做单样本过拟合测试 → Loss 可以降到 0.0001，说明模型结构 OK
2. 换小数据集（200 样本）测试 → Loss 也能下降，说明数据集 OK
3. 回查 `train_epoch` → 发现 `optimizer.step()` **从未被调用**！

**根因**：在引入 `_NoamWrapper` 包装类时，为了让 `train_epoch` 保持接口不变，把 `scheduler.step()` 变成了 `pass`，但忘记在 `train_epoch` 里单独调用 `scheduler.optimizer.step()`。

**教训**：参数完全没有更新时，Loss 表现为在随机初始化水平附近随机抖动，而不是线性高位——这是个很典型的"loss 不动"症状，以后遇到先检查 `optimizer.step()` 是否被调用。

### 阶段四：Noam Scheduler 不适合小数据集

**现象**：即使修复了 `optimizer.step()` 后，用 Noam Scheduler 仍然收敛很慢。

**分析**：`d_model=64` 时 Noam 峰值 lr = `64^(-0.5) / sqrt(200) ≈ 8.8e-4`，比 `AdamW` 默认 lr `1e-3` 更低，且 Warmup 期间 lr 更低，对 10k 样本的小数据集来说步数太少撑不过 Warmup。

**优化**：翻译任务改用 `AdamW(lr=1e-3) + CosineAnnealingLR`，保留 `NoamScheduler` 类作为教学展示。

**注意**：Noam Scheduler 在原论文（WMT 百万级数据，d_model=512）中表现很好，它更适合大数据 + 大模型。

### 阶段五：import 顺序规范化

用户通过编辑器自动格式化，将所有文件的 import 顺序调整为 PEP 8 标准：
1. 标准库（`import math`、`import os`）
2. 第三方库（`import torch`、`import matplotlib`）
3. 本地库（`from encoder import Encoder`）

各组之间加空行。这是纯粹的风格优化，不影响功能。

---

## 常见问题 / 注意事项

### Q: 某个文件运行报 `ModuleNotFoundError`
确保在项目根目录下运行，或激活 `.venv`：
```bash
cd /Users/liulei/aiS/transformerS
.venv/bin/python 04_tasks/translation_task.py
```

### Q: `run_all.py` 报 `exec` 相关错误
`exec_file` 函数会把文件内容作为 `__main__` 执行，如果文件内部有复杂的相对导入可能出问题。优先单独运行各文件。

### Q: 翻译任务 Loss 卡住不动
检查 `train_epoch` 函数，确认 `optimizer.step()` 是否在 `loss.backward()` 之后被调用。

### Q: Mask 操作报 `NotImplementedError`
检查所有 `torch.tril(torch.ones(...))` 是否都加了 `.bool()`。

### Q: 在 Jupyter 中运行，图表不显示
将 `visualize.py` 顶部的 `matplotlib.use("Agg")` 改为注释掉，Jupyter 会自动用 `inline` 后端。

---

## 待优化方向（下次继续时参考）

1. **Beam Search 验证**：`transformer.py` 中实现了 Beam Search，但在测试时贪心解码就已经足够，Beam Search 路径未做完整验证
2. **visualize.py 集成到任务**：目前可视化工具独立，可以在训练任务中加入注意力热力图自动保存
3. **BPETokenizer 完整测试**：`tokenizer.py` 中的 BPE 实现是简化版，没有在实际任务中调用
4. **CUDA 测试**：所有验证都在 CPU 上完成，有 GPU 时可以调大模型（`d_model=512`，`num_layers=6`）测试性能
5. **真实数据集**：翻译任务可以换成 WMT de-en 等真实数据，分类任务可以换成 SST-2 等

---

## 项目文件索引（快速定位）

| 想找什么 | 去哪里看 |
|---------|---------|
| 位置编码原理 | `01_basics/01_embedding.py` |
| Mask 的完整实现 | `02_full_transformer/masks.py` |
| BERT 模型结构 | `03_variants/only_encoder.py` |
| GPT 文本生成 | `03_variants/only_decoder.py` 中的 `generate` 方法 |
| Teacher Forcing 代码 | `04_tasks/translation_task.py` 中的 `train_epoch` |
| 注意力可视化 | `utils/visualize.py` 中的 `plot_attention_heatmap` |
| BPE 分词器 | `utils/tokenizer.py` 中的 `BPETokenizer` |
| 整体运行入口 | `run_all.py` |

