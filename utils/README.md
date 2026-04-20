# utils — 工具模块

> 提供可视化和分词两类辅助工具，服务于课程各阶段的调试和理解。

---

## 文件说明

### `visualize.py` — 可视化工具

将注意力权重、位置编码等抽象向量可视化为直观图表，帮助理解模型内部工作机制。

| 函数 | 输入 | 输出 | 用途 |
|------|------|------|------|
| `plot_attention_heatmap` | 注意力权重张量 + token 列表 | 热力图 PNG | 观察某个头关注了哪些位置 |
| `plot_multi_head_attention` | 多头注意力权重 | 多子图热力图 | 对比不同头的注意力模式差异 |
| `plot_positional_encoding` | d_model, max_len | PE 矩阵热力图 | 直观理解 sin/cos 位置编码的规律 |
| `plot_training_curves` | loss/accuracy 列表 | 折线图 | 监控训练过程，判断是否收敛 |
| `plot_embedding_similarity` | 词向量矩阵 + 词列表 | 余弦相似度热力图 | 分析词向量空间的语义关系 |

**使用示例**：
```python
from utils.visualize import plot_attention_heatmap

# 假设 attention_weights 形状为 [num_heads, seq_len, seq_len]
plot_attention_heatmap(
    attention_weights[0],          # 第 0 个头
    src_tokens=["I", "love", "NLP"],
    tgt_tokens=["我", "爱", "NLP"],
    save_path="attention.png"
)
```

**后端说明**：默认使用 `matplotlib.use("Agg")` 非交互式后端，适合脚本和服务器环境。
在 Jupyter Notebook 中使用时，可将 `Agg` 改为 `inline`。

---

### `tokenizer.py` — 分词器

实现三种从简单到复杂的分词器，展示现代 NLP 分词技术的演进。

| 类 | 粒度 | 词表大小 | 典型用途 |
|----|------|---------|---------|
| `CharTokenizer` | 字符级 | ~128（ASCII）| 字符级 LM 演示，词表最小 |
| `WordTokenizer` | 词级 | 数万～数十万 | 经典 NLP，简单但 OOV 问题严重 |
| `BPETokenizer` | 子词级 | 3万～5万 | GPT/BERT 等大模型的实际做法 |

**通用接口**（三种分词器均实现）：

```python
tokenizer = CharTokenizer()           # 或 WordTokenizer / BPETokenizer

# 建立词表
tokenizer.build_vocab(text)           # 从文本学习词表
tokenizer.save("vocab.json")          # 保存
tokenizer.load("vocab.json")          # 加载

# 编码 / 解码
ids = tokenizer.encode("hello world") # str → List[int]
text = tokenizer.decode(ids)          # List[int] → str

# 批处理（自动 padding）
batch = tokenizer.encode_batch(texts, max_length=64)  # List[str] → Tensor
```

**BPE（Byte-Pair Encoding）工作原理**：
1. 初始词表 = 所有字符
2. 统计相邻 token 对的出现频率
3. 合并频率最高的 token 对，加入词表
4. 重复直到词表达到目标大小

**特殊 Token 约定**（四个文件统一）：

| Token | ID | 说明 |
|-------|----|------|
| `<pad>` | 0 | 填充位，注意力 Mask 会屏蔽 |
| `<bos>` | 1 | 序列开始，Decoder 生成的起点 |
| `<eos>` | 2 | 序列结束，生成时遇到则停止 |
| `<unk>` | 3 | 未知词，词表外的 token 映射到此 |
| `<mask>` | 4 | 掩码，BERT 预训练 MLM 任务用 |

---

### `__init__.py` — 包初始化

导出 `plot_attention_heatmap`、`CharTokenizer` 等常用符号。

---

## 运行方式

```bash
# 测试可视化（会在当前目录生成 PNG 图片）
python utils/visualize.py

# 测试分词器
python utils/tokenizer.py
```

---

## 与其他模块的关系

- `01_basics/01_embedding.py` 中的位置编码验证调用了 `visualize.py`
- `04_tasks/lm_task.py` 的字符级数据处理与 `CharTokenizer` 逻辑类似（任务文件内自包含）
- 分词器可独立用于其他 NLP 项目，与 Transformer 架构解耦

