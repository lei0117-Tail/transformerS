# 02_full_transformer — 完整 Transformer

> 本目录将 `01_basics` 中的六个构建块组装成一个完整的 Encoder-Decoder Transformer，
> 同时包含所有生产级的 Mask 工具和自回归解码逻辑。

---

## 文件说明

### `masks.py` — 所有 Mask 工具函数

Mask 是 Transformer 中最容易出错的部分。本文件集中管理所有 Mask 的生成逻辑。

| 函数 | 返回形状 | 用途 |
|------|---------|------|
| `make_padding_mask(seq, pad_idx)` | `[B, 1, 1, S]` | 屏蔽 `<pad>` 位置，用于 Encoder 自注意力 |
| `make_causal_mask(seq_len)` | `[1, 1, S, S]` | 下三角掩码，防止 Decoder 看到未来 token |
| `make_decoder_self_attn_mask(tgt)` | `[B, 1, T, T]` | Causal Mask ∧ Padding Mask，Decoder 自注意力专用 |
| `make_cross_attn_mask(src)` | `[B, 1, 1, S]` | Encoder-Decoder 交叉注意力用的 Padding Mask |

**Mask 形状设计原因**：形状中的 `1` 维是为了利用 PyTorch 广播机制，自动扩展到 `[B, num_heads, Q_len, K_len]`。

> **关键细节**：`torch.tril(torch.ones(...))` 返回 Float，必须加 `.bool()` 才能与 `(seq != pad_idx)` 做 `&` 运算。

---

### `encoder.py` — 完整 Transformer Encoder

将 N 个 `EncoderLayer` 堆叠，加上词嵌入和位置编码，构成完整 Encoder。

| 类 | 说明 |
|----|------|
| `Encoder` | 完整 Encoder：Embedding + N × EncoderLayer + LayerNorm |

**主要参数**：
- `vocab_size`：词表大小
- `d_model`：隐藏层维度（原论文 512）
- `num_heads`：注意力头数（原论文 8）
- `num_layers`：Encoder 层数 N（原论文 6）
- `d_ff`：FFN 中间层维度（原论文 2048）

**输入输出**：
```
输入：token ids [batch, src_len]
输出：上下文表示 [batch, src_len, d_model]
```

---

### `decoder.py` — 完整 Transformer Decoder

Decoder 比 Encoder 多一个**交叉注意力**子层，用于从 Encoder 输出中提取信息。

| 类 | 说明 |
|----|------|
| `DecoderLayer` | 单层 Decoder：MaskedMHA → Add&Norm → CrossMHA → Add&Norm → FFN → Add&Norm |
| `Decoder` | 完整 Decoder：Embedding + N × DecoderLayer + LayerNorm |

**数据流**：
```
目标序列 tgt ──→ Masked 自注意力 ──→
                                     → 交叉注意力（Q来自Decoder，K/V来自Encoder）→ FFN → 输出
Encoder 输出 enc_out ──────────────→
```

**推理时的 Mask**：
- 自注意力用 `make_decoder_self_attn_mask`（Causal + Padding）
- 交叉注意力用 `make_cross_attn_mask`（Encoder 的 Padding Mask）

---

### `transformer.py` — 完整 Encoder-Decoder Transformer

整合 Encoder 和 Decoder，提供训练和推理接口。

| 类/方法 | 说明 |
|---------|------|
| `Transformer` | 完整模型，包含 Encoder、Decoder、输出投影层 |
| `.forward(src, tgt)` | 训练模式：Teacher Forcing，返回 logits |
| `.encode(src)` | 只运行 Encoder，返回编码结果 |
| `.decode(tgt, enc_out, src_mask)` | 只运行 Decoder |
| `.greedy_decode(src, max_len)` | 贪心解码（推理） |
| `.beam_search(src, beam_size)` | Beam Search 解码（推理） |

**Teacher Forcing（训练时）**：
```
src: [I, love, NLP]
tgt_input:  [<bos>, 我, 爱]         ← Decoder 输入
tgt_target: [我, 爱, NLP]           ← 训练目标（向右移一位）
```

---

### `__init__.py` — 包初始化

导出 `Transformer`、`Encoder`、`Decoder` 等主要类，方便外部导入。

---

## 运行方式

```bash
# 按顺序验证
python 02_full_transformer/masks.py
python 02_full_transformer/encoder.py
python 02_full_transformer/decoder.py
python 02_full_transformer/transformer.py

# 或通过主入口
python run_all.py --step 2
```

---

## 与 01_basics 的关系

```
01_basics/06_encoder_layer.py
         ↓
02_full_transformer/encoder.py   (堆叠 N 层 EncoderLayer)
02_full_transformer/decoder.py   (新增交叉注意力)
02_full_transformer/transformer.py (组合 Encoder + Decoder)
```

`02_full_transformer` 中的各文件**自包含**（内联了必要的基础组件），不直接 import `01_basics`，方便独立运行和理解。

