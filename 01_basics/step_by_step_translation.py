"""
端到端翻译演示：每一步都做了什么？（三句不同长度 Batch 版）
=============================================================

用三个**不同长度**的句子，完整展示 Transformer 翻译的每一个步骤：
  句子A (短): "五"           → "five"            (src_len=1,  tgt_len=2)
  句子B (中): "三 一 四"     → "three one four"   (src_len=3,  tgt_len=4)
  句子C (长): "二 五 三 一 四" → "two five three one four" (src_len=5, tgt_len=6)

关键亮点：
  - 三句话同时送入模型（batch=3），展示并行计算
  - 不同长度需要 Padding 到同一长度
  - Cross-Attention 中 src_len ≠ tgt_len 的真实场景

运行方式:
    python 01_basics/step_by_step_translation.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ══════════════════════════════════════════════
# 第0部分：定义模型组件（与原 translation_task.py 精简版一致）
# ══════════════════════════════════════════════

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

# ── 词表（扩展以支持更多词）──
SRC_WORD2IDX = {
    "<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3,
    "一": 4, "二": 5, "三": 6, "四": 7, "五": 8,
}
SRC_VOCAB_SIZE = len(SRC_WORD2IDX)
SRC_IDX2WORD = {v: k for k, v in SRC_WORD2IDX.items()}

TGT_WORD2IDX = {
    "<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3,
    "one": 4, "two": 5, "three": 6, "four": 7, "five": 8,
}
TGT_VOCAB_SIZE = len(TGT_WORD2IDX)
TGT_IDX2WORD = {v: k for k, v in TGT_WORD2IDX.items()}


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    return torch.matmul(weights, V), weights


class MHA(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, Q, K, V, mask=None):
        B, S_Q, _ = Q.shape
        _, S_K, _ = K.shape
        Q, K, V = self.W_Q(Q), self.W_K(K), self.W_V(V)
        Q = Q.view(B, S_Q, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, S_K, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, S_K, self.num_heads, self.d_k).transpose(1, 2)
        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(B, S_Q, -1)
        return self.W_O(out)


class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
    def forward(self, x):
        return self.net(x)


class EncLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = MHA(d_model, num_heads)
        self.ffn = FFN(d_model, d_ff)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        r = x; x = self.n1(x); x = r + self.attn(x, x, x, mask)
        r = x; x = self.n2(x); x = r + self.ffn(x)
        return x


class DecLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MHA(d_model, num_heads)
        self.cross_attn = MHA(d_model, num_heads)
        self.ffn = FFN(d_model, d_ff)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.n3 = nn.LayerNorm(d_model)

    def forward(self, tgt, enc_out, tgt_mask=None, src_mask=None):
        r = tgt; tgt = self.n1(tgt); tgt = r + self.self_attn(tgt, tgt, tgt, tgt_mask)
        r = tgt; tgt = self.n2(tgt); tgt = r + self.cross_attn(tgt, enc_out, enc_out, src_mask)
        r = tgt; tgt = self.n3(tgt); tgt = r + self.ffn(tgt)
        return tgt


class TransformerForTranslation(nn.Module):
    """完整 Encoder-Decoder Transformer"""

    def __init__(self, src_vocab, tgt_vocab, d_model=32, num_heads=4, num_layers=1, d_ff=64):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab, d_model, padding_idx=PAD_IDX)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=PAD_IDX)

        pe = torch.zeros(100, d_model)
        pos = torch.arange(0, 100).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

        self.enc_layer = EncLayer(d_model, num_heads, d_ff)
        self.dec_layer = DecLayer(d_model, num_heads, d_ff)
        self.enc_norm = nn.LayerNorm(d_model)
        self.dec_norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, tgt_vocab)

    def forward(self, src, tgt):
        # Mask
        src_mask = (src != PAD_IDX).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.size(1)
        causal = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool().unsqueeze(0).unsqueeze(0)
        pad_mask = (tgt != PAD_IDX).unsqueeze(1).unsqueeze(2)
        tgt_mask = causal & pad_mask

        # Encoder
        src_emb = self.src_emb(src) * math.sqrt(self.d_model) + self.pe[:, :src.size(1)]
        enc_out = self.enc_norm(self.enc_layer(src_emb, mask=src_mask))

        # Decoder
        tgt_emb = self.tgt_emb(tgt) * math.sqrt(self.d_model) + self.pe[:, :tgt.size(1)]
        dec_out = self.dec_norm(self.dec_layer(tgt_emb, enc_out, tgt_mask=tgt_mask, src_mask=src_mask))

        logits = self.proj(dec_out)
        return logits, {"enc_out": enc_out, "dec_out": dec_out,
                        "src_emb": src_emb, "tgt_emb": tgt_emb,
                        "src_mask": src_mask, "tgt_mask": tgt_mask}


# ══════════════════════════════════════════════
# 第1部分：三个句子的数据定义
# ══════════════════════════════════════════════

# ── 三个例子（不同长度！）──
EXAMPLES = [
    {
        "name": "句子A (短)",
        "src":  ["五"],
        "tgt":  ["<bos>", "five"],
    },
    {
        "name": "句子B (中)",
        "src":  ["三", "一", "四"],
        "tgt":  ["<bos>", "three", "one", "four"],
    },
    {
        "name": "句子C (长)",
        "src":  ["二", "五", "三", "一", "四"],
        "tgt":  ["<bos>", "two", "five", "three", "one", "four"],
    },
]


def build_batch():
    """将三个句子 padding 成一个 batch"""
    max_src_len = max(len(ex["src"]) for ex in EXAMPLES)
    max_tgt_len = max(len(ex["tgt"]) for ex in EXAMPLES)

    src_batch = []
    tgt_batch = []
    src_texts = []
    tgt_texts = []

    for ex in EXAMPLES:
        # 源语言: 填充 PAD 到 max_src_len
        src_ids = [SRC_WORD2IDX[w] for w in ex["src"]] + [PAD_IDX] * (max_src_len - len(ex["src"]))
        src_batch.append(src_ids)
        src_texts.append(ex["src"] + ["<pad>"] * (max_src_len - len(ex["src"])))

        # 目标语言: 填充 PAD 到 max_tgt_len
        tgt_ids = [TGT_WORD2IDX[w] for w in ex["tgt"]] + [PAD_IDX] * (max_tgt_len - len(ex["tgt"]))
        tgt_batch.append(tgt_ids)
        tgt_texts.append(ex["tgt"] + ["<pad>"] * (max_tgt_len - len(ex["tgt"])))

    return (
        torch.tensor(src_batch),                          # [3, max_src_len]
        torch.tensor(tgt_batch),                          # [3, max_tgt_len]
        src_texts, tgt_texts,
        max_src_len, max_tgt_len,
    )


# ══════════════════════════════════════════════
# 第2部分：主函数 —— 分步展示
# ══════════════════════════════════════════════

def main():
    D_MODEL = 32
    NUM_HEADS = 4
    D_K = D_MODEL // NUM_HEADS
    D_FF = 64

    # ── 标题 ──
    print("╔" + "═" * 78 + "╗")
    print("║" + "  Transformer 翻译任务：每一步都做了什么？（Batch=3, 不同长度）".center(78, " ") + "║")
    print("║" + "  句子A: '五' → 'five'     |  句子B: '三 一 四' → 'three one four'".center(76, " ") + " ║")
    print("║" + "  句子C: '二 五 三 一 四' → 'two five three one four'".center(76, " ") + " ║")
    print("╚" + "═" * 78 + "╝")

    print(f"\n📦 模型配置:  d_model={D_MODEL}, heads={NUM_HEADS} (d_k={D_K}), d_ff={D_FF}")

    model = TransformerForTranslation(
        src_vocab=SRC_VOCAB_SIZE, tgt_vocab=TGT_VOCAB_SIZE,
        d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF,
    )

    # ────────────────────────────────────────
    # 📍 步骤 0：输入数据 & Padding
    # ────────────────────────────────────────
    src_ids, tgt_ids, src_texts, tgt_texts, max_src_len, max_tgt_len = build_batch()

    print(f"\n{'━'*78}")
    print(f"  📍 步骤 0：输入数据 + Padding（关键！不同长度要补齐）")
    print(f"{'━'*78}")

    print(f"\n  ⚠️ Transformer 要求同一 batch 内张量形状相同，所以短的要补 <pad>")
    print(f"\n  {'批次':>4}  {'源句子':>22}  {'Token IDs':>24}  {'原始长度'}")
    print(f"  {'-'*70}")
    for i, ex in enumerate(EXAMPLES):
        raw_src = ' '.join(ex["src"])
        raw_tgt = ' '.join(ex["tgt"])
        print(f"  {i:>4}  {raw_src:>22}  {str(src_ids[i].tolist()):>24}  src_len={len(ex['src'])}")
    print(f"\n  Padding 后 batch 形状:")
    print(f"    src_ids: {src_ids.shape}  [batch=3, src_len={max_src_len}]")
    print(f"    tgt_ids: {tgt_ids.shape}  [batch=3, tgt_len={max_tgt_len}]")

    for i, ex in enumerate(EXAMPLES):
        print(f"\n    批次{i} ({ex['name']}):")
        print(f"      源: {src_texts[i]}  ids={src_ids[i].tolist()}")
        print(f"      目: {tgt_texts[i]}  ids={tgt_ids[i].tolist()}")

    # ────────────────────────────────────────
    # 📍 步骤 1：Embedding（含位置编码）
    # ────────────────────────────────────────
    print(f"\n{'━'*78}")
    print(f"  📍 步骤 1：Embedding（查表 ×√d_model + 位置编码）")
    print(f"{'━'*78}")

    src_emb = model.src_emb(src_ids) * math.sqrt(D_MODEL) + model.pe[:, :max_src_len]
    tgt_emb = model.tgt_emb(tgt_ids) * math.sqrt(D_MODEL) + model.pe[:, :max_tgt_len]

    print(f"\n  源语言 Embedding:")
    print(f"    输入: {src_ids.shape}  [batch, src_len]")
    print(f"    Embedding 权重: {model.src_emb.weight.shape}  [vocab={SRC_VOCAB_SIZE}, d_model={D_MODEL}]")
    print(f"    输出: {src_emb.shape}  [batch, src_len, d_model]")
    print(f"\n  目标语言 Embedding:")
    print(f"    输入: {tgt_ids.shape}  [batch, tgt_len]")
    print(f"    输出: {tgt_emb.shape}  [batch, tgt_len, d_model]")

    print(f"\n  🔍 每个样本每个位置的向量（前4维）:")
    for b in range(3):
        print(f"\n    ── 批次{b} ({EXAMPLES[b]['name']}) ──")
        print(f"    {'位置':>4} {'源词':>6} {'向量(前4维)':>30}  |  {'目标词':>8} {'向量(前4维)':>30}")
        for s in range(max_src_len):
            svec = src_emb[b, s, :4].detach().numpy().round(3)
            sw = src_texts[b][s]
            tw = tgt_texts[b][s] if s < len(tgt_texts[b]) else "-"
            tvec = tgt_emb[b, s, :4].detach().numpy().round(3) if s < max_tgt_len else None
            tcol = f"{str(tvec.tolist()):>30}" if tvec is not None else " "*30
            pad_mark = " ← <pad>" if sw == "<pad>" else ""
            print(f"    {s:>4} {sw:>6} {str(svec.tolist()):>30}{pad_mark}  |  {tw:>8} {tcol}")

    # ────────────────────────────────────────
    # 📍 步骤 2：Encoder Self-Attention
    # ────────────────────────────────────────
    print(f"\n{'━'*78}")
    print(f"  📍 步骤 2：Encoder 自注意力（Self-Attention）—— 双向关注！")
    print(f"{'━'*78}")

    src_mask = (src_ids != PAD_IDX).unsqueeze(1).unsqueeze(2)
    print(f"\n  Padding Mask（告诉模型哪些是真正的词，哪些是填充的）:")
    print(f"    shape: {src_mask.shape}  [batch, 1, 1, src_len]")
    for b in range(3):
        mask_vals = src_mask[b, 0, 0].int().tolist()
        words = src_texts[b]
        mask_str = "  ".join([f"{'✓':>3}" if v else f"{'✗':>3}" for v in mask_vals])
        word_str = "  ".join([f"{w:>3}" for w in words])
        print(f"    批次{b}: [{word_str}]")
        print(f"           [{mask_str}]  (✓=看得到/参与计算, ✗=<pad>被屏蔽)")

    # 手动算注意力展示（用批次1=句子B，因为它长度适中）
    demo_b = 1  # 用句子B做详细演示
    demo_src_len = len(EXAMPLES[demo_b]["src"])
    W_Q = model.enc_layer.attn.W_Q
    W_K = model.enc_layer.attn.W_K

    Q_raw = W_Q(src_emb[demo_b:demo_b+1, :demo_src_len])  # [1, 3, 32]
    K_raw = W_K(src_emb[demo_b:demo_b+1, :demo_src_len])
    Q_multi = Q_raw.view(1, demo_src_len, NUM_HEADS, D_K).transpose(1, 2)
    K_multi = K_raw.view(1, demo_src_len, NUM_HEADS, D_K).transpose(1, 2)
    scores = torch.matmul(Q_multi, K_multi.transpose(-2, -1)) / math.sqrt(D_K)
    attn_weights = F.softmax(scores, dim=-1)

    print(f"\n  🔍 以「{EXAMPLES[demo_b]['name']}」为例，Head 0 的 Encoder Self-Attention 权重:")
    w = attn_weights[0, 0].detach().numpy().round(3)
    demo_words = EXAMPLES[demo_b]["src"]
    header = f"           {'':>8}" + "".join([f" {w_:>10}" for w_ in demo_words])
    print(header)
    for i, qi in enumerate(demo_words):
        row = f"        {qi:>8}"
        for j in range(demo_src_len):
            row += f" {w[i,j]:>10.3f}"
        print(row)
    print(f"    → Encoder 是双向的！每个词都能看到句子中所有其他词（包括后面的）")

    # 完整 Encoder 输出
    enc_out_padded = model.enc_layer(src_emb, mask=src_mask)
    enc_out = model.enc_norm(enc_out_padded)
    print(f"\n  ✅ Encoder 输出: {enc_out.shape}  [batch, src_len={max_src_len}, d_model={D_MODEL}]")
    print(f"    （注意：<pad> 位置的输出无意义，后续会被 mask 屏蔽掉）")

    # ────────────────────────────────────────
    # 📍 步骤 3：Decoder Masked Self-Attention
    # ────────────────────────────────────────
    print(f"\n{'━'*78}")
    print(f"  📍 步骤 3：Decoder Masked Self-Attention —— 只能看过去！")
    print(f"{'━'*78}")

    tgt_len_demo = len(EXAMPLES[demo_b]["tgt"])
    causal = torch.tril(torch.ones(tgt_len_demo, tgt_len_demo)).bool().unsqueeze(0).unsqueeze(0)

    print(f"\n  ⚠️ Causal Mask（下三角）：Decoder 生成时不能偷看未来！")
    cm = causal[0, 0].int()
    demo_tgt_words = EXAMPLES[demo_b]["tgt"]
    header = f"           {'':>10}" + "".join(f" {w:>10}" for w in demo_tgt_words)
    print(header)
    for i, wi in enumerate(demo_tgt_words):
        row = f"        {wi:>10}"
        for j in range(tgt_len_demo):
            row += f" {cm[i,j]:>10d}"
        print(row)
    print(f"    上三角全是 0 → 位置 i 只能看到位置 0~i")

    # Decoder Self-Attn 权重
    dec_self_Q = model.dec_layer.self_attn.W_Q(tgt_emb[demo_b:demo_b+1, :tgt_len_demo])
    dec_self_K = model.dec_layer.self_attn.W_K(tgt_emb[demo_b:demo_b+1, :tgt_len_demo])
    dec_Q = dec_self_Q.view(1, tgt_len_demo, NUM_HEADS, D_K).transpose(1, 2)
    dec_K = dec_self_K.view(1, tgt_len_demo, NUM_HEADS, D_K).transpose(1, 2)
    dec_scores = torch.matmul(dec_Q, dec_K.transpose(-2, -1)) / math.sqrt(D_K)
    dec_scores_masked = dec_scores.masked_fill(~causal, float("-inf"))
    dec_weights = F.softmax(dec_scores_masked, dim=-1)

    print(f"\n  🔍 Decoder Self-Attention 权重（Head 0，已应用 Causal Mask）:")
    dw = dec_weights[0, 0].detach().numpy().round(3)
    header = f"           {'':>10}" + "".join(f" {w:>10}" for w in demo_tgt_words)
    print(header)
    for i, wi in enumerate(demo_tgt_words):
        row = f"        {wi:>10}"
        for j in range(tgt_len_demo):
            val = dw[i, j]
            marker = "" if j <= i else " ✗"
            row += f" {val:>9.3f}{marker}"
        print(row)

    # ────────────────────────────────────────
    # 📍 步骤 4：Cross-Attention（核心亮点！）
    # ────────────────────────────────────────
    print(f"\n{'━'*78}")
    print(f"  📍 步骤 4：Cross-Attention —— Q来自Decoder，K/V来自Encoder")
    print(f"{'━'*78}")

    print(f"\n  ⭐ 这是 Encoder-Decoder 最核心的操作！")
    print(f"     Q = Decoder 当前状态（我已经生成了什么？下一步该生成啥？）")
    print(f"     K = Encoder 输出（原文到底说了哪些词？）")
    print(f"     V = Encoder 输出（原文的信息内容）")
    print(f"\n  💡 关键维度变化:")

    for b_idx, ex in enumerate(EXAMPLES):
        real_src_len = len(ex["src"])
        real_tgt_len = len(ex["tgt"])
        print(f"\n    批次{b_idx} ({ex['name']}):")
        print(f"      Q 来自 Decoder: shape=[1, {real_tgt_len}, {D_MODEL}]  （{real_tgt_len}个目标位置）")
        print(f"      K 来自 Encoder: shape=[1, {real_src_len}, {D_MODEL}]  （{real_src_len}个源词）")
        print(f"      V 来自 Encoder: shape=[1, {real_src_len}, {D_MODEL}]")
        print(f"      → Attention 矩阵: [{NUM_HEADS}, {real_tgt_len}, {real_src_len}]")
        print(f"         （每一行=Decoder的一个位置，每一列=Encoder的一个源词）")

    # 展示三个批次的 Cross-Attn
    print(f"\n  🔍 三个句子的 Cross-Attention 权重对比（Head 0）:")
    for b_idx, ex in enumerate(EXAMPLES):
        real_src_len = len(ex["src"])
        real_tgt_len = len(ex["tgt"])

        cQ = model.dec_layer.cross_attn.W_Q(tgt_emb[b_idx:b_idx+1, :real_tgt_len])
        cK = model.dec_layer.cross_attn.W_K(enc_out[b_idx:b_idx+1, :real_src_len])
        cV = model.dec_layer.cross_attn.W_V(enc_out[b_idx:b_idx+1, :real_src_len])

        cQ_m = cQ.view(1, real_tgt_len, NUM_HEADS, D_K).transpose(1, 2)
        cK_m = cK.view(1, real_src_len, NUM_HEADS, D_K).transpose(1, 2)

        cscores = torch.matmul(cQ_m, cK_m.transpose(-2, -1)) / math.sqrt(D_K)
        cweights = F.softmax(cscores, dim=-1)
        cw = cweights[0, 0].detach().numpy().round(3)

        src_w = ex["src"]
        tgt_w = ex["tgt"]

        print(f"\n    ━━ {ex['name']} (src_len={real_src_len}, tgt_len={real_tgt_len}) ━━")
        header = f"             {'':>12}" + "".join(f" {s:>12}(Enc)" for s in src_w)
        print(header)
        for ti, tw in enumerate(tgt_w):
            row = f"        {tw:>12}"
            for si in range(real_src_len):
                row += f" {cw[ti,si]:>12.3f}"
            print(row)

    # ────────────────────────────────────────
    # 📍 步骤 5：输出投影 → 预测概率
    # ────────────────────────────────────────
    print(f"\n{'━'*78}")
    print(f"  📍 步骤 5：输出投影 → 预测每个位置的下一个词")
    print(f"{'━'*78}")

    logits, info = model(src_ids, tgt_ids)
    probs = F.softmax(logits, dim=-1)
    preds = logits.argmax(dim=-1)

    print(f"\n  Decoder 输出: {info['dec_out'].shape}  [batch, tgt_len, d_model]")
    print(f"  投影层: Linear({D_MODEL} → vocab={TGT_VOCAB_SIZE})")
    print(f"  Logits:  {logits.shape}  [batch, tgt_len, vocab_size]\n")

    for b_idx, ex in enumerate(EXAMPLES):
        real_tgt_len = len(ex["tgt"])
        print(f"  ┌─ {ex['name']} ─────────────────────────────────────────┐")
        print(f"  {'位置':>4} {'输入':>10} {'应预测':>8} {'实际预测':>10} {'Top-3候选'}")
        print(f"  {'-'*62}")
        for i in range(real_tgt_len):
            inp_word = tgt_texts[b_idx][i]
            # 应预测的答案 = 下一位置的目标词（或 EOS）
            expected = ex["tgt"][i+1] if i+1 < len(ex["tgt"]) else "(结束)"
            pred_id = preds[b_idx, i].item()
            pred_word = TGT_IDX2WORD.get(pred_id, "?")
            top3_p, top3_id = probs[b_idx, i].topk(3)
            top3 = [f"{TGT_IDX2WORD[idx.item()]}({p.item():.1%})" for p, idx in zip(top3_p, top3_id)]
            mark = " ✓" if pred_word == expected or expected == "(结束)" else ""
            print(f"  {i:>4} {inp_word:>10} {expected:>8} {pred_word:>10}{mark}  {', '.join(top3)}")
        print(f"  └────────────────────────────────────────────────────┘")

    print(f"\n  💡 训练目标（Teacher Forcing）:")
    print(f"     每个位置输入当前词 → 预测下一个词")
    print(f"     例如句子B: <bos>→three, three→one, one→four, four→<eos>")
    print(f"  （注：未训练的随机模型，预测不准。训练后正确率→100%）")

    # ────────────────────────────────────────
    # 总结：完整数据流一览
    # ────────────────────────────────────────
    print(f"\n{'╔' + '═'*78 + '╗'}")
    print(f"║{'  📊 完整数据流总结（Batch=3, 不同长度）':^78}║")
    print(f"{'╚' + '═'*78 + '╝'}")
    print(f"""
  ┌──────────────────────────────────────────────────────────────────────┐
  │                    Transformer 翻译流程 (batch=3)                     │
  ├──────────────────────────────────────────────────────────────────────┤
  │                                                                      │
  │   句子A: "五"              句子B: "三 一 四"       句子C: "二五三一四"│
  │    ↓ Token IDs              ↓ Token IDs              ↓ Token IDs       │
  │   [8]                      [6,4,7]                  [5,8,6,4,7]       │
  │    ↓ Pad to 5               ↓ Pad to 5                ↓ Pad to 5        │
  │   [8,0,0,0,0]              [6,4,7,0,0]              [5,8,6,4,7]       │
  │    ↓ Embedding               ↓ Embedding               ↓ Embedding       │
  │  [1,5,{D_MODEL}]             [1,5,{D_MODEL}]              [1,5,{D_MODEL}]         │
  │    ↓ + Position Encoding     ↓ + Position Encoding     ↓ + Position Enc  │
  │                            ↓                                ↓          │
  │  ┌──────────────────────────────────────────────────────────────┐    │
  │  │  ENCODER (共享权重，3个样本并行计算)                           │    │
  │  │   Self-Attention (双向) + FFN + LayerNorm                    │    │
  │  │   → 输出: [3, 5, {D_MODEL}]  每个位置融合了全局信息              │    │
  │  └──────────────────────────┬───────────────────────────────────┘    │
  │                             │                                        │
  │  ┌──────────────────────────────────────────────────────────────┐    │
  │  │  DECODER (共享权重，3个样本并行计算)                           │    │
  │  │                                                              │    │
  │  │  ① Masked Self-Attn (单向! Causal Mask)                       │    │
  │  │  ② Cross-Attn: Q=Dec状态, K/V=Enc输出  ← 关键桥梁！          │    │
  │  │  ③ FFN + LayerNorm                                          │    │
  │  └──────────────────────────┬───────────────────────────────────┘    │
  │                             ↓                                        │
  │                   Linear({D_MODEL}→vocab)                              │
  │                             ↓                                        │
  │              预测: ["five","three one four","two five three one four"]│
  │                                                                      │
  └──────────────────────────────────────────────────────────────────────┘

  🎯 核心要点:
    ① 不同长度 → Padding 对齐到同一 tensor（<pad>=0）
    ② Padding Mask → 让模型忽略 <pad> 位置（不参与注意力计算）
    ③ Causal Mask → Decoder 只能看过去（从左到右逐字生成）
    ④ Cross-Attention → src_len ≠ tgt_len 时矩阵不是方阵！（如句子A: 1×2）
    ⑤ 共享权重 → 3个样本完全并行，一次前向传播同时处理

  维度速查:
    d_model = {D_MODEL}    每个 token 向量长度
    d_k     = {D_K}       每个注意力头维度
    batch   = 3           同时处理 3 个句子
    src_len = 5 (padded)  源语言最大长度
    tgt_len = 6 (padded)  目标语言最大长度
""")


if __name__ == "__main__":
    main()

