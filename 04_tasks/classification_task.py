"""
任务1: 文本分类（Only Encoder / BERT 风格）
=============================================
使用 Only Encoder 做文本分类。

任务说明：
  - 输入：文本序列（token ids）
  - 输出：类别预测
  - 原理：取 [CLS] token 的表示，接分类头

数据集选项（通过 run_classification_task(use_real_data=True/False) 切换）：
  - 合成数据（默认）：随机生成，无需任何依赖，验证模型流程
  - 真实数据：sklearn 内置 20newsgroups（2类：科技 vs 体育），无需下载

真实数据使用方式：
    python 04_tasks/classification_task.py --real
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────
# 合成数据集
# ─────────────────────────────────────────────
class SyntheticSentimentDataset(Dataset):
    """
    合成情感分析数据集
    规则：
      - 类别0（负面）：包含 token 1-100（低值词）
      - 类别1（正面）：包含 token 201-300（高值词）
      - 类别2（中性）：混合

    这不是真实语言数据，但足以验证模型能否学习简单规律。
    """

    def __init__(self, num_samples: int = 1000, seq_len: int = 32, vocab_size: int = 500):
        self.data = []
        self.labels = []
        self.seq_len = seq_len

        for _ in range(num_samples):
            label = random.randint(0, 2)
            # 生成不同类别的特征 token
            if label == 0:
                # 负面：用更多低值 token
                key_tokens = [random.randint(10, 80) for _ in range(5)]
            elif label == 1:
                # 正面：用更多高值 token
                key_tokens = [random.randint(200, 280) for _ in range(5)]
            else:
                # 中性：混合
                key_tokens = [random.randint(100, 200) for _ in range(5)]

            # 填充随机 token，加入类别特征 token
            tokens = [1]  # [CLS]
            tokens += key_tokens
            tokens += [random.randint(1, vocab_size - 1) for _ in range(seq_len - 6)]
            tokens.append(2)  # [SEP]

            # 截断/补全到 seq_len
            tokens = tokens[:seq_len]
            while len(tokens) < seq_len:
                tokens.append(0)  # [PAD]

            self.data.append(tokens)
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ─────────────────────────────────────────────
# 真实数据集：20 Newsgroups（sklearn 内置，无需下载）
# ─────────────────────────────────────────────
class NewsGroupsDataset(Dataset):
    """
    使用 sklearn 内置的 20newsgroups 数据集做二分类：
      - 类别 0: rec.sport.baseball（体育）
      - 类别 1: sci.space（科技）

    分词方式：空格分词 + 词频词表（word-level），简单实现，无需外部分词器。

    Args:
        subset:     'train' 或 'test'
        max_len:    序列截断长度
        vocab_size: 词表大小（按词频取前 N 个）
        vocab:      复用已有词表（val/test 时传入 train 的词表）
    """

    def __init__(self, subset: str = "train", max_len: int = 128,
                 vocab_size: int = 8000, vocab: dict = None):
        from sklearn.datasets import fetch_20newsgroups

        categories = ["rec.sport.baseball", "sci.space"]
        data = fetch_20newsgroups(
            subset=subset,
            categories=categories,
            remove=("headers", "footers", "quotes"),  # 去掉元信息，防止泄露
        )

        self.texts = data.data
        self.labels = data.target.tolist()
        self.max_len = max_len

        # 建立词表（仅 train 时建，test/val 复用）
        if vocab is None:
            word_freq: dict = {}
            for text in self.texts:
                for w in text.lower().split():
                    w = w.strip(".,!?;:\"'()[]")
                    if w:
                        word_freq[w] = word_freq.get(w, 0) + 1
            # 保留最高频的 vocab_size-4 个词，前 4 位留给特殊 token
            top_words = sorted(word_freq, key=lambda x: -word_freq[x])[:vocab_size - 4]
            self.vocab = {"<pad>": 0, "<cls>": 1, "<sep>": 2, "<unk>": 3}
            for w in top_words:
                self.vocab[w] = len(self.vocab)
        else:
            self.vocab = vocab

        self.vocab_size = len(self.vocab)

    def _tokenize(self, text: str):
        """文本 → token id 列表，带 [CLS] 和截断"""
        tokens = [1]  # <cls>
        for w in text.lower().split():
            w = w.strip(".,!?;:\"'()[]")
            if w:
                tokens.append(self.vocab.get(w, 3))  # 3 = <unk>
        tokens = tokens[: self.max_len]
        # padding
        tokens += [0] * (self.max_len - len(tokens))
        return tokens

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = self._tokenize(self.texts[idx])
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ─────────────────────────────────────────────
# 模型（Only Encoder）
# ─────────────────────────────────────────────

def scaled_dot_product_attention(Q, K, V, mask=None, dropout=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    if dropout is not None:
        weights = dropout(weights)
    return torch.matmul(weights, V), weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x):
        B, S, _ = x.shape
        return x.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

    def _merge(self, x):
        B, _, S, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, S, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q, K, V = self.W_Q(Q), self.W_K(K), self.W_V(V)
        Q, K, V = self._split(Q), self._split(K), self._split(V)
        out, _ = scaled_dot_product_attention(Q, K, V, mask=mask, dropout=self.dropout)
        return self.W_O(self._merge(out))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn))
        ffn = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn))
        return x


class TextClassifier(nn.Module):
    """
    文本分类模型（Only Encoder）

    架构：
      Embedding → N×EncoderLayer → [CLS] pooling → 分类头
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        num_classes: int,
        d_ff: int = None,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        d_ff = d_ff or 4 * d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len]
        Returns:
            logits: [batch, num_classes]
        """
        B, S = input_ids.shape
        device = input_ids.device

        # Padding mask
        mask = (input_ids != self.pad_idx).unsqueeze(1).unsqueeze(2)

        # 嵌入
        pos_ids = torch.arange(S, device=device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.pos_embedding(pos_ids)
        x = self.emb_dropout(x)

        # Encoder 层
        for layer in self.layers:
            x = layer(x, mask=mask)
        x = self.norm(x)

        # 取 [CLS] token（位置 0）做分类
        cls_repr = x[:, 0, :]
        return self.classifier(cls_repr)


# ─────────────────────────────────────────────
# 训练 + 评估
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for input_ids, labels in loader:
        input_ids, labels = input_ids.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        # 梯度裁剪（防止梯度爆炸）
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for input_ids, labels in loader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        logits = model(input_ids)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += len(labels)
    return correct / total


def run_classification_task(use_real_data: bool = False):
    """
    Args:
        use_real_data: False → 合成数据（快，3 类，验证流程）
                       True  → 20newsgroups 真实文本（体育 vs 科技，2 类）
    """
    print("=" * 60)
    print("任务1: 文本分类（Only Encoder / BERT 风格）")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")

    if use_real_data:
        # ── 真实数据集：20newsgroups ──────────────────────────────
        print("数据集: 20newsgroups（体育 vs 科技，sklearn 内置）")
        SEQ_LEN    = 128
        D_MODEL    = 128
        NUM_HEADS  = 4
        NUM_LAYERS = 3
        NUM_CLASSES = 2
        BATCH_SIZE = 32
        LR         = 3e-4
        EPOCHS     = 15

        train_dataset = NewsGroupsDataset(subset="train", max_len=SEQ_LEN, vocab_size=8000)
        val_dataset   = NewsGroupsDataset(subset="test",  max_len=SEQ_LEN, vocab=train_dataset.vocab)
        VOCAB_SIZE = train_dataset.vocab_size
        print(f"词表大小: {VOCAB_SIZE}")
        print(f"训练样本: {len(train_dataset)},  测试样本: {len(val_dataset)}")
        print(f"类别: 0=rec.sport.baseball, 1=sci.space")
    else:
        # ── 合成数据（默认）──────────────────────────────────────
        print("数据集: 合成情感数据（三分类，快速验证）")
        VOCAB_SIZE = 500
        SEQ_LEN    = 32
        D_MODEL    = 64
        NUM_HEADS  = 4
        NUM_LAYERS = 2
        NUM_CLASSES = 3
        BATCH_SIZE  = 32
        LR          = 3e-4
        EPOCHS      = 10
        train_dataset = SyntheticSentimentDataset(num_samples=2000, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE)
        val_dataset   = SyntheticSentimentDataset(num_samples=500,  seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

    # 模型
    model = TextClassifier(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        max_len=SEQ_LEN,
        dropout=0.1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 打印数据流形状
    sample_input, sample_label = next(iter(train_loader))
    print(f"\n{'='*50}")
    print("数据流形状追踪:")
    print(f"{'='*50}")
    print(f"  输入 input_ids:  {sample_input.shape}   [batch={BATCH_SIZE}, seq_len={SEQ_LEN}]")
    print(f"  标签 labels:      {sample_label.shape}   [batch={BATCH_SIZE}] (类别 0~{NUM_CLASSES-1})")

    # 单个样本前向看形状变化
    model.eval()
    single_input = sample_input[:1].to(device)
    with torch.no_grad():
        emb = model.embedding(single_input) if hasattr(model, 'embedding') else None
        if emb is not None:
            print(f"  Embedding 后:     {emb.shape}       [1, seq_len, d_model={D_MODEL}]")
        logits_sample = model(single_input)
    print(f"  logits (输出):    {logits_sample.shape}   [1, num_classes={NUM_CLASSES}] ← 分类概率（未 softmax）")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\n开始训练（{EPOCHS} epochs）...")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} {'Val Acc':>10}")
    print("-" * 45)

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(f"{epoch:>6} {train_loss:>12.4f} {train_acc:>9.1%} {val_acc:>9.1%}")

    print(f"\n最佳验证准确率: {best_val_acc:.1%}")
    print("\n✅ 分类任务完成！")
    return model


if __name__ == "__main__":
    import sys
    use_real = "--real" in sys.argv
    run_classification_task(use_real_data=use_real)

