"""
任务1: 文本分类（Only Encoder / BERT 风格）
=============================================
使用 Only Encoder 做情感分析（Sentiment Analysis）。

任务说明：
  - 输入：文本序列（token ids）
  - 输出：类别预测（如：正面/负面/中性）
  - 原理：取 [CLS] token 的表示，接分类头

训练流程：
  1. 构造合成数据集（正面/负面文本）
  2. 构建 DataLoader
  3. 训练 Only Encoder 分类器
  4. 评估准确率

本示例使用合成数据，演示完整的训练循环，无需下载真实数据集。
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


def run_classification_task():
    print("=" * 60)
    print("任务1: 文本分类（Only Encoder / BERT 风格）")
    print("=" * 60)

    # 超参数
    VOCAB_SIZE = 500
    SEQ_LEN = 32
    D_MODEL = 64
    NUM_HEADS = 4
    NUM_LAYERS = 2
    NUM_CLASSES = 3
    BATCH_SIZE = 32
    LR = 3e-4
    EPOCHS = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")

    # 数据集
    train_dataset = SyntheticSentimentDataset(num_samples=2000, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE)
    val_dataset = SyntheticSentimentDataset(num_samples=500, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

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
    run_classification_task()

