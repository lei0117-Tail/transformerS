"""
任务3: 序列到序列翻译（Encoder-Decoder）
==========================================
使用完整 Encoder-Decoder 架构，训练一个数字→单词的"翻译"任务。

任务说明：
  - 输入（源语言）：数字序列，如 [3, 1, 4, 1, 5]
  - 输出（目标语言）：对应的英文单词序列，如 "three one four one five"
  - 简化了真实翻译，但完整演示了 Encoder-Decoder 的训练/推理流程

特殊 token：
  - <pad> = 0    ：填充
  - <bos> = 1    ：序列开始（Begin of Sequence）
  - <eos> = 2    ：序列结束（End of Sequence）
  - <unk> = 3    ：未知词

训练技巧演示：
  - Teacher Forcing（训练时用真实目标序列）
  - 学习率 Warmup（原论文的 Noam Scheduler）
  - Label Smoothing（防止模型过度自信）
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────
# 数字→单词 合成翻译数据集
# ─────────────────────────────────────────────

# 简单的数字词汇表
NUM2WORD = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
}

# 特殊 token
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

# 源语言词表（数字 0-9，offset by 4 for special tokens）
SRC_VOCAB_SIZE = 14  # 4 special + 10 digits

# 目标语言词表
TGT_WORDS = ["<pad>", "<bos>", "<eos>", "<unk>"] + list(NUM2WORD.values())
TGT_WORD2IDX = {w: i for i, w in enumerate(TGT_WORDS)}
TGT_IDX2WORD = {i: w for w, i in TGT_WORD2IDX.items()}
TGT_VOCAB_SIZE = len(TGT_WORDS)


class NumberToWordDataset(Dataset):
    """
    合成翻译数据集：
      src: 数字序列 [3, 1, 4] → token ids [7, 5, 8]（数字+4偏移）
      tgt: 单词序列 <bos> three one four <eos>
    """

    def __init__(self, num_samples: int = 5000, max_len: int = 6):
        self.samples = []
        for _ in range(num_samples):
            length = random.randint(2, max_len)
            numbers = [random.randint(0, 9) for _ in range(length)]
            self.samples.append(numbers)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        numbers = self.samples[idx]
        # 源序列：数字 id（+4 偏移，保留特殊 token）
        src = [n + 4 for n in numbers]
        # 目标序列：<bos> + 单词 id + <eos>
        tgt = [BOS_IDX] + [TGT_WORD2IDX[NUM2WORD[n]] for n in numbers] + [EOS_IDX]
        return src, tgt


def collate_fn(batch):
    """
    Batch 内序列长度可能不同，需要 padding 到相同长度
    """
    src_list, tgt_list = zip(*batch)

    # Padding 到 batch 内最长序列
    src_max = max(len(s) for s in src_list)
    tgt_max = max(len(t) for t in tgt_list)

    src_padded = torch.zeros(len(src_list), src_max, dtype=torch.long)
    tgt_padded = torch.zeros(len(tgt_list), tgt_max, dtype=torch.long)

    for i, (src, tgt) in enumerate(zip(src_list, tgt_list)):
        src_padded[i, :len(src)] = torch.tensor(src)
        tgt_padded[i, :len(tgt)] = torch.tensor(tgt)

    return src_padded, tgt_padded


# ─────────────────────────────────────────────
# Encoder-Decoder 模型（自包含）
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


class MHA(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.weights = None

    def _split(self, x):
        B, S, _ = x.shape
        return x.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

    def _merge(self, x):
        B, _, S, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, S, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q, K, V = self.W_Q(Q), self.W_K(K), self.W_V(V)
        Q, K, V = self._split(Q), self._split(K), self._split(V)
        out, self.weights = scaled_dot_product_attention(Q, K, V, mask, self.dropout)
        return self.W_O(self._merge(out))


class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EncLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()
        self.attn = MHA(d_model, num_heads, dropout)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        r = x; x = self.n1(x)
        x = r + self.drop(self.attn(x, x, x, mask))
        r = x; x = self.n2(x)
        x = r + self.drop(self.ffn(x))
        return x


class DecLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()
        self.self_attn = MHA(d_model, num_heads, dropout)
        self.cross_attn = MHA(d_model, num_heads, dropout)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.n3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, tgt, enc_out, tgt_mask=None, src_mask=None):
        r = tgt; tgt = self.n1(tgt)
        tgt = r + self.drop(self.self_attn(tgt, tgt, tgt, tgt_mask))
        r = tgt; tgt = self.n2(tgt)
        tgt = r + self.drop(self.cross_attn(tgt, enc_out, enc_out, src_mask))
        r = tgt; tgt = self.n3(tgt)
        tgt = r + self.drop(self.ffn(tgt))
        return tgt


class Seq2SeqTransformer(nn.Module):
    """
    完整 Encoder-Decoder Transformer（用于翻译任务）
    """

    def __init__(self, src_vocab, tgt_vocab, d_model=64, num_heads=4, num_layers=2, d_ff=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.src_emb = nn.Embedding(src_vocab, d_model, padding_idx=PAD_IDX)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=PAD_IDX)

        # 正弦位置编码
        pe = torch.zeros(200, d_model)
        pos = torch.arange(0, 200).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

        self.enc_layers = nn.ModuleList([EncLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dec_layers = nn.ModuleList([DecLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.enc_norm = nn.LayerNorm(d_model)
        self.dec_norm = nn.LayerNorm(d_model)

        self.proj = nn.Linear(d_model, tgt_vocab)

    def _emb(self, x, emb):
        e = emb(x) * math.sqrt(self.d_model)
        e = e + self.pe[:, :x.size(1)]
        return e

    def _src_mask(self, src):
        return (src != PAD_IDX).unsqueeze(1).unsqueeze(2)

    def _tgt_mask(self, tgt):
        L = tgt.size(1)
        causal = torch.tril(torch.ones(L, L, device=tgt.device)).bool().unsqueeze(0).unsqueeze(0)
        pad = (tgt != PAD_IDX).unsqueeze(1).unsqueeze(2)
        return causal & pad

    def encode(self, src):
        src_mask = self._src_mask(src)
        x = self._emb(src, self.src_emb)
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        return self.enc_norm(x), src_mask

    def decode(self, tgt, enc_out, src_mask):
        tgt_mask = self._tgt_mask(tgt)
        x = self._emb(tgt, self.tgt_emb)
        for layer in self.dec_layers:
            x = layer(x, enc_out, tgt_mask=tgt_mask, src_mask=src_mask)
        return self.dec_norm(x)

    def forward(self, src, tgt):
        """Teacher Forcing 前向"""
        enc_out, src_mask = self.encode(src)
        dec_out = self.decode(tgt, enc_out, src_mask)
        return self.proj(dec_out)

    @torch.no_grad()
    def translate(self, src, max_len=20):
        """
        贪心解码翻译

        Args:
            src: [1, src_len]
        Returns:
            list of token ids（不含 <bos>）
        """
        self.eval()
        device = next(self.parameters()).device
        src = src.to(device)

        enc_out, src_mask = self.encode(src)
        dec_input = torch.tensor([[BOS_IDX]], dtype=torch.long, device=device)
        result = []

        for _ in range(max_len):
            dec_out = self.decode(dec_input, enc_out, src_mask)
            logits = self.proj(dec_out[:, -1, :])
            next_id = logits.argmax(-1).item()
            if next_id == EOS_IDX:
                break
            result.append(next_id)
            dec_input = torch.cat([dec_input, torch.tensor([[next_id]], device=device)], dim=1)

        return result


# ─────────────────────────────────────────────
# Noam Warmup Scheduler（原论文）
# ─────────────────────────────────────────────

class NoamScheduler:
    """
    原论文使用的学习率调度：
    lr = d_model^(-0.5) × min(step^(-0.5), step × warmup_steps^(-1.5))

    特点：先线性上升到峰值，然后按 1/√step 下降
    """

    def __init__(self, optimizer, d_model: int, warmup_steps: int = 400):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * self.warmup_steps ** -1.5
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


# ─────────────────────────────────────────────
# Label Smoothing Loss
# ─────────────────────────────────────────────
class LabelSmoothingLoss(nn.Module):
    """
    标签平滑：将 one-hot 目标分布平滑为软分布
    防止模型对预测过度自信（提升泛化）

    例：smoothing=0.1，真实类别概率从 1.0 → 0.9，
    剩余 0.1 均匀分配给其他类别
    """

    def __init__(self, vocab_size: int, smoothing: float = 0.1, ignore_index: int = PAD_IDX):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [N, vocab_size]
            targets: [N]
        """
        log_probs = F.log_softmax(logits, dim=-1)

        # 构造软目标分布
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
            smooth_targets.scatter_(-1, targets.unsqueeze(-1), 1.0 - self.smoothing)
            smooth_targets[:, self.ignore_index] = 0

        # 忽略 padding 位置
        mask = (targets != self.ignore_index).float()
        loss = -(smooth_targets * log_probs).sum(dim=-1) * mask
        return loss.sum() / mask.sum()


# ─────────────────────────────────────────────
# 训练
# ─────────────────────────────────────────────

def train_epoch(model, loader, criterion, scheduler, device):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)

        # Teacher Forcing：Decoder 输入去掉最后一个 token，目标去掉第一个 token
        tgt_input = tgt[:, :-1]    # <bos> ... (去掉 <eos>)
        tgt_target = tgt[:, 1:]    # ... <eos> (去掉 <bos>)

        logits = model(src, tgt_input)  # [B, tgt_len-1, vocab]

        # 计算 loss
        loss = criterion(
            logits.contiguous().view(-1, logits.size(-1)),
            tgt_target.contiguous().view(-1),
        )

        scheduler.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scheduler.optimizer.step()
        scheduler.step()

        num_tokens = (tgt_target != PAD_IDX).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    return total_loss / total_tokens


@torch.no_grad()
def evaluate_accuracy(model, dataset, num_samples: int = 200, device: torch.device = None):
    """
    评估翻译准确率：序列完全匹配
    """
    model.eval()
    correct = 0
    for _ in range(num_samples):
        numbers = random.choice(dataset.samples)
        src = torch.tensor([[n + 4 for n in numbers]], dtype=torch.long)
        predicted_ids = model.translate(src)
        predicted_words = [TGT_IDX2WORD.get(i, "<unk>") for i in predicted_ids]
        expected_words = [NUM2WORD[n] for n in numbers]
        if predicted_words == expected_words:
            correct += 1
    return correct / num_samples


def run_translation_task():
    print("=" * 60)
    print("任务3: 序列到序列翻译（Encoder-Decoder）")
    print("=" * 60)
    print("任务：数字序列 → 英文单词序列")
    print("  输入: [3, 1, 4]  →  输出: three one four")

    D_MODEL = 64
    NUM_HEADS = 4
    NUM_LAYERS = 3
    BATCH_SIZE = 64
    LR = 1e-3
    EPOCHS = 30

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")
    print(f"源语言词表: {SRC_VOCAB_SIZE}, 目标语言词表: {TGT_VOCAB_SIZE}")
    print(f"目标词汇: {TGT_WORDS[4:]}")

    # 数据
    train_dataset = NumberToWordDataset(num_samples=10000, max_len=6)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, drop_last=True
    )

    # 模型
    model = Seq2SeqTransformer(
        src_vocab=SRC_VOCAB_SIZE,
        tgt_vocab=TGT_VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_MODEL * 4,
        dropout=0.1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 使用 AdamW + CosineAnnealing（小数据集更稳定）
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.05)
    # 用 Noam 类包装以复用 train_epoch 接口
    class _NoamWrapper:
        def __init__(self, opt, sched):
            self.optimizer = opt
            self._sched = sched
        def step(self):
            pass   # step 在 epoch 结束后调用
    scheduler = _NoamWrapper(optimizer, scheduler_cos)

    # 标准交叉熵（带 label smoothing，PyTorch 原生支持）
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)

    print(f"\n开始训练（{EPOCHS} epochs）...")
    print(f"{'Epoch':>6} {'Loss':>10} {'Trans Acc':>12}")
    print("-" * 35)

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, criterion, scheduler, device)
        scheduler_cos.step()

        if epoch % 5 == 0 or epoch == 1:
            acc = evaluate_accuracy(model, train_dataset, num_samples=300, device=device)
            print(f"{epoch:>6} {loss:>10.4f} {acc:>11.1%}")

            # 展示几个翻译样例
            test_cases = [[3, 1, 4], [9, 2, 7, 5], [0, 0, 1]]
            for nums in test_cases:
                src = torch.tensor([[n + 4 for n in nums]])
                pred_ids = model.translate(src)
                pred_words = [TGT_IDX2WORD.get(i, "<unk>") for i in pred_ids]
                true_words = [NUM2WORD[n] for n in nums]
                status = "✓" if pred_words == true_words else "✗"
                print(f"  {status} {nums} → {' '.join(pred_words)!r:30} (期望: {' '.join(true_words)!r})")
            print()
        else:
            print(f"{epoch:>6} {loss:>10.4f}")

    print("\n✅ 翻译任务完成！")
    return model


if __name__ == "__main__":
    run_translation_task()

