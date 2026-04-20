"""
任务2: 字符级语言模型（Only Decoder / GPT 风格）
=================================================
使用 Only Decoder 训练一个字符级语言模型，学习生成文本。

任务说明：
  - 输入：字符序列（上下文）
  - 输出：预测下一个字符
  - 训练目标：最小化 Causal Language Modeling (CLM) 损失

特点：
  - 字符级（char-level）：词表只有所有可能的字符，约 128 个 ASCII
  - 无需分词器，直接操作字符
  - 训练后可以用前缀（prompt）生成文本

本示例使用莎士比亚文本（或自定义文本）训练字符级 LM。
数据：合成的字符序列，演示完整训练流程。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────
# 字符级数据集
# ─────────────────────────────────────────────
class CharDataset(Dataset):
    """
    字符级数据集：从文本中滑动窗口切割序列

    Example:
        文本: "hello world"
        seq_len=4:
            输入: "hell", 目标: "ello"
            输入: "ello", 目标: "llo "
            ...
    """

    def __init__(self, text: str, seq_len: int = 64):
        # 构建字符词表
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.char2idx = {c: i for i, c in enumerate(chars)}
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self.seq_len = seq_len

        # 编码全文
        self.data = torch.tensor(
            [self.char2idx[c] for c in text], dtype=torch.long
        )

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        # 输入：idx 到 idx+seq_len
        # 目标：idx+1 到 idx+seq_len+1（右移一位）
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y

    def encode(self, text: str) -> torch.Tensor:
        """字符串 → token ids"""
        return torch.tensor([self.char2idx.get(c, 0) for c in text], dtype=torch.long)

    def decode(self, ids) -> str:
        """token ids → 字符串"""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self.idx2char.get(i, "?") for i in ids)


# ─────────────────────────────────────────────
# GPT 风格 Only Decoder 模型（自包含）
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


class CausalSelfAttention(nn.Module):
    """因果自注意力（Masked Self-Attention）"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # 合并 QKV 到一个线性层（高效）
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, S, C = x.shape
        # 一次性计算 Q, K, V
        qkv = self.qkv_proj(x)  # [B, S, 3*C]
        Q, K, V = qkv.split(self.d_model, dim=-1)

        # 拆头
        def split(t):
            return t.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

        Q, K, V = split(Q), split(K), split(V)

        attn_out, _ = scaled_dot_product_attention(Q, K, V, mask=mask, dropout=self.attn_dropout)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, C)
        return self.resid_dropout(self.out_proj(attn_out))


class GPTBlock(nn.Module):
    """GPT 基础块：Masked Attn + FFN（Pre-LN）"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """
    字符级语言模型

    GPT 风格：可学习位置编码 + Pre-LN + Causal Mask
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: int = 512,
        seq_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            GPTBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)

        # 输出头（与 token embedding 共享权重）
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.tril(torch.ones(seq_len, seq_len, device=device)).bool().unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len]
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        B, S = x.shape
        assert S <= self.seq_len, f"序列长度 {S} 超过最大长度 {self.seq_len}"
        device = x.device

        pos = torch.arange(S, device=device).unsqueeze(0)
        h = self.drop(self.token_emb(x) + self.pos_emb(pos))

        causal_mask = self.make_causal_mask(S, device)
        for block in self.blocks:
            h = block(h, causal_mask)

        h = self.ln_final(h)
        return self.lm_head(h)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        """
        自回归生成文本

        Args:
            idx:            [1, prompt_len]
            max_new_tokens: 生成多少新字符
            temperature:    温度（>1更随机，<1更保守）
            top_k:          top-k 采样（0=不限制）
        """
        self.eval()
        for _ in range(max_new_tokens):
            # 截断到 seq_len
            idx_cond = idx[:, -self.seq_len:]
            logits = self.forward(idx_cond)
            # 取最后一步
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_char = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_char], dim=1)

        return idx


# ─────────────────────────────────────────────
# 训练
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)  # [B, S, V]
        # CLM loss：每个位置预测下一个字符
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def run_lm_task():
    print("=" * 60)
    print("任务2: 字符级语言模型（Only Decoder / GPT 风格）")
    print("=" * 60)

    # 合成训练文本（重复的字母序列 + 自定义片段）
    # 真实使用时换成 Shakespeare、代码、新闻等文本
    sample_text = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die, to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream. Ay, there's the rub,
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause. There's the respect
That makes calamity of so long life.
""" * 20  # 重复 20 次增加训练数据量

    SEQ_LEN = 64
    D_MODEL = 128
    NUM_HEADS = 4
    NUM_LAYERS = 4
    BATCH_SIZE = 32
    LR = 3e-4
    EPOCHS = 15

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")
    print(f"文本长度: {len(sample_text)} 字符")

    # 数据集
    dataset = CharDataset(sample_text, seq_len=SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    print(f"词表大小: {dataset.vocab_size} 字符")
    print(f"训练样本数: {len(dataset)}")
    print(f"词表: {''.join(sorted(dataset.char2idx.keys()))[:50]}...")

    # 模型
    model = MiniGPT(
        vocab_size=dataset.vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_MODEL * 4,
        seq_len=SEQ_LEN,
        dropout=0.1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.1)

    print(f"\n开始训练（{EPOCHS} epochs）...")
    print(f"{'Epoch':>6} {'Loss':>10} {'Perplexity':>12}")
    print("-" * 32)

    for epoch in range(1, EPOCHS + 1):
        loss, ppl = train_epoch(model, loader, optimizer, device)
        scheduler.step()

        if epoch % 3 == 0 or epoch == 1:
            # 每隔几轮生成一段文本看效果
            prompt_str = "To be, or"
            prompt_ids = dataset.encode(prompt_str).unsqueeze(0).to(device)
            generated = model.generate(prompt_ids, max_new_tokens=80, temperature=0.8, top_k=10)
            generated_text = dataset.decode(generated[0])
            print(f"\n  [Epoch {epoch}] 生成样本:")
            print(f"  >>> {generated_text}")
            print()

        print(f"{epoch:>6} {loss:>10.4f} {ppl:>12.2f}")

    print("\n✅ 语言模型任务完成！")
    return model, dataset


if __name__ == "__main__":
    run_lm_task()

