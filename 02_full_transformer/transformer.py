"""
完整 Transformer (Encoder-Decoder)
=====================================
将 Encoder 和 Decoder 组合，加上输出投影层，构成用于序列到序列任务的完整 Transformer。

论文：Attention Is All You Need (Vaswani et al., 2017)

整体架构：
  src (源序列)                      tgt (目标序列，右移一位)
       ↓                                    ↓
  [Encoder]                           [Decoder]
       ↓                                    ↓
  enc_output ─────────────────→  cross-attn K/V
                                          ↓
                                   [Linear + Softmax]
                                          ↓
                                   预测下一个 token 的概率分布

训练技巧 - Teacher Forcing：
  训练时 Decoder 的输入是真实目标序列（右移一位），
  不使用上一步的预测结果，可以加速收敛。

  例：目标 "I love NLP !"
    Decoder 输入:  <bos> I love NLP
    Decoder 目标:  I love NLP !

推理 - 自回归生成：
  逐步预测：每次用已生成的序列作为 Decoder 输入，
  预测下一个 token，直到生成 <eos>。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from decoder import Decoder  # noqa: E402
# ─────────────────────────────────────────────
# 复用 Encoder / Decoder
# ─────────────────────────────────────────────
from encoder import Encoder  # noqa: E402 (same package, relative import)


# ─────────────────────────────────────────────
# 完整 Transformer
# ─────────────────────────────────────────────
class Transformer(nn.Module):
    """
    Encoder-Decoder Transformer

    Args:
        src_vocab_size: 源语言词表大小
        tgt_vocab_size: 目标语言词表大小
        d_model:        模型维度（默认 512）
        num_heads:      注意力头数（默认 8）
        num_layers:     Encoder/Decoder 层数（默认 6）
        d_ff:           FFN 中间层维度（默认 2048）
        max_len:        最大序列长度（默认 5000）
        dropout:        dropout 概率（默认 0.1）
        pad_idx:        padding token id（默认 0）

    典型配置（原论文 base）：
        d_model=512, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
        pad_idx: int = 0,
        share_embedding: bool = False,  # 是否共享 src/tgt embedding（同语言时可用）
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        # Encoder
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
            pad_idx=pad_idx,
        )

        # Decoder
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
            pad_idx=pad_idx,
        )

        # 输出投影：d_model → tgt_vocab_size
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # 可选：共享 embedding 权重（论文 Section 3.4）
        if share_embedding:
            assert src_vocab_size == tgt_vocab_size, "共享 embedding 要求 src/tgt 词表大小相同"
            self.decoder.token_embedding.weight = self.encoder.token_embedding.weight

        # 输出投影也与 embedding 共享（减少参数，提升效果）
        # 参考：Press & Wolf, 2017 "Using the Output Embedding to Improve Language Models"
        self.output_projection.weight = self.decoder.token_embedding.weight

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        单独跑 Encoder（推理时可缓存，避免重复计算）

        Args:
            src: [batch, src_len]
        Returns:
            [batch, src_len, d_model]
        """
        return self.encoder(src, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        enc_output: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        src_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        单独跑 Decoder

        Args:
            tgt:        [batch, tgt_len]
            enc_output: [batch, src_len, d_model]
        Returns:
            [batch, tgt_len, d_model]
        """
        return self.decoder(tgt, enc_output, tgt_mask, src_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        训练时的前向传播（Teacher Forcing）

        Args:
            src: [batch, src_len]  — 源序列
            tgt: [batch, tgt_len]  — 目标序列（右移，以 <bos> 开头）

        Returns:
            logits: [batch, tgt_len, tgt_vocab_size]  — 未归一化的预测概率

        使用示例：
            # 翻译任务
            src = tokenize("Hello World")        # 源语言
            tgt = [<bos>] + tokenize("你好 世界") # 目标语言（右移）
            logits = model(src, tgt)
            # logits 对应 tokenize("你好 世界") + [<eos>]
        """
        # Step 1: Encoder
        if src_mask is None:
            src_mask = self.encoder.make_src_mask(src)
        enc_output = self.encoder(src, src_mask)

        # Step 2: Decoder
        dec_output = self.decoder(tgt, enc_output, tgt_mask, src_mask)

        # Step 3: 投影到词表
        logits = self.output_projection(dec_output)  # [batch, tgt_len, vocab_size]

        return logits

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        max_len: int = 50,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        贪心解码（推理用）：每步取概率最高的 token

        Args:
            src:     [1, src_len]  — 单个源序列（批次大小=1）
            bos_idx: <bos> token id
            eos_idx: <eos> token id
            max_len: 最大生成长度

        Returns:
            generated: 生成的 token id 序列（不含 <bos>）
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        src = src.to(device)

        # 一次性编码源序列
        src_mask = self.encoder.make_src_mask(src)
        enc_output = self.encode(src, src_mask)

        # 初始化 Decoder 输入（只有 <bos>）
        tgt = torch.tensor([[bos_idx]], dtype=torch.long, device=device)

        generated = []
        for _ in range(max_len):
            # Decoder 前向
            dec_output = self.decode(tgt, enc_output, src_mask=src_mask)

            # 取最后一步的预测
            logits = self.output_projection(dec_output[:, -1, :])  # [1, vocab]
            next_token = logits.argmax(dim=-1).item()

            if next_token == eos_idx:
                break

            generated.append(next_token)
            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            tgt = torch.cat([tgt, next_token_tensor], dim=1)

        return generated

    @torch.no_grad()
    def beam_search(
        self,
        src: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        beam_size: int = 4,
        max_len: int = 50,
        length_penalty: float = 0.6,
    ) -> list:
        """
        Beam Search 解码（比贪心质量更高）

        Args:
            src:            [1, src_len]
            beam_size:      beam 数量
            length_penalty: 长度惩罚因子（防止生成过短序列）

        Returns:
            best_sequence: 最佳生成 token id 列表
        """
        device = next(self.parameters()).device
        src = src.to(device)

        src_mask = self.encoder.make_src_mask(src)
        enc_output = self.encode(src, src_mask)

        # 扩展到 beam_size 倍（复制）
        enc_output = enc_output.expand(beam_size, -1, -1)
        src_mask = src_mask.expand(beam_size, -1, -1, -1)

        # beam: list of (score, sequence)
        beams = [(0.0, [bos_idx])]
        completed = []

        for step in range(max_len):
            candidates = []
            for score, seq in beams:
                if seq[-1] == eos_idx:
                    # 此 beam 已完成
                    pen = ((5 + len(seq)) / 6) ** length_penalty
                    completed.append((score / pen, seq[1:]))  # 去掉 <bos>
                    continue

                tgt = torch.tensor([seq], dtype=torch.long, device=device)
                # 扩展到 beam_size（只取当前 beam）
                dec_out = self.decode(tgt[:1], enc_output[:1], src_mask=src_mask[:1])
                logits = self.output_projection(dec_out[:, -1, :])  # [1, vocab]
                log_probs = F.log_softmax(logits, dim=-1)[0]  # [vocab]

                # 取 top-k 候选
                topk_probs, topk_ids = log_probs.topk(beam_size)
                for prob, token_id in zip(topk_probs.tolist(), topk_ids.tolist()):
                    candidates.append((score + prob, seq + [token_id]))

            if not candidates:
                break

            # 保留分数最高的 beam_size 个 beam
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_size]

        # 加入未完成的 beam
        for score, seq in beams:
            pen = ((5 + len(seq)) / 6) ** length_penalty
            completed.append((score / pen, seq[1:]))  # 去掉 <bos>

        completed.sort(key=lambda x: x[0], reverse=True)
        return completed[0][1] if completed else []


# ─────────────────────────────────────────────
# 验证
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("完整 Transformer (Encoder-Decoder)")
    print("=" * 60)

    # 小模型配置（快速验证）
    model = Transformer(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=512,
        dropout=0.0,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n参数总量: {total_params:,}")

    # 训练模式前向传播（Teacher Forcing）
    batch_size = 2
    src_len = 12
    tgt_len = 8

    src = torch.randint(1, 1000, (batch_size, src_len))
    tgt = torch.randint(1, 1000, (batch_size, tgt_len))
    src[0, -3:] = 0  # padding

    print(f"\n[训练] src shape: {src.shape}, tgt shape: {tgt.shape}")
    logits = model(src, tgt)
    print(f"[训练] logits shape: {logits.shape}")  # [2, 8, 1000]

    # 计算损失示例
    # tgt_shifted 是 tgt 右移一位后的目标（去掉 <bos>，加上 <eos>）
    target = torch.randint(1, 1000, (batch_size, tgt_len))
    loss = F.cross_entropy(
        logits.view(-1, 1000),
        target.view(-1),
        ignore_index=0,  # 忽略 padding
    )
    print(f"[训练] Cross-Entropy Loss: {loss.item():.4f}")

    # 推理模式（贪心解码）
    src_single = torch.randint(1, 1000, (1, 10))
    generated = model.greedy_decode(src_single, bos_idx=2, eos_idx=3, max_len=20)
    print(f"\n[推理] 贪心解码结果 (token ids): {generated}")

    print("\n✅ Transformer 验证通过！")

