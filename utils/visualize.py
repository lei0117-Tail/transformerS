"""
注意力可视化工具
================
提供可视化 Transformer 注意力权重的工具函数。

功能：
  1. 注意力热力图（单头/多头）
  2. 位置编码可视化
  3. 词嵌入相似度矩阵
  4. 多层注意力 subplot 展示
"""

import math

import matplotlib
import torch

matplotlib.use("Agg")  # 非交互式后端（Jupyter 可改为 TkAgg 或 inline）
import matplotlib.pyplot as plt
import numpy as np
from typing import List


def plot_attention_heatmap(
    attention_weights: torch.Tensor,
    src_tokens: List[str] = None,
    tgt_tokens: List[str] = None,
    title: str = "Attention Weights",
    save_path: str = None,
    figsize: tuple = (8, 6),
    cmap: str = "Blues",
) -> None:
    """
    绘制注意力权重热力图

    Args:
        attention_weights: [seq_q, seq_k] — 注意力权重矩阵
        src_tokens:        横轴（Key 序列）的 token 标签
        tgt_tokens:        纵轴（Query 序列）的 token 标签
        title:             图标题
        save_path:         保存路径（None=不保存）
    """
    if isinstance(attention_weights, torch.Tensor):
        weights = attention_weights.detach().cpu().numpy()
    else:
        weights = attention_weights

    seq_q, seq_k = weights.shape

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(weights, cmap=cmap, vmin=0, vmax=weights.max())
    plt.colorbar(im, ax=ax)

    # Token 标签
    if src_tokens:
        ax.set_xticks(range(len(src_tokens)))
        ax.set_xticklabels(src_tokens, rotation=45, ha="right", fontsize=10)
    else:
        ax.set_xticks(range(seq_k))
        ax.set_xticklabels([f"k{i}" for i in range(seq_k)], rotation=45)

    if tgt_tokens:
        ax.set_yticks(range(len(tgt_tokens)))
        ax.set_yticklabels(tgt_tokens, fontsize=10)
    else:
        ax.set_yticks(range(seq_q))
        ax.set_yticklabels([f"q{i}" for i in range(seq_q)])

    # 在每个格子中显示数值
    if seq_q <= 12 and seq_k <= 12:  # 序列不太长时显示数值
        for i in range(seq_q):
            for j in range(seq_k):
                ax.text(j, i, f"{weights[i, j]:.2f}",
                        ha="center", va="center",
                        color="white" if weights[i, j] > 0.5 else "black",
                        fontsize=8)

    ax.set_xlabel("Key (attended to)", fontsize=12)
    ax.set_ylabel("Query (attending from)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ 已保存到 {save_path}")
    plt.show()
    plt.close()


def plot_multihead_attention(
    attention_weights: torch.Tensor,
    tokens: List[str] = None,
    save_path: str = None,
    max_heads: int = 8,
) -> None:
    """
    多头注意力可视化（每个头单独显示）

    Args:
        attention_weights: [num_heads, seq_q, seq_k]
        tokens:            token 标签（自注意力时 src=tgt）
    """
    if isinstance(attention_weights, torch.Tensor):
        weights = attention_weights.detach().cpu().numpy()
    else:
        weights = attention_weights

    num_heads = min(weights.shape[0], max_heads)
    n_cols = min(4, num_heads)
    n_rows = math.ceil(num_heads / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
    axes = np.array(axes).flatten()

    for head_idx in range(num_heads):
        ax = axes[head_idx]
        im = ax.imshow(weights[head_idx], cmap="Blues", vmin=0)
        plt.colorbar(im, ax=ax, shrink=0.8)

        if tokens:
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(tokens)))
            ax.set_yticklabels(tokens, fontsize=8)

        ax.set_title(f"Head {head_idx + 1}", fontsize=11)

    # 隐藏多余的子图
    for i in range(num_heads, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f"Multi-Head Attention ({num_heads} heads)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ 已保存到 {save_path}")
    plt.show()
    plt.close()


def plot_positional_encoding(
    d_model: int = 64,
    max_len: int = 100,
    save_path: str = None,
) -> None:
    """
    可视化位置编码矩阵

    展示 sin/cos 的周期性：不同维度有不同频率
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe_matrix = pe.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 热力图
    im = axes[0].pcolormesh(pe_matrix.T, cmap="RdBu", vmin=-1, vmax=1)
    axes[0].set_xlabel("Position", fontsize=12)
    axes[0].set_ylabel("Dimension", fontsize=12)
    axes[0].set_title(f"Positional Encoding Heatmap\n(d_model={d_model}, max_len={max_len})", fontsize=12)
    plt.colorbar(im, ax=axes[0])

    # 前几维的波形图
    dims_to_plot = [0, 2, 4, 10, 20, 40]
    for dim in dims_to_plot:
        if dim < d_model:
            axes[1].plot(pe_matrix[:, dim], label=f"dim {dim}", alpha=0.8)
    axes[1].set_xlabel("Position", fontsize=12)
    axes[1].set_ylabel("Value", fontsize=12)
    axes[1].set_title("PE Values for Different Dimensions\n(lower dim = higher frequency)", fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].axhline(y=0, color="black", linestyle="--", alpha=0.3)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ 已保存到 {save_path}")
    plt.show()
    plt.close()


def plot_training_curve(
    train_losses: List[float],
    val_losses: List[float] = None,
    metrics: dict = None,
    title: str = "Training Curve",
    save_path: str = None,
) -> None:
    """
    绘制训练曲线

    Args:
        train_losses: 每 epoch 的训练损失
        val_losses:   验证损失（可选）
        metrics:      其他指标 {'name': [values]}（可选）
        title:        图标题
    """
    has_metrics = metrics and len(metrics) > 0
    n_plots = 1 + (1 if has_metrics else 0)

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    # 损失曲线
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, "b-o", label="Train Loss", markersize=4)
    if val_losses:
        axes[0].plot(epochs, val_losses, "r-o", label="Val Loss", markersize=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 其他指标
    if has_metrics:
        for name, values in metrics.items():
            axes[1].plot(range(1, len(values) + 1), values, "-o", label=name, markersize=4)
        axes[1].set_xlabel("Epoch")
        axes[1].set_title("Metrics")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ 已保存到 {save_path}")
    plt.show()
    plt.close()


def plot_embedding_similarity(
    embedding_weight: torch.Tensor,
    token_labels: List[str] = None,
    title: str = "Embedding Cosine Similarity",
    save_path: str = None,
) -> None:
    """
    可视化 Embedding 的余弦相似度矩阵

    Args:
        embedding_weight: [vocab_size, d_model] — Embedding 权重
        token_labels:     token 标签列表
    """
    if isinstance(embedding_weight, torch.Tensor):
        emb = embedding_weight.detach().cpu()
    else:
        emb = torch.tensor(embedding_weight)

    # 计算余弦相似度
    emb_norm = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
    sim = torch.matmul(emb_norm, emb_norm.T).numpy()

    n = sim.shape[0]
    fig, ax = plt.subplots(figsize=(max(6, n * 0.6), max(5, n * 0.5)))
    im = ax.imshow(sim, cmap="RdBu", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)

    if token_labels:
        ax.set_xticks(range(len(token_labels)))
        ax.set_xticklabels(token_labels, rotation=45, ha="right")
        ax.set_yticks(range(len(token_labels)))
        ax.set_yticklabels(token_labels)

    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ 已保存到 {save_path}")
    plt.show()
    plt.close()


def plot_model_architecture_summary(model: torch.nn.Module, title: str = "Model Summary") -> None:
    """
    打印模型结构摘要（参数量统计）
    """
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")
    print(f"{'Layer':<40} {'Params':>12}")
    print(f"{'-'*55}")

    total = 0
    trainable = 0
    for name, module in model.named_modules():
        params = sum(p.numel() for p in module.parameters(recurse=False))
        if params > 0:
            print(f"  {name:<38} {params:>12,}")
            total += params

    for p in model.parameters():
        if p.requires_grad:
            trainable += p.numel()

    print(f"{'-'*55}")
    print(f"  {'Total Parameters':<38} {total:>12,}")
    print(f"  {'Trainable Parameters':<38} {trainable:>12,}")
    print(f"  {'Non-trainable Parameters':<38} {total - trainable:>12,}")
    print(f"  {'Estimated Size (MB)':<38} {total * 4 / 1024 / 1024:>11.2f}")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────
# 快速演示
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("可视化工具演示")
    print("=" * 60)

    # ── 1. 注意力热力图 ──
    tokens = ["I", "love", "trans-", "former", "!"]
    weights = torch.softmax(torch.randn(5, 5), dim=-1)
    print("\n1. 注意力权重热力图")
    plot_attention_heatmap(
        weights,
        src_tokens=tokens,
        tgt_tokens=tokens,
        title="Self-Attention Weights",
        save_path="attention_heatmap.png"
    )

    # ── 2. 多头注意力 ──
    print("\n2. 多头注意力可视化（4 heads）")
    multihead_weights = torch.softmax(torch.randn(4, 5, 5), dim=-1)
    plot_multihead_attention(
        multihead_weights,
        tokens=tokens,
        save_path="multihead_attention.png"
    )

    # ── 3. 位置编码 ──
    print("\n3. 位置编码可视化")
    plot_positional_encoding(d_model=64, max_len=100, save_path="positional_encoding.png")

    # ── 4. 训练曲线 ──
    print("\n4. 训练曲线")
    train_losses = [3.5 - 0.2 * i + 0.05 * torch.randn(1).item() for i in range(15)]
    val_losses = [3.8 - 0.18 * i + 0.08 * torch.randn(1).item() for i in range(15)]
    plot_training_curve(
        train_losses, val_losses,
        metrics={"Accuracy": [0.1 + 0.05 * i for i in range(15)]},
        title="Transformer Training",
        save_path="training_curve.png"
    )

    print("\n✅ 所有可视化完成！")

