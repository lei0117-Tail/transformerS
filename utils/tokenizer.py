"""
简单分词器
==========
为课程实验提供几种常用的分词器实现，从简单到复杂：

1. CharTokenizer    — 字符级分词（最简单）
2. WordTokenizer    — 空格分词（简单）
3. BPETokenizer     — Byte-Pair Encoding（简化版，最接近实际 LLM）

特殊 token 约定：
  <pad> = 0  : 填充
  <bos> = 1  : 序列开始
  <eos> = 2  : 序列结束
  <unk> = 3  : 未知词
  <mask> = 4 : 掩码（BERT 预训练用）

使用方法：
    tokenizer = CharTokenizer()
    tokenizer.build_vocab("your text here")
    ids = tokenizer.encode("hello")
    text = tokenizer.decode(ids)
"""

import json
import os
import re
from collections import Counter
from typing import List, Dict

# ─────────────────────────────────────────────
# 特殊 Token 常量
# ─────────────────────────────────────────────
SPECIAL_TOKENS = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<unk>": 3,
    "<mask>": 4,
}
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
MASK_IDX = 4


class BaseTokenizer:
    """所有分词器的基类"""

    def __init__(self):
        self.token2id: Dict[str, int] = dict(SPECIAL_TOKENS)
        self.id2token: Dict[int, str] = {v: k for k, v in self.token2id.items()}
        self.vocab_size: int = len(self.token2id)

    def _add_token(self, token: str) -> int:
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token
            self.vocab_size += 1
        return self.token2id[token]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        raise NotImplementedError

    def batch_encode(
        self,
        texts: List[str],
        max_length: int = None,
        padding: bool = True,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> Dict:
        """
        批量编码，支持 padding 和截断

        Returns:
            {
                "input_ids": List[List[int]],
                "attention_mask": List[List[int]]
            }
        """
        encoded = [self.encode(t, add_bos, add_eos) for t in texts]

        if max_length:
            encoded = [e[:max_length] for e in encoded]

        max_len = max(len(e) for e in encoded)

        input_ids = []
        attention_mask = []

        for ids in encoded:
            pad_len = max_len - len(ids)
            if padding:
                input_ids.append(ids + [PAD_IDX] * pad_len)
                attention_mask.append([1] * len(ids) + [0] * pad_len)
            else:
                input_ids.append(ids)
                attention_mask.append([1] * len(ids))

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def save(self, path: str):
        """保存词表到 JSON 文件"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"token2id": self.token2id}, f, ensure_ascii=False, indent=2)
        print(f"词表已保存到 {path}")

    @classmethod
    def load(cls, path: str):
        """从 JSON 文件加载词表"""
        tokenizer = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tokenizer.token2id = data["token2id"]
        tokenizer.id2token = {int(v): k for k, v in tokenizer.token2id.items()}
        tokenizer.vocab_size = len(tokenizer.token2id)
        return tokenizer


# ─────────────────────────────────────────────
# 1. 字符级分词器
# ─────────────────────────────────────────────
class CharTokenizer(BaseTokenizer):
    """
    字符级分词器：每个字符是一个 token。

    优点：词表小（约128个 ASCII 字符），无 OOV 问题
    缺点：序列长，模型需要学习字符组合规律
    适用：字符级语言模型（如任务2）

    Example:
        tokenizer = CharTokenizer()
        tokenizer.build_vocab("hello world")
        ids = tokenizer.encode("hello")  → [5, 6, 7, 7, 8]（举例）
    """

    def build_vocab(self, text: str) -> "CharTokenizer":
        """从文本中构建字符词表"""
        for char in sorted(set(text)):
            self._add_token(char)
        print(f"字符级词表大小: {self.vocab_size} (含 {len(SPECIAL_TOKENS)} 个特殊 token)")
        return self

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = [self.token2id.get(c, UNK_IDX) for c in text]
        if add_bos:
            ids = [BOS_IDX] + ids
        if add_eos:
            ids = ids + [EOS_IDX]
        return ids

    def decode(self, ids, skip_special: bool = True) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        special_ids = set(SPECIAL_TOKENS.values())
        chars = []
        for i in ids:
            if skip_special and i in special_ids:
                continue
            chars.append(self.id2token.get(i, "?"))
        return "".join(chars)


# ─────────────────────────────────────────────
# 2. 词级分词器（空格/标点）
# ─────────────────────────────────────────────
class WordTokenizer(BaseTokenizer):
    """
    词级分词器：按空格和标点分割。

    优点：直观，序列较短
    缺点：词表大，存在 OOV（out-of-vocabulary）问题
    适用：英文文本分类等简单任务

    Example:
        tokenizer = WordTokenizer()
        tokenizer.build_vocab(["Hello world", "I love NLP"])
        ids = tokenizer.encode("I love NLP")  → [7, 5, 6]（举例）
    """

    def __init__(self, lowercase: bool = True):
        super().__init__()
        self.lowercase = lowercase

    def tokenize(self, text: str) -> List[str]:
        """分词：小写 + 按空格/标点分割"""
        if self.lowercase:
            text = text.lower()
        # 在标点前后插入空格再 split
        text = re.sub(r"([^\w\s])", r" \1 ", text)
        return [t for t in text.split() if t.strip()]

    def build_vocab(
        self,
        texts: List[str],
        min_freq: int = 1,
        max_vocab: int = None,
    ) -> "WordTokenizer":
        """
        从文本列表构建词表

        Args:
            texts:     文本列表
            min_freq:  最小词频（低频词用 <unk> 代替）
            max_vocab: 最大词表大小（None=不限制）
        """
        counter = Counter()
        for text in texts:
            counter.update(self.tokenize(text))

        # 按频率排序
        vocab = [word for word, freq in counter.most_common(max_vocab) if freq >= min_freq]
        for word in vocab:
            self._add_token(word)

        print(f"词级词表大小: {self.vocab_size} (原始词: {len(counter)}, min_freq={min_freq})")
        return self

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        tokens = self.tokenize(text)
        ids = [self.token2id.get(t, UNK_IDX) for t in tokens]
        if add_bos:
            ids = [BOS_IDX] + ids
        if add_eos:
            ids = ids + [EOS_IDX]
        return ids

    def decode(self, ids, skip_special: bool = True) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        special_ids = set(SPECIAL_TOKENS.values())
        words = []
        for i in ids:
            if skip_special and i in special_ids:
                continue
            words.append(self.id2token.get(i, "<unk>"))
        return " ".join(words)


# ─────────────────────────────────────────────
# 3. 简化版 BPE 分词器
# ─────────────────────────────────────────────
class SimpleBPETokenizer(BaseTokenizer):
    """
    简化版 Byte-Pair Encoding (BPE) 分词器

    BPE 核心思想（数据驱动的子词分割）：
      1. 初始：每个字符是一个 token
      2. 统计相邻 token pair 的频率
      3. 合并频率最高的 pair，生成新 token
      4. 重复直到词表大小达到目标

    优点：平衡词表大小和序列长度，处理 OOV（未见词会分解为字符）
    代表：GPT-2/3 的 BPE，LLaMA 的 SentencePiece（变体）

    注意：这是教学用的简化实现，真实的 BPE 更复杂。
    """

    def __init__(self):
        super().__init__()
        self.merges: List[tuple] = []  # 合并规则列表：[(token_a, token_b), ...]

    def _get_pairs(self, vocab: Dict[str, int]) -> Counter:
        """统计所有词中相邻 token pair 的频率"""
        pairs = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_vocab(self, pair: tuple, vocab: Dict[str, int]) -> Dict[str, int]:
        """在词表中合并指定 pair"""
        new_vocab = {}
        bigram = " ".join(pair)
        replacement = "".join(pair)
        for word, freq in vocab.items():
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = freq
        return new_vocab

    def learn_bpe(
        self,
        text: str,
        num_merges: int = 100,
        verbose: bool = True,
    ) -> "SimpleBPETokenizer":
        """
        从文本中学习 BPE 合并规则

        Args:
            text:       训练文本
            num_merges: BPE 合并次数（= 新增子词数量）
        """
        # Step 1: 初始词表（字符级），每个词末尾加 </w> 表示词边界
        words = text.lower().split()
        word_freq = Counter(words)
        # 将每个词表示为字符序列：'hello' → 'h e l l o </w>'
        vocab = {" ".join(list(w)) + " </w>": freq for w, freq in word_freq.items()}

        # 添加初始字符 token
        for word in vocab:
            for char in word.split():
                self._add_token(char)

        if verbose:
            print(f"初始字符词表大小: {self.vocab_size}")

        # Step 2: 迭代合并
        for merge_idx in range(num_merges):
            pairs = self._get_pairs(vocab)
            if not pairs:
                break

            # 找到最高频的 pair
            best_pair = max(pairs, key=pairs.get)
            best_freq = pairs[best_pair]

            if best_freq < 2:  # 频率低于 2，停止合并
                break

            # 合并并添加新 token
            new_token = "".join(best_pair)
            self.merges.append(best_pair)
            self._add_token(new_token)
            vocab = self._merge_vocab(best_pair, vocab)

            if verbose and (merge_idx + 1) % 20 == 0:
                print(f"  Merge {merge_idx + 1}: {best_pair} → {new_token!r} (freq={best_freq})")

        if verbose:
            print(f"BPE 后词表大小: {self.vocab_size} (学习了 {len(self.merges)} 条合并规则)")

        return self

    def tokenize(self, word: str) -> List[str]:
        """
        将单个词用 BPE 规则分割为子词

        核心逻辑：从字符级分割开始，按照学习到的 merge 规则依次合并
        """
        # 初始分割为字符
        word_chars = list(word) + ["</w>"]
        tokens = word_chars[:]

        # 按照 merge 规则依次合并
        for pair in self.merges:
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """将文本编码为 token ids"""
        all_tokens = []
        for word in text.lower().split():
            bpe_tokens = self.tokenize(word)
            all_tokens.extend(bpe_tokens)

        ids = [self.token2id.get(t, UNK_IDX) for t in all_tokens]
        if add_bos:
            ids = [BOS_IDX] + ids
        if add_eos:
            ids = ids + [EOS_IDX]
        return ids

    def decode(self, ids, skip_special: bool = True) -> str:
        """将 token ids 解码为文本"""
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        special_ids = set(SPECIAL_TOKENS.values())
        tokens = []
        for i in ids:
            if skip_special and i in special_ids:
                continue
            tokens.append(self.id2token.get(i, "<unk>"))

        # 合并 tokens（去掉 </w>）
        text = " ".join(tokens)
        text = text.replace(" </w>", " ").replace("</w>", "").strip()
        # 去掉 BPE 内部空格
        words = []
        current = []
        for token in tokens:
            if token.endswith("</w>"):
                current.append(token.replace("</w>", ""))
                words.append("".join(current))
                current = []
            else:
                current.append(token)
        if current:
            words.append("".join(current))
        return " ".join(words)


# ─────────────────────────────────────────────
# 演示 / 测试
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("分词器演示")
    print("=" * 60)

    sample_texts = [
        "Hello, world! I love Transformers.",
        "Natural language processing is fascinating.",
        "The quick brown fox jumps over the lazy dog.",
    ]

    # ── 1. 字符级分词器 ──
    print("\n" + "─" * 40)
    print("1. CharTokenizer（字符级）")
    print("─" * 40)

    char_tok = CharTokenizer()
    all_text = " ".join(sample_texts)
    char_tok.build_vocab(all_text)

    test = "Hello!"
    ids = char_tok.encode(test)
    decoded = char_tok.decode(ids)
    print(f"原文:  {test!r}")
    print(f"编码:  {ids}")
    print(f"解码:  {decoded!r}")
    print(f"add_bos/eos: {char_tok.encode(test, add_bos=True, add_eos=True)}")

    # ── 2. 词级分词器 ──
    print("\n" + "─" * 40)
    print("2. WordTokenizer（词级）")
    print("─" * 40)

    word_tok = WordTokenizer(lowercase=True)
    word_tok.build_vocab(sample_texts, min_freq=1)

    test = "I love Transformers."
    ids = word_tok.encode(test)
    decoded = word_tok.decode(ids)
    print(f"原文:  {test!r}")
    print(f"token: {word_tok.tokenize(test)}")
    print(f"编码:  {ids}")
    print(f"解码:  {decoded!r}")

    # OOV 测试
    oov_test = "I love BERT architectures."
    oov_ids = word_tok.encode(oov_test)
    print(f"\nOOV 测试: {oov_test!r}")
    print(f"编码（unk={UNK_IDX}）: {oov_ids}")

    # ── 3. BPE 分词器 ──
    print("\n" + "─" * 40)
    print("3. SimpleBPETokenizer（BPE 子词）")
    print("─" * 40)

    bpe_tok = SimpleBPETokenizer()
    bpe_tok.learn_bpe(all_text, num_merges=50, verbose=True)

    test_words = ["love", "natural", "processing", "transformers"]
    for word in test_words:
        tokens = bpe_tok.tokenize(word)
        print(f"  {word!r:15} → {tokens}")

    ids = bpe_tok.encode("Hello world")
    decoded = bpe_tok.decode(ids)
    print(f"\n编码 'Hello world': {ids}")
    print(f"解码: {decoded!r}")

    # ── 4. 批量编码 ──
    print("\n" + "─" * 40)
    print("4. 批量编码（padding）")
    print("─" * 40)

    batch_texts = ["hello", "hello world", "hello world how are you"]
    result = word_tok.batch_encode(batch_texts, max_length=10, padding=True)
    print(f"输入文本:")
    for t in batch_texts:
        print(f"  {t!r}")
    print(f"\ninput_ids (padded to max_length):")
    for ids in result["input_ids"]:
        print(f"  {ids}")
    print(f"\nattention_mask:")
    for mask in result["attention_mask"]:
        print(f"  {mask}")

    print("\n✅ 分词器演示完成！")

