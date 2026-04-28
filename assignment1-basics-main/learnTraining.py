"""
=============================================================================
BPE 分词器 & 训练基础设施 -- 从零理解完整流水线
=============================================================================

本文件是一个「学习笔记 + 可运行代码」的导读,
覆盖 homework/ 下 Transformer 架构之外的所有模块:
分词器 (Tokenizer) 和训练基础设施 (Training Infrastructure)。

层级关系 (从底到顶):
─────────────────────────────────────────────────────────
Part A: BPE 分词器
    A1. Unicode & UTF-8 编码基础
    A2. 预分词 (Pre-tokenization)
    A3. BPE 训练算法 (字节对合并)
    A4. BPE 编码 & 解码
    A5. 流式编码 (大文件处理)

Part B: 训练基础设施
    B1. Cross-Entropy Loss (数值稳定)
    B2. Data Loading (随机采样 mini-batch)
    B3. AdamW 优化器 (自适应学习率 + 解耦权重衰减)
    B4. 余弦学习率调度 (Cosine Annealing with Warmup)
    B5. 梯度裁剪 (Gradient Clipping)
    B6. Checkpoint (断点续训)

Part C: 资源核算
    C1. 内存分析 (参数/梯度/优化器/激活值)
    C2. FLOPs 分析
─────────────────────────────────────────────────────────

下面我们逐一剖析。
"""

import math
import numpy as np
import torch
import torch.nn as nn
from collections import Counter, defaultdict


# ============================================================================
# Part A: BPE 分词器
# ============================================================================

# ── A1. Unicode & UTF-8 编码基础 ───────────────────────
"""
★ 为什么要做这个?
  神经网络只能处理数字, 不能直接处理文本。
  我们需要把任意语言的文本转换成整数序列 (token IDs), 这就是分词器的任务。
  而分词器的第一步, 就是选择 "最小单元": 字符? 词? 还是字节?
  理解 Unicode 和 UTF-8 是理解为什么选择「字节」作为起点的基础。

为什么从字节开始?

  字符级别: "hello" → ['h','e','l','l','o']   简单, 但词表爆炸 (Unicode 14万+)
  词级别:   "hello" → ['hello']               高效, 但处理不了新词 (OOV 问题)
  字节级别: "hello" → [104,101,108,108,111]    初始词表仅 256, 能表示一切

UTF-8 编码:
  - ASCII 字符 (英文/数字): 1 字节
  - 中文字符: 3 字节  (如 '你' → 0xe4, 0xbd, 0xa0)
  - Emoji:    4 字节  (如 '😅' → 0xf0, 0x9f, 0x98, 0x85)

为什么 UTF-8 > UTF-16/32 ?
  - UTF-8 对英文只用 1 字节, 空间利用率最高
  - UTF-16/32 会给每个 ASCII 字符填充无用的 0x00 字节
  - BPE 在字节流上学习, UTF-8 的紧凑性让合并模式更有意义

代码位置: homework/bpe_tokenizer.py
"""


def demo_unicode_basics():
    """演示 Unicode 和 UTF-8 编码。"""
    print("=" * 60)
    print("A1. Unicode & UTF-8 编码基础")
    print("=" * 60)

    examples = [
        ("hello", "英文"),
        ("你好", "中文"),
        ("café", "带重音字符"),
        ("😅", "Emoji"),
    ]

    for text, desc in examples:
        utf8 = text.encode("utf-8")
        byte_list = list(utf8)
        print(f"\n  '{text}' ({desc}):")
        print(f"    字符数: {len(text)}")
        print(f"    UTF-8 字节数: {len(utf8)}")
        print(f"    字节值: {byte_list}")
        print(f"    十六进制: {[f'0x{b:02x}' for b in byte_list]}")

    # 关键：一个字节不一定对应一个字符
    print("\n  关键结论:")
    print("    - 英文: 1 字符 = 1 字节")
    print("    - 中文: 1 字符 = 3 字节")
    print("    - UTF-8 是变长编码, 不能逐字节 decode")

    # 演示逐字节 decode 的错误
    print("\n  错误示例: 逐字节 decode 多字节字符")
    try:
        bad = b"caf\xc3\xa9"  # "café" 的 UTF-8
        _ = "".join([bytes([b]).decode("utf-8") for b in bad])
    except UnicodeDecodeError as e:
        print(f"    UnicodeDecodeError: {e}")
        print("    原因: 'é' 需要 2 字节 (0xc3, 0xa9), 不能拆开 decode")


# ── A2. 预分词 (Pre-tokenization) ──────────────────────
"""
★ 为什么要做这个?
  BPE 只关心统计频率, 它不懂语言。如果不加限制, 它可能合并 "。我" 变成一个 token,
  或者把 "hello" 的 "o" 和 "world" 的 "w" 合在一起。这些跨边界的合并毫无意义,
  还会浪费宝贵的词表空间。预分词就是在 BPE 之前画好 "不可逾越的边界",
  让 BPE 只在有意义的单元内部做合并。

为什么需要预分词?

  如果直接在整个文本的字节流上做 BPE:
    "hello world" 可能学到合并 "o w" → "o w"
    跨越了单词边界, 语义上不合理

  预分词: 先按正则把文本分成语言学单元, 然后每个单元内部独立做 BPE。
  这保证了:
    - 不会跨单词合并
    - 空格被保留为 token 的一部分 (如 " the" 是一个 token)
    - 标点和数字被独立分出

GPT-2 的预分词正则:

  r"'(?:[sdmt]|ll|ve|re)| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"

  匹配顺序:
    1. 缩写后缀: 's, 'd, 'm, 't, 'll, 've, 're
    2. 可选空格 + 字母序列:  " hello"
    3. 可选空格 + 数字序列:  " 123"
    4. 可选空格 + 其他字符:  " ,"
    5. 行尾空白 / 其他空白

代码位置: homework/bpe_tokenizer.py
"""


def demo_pretokenization():
    """演示 GPT-2 风格的预分词。"""
    import regex

    print("\n" + "=" * 60)
    print("A2. 预分词 (Pre-tokenization)")
    print("=" * 60)

    GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    examples = [
        "Hello, world!",
        "I'll be there at 3:30pm.",
        "some text that i'll pre-tokenize",
        "price is $99.99 today",
    ]

    for text in examples:
        tokens = regex.findall(GPT2_PAT, text)
        print(f"\n  '{text}'")
        print(f"  → {tokens}")

    print("\n  注意:")
    print("    - 空格被归入下一个单词: ' world' 而非 'world'")
    print("    - 缩写被拆分: \"i'll\" → \"i\" + \"'ll\"")
    print("    - 标点和数字单独分出")


# ── A3. BPE 训练算法 ──────────────────────────────────
"""
★ 为什么要做这个?
  初始的 256 字节词表太细粒度了: 一个普通英文单词需要 5-10 个 token,
  中文一个字就要 3 个 token。这让序列变得很长, 注意力计算 O(T²) 会爆炸。
  BPE 训练通过统计频率, 自动发现 "值得合并" 的字节组合,
  把高频模式压缩成单个 token (如 " the"、"ing"、"tion"),
  从而大幅缩短序列长度, 让 Transformer 能高效处理。

BPE (Byte-Pair Encoding) 训练算法:

  输入: 文本语料 + 目标词汇量 V
  输出: 词汇表 vocab + 合并规则 merges

  初始化:
    vocab = {0: b'\x00', 1: b'\x01', ..., 255: b'\xff'}  (256 个基础字节)
    加入 special tokens (如 <|endoftext|>)

  主循环 (直到 |vocab| = V):
    1. 统计语料中所有相邻 token 对的频率
    2. 选出频率最高的对 (p1, p2)
       平局时选字典序最大的 (保证确定性)
    3. 创建新 token: new = p1 + p2
    4. 把 new 加入 vocab, 把 (p1, p2) 加入 merges
    5. 在所有序列中把 p1,p2 替换为 new
    6. 增量更新 pair 频率 (性能优化的关键!)

  增量更新优化:
    朴素方法: 每次合并后重新扫描整个语料统计频率 → O(merges × corpus)
    优化方法: 只更新受影响的 pair → 大幅减少计算量
      合并 (A,B) → AB 后:
        - 删除: (X,A), (A,B), (B,Y) 的计数
        - 添加: (X,AB), (AB,Y) 的计数

代码位置: homework/bpe_tokenizer.py → run_train_bpe()
"""


def demo_bpe_training():
    """手动模拟 BPE 训练过程。"""
    print("\n" + "=" * 60)
    print("A3. BPE 训练算法 (手动模拟)")
    print("=" * 60)

    # 语料: 带频率的预分词结果
    corpus_with_freq = {
        # 每个 word 出现的次数
        tuple(bytes([b]) for b in "hello".encode()): 3,
        tuple(bytes([b]) for b in "hell".encode()): 2,
        tuple(bytes([b]) for b in " world".encode()): 1,
    }

    print("\n  初始语料 (预分词后):")
    for seq, freq in corpus_with_freq.items():
        readable = [t.decode("latin-1") for t in seq]
        print(f"    {readable} × {freq}")

    # 初始词汇表
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    target_size = 260

    print(f"\n  初始词汇表: {len(vocab)} 个 (256 基础字节)")
    print(f"  目标词汇表: {target_size}")
    print()

    iteration = 0
    while len(vocab) < target_size:
        iteration += 1

        # Step 1: 统计 pair 频率
        pair_counts = Counter()
        for seq, freq in corpus_with_freq.items():
            for i in range(len(seq) - 1):
                pair_counts[(seq[i], seq[i + 1])] += freq

        if not pair_counts:
            break

        # Step 2: 选最频繁的 pair (平局选字典序最大)
        max_count = max(pair_counts.values())
        candidates = [p for p, c in pair_counts.items() if c == max_count]
        best_pair = max(candidates)

        # Step 3: 创建新 token
        new_token = best_pair[0] + best_pair[1]
        new_id = len(vocab)
        vocab[new_id] = new_token
        merges.append(best_pair)

        # Step 4: 在语料中执行合并
        new_corpus = {}
        for seq, freq in corpus_with_freq.items():
            new_seq = _merge_pair(seq, best_pair, new_token)
            new_corpus[new_seq] = new_corpus.get(new_seq, 0) + freq
        corpus_with_freq = new_corpus

        # 打印
        p1 = best_pair[0].decode("latin-1")
        p2 = best_pair[1].decode("latin-1")
        merged = new_token.decode("latin-1")
        print(f"  迭代 {iteration}: '{p1}' + '{p2}' → '{merged}' "
              f"(频率={max_count}, ID={new_id})")

    print(f"\n  最终合并规则 ({len(merges)} 条):")
    for i, (p1, p2) in enumerate(merges):
        print(f"    {i + 1}. {p1.decode('latin-1')!r} + "
              f"{p2.decode('latin-1')!r} → "
              f"{(p1 + p2).decode('latin-1')!r}")

    print(f"\n  最终语料:")
    for seq, freq in corpus_with_freq.items():
        readable = [t.decode("latin-1") for t in seq]
        print(f"    {readable} × {freq}")


def _merge_pair(seq, pair, new_token):
    """在 token 序列中执行一次合并。"""
    result = []
    i = 0
    while i < len(seq):
        if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            result.append(new_token)
            i += 2
        else:
            result.append(seq[i])
            i += 1
    return tuple(result)


# ── A4. BPE 编码 & 解码 ───────────────────────────────
"""
★ 为什么要做这个?
  训练好词表和合并规则后, 还需要一个能把任意文本转成 token ID 序列的编码器,
  以及一个能把 token ID 序列还原为文本的解码器。
  Encode 是 Transformer 的 "入口": 用户的文本必须先 encode 成整数才能喂给模型。
  Decode 是 Transformer 的 "出口": 模型输出的 token ID 必须 decode 回文本才能给人看。
  Encode + Decode 必须互逆 (roundtrip), 否则信息就会丢失。

── Encode (文本 → Token IDs) ──

  1. 处理特殊 token: 先按特殊 token 分割文本
     "hello<|endoftext|>world" → ["hello", "<|endoftext|>", "world"]

  2. 对每个非特殊段做预分词:
     "hello world" → ["hello", " world"]

  3. 每个预分词单元 → UTF-8 字节 → 初始单字节 token 序列:
     "hello" → [b'h', b'e', b'l', b'l', b'o']

  4. 按合并规则的优先级迭代合并:
     对序列中所有相邻对, 找到在 merges 中最靠前 (优先级最高) 的对,
     执行合并, 重复直到没有可合并的对
     [b'h', b'e', b'l', b'l', b'o']
       → merge (h,e): [b'he', b'l', b'l', b'o']
       → merge (l,l): [b'he', b'll', b'o']
       → 没有更多可合并的对, 停止

  5. 查表: token bytes → token ID:
     [b'he', b'll', b'o'] → [256, 257, 111]


── Decode (Token IDs → 文本) ──

  1. 查表: token ID → token bytes:
     [256, 257, 111] → [b'he', b'll', b'o']

  2. 拼接所有字节:
     b'he' + b'll' + b'o' = b'hello'

  3. UTF-8 解码:
     b'hello'.decode('utf-8') → "hello"

  Decode 比 Encode 简单得多!

代码位置: homework/bpe_tokenizer.py → Tokenizer 类
"""


def demo_bpe_encode_decode():
    """手动模拟 BPE 编码和解码。"""
    print("\n" + "=" * 60)
    print("A4. BPE 编码 & 解码 (手动模拟)")
    print("=" * 60)

    # 模拟一个小词汇表
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = b"he"
    vocab[257] = b"ll"
    vocab[258] = b"hel"
    vocab[259] = b"hell"
    vocab[260] = b"hello"

    merges = [
        (b"h", b"e"),      # → he
        (b"l", b"l"),      # → ll
        (b"he", b"l"),     # → hel
        (b"hel", b"l"),    # → hell
        (b"hell", b"o"),   # → hello
    ]

    bytes_to_id = {v: k for k, v in vocab.items()}

    # ── Encode ──
    text = "hello"
    print(f"\n  Encode: '{text}'")

    # Step 1: 文本 → UTF-8 字节
    raw_bytes = text.encode("utf-8")
    tokens = [bytes([b]) for b in raw_bytes]
    print(f"    1. UTF-8 字节: {[t.decode('latin-1') for t in tokens]}")

    # Step 2: 迭代合并
    step = 0
    while True:
        # 找优先级最高的可合并对
        best_idx = -1
        best_priority = len(merges)
        best_pair = None

        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            if pair in merges:
                priority = merges.index(pair)
                if priority < best_priority:
                    best_priority = priority
                    best_idx = i
                    best_pair = pair

        if best_pair is None:
            break

        step += 1
        merged = best_pair[0] + best_pair[1]
        tokens = tokens[:best_idx] + [merged] + tokens[best_idx + 2:]
        print(f"    2.{step} 合并 {best_pair[0].decode('latin-1')!r} + "
              f"{best_pair[1].decode('latin-1')!r} → "
              f"{merged.decode('latin-1')!r}  "
              f"序列: {[t.decode('latin-1') for t in tokens]}")

    # Step 3: 查表得 ID
    token_ids = [bytes_to_id[t] for t in tokens]
    print(f"    3. Token IDs: {token_ids}")

    # ── Decode ──
    print(f"\n  Decode: {token_ids}")
    decoded_bytes = b"".join(vocab[tid] for tid in token_ids)
    decoded_text = decoded_bytes.decode("utf-8")
    print(f"    1. 查表: {[vocab[tid] for tid in token_ids]}")
    print(f"    2. 拼接: {decoded_bytes}")
    print(f"    3. UTF-8 解码: '{decoded_text}'")
    print(f"\n  Roundtrip 验证: '{text}' → {token_ids} → '{decoded_text}' "
          f"{'OK' if text == decoded_text else 'FAIL'}")


# ── A5. 流式编码 ──────────────────────────────────────
"""
★ 为什么要做这个?
  真实训练数据动辄数 GB 甚至 TB (OpenWebText ≈ 11GB, The Pile ≈ 800GB)。
  如果一次性把整个文件读入内存再编码, 内存会直接爆掉。
  流式编码 (encode_iterable) 让我们逐行处理, 内存占用只和单行大小有关,
  这是工程上处理大规模数据的必备能力。

处理大文件 (GB 级别) 时不能一次性读入内存。

  encode_iterable(text_iterable):
    逐行读取 → 预分词 → BPE 编码 → yield token IDs

  内存占用 = O(单行大小), 与文件总大小无关。

  特殊处理:
    - 跨行的预分词单元: 需要缓存上一行的尾部
    - 特殊 token 分割: 在每行内独立处理

代码位置: homework/bpe_tokenizer.py → Tokenizer.encode_iterable()
"""


# ============================================================================
# Part B: 训练基础设施
# ============================================================================

# ── B1. Cross-Entropy Loss ─────────────────────────────
"""
★ 为什么要做这个?
  训练 = 让模型的预测越来越接近正确答案。但 "接近" 需要一个数字来衡量,
  这就是损失函数 (loss)。对于语言模型, 每个位置要从 vocab_size 个候选中
  预测正确的下一个 token, 本质是一个分类问题。
  交叉熵 (Cross-Entropy) 是分类问题的标准损失函数:
    - 预测对了 → loss 接近 0
    - 预测错了 → loss 很大
  它是整个训练循环的 "指南针", 反向传播的起点。

数学:
  对每个样本:  CE = -log P(target) = -log softmax(logits)[target]

展开:
  CE = -logit_target + log(sum(exp(logit_j)))

数值稳定技巧:
  令 c = max(logits), 则:
    log(sum(exp(logit_j))) = c + log(sum(exp(logit_j - c)))
  减去 max 防止 exp 溢出, 数学上等价。

为什么不先算 softmax 再取 log?
  softmax 输出可能非常接近 0, 取 log 后变成 -inf → 精度丢失
  直接用 log-sum-exp 形式更稳定。

代码位置: homework/cross_entropy.py
"""


def demo_cross_entropy():
    """演示交叉熵损失的计算和数值稳定性。"""
    from homework.cross_entropy import cross_entropy

    print("\n" + "=" * 60)
    print("B1. Cross-Entropy Loss")
    print("=" * 60)

    vocab_size = 10
    batch_size = 4

    # 场景 1: 随机 logits (模型未训练)
    logits = torch.randn(batch_size, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size,))

    loss = cross_entropy(logits, targets)
    expected = math.log(vocab_size)
    print(f"\n  场景 1: 随机 logits")
    print(f"    损失: {loss.item():.4f}")
    print(f"    随机猜测的期望: ln({vocab_size}) = {expected:.4f}")
    print(f"    (两者应接近, 因为 logits 是随机的)")

    # 场景 2: 完美预测
    logits_perfect = torch.zeros(batch_size, vocab_size)
    logits_perfect.scatter_(1, targets.unsqueeze(1), 100.0)  # target 位置 logit=100

    loss_perfect = cross_entropy(logits_perfect, targets)
    print(f"\n  场景 2: 完美预测 (target logit=100)")
    print(f"    损失: {loss_perfect.item():.6f}")
    print(f"    (应接近 0)")

    # 场景 3: 数值稳定性测试
    logits_large = torch.tensor([[1000.0, 0.0, 0.0]])
    targets_large = torch.tensor([0])
    loss_large = cross_entropy(logits_large, targets_large)
    print(f"\n  场景 3: 极端 logits [1000, 0, 0]")
    print(f"    损失: {loss_large.item():.6f}")
    print(f"    (减去 max 后: [0, -1000, -1000] → 不会溢出)")


# ── B2. Data Loading ───────────────────────────────────
"""
★ 为什么要做这个?
  训练数据是一个巨大的 token 序列 (数十亿个 token), 不可能一次性全部喂给模型。
  我们需要把它切成一小块一小块的 mini-batch, 每次只处理几个固定长度的序列。
  Data Loader 就是这个 "切片机": 它随机采样起始位置, 切出输入和目标,
  保证每个 batch 的序列等长 (GPU 最高效), 同时支持 next-token prediction 的格式。

语言模型训练数据:
  多个文档拼接成一个超长 token 序列, 用 <|endoftext|> 分隔:
    [doc1_tokens... <eos> doc2_tokens... <eos> ...]

get_batch(dataset, batch_size, context_length, device):
  1. 随机采样 batch_size 个起始位置
     范围: [0, len(dataset) - context_length)
  2. 切出 context_length 长度的窗口作为输入 x
  3. x 右移一位作为目标 y (next-token prediction)

  x[i] = dataset[start : start + context_length]
  y[i] = dataset[start+1 : start+1 + context_length]

示意图:
  dataset: [a, b, c, d, e, f, g, h, ...]
                      ↑ start=2, context_length=4
  x:       [c, d, e, f]
  y:       [d, e, f, g]   (每个位置预测下一个 token)

代码位置: homework/data_loading.py
"""


def demo_data_loading():
    """演示训练数据的加载方式。"""
    from homework.data_loading import get_batch

    print("\n" + "=" * 60)
    print("B2. Data Loading")
    print("=" * 60)

    dataset = np.arange(100)  # 简单数据: [0, 1, 2, ..., 99]

    x, y = get_batch(dataset, batch_size=3, context_length=5, device="cpu")

    print(f"\n  dataset: [0, 1, 2, ..., 99]  (100 个 token)")
    print(f"  batch_size=3, context_length=5")
    print(f"\n  采样结果:")
    for i in range(3):
        x_list = x[i].tolist()
        y_list = y[i].tolist()
        ok = all(y_list[j] == x_list[j] + 1 for j in range(len(x_list)))
        print(f"    序列 {i}: x={x_list}")
        print(f"             y={y_list}  "
              f"{'(y = x 右移一位)' if ok else ''}")

    print(f"\n  关键特性:")
    print(f"    - 起始位置随机采样 → 每个 epoch 看到不同的窗口")
    print(f"    - x 和 y 只差一个位置 → next-token prediction")
    print(f"    - 所有序列等长 → 无需 padding, GPU 利用率高")


# ── B3. AdamW 优化器 ──────────────────────────────────
"""
★ 为什么要做这个?
  反向传播算出了梯度 (告诉我们每个参数应该往哪个方向调整),
  但 "调整多少" 才是关键。最简单的 SGD 对所有参数用同一个步长,
  但 Transformer 有数十亿参数, 不同参数的 "地形" 差异很大:
  有的参数梯度大且稳定, 有的参数梯度小且嘈杂。
  AdamW 为每个参数自动计算合适的步长 (自适应学习率),
  还通过动量平滑噪声, 加上解耦的权重衰减防止过拟合。
  它是目前训练大型语言模型的标准优化器。

AdamW = Adam + 解耦权重衰减

核心状态 (每个参数):
  m: 一阶矩 (梯度的指数移动平均 ≈ 梯度均值)
  v: 二阶矩 (梯度平方的指数移动平均 ≈ 梯度方差)

更新公式 (每步):
  1. 更新矩估计:
       m ← β₁·m + (1-β₁)·g           β₁=0.9 (偏向近期梯度)
       v ← β₂·v + (1-β₂)·g²          β₂=0.999 (更平滑的方差估计)

  2. 偏差校正 (训练初期 m,v 偏向 0, 需要修正):
       α_t = lr × √(1-β₂ᵗ) / (1-β₁ᵗ)

  3. 参数更新 (自适应学习率):
       θ ← θ - α_t × m / (√v + ε)

     直觉: 梯度方差大的参数用小步长, 方差小的用大步长
           → 自动适应不同参数的 "地形"

  4. 解耦权重衰减:
       θ ← θ - lr × λ × θ

     为什么 "解耦"?
       原始 Adam: weight decay 加在梯度上 → 被自适应学习率缩放, 不同参数衰减不一致
       AdamW: weight decay 直接作用于参数 → 所有参数以相同比例衰减, 更合理

代码位置: homework/adamw.py

内存开销:
  每个参数需要额外存储 m 和 v → 优化器状态 = 2 × 参数量 × 4 bytes
  GPT-2 XL (1.5B 参数): 优化器状态 ≈ 12 GB
"""


def demo_adamw():
    """演示 AdamW 优化器的工作原理和与 PyTorch 的对比。"""
    from homework.adamw import AdamW

    print("\n" + "=" * 60)
    print("B3. AdamW 优化器")
    print("=" * 60)

    # 演示: 优化 f(x) = x^2 的最小值
    print("\n  任务: 最小化 f(x) = sum(x^2), 初始 x ~ N(0,5)")

    torch.manual_seed(42)
    x_ours = nn.Parameter(5 * torch.randn(10))
    x_ref = nn.Parameter(x_ours.data.clone())

    opt_ours = AdamW([x_ours], lr=0.1, weight_decay=0.01)
    opt_ref = torch.optim.AdamW([x_ref], lr=0.1, weight_decay=0.01)

    print(f"\n  {'步数':>4} | {'我们的损失':>10} | {'PyTorch损失':>10} | {'差异':>10}")
    print("  " + "-" * 50)

    for step in range(20):
        # 我们的实现
        opt_ours.zero_grad()
        loss_ours = (x_ours**2).mean()
        loss_ours.backward()
        opt_ours.step()

        # PyTorch 参考实现
        opt_ref.zero_grad()
        loss_ref = (x_ref**2).mean()
        loss_ref.backward()
        opt_ref.step()

        diff = (x_ours.data - x_ref.data).abs().max().item()

        if step % 5 == 0 or step == 19:
            print(f"  {step:4d} | {loss_ours.item():10.6f} | "
                  f"{loss_ref.item():10.6f} | {diff:10.2e}")

    final_diff = (x_ours.data - x_ref.data).abs().max().item()
    print(f"\n  最终参数差异: {final_diff:.2e}")
    print(f"  与 PyTorch AdamW 匹配: {'YES' if final_diff < 1e-5 else 'NO'}")

    # 解释 AdamW vs SGD 的优势
    print(f"\n  AdamW 的优势:")
    print(f"    - 自适应学习率: 不同参数自动用不同步长")
    print(f"    - 动量: 指数移动平均平滑梯度噪声")
    print(f"    - 解耦 weight decay: 正则化效果更一致")


# ── B4. 余弦学习率调度 ────────────────────────────────
"""
★ 为什么要做这个?
  即使有了 AdamW, 全程用同一个学习率也不是最优的:
    - 训练初期: Adam 的矩估计 m,v 还不准, 大学习率会导致参数跑偏
    - 训练中期: 需要大学习率充分探索参数空间
    - 训练后期: 需要小学习率精细调整, 逼近最优解
  学习率调度就是 "自动挡": 先慢慢加速 (warmup), 再逐渐减速 (cosine decay),
  让训练过程又稳又快。几乎所有 LLM 训练都使用这种策略。

三个阶段:

  (1) 线性预热 (Warmup): t < T_w
      lr = (t / T_w) × lr_max

      为什么?
        训练初期, Adam 的矩估计 m,v 不准确 (初始化为 0)。
        如果一开始就用大学习率, 更新方向不可靠, 容易发散。
        线性 warmup 给 Adam 时间来稳定矩估计。

  (2) 余弦退火 (Cosine Annealing): T_w ≤ t ≤ T_c
      lr = lr_min + 0.5 × (1 + cos(π × (t - T_w) / (T_c - T_w))) × (lr_max - lr_min)

      为什么余弦而不是线性衰减?
        余弦在开头和结尾变化慢, 中间变化快。
        开头: 保持较大学习率更久, 充分探索
        结尾: 平稳过渡到最小学习率, 精细调整

  (3) 退火后 (Post-annealing): t > T_c
      lr = lr_min

曲线图:
  lr
  ↑ lr_max ···/‾‾‾‾‾‾‾‾‾‾‾\
  |        /                  \
  |       /                    \
  |      /                      \\___________
  | lr_min                                  →  t
  └──┬──┬────────────────┬──────
    0  T_w               T_c

代码位置: homework/lr_schedule.py
"""


def demo_lr_schedule():
    """演示余弦学习率调度。"""
    from homework.lr_schedule import lr_cosine_schedule

    print("\n" + "=" * 60)
    print("B4. 余弦学习率调度")
    print("=" * 60)

    max_lr = 1.0
    min_lr = 0.1
    warmup = 10
    total = 50

    print(f"\n  参数: max_lr={max_lr}, min_lr={min_lr}, "
          f"warmup={warmup}, total={total}")
    print(f"\n  {'步数':>4} | {'学习率':>8} | {'阶段':>8} | 可视化")
    print("  " + "-" * 55)

    for t in range(total + 5):
        lr = lr_cosine_schedule(t, max_lr, min_lr, warmup, total)

        if t < warmup:
            phase = "预热"
        elif t <= total:
            phase = "余弦退火"
        else:
            phase = "退火后"

        bar = "#" * int(lr * 30)

        if t % 3 == 0 or t == warmup or t == total:
            print(f"  {t:4d} | {lr:8.4f} | {phase:>8} | {bar}")

    print(f"\n  关键直觉:")
    print(f"    - 预热: 让 Adam 的矩估计稳定下来")
    print(f"    - 余弦衰减: 前期充分探索, 后期精细调整")
    print(f"    - 不要骤降: 余弦比线性更 '温柔'")


# ── B5. 梯度裁剪 ──────────────────────────────────────
"""
★ 为什么要做这个?
  训练过程中, 偶尔会遇到一个 "坏 batch" (比如包含异常数据或罕见 token),
  导致梯度突然变得非常大 (gradient spike)。如果不加限制, 这一步的巨大更新
  会把参数推到极端值, 之后的 loss 飙升, 甚至出现 NaN, 整个训练崩溃。
  梯度裁剪就是一个 "安全阀": 正常时不干预, 梯度太大时自动缩小,
  保证每一步的更新幅度在可控范围内。这是大模型训练稳定性的关键保障。

问题: 训练中偶尔出现异常大的梯度 (gradient spike), 会导致:
  - 参数被更新到极端值
  - loss 突然飙升
  - 训练不稳定甚至 NaN

解决: 梯度裁剪 (Gradient Clipping by Global L2 Norm)

算法:
  1. 计算全局梯度范数: ||g||₂ = √(Σ ||g_param||₂²)
     (把所有参数的梯度拼成一个大向量, 算 L2 范数)

  2. 如果 ||g||₂ ≤ M:  不做任何事
     如果 ||g||₂ > M:  g ← g × M / (||g||₂ + ε)

  效果: 保持梯度方向不变, 只在 "步子太大" 时等比例缩短。

  比喻: 不管你想往哪个方向走, 每步最多走 M 米。

代码位置: homework/gradient_clipping.py
"""


def demo_gradient_clipping():
    """演示梯度裁剪的效果。"""
    from homework.gradient_clipping import gradient_clipping

    print("\n" + "=" * 60)
    print("B5. 梯度裁剪")
    print("=" * 60)

    max_norm = 1.0

    # 创建一些参数并计算梯度
    torch.manual_seed(42)
    params = [nn.Parameter(torch.randn(5, 5)) for _ in range(3)]

    # 模拟一个产生大梯度的 loss
    loss = sum((p * 10).sum() for p in params)
    loss.backward()

    # 裁剪前
    norm_before = torch.sqrt(
        sum(p.grad.pow(2).sum() for p in params)
    ).item()

    print(f"\n  max_norm = {max_norm}")
    print(f"\n  裁剪前:")
    print(f"    全局梯度 L2 范数: {norm_before:.4f}")
    for i, p in enumerate(params):
        print(f"    参数 {i} 梯度范数: {p.grad.norm().item():.4f}")

    # 保存裁剪前的梯度方向
    dirs_before = [p.grad.clone() / p.grad.norm() for p in params]

    # 执行裁剪
    gradient_clipping(params, max_norm)

    # 裁剪后
    norm_after = torch.sqrt(
        sum(p.grad.pow(2).sum() for p in params)
    ).item()

    print(f"\n  裁剪后:")
    print(f"    全局梯度 L2 范数: {norm_after:.4f} (≤ {max_norm})")
    for i, p in enumerate(params):
        print(f"    参数 {i} 梯度范数: {p.grad.norm().item():.4f}")

    # 验证方向不变
    print(f"\n  方向保持验证:")
    for i, p in enumerate(params):
        dir_after = p.grad / p.grad.norm()
        cos_sim = (dirs_before[i] * dir_after).sum().item()
        print(f"    参数 {i} 余弦相似度: {cos_sim:.6f} (应为 1.0)")


# ── B6. Checkpoint ────────────────────────────────────
"""
★ 为什么要做这个?
  训练一个大模型可能需要数天甚至数周。如果训练到一半机器宕机、断电、
  或者 CUDA OOM, 没有 checkpoint 就意味着从头再来, 浪费大量时间和算力。
  Checkpoint 定期把模型参数、优化器状态、当前步数保存到磁盘,
  出问题后可以从最近的 checkpoint "无缝" 恢复, 继续训练。
  这是大规模训练的 "保险机制", 工程上必不可少。

保存训练状态以支持断点续训。

保存内容:
  {
      "model_state_dict": model.state_dict(),       # 所有参数
      "optimizer_state_dict": optimizer.state_dict(),  # m, v, step 等
      "iteration": step_number,                      # 当前步数
  }

为什么要保存优化器状态?
  AdamW 的 m (一阶矩) 和 v (二阶矩) 是训练历史的累积。
  如果只恢复模型参数但重置优化器:
    - m, v 从零开始, 偏差校正会使初始更新偏大
    - 学习率调度从头开始 (warmup 又来一遍)
    - loss 可能先上升再下降 ("断点跳跃")
  保存 + 恢复优化器状态可以 "无缝" 继续训练。

代码位置: homework/checkpoint.py
"""


def demo_checkpoint():
    """演示模型检查点的保存和加载。"""
    import io
    from homework.checkpoint import save_checkpoint, load_checkpoint

    print("\n" + "=" * 60)
    print("B6. Checkpoint (断点续训)")
    print("=" * 60)

    # 创建一个小模型
    model = nn.Linear(4, 2, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 模拟训练几步
    for _ in range(5):
        optimizer.zero_grad()
        x = torch.randn(3, 4)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

    # 保存
    buffer = io.BytesIO()
    save_checkpoint(model, optimizer, iteration=5, out=buffer)

    print(f"\n  训练 5 步后:")
    print(f"    模型参数:\n      {model.weight.data}")
    print(f"    Checkpoint 大小: {buffer.tell()} bytes")

    # 修改模型 (模拟训练中断后的新实例)
    model_new = nn.Linear(4, 2, bias=False)
    optimizer_new = torch.optim.AdamW(model_new.parameters(), lr=1e-3)

    print(f"\n  新实例 (未加载 checkpoint):")
    print(f"    模型参数:\n      {model_new.weight.data}")

    # 加载
    buffer.seek(0)
    iteration = load_checkpoint(buffer, model_new, optimizer_new)

    print(f"\n  加载 checkpoint 后:")
    print(f"    模型参数:\n      {model_new.weight.data}")
    print(f"    恢复到步数: {iteration}")

    # 验证参数一致
    match = torch.allclose(model.weight.data, model_new.weight.data)
    print(f"\n  参数完全匹配: {'YES' if match else 'NO'}")
    print(f"\n  保存的内容:")
    print(f"    - model_state_dict: 所有可学习参数")
    print(f"    - optimizer_state_dict: Adam 的 m, v 矩估计")
    print(f"    - iteration: 当前步数 (用于恢复 lr schedule)")


# ============================================================================
# Part C: 资源核算
# ============================================================================

# ── C1. 内存分析 ──────────────────────────────────────
"""
★ 为什么要做这个?
  GPU 显存是训练大模型最稀缺的资源。一张 A100 只有 80GB,
  但模型参数 + 梯度 + 优化器状态 + 激活值可能需要数百 GB。
  不算清楚内存, 你甚至不知道能不能把模型放进 GPU, 最大 batch_size 是多少。
  内存分析让你在写代码之前就知道: 需要多少卡? 要不要用梯度累积?
  要不要用混合精度? 这是设计训练方案的第一步。

使用 AdamW + float32 训练一个 Transformer LM, 峰值内存分为四部分:

  1. 参数 (Parameters):     4P bytes
     P = V·d + L·(12d² + 2d) + d
     └ embedding  └ 每层: 4个投影d² + FFN 8d² + 2个 RMSNorm d

  2. 梯度 (Gradients):      4P bytes
     每个参数对应一个梯度, 大小相同

  3. 优化器状态 (Optimizer): 8P bytes
     AdamW 需要 m + v, 各 4P bytes

  4. 激活值 (Activations):
     每层: 16BTd + 2BHT²  (注意力分数 BHT² 在长序列时很大!)
     输出: BTd + 2BTV
     总计: L·(16BTd + 2BHT²) + BTd + 2BTV

  总计: 16P + 4·Activations bytes
"""


def demo_memory_analysis():
    """计算 GPT-2 XL 的内存需求。"""
    print("\n" + "=" * 60)
    print("C1. 内存分析 (GPT-2 XL)")
    print("=" * 60)

    V, T, L, d, H = 50257, 1024, 48, 1600, 25

    # 参数量
    P = V * d + L * (12 * d**2 + 2 * d) + d
    print(f"\n  GPT-2 XL: V={V}, T={T}, L={L}, d={d}, H={H}")
    print(f"\n  参数量 P = {P:,} ({P / 1e9:.2f}B)")

    # 各部分内存
    mem_params = 4 * P
    mem_grads = 4 * P
    mem_optim = 8 * P
    mem_static = mem_params + mem_grads + mem_optim

    print(f"\n  静态内存 (与 batch_size 无关):")
    print(f"    参数:     4P = {mem_params / 1e9:.2f} GB")
    print(f"    梯度:     4P = {mem_grads / 1e9:.2f} GB")
    print(f"    优化器:   8P = {mem_optim / 1e9:.2f} GB")
    print(f"    ────────────────────")
    print(f"    合计:    16P = {mem_static / 1e9:.2f} GB")

    # 激活值 (与 batch_size 成正比)
    act_per_B = L * (16 * T * d + 2 * H * T**2) + T * d + 2 * T * V
    mem_act_per_B = 4 * act_per_B

    print(f"\n  动态内存 (激活值, 与 B 成正比):")
    print(f"    每个样本: {mem_act_per_B / 1e9:.2f} GB")

    # 最大 batch size
    gpu_mem = 80  # A100 80GB
    max_B = (gpu_mem * 1e9 - mem_static) / mem_act_per_B

    print(f"\n  A100 (80GB) 最大 batch_size:")
    print(f"    80 GB = {mem_static / 1e9:.2f} + {mem_act_per_B / 1e9:.2f} × B")
    print(f"    B ≤ {max_B:.2f}")
    print(f"    最大 batch_size = {int(max_B)}")

    print(f"\n  结论: 单卡只能放 {int(max_B)} 个样本!")
    print(f"    实际训练用 batch_size=1024 需要:")
    print(f"    - 梯度累积 (Gradient Accumulation)")
    print(f"    - 数据并行 (Data Parallelism)")
    print(f"    - 激活值重计算 (Gradient Checkpointing)")


# ── C2. FLOPs 分析 ────────────────────────────────────
"""
★ 为什么要做这个?
  即使内存够用, 训练还受限于计算速度。FLOPs 分析告诉你:
  训练需要多少总计算量? 单卡要跑多久? 需要多少卡才能在合理时间内完成?
  这直接决定了训练预算 (GPU 租用费用) 和项目时间表。
  理解 FLOPs 也帮你判断: 模型的计算瓶颈在哪里? 注意力 O(T²d) 还是 FFN O(Td²)?
  从而做出明智的架构选择 (如 Flash Attention, GQA 等优化)。

前向传播的 FLOPs (矩阵乘法主导):

  矩阵乘法 Y = X @ W, X∈(m,k), W∈(k,n): FLOPs = 2mkn

  每个 Transformer Block:
    QKV 投影:   3 × 2BTd² = 6BTd²
    Q^T K:      2BT²d
    Attn × V:   2BT²d
    输出投影:   2BTd²
    FFN W1:     2BT·d·4d = 8BTd²
    FFN W2:     2BT·4d·d = 8BTd²
    ─────────
    每层合计:   24BTd² + 4BT²d

  输出嵌入:     2BTVd

  前向 FLOPs = L·(24BTd² + 4BT²d) + 2BTVd
  反向 ≈ 2× 前向
  总计 ≈ 3× 前向
"""


def demo_flops_analysis():
    """计算训练 GPT-2 XL 所需的 FLOPs 和时间。"""
    print("\n" + "=" * 60)
    print("C2. FLOPs 分析 (GPT-2 XL 训练)")
    print("=" * 60)

    V, T, L, d, H = 50257, 1024, 48, 1600, 25
    B = 1024
    steps = 400_000

    # 前向 FLOPs
    per_block = 24 * B * T * d**2 + 4 * B * T**2 * d
    F_forward = L * per_block + 2 * B * T * V * d
    F_step = 3 * F_forward  # 前向 + 反向(2x)

    total_flops = steps * F_step

    # A100 吞吐
    peak_tflops = 19.5  # TFLOP/s (float32)
    mfu = 0.5  # 模型利用率
    effective = peak_tflops * 1e12 * mfu

    time_sec = total_flops / effective
    time_days = time_sec / 86400

    print(f"\n  训练配置: B={B}, steps={steps:,}")
    print(f"\n  每步 FLOPs: {F_step:.3e}")
    print(f"  总 FLOPs:   {total_flops:.3e}")
    print(f"\n  A100 (50% MFU): {effective:.2e} FLOP/s")
    print(f"  每步耗时: {F_step / effective:.1f} 秒")
    print(f"\n  单卡训练时间: {time_days:,.0f} 天 ≈ {time_days / 365:.1f} 年")
    print(f"\n  这就是为什么需要多卡并行!")
    print(f"    用 256 张 A100: {time_days / 256:,.0f} 天 ≈ {time_days / 256:.1f} 天")
    print(f"    总 tokens: {steps * B * T / 1e9:.1f}B")


# ============================================================================
# 完整训练循环伪代码
# ============================================================================
"""
── 把所有模块串联: 完整训练循环 ─────────────────────────

  # 准备
  tokenizer = Tokenizer.from_files(vocab_path, merges_path)
  dataset = np.memmap(tokens_path, dtype=np.uint16)
  model = TransformerLM(vocab_size, context_length, d_model, ...)
  optimizer = AdamW(model.parameters(), lr=max_lr, weight_decay=0.1)

  for step in range(total_steps):
      # 1. 学习率调度
      lr = lr_cosine_schedule(step, max_lr, min_lr, warmup, total)
      for pg in optimizer.param_groups:
          pg['lr'] = lr

      # 2. 获取 mini-batch
      x, y = get_batch(dataset, batch_size, context_length, device)

      # 3. 前向传播
      logits = model(x)
      loss = cross_entropy(logits.view(-1, V), y.view(-1))

      # 4. 反向传播
      loss.backward()

      # 5. 梯度裁剪
      gradient_clipping(model.parameters(), max_norm=1.0)

      # 6. 优化器更新
      optimizer.step()
      optimizer.zero_grad()

      # 7. 定期保存 checkpoint
      if step % save_every == 0:
          save_checkpoint(model, optimizer, step, f"ckpt_{step}.pt")

数据流总览:
  文本 → [BPE 分词] → token 序列 → [get_batch 采样]
  → input_ids → [Transformer LM] → logits → [Cross-Entropy] → loss
  → [反向传播] → gradients → [梯度裁剪] → clipped_grads
  → [AdamW + LR Schedule] → 更新参数 → [Checkpoint 保存]
"""


# ============================================================================
# 主入口
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("CS336 Assignment 1: BPE 分词器 & 训练基础设施")
    print("=" * 60)

    # ── Part A: BPE 分词器 ──
    print("\n" + ">" * 60)
    print("Part A: BPE 分词器")
    print(">" * 60)

    demo_unicode_basics()
    demo_pretokenization()
    demo_bpe_training()
    demo_bpe_encode_decode()

    # ── Part B: 训练基础设施 ──
    print("\n\n" + ">" * 60)
    print("Part B: 训练基础设施")
    print(">" * 60)

    demo_cross_entropy()
    demo_data_loading()
    demo_adamw()
    demo_lr_schedule()
    demo_gradient_clipping()
    demo_checkpoint()

    # ── Part C: 资源核算 ──
    print("\n\n" + ">" * 60)
    print("Part C: 资源核算")
    print(">" * 60)

    demo_memory_analysis()
    demo_flops_analysis()

    # ── 总结 ──
    print("\n\n" + "=" * 60)
    print("学习路径建议")
    print("=" * 60)
    print("""
  分词器:
    1. 运行本文件, 理解 BPE 的每一步
    2. 阅读 homework/bpe_tokenizer.py → 完整实现
    3. 关注增量更新优化 → 性能从 O(merges*corpus) 到 O(affected)
    4. 对比 TinyStories vs OpenWebText 词汇表差异

  训练基础设施:
    5. 阅读 homework/cross_entropy.py → 数值稳定性技巧
    6. 阅读 homework/adamw.py → 理解自适应学习率 + 解耦 weight decay
    7. 阅读 homework/lr_schedule.py → 理解 warmup 的必要性
    8. 阅读 homework/gradient_clipping.py → 防止梯度爆炸
    9. 串联 learnTransformer.py → 理解完整的训练流水线

  资源核算:
    10. 用 GPT-2 XL 参数代入公式, 建立直觉
    11. 理解为什么大模型训练需要多卡并行
    """)
