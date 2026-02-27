"""
BPE (Byte Pair Encoding) 分词器训练实现

本模块实现了字节级别的 BPE 分词器训练算法，包括：
1. 字节级别的 BPE 训练
2. 特殊 tokens 处理
3. GPT-2 风格的预分词
4. 增量更新优化

主要参考: Sennrich et al., 2016 - https://arxiv.org/abs/1508.07909
"""

import os
import re
from collections import defaultdict
from typing import List, Tuple, Dict
import regex

# GPT-2 预分词模式
# 这个正则表达式保留了缩写、标点符号、空格等语言学特征
GPT2_PATTERN = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练一个字节级别的 BPE 分词器

    ================== 算法流程 ==================

    1. 初始化词汇表：
       - 256 个基础字节 (0-255)
       - 特殊 tokens（优先添加）

    2. 预处理语料：
       - 按特殊 tokens 分割文本
       - 使用 GPT-2 regex 预分词
       - 转换为字节序列

    3. 迭代合并：
       重复直到词汇表达到 vocab_size：
       a. 统计相邻字节对频率
       b. 选择最频繁的字节对
       c. 创建新 token = pair[0] + pair[1]
       d. 在所有序列中合并该 pair
       e. 增量更新 pair 频率（优化）

    ===============================================

    参数：
        input_path: 训练文本文件路径
        vocab_size: 目标词汇表大小（包括基础字节、特殊 tokens、合并产生的新 tokens）
        special_tokens: 特殊 token 列表，如 ["<eos>", "<pad>"]

    返回：
        vocab: dict[int, bytes] - Token ID 到 Token 字节的映射
               例如: {0: b'\x00', ..., 256: b'<eos>', 257: b'ing', ...}

        merges: list[tuple[bytes, bytes]] - 按合并顺序排列的合并规则
               例如: [(b'e', b'r'), (b'er', b'ing'), ...]
               表示 b'e' + b'er' -> b'ering'
    """

    # ==================== 第1步：参数校验 ====================
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("vocab_size 必须是正整数")

    # ==================== 第2步：初始化词汇表 ====================
    # 基础词汇表：所有 256 个字节
    # 每个字节是一个独立的 token，ID 从 0 到 255
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_token_id: int = 256  # 新 token ID 从 256 开始

    # 集合用于快速检查字节值是否已存在
    existing_byte_values = set(vocab.values())

    # 添加特殊 tokens 到词汇表（优先级最高）
    for st_str in special_tokens:
        # 如果词汇表已满，停止添加
        if len(vocab) >= vocab_size:
            break

        st_bytes = st_str.encode("utf-8")  # 将特殊 token 转为字节

        # 避免重复：只在字节值不存在时添加
        # 例如：特殊 token "a" 和基础字节 b'a' 不能重复
        if st_bytes not in existing_byte_values:
            vocab[next_token_id] = st_bytes
            existing_byte_values.add(st_bytes)
            next_token_id += 1

    # ==================== 第3步：读取训练语料 ====================
    try:
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except FileNotFoundError:
        text = ""  # 文件不存在时视为空文本

    # ==================== 第4步：预处理和预分词 ====================

    # 4.1 按特殊 tokens 分割文本
    # 关键：防止跨特殊 token 边界进行合并
    # 例如："Hello<eos>World" 应该分割为 ["Hello", "World"]
    if special_tokens:
        # 构建分割模式，用 | 连接所有特殊 tokens
        # regex.escape 确保 | 等特殊字符被正确转义
        split_pattern = "|".join(re.escape(tok) for tok in special_tokens)
        chunks = regex.split(split_pattern, text)
    else:
        chunks = [text]

    # 4.2 对每个 chunk 进行预分词
    # 使用 GPT-2 regex 模式进行预分词
    # 这会保留语言学特征，如缩写、标点、空格等
    token_frequency_table = defaultdict(int)  # 统计每个字节序列的频率

    for chunk in chunks:
        # 使用 GPT-2 regex 找到所有预 tokens
        for word in regex.findall(GPT2_PATTERN, chunk):
            # 将预 token 转换为字节序列
            word_bytes = word.encode("utf-8")

            # 转换为字节元组（可哈希，可作为字典键）
            # 例如：b'hello' -> (b'h', b'e', b'l', b'l', b'o')
            byte_tuple = tuple(bytes([b]) for b in word_bytes)

            # 统计频率
            token_frequency_table[byte_tuple] += 1

    # ==================== 第5步：初始化 pair 频率统计 ====================
    # 使用 defaultdict 方便计数
    pair_counts = defaultdict(int)

    # 遍历所有字节序列，统计相邻字节对频率
    for byte_tuple, freq in token_frequency_table.items():
        for i in range(len(byte_tuple) - 1):
            pair = (byte_tuple[i], byte_tuple[i + 1])
            pair_counts[pair] += freq

    # ==================== 第6步：BPE 迭代合并 ====================
    merges: List[Tuple[bytes, bytes]] = []  # 记录合并操作

    while len(vocab) < vocab_size:
        # 6.1 检查终止条件
        if not pair_counts:
            break  # 没有更多 pair 可以合并

        # 6.2 选择最频繁的 pair
        # 找到最大频率
        max_count = max(pair_counts.values())

        # 可能存在多个 pair 具有相同最高频率
        # 选择字典序最大的 pair（确保与参考实现一致）
        candidates = [pair for pair, count in pair_counts.items() if count == max_count]
        best_pair = max(candidates)  # 元组比较：逐元素比较，选择最大的

        # 6.3 创建新 token
        new_token_bytes = best_pair[0] + best_pair[1]

        # 添加到词汇表
        vocab[next_token_id] = new_token_bytes
        next_token_id += 1

        # 记录合并操作（按顺序）
        merges.append(best_pair)

        # 6.4 增量更新 pair_counts（性能优化的关键）
        # 只有包含 best_pair 的字节序列才会受影响
        # 只需更新这些受影响的序列，而非全部重新统计

        # 找出所有包含 best_pair 的字节序列
        affected_tokens = []
        for byte_tuple, freq in token_frequency_table.items():
            # 检查字节序列中是否包含 best_pair
            has_pair = False
            for i in range(len(byte_tuple) - 1):
                if byte_tuple[i:i+2] == best_pair:
                    has_pair = True
                    break
            if has_pair:
                affected_tokens.append((byte_tuple, freq))

        # 对每个受影响的字节序列进行更新
        for byte_tuple, freq in affected_tokens:
            # a. 删除旧的 pair 计数
            for i in range(len(byte_tuple) - 1):
                old_pair = (byte_tuple[i], byte_tuple[i + 1])
                pair_counts[old_pair] -= freq
                # 删除计数为 0 或负的 pair（清理）
                if pair_counts[old_pair] <= 0:
                    del pair_counts[old_pair]

            # b. 在字节序列中合并 best_pair
            new_byte_tuple = _merge_pair_in_sequence(byte_tuple, best_pair, new_token_bytes)

            # c. 添加新的 pair 计数
            for i in range(len(new_byte_tuple) - 1):
                new_pair = (new_byte_tuple[i], new_byte_tuple[i + 1])
                pair_counts[new_pair] += freq

            # d. 更新字节序列频率表
            del token_frequency_table[byte_tuple]
            token_frequency_table[new_byte_tuple] += freq

    # ==================== 第7步：返回结果 ====================
    return vocab, merges


def _merge_pair_in_sequence(
    byte_sequence: Tuple[bytes, ...],
    pair: Tuple[bytes, bytes],
    new_token: bytes,
) -> Tuple[bytes, ...]:
    """
    在字节序列中合并指定的 pair

    辅助函数：将所有出现的 pair 替换为 new_token

    参数：
        byte_sequence: 字节元组，如 (b'h', b'e', b'l', b'l', b'o')
        pair: 要合并的字节对，如 (b'l', b'l')
        new_token: 合并后的新 token，如 b'll'

    返回：
        合并后的新字节元组，如 (b'h', b'e', b'll', b'o')

    示例：
        >>> _merge_pair_in_sequence((b'h', b'e', b'l', b'l', b'o'), (b'l', b'l'), b'll')
        (b'h', b'e', b'll', b'o')
    """
    new_sequence = []
    i = 0

    while i < len(byte_sequence):
        # 检查当前位置是否是要合并的 pair
        if i < len(byte_sequence) - 1 and byte_sequence[i] == pair[0] and byte_sequence[i + 1] == pair[1]:
            # 找到要合并的 pair，替换为新 token
            new_sequence.append(new_token)
            i += 2  # 跳过两个元素
        else:
            # 不匹配，保留当前元素
            new_sequence.append(byte_sequence[i])
            i += 1

    return tuple(new_sequence)


# ==================== 辅助函数：保存和加载 ====================

def save_vocab_and_merges(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_path: str,
    merges_path: str,
):
    """
    保存词汇表和合并规则到文件

    参数：
        vocab: 词汇表
        merges: 合并规则列表
        vocab_path: 词汇表保存路径
        merges_path: 合并规则保存路径
    """
    import json

    # 保存词汇表（可读格式）
    # 将 bytes 转换为字符串以便存储
    vocab_str = {
        token_id: token_bytes.decode("utf-8", errors="replace")
        for token_id, token_bytes in vocab.items()
    }
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_str, f, ensure_ascii=False, indent=2)

    # 保存合并规则
    with open(merges_path, "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            p1_str = p1.decode("utf-8", errors="replace")
            p2_str = p2.decode("utf-8", errors="replace")
            f.write(f"{p1_str} {p2_str}\n")


# ==================== 测试和调试 ====================

if __name__ == "__main__":
    # 简单测试
    test_text = "hello world hello world"
    test_file = "/tmp/test_bpe.txt"

    # 创建测试文件
    with open(test_file, "w") as f:
        f.write(test_text)

    # 训练 BPE
    vocab, merges = run_train_bpe(
        input_path=test_file,
        vocab_size=300,
        special_tokens=["<eos>"]
    )

    print(f"词汇表大小: {len(vocab)}")
    print(f"合并次数: {len(merges)}")
    print(f"\n前 10 个合并:")
    for i, (p1, p2) in enumerate(merges[:10]):
        print(f"  {i+1}. {p1} + {p2} -> {p1 + p2}")
