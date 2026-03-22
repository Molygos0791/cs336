# 备份文件：优化后的 BPE 实现
import os
from collections import defaultdict
from typing import Dict, List, Tuple
import regex


GPT2_PATTERN = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _split_by_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """按特殊 tokens 分割文本"""
    if not special_tokens:
        return [text]
    sorted_tokens = sorted(special_tokens, key=len, reverse=True)
    split_pattern = "|".join(regex.escape(tok) for tok in sorted_tokens)
    return regex.split(split_pattern, text)


def _pretokenize_chunk(chunk: str) -> list[str]:
    """使用 GPT-2 regex 对文本块进行预分词"""
    return regex.findall(GPT2_PATTERN, chunk)


def _preprocess_text(text: str, special_tokens: list[str]) -> list[str]:
    """统一的文本预处理流程：分割特殊 tokens + GPT-2 预分词"""
    chunks = _split_by_special_tokens(text, special_tokens)
    result = []
    for chunk in chunks:
        if chunk not in special_tokens:
            result.extend(_pretokenize_chunk(chunk))
    return result


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练一个字节级别的 BPE 分词器（优化版本）
    """
    # 参数校验
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("vocab_size 必须是正整数")

    # 初始化词汇表
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_token_id: int = 256
    existing_byte_values = set(vocab.values())

    # 添加特殊 tokens
    for st_str in special_tokens:
        if len(vocab) >= vocab_size:
            break
        st_bytes = st_str.encode("utf-8")
        if st_bytes not in existing_byte_values:
            vocab[next_token_id] = st_bytes
            existing_byte_values.add(st_bytes)
            next_token_id += 1

    # 读取训练语料
    try:
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except FileNotFoundError:
        text = ""

    # 预处理和预分词（使用优化后的统一函数）
    pre_tokens = _preprocess_text(text, special_tokens)
    token_frequency_table = defaultdict(int)

    for word in pre_tokens:
        word_bytes = word.encode("utf-8")
        byte_tuple = tuple(bytes([b]) for b in word_bytes)
        token_frequency_table[byte_tuple] += 1

    # 初始化 pair 频率统计
    pair_counts = defaultdict(int)
    for byte_tuple, freq in token_frequency_table.items():
        for i in range(len(byte_tuple) - 1):
            pair = (byte_tuple[i], byte_tuple[i + 1])
            pair_counts[pair] += freq

    # BPE 迭代合并
    merges: List[Tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        # 选择最频繁的 pair
        max_count = max(pair_counts.values())
        candidates = [pair for pair, count in pair_counts.items() if count == max_count]
        best_pair = max(candidates)

        # 创建新 token
        new_token_bytes = best_pair[0] + best_pair[1]
        vocab[next_token_id] = new_token_bytes
        next_token_id += 1
        merges.append(best_pair)

        # 增量更新 pair_counts
        affected_tokens = []
        for byte_tuple, freq in token_frequency_table.items():
            has_pair = False
            for i in range(len(byte_tuple) - 1):
                if byte_tuple[i:i+2] == best_pair:
                    has_pair = True
                    break
            if has_pair:
                affected_tokens.append((byte_tuple, freq))

        for byte_tuple, freq in affected_tokens:
            # 删除旧的 pair 计数
            for i in range(len(byte_tuple) - 1):
                old_pair = (byte_tuple[i], byte_tuple[i + 1])
                pair_counts[old_pair] -= freq
                if pair_counts[old_pair] <= 0:
                    del pair_counts[old_pair]

            # 合并 pair
            new_byte_tuple = _merge_pair_in_sequence(byte_tuple, best_pair, new_token_bytes)

            # 添加新的 pair 计数
            for i in range(len(new_byte_tuple) - 1):
                new_pair = (new_byte_tuple[i], new_byte_tuple[i + 1])
                pair_counts[new_pair] += freq

            # 更新频率表
            del token_frequency_table[byte_tuple]
            token_frequency_table[new_byte_tuple] += freq

    return vocab, merges


def _merge_pair_in_sequence(
    byte_sequence: tuple[bytes, ...],
    pair: tuple[bytes, bytes],
    new_token: bytes,
) -> tuple[bytes, ...]:
    """在字节序列中合并指定的 pair（优化版本）"""
    pair_len = len(byte_sequence)
    new_sequence = []
    i = 0

    # 提取 pair 的两个元素（优化：避免重复索引访问）
    p1, p2 = pair

    while i < pair_len:
        if i + 1 < pair_len and byte_sequence[i] == p1 and byte_sequence[i + 1] == p2:
            new_sequence.append(new_token)
            i += 2
        else:
            new_sequence.append(byte_sequence[i])
            i += 1

    return tuple(new_sequence)


def save_vocab_and_merges(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_path: str,
    merges_path: str,
):
    """保存词汇表和合并规则到文件"""
    import json

    vocab_str = {
        token_id: token_bytes.decode("utf-8", errors="replace")
        for token_id, token_bytes in vocab.items()
    }
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_str, f, ensure_ascii=False, indent=2)

    with open(merges_path, "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            p1_str = p1.decode("utf-8", errors="replace")
            p2_str = p2.decode("utf-8", errors="replace")
            f.write(f"{p1_str} {p2_str}\n")
