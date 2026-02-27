#!/usr/bin/env python3
"""
BPE Training on OpenWebText Dataset

作业要求：
- 在 OpenWebText 数据集上训练字节级别的 BPE 分词器
- 词汇表大小：32,000
- 分析最长 token
- 比较 TinyStories vs OpenWebText 分词器
"""

import time
import psutil
import os
import sys
from pathlib import Path

# 添加当前目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from bpe_tokenizer import run_train_bpe, save_vocab_and_merges

# ==================== 配置参数 ====================
INPUT_PATH = "/Users/apple/Desktop/cs336/assignment1-basics-main/data/owt_train.txt"
VOCAB_SIZE = 32000
SPECIAL_TOKENS = ["<|endoftext|>"]
VOCAB_OUTPUT = "/Users/apple/Desktop/cs336/assignment1-basics-main/data/owt_vocab.json"
MERGES_OUTPUT = "/Users/apple/Desktop/cs336/assignment1-basics-main/data/owt_merges.txt"


def main():
    """主函数：执行 BPE 训练并分析结果"""

    print("=" * 70)
    print("BPE Training on OpenWebText Dataset")
    print("=" * 70)
    print(f"输入文件: {INPUT_PATH}")
    print(f"目标词汇表大小: {VOCAB_SIZE:,}")
    print(f"特殊tokens: {SPECIAL_TOKENS}")
    print("=" * 70)

    # 检查输入文件是否存在
    if not os.path.exists(INPUT_PATH):
        print(f"\n❌ 错误: 输入文件不存在!")
        print(f"   路径: {INPUT_PATH}")
        return

    # 获取文件大小
    file_size = os.path.getsize(INPUT_PATH)
    print(f"\n📊 输入文件信息:")
    print(f"   文件大小: {file_size / 1024 / 1024 / 1024:.2f} GB")

    # 开始训练
    print(f"\n🚀 开始训练...")
    process = psutil.Process(os.getpid())
    start_time = time.time()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB

    try:
        vocab, merges = run_train_bpe(
            input_path=INPUT_PATH,
            vocab_size=VOCAB_SIZE,
            special_tokens=SPECIAL_TOKENS
        )

        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 计算统计信息
        training_time = end_time - start_time
        memory_used = end_memory - start_memory

        print(f"\n✅ 训练成功完成!")
        print(f"=" * 70)
        print(f"📈 训练统计:")
        print(f"   训练时间: {training_time:.2f} 秒 ({training_time / 60:.2f} 分钟, {training_time / 3600:.2f} 小时)")
        print(f"   内存使用: {end_memory:.2f} MB ({end_memory / 1024:.2f} GB)")
        print(f"   内存增长: {memory_used:.2f} MB")
        print(f"   词汇表大小: {len(vocab):,} 个tokens")
        print(f"   合并次数: {len(merges):,} 次")

        # 保存结果
        print(f"\n💾 保存结果到磁盘...")
        save_vocab_and_merges(vocab, merges, VOCAB_OUTPUT, MERGES_OUTPUT)
        print(f"   ✓ 词汇表已保存到: {VOCAB_OUTPUT}")
        print(f"   ✓ 合并规则已保存到: {MERGES_OUTPUT}")

        # 分析词汇表
        print(f"\n🔍 词汇表分析:")
        analyze_vocab(vocab, merges)

        # 分析特殊tokens
        print(f"\n🎯 特殊Tokens验证:")
        verify_special_tokens(vocab, SPECIAL_TOKENS)

        # 与 TinyStories 比较
        print(f"\n📊 与 TinyStories 分词器比较:")
        compare_with_tinystories(vocab)

    except Exception as e:
        print(f"\n❌ 训练过程中发生错误:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


def analyze_vocab(vocab, merges):
    """分析词汇表特征"""

    # 1. 找最长的token
    longest_tokens = sorted(vocab.items(), key=lambda x: len(x[1]), reverse=True)[:10]

    print(f"   前10个最长的tokens:")
    for i, (token_id, token_bytes) in enumerate(longest_tokens, 1):
        try:
            token_str = token_bytes.decode('utf-8')
        except:
            token_str = repr(token_bytes)

        # 显示前100个字符
        display_str = token_str[:100] + "..." if len(token_str) > 100 else token_str
        print(f"      {i}. ID={token_id:5d}, 长度={len(token_bytes):3d} bytes")
        print(f"         内容: {display_str}")

    # 2. 统计token长度分布
    token_lengths = [len(v) for v in vocab.values()]
    print(f"\n   Token长度分布:")
    print(f"      最小长度: {min(token_lengths)} bytes")
    print(f"      最大长度: {max(token_lengths)} bytes")
    print(f"      平均长度: {sum(token_lengths) / len(token_lengths):.2f} bytes")
    print(f"      中位数长度: {sorted(token_lengths)[len(token_lengths) // 2]} bytes")

    # 3. 多字节token统计
    multi_byte_count = sum(1 for length in token_lengths if length > 1)
    print(f"\n   多字节token数量 (>1 byte): {multi_byte_count:,} / {len(vocab):,} ({multi_byte_count/len(vocab)*100:.1f}%)")

    # 4. 显示前几个合并规则
    print(f"\n   前10个合并规则:")
    for i, (p1, p2) in enumerate(merges[:10], 1):
        try:
            p1_str = p1.decode('utf-8', errors='replace')
            p2_str = p2.decode('utf-8', errors='replace')
            merged = (p1 + p2).decode('utf-8', errors='replace')
            print(f"      {i}. {repr(p1_str)} + {repr(p2_str)} → {repr(merged)}")
        except:
            print(f"      {i}. {p1} + {p2} → {p1 + p2}")


def verify_special_tokens(vocab, special_tokens):
    """验证特殊tokens是否正确添加到词汇表"""

    for token_str in special_tokens:
        token_bytes = token_str.encode('utf-8')

        # 查找token ID
        token_id = None
        for tid, tbytes in vocab.items():
            if tbytes == token_bytes:
                token_id = tid
                break

        if token_id is not None:
            print(f"   ✓ '{token_str}'")
            print(f"      ID: {token_id}")
            print(f"      编码长度: {len(token_bytes)} bytes")
        else:
            print(f"   ✗ '{token_str}' 未找到!")


def compare_with_tinystories(owt_vocab):
    """与 TinyStories 分词器比较"""

    ts_vocab_path = "/Users/apple/Desktop/cs336/assignment1-basics-main/data/tinystories_vocab.json"

    if not os.path.exists(ts_vocab_path):
        print(f"   TinyStories 词汇表不存在: {ts_vocab_path}")
        print(f"   请先完成 TinyStories 训练")
        return

    import json

    # 加载 TinyStories 词汇表
    with open(ts_vocab_path, "r", encoding="utf-8") as f:
        ts_vocab_str = json.load(f)

    ts_vocab = {
        int(k): v.encode("utf-8", errors="replace")
        for k, v in ts_vocab_str.items()
    }

    # 比较最长 token
    owt_sorted = sorted(owt_vocab.items(), key=lambda x: len(x[1]), reverse=True)
    ts_sorted = sorted(ts_vocab.items(), key=lambda x: len(x[1]), reverse=True)

    owt_longest = owt_sorted[0]
    ts_longest = ts_sorted[0]

    print(f"   TinyStories 最长token:")
    print(f"      ID={ts_longest[0]}, 长度={len(ts_longest[1])} bytes")
    try:
        ts_str = ts_longest[1].decode('utf-8')
        print(f"      内容: {repr(ts_str[:100])}..." if len(ts_str) > 100 else f"      内容: {repr(ts_str)}")
    except:
        print(f"      内容: {repr(ts_longest[1])}")

    print(f"\n   OpenWebText 最长token:")
    print(f"      ID={owt_longest[0]}, 长度={len(owt_longest[1])} bytes")
    try:
        owt_str = owt_longest[1].decode('utf-8')
        print(f"      内容: {repr(owt_str[:100])}..." if len(owt_str) > 100 else f"      内容: {repr(owt_str)}")
    except:
        print(f"      内容: {repr(owt_longest[1])}")

    # 分析差异
    print(f"\n   对比分析:")
    if len(owt_longest[1]) > len(ts_longest[1]):
        print(f"      ✓ OpenWebText 最长token ({len(owt_longest[1])} bytes) ")
        print(f"        比 TinyStories ({len(ts_longest[1])} bytes) 长 {len(owt_longest[1]) - len(ts_longest[1])} bytes")
    print(f"      ✓ OpenWebText 包含更多样化的文本（网络内容），")
    print(f"        可能有更长的重复模式（URL、代码片段、技术术语等）")
    print(f"      ✓ TinyStories 是简单的儿童故事，语言模式相对简单")


if __name__ == "__main__":
    main()
