"""
BPE Training on TinyStories Dataset

训练字节级别的BPE分词器，并分析结果。
"""

import time
import psutil
import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from bpe_tokenizer import run_train_bpe, save_vocab_and_merges

# ==================== 配置参数 ====================
INPUT_PATH = "/Users/apple/Desktop/cs336/assignment1-basics-main/data/TinyStories-train.txt"
VOCAB_SIZE = 10000
SPECIAL_TOKENS = ["<|endoftext|>"]
VOCAB_OUTPUT = "/Users/apple/Desktop/cs336/assignment1-basics-main/data/tinystories_vocab.json"
MERGES_OUTPUT = "/Users/apple/Desktop/cs336/assignment1-basics-main/data/tinystories_merges.txt"


def main():
    """主函数：执行BPE训练并分析结果"""

    print("=" * 70)
    print("BPE Training on TinyStories Dataset")
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
        print(f"   内存使用: {end_memory:.2f} MB")
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

    # 3. 显示前几个合并规则
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


def performance_analysis():
    """
    性能分析：找出训练过程中的瓶颈

    使用cProfile分析哪个函数最耗时
    """
    import cProfile
    import pstats
    from io import StringIO

    print("\n" + "=" * 70)
    print("🔬 性能分析模式")
    print("=" * 70)

    pr = cProfile.Profile()
    pr.enable()

    # 运行训练（使用较小的词汇表以加快分析）
    vocab, merges = run_train_bpe(
        input_path=INPUT_PATH,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS
    )

    pr.disable()

    # 验证训练结果
    print(f"\n✅ 性能分析训练完成:")
    print(f"   词汇表大小: {len(vocab):,}")
    print(f"   合并次数: {len(merges):,}")

    # 打印性能分析结果
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # 打印前30个最耗时的函数

    print("\n📊 性能分析结果 (按累计时间排序, 前30名):")
    print("-" * 70)
    print(s.getvalue())


if __name__ == "__main__":
    # 可以选择运行模式
    import argparse

    parser = argparse.ArgumentParser(description='BPE Training on TinyStories')
    parser.add_argument('--profile', action='store_true',
                       help='启用性能分析模式')
    parser.add_argument('--quick-test', action='store_true',
                       help='快速测试模式 (只使用前1000行)')

    args = parser.parse_args()

    if args.profile:
        performance_analysis()
    elif args.quick_test:
        print("\n⚡ 快速测试模式 - 只使用前1000行\n")
        # 创建临时测试文件
        test_file = "/tmp/tinystories_test.txt"
        with open(INPUT_PATH, "r", encoding="utf-8", errors="ignore") as f:
            lines = [f.readline() for _ in range(1000)]

        with open(test_file, "w", encoding="utf-8") as f:
            f.writelines(lines)

        # 修改输入路径
        INPUT_PATH = test_file
        VOCAB_OUTPUT = "/tmp/tinystories_test_vocab.json"
        MERGES_OUTPUT = "/tmp/tinystories_test_merges.txt"

        main()

        print(f"\n✅ 快速测试完成! 测试文件已保存到: {test_file}")
    else:
        main()
