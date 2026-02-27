"""
BPE 分词器 - 互动学习脚本

适合：懂一点 Python 的同学
目标：通过动手实验理解 BPE 算法

运行方法：
    python learn_bpe_step_by_step.py
"""

print("=" * 60)
print("BPE 分词器 - 互动学习")
print("=" * 60)

# ==================== 第1课：理解什么是字节 ====================

print("\n【第1课】理解字节和字符串")
print("-" * 40)

text = "hello"
print(f"原始字符串: {text}")
print(f"字符串类型: {type(text)}")

# 转换为字节
bytes_text = text.encode("utf-8")
print(f"\n编码为字节: {bytes_text}")
print(f"字节类型: {type(bytes_text)}")

# 拆分成单个字节
byte_list = [bytes([b]) for b in bytes_text]
print(f"\n拆分成单字节列表:")
for i, b in enumerate(byte_list):
    print(f"  位置{i}: {b} (值: {ord(b.decode())})")

print("\n💡 关键概念:")
print("  - 字符串 'hello' → 字节 b'hello'")
print("  - 每个字符对应1个字节 (英文)")


# ==================== 第2课：统计相邻对 ====================

print("\n\n【第2课】统计相邻字节对")
print("-" * 40)

# 准备一个简单的语料
corpus = ["hello", "hello", "world"]
print(f"语料: {corpus}")

# 统计 pair 频率
from collections import Counter

all_pairs = []
for word in corpus:
    word_bytes = [bytes([b]) for b in word.encode("utf-8")]
    pairs = [(word_bytes[i], word_bytes[i+1]) for i in range(len(word_bytes)-1)]
    all_pairs.extend(pairs)

pair_counts = Counter(all_pairs)
print(f"\n所有相邻对及频率:")
for pair, count in pair_counts.most_common():
    print(f"  {pair[0].decode()} + {pair[1].decode()}: {count}次")

print("\n💡 BPE 核心思想:")
print(f"  - 最频繁的对是: {pair_counts.most_common(1)[0][0]}")
print("  - 我们应该先合并这个!")


# ==================== 第3课：手动模拟一次 BPE 合并 ====================

print("\n\n【第3课】手动模拟一次 BPE 合并")
print("-" * 40)

# 初始状态
vocab = {i: bytes([i]) for i in range(256)}  # 基础256字节
sequences = [
    tuple(bytes([b]) for b in "hello".encode("utf-8")),
    tuple(bytes([b]) for b in "world".encode("utf-8")),
]

print(f"初始词汇表大小: {len(vocab)}")
print(f"初始序列:")
for seq in sequences:
    print(f"  {seq}")

# 第1次合并
print("\n--- 第1次合并 ---")

# 统计所有 pairs
pair_counts = Counter()
for seq in sequences:
    for i in range(len(seq) - 1):
        pair = (seq[i], seq[i+1])
        pair_counts[pair] += 1

print("Pair 频率统计:")
for pair, count in pair_counts.most_common(5):
    print(f"  {pair[0].decode()} + {pair[1].decode()}: {count}次")

# 选择最频繁的 pair
best_pair = max(pair_counts, key=pair_counts.get)
print(f"\n选择最频繁的 pair: {best_pair[0].decode()} + {best_pair[1].decode()}")

# 创建新 token
new_token = best_pair[0] + best_pair[1]
new_id = len(vocab)
vocab[new_id] = new_token

print(f"创建新 token: ID={new_id}, 值={new_token}")

# 在序列中合并
def merge_in_sequence(seq, pair, new_token):
    """在序列中合并指定的 pair"""
    new_seq = []
    i = 0
    while i < len(seq):
        if i < len(seq) - 1 and seq[i] == pair[0] and seq[i+1] == pair[1]:
            new_seq.append(new_token)
            i += 2
        else:
            new_seq.append(seq[i])
            i += 1
    return tuple(new_seq)

sequences = [merge_in_sequence(seq, best_pair, new_token) for seq in sequences]

print(f"\n合并后的序列:")
for seq in sequences:
    print(f"  {seq}")

print(f"\n词汇表现在大小: {len(vocab)}")


# ==================== 第4课：完整 BPE 训练循环 ====================

print("\n\n【第4课】完整 BPE 训练循环")
print("-" * 40)

# 重新初始化
vocab = {i: bytes([i]) for i in range(256)}
sequences = [
    tuple(bytes([b]) for b in word.encode("utf-8"))
    for word in ["hello", "hello", "world", "hell", "hello"]
]

target_vocab_size = 260  # 只合并4次
merges = []

print(f"目标词汇表大小: {target_vocab_size}")
print(f"初始词汇表: {len(vocab)} 个 tokens")

iteration = 0
while len(vocab) < target_vocab_size:
    iteration += 1

    # 统计 pair 频率
    pair_counts = Counter()
    for seq in sequences:
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i+1])
            pair_counts[pair] += 1

    if not pair_counts:
        break

    # 选择最频繁的 pair
    best_pair = max(pair_counts, key=pair_counts.get)
    max_count = pair_counts[best_pair]

    # 创建新 token
    new_token = best_pair[0] + best_pair[1]
    new_id = len(vocab)
    vocab[new_id] = new_token

    # 记录合并
    merges.append(best_pair)

    # 在所有序列中合并
    sequences = [merge_in_sequence(seq, best_pair, new_token) for seq in sequences]

    print(f"\n迭代 {iteration}:")
    print(f"  合并: {best_pair[0].decode()} + {best_pair[1].decode()} → {new_token.decode()}")
    print(f"  频率: {max_count}次")
    print(f"  新 ID: {new_id}")
    print(f"  词汇表大小: {len(vocab)}")

print("\n" + "=" * 60)
print("📊 最终结果:")
print("=" * 60)
print(f"词汇表大小: {len(vocab)}")
print(f"\n合并规则:")
for i, (p1, p2) in enumerate(merges):
    print(f"  {i+1}. {p1.decode()} + {p2.decode()} → {(p1+p2).decode()}")

print(f"\n最终序列:")
for seq in sequences:
    decoded = b''.join(seq).decode("utf-8")
    print(f"  {list(seq)} → '{decoded}'")


# ==================== 第5课：为什么要字节级别？ ====================

print("\n\n【第5课】为什么用字节级别？")
print("-" * 40)

print("问题1: 中文字符怎么办？")
chinese = "你好"
chinese_bytes = chinese.encode("utf-8")
print(f"  '{chinese}' → {chinese_bytes}")
print(f"  占用 {len(chinese_bytes)} 个字节")

print("\n问题2: Emoji 怎么办？")
emoji = "😊"
emoji_bytes = emoji.encode("utf-8")
print(f"  '{emoji}' → {emoji_bytes}")
print(f"  占用 {len(emoji_bytes)} 个字节")

print("\n💡 字节级别的优势:")
print("  - 统一处理所有语言")
print("  - 不需要预先知道词汇表")
print("  - 可以处理任意 Unicode 字符")


# ==================== 第6课：理解增量更新优化 ====================

print("\n\n【第6课】理解增量更新优化（进阶）")
print("-" * 40)

print("朴素方法（慢）:")
print("  每次合并都重新统计所有 pairs")
print("  时间复杂度: O(merges × corpus_size)")

print("\n优化方法（快）:")
print("  只更新受影响的 pairs")
print("  例如合并 (e,r) → er")
print("  - 删除: (h,e), (e,r), (r,ing)")
print("  - 新增: (h,er), (er,ing)")
print("  - 其他 pairs 不变!")

print("\n💡 性能提升:")
print("  朴素: 3秒")
print("  优化: 0.38秒")
print("  加速: ~8倍")


# ==================== 总结 ====================

print("\n\n" + "=" * 60)
print("🎓 恭喜！你已经理解了 BPE 的核心概念")
print("=" * 60)

print("\n下一步建议:")
print("  1. 阅读 bpe_tokenizer.py 的 run_train_bpe 函数")
print("  2. 对比每个步骤和这里的示例")
print("  3. 尝试修改代码，观察效果变化")

print("\n关键要点:")
print("  ✅ BPE = 不断合并最频繁的字节对")
print("  ✅ 字节级别 = 可以处理任意语言")
print("  ✅ 增量更新 = 只更新受影响的部分")
print("\n")
