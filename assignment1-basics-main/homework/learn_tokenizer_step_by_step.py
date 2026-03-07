"""
Tokenizer 分词器 - 互动学习脚本

适合：已经理解 BPE 训练，想学习如何使用训练好的分词器的同学
目标：通过动手实验理解 Tokenizer 的 encode 和 decode 原理

运行方法：
    python learn_tokenizer_step_by_step.py
"""

print("=" * 60)
print("Tokenizer 分词器 - 互动学习")
print("=" * 60)

# ==================== 第1课：理解词汇表 ====================

print("\n【第1课】理解词汇表 (Vocabulary)")
print("-" * 40)

# 一个简单的词汇表示例
vocab = {
    0: b'a',
    1: b'b',
    2: b'c',
    3: b'ab',
    4: b'abc',
}

print("词汇表示例:")
for token_id, token_bytes in vocab.items():
    print(f"  ID {token_id}: {token_bytes} → '{token_bytes.decode()}'")

print("\n💡 关键概念:")
print("  - 词汇表 = Token ID ↔ Token Bytes 的映射")
print("  - 训练 BPE 后，我们会得到一个词汇表")
print("  - 词汇表包含：基础字节 + 合并产生的新 tokens")


# ==================== 第2课：理解合并规则 ====================

print("\n\n【第2课】理解合并规则 (Merges)")
print("-" * 40)

# 合并规则示例
merges = [
    (b'a', b'b'),      # 第一次合并：a + b → ab
    (b'ab', b'c'),     # 第二次合并：ab + c → abc
]

print("合并规则示例 (按顺序):")
for i, (p1, p2) in enumerate(merges):
    merged = p1 + p2
    print(f"  {i+1}. {p1.decode()} + {p2.decode()} → {merged.decode()}")

print("\n💡 关键概念:")
print("  - 合并规则的顺序很重要！")
print("  - 编码时按顺序应用合并规则")
print("  - 先合并的优先级更高")


# ==================== 第3课：手动模拟 Encode 过程 ====================

print("\n\n【第3课】手动模拟 Encode 过程")
print("-" * 40)

text = "abc"
print(f"输入文本: '{text}'")

# Step 1: 转换为字节
text_bytes = text.encode("utf-8")
print(f"Step 1 - 转为字节: {text_bytes}")

# Step 2: 拆分为单字节 tokens
tokens = [bytes([b]) for b in text_bytes]
print(f"Step 2 - 初始 tokens: {tokens}")

# Step 3: 应用合并规则
print(f"\nStep 3 - 应用合并规则:")

# 创建 bytes -> id 的映射
vocab_bytes_to_id = {v: k for k, v in vocab.items()}

iteration = 0
while True:
    iteration += 1

    # 查找可以合并的 pairs
    mergeable_pairs = []
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        if pair in merges:
            # 找到这个 pair 在 merges 中的位置（优先级）
            priority = merges.index(pair)
            mergeable_pairs.append((priority, i, pair))

    if not mergeable_pairs:
        print(f"  迭代 {iteration}: 没有更多可合并的 pairs")
        break

    # 选择优先级最高的（在 merges 中最靠前的）
    mergeable_pairs.sort()  # 按 priority 排序
    _, best_pos, best_pair = mergeable_pairs[0]

    # 合并
    merged_token = best_pair[0] + best_pair[1]
    print(f"  迭代 {iteration}: 在位置 {best_pos} 合并 {best_pair[0].decode()} + {best_pair[1].decode()} → {merged_token.decode()}")

    # 执行合并
    new_tokens = tokens[:best_pos] + [merged_token] + tokens[best_pos + 2:]
    tokens = new_tokens
    print(f"           当前后 tokens: {tokens}")

# Step 4: 转换为 IDs
print(f"\nStep 4 - 转换为 IDs:")
token_ids = [vocab_bytes_to_id[t] for t in tokens if t in vocab_bytes_to_id]
print(f"  最终 token IDs: {token_ids}")

print("\n💡 Encode 过程总结:")
print("  1. 文本 → 字节")
print("  2. 字节 → 初始单字节 tokens")
print("  3. 按合并规则顺序迭代合并")
print("  4. Tokens → Token IDs")


# ==================== 第4课：理解 Decode 过程 ====================

print("\n\n【第4课】理解 Decode 过程")
print("-" * 40)

token_ids = [4]  # abc 的 ID
print(f"输入 token IDs: {token_ids}")

# Step 1: 从词汇表查找 token bytes
print(f"\nStep 1 - 查找 token bytes:")
all_bytes = []
for tid in token_ids:
    if tid in vocab:
        token_bytes = vocab[tid]
        all_bytes.append(token_bytes)
        print(f"  ID {tid} → {token_bytes}")

# Step 2: 合并所有字节
combined = b"".join(all_bytes)
print(f"\nStep 2 - 合并字节: {combined}")

# Step 3: 解码为文本
decoded_text = combined.decode("utf-8")
print(f"\nStep 3 - 解码为文本: '{decoded_text}'")

print("\n💡 Decode 过程总结:")
print("  1. Token IDs → Token Bytes (查词汇表)")
print("  2. 合并所有 bytes")
print("  3. Bytes → 文本 (UTF-8 解码)")


# ==================== 第5课：理解预分词 (Pre-tokenization) ====================

print("\n\n【第5课】理解预分词 (Pre-tokenization)")
print("-" * 40)

text = "Hello, world!"
print(f"输入: '{text}'")

# GPT-2 预分词模式
import regex
GPT2_PATTERN = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

# 预分词
pre_tokens = regex.findall(GPT2_PATTERN, text)
print(f"\n预分词结果:")
for i, token in enumerate(pre_tokens):
    print(f"  {i+1}. '{token}'")

print("\n💡 为什么要预分词?")
print("  - 把文本分成语言学单元")
print("  - 保持单词边界，避免跨单词合并")
print("  - 例如：'hello' 和 'world' 分别处理，不会产生 'helloworld'")


# ==================== 第6课：处理特殊 Tokens ====================

print("\n\n【第6课】处理特殊 Tokens (Special Tokens)")
print("-" * 40)

# 扩展词汇表，添加特殊 tokens
special_tokens = ["<eos>", "<pad>"]
print(f"特殊 tokens: {special_tokens}")

# 添加到词汇表
vocab_with_special = vocab.copy()
next_id = max(vocab_with_special.keys()) + 1

for st in special_tokens:
    vocab_with_special[next_id] = st.encode("utf-8")
    print(f"  添加 {st} → ID {next_id}")
    next_id += 1

print("\n💡 特殊 tokens 的特点:")
print("  - 不会被拆分")
print("  - 始终作为单个 token")
print("  - 常用: <eos>(结束), <pad>(填充), <unk>(未知)")


# ==================== 第7课：完整的 Encode/Decode 示例 ====================

print("\n\n【第7课】完整的 Encode/Decode 示例")
print("-" * 40)

# 使用一个更真实的词汇表
print("创建一个简单的词汇表...")

# 基础字节 (简化版，只显示部分)
simple_vocab = {i: bytes([i]) for i in range(256)}

# 添加一些合并后的 tokens
simple_vocab[256] = b'he'
simple_vocab[257] = b'll'
simple_vocab[258] = b'o'
simple_vocab[259] = b'hello'

# 创建反向映射
vocab_bytes_to_id = {v: k for k, v in simple_vocab.items()}
merges = [
    (b'h', b'e'),      # he
    (b'l', b'l'),      # ll
]

print("词汇表包含 (部分):")
for tid in [ord('h'), ord('e'), ord('l'), ord('o'), 256, 257, 259]:
    token_bytes = simple_vocab[tid]
    print(f"  ID {tid}: {token_bytes} → '{token_bytes.decode('utf-8', errors='replace')}'")

# Encode "hello"
text = "hello"
print(f"\n📝 Encode: '{text}'")

# 预分词
word_bytes = text.encode("utf-8")
tokens = [bytes([b]) for b in word_bytes]
print(f"  初始 tokens: {[t.decode() for t in tokens]}")

# 应用合并
def apply_bpe_merge(tokens, merges, vocab):
    """应用 BPE 合并"""
    while True:
        # 找到优先级最高的可合并 pair
        best_pair = None
        best_pos = -1
        best_priority = len(merges)

        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            if pair in merges:
                priority = merges.index(pair)
                if priority < best_priority:
                    best_priority = priority
                    best_pair = pair
                    best_pos = i

        if best_pair is None:
            break

        # 合并
        merged = best_pair[0] + best_pair[1]
        tokens = tokens[:best_pos] + [merged] + tokens[best_pos + 2:]

    return tokens

tokens = apply_bpe_merge(tokens, merges, simple_vocab)
print(f"  合并后 tokens: {[t.decode() for t in tokens]}")

# 转为 IDs
token_ids = [vocab_bytes_to_id[t] for t in tokens]
print(f"  Token IDs: {token_ids}")

# Decode
print(f"\n📖 Decode: {token_ids}")
all_bytes = [simple_vocab[tid] for tid in token_ids]
combined = b"".join(all_bytes)
decoded = combined.decode("utf-8")
print(f"  解码结果: '{decoded}'")

print("\n✅ Roundtrip 成功!")


# ==================== 第8课：流式编码大文件 ====================

print("\n\n【第8课】流式编码大文件 (Memory Efficient)")
print("-" * 40)

print("假设我们有一个 1GB 的文本文件:")
print("  ❌ 错误方法: 全部读入内存 → encode()")
print("              需要: 1GB + token_ids 内存")
print("")
print("  ✅ 正确方法: 逐行读取 → encode_iterable()")
print("              需要: 1 行文本 + 当前 token_ids 内存")

# 模拟流式处理
def mock_file_lines():
    """模拟文件行"""
    yield "hello world\n"
    yield "this is a test\n"
    yield "end of file\n"

print("\n示例代码:")
print("  with open('large_file.txt') as f:")
print("      for token_id in tokenizer.encode_iterable(f):")
print("          # 处理 token_id")
print("          pass")

print("\n💡 encode_iterable 的优势:")
print("  - 不会一次性加载整个文件")
print("  - 逐行 yield token IDs")
print("  - 内存占用 = O(单行大小)")


# ==================== 总结 ====================

print("\n\n" + "=" * 60)
print("🎓 恭喜！你已经理解了 Tokenizer 的核心概念")
print("=" * 60)

print("\n关键要点:")
print("  ✅ Encode: 文本 → 预分词 → 字节 → BPE合并 → Token IDs")
print("  ✅ Decode: Token IDs → Token Bytes → 合并 → 文本")
print("  ✅ 预分词: 保持单词边界")
print("  ✅ 特殊 tokens: 不拆分，单独处理")
print("  ✅ 流式编码: 内存高效处理大文件")

print("\n下一步建议:")
print("  1. 阅读 bpe_tokenizer.py 的 Tokenizer 类")
print("  2. 对比每个方法和这里的示例")
print("  3. 使用 from_files() 加载 GPT-2 词汇表测试")
print("  4. 运行 pytest tests/test_tokenizer.py 验证实现")

print("\n")
