"""
=============================================================================
Transformer Language Model -- 从零理解完整架构
=============================================================================

本文件是一个「学习笔记 + 可运行代码」的导读，
按照 **自底向上** 的顺序，把 homework/ 下的各个模块串联成完整的 Transformer。

层级关系 (从底到顶):
─────────────────────────────────────────────────────────
Level 0: 基础算子
    softmax  ──  数值稳定的 softmax
    cross_entropy  ──  交叉熵损失

Level 1: 基础层
    Linear   ──  无偏置线性变换
    Embedding ──  token → 向量查表

Level 2: 归一化 & 位置编码 & 激活
    RMSNorm  ──  均方根归一化 (比 LayerNorm 更快)
    RoPE     ──  旋转位置编码 (让 attention 感知相对距离)
    SwiGLU   ──  门控 FFN (silu 激活 + 门控线性单元)

Level 3: 注意力
    scaled_dot_product_attention  ──  单头注意力核心
    MultiHeadSelfAttention        ──  多头自注意力 + RoPE + 因果遮罩

Level 4: Transformer Block
    TransformerBlock  ──  Pre-Norm 残差块 (Attention + FFN)

Level 5: 完整模型
    TransformerLM  ──  Embedding → N × Block → RMSNorm → LM Head
─────────────────────────────────────────────────────────

下面我们逐层剖析。
"""

import torch
import torch.nn as nn

# ============================================================================
# Level 0: 基础算子
# ============================================================================
"""
── softmax(x, dim) ──────────────────────────────────────

数学:  softmax(x_i) = exp(x_i) / Σ exp(x_j)

数值稳定技巧:  先减去 max(x), 防止 exp 溢出

代码位置: homework/softmax.py

    def softmax(x, dim):
        x_max = x.max(dim=dim, keepdim=True)[0]
        x_exp = torch.exp(x - x_max)          # 减去 max 防溢出
        return x_exp / x_exp.sum(dim=dim, keepdim=True)


── cross_entropy(logits, targets) ───────────────────────

数学:  CE = -log P(target)
         = -logit_target + log(Σ exp(logit_j))

也用了减去 max 的技巧, 避免 exp 上溢。

代码位置: homework/cross_entropy.py
"""


# ============================================================================
# Level 1: 基础层
# ============================================================================
"""
── Embedding(vocab_size, d_model) ───────────────────────

本质: 一个 (vocab_size, d_model) 的查表矩阵。
      给定 token_id, 返回对应行向量。

    weight[token_id]  →  向量 ∈ R^d_model

初始化: trunc_normal_ (截断正态)

代码位置: homework/embedding.py


── Linear(in_features, out_features) ────────────────────

本质: y = x @ W^T  (无偏置)

为什么无偏置?
  现代 LLM (LLaMA, GPT 等) 在大部分线性层中去掉 bias,
  因为在有 RMSNorm 的情况下 bias 几乎不影响表达能力，
  去掉后可以节省参数和计算。

代码位置: homework/linear.py
"""


# ============================================================================
# Level 2: 归一化 & 位置编码 & 激活
# ============================================================================
"""
── RMSNorm(d_model, eps) ────────────────────────────────

数学:  RMSNorm(x) = x / √(mean(x²) + eps) * weight

与 LayerNorm 的区别:
  - LayerNorm:  先减均值 (center), 再除标准差 (scale), 有两个可学习参数 (γ, β)
  - RMSNorm:    跳过 center, 只做 scale, 只有一个可学习参数 (weight)
  - RMSNorm 更快 ~10-15%, 效果几乎一样

实现注意点:
  - 先转 float32 算 variance, 再转回原始 dtype (混合精度训练必需)

代码位置: homework/rmsnorm.py


── RoPE (Rotary Positional Embedding) ───────────────────

核心思想:
  通过对 Q, K 向量的每一对相邻元素做 2D 旋转来注入位置信息。
  旋转角 = position × frequency, 其中 frequency 随维度指数衰减。

数学:
  freq_i = 1 / (theta ^ (2i / d_k))     i = 0, 1, ..., d_k/2 - 1
  angle  = position × freq_i

  对 (x[2i], x[2i+1]) 做 2D 旋转:
    x_new[2i]   = x[2i]   × cos(angle) - x[2i+1] × sin(angle)
    x_new[2i+1] = x[2i]   × sin(angle) + x[2i+1] × cos(angle)

为什么有效?
  Q 在位置 m 旋转了角度 m·freq, K 在位置 n 旋转了角度 n·freq。
  Q·K^T 的结果只取决于 (m-n)·freq, 即 **相对位置**。
  这让模型能推广到训练时没见过的更长序列。

实现优化:
  预先计算好 cos_cache 和 sin_cache (max_seq_len, d_k//2),
  forward 时只需查表和乘法, 无需重新计算。

代码位置: homework/rope.py


── SwiGLU Feed-Forward Network ──────────────────────────

数学:  SwiGLU(x) = (silu(x @ W1^T) ⊙ (x @ W3^T)) @ W2^T

其中:
  silu(z) = z * sigmoid(z)  (Swish 激活函数)
  ⊙ 表示逐元素乘法 (门控机制)

结构:
  输入 x ∈ R^d_model
  ──→ 分两路:
       路1: x @ W1 → silu()  (产生"信息")
       路3: x @ W3            (产生"门控信号")
  ──→ 路1 ⊙ 路3              (门控: 选择性地通过信息)
  ──→ @ W2                    (投影回 d_model)

为什么比标准 FFN 好?
  标准 FFN:  ReLU(x @ W1) @ W2
  SwiGLU:    门控机制让网络可以更精细地控制信息流动,
             实验表明在相同参数量下效果更好。

隐藏层维度: 默认 d_ff = (8/3) × d_model, 向上取整到 64 的倍数。

代码位置: homework/swiglu.py
"""


# ============================================================================
# Level 3: 注意力机制
# ============================================================================
"""
── Scaled Dot-Product Attention ─────────────────────────

数学:  Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V

分步:
  1. scores = Q @ K^T / √d_k     相似度打分
  2. scores += causal_mask         因果遮罩 (未来位置设为 -inf)
  3. weights = softmax(scores)     归一化为概率
  4. output  = weights @ V         加权聚合 value

为什么要除以 √d_k ?
  Q 和 K 的点积结果的方差 ≈ d_k (假设 Q, K 各元素均值 0, 方差 1)。
  除以 √d_k 使方差回到 1, 避免 softmax 进入梯度极小的饱和区。

代码位置: homework/attention.py


── Multi-Head Self-Attention ────────────────────────────

数学:
  head_i = Attention(Q_i, K_i, V_i)
  output = Concat(head_1, ..., head_h) @ W_o

分步:
  1. Q = x @ W_q,  K = x @ W_k,  V = x @ W_v    (线性投影)
  2. 把 Q, K, V 拆成 h 个头:
     reshape (batch, seq, d_model) → (batch, h, seq, d_k)
     其中 d_k = d_model / h
  3. 对 Q, K 应用 RoPE (注入位置信息)
  4. 生成因果遮罩: 下三角矩阵, mask[i,j]=True 当 i≥j
  5. 每个头独立做 scaled dot-product attention
  6. 拼接各头: (batch, h, seq, d_k) → (batch, seq, d_model)
  7. output = concat @ W_o    (输出投影)

为什么多头?
  每个头学习不同的 "关注模式":
  - 某些头关注语法关系
  - 某些头关注语义相似性
  - 某些头关注局部上下文
  多头让模型同时捕获多种类型的依赖关系。

代码位置: homework/multihead_attention.py
"""
# ============================================================================
# Level 4: Transformer Block
# ============================================================================
"""
── Pre-Norm Transformer Block ──────────────────────────

架构 (Pre-Norm, LLaMA 风格):

    ┌──────────────────────────────────────┐
    │  输入 x                              │
    │  ├── 残差分支: x 直接向下传递         │
    │  └── 主路径:                         │
    │       x → RMSNorm → MultiHeadAttn    │
    │  x = x + Attn(RMSNorm(x))     ← 残差连接 1
    │                                      │
    │  ├── 残差分支: x 直接向下传递         │
    │  └── 主路径:                         │
    │       x → RMSNorm → SwiGLU FFN      │
    │  x = x + FFN(RMSNorm(x))      ← 残差连接 2
    │                                      │
    │  输出 x                              │
    └──────────────────────────────────────┘

Pre-Norm vs Post-Norm:
  Post-Norm (原始论文):  x = LayerNorm(x + Sublayer(x))
  Pre-Norm  (LLaMA):     x = x + Sublayer(Norm(x))

  Pre-Norm 的优点:
  - 训练更稳定 (梯度流更顺畅)
  - 不需要精心的学习率 warmup
  - 更容易收敛

残差连接的作用:
  - 缓解梯度消失问题
  - 允许堆叠更多层 (原始 Transformer 6 层, 现代 LLM 100+ 层)
  - 让每层只需学习 "残差" (增量修正), 比学习完整映射更容易

代码位置: homework/transformer_block.py
"""


# ============================================================================
# Level 5: 完整的 Transformer Language Model
# ============================================================================
"""
── TransformerLM ────────────────────────────────────────

完整前向传播流程:

  token_ids (batch, seq)                          整数 token ID
       │
       ▼
  ┌─────────────────┐
  │ Token Embedding  │  查表: id → 向量
  └─────────────────┘
       │  (batch, seq, d_model)
       ▼
  ┌─────────────────┐
  │ Transformer      │  × N 层
  │ Block 1..N       │  每层: Attn + FFN (都有残差 + RMSNorm)
  └─────────────────┘
       │  (batch, seq, d_model)
       ▼
  ┌─────────────────┐
  │ Final RMSNorm    │  最终归一化
  └─────────────────┘
       │  (batch, seq, d_model)
       ▼
  ┌─────────────────┐
  │ LM Head (Linear) │  投影到词表: d_model → vocab_size
  └─────────────────┘
       │  (batch, seq, vocab_size)
       ▼
     logits                                       每个位置对每个 token 的打分

训练时:
  loss = cross_entropy(logits.view(-1, V), targets.view(-1))
  → 反向传播 → AdamW 优化

推理时:
  next_token = argmax(logits[:, -1, :])   或者用 temperature sampling

代码位置: homework/transformer_lm.py
"""


# ============================================================================
# 可运行的实验代码: 把所有组件组装起来
# ============================================================================
from homework.transformer_lm import TransformerLM
from homework.cross_entropy import cross_entropy


def demo_forward_pass():
    """演示完整的前向传播过程。"""
    # ── 超参数 (小模型, 方便调试) ──
    vocab_size = 256        # 词表大小 (这里用字节级别)
    context_length = 128    # 最大序列长度
    d_model = 64            # 隐藏维度
    num_layers = 2          # Transformer 层数
    num_heads = 4           # 注意力头数 (d_k = d_model / num_heads = 16)
    d_ff = 172              # FFN 隐藏维度 (≈ 8/3 * 64, 取 64 的倍数: 192 → 172)
    rope_theta = 10000.0    # RoPE 基频 (标准值)

    # ── 构建模型 ──
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    print(f"模型结构:\n{model}\n")

    # ── 构造输入 ──
    batch_size = 2
    seq_len = 16
    # 随机 token ids (模拟输入)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    # targets: 向右移一位 (语言模型的标准做法)
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    print(f"输入 shape:  {input_ids.shape}")
    print(f"目标 shape:  {target_ids.shape}")

    # ── 前向传播 ──
    logits = model(input_ids)
    print(f"输出 logits shape: {logits.shape}")
    print(f"  → 期望: (batch={batch_size}, seq={seq_len}, vocab={vocab_size})")

    # ── 计算损失 ──
    # cross_entropy 期望 (batch_size, vocab_size) 和 (batch_size,)
    # 所以需要展平 seq 维度
    logits_flat = logits.view(-1, vocab_size)    # (batch*seq, vocab)
    targets_flat = target_ids.view(-1)            # (batch*seq,)
    loss = cross_entropy(logits_flat, targets_flat)
    print(f"\n交叉熵损失: {loss.item():.4f}")
    print(f"  → 随机猜测的期望损失: {torch.log(torch.tensor(float(vocab_size))).item():.4f}")
    print(f"  (两者应该接近, 因为模型还没训练)")

    # ── 反向传播 ──
    loss.backward()
    grad_norms = {
        name: p.grad.norm().item()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    print(f"\n梯度范数 (部分参数):")
    for name, norm in list(grad_norms.items())[:5]:
        print(f"  {name}: {norm:.6f}")


def demo_layer_by_layer():
    """逐层演示数据流动, 帮助理解每一步的 shape 变化。"""
    from homework.embedding import Embedding
    from homework.rmsnorm import RMSNorm
    from homework.rope import RotaryPositionalEmbedding
    from homework.swiglu import SwiGLU
    from homework.linear import Linear
    from homework.softmax import softmax

    print("=" * 60)
    print("逐层 Shape 追踪")
    print("=" * 60)

    d_model = 64
    num_heads = 4
    d_k = d_model // num_heads  # = 16
    vocab_size = 256
    batch, seq = 2, 8

    # Step 1: Embedding
    emb = Embedding(vocab_size, d_model)
    token_ids = torch.randint(0, vocab_size, (batch, seq))
    x = emb(token_ids)
    print(f"\n1. Embedding:     {token_ids.shape} → {x.shape}")
    print(f"   token_ids (int) → 查表得到 dense vectors")

    # Step 2: RMSNorm
    norm = RMSNorm(d_model)
    x_norm = norm(x)
    print(f"\n2. RMSNorm:       {x.shape} → {x_norm.shape}")
    print(f"   x / sqrt(mean(x^2) + eps) * weight")

    # Step 3: Q, K, V 投影
    q_proj = nn.Linear(d_model, d_model, bias=False)
    Q = q_proj(x_norm)
    print(f"\n3. Q/K/V 投影:    {x_norm.shape} → {Q.shape}")
    print(f"   x @ W_q → Q (同理 K, V)")

    # Step 4: 拆分为多头
    from einops import rearrange
    Q_heads = rearrange(Q, "b s (h d) -> b h s d", h=num_heads)
    print(f"\n4. 拆分多头:      {Q.shape} → {Q_heads.shape}")
    print(f"   (batch, seq, d_model) → (batch, heads, seq, d_k)")

    # Step 5: RoPE
    rope = RotaryPositionalEmbedding(theta=10000.0, d_k=d_k, max_seq_len=128)
    positions = torch.arange(seq).unsqueeze(0).expand(batch, -1)
    # positions 需要加 heads 维度以便广播: (batch, seq) → (batch, 1, seq)
    rope_positions = positions.unsqueeze(1)
    Q_rotated = rope(Q_heads, rope_positions)
    print(f"\n5. RoPE:          {Q_heads.shape} → {Q_rotated.shape}")
    print(f"   对每对 (x[2i], x[2i+1]) 做 2D 旋转, 注入位置信息")

    # Step 6: Attention scores
    import math
    K_heads = rearrange(q_proj(x_norm), "b s (h d) -> b h s d", h=num_heads)
    K_rotated = rope(K_heads, rope_positions)
    scores = torch.einsum("bhqd, bhkd -> bhqk", Q_rotated, K_rotated) / math.sqrt(d_k)
    print(f"\n6. Attention 分数: Q@K^T/√d_k → {scores.shape}")
    print(f"   (batch, heads, seq_q, seq_k)")

    # Step 7: Causal mask + Softmax
    causal_mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
    scores_masked = scores.masked_fill(~causal_mask, float("-inf"))
    attn_weights = softmax(scores_masked, dim=-1)
    print(f"\n7. Mask + Softmax: {scores.shape} → {attn_weights.shape}")
    print(f"   上三角设为 -inf, softmax 归一化")
    print(f"   第一行 (位置0): 只能看自己")
    print(f"   最后一行 (位置{seq-1}): 能看所有位置")

    # Step 8: Weighted sum
    V_heads = rearrange(q_proj(x_norm), "b s (h d) -> b h s d", h=num_heads)
    attn_out = torch.einsum("bhqk, bhkd -> bhqd", attn_weights, V_heads)
    print(f"\n8. 加权聚合:      weights@V → {attn_out.shape}")

    # Step 9: 拼接多头 + 输出投影
    concat = rearrange(attn_out, "b h s d -> b s (h d)")
    out_proj = nn.Linear(d_model, d_model, bias=False)
    attn_output = out_proj(concat)
    print(f"\n9. 拼接+投影:     {attn_out.shape} → {concat.shape} → {attn_output.shape}")

    # Step 10: 残差连接
    x_after_attn = x + attn_output
    print(f"\n10. 残差连接:     x + Attn(Norm(x)) → {x_after_attn.shape}")

    # Step 11: FFN (SwiGLU)
    ffn = SwiGLU(d_model=d_model)
    ffn_out = ffn(norm(x_after_attn))
    x_after_ffn = x_after_attn + ffn_out
    print(f"\n11. SwiGLU FFN:   {x_after_attn.shape} → {ffn_out.shape}")
    print(f"    silu(x@W1) ⊙ (x@W3) → @W2")
    print(f"    + 残差连接 → {x_after_ffn.shape}")

    # Step 12: Final Norm + LM Head
    final_norm = RMSNorm(d_model)
    lm_head = Linear(d_model, vocab_size)
    logits = lm_head(final_norm(x_after_ffn))
    print(f"\n12. LM Head:      {x_after_ffn.shape} → {logits.shape}")
    print(f"    RMSNorm → Linear(d_model → vocab_size)")
    print(f"    每个位置输出 {vocab_size} 维 logits, 表示下一个 token 的概率分布")


def demo_attention_pattern():
    """可视化因果注意力遮罩, 帮助理解自回归生成。"""
    print("\n" + "=" * 60)
    print("因果注意力遮罩 (Causal Mask)")
    print("=" * 60)

    seq_len = 6
    mask = torch.tril(torch.ones(seq_len, seq_len))

    print(f"\n遮罩矩阵 ({seq_len}x{seq_len}):")
    print("行=query位置, 列=key位置, 1=可见, 0=遮蔽\n")

    # 打印列标签
    print("      ", end="")
    for j in range(seq_len):
        print(f"pos{j} ", end="")
    print()

    for i in range(seq_len):
        print(f"pos{i}: ", end="")
        for j in range(seq_len):
            if mask[i, j] == 1:
                print("  ●   ", end="")   # 可以看到
            else:
                print("  ○   ", end="")   # 被遮蔽
        print(f"  ← 位置 {i} 可以看到 {int(mask[i].sum())} 个位置")

    print("""
解读:
  - 位置 0 只能看自己 (生成第一个 token 时没有上下文)
  - 位置 5 可以看到 0~5 所有位置 (拥有完整上下文)
  - 这就是 "自回归" 的含义: 每个位置只依赖之前的 token
  """)


# ============================================================================
# 训练循环概览
# ============================================================================
"""
── 训练一个 Transformer LM 的完整流程 ──────────────────

1. 数据准备:
   - 文本 → BPE 分词 → token_ids 序列
   - 切成固定长度窗口 (context_length)
   - input  = tokens[:-1]   (输入)
   - target = tokens[1:]    (目标, 右移一位)

   代码: homework/data_loading.py → get_batch()

2. 前向传播:
   logits = model(input_ids)
   loss = cross_entropy(logits.view(-1, V), targets.view(-1))

3. 反向传播:
   loss.backward()

4. 梯度裁剪 (防止梯度爆炸):
   gradient_clipping(model.parameters(), max_norm=1.0)
   代码: homework/gradient_clipping.py

5. 优化器更新 (AdamW):
   optimizer.step()
   代码: homework/adamw.py

6. 学习率调度 (Cosine with Warmup):
   lr = cosine_schedule(step, warmup_steps, total_steps, max_lr, min_lr)
   代码: homework/lr_schedule.py

7. 检查点保存:
   save_checkpoint(model, optimizer, step, path)
   代码: homework/checkpoint.py

重复 2-7 直到收敛。
"""


# ============================================================================
# 主入口
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Transformer Language Model 学习指南")
    print("=" * 60)

    # Demo 1: 完整前向传播
    print("\n>>> Demo 1: 完整前向传播")
    print("-" * 40)
    demo_forward_pass()

    # Demo 2: 逐层 shape 追踪
    print("\n\n>>> Demo 2: 逐层 Shape 追踪")
    print("-" * 40)
    demo_layer_by_layer()

    # Demo 3: 因果遮罩可视化
    print("\n\n>>> Demo 3: 因果注意力遮罩")
    print("-" * 40)
    demo_attention_pattern()

    print("\n" + "=" * 60)
    print("学习路径建议:")
    print("=" * 60) 
    print("""
    1. 先运行本文件, 观察每一层的 shape 变化
    2. 阅读 homework/softmax.py     → 理解数值稳定性
    3. 阅读 homework/attention.py   → 理解注意力核心
    4. 阅读 homework/rope.py        → 理解位置编码
    5. 阅读 homework/multihead_attention.py → 理解多头机制
    6. 阅读 homework/transformer_block.py   → 理解残差 + Pre-Norm
    7. 阅读 homework/transformer_lm.py      → 理解完整模型
    8. 尝试修改超参数, 观察参数量变化
    9. 用小数据集实际训练一个小模型
    """)
