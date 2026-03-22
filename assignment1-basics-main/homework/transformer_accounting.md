# Transformer LM Resource Accounting

## 问题描述

> **Problem (transformer_accounting): Transformer LM resource accounting (5 points)**

---

## Part (a): 参数量和内存

### 问题

> Consider GPT-2 XL, which has the following configuration:
> - vocab_size: 50,257
> - context_length: 1,024
> - num_layers: 48
> - d_model: 1,600
> - num_heads: 25
> - d_ff: 6,400
>
> Suppose we constructed our model using this configuration. How many trainable parameters would our model have? Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model?
>
> **Deliverable**: A one-to-two sentence response.

### 解答

### GPT-2 XL 配置
| 参数 | 值 |
|------|-----|
| vocab_size | 50,257 |
| context_length | 1,024 |
| num_layers | 48 |
| d_model | 1,600 |
| num_heads | 25 |
| d_ff | 6,400 |

### 参数量计算

**1. Token Embeddings**
$$\text{params} = \text{vocab\_size} \times \text{d\_model} = 50,257 \times 1,600 = 80,411,200$$

**2. 每个 Transformer Block**
- Q, K, V, O 投影：$4 \times d_{model} \times d_{model} = 4 \times 1,600 \times 1,600 = 10,240,000$
- FFN (SwiGLU)：$3 \times d_{model} \times d_{ff} = 3 \times 1,600 \times 6,400 = 30,720,000$
- RMSNorm (ln1, ln2)：$2 \times d_{model} = 2 \times 1,600 = 3,200$
- **每层总计**：$10,240,000 + 30,720,000 + 3,200 = 40,963,200$

**3. 所有层**
$$48 \times 40,963,200 = 1,966,233,600$$

**4. 最终层归一化 (ln_final)**
$$d_{model} = 1,600$$

**5. LM Head**
$$\text{vocab\_size} \times d_{model} = 50,257 \times 1,600 = 80,411,200$$

### 总参数量
$$80,411,200 + 1,966,233,600 + 1,600 + 80,411,200 = \boxed{2,127,057,600} \approx 2.13 \text{ B}$$

### 内存需求 (float32 = 4 bytes)
$$2,127,057,600 \times 4 \text{ bytes} = 8,508,230,400 \text{ bytes} \approx \boxed{8.51 \text{ GB}}$$

### 最终答案

GPT-2 XL 模型共有 **2.13B 可训练参数**，使用单精度浮点数（float32）加载需要约 **8.51 GB** 内存。

---

## Part (b): FLOPs 分析

### 问题

> Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped model. How many FLOPs do these matrix multiplies require in total? Assume that our input sequence has context_length tokens.
>
> **Deliverable**: A list of matrix multiplies (with descriptions), and the total number of FLOPs required.

### 解答

假设 batch_size = 1，sequence_length = context_length = 1024

#### 每个 Transformer Block 的矩阵乘法

| 操作 | 描述 | 形状 | FLOPs 公式 | FLOPs |
|------|------|------|------------|-------|
| Q 投影 | input @ W_q | (seq, d_model) @ (d_model, d_model) | $2 \times seq \times d_{model}^2$ | $2 \times 1024 \times 1600^2 = 5,242,880,000$ |
| K 投影 | input @ W_k | (seq, d_model) @ (d_model, d_model) | $2 \times seq \times d_{model}^2$ | $5,242,880,000$ |
| V 投影 | input @ W_v | (seq, d_model) @ (d_model, d_model) | $2 \times seq \times d_{model}^2$ | $5,242,880,000$ |
| Q @ K^T | 注意力分数 | (heads, seq, d_k) @ (heads, d_k, seq) | $2 \times heads \times seq^2 \times d_k$ | $2 \times 25 \times 1024^2 \times 64 = 3,355,443,200$ |
| scores @ V | 注意力输出 | (heads, seq, seq) @ (heads, seq, d_k) | $2 \times heads \times seq^2 \times d_k$ | $3,355,443,200$ |
| O 投影 | attn_out @ W_o | (seq, d_model) @ (d_model, d_model) | $2 \times seq \times d_{model}^2$ | $5,242,880,000$ |
| FFN w1 | up-projection | (seq, d_model) @ (d_model, d_ff) | $2 \times seq \times d_{model} \times d_{ff}$ | $2 \times 1024 \times 1600 \times 6400 = 20,971,520,000$ |
| FFN w3 | gate-projection | (seq, d_model) @ (d_model, d_ff) | $2 \times seq \times d_{model} \times d_{ff}$ | $20,971,520,000$ |
| FFN w2 | down-projection | (seq, d_ff) @ (d_ff, d_model) | $2 \times seq \times d_{model} \times d_{ff}$ | $20,971,520,000$ |
| **每层总计** | | | | **90,643,206,400** |

#### 最终层

| 操作 | 形状 | FLOPs |
|------|------|-------|
| LM Head | (seq, d_model) @ (d_model, vocab_size) | $2 \times 1024 \times 1600 \times 50257 = 164,477,337,600$ |

### 总 FLOPs

$$\text{Total} = 48 \times 90,643,206,400 + 164,477,337,600 = \boxed{4,515,351,244,800} \approx 4.52 \text{ TFLOPs}$$

---

## Part (c): 计算瓶颈分析

### 问题

> Based on your analysis above, which parts of the model require the most FLOPs?
>
> **Deliverable**: A one-to-two sentence response.

### 解答

### 各组件 FLOPs 占比

| 组件 | 每层 FLOPs | 48层总计 | 占比 |
|------|-----------|----------|------|
| Attention 投影 (Q, K, V, O) | $8 \times seq \times d_{model}^2 = 20.97B$ | 1,006.6B | 22.3% |
| Attention 计算 (QK^T, scores@V) | $4 \times seq^2 \times d_{model} = 6.71B$ | 322.1B | 7.1% |
| FFN (w1, w2, w3) | $6 \times seq \times d_{model} \times d_{ff} = 62.91B$ | 3,020.0B | 66.9% |
| LM Head | - | 164.5B | 3.6% |

### 最终答案

**FFN (SwiGLU) 消耗最多的 FLOPs（约 67%）**，因为 $d_{ff} = 4 \times d_{model}$，使得 FFN 的矩阵乘法规模最大。Attention 投影次之（约 22%），而实际的 attention 计算（QK^T 和 scores@V）仅占约 7%。

---

## Part (d): 不同模型规模对比

### 问题

> Repeat your analysis with GPT-2 small (12 layers, 768 d_model, 12 heads), GPT-2 medium (24 layers, 1024 d_model, 16 heads), and GPT-2 large (36 layers, 1280 d_model, 20 heads). As the model size increases, which parts of the Transformer LM take up proportionally more or less of the total FLOPs?
>
> **Deliverable**: For each model, provide a breakdown of model components and its associated FLOPs (as a proportion of the total FLOPs required for a forward pass). In addition, provide a one-to-two sentence description of how varying the model size changes the proportional FLOPs of each component.

### 解答

### 模型配置

| 模型 | layers | d_model | heads | d_ff |
|------|--------|---------|-------|------|
| GPT-2 Small | 12 | 768 | 12 | 3072 |
| GPT-2 Medium | 24 | 1024 | 16 | 4096 |
| GPT-2 Large | 36 | 1280 | 20 | 5120 |
| GPT-2 XL | 48 | 1600 | 25 | 6400 |

### FLOPs 公式总结

设 $n$ = num_layers, $d$ = d_model, $f$ = d_ff, $s$ = seq_len, $v$ = vocab_size

- **Attention 投影**: $8nsd^2$
- **Attention 计算**: $4ns^2d$
- **FFN**: $6nsdf$
- **LM Head**: $2sdv$

### 各模型 FLOPs 占比 (seq = 1024)

| 组件 | GPT-2 Small | GPT-2 Medium | GPT-2 Large | GPT-2 XL |
|------|-------------|--------------|-------------|----------|
| Attn 投影 | 21.4% | 21.9% | 22.1% | 22.3% |
| Attn 计算 | 7.1% | 7.0% | 6.9% | 7.1% |
| FFN | 67.9% | 67.6% | 67.8% | 66.9% |
| LM Head | 3.6% | 3.5% | 3.2% | 3.6% |
| **Total FLOPs** | **0.34T** | **1.03T** | **2.24T** | **4.51T** |

### 详细计算

**GPT-2 Small (n=12, d=768, f=3072)**:
- Attn 投影: $8 \times 12 \times 1024 \times 768^2 = 45.5B$ (21.4%)
- Attn 计算: $4 \times 12 \times 1024^2 \times 768 = 38.7B$ (18.2%)
- FFN: $6 \times 12 \times 1024 \times 768 \times 3072 = 173.2B$ (81.4%)
- LM Head: $2 \times 1024 \times 768 \times 50257 = 79.1B$
- **Total**: 336.5B ≈ 0.34T

**GPT-2 Medium (n=24, d=1024, f=4096)**:
- Attn 投影: $8 \times 24 \times 1024 \times 1024^2 = 201.3B$
- Attn 计算: $4 \times 24 \times 1024^2 \times 1024 = 102.4B$
- FFN: $6 \times 24 \times 1024 \times 1024 \times 4096 = 619.3B$
- LM Head: $2 \times 1024 \times 1024 \times 50257 = 105.4B$
- **Total**: 1,028.4B ≈ 1.03T

**GPT-2 Large (n=36, d=1280, f=5120)**:
- Attn 投影: $8 \times 36 \times 1024 \times 1280^2 = 480.0B$
- Attn 计算: $4 \times 36 \times 1024^2 \times 1280 = 192.0B$
- FFN: $6 \times 36 \times 1024 \times 1280 \times 5120 = 1,440.0B$
- LM Head: $2 \times 1024 \times 1280 \times 50257 = 131.7B$
- **Total**: 2,243.7B ≈ 2.24T

**GPT-2 XL (n=48, d=1600, f=6400)**:
- Attn 投影: $8 \times 48 \times 1024 \times 1600^2 = 1,006.6B$
- Attn 计算: $4 \times 48 \times 1024^2 \times 1600 = 322.1B$
- FFN: $6 \times 48 \times 1024 \times 1600 \times 6400 = 3,020.0B$
- LM Head: $2 \times 1024 \times 1600 \times 50257 = 164.5B$
- **Total**: 4,513.2B ≈ 4.51T

### 最终答案

随着模型规模增大，各组件的 FLOPs 占比**基本保持稳定**：
- **FFN** 始终占约 67-68%，是主要计算开销
- **Attention 投影** 占约 21-22%
- **Attention 计算** 占约 7%
- **LM Head** 占约 3-4%

这是因为 GPT-2 系列保持 $d_{ff} = 4 \times d_{model}$ 的比例，使得各组件按相同比例缩放，因此各部分的相对占比几乎不变。

---

## Part (e): 长上下文的影响

### 问题

> Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one forward pass change? How do the relative contribution of FLOPs of the model components change?
>
> **Deliverable**: A one-to-two sentence response.

### 解答

### GPT-2 XL with context_length = 16,384

将序列长度从 1,024 增加到 16,384（16倍）

| 组件 | 原始 (s=1024) | 新 (s=16384) | 倍数 |
|------|---------------|--------------|------|
| Attn 投影 | 1,006.6B | 16,105.2B | 16× |
| Attn 计算 | 322.1B | 82,176.0B | **256×** |
| FFN | 3,020.0B | 48,320.0B | 16× |
| LM Head | 164.5B | 2,631.6B | 16× |
| **Total** | **4,513.2B** | **149,232.8B** | **33×** |

### 分析

1. **总 FLOPs 变化**: 从 4.5 TFLOPs 增加到 149.2 TFLOPs，增加约 **33 倍**

2. **相对占比变化**:
   - **Attention 计算** 从 7.1% 激增到 **55.0%**（因为与 $s^2$ 成正比）
   - **FFN** 从 66.9% 下降到 **32.4%**
   - **Attention 投影** 从 22.3% 下降到 **10.8%**
   - **LM Head** 从 3.6% 下降到 **1.8%**

### 最终答案

**总 FLOPs 增加约 33 倍**（从 4.5T 到 149.2T）。最显著的变化是 **attention 计算的占比从 7% 激增到 55%**，因为它与序列长度呈 $O(s^2)$ 关系，而其他组件仅呈线性增长。这使得 attention 计算成为长上下文模型的主要瓶颈，也解释了为什么 Flash Attention 等优化技术对长上下文模型如此重要。

---

## 解题思路总结

这类问题需要我们：
1. **理解模型架构**：明确每个组件的参数和计算
2. **参数量计算**：统计所有可训练参数的数量
3. **FLOPs 计算**：矩阵乘法 `C = A @ B`，若 A 为 (M, K)，B 为 (K, N)，则 FLOPs = 2MKN
4. **分析占比**：比较各组件的计算量占比

---

## 公式速查表

| 组件 | 参数量 | FLOPs (per token) |
|------|--------|-------------------|
| Token Embed | $v \times d$ | 0 (lookup) |
| Attn 投影 | $4nd^2$ | $8nsd^2$ |
| Attn 计算 | 0 | $4ns^2d$ |
| FFN (SwiGLU) | $3ndf$ | $6nsdf$ |
| Final LN | $d$ | $2d$ |
| LM Head | $vd$ | $2sdv$ |

其中：$n$ = layers, $d$ = d_model, $f$ = d_ff, $s$ = seq_len, $v$ = vocab_size
