# 面试 Q&A 合集

---

## 一面

### 1. PPO vs GRPO 区别，为什么用 GRPO

| | PPO | GRPO |
|---|---|---|
| **Advantage 来源** | Critic 网络 V(s)：A = R − V(s)（或 GAE） | 组内统计：Â = (R − μ) / σ |
| **需要的模型** | Actor + Critic + Reference + Reward（4 个） | Actor + Reference（2 个） |
| **显存占用** | Critic 与 Actor 同等大小，显存翻倍 | 无 Critic，省一半显存 |
| **核心目标** | min(r·A, clip(r)·A) | 完全相同的 clip 目标 |
| **适用场景** | 通用（对话、任意 reward） | 有可验证 reward 的任务（数学、代码） |

**为什么用 GRPO**：数学任务的 reward 是确定性的（答对=1，答错=0），同一道题采 G=8 条回答就能得到可靠的组内 baseline，不需要训一个 Critic 来估计 value。省掉 Critic = 省显存 + 省工程复杂度 + 避免 Critic 训崩的风险。

### 2. DPO 损失函数

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \left( \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right) \right]$$

直觉：让 π_θ 对 preferred response y_w 的概率比 y_l **相对于 reference** 更高。β 控制偏离 reference 的程度。本质上把 reward model + RL 两步合成一步，直接从 preference pair 优化策略。

### 3. FSDP vs DDP

| | DDP | FSDP |
|---|---|---|
| **分片内容** | 不分片，每张卡持有完整模型副本 | 对 **参数 + 梯度 + 优化器状态** 三者做分片 |
| **通信** | AllReduce 梯度 | 前向/反向时 AllGather 参数，反向后 ReduceScatter 梯度 |
| **显存** | 每卡 = 完整模型 + 梯度 + 优化器 | 每卡 ≈ 模型/N + 梯度/N + 优化器/N |
| **适用** | 模型能放进单卡 | 模型放不进单卡 |

FSDP 分片策略：
- `FULL_SHARD`：参数 + 梯度 + 优化器全分片（最省显存）
- `SHARD_GRAD_OP`：只分片梯度和优化器（前向不需要 AllGather，更快）
- `NO_SHARD`：退化为 DDP

### 4. Megatron 并行策略

| 并行方式 | 切什么 | 通信 |
|---|---|---|
| **数据并行 (DP)** | 数据 batch | AllReduce 梯度 |
| **张量并行 (TP)** | 单层权重矩阵按列/行切分到多卡 | 每层 AllReduce 激活值 |
| **流水线并行 (PP)** | 不同层放不同卡 | 层间传递激活值（micro-batch 流水线） |
| **序列并行 (SP)** | 沿序列维度切分 LayerNorm/Dropout | 与 TP 配合减少激活显存 |

实践中 3D 并行：TP 在节点内（NVLink 高带宽），PP 跨节点，DP 最外层。

### 5. KV Cache 原理 + vLLM 两个优化

**KV Cache**：自回归生成时，每个新 token 的 attention 需要所有历史 token 的 K、V。如果不缓存，每步都要对整个序列重新计算 KV，复杂度 O(n²)。缓存后每步只算新 token 的 Q，与缓存的 KV 做 attention，变为 O(n)。

**vLLM 优化**：
1. **PagedAttention**：KV cache 按"页"（block）分配，不再为每个序列预分配最大长度的连续显存。解决显存碎片化问题，利用率从 ~20% 提升到 >90%
2. **Continuous Batching**：不等整个 batch 生成完毕才处理下一批，而是序列完成后立即插入新请求，GPU 利用率大幅提升

### 6. AWQ vs GPTQ

| | GPTQ | AWQ |
|---|---|---|
| **思路** | 逐层量化，用 Hessian 信息做最优权重舍入 | 观察激活分布，保护"重要"权重通道（per-channel scaling） |
| **校准** | 需要校准数据集 + 较长时间（逐层优化） | 只需少量校准数据，快得多 |
| **核心洞察** | 最小化量化误差的二阶近似 | 1% 的 salient weights 决定模型质量，保护它们即可 |
| **推理速度** | 快（INT4 kernel） | 更快（对硬件更友好的量化格式） |

### 7. LoRA 调参 / 位置编码 / OOM / Activation Checkpointing

**LoRA 关键参数**：
- `r`（秩）：通常 8-64，越大表达力越强但参数越多
- `alpha`：缩放因子，实际缩放 = alpha/r，通常 alpha = 2r
- `target_modules`：一般选 q_proj, v_proj（最低配）；加 k_proj, o_proj, gate_proj, up_proj, down_proj 效果更好

**位置编码**：
- **绝对位置编码**（原始 Transformer）：学习 PE ∈ R^{L×d}，加到 embedding 上
- **RoPE**（Qwen/LLaMA 系列）：在 Q、K 上施加旋转矩阵，使 attention score 只依赖相对位置差。支持长度外推（NTK-aware scaling / YaRN）
- **ALiBi**：不加位置编码，直接在 attention score 上减去与距离成正比的偏置

**OOM 怎么办**（按优先级）：
1. 减小 batch size / 增大 gradient accumulation
2. 开启混合精度（bf16/fp16）
3. 开启 gradient checkpointing
4. 使用 LoRA / QLoRA 减少可训练参数
5. DeepSpeed ZeRO / FSDP 分片
6. 减小序列长度

**Activation Checkpointing 原理**：
- 正常反向传播需要保留所有层的前向激活值 → 显存 O(L)
- Checkpointing：只保留部分层（如每 √L 层）的激活，其余在反向传播时重新计算
- 时间换空间：显存降为 O(√L)，计算量增加约 33%（每个 segment 前向算两次）

### 8. Qwen3-VL 改进 + 多模态架构

**典型多模态 LLM 架构**：
```
图像 → Vision Encoder (ViT) → Projector/Adapter → LLM Backbone → 文本输出
```

**Qwen2-VL 改进**：
- Naive Dynamic Resolution：不固定图像分辨率，按原始比例切分为 patches
- 3D RoPE：对视觉 token 施加 (时间, 高度, 宽度) 三维位置编码，支持视频理解
- 更大的 ViT（675M 参数），与 LLM 联合训练

---

## 补充面

### Q1. 为什么 Qwen 训练中要加后训练（Post-Training），作用是什么？

预训练（Pretraining）只做 next-token prediction，学到的是通用语言建模能力，但模型不会"听话"——不能对齐人类意图、不会拒绝有害请求、也不会按格式回答。

**后训练阶段**一般包含两步：
1. **SFT（Supervised Fine-Tuning）**：用高质量 instruction-response 数据微调，让模型学会遵循指令、结构化输出
2. **偏好对齐（RLHF / DPO / GRPO）**：进一步根据人类偏好优化，让回答更有帮助、无害、诚实

作用总结：把"知识丰富但不可控"的 base model 变成"可控、安全、好用"的 chat model。

### Q2. 偏好对齐这一步的作用是什么？

SFT 后模型已经会遵循指令，但仍有问题：
- 生成质量参差不齐（好坏回答都在 SFT 分布中）
- 可能生成有害/不安全内容
- 风格不一致

偏好对齐通过 preference pair (y_w, y_l) 让模型**学会区分好坏回答**，在生成分布上做精细调整：
- 提升 preferred response 的概率
- 降低 rejected response 的概率
- 通过 KL 约束防止偏离太远

本质：SFT 教模型"怎么回答"，偏好对齐教模型"哪种回答更好"。

### Q3. SD（Stable Diffusion）和 FLUX 的区别

| | Stable Diffusion (1.x/2.x/XL) | FLUX |
|---|---|---|
| **去噪骨架** | UNet（下采样-中间-上采样 + skip connection） | DiT（Transformer block 堆叠） |
| **文本编码** | CLIP text encoder（SD1 单 CLIP，SDXL 双 CLIP） | T5-XXL + CLIP 双编码器 |
| **架构** | CNN-based UNet + cross-attention | 双流 Transformer → 单流 Transformer |
| **扩展性** | UNet 扩展困难（归纳偏置强） | Transformer scaling law，容易扩参数 |
| **生成质量** | 好，但文字渲染弱 | 文字渲染、构图、细节显著提升 |
| **推理** | 较快 | 较慢（Transformer 计算量大） |

### Q4. 使用 DiT 相比 UNet 的优势

1. **Scaling 友好**：Transformer 遵循 scaling law，增大参数量 → 性能稳定提升；UNet 扩展效果有天花板
2. **无归纳偏置**：UNet 的卷积有局部性偏置，DiT 全局注意力从第一层就能建模长程依赖
3. **统一架构**：与 LLM 共用 Transformer 架构，方便多模态融合（共享权重、联合训练）
4. **灵活条件注入**：通过 adaLN-Zero、cross-attention 等方式灵活注入条件，比 UNet 的 cross-attention 层更统一
5. **工程生态**：大量 Transformer 优化（FlashAttention、tensor parallel）可直接复用

### Q5. CFG（Classifier-Free Guidance）原理

核心思想：在训练时随机 drop 条件（以一定概率把 condition 替换为空），让模型同时学会 **有条件生成 ε(x_t, c)** 和 **无条件生成 ε(x_t, ∅)**。

推理时合成引导信号：

$$\tilde{\epsilon} = \epsilon(x_t, \varnothing) + w \cdot \big(\epsilon(x_t, c) - \epsilon(x_t, \varnothing)\big)$$

- w = 1：标准条件生成
- w > 1：放大条件的影响，生成更符合 prompt 但多样性降低
- w = 0：无条件生成

直觉：用"有条件 − 无条件"的差值作为"条件方向"，w 控制沿这个方向走多远。相比 Classifier Guidance 不需要额外训练分类器。

### Q6. LoRA 微调的原理

对预训练权重矩阵 W ∈ R^{d×k}，LoRA **冻结 W**，只训练一个低秩分解的增量：

$$W' = W + \Delta W = W + B \cdot A$$

- A ∈ R^{r×k}（高斯初始化），B ∈ R^{d×r}（零初始化）
- r ≪ min(d, k)，通常 r = 8~64
- 实际缩放：ΔW = (α/r) · B·A，α 控制增量幅度

优势：
- 可训练参数从 d×k 降到 (d+k)×r，减少 99%+
- 推理时可合并回原权重：W' = W + BA，**无额外延迟**
- 多任务可共享 base model，只切换不同的 BA

### Q7. DDPM 和 DDIM 的区别

| | DDPM | DDIM |
|---|---|---|
| **前向过程** | 马尔可夫链，每步加高斯噪声 | 相同 |
| **反向过程** | **随机**：每步采样加噪声 | **确定性**：给定 x_T，反向路径唯一 |
| **采样步数** | 需要 T 步（通常 1000） | 可跳步，50 甚至 10 步即可 |
| **生成多样性** | 每次采样不同结果 | 同一 x_T → 同一结果（deterministic） |
| **训练** | 完全相同的训练目标 | **不需要重新训练**，直接用 DDPM 的模型 |

DDIM 关键公式在反向步中去掉了随机噪声项（σ=0），使得采样可以用更大步长跳步。

### Q8. PPO 和 DPO 的区别

| | PPO | DPO |
|---|---|---|
| **阶段** | 两阶段：先训 Reward Model，再 RL | 一阶段：直接从 preference pair 优化 |
| **需要的模型** | Actor + Critic + Reward + Reference | Policy + Reference |
| **优化方式** | 在线 RL：采样 → 打分 → 更新 | 离线：直接在 preference 数据上做 supervised loss |
| **训练复杂度** | 高（多模型协调、reward hacking 风险） | 低（一个 loss，标准训练循环） |
| **理论关系** | 最大化 reward - KL | 等价于隐式 reward = β·log(π/π_ref) |
| **适用** | 需要精细 reward 信号的场景 | preference pair 充足、想简化训练的场景 |

DPO 本质上证明了：最优策略可以用 π_ref 和 preference pair 解析求解，无需显式 reward model。

### Q9. 什么是 SMPL？有哪些参数？

SMPL（Skinned Multi-Person Linear Model）是一个**参数化人体模型**，用一组低维参数表示人体 3D 网格。

**核心参数**：
| 参数 | 维度 | 含义 |
|---|---|---|
| **Shape β** | 10 维（或更多） | 控制体型（高矮胖瘦），通过 PCA 基变形 |
| **Pose θ** | 72 维（24 个关节 × 3 轴旋转） | 控制姿态（关节角度），用轴角表示 |
| **Translation t** | 3 维 | 全局平移 |

**输出**：6890 个顶点的三角网格

流程：`T-pose 模板 → Shape Blend Shape（β）→ Pose Blend Shape（θ）→ 线性蒙皮（LBS）→ 最终 mesh`

### Q10. SMPL 穿宽松衣服时的问题

SMPL 建模的是**裸体人体表面**，宽松衣服会导致：

1. **衣服穿透**：衣服表面在 SMPL 体表之内（尤其裙子、大衣），物理不合理
2. **无法表示衣物形变**：SMPL 的顶点贴合人体，无法表示飘动、褶皱等衣物独有的形变
3. **重建歧义**：从图像重建时，宽松衣服遮挡体型，β 和 θ 估计不准
4. **碰撞检测失效**：SMPL mesh 比衣物小，无法作为碰撞体

解决方案：SMPL+D（加 displacement）、ClothWild、SCAPE 变体，或直接用隐式表示（NeRF/SDF）建模穿衣人体。

### Q11. 介绍一下 Markov

**马尔可夫性质（Markov Property）**：未来状态只取决于当前状态，与历史无关。

$$P(X_{t+1} | X_t, X_{t-1}, \ldots, X_0) = P(X_{t+1} | X_t)$$

**马尔可夫链**：满足马尔可夫性的随机过程，由转移概率矩阵 P 完全定义。

在 AI 中的应用：
- **DDPM 前向过程**：q(x_t | x_{t-1}) 是马尔可夫链，每步只依赖上一步
- **DDPM 反向过程**：p(x_{t-1} | x_t) 也建模为马尔可夫链
- **RL**：MDP（马尔可夫决策过程）假设 state 满足马尔可夫性
- **HMM**：隐马尔可夫模型用于序列建模
- **MCMC**：马尔可夫链蒙特卡洛采样

关键性质：遍历性（ergodicity）→ 存在唯一平稳分布 π，长期采样收敛到 π。

### Q12. 单流和双流 Transformer Block

**双流（Two-stream / MM-DiT）**：
- 图像 token 和文本 token 分别通过**独立的** self-attention 和 FFN
- 在 attention 阶段共享 K、V（cross-attention）或拼接后做 joint attention
- 每种模态有自己的 LayerNorm 和投影矩阵
- 优势：每种模态保留独立的表示空间，适合早期层
- 例子：FLUX 的前半部分、SD3 的 MM-DiT

**单流（Single-stream）**：
- 图像 token 和文本 token 拼接成一个序列，共享同一套 self-attention + FFN
- 模态间交互完全通过 self-attention 自然发生
- 优势：更简单、参数更少、深层模态融合更充分
- 例子：FLUX 的后半部分

FLUX 的设计：**前半双流（保留模态特异性）→ 后半单流（深度融合）**，兼顾两者优势。

### Q13. DiT 使用的是 BN 还是 LN，为什么？

DiT 使用 **LayerNorm (LN)**，具体是 **adaLN-Zero**（自适应 LayerNorm）。

为什么不用 BN：
1. **Batch 依赖**：BN 沿 batch 维度统计均值/方差，batch size 小时统计不稳定
2. **不适合变长序列**：不同图像分辨率 → token 数不同，BN 统计无意义
3. **分布式训练困难**：多卡间需同步 BN 统计量
4. **条件注入不便**：DiT 需要根据时间步 t 和类别 c 动态调整 normalization，LN 更容易改造为 adaLN

adaLN-Zero：用条件嵌入（t, c）生成 γ, β, α 三个参数，分别做 scale、shift 和残差门控，**Zero 指 α 初始化为 0**（初始时残差支路输出为零，训练更稳定）。

### Q14. BN 在训练和推理时有什么区别？

| | 训练 | 推理 |
|---|---|---|
| **统计量来源** | 当前 mini-batch 的均值和方差 | 训练期间累积的 **running mean / running var**（指数移动平均） |
| **行为** | 每个 batch 统计不同 → 引入正则化噪声 | 固定统计量 → 确定性输出 |
| **模式切换** | `model.train()` | `model.eval()`（必须切换，否则结果不对） |

关键问题：
- 训练和推理的分布不一致 → batch size 越小差异越大
- 小 batch 时 BN 统计量方差大，训练不稳定
- 这也是为什么 Transformer / DiT 普遍选择 LN 的原因之一

---

## 手撕题

### 最长回文子串（DP）

```python
def longest_palindrome(s: str) -> tuple[int, str]:
    """返回最长回文子串的长度和子串本身"""
    n = len(s)
    if n <= 1:
        return n, s

    # dp[i][j] = True 表示 s[i..j] 是回文
    dp = [[False] * n for _ in range(n)]
    start, max_len = 0, 1

    # 单字符都是回文
    for i in range(n):
        dp[i][i] = True

    # 枚举子串长度从 2 到 n
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                dp[i][j] = (length == 2) or dp[i + 1][j - 1]
            if dp[i][j] and length > max_len:
                start, max_len = i, length

    return max_len, s[start:start + max_len]
```

### 手撕 Multi-Head Attention（einops 版）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        # self-attention 场景：一次性投影出 Q/K/V
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                       # (B, N, d_model)
        mask: torch.Tensor | None = None,      # (B, N) 或 (B, N, N)，True=保留
    ) -> torch.Tensor:
        # ── Step 1: 投影 + 拆 Q/K/V + 拆头，一行搞定 ──
        qkv = self.qkv_proj(x)                                          # (B, N, 3*d_model)
        q, k, v = rearrange(
            qkv, 'b n (three h d) -> three b h n d', three=3, h=self.n_heads
        )                                                               # 3 × (B, h, N, d_head)

        # ── Step 2: Scaled Dot-Product Attention ──
        # 用 einsum 写 QK^T，下标即文档：i=query 位置，j=key 位置
        scores = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if mask is not None:
            if mask.dim() == 2:                                         # (B, N) → key padding mask
                mask = rearrange(mask, 'b j -> b 1 1 j')
            else:                                                       # (B, N, N) → 自定义 mask
                mask = rearrange(mask, 'b i j -> b 1 i j')
            scores = scores.masked_fill(~mask, float('-inf'))

        attn = self.dropout(F.softmax(scores, dim=-1))                  # (B, h, N, N)

        # ── Step 3: 加权求和 + 合并多头 ──
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)      # (B, h, N, d_head)
        out = rearrange(out, 'b h n d -> b n (h d)')                    # (B, N, d_model)
        return self.out_proj(out)
```

**einops 改写要点**：

| 操作 | 传统写法 | einops 写法 |
|---|---|---|
| 拆 Q/K/V + 拆头 + 转置 | `view + chunk + transpose` 三步 | `rearrange('b n (three h d) -> three b h n d')` 一行 |
| 计算 QK^T | `Q @ K.transpose(-2, -1)` | `einsum('b h i d, b h j d -> b h i j')`（下标自带语义） |
| 合并多头 | `transpose(1, 2).contiguous().view(B, N, d_model)` | `rearrange('b h n d -> b n (h d)')` |
| mask 广播 | 多次 `unsqueeze` | `rearrange('b j -> b 1 1 j')` 显式声明广播维度 |

不可变性：`rearrange` 返回新 view，不修改输入张量；`out_proj` 输出新张量，符合不可变风格。

---

## 二面

### 9. Attention 怎么算？为什么除 √d？除 d 行不行？

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**为什么除 √d_k**：
- Q 和 K 的每个元素假设为均值 0、方差 1 的独立随机变量
- 点积 q·k = Σq_i·k_i，其方差 = d_k（d_k 个方差为 1 的独立项求和）
- 不 scale 时，d_k 越大，点积值越大，softmax 趋近 one-hot → 梯度消失
- 除以 √d_k 使点积方差归一化为 1

**除 d 行不行**：数学上可以，但会让 attention score 方差变为 1/d_k（太小），softmax 输出趋于均匀分布 → 注意力无法区分 token，模型表达力下降。√d_k 是方差归一化的最优选择。

### 10. GRPO 在 Dense vs MoE 模型上的训练差异

| | Dense | MoE |
|---|---|---|
| **梯度更新** | 所有参数每步都更新 | 只有被激活的 expert 被更新 |
| **Entropy collapse 风险** | 均匀 | 更高——部分 expert 可能持续被选中并过度强化 |
| **额外需要** | 无 | 辅助 loss（load balancing）防止 expert 退化 |
| **KL 正则** | 标准做法 | 更关键——防止 router 将所有 token 路由到少数 expert |
| **显存** | 参数量 = 计算量 | 总参数量大但每步只激活一部分 |

### 11. PPO 的 Value Model

**Value Model（Critic）**：
- 与 Actor 通常共享 backbone，但**输出头不同**
- Actor 输出：π(a|s)，一个 token 词表上的概率分布（softmax）
- Critic 输出：V(s)，一个标量，预测从当前状态到回合结束的期望累计 reward
- 实现上：backbone 最后一层接一个 Linear(hidden_dim, 1) 投影头

---

## 手撕公式 + 代码

### PPO 公式

$$\mathcal{L}_{\text{PPO}} = -\mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta), 1{-}\epsilon, 1{+}\epsilon) \hat{A}_t \right) \right]$$

- $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\text{old}}(a_t|s_t)$
- $\hat{A}_t = \text{GAE}(\delta_t) = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$，$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

### GRPO 公式

$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}_{x} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \min\left( r_{i,t} \hat{A}_i,\ \text{clip}(r_{i,t}, 1{-}\epsilon, 1{+}\epsilon) \hat{A}_i \right) \right] + \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

- $r_{i,t}(\theta) = \pi_\theta(y_{i,t}|x, y_{i,<t}) / \pi_{\text{old}}(y_{i,t}|x, y_{i,<t})$
- $\hat{A}_i = (R_i - \mu_G) / \sigma_G$（**组内归一化，无 Critic**）
- KL 用 Schulman 无偏估计器：$D_{\text{KL}} \approx \mathbb{E}_\theta[\pi_{\text{ref}}/\pi_\theta - \log(\pi_{\text{ref}}/\pi_\theta) - 1]$

### PPO vs GRPO 核心区别

| | PPO | GRPO |
|---|---|---|
| Advantage | GAE(V(s)) — 依赖 Critic | (R − μ)/σ — 组内统计 |
| KL 约束 | 靠 clip 隐式约束 | clip + 显式 β·D_KL |
| 粒度 | token-level advantage | **sequence-level** advantage |

### 手撕 GRPO Loss（PyTorch）

```python
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat


def grpo_loss(
    logits: torch.Tensor,        # (B*G, T, V) 当前策略的 logits
    input_ids: torch.Tensor,     # (B*G, T) token ids
    old_log_probs: torch.Tensor, # (B*G, T) 采样时 π_old 的 log prob
    ref_log_probs: torch.Tensor, # (B*G, T) reference model 的 log prob
    rewards: torch.Tensor,       # (B*G,) 每条回答的 reward
    response_mask: torch.Tensor, # (B*G, T) 1=response token, 0=prompt token
    group_size: int,             # G
    clip_eps: float = 0.2,
    kl_beta: float = 0.1,
) -> torch.Tensor:
    """
    GRPO-Clip loss with KL regularization (einops version).
    B = batch of prompts, G = rollouts per prompt, T = seq len, V = vocab size
    """

    # ── Step 1: 当前策略的 per-token log prob ──
    log_probs = F.log_softmax(logits, dim=-1)                      # (B*G, T, V)
    token_log_probs = log_probs.gather(
        dim=-1, index=rearrange(input_ids, 'bg t -> bg t 1')
    )                                                               # (B*G, T, 1)
    token_log_probs = rearrange(token_log_probs, 'bg t 1 -> bg t') # (B*G, T)

    # ── Step 2: 组内归一化 Advantage（无 Critic）──
    #   reshape (B*G,) → (B, G)，组内求 mean/std，再铺回 (B*G,)
    r_grouped = rearrange(rewards, '(b g) -> b g', g=group_size)   # (B, G)
    mean_r = reduce(r_grouped, 'b g -> b 1', 'mean')               # (B, 1)
    std_r = r_grouped.std(dim=1, keepdim=True).clamp(min=1e-8)     # (B, 1)
    advantages = rearrange(
        (r_grouped - mean_r) / std_r, 'b g -> (b g)'
    )                                                               # (B*G,)

    # ── Step 3: 重要性采样比 r(θ) = π_new / π_old ──
    ratio = torch.exp(token_log_probs - old_log_probs)              # (B*G, T)

    # ── Step 4: PPO-style clip objective ──
    #   sequence-level advantage 广播到每个 token
    adv = repeat(advantages, 'bg -> bg t', t=ratio.shape[1])       # (B*G, T)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    clip_obj = torch.min(surr1, surr2)                              # (B*G, T)

    # masked mean：只算 response token，per-sequence 长度归一化
    resp_len = reduce(response_mask, 'bg t -> bg', 'sum').clamp(min=1)  # (B*G,)
    policy_loss = -reduce(
        clip_obj * response_mask, 'bg t -> bg', 'sum'
    ) / resp_len                                                    # (B*G,)
    policy_loss = policy_loss.mean()

    # ── Step 5: KL 正则（Schulman 无偏估计器）──
    #   D_KL ≈ E_θ[ π_ref/π_θ − log(π_ref/π_θ) − 1 ]
    log_r = ref_log_probs - token_log_probs                         # log(π_ref/π_θ)
    kl_per_token = torch.exp(log_r) - log_r - 1                    # (B*G, T)
    kl_loss = reduce(
        kl_per_token * response_mask, 'bg t -> bg', 'sum'
    ) / resp_len                                                    # (B*G,)
    kl_loss = kl_loss.mean()

    # ── Total loss ──
    return policy_loss + kl_beta * kl_loss
```

### 代码与公式对应

| 代码 (einops) | 公式 | 意义 |
|---|---|---|
| `rearrange(rewards, '(b g) -> b g')` | 将 B*G 条 reward 按 prompt 分组 | 为组内归一化做准备 |
| `(r_grouped - mean_r) / std_r` | Â = (R − μ) / σ | 组内归一化 advantage，**替代 Critic** |
| `repeat(adv, 'bg -> bg t')` | 整条回答共享 Â_i | Sequence-level advantage 广播到 token |
| `torch.min(surr1, surr2)` | min(r·A, clip(r)·A) | PPO-style clip 防止过大更新 |
| `reduce(... * mask, 'bg t -> bg', 'sum') / resp_len` | (1/\|y_i\|) Σ_t | Response masking + 长度归一化 |
| `exp(log_r) - log_r - 1` | π_ref/π_θ − log(π_ref/π_θ) − 1 | Schulman KL 无偏估计 |
