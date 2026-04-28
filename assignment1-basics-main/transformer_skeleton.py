"""
Transformer 实现练习 - 填空版
按照 TODO 标记依次实现各个组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum, reduce
import math


# ============================================================================
# Level 0: 基础算子
# ============================================================================

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """数值稳定的 softmax"""
    # TODO: 实现 softmax
    x_max=x.max(dim=dim,keepdim=True)[0]
    x_exp=torch.exp(x-x_max)
    return x_exp/x_exp.sum(dim=dim,keepdim=True)
    # 提示: 先减去 max(x) 防止溢出
    # 1. x_max = x.max(dim=dim, keepdim=True)[0]
    # 2. x_exp = torch.exp(x - x_max)
    # 3. return x_exp / x_exp.sum(dim=dim, keepdim=True)

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    交叉熵损失
    logits: (batch, vocab_size)
    targets: (batch,) 整数索引
    """
    x_max=reduce(logits,"b v -> b 1", "max")
    logits_stable=logits-x_max
    log_sum_exp=torch.log(reduce(torch.exp(logits_stable),"b v -> b 1", "sum"))
    target_logits=logits_stable[torch.arange(logits.shape[0]),targets]
    losses=-target_logits+log_sum_exp
    return reduce(losses,"b ->","mean")
    # TODO: 实现交叉熵
    # 提示: CE = -target_logit + log(sum(exp(logits)))
    # 1. 减去 max 保证数值稳定
    # 2. log_sum_exp = log(sum(exp(logits_stable)))
    # 3. target_logits = logits_stable[range(batch), targets]
    # 4. loss = -target_logits + log_sum_exp
    # 5. return loss.mean()


# ============================================================================
# Level 1: 基础层
# ============================================================================

class Embedding(nn.Module):
    """Token embedding 层"""
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.weight=nn.Parameter(torch.empty(vocab_size,d_model))
        nn.init.trunc_normal_(self.weight)
        # TODO: 创建 weight 参数
        # self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        # nn.init.trunc_normal_(self.weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
        # TODO: 查表返回 embedding
        # return self.weight[token_ids]



class Linear(nn.Module):
    """无偏置线性层"""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight=nn.Parameter(torch.empty(out_features,in_features))
        nn.init.trunc_normal_(self.weight)
        # TODO: 创建 weight 参数 (out_features, in_features)
        # self.weight = nn.Parameter(torch.empty(out_features, in_features))
        # nn.init.trunc_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... inp, out inp -> ... out")
        # TODO: 实现 x @ W^T
        # return einsum(x, self.weight, "... inp, out inp -> ... out")


# ============================================================================
# Level 2: 归一化 & 位置编码 & 激活
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight=nn.Parameter(torch.ones(d_model))
        # TODO: 创建可学习的 weight 参数
        # self.weight = nn.Parameter(torch.ones(d_model))
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float=x.float()
        variance=reduce(x_float**2, "... d-> ... 1","mean")
        x_norm=x_float*torch.rsqrt(variance+self.eps)
        x_norm=x_norm.to(x.dtype)
        return x_norm*self.weight
        # TODO: 实现 RMSNorm
        # 1. 转 float32: x_float = x.float()
        # 2. variance = mean(x^2): reduce(x_float**2, "... d -> ... 1", "mean")
        # 3. x_norm = x_float / sqrt(variance + eps)
        # 4. 转回原 dtype 并乘 weight


class RotaryPositionalEmbedding(nn.Module):
    """RoPE 旋转位置编码"""
    def __init__(self, theta: float, d_k: int, max_seq_len: int):
        super().__init__()
        if d_k%2 != 0:
            raise ValueError(f"d_k must be even, got {d_k}")
        freq_indices=torch.arange(0,d_k,2,dtype=torch.float32)
        freqs=1.0/(theta**(freq_indices/d_k))
        positions=torch.arange(max_seq_len)
        angles=einsum(positions,freqs, "pos, freq-> pos freq")
        self.register_buffer("cos_cache",angles.cos(),persistent=False)
        self.register_buffer("sin_cache",angles.sin(),persistent=False)

        # TODO: 预计算 cos/sin cache
        # 1. freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2) / d_k))
        # 2. positions = torch.arange(max_seq_len)
        # 3. angles = einsum(positions, freqs, "pos, freq -> pos freq")
        # 4. self.register_buffer("cos_cache", angles.cos())
        # 5. self.register_buffer("sin_cache", angles.sin())
        

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: (..., seq_len, d_k)
        token_positions: (..., seq_len)
        """
        cos=self.cos_cache[token_positions]
        sin=self.sin_cache[token_positions]
        x_pairs=rearrange(x, "... (d pair)-> ... d pair",pair=2)
        x_even,x_odd=x_pairs[...,0],x_pairs[...,1]
        rotated_even=x_even*cos-x_odd*sin
        rotated_odd=x_even*sin+x_odd*cos
        return rearrange([rotated_even,rotated_odd],"pair ... seq d -> ... seq (d pair)")
        # TODO: 应用旋转
        # 1. cos = self.cos_cache[token_positions]
        # 2. sin = self.sin_cache[token_positions]
        # 3. x_pairs = rearrange(x, "... (d pair) -> ... d pair", pair=2)
        # 4. x_even, x_odd = x_pairs[..., 0], x_pairs[..., 1]
        # 5. rotated_even = x_even * cos - x_odd * sin
        # 6. rotated_odd = x_even * sin + x_odd * cos
        # 7. return rearrange([rotated_even, rotated_odd], "pair ... d -> ... (d pair)") 


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network"""
    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        if d_ff is None:
            d_ff = int(8 * d_model / 3)
            d_ff = ((d_ff + 63) // 64) * 64
        self.w1=nn.Linear(d_model,d_ff,bias=False)
        self.w2=nn.Linear(d_ff,d_model,bias=False)
        self.w3=nn.Linear(d_model,d_ff,bias=False)
        # TODO: 创建三个线性层 w1, w2, w3
        # self.w1 = nn.Linear(d_model, d_ff, bias=False)
        # self.w2 = nn.Linear(d_ff, d_model, bias=False)
        # self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x))*self.w3(x))
        # TODO: 实现 SwiGLU
        # return self.w2(F.silu(self.w1(x)) * self.w3(x))
        

# ============================================================================
# Level 3: 注意力
# ============================================================================

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Q, K, V: (..., seq, d_k)
    mask: (..., seq_q, seq_k) bool, True=可见
    """
    d_k=Q.shape[-1]
    score=einsum(Q,K, "... q d, ... k d -> ... q k ")/math.sqrt(d_k)
    if mask is not None:
        score=score.masked_fill(~mask,float("-inf"))
    attn_weights=softmax(score,dim=-1)
    output=einsum(attn_weights,V,"... q k, ... k d -> ... q d")
    return output
    # TODO: 实现注意力
    # 1. d_k = Q.shape[-1]
    # 2. scores = einsum(Q, K, "... q d, ... k d -> ... q k") / sqrt(d_k)
    # 3. if mask: scores = scores.masked_fill(~mask, float("-inf"))
    # 4. attn_weights = softmax(scores, dim=-1)
    # 5. output = einsum(attn_weights, V, "... q k, ... k d -> ... q d")


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力"""
    def __init__(self, d_model: int, num_heads: int, rope: RotaryPositionalEmbedding = None):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rope = rope
        self.q_proj=nn.Linear(d_model,d_model,bias=False)
        self.k_proj=nn.Linear(d_model,d_model,bias=False)
        self.v_proj=nn.Linear(d_model,d_model,bias=False)
        self.output_proj=nn.Linear(d_model,d_model,bias=False)
        # TODO: 创建 Q, K, V, output 投影层
        # self.q_proj = nn.Linear(d_model, d_model, bias=False)
        # self.k_proj = nn.Linear(d_model, d_model, bias=False)
        # self.v_proj = nn.Linear(d_model, d_model, bias=False)
        # self.output_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        """x: (batch, seq, d_model)"""
        batch, seq_len, d_model = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        Q = rearrange(Q,"... seq (head d) -> ... head seq d",head=self.num_heads)
        K = rearrange(K,"... seq (head d) -> ... head seq d",head=self.num_heads)
        V = rearrange(V,"... seq (head d) -> ... head seq d",head=self.num_heads)
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len)
            rope_positions = token_positions.unsqueeze(-2) if token_positions.dim() >=1 else token_positions
            Q = self.rope(Q, rope_positions)
            K = self.rope(K, rope_positions)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        attn_out =scaled_dot_product_attention(Q,K,V,mask=causal_mask)
        attn_output=rearrange(attn_out," ... head seq d -> ... seq (head d)")
        output=self.output_proj(attn_output)
        return output
        # TODO: 实现多头注意力
        # 1. Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # 2. 拆分多头: rearrange(Q, "b s (h d) -> b h s d", h=self.num_heads)
        # 3. 如果有 RoPE:
        #    rope_pos = token_positions.unsqueeze(-2) if token_positions.dim() >= 1 else token_positions
        #    Q = self.rope(Q, rope_pos)
        #    K = self.rope(K, rope_pos)
        # 4. 因果遮罩: causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        # 5. attn_out = scaled_dot_product_attention(Q, K, V, causal_mask)
        # 6. 拼接: rearrange(attn_out, "b h s d -> b s (h d)")
        # 7. output = self.output_proj(concat)


# ============================================================================
# Level 4: Transformer Block
# ============================================================================

class TransformerBlock(nn.Module):
    """Pre-Norm Transformer Block"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.ln1=RMSNorm(d_model)
        self.ln2=RMSNorm(d_model)
        rope=RotaryPositionalEmbedding(theta,d_model//num_heads,max_seq_len)
        self.attn=MultiHeadSelfAttention(d_model,num_heads,rope)
        self.ffn=SwiGLU(d_model,d_ff)
        # TODO: 创建组件
        # self.ln1 = RMSNorm(d_model)
        # self.ln2 = RMSNorm(d_model)
        # rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)
        # self.attn = MultiHeadSelfAttention(d_model, num_heads, rope)
        # self.ffn = SwiGLU(d_model, d_ff)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        # TODO: 实现 Pre-Norm 残差块
        x+=self.attn(self.ln1(x), token_positions)
        x+=self.ffn(self.ln2(x))
        return x
        # x = x + self.attn(self.ln1(x), token_positions)
        # x = x + self.ffn(self.ln2(x))
        # return x


# ============================================================================
# Level 5: 完整模型
# ============================================================================

class TransformerLM(nn.Module):
    """Transformer Language Model"""
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0
    ):
        super().__init__()
        # TODO: 创建所有组件
        # self.token_embeddings = Embedding(vocab_size, d_model)
        # self.layers = nn.ModuleList([
        #     TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
        #     for _ in range(num_layers)
        # ])
        # self.ln_final = RMSNorm(d_model)
        # self.lm_head = Linear(d_model, vocab_size)
        pass

    def forward(self, input_ids: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        """
        input_ids: (batch, seq)
        return: (batch, seq, vocab_size)
        """
        # TODO: 实现前向传播
        # 1. x = self.token_embeddings(input_ids)
        # 2. if token_positions is None:
        #    token_positions = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
        # 3. for layer in self.layers: x = layer(x, token_positions)
        # 4. x = self.ln_final(x)
        # 5. logits = self.lm_head(x)
        # 6. return logits
        pass


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("测试你的实现...")

    # 测试 softmax
    x = torch.randn(2, 5)
    out = softmax(x, dim=-1)
    assert out.shape == x.shape
    assert torch.allclose(out.sum(dim=-1), torch.ones(2))
    print("✓ softmax 通过")

    # 测试 Embedding
    emb = Embedding(100, 64)
    tokens = torch.randint(0, 100, (2, 10))
    out = emb(tokens)
    assert out.shape == (2, 10, 64)
    print("✓ Embedding 通过")

    # 测试 RMSNorm
    norm = RMSNorm(64)
    x = torch.randn(2, 10, 64)
    out = norm(x)
    assert out.shape == x.shape
    print("✓ RMSNorm 通过")

    # 测试 RoPE
    rope = RotaryPositionalEmbedding(10000.0, 16, 128)
    x = torch.randn(2, 4, 8, 16)  # (batch, heads, seq, d_k)
    pos = torch.arange(8).unsqueeze(0).unsqueeze(0).expand(2, 1, -1)
    out = rope(x, pos)
    assert out.shape == x.shape
    print("✓ RoPE 通过")

    # 测试 SwiGLU
    ffn = SwiGLU(64)
    x = torch.randn(2, 10, 64)
    out = ffn(x)
    assert out.shape == x.shape
    print("✓ SwiGLU 通过")

    # 测试 Attention
    Q = K = V = torch.randn(2, 4, 8, 16)
    mask = torch.tril(torch.ones(8, 8, dtype=torch.bool))
    out = scaled_dot_product_attention(Q, K, V, mask)
    assert out.shape == (2, 4, 8, 16)
    print("✓ Attention 通过")

    # 测试 MultiHeadSelfAttention
    rope = RotaryPositionalEmbedding(10000.0, 16, 128)
    attn = MultiHeadSelfAttention(64, 4, rope)
    x = torch.randn(2, 8, 64)
    pos = torch.arange(8).unsqueeze(0).expand(2, -1)
    out = attn(x, pos)
    assert out.shape == x.shape
    print("✓ MultiHeadSelfAttention 通过")

    # 测试 TransformerBlock
    block = TransformerBlock(64, 4, 256, 128, 10000.0)
    x = torch.randn(2, 8, 64)
    out = block(x, pos)
    assert out.shape == x.shape
    print("✓ TransformerBlock 通过")

    # 测试完整模型
    model = TransformerLM(256, 128, 64, 2, 4, 256)
    input_ids = torch.randint(0, 256, (2, 16))
    logits = model(input_ids)
    assert logits.shape == (2, 16, 256)
    print("✓ TransformerLM 通过")

    # 测试损失
    targets = torch.randint(0, 256, (2, 16))
    loss = cross_entropy(logits.view(-1, 256), targets.view(-1))
    assert loss.dim() == 0
    print("✓ cross_entropy 通过")

    print("\n所有测试通过! 🎉")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
