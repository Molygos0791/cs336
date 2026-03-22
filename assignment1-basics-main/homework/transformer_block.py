"""
Pre-norm Transformer Block implementation.

This implements a pre-normalization Transformer block with:
- RMSNorm for layer normalization
- Multi-head self-attention with RoPE
- SwiGLU feed-forward network
"""

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from homework.rmsnorm import RMSNorm
from homework.swiglu import SwiGLU
from homework.rope import RotaryPositionalEmbedding
from homework.multihead_attention import MultiHeadSelfAttention


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block.

    Architecture:
        x_out = x + Attention(RMSNorm(x))
        x_out = x_out + FFN(RMSNorm(x_out))

    Where:
        - RMSNorm is root mean square layer normalization
        - Attention is causal multi-head self-attention with RoPE
        - FFN is SwiGLU feed-forward network

    Args:
        d_model (int): Dimensionality of the Transformer block inputs.
        num_heads (int): Number of heads to use in multi-head self-attention.
        d_ff (int): Dimensionality of the position-wise feed-forward inner layer.
        max_seq_len (int): Maximum sequence length for RoPE cache.
        theta (float): Base frequency for RoPE.
        device (torch.device | None): Device to store parameters on.
        dtype (torch.dtype | None): Data type for parameters.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # Pre-norm layers
        self.ln1 = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)

        # RoPE for positional encoding
        d_k = d_model // num_heads
        rope = RotaryPositionalEmbedding(
            theta=theta,
            d_k=d_k,
            max_seq_len=max_seq_len,
            device=device,
        )

        # Multi-head self-attention with RoPE
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope=rope,
            device=device,
            dtype=dtype,
        )

        # Feed-forward network (SwiGLU)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, "batch seq_len d_model"],
        token_positions: Int[Tensor, "batch seq_len"] | None = None,
    ) -> Float[Tensor, "batch seq_len d_model"]:
        """
        Forward pass for the Transformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            token_positions: Token positions for RoPE. If None, uses [0, 1, ..., seq_len-1].

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        # Pre-norm attention with residual connection
        # x_out = x + Attention(RMSNorm(x))
        x_norm = self.ln1(x)
        attn_out = self.attn(x_norm, token_positions)
        x = x + attn_out

        # Pre-norm FFN with residual connection
        # x_out = x_out + FFN(RMSNorm(x_out))
        x_norm = self.ln2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out

        return x
