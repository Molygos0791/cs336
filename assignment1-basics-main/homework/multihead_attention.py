"""
Multi-head self-attention module implementation.

Following Vaswani et al. [2017], "Attention Is All You Need".
"""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from einops import rearrange
from typing import TYPE_CHECKING

from homework.attention import scaled_dot_product_attention

if TYPE_CHECKING:
    from homework.rope import RotaryPositionalEmbedding


class MultiHeadSelfAttention(nn.Module):
    """
    Causal multi-head self-attention module.

    This implements the multi-head attention mechanism from "Attention Is All You Need"
    (Vaswani et al., 2017) with causal masking for autoregressive language modeling.

    Args:
        d_model (int): Dimensionality of the Transformer block inputs.
        num_heads (int): Number of attention heads. d_model must be divisible by num_heads.
        rope (RotaryPositionalEmbedding | None): Optional RoPE module for positional encoding.
        device (torch.device | None): Device to store parameters on.
        dtype (torch.dtype | None): Data type for parameters.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Per-head dimension
        self.d_v = d_model // num_heads  # Per-head dimension
        self.rope = rope

        # Q, K, V projection layers (no bias, following the paper)
        # Weight shape: (d_model, d_model) - projects from d_model to d_model
        # Then we split into num_heads heads, each with d_k dimensions
        self.q_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)

        # Output projection layer
        self.output_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
    ) -> Float[Tensor, "... seq_len d_model"]:
        """
        Forward pass for multi-head self-attention.

        Args:
            x: Input tensor of shape (..., seq_len, d_model).
               Arbitrary batch dimensions are supported.
            token_positions: Optional token positions for RoPE.
                            Required if rope was provided during initialization.

        Returns:
            Output tensor of shape (..., seq_len, d_model).
        """
        # Get input shape
        *batch_dims, seq_len, d_model = x.shape

        # Project to Q, K, V
        # Shape: (..., seq_len, d_model)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention
        # From (..., seq_len, d_model) to (..., num_heads, seq_len, d_k)
        Q = rearrange(Q, "... seq (head d) -> ... head seq d", head=self.num_heads)
        K = rearrange(K, "... seq (head d) -> ... head seq d", head=self.num_heads)
        V = rearrange(V, "... seq (head d) -> ... head seq d", head=self.num_heads)

        # Apply RoPE if available
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # Create causal mask
        # Shape: (seq_len, seq_len)
        # mask[i, j] = True if position i can attend to position j
        # For causal attention: i >= j (each position can only attend to itself and earlier positions)
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        )

        # Apply scaled dot-product attention with causal mask
        # Output shape: (..., num_heads, seq_len, d_k)
        attn_output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        # Reshape back to (..., seq_len, d_model)
        attn_output = rearrange(attn_output, "... head seq d -> ... seq (head d)")

        # Apply output projection
        output = self.output_proj(attn_output)

        return output
