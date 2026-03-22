"""Scaled dot-product attention implementation."""

import torch
from jaxtyping import Bool, Float
from torch import Tensor
import math
from einops import einsum


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Compute scaled dot-product attention.

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Args:
        Q: Query tensor of shape (..., seq_len_q, d_k)
        K: Key tensor of shape (..., seq_len_k, d_k)
        V: Value tensor of shape (..., seq_len_k, d_v)
        mask: Optional boolean mask of shape (..., seq_len_q, seq_len_k).
              True means attend, False means mask out (set to -inf before softmax).

    Returns:
        Output tensor of shape (..., seq_len_q, d_v)
    """
    d_k = Q.shape[-1]

    # Compute attention scores: Q @ K^T / sqrt(d_k)
    # Q: (..., queries, d_k), K: (..., keys, d_k)
    # scores: (..., queries, keys)
    scores = einsum(Q, K, "... queries d, ... keys d -> ... queries keys") / math.sqrt(d_k)

    # Apply mask: set False positions to -inf
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # Apply softmax along the last dimension (keys dimension)
    # Using our softmax implementation for numerical stability
    from homework.softmax import softmax
    attn_weights = softmax(scores, dim=-1)

    # Compute output: attention_weights @ V
    # attn_weights: (..., queries, keys), V: (..., keys, d_v)
    # output: (..., queries, d_v)
    output = einsum(attn_weights, V, "... queries keys, ... keys d -> ... queries d")

    return output
