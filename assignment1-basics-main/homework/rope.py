import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from einops import rearrange


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) module.

    RoPE encodes position information by rotating the query and key vectors
    in the attention mechanism. The rotation angle is determined by the
    position of the token in the sequence.

    Args:
        theta (float): Base frequency for the rotary embeddings (Θ in the paper).
        d_k (int): Dimension of the query/key vectors. Must be even.
        max_seq_len (int): Maximum sequence length to precompute caches for.
        device (torch.device | None): Device to store the buffers on.
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even, got {d_k}")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Compute the frequency bands: 1 / (theta^(2i/d_k)) for i in [0, d_k/2)
        # Shape: (d_k // 2,)
        freq_indices = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)
        freqs = 1.0 / (theta ** (freq_indices / d_k))

        # Compute position indices: [0, 1, 2, ..., max_seq_len - 1]
        # Shape: (max_seq_len,)
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)

        # Compute the angles: position * freq for each position and frequency
        # Shape: (max_seq_len, d_k // 2)
        angles = torch.outer(positions, freqs)

        # Precompute cos and sin caches
        self.register_buffer("cos_cache", angles.cos(), persistent=False)
        self.register_buffer("sin_cache", angles.sin(), persistent=False)

    def forward(
        self, x: Float[Tensor, "... seq_len d_k"], token_positions: Int[Tensor, "... seq_len"]
    ) -> Float[Tensor, "... seq_len d_k"]:
        """
        Apply rotary positional embedding to the input tensor.

        Args:
            x: Input tensor of shape (..., seq_len, d_k) with arbitrary batch dimensions.
            token_positions: Tensor of shape (..., seq_len) specifying the position
                           of each token in the sequence.

        Returns:
            Tensor of the same shape as x with rotary embeddings applied.
        """
        # Get the cos and sin values for the specified positions
        # token_positions: (..., seq_len)
        # cos_cache, sin_cache: (max_seq_len, d_k // 2)
        # After indexing: (..., seq_len, d_k // 2)
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        # Split x into even and odd indices for the rotation using rearrange
        # x shape: (..., seq_len, d_k) -> (..., seq_len, d_k // 2, 2)
        x_pairs = rearrange(x, "... (d pair) -> ... d pair", pair=2)
        x_even = x_pairs[..., 0]  # Shape: (..., seq_len, d_k // 2)
        x_odd = x_pairs[..., 1]   # Shape: (..., seq_len, d_k // 2)

        # Apply the rotation:
        # rotated_even = x_even * cos - x_odd * sin
        # rotated_odd = x_even * sin + x_odd * cos
        # This is equivalent to a 2D rotation for each pair (x[2i], x[2i+1])
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        # Interleave the rotated even and odd indices back together using rearrange
        # (..., seq_len, d_k // 2, 2) -> (..., seq_len, d_k)
        return rearrange([rotated_even, rotated_odd], "pair ... seq d -> ... seq (d pair)")
