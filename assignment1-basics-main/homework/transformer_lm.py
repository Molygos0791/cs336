"""
Transformer Language Model implementation.

This implements a complete Transformer language model with:
- Token embeddings
- Multiple Transformer blocks with pre-norm, RoPE, and SwiGLU
- Final RMSNorm
- Language model head for vocabulary projection
"""

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from einops import repeat

from homework.embedding import Embedding
from homework.linear import Linear
from homework.rmsnorm import RMSNorm
from homework.transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    """
    Transformer Language Model.

    Architecture:
        1. Token embeddings
        2. Multiple Transformer blocks (pre-norm with RoPE and SwiGLU)
        3. Final RMSNorm
        4. Language model head (linear projection to vocabulary)

    Args:
        vocab_size (int): Size of the vocabulary.
        context_length (int): Maximum context length (sequence length).
        d_model (int): Dimensionality of the model embeddings.
        num_layers (int): Number of Transformer blocks.
        num_heads (int): Number of attention heads.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        rope_theta (float): Base frequency for RoPE.
        device (torch.device | None): Device to store parameters on.
        dtype (torch.dtype | None): Data type for parameters.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        # Token embeddings
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])

        # Final layer normalization
        self.ln_final = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)

        # Language model head (project to vocabulary)
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        input_ids: Int[Tensor, " batch_size sequence_length"],
        token_positions: Int[Tensor, " batch_size sequence_length"] | None = None,
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        """
        Forward pass for the Transformer language model.

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length).
            token_positions: Token positions for RoPE. If None, uses [0, 1, ..., seq_len-1].

        Returns:
            Logits tensor of shape (batch_size, sequence_length, vocab_size).
        """
        # Get sequence length from input
        batch_size, seq_len = input_ids.shape

        # Generate token positions if not provided
        if token_positions is None:
            token_positions = repeat(
                torch.arange(seq_len, device=input_ids.device),
                "s -> b s", b=batch_size,
            )

        # Step 1: Token embeddings
        # Shape: (batch_size, sequence_length, d_model)
        x = self.token_embeddings(input_ids)

        # Step 2: Pass through transformer layers
        for layer in self.layers:
            x = layer(x, token_positions)

        # Step 3: Final layer normalization
        x = self.ln_final(x)

        # Step 4: Language model head (project to vocabulary)
        # Shape: (batch_size, sequence_length, vocab_size)
        logits = self.lm_head(x)

        return logits
