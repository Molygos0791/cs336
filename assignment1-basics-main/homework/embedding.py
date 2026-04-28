"""Token embedding layer."""

import torch
import torch.nn as nn
from torch import Tensor


class Embedding(nn.Module):
    """Maps token IDs to dense vectors via a lookup table."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]
