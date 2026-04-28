"""Linear layer implementation (no bias)."""

import torch
import torch.nn as nn
from torch import Tensor
from einops import einsum


class Linear(nn.Module):
    """Linear transformation without bias.

    Weight shape follows PyTorch convention: (out_features, in_features).

    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        device: Device to store parameters on.
        dtype: Data type for parameters.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        return einsum(x, self.weight, "... inp, out inp -> ... out")
