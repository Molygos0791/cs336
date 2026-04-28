"""RMSNorm (Root Mean Square Layer Normalization)."""

import torch
import torch.nn as nn
from torch import Tensor
from einops import reduce


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Compared to LayerNorm, RMSNorm skips mean centering and only uses
    the root mean square for normalization, improving efficiency.

    Args:
        d_model: Hidden dimension size.
        eps: Small constant for numerical stability.
        device: Device to store parameters on.
        dtype: Data type for parameters.
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: Tensor) -> Tensor:
        original_dtype = x.dtype
        x_float = x.to(torch.float32)
        variance = reduce(x_float ** 2, "... d -> ... 1", "mean")
        x_norm = x_float * torch.rsqrt(variance + self.eps)
        x_norm = x_norm.to(original_dtype)
        return x_norm * self.weight
