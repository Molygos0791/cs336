"""SwiGLU feed-forward network (LLaMA-style)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def swiglu_ffn(d_model: int, d_ff: int | None = None) -> int:
    """Compute SwiGLU inner dimension: ~(8/3)*d_model rounded up to multiple of 64."""
    if d_ff is not None:
        return d_ff
    target_d_ff = int(8 * d_model / 3)
    return ((target_d_ff + 63) // 64) * 64


class SwiGLU(nn.Module):
    """SwiGLU (Swish-Gated Linear Unit) feed-forward network.

    Formula: output = (silu(x @ w1.T) * (x @ w3.T)) @ w2.T

    Args:
        d_model: Model hidden dimension.
        d_ff: Inner FFN dimension. If None, auto-computed as (8/3)*d_model.
        device: Device to store parameters on.
        dtype: Data type for parameters.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = swiglu_ffn(d_model, d_ff)
        self.w1 = nn.Linear(d_model, self.d_ff, bias=False, device=device, dtype=dtype)
        self.w2 = nn.Linear(self.d_ff, d_model, bias=False, device=device, dtype=dtype)
        self.w3 = nn.Linear(d_model, self.d_ff, bias=False, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
