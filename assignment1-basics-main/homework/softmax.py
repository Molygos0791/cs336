"""Softmax implementation with numerical stability."""

import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Apply softmax to the specified dimension of the input tensor.

    Uses the numerical stability trick of subtracting the maximum value
    along the specified dimension before computing exponentials.

    Args:
        x: Input tensor. Shape is arbitrary.
        dim: Dimension to apply softmax to.

    Returns:
        Tensor with the same shape as input with softmax applied along the specified dim.
    """
    x_max = x.max(dim=dim, keepdim=True)[0]
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)
