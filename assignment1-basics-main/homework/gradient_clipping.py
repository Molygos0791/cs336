"""Gradient clipping by global L2 norm."""

import torch
from collections.abc import Iterable


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Clip combined gradient L2 norm to at most max_l2_norm (in-place).

    If ||g||_2 <= M, leave gradients unchanged.
    Otherwise, scale every gradient by M / (||g||_2 + eps).

    Args:
        parameters: Iterable of parameters whose .grad to clip.
        max_l2_norm: Maximum allowed L2 norm.
    """
    eps = 1e-6

    # Collect all gradients (skip parameters without grad)
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return

    # Compute global L2 norm: sqrt(sum of squared norms of each gradient)
    total_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))

    # Clip: scale down if total_norm > max_l2_norm
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)
