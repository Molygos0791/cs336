"""AdamW optimizer with decoupled weight decay."""

import math
from collections.abc import Callable
from typing import Optional

import torch


class AdamW(torch.optim.Optimizer):
    """AdamW optimizer following Algorithm 1 in the assignment.

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate (alpha).
        betas: Coefficients for computing running averages of gradient and its square.
        eps: Term added to denominator for numerical stability.
        weight_decay: Decoupled weight decay coefficient (lambda).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                state["t"] += 1
                t = state["t"]
                m, v = state["m"], state["v"]

                # Update moment estimates
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias-corrected learning rate
                alpha_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                # Parameter update
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-alpha_t)

                # Decoupled weight decay
                p.data.add_(p.data, alpha=-lr * weight_decay)

        return loss
