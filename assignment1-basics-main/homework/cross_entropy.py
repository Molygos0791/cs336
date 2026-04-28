"""Cross-entropy loss with numerical stability."""

import torch
from einops import reduce
from jaxtyping import Float, Int
from torch import Tensor


def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"],
    targets: Int[Tensor, " batch_size"],
) -> Float[Tensor, ""]:
    """
    Compute the average cross-entropy loss across examples.

    Uses numerical stability tricks:
    - Subtract the max logit to prevent overflow in exp.
    - Cancel log and exp: log(softmax(x)[t]) = x[t] - log(sum(exp(x))).

    Args:
        inputs: Unnormalized logits of shape (batch_size, vocab_size).
        targets: Target class indices of shape (batch_size,).

    Returns:
        Scalar tensor with the average cross-entropy loss.
    """
    # Subtract max for numerical stability
    x_max = reduce(inputs, "b v -> b 1", "max")
    logits_stable = inputs - x_max

    # log-sum-exp per example
    log_sum_exp = torch.log(reduce(torch.exp(logits_stable), "b v -> b", "sum"))

    # Gather target logit per example
    target_logits = logits_stable[torch.arange(inputs.shape[0], device=inputs.device), targets]

    # CE = -target_logit + log_sum_exp, averaged over batch
    losses = -target_logits + log_sum_exp

    return reduce(losses, "b -> ", "mean")
