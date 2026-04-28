"""Data loading for language model training."""

import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of input sequences and next-token targets.

    Randomly samples `batch_size` starting indices from valid positions
    in the dataset, then slices contiguous windows of `context_length`
    for inputs and the corresponding shifted-by-one windows for targets.

    Args:
        dataset: 1D numpy array of integer token IDs.
        batch_size: Number of sequences per batch.
        context_length: Length of each sequence.
        device: PyTorch device string (e.g., 'cpu', 'cuda:0', 'mps').

    Returns:
        (x, y) where both are LongTensors of shape (batch_size, context_length).
        x contains input token IDs, y contains the next-token targets.
    """
    # Valid starting indices: 0 .. len(dataset) - context_length - 1
    max_start = len(dataset) - context_length
    starts = np.random.randint(0, max_start, size=batch_size)

    # Build input and target arrays
    x = np.stack([dataset[s : s + context_length] for s in starts])
    y = np.stack([dataset[s + 1 : s + 1 + context_length] for s in starts])

    return (
        torch.tensor(x, dtype=torch.long, device=device),
        torch.tensor(y, dtype=torch.long, device=device),
    )
