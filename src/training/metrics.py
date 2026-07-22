# src/training/metrics.py

from __future__ import annotations

import torch
from torch import Tensor


def accuracy(logits: Tensor, labels: Tensor) -> float:
    """
    Compute classification accuracy.

    Parameters
    ----------
    logits:
        Model outputs before softmax.
        Shape: (batch_size, num_classes)

    labels:
        Ground truth class indices.
        Shape: (batch_size,)

    Returns
    -------
    float
        Accuracy value in range [0, 1].
    """

    predictions = torch.argmax(
        logits,
        dim=1,
    )

    correct = torch.eq(
        predictions,
        labels,
    ).sum()

    return float(
        correct.item() / labels.size(0)
    )

def accuracy_counts(logits: Tensor, labels: Tensor) -> tuple[int, int]:
    """
    Return (correct_predictions, total_samples).
    """

    predictions = torch.argmax(logits, dim=1)
    correct = torch.eq(predictions, labels).sum().item()

    return int(correct), labels.size(0)