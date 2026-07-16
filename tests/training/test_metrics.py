from __future__ import annotations

import torch

from src.training.metrics import accuracy


def test_perfect_accuracy() -> None:
    logits = torch.tensor(
        [
            [3.0, 1.0],
            [0.5, 2.0],
        ]
    )

    labels = torch.tensor(
        [
            0,
            1,
        ]
    )

    result = accuracy(
        logits,
        labels,
    )

    assert result == 1.0


def test_zero_accuracy() -> None:
    logits = torch.tensor(
        [
            [0.5, 2.0],
            [3.0, 1.0],
        ]
    )

    labels = torch.tensor(
        [
            0,
            1,
        ]
    )

    result = accuracy(
        logits,
        labels,
    )

    assert result == 0.0


def test_partial_accuracy() -> None:
    logits = torch.tensor(
        [
            [3.0, 1.0],
            [0.5, 2.0],
            [4.0, 1.0],
            [1.0, 5.0],
        ]
    )

    labels = torch.tensor(
        [
            0,
            0,
            0,
            1,
        ]
    )

    result = accuracy(
        logits,
        labels,
    )

    assert result == 0.75