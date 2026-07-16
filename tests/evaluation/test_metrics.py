"""Tests for evaluation metrics."""

from __future__ import annotations

import pytest
import torch

from src.evaluation.metrics import (
    accuracy,
    f1,
    precision,
    recall,
    roc_auc,
)


def test_accuracy() -> None:
    """Accuracy is computed correctly."""

    predictions = torch.tensor([0, 1, 1, 0])
    targets = torch.tensor([0, 1, 0, 0])

    assert accuracy(predictions, targets) == pytest.approx(0.75)


def test_precision() -> None:
    predictions = torch.tensor([0, 1, 1, 0])
    targets = torch.tensor([0, 1, 0, 0])

    assert precision(predictions, targets) == pytest.approx(0.75)


def test_recall() -> None:
    predictions = torch.tensor([0, 1, 1, 0])
    targets = torch.tensor([0, 1, 0, 0])

    assert recall(predictions, targets) == pytest.approx(0.8333333333)


def test_f1() -> None:
    """Macro F1 is computed correctly."""

    predictions = torch.tensor([0, 1, 1, 0])
    targets = torch.tensor([0, 1, 0, 0])

    assert f1(predictions, targets) == pytest.approx(0.7333333333)


def test_binary_roc_auc() -> None:
    """Binary ROC-AUC is computed correctly."""

    probabilities = torch.tensor(
        [
            [0.90, 0.10],
            [0.20, 0.80],
            [0.80, 0.20],
            [0.10, 0.90],
        ]
    )

    targets = torch.tensor([0, 1, 0, 1])

    assert roc_auc(probabilities, targets) == pytest.approx(1.0)


def test_multiclass_roc_auc() -> None:
    """Multiclass ROC-AUC is computed correctly."""

    probabilities = torch.tensor(
        [
            [0.90, 0.05, 0.05],
            [0.05, 0.90, 0.05],
            [0.05, 0.05, 0.90],
        ]
    )

    targets = torch.tensor([0, 1, 2])

    assert roc_auc(probabilities, targets) == pytest.approx(1.0)