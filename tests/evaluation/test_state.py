"""Tests for evaluation state."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from src.evaluation.state import EvaluationResult


def test_evaluation_result_creation() -> None:
    """EvaluationResult stores all provided values."""

    predictions = torch.tensor([0, 1, 2])
    targets = torch.tensor([0, 2, 2])
    probabilities = torch.tensor(
        [
            [0.90, 0.05, 0.05],
            [0.10, 0.80, 0.10],
            [0.05, 0.10, 0.85],
        ]
    )

    result = EvaluationResult(
        loss=0.123,
        accuracy=0.90,
        precision=0.91,
        recall=0.89,
        f1=0.90,
        roc_auc=None,
        predictions=predictions,
        targets=targets,
        probabilities=probabilities,
    )

    assert result.loss == pytest.approx(0.123)
    assert result.accuracy == pytest.approx(0.90)
    assert result.precision == pytest.approx(0.91)
    assert result.recall == pytest.approx(0.89)
    assert result.f1 == pytest.approx(0.90)
    assert result.roc_auc is None

    assert torch.equal(result.predictions, predictions)
    assert torch.equal(result.targets, targets)
    assert torch.equal(result.probabilities, probabilities)


def test_evaluation_result_accepts_roc_auc() -> None:
    """EvaluationResult accepts a ROC-AUC score."""

    result = EvaluationResult(
        loss=0.1,
        accuracy=1.0,
        precision=1.0,
        recall=1.0,
        f1=1.0,
        roc_auc=0.998,
        predictions=torch.tensor([1]),
        targets=torch.tensor([1]),
        probabilities=torch.tensor([[0.01, 0.99]]),
    )

    assert result.roc_auc == pytest.approx(0.998)


def test_evaluation_result_is_frozen() -> None:
    """EvaluationResult is immutable."""

    result = EvaluationResult(
        loss=0.1,
        accuracy=1.0,
        precision=1.0,
        recall=1.0,
        f1=1.0,
        roc_auc=None,
        predictions=torch.tensor([0]),
        targets=torch.tensor([0]),
        probabilities=torch.tensor([[1.0]]),
    )

    with pytest.raises(FrozenInstanceError):
        result.loss = 0.5