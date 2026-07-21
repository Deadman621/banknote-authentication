"""Tests for evaluation state."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch

from src.evaluation.state import EvaluationMetrics, EvaluationResult


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
    metrics = EvaluationMetrics(
        accuracy=0.90,
        precision=0.91,
        recall=0.89,
        f1=0.90,
        roc_auc=None,
    )
    confusion_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, 0, 1]])

    result = EvaluationResult(
        loss=0.123,
        metrics=metrics,
        predictions=predictions,
        targets=targets,
        probabilities=probabilities,
        confusion_matrix=confusion_matrix,
    )

    assert result.loss == pytest.approx(0.123)
    assert result.metrics.accuracy == pytest.approx(0.90)
    assert result.metrics.precision == pytest.approx(0.91)
    assert result.metrics.recall == pytest.approx(0.89)
    assert result.metrics.f1 == pytest.approx(0.90)
    assert result.metrics.roc_auc is None

    assert torch.equal(result.predictions, predictions)
    assert torch.equal(result.targets, targets)
    assert torch.equal(result.probabilities, probabilities)


def test_evaluation_result_accepts_roc_auc() -> None:
    """EvaluationResult accepts a ROC-AUC score."""

    metrics = EvaluationMetrics(
        accuracy=1.0,
        precision=1.0,
        recall=1.0,
        f1=1.0,
        roc_auc=0.998,
    )
    result = EvaluationResult(
        loss=0.1,
        metrics=metrics,
        predictions=torch.tensor([1]),
        targets=torch.tensor([1]),
        probabilities=torch.tensor([[0.01, 0.99]]),
        confusion_matrix=np.array([[1]]),
    )

    assert result.metrics.roc_auc == pytest.approx(0.998)


def test_evaluation_result_is_frozen() -> None:
    """EvaluationResult is immutable."""

    metrics = EvaluationMetrics(
        accuracy=1.0,
        precision=1.0,
        recall=1.0,
        f1=1.0,
        roc_auc=None,
    )
    result = EvaluationResult(
        loss=0.1,
        metrics=metrics,
        predictions=torch.tensor([0]),
        targets=torch.tensor([0]),
        probabilities=torch.tensor([[1.0]]),
        confusion_matrix=np.array([[1]]),
    )

    with pytest.raises(FrozenInstanceError):
        result.loss = 0.5