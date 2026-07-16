"""Tests for evaluator."""

from __future__ import annotations

import torch
import pytest

from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.evaluator import Evaluator
from src.evaluation.state import EvaluationResult

class DummyModel(nn.Module):
    """Simple deterministic classifier."""

    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        return images


def create_dataloader() -> DataLoader:
    """Create a small evaluation dataloader."""

    logits = torch.tensor(
        [
            [5.0, 0.0],
            [0.0, 5.0],
            [5.0, 0.0],
            [0.0, 5.0],
        ]
    )

    labels = torch.tensor(
        [
            0,
            1,
            0,
            1,
        ]
    )

    dataset = TensorDataset(
        logits,
        labels,
    )

    return DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )


def test_evaluator_creation() -> None:
    """Evaluator stores dependencies."""

    evaluator = Evaluator(
        model=DummyModel(),
        loss_fn=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
    )

    assert evaluator.model is not None
    assert evaluator.loss_fn is not None
    assert evaluator.device.type == "cpu"


def test_evaluate_returns_result() -> None:
    """Evaluation returns an EvaluationResult."""

    evaluator = Evaluator(
        model=DummyModel(),
        loss_fn=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
    )

    result = evaluator.evaluate(
        create_dataloader(),
    )

    assert isinstance(
        result,
        EvaluationResult,
    )


def test_perfect_predictions() -> None:
    """Perfect predictions produce perfect metrics."""

    evaluator = Evaluator(
        model=DummyModel(),
        loss_fn=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
    )

    result = evaluator.evaluate(
        create_dataloader(),
    )

    assert result.metrics.accuracy == 1.0
    assert result.metrics.precision == 1.0
    assert result.metrics.recall == 1.0
    assert result.metrics.f1 == 1.0
    assert result.metrics.roc_auc == 1.0

    assert result.predictions.shape == (4,)
    assert result.targets.shape == (4,)
    assert result.probabilities.shape == (4, 2)
    assert result.confusion_matrix.shape == (2, 2)


def test_loss_is_non_negative() -> None:
    """Loss is always non-negative."""

    evaluator = Evaluator(
        model=DummyModel(),
        loss_fn=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
    )

    result = evaluator.evaluate(
        create_dataloader(),
    )

    assert result.loss >= 0.0

def test_empty_dataloader_raises() -> None:
    """Empty dataloaders are rejected."""

    dataset = TensorDataset(
        torch.empty((0, 2)),
        torch.empty((0,), dtype=torch.long),
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
    )

    evaluator = Evaluator(
        model=DummyModel(),
        loss_fn=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
    )

    with pytest.raises(ValueError):
        evaluator.evaluate(loader)