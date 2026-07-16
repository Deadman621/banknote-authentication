"""End-to-end integration tests for the evaluation pipeline."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.evaluator import Evaluator
from src.evaluation.state import EvaluationResult


class DummyModel(nn.Module):
    """A deterministic model for integration testing."""

    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        return images


def create_dataloader() -> DataLoader:
    """Create a deterministic evaluation dataloader."""

    logits = torch.tensor(
        [
            [8.0, 1.0],
            [1.0, 8.0],
            [8.0, 1.0],
            [1.0, 8.0],
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


def test_evaluation_pipeline() -> None:
    """
    Verify the complete evaluation pipeline.
    """

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

    assert result.loss >= 0.0

    assert result.metrics.accuracy == 1.0
    assert result.metrics.precision == 1.0
    assert result.metrics.recall == 1.0
    assert result.metrics.f1 == 1.0
    assert result.metrics.roc_auc == 1.0

    assert result.predictions.shape == (4,)
    assert result.targets.shape == (4,)
    assert result.probabilities.shape == (4, 2)

    assert result.confusion_matrix.shape == (2, 2)

    assert result.confusion_matrix[0, 0] == 2
    assert result.confusion_matrix[1, 1] == 2