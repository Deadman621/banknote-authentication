"""
Evaluation metrics.

Provides pure functions for computing classification metrics.
"""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    f1_score as sklearn_f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from torch import Tensor
from src.evaluation.state import EvaluationMetrics

def classification_metrics(predictions: Tensor, targets: Tensor, probabilities: Tensor) -> EvaluationMetrics:
    """Compute all classification metrics."""

    return EvaluationMetrics(
        accuracy=accuracy(predictions, targets),
        precision=precision(predictions, targets),
        recall=recall(predictions, targets),
        f1=f1(predictions, targets),
        roc_auc=roc_auc(probabilities, targets),
    )


def accuracy(predictions: Tensor, targets: Tensor) -> float:
    """
    Compute classification accuracy.

    Args:
        predictions:
            Predicted class indices.

        targets:
            Ground-truth class indices.

    Returns:
        Classification accuracy.
    """

    return float(
        accuracy_score(
            targets.cpu().numpy(),
            predictions.cpu().numpy(),
        )
    )


def precision(predictions: Tensor, targets: Tensor) -> float:
    """
    Compute macro-averaged precision.
    """

    return float(
        precision_score(
            targets.cpu().numpy(),
            predictions.cpu().numpy(),
            average="macro",
            zero_division=0,
        )
    )


def recall(predictions: Tensor, targets: Tensor) -> float:
    """
    Compute macro-averaged recall.
    """

    return float(
        recall_score(
            targets.cpu().numpy(),
            predictions.cpu().numpy(),
            average="macro",
            zero_division=0,
        )
    )


def f1(predictions: Tensor, targets: Tensor) -> float:
    """
    Compute macro-averaged F1 score.
    """

    return float(
        sklearn_f1_score(
            targets.cpu().numpy(),
            predictions.cpu().numpy(),
            average="macro",
            zero_division=0,
        )
    )


def roc_auc(probabilities: Tensor, targets: Tensor) -> float:
    """
    Compute ROC-AUC.

    Supports both binary and multiclass classification.

    Args:
        probabilities:
            Class probabilities.

        targets:
            Ground-truth class indices.

    Returns:
        ROC-AUC score.
    """

    y_true = targets.cpu().numpy()
    y_score = probabilities.cpu().numpy()

    if probabilities.shape[1] == 2:
        return float(roc_auc_score(y_true, y_score[:, 1]))

    return float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))
        