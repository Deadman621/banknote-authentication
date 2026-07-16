"""
Evaluation state.

Contains immutable data structures representing the results of a model
evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """
    Stores the results of evaluating a model on a dataset.

    Attributes:
        loss:
            Average evaluation loss.

        accuracy:
            Classification accuracy.

        precision:
            Macro-averaged precision.

        recall:
            Macro-averaged recall.

        f1:
            Macro-averaged F1 score.

        roc_auc:
            ROC-AUC score for binary classification.
            None for tasks where ROC-AUC is not applicable.

        predictions:
            Predicted class indices.

        targets:
            Ground-truth class indices.

        probabilities:
            Class probabilities after softmax.
    """

    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    predictions: Tensor
    targets: Tensor
    probabilities: Tensor