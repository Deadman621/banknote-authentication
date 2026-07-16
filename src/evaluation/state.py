from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from torch import Tensor


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """
    Classification metrics computed during evaluation.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """
    Complete output of a model evaluation.
    """

    loss: float

    metrics: EvaluationMetrics

    predictions: Tensor
    targets: Tensor
    probabilities: Tensor

    confusion_matrix: NDArray[np.int_]