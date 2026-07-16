"""
Confusion matrix utilities.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix
from torch import Tensor

def compute_confusion_matrix(predictions: Tensor, targets: Tensor) -> NDArray[np.int_]:
    """
    Compute the confusion matrix.

    Args:
        predictions:
            Predicted class indices.

        targets:
            Ground-truth class indices.

    Returns:
        Confusion matrix.
    """

    return confusion_matrix(targets.cpu().numpy(), predictions.cpu().numpy())