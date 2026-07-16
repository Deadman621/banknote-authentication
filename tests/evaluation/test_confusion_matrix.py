"""Tests for confusion matrix utilities."""

from __future__ import annotations

import numpy as np
import torch

from src.evaluation.confusion_matrix import compute_confusion_matrix


def test_compute_confusion_matrix() -> None:
    """Confusion matrix is computed correctly."""

    predictions = torch.tensor([0, 1, 1, 0])
    targets = torch.tensor([0, 1, 0, 0])

    matrix = compute_confusion_matrix(
        predictions,
        targets,
    )

    expected = np.array(
        [
            [2, 1],
            [0, 1],
        ]
    )

    np.testing.assert_array_equal(matrix, expected)


def test_confusion_matrix_shape() -> None:
    """Confusion matrix has the expected shape."""

    predictions = torch.tensor([0, 1, 2, 0, 1, 2])
    targets = torch.tensor([0, 1, 2, 1, 2, 0])

    matrix = compute_confusion_matrix(
        predictions,
        targets,
    )

    assert matrix.shape == (3, 3)