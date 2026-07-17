from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from src.inference.state import (
    BatchPredictionResult,
    PredictionResult,
)


def test_prediction_result_creation() -> None:
    probabilities = np.array(
        [0.1, 0.2, 0.7],
        dtype=np.float32,
    )

    result = PredictionResult(
        predicted_class=2,
        confidence=0.7,
        probabilities=probabilities,
    )

    assert result.predicted_class == 2
    assert result.confidence == pytest.approx(0.7)
    assert result.probabilities is probabilities


def test_prediction_result_is_frozen() -> None:
    result = PredictionResult(
        predicted_class=1,
        confidence=0.95,
        probabilities=np.array(
            [0.05, 0.95],
            dtype=np.float32,
        ),
    )

    with pytest.raises(FrozenInstanceError):
        result.predicted_class = 0  # type: ignore[misc]


def test_prediction_result_uses_slots() -> None:
    result = PredictionResult(
        predicted_class=0,
        confidence=1.0,
        probabilities=np.array(
            [1.0],
            dtype=np.float32,
        ),
    )

    assert not hasattr(result, "__dict__")


def test_batch_prediction_result_creation() -> None:
    prediction_1 = PredictionResult(
        predicted_class=0,
        confidence=0.8,
        probabilities=np.array(
            [0.8, 0.2],
            dtype=np.float32,
        ),
    )

    prediction_2 = PredictionResult(
        predicted_class=1,
        confidence=0.9,
        probabilities=np.array(
            [0.1, 0.9],
            dtype=np.float32,
        ),
    )

    batch = BatchPredictionResult(
        predictions=(prediction_1, prediction_2),
    )

    assert len(batch.predictions) == 2
    assert batch.predictions[0] == prediction_1
    assert batch.predictions[1] == prediction_2


def test_batch_prediction_result_is_frozen() -> None:
    batch = BatchPredictionResult(
        predictions=(),
    )

    with pytest.raises(FrozenInstanceError):
        batch.predictions = ()  # type: ignore[misc]


def test_batch_prediction_result_uses_slots() -> None:
    batch = BatchPredictionResult(
        predictions=(),
    )

    assert not hasattr(batch, "__dict__")


def test_prediction_probabilities_are_float32() -> None:
    probabilities = np.array(
        [0.2, 0.3, 0.5],
        dtype=np.float32,
    )

    result = PredictionResult(
        predicted_class=2,
        confidence=0.5,
        probabilities=probabilities,
    )

    assert result.probabilities.dtype == np.float32


def test_empty_batch_prediction_result() -> None:
    batch = BatchPredictionResult(
        predictions=(),
    )

    assert batch.predictions == ()