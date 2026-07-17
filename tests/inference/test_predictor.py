from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor
from torch import nn

from src.inference.predictor import Predictor
from src.inference.state import (
    BatchPredictionResult,
    PredictionResult,
)


class ToyModel(nn.Module):
    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        batch_size = images.shape[0]

        logits = torch.tensor(
            [[1.0, 2.0, 5.0]],
            dtype=torch.float32,
        )

        return logits.repeat(batch_size, 1)


def test_predict_returns_prediction_result() -> None:
    predictor = Predictor(ToyModel())

    image = torch.randn(
        1,
        3,
        224,
        224,
    )

    result = predictor.predict(image)

    assert isinstance(result, PredictionResult)


def test_predict_returns_expected_class() -> None:
    predictor = Predictor(ToyModel())

    image = torch.randn(
        1,
        3,
        224,
        224,
    )

    result = predictor.predict(image)

    assert result.predicted_class == 2


def test_predict_confidence_matches_max_probability() -> None:
    predictor = Predictor(ToyModel())

    image = torch.randn(
        1,
        3,
        224,
        224,
    )

    result = predictor.predict(image)

    assert result.confidence == pytest.approx(
        float(result.probabilities.max())
    )


def test_predict_probabilities_sum_to_one() -> None:
    predictor = Predictor(ToyModel())

    image = torch.randn(
        1,
        3,
        224,
        224,
    )

    result = predictor.predict(image)

    assert result.probabilities.sum() == pytest.approx(
        1.0,
        abs=1e-6,
    )


def test_predict_probabilities_are_float32() -> None:
    predictor = Predictor(ToyModel())

    image = torch.randn(
        1,
        3,
        224,
        224,
    )

    result = predictor.predict(image)

    assert result.probabilities.dtype == np.float32


def test_predict_batch_returns_batch_prediction_result() -> None:
    predictor = Predictor(ToyModel())

    images = torch.randn(
        4,
        3,
        224,
        224,
    )

    result = predictor.predict_batch(images)

    assert isinstance(result, BatchPredictionResult)


def test_predict_batch_returns_expected_number_of_predictions() -> None:
    predictor = Predictor(ToyModel())

    images = torch.randn(
        5,
        3,
        224,
        224,
    )

    result = predictor.predict_batch(images)

    assert len(result.predictions) == 5


def test_predict_batch_predictions_are_correct() -> None:
    predictor = Predictor(ToyModel())

    images = torch.randn(
        3,
        3,
        224,
        224,
    )

    result = predictor.predict_batch(images)

    for prediction in result.predictions:
        assert prediction.predicted_class == 2


def test_predict_batch_confidence_matches_probability() -> None:
    predictor = Predictor(ToyModel())

    images = torch.randn(
        2,
        3,
        224,
        224,
    )

    result = predictor.predict_batch(images)

    for prediction in result.predictions:
        assert prediction.confidence == pytest.approx(
            float(prediction.probabilities.max())
        )


def test_predict_batch_probabilities_sum_to_one() -> None:
    predictor = Predictor(ToyModel())

    images = torch.randn(
        2,
        3,
        224,
        224,
    )

    result = predictor.predict_batch(images)

    for prediction in result.predictions:
        assert prediction.probabilities.sum() == pytest.approx(
            1.0,
            abs=1e-6,
        )


def test_predict_is_deterministic() -> None:
    predictor = Predictor(ToyModel())

    image = torch.randn(
        1,
        3,
        224,
        224,
    )

    first = predictor.predict(image)
    second = predictor.predict(image)

    assert first.predicted_class == second.predicted_class
    assert first.confidence == pytest.approx(second.confidence)
    assert np.array_equal(
        first.probabilities,
        second.probabilities,
    )


def test_predict_batch_is_deterministic() -> None:
    predictor = Predictor(ToyModel())

    images = torch.randn(
        4,
        3,
        224,
        224,
    )

    first = predictor.predict_batch(images)
    second = predictor.predict_batch(images)

    assert len(first.predictions) == len(second.predictions)

    for lhs, rhs in zip(
        first.predictions,
        second.predictions,
        strict=True,
    ):
        assert lhs.predicted_class == rhs.predicted_class
        assert lhs.confidence == pytest.approx(rhs.confidence)
        assert np.array_equal(
            lhs.probabilities,
            rhs.probabilities,
        )