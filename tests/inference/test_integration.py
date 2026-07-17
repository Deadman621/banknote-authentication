from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import Tensor
from torch import nn
from torch.optim import SGD

from src.checkpoint.io import save_checkpoint
from src.inference.loader import prepare_model
from src.inference.predictor import Predictor
from src.inference.preprocessing import preprocess_image
from src.inference.state import (
    BatchPredictionResult,
    PredictionResult,
    PreprocessingConfig,
)


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.linear = nn.Linear(
            in_features=3 * 224 * 224,
            out_features=3,
        )

        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.copy_(
                torch.tensor(
                    [1.0, 2.0, 5.0],
                    dtype=torch.float32,
                )
            )

    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        images = images.view(images.shape[0], -1)
        return self.linear(images)


def _create_checkpoint(
    path: Path,
) -> None:
    model = ToyModel()

    optimizer = SGD(
        model.parameters(),
        lr=0.1,
    )

    save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        global_step=10,
    )


def _config() -> PreprocessingConfig:
    return PreprocessingConfig(
        image_size=(224, 224),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )


def test_end_to_end_single_prediction(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.png"

    Image.new(
        mode="RGB",
        size=(128, 128),
        color=(255, 0, 0),
    ).save(image_path)

    checkpoint = tmp_path / "model.pt"
    _create_checkpoint(checkpoint)

    model = ToyModel()

    prepare_model(
        checkpoint_path=checkpoint,
        model=model,
        device=torch.device("cpu"),
    )

    predictor = Predictor(model)

    image = preprocess_image(
        image_path,
        _config(),
    )

    result = predictor.predict(image)

    assert isinstance(result, PredictionResult)
    assert result.predicted_class == 2
    assert result.confidence == pytest.approx(
        float(result.probabilities.max())
    )
    assert result.probabilities.sum() == pytest.approx(
        1.0,
        abs=1e-6,
    )
    assert result.probabilities.dtype == np.float32


def test_end_to_end_batch_prediction(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    _create_checkpoint(checkpoint)

    model = ToyModel()

    prepare_model(
        checkpoint_path=checkpoint,
        model=model,
        device=torch.device("cpu"),
    )

    predictor = Predictor(model)

    images = []

    for index in range(3):
        image_path = tmp_path / f"{index}.png"

        Image.new(
            mode="RGB",
            size=(64, 64),
            color=(index * 50, 0, 0),
        ).save(image_path)

        images.append(
            preprocess_image(
                image_path,
                _config(),
            )
        )

    batch = torch.cat(
        images,
        dim=0,
    )

    result = predictor.predict_batch(batch)

    assert isinstance(result, BatchPredictionResult)
    assert len(result.predictions) == 3

    for prediction in result.predictions:
        assert prediction.predicted_class == 2
        assert prediction.confidence == pytest.approx(
            float(prediction.probabilities.max())
        )
        assert prediction.probabilities.sum() == pytest.approx(
            1.0,
            abs=1e-6,
        )


def test_prepare_model_puts_model_in_eval_mode(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    _create_checkpoint(checkpoint)

    model = ToyModel()

    assert model.training

    prepare_model(
        checkpoint_path=checkpoint,
        model=model,
        device=torch.device("cpu"),
    )

    assert not model.training