from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from src.explainability.gradcam import GradCAM
from src.explainability.state import GradCAMResult


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            3,
            8,
            kernel_size=3,
            padding=1,
        )

        self.relu = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(
            8,
            2,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)

        return self.fc(x)


@pytest.fixture
def model() -> ToyModel:
    return ToyModel()


@pytest.fixture
def gradcam(
    model: ToyModel,
) -> GradCAM:
    return GradCAM(
        model=model,
        target_layer=model.conv,
    )


def test_constructor(
    gradcam: GradCAM,
) -> None:
    assert gradcam is not None


def test_compute_weights_shape(
    gradcam: GradCAM,
) -> None:
    gradients = torch.randn(
        2,
        8,
        16,
        16,
    )

    weights = gradcam._compute_weights(
        gradients,
    )

    assert weights.shape == (
        2,
        8,
        1,
        1,
    )


def test_compute_weights_dtype(
    gradcam: GradCAM,
) -> None:
    gradients = torch.randn(
        1,
        8,
        8,
        8,
    )

    weights = gradcam._compute_weights(
        gradients,
    )

    assert weights.dtype == gradients.dtype


def test_compute_heatmap_shape(
    gradcam: GradCAM,
) -> None:
    activations = torch.randn(
        1,
        8,
        16,
        16,
    )

    weights = torch.randn(
        1,
        8,
        1,
        1,
    )

    heatmap = gradcam._compute_heatmap(
        activations,
        weights,
    )

    assert heatmap.shape == (
        1,
        1,
        16,
        16,
    )


def test_compute_heatmap_non_negative(
    gradcam: GradCAM,
) -> None:
    activations = torch.randn(
        1,
        8,
        16,
        16,
    )

    weights = torch.randn(
        1,
        8,
        1,
        1,
    )

    heatmap = gradcam._compute_heatmap(
        activations,
        weights,
    )

    assert torch.all(
        heatmap >= 0,
    )


def test_resize_heatmap(
    gradcam: GradCAM,
) -> None:
    heatmap = torch.randn(
        1,
        1,
        8,
        8,
    )

    resized = gradcam._resize_heatmap(
        heatmap,
        size=(32, 32),
    )

    assert resized.shape == (
        1,
        1,
        32,
        32,
    )


def test_normalize_heatmap_range(
    gradcam: GradCAM,
) -> None:
    heatmap = torch.randn(
        1,
        1,
        32,
        32,
    )

    normalized = gradcam._normalize_heatmap(
        heatmap,
    )

    assert torch.isclose(
        normalized.min(),
        torch.tensor(0.0),
    )

    assert torch.isclose(
        normalized.max(),
        torch.tensor(1.0),
    )


def test_normalize_heatmap_constant(
    gradcam: GradCAM,
) -> None:
    heatmap = torch.ones(
        1,
        1,
        16,
        16,
    )

    normalized = gradcam._normalize_heatmap(
        heatmap,
    )

    assert torch.all(
        normalized == 0,
    )


def test_generate_returns_result(
    gradcam: GradCAM,
) -> None:
    image = torch.randn(
        3,
        64,
        64,
    )

    result = gradcam.generate(
        image,
    )

    assert isinstance(
        result,
        GradCAMResult,
    )


def test_generate_heatmap_shape(
    gradcam: GradCAM,
) -> None:
    image = torch.randn(
        3,
        64,
        64,
    )

    result = gradcam.generate(
        image,
    )

    assert result.heatmap.shape == (
        64,
        64,
    )


def test_generate_heatmap_dtype(
    gradcam: GradCAM,
) -> None:
    image = torch.randn(
        3,
        64,
        64,
    )

    result = gradcam.generate(
        image,
    )

    assert result.heatmap.dtype == np.float32


def test_generate_heatmap_normalized(
    gradcam: GradCAM,
) -> None:
    image = torch.randn(
        3,
        64,
        64,
    )

    result = gradcam.generate(
        image,
    )

    assert result.heatmap.min() >= 0.0
    assert result.heatmap.max() <= 1.0


@pytest.mark.parametrize(
    "shape",
    [
        (64, 64),
        (1, 3, 64, 64),
        (3, 64),
    ],
)
def test_generate_invalid_shape(
    gradcam: GradCAM,
    shape: tuple[int, ...],
) -> None:
    image = torch.randn(shape)

    with pytest.raises(
        ValueError,
    ):
        gradcam.generate(
            image,
        )


def test_generate_invalid_target_class(
    gradcam: GradCAM,
) -> None:
    image = torch.randn(
        3,
        64,
        64,
    )

    with pytest.raises(
        ValueError,
    ):
        gradcam.generate(
            image,
            target_class=10,
        )


def test_generate_restores_training_mode(
    model: ToyModel,
) -> None:
    model.train()

    gradcam = GradCAM(
        model=model,
        target_layer=model.conv,
    )

    image = torch.randn(
        3,
        64,
        64,
    )

    gradcam.generate(
        image,
    )

    assert model.training


def test_generate_multiple_calls(
    gradcam: GradCAM,
) -> None:
    image = torch.randn(
        3,
        64,
        64,
    )

    first = gradcam.generate(
        image,
    )

    second = gradcam.generate(
        image,
    )

    assert first.heatmap.shape == second.heatmap.shape

def test_generate_explicit_target_class(
    gradcam: GradCAM,
) -> None:
    image = torch.randn(
        3,
        64,
        64,
    )

    result = gradcam.generate(
        image,
        target_class=0,
    )

    assert result.target_class == 0