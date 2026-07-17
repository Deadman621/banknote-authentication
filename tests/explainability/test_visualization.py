from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.explainability.visualization import (
    apply_colormap,
    overlay_heatmap,
    validate_heatmap,
    validate_image,
)


@pytest.fixture
def image() -> NDArray[np.uint8]:
    return np.zeros(
        (64, 64, 3),
        dtype=np.uint8,
    )


@pytest.fixture
def heatmap() -> NDArray[np.float32]:
    return np.zeros(
        (64, 64),
        dtype=np.float32,
    )


def test_validate_image_valid(
    image: NDArray[np.uint8],
) -> None:
    validate_image(image)


@pytest.mark.parametrize(
    "invalid_image",
    [
        np.zeros(
            (64, 64),
            dtype=np.uint8,
        ),
        np.zeros(
            (64, 64, 1),
            dtype=np.uint8,
        ),
        np.zeros(
            (64, 64, 3),
            dtype=np.float32,
        ),
    ],
)
def test_validate_image_invalid(
    invalid_image: NDArray[np.generic],
) -> None:
    with pytest.raises(
        ValueError,
    ):
        validate_image(
            invalid_image,
        )


def test_validate_heatmap_valid(
    heatmap: NDArray[np.float32],
) -> None:
    validate_heatmap(
        heatmap,
        image_shape=(64, 64),
    )


@pytest.mark.parametrize(
    "invalid_heatmap",
    [
        np.zeros(
            (64, 64, 1),
            dtype=np.float32,
        ),
        np.zeros(
            (64, 32),
            dtype=np.float32,
        ),
        np.zeros(
            (64, 64),
            dtype=np.float64,
        ),
    ],
)
def test_validate_heatmap_invalid(
    invalid_heatmap: NDArray[np.generic],
) -> None:
    with pytest.raises(
        ValueError,
    ):
        validate_heatmap(
            invalid_heatmap,
            image_shape=(64, 64),
        )


def test_apply_colormap_shape(
    heatmap: NDArray[np.float32],
) -> None:
    result = apply_colormap(
        heatmap,
    )

    assert result.shape == (
        64,
        64,
        3,
    )


def test_apply_colormap_dtype(
    heatmap: NDArray[np.float32],
) -> None:
    result = apply_colormap(
        heatmap,
    )

    assert result.dtype == np.uint8


def test_overlay_heatmap_shape(
    image: NDArray[np.uint8],
    heatmap: NDArray[np.float32],
) -> None:
    result = overlay_heatmap(
        image,
        heatmap,
    )

    assert result.shape == (
        64,
        64,
        3,
    )


def test_overlay_heatmap_dtype(
    image: NDArray[np.uint8],
    heatmap: NDArray[np.float32],
) -> None:
    result = overlay_heatmap(
        image,
        heatmap,
    )

    assert result.dtype == np.uint8


@pytest.mark.parametrize(
    "alpha",
    [
        -0.1,
        1.1,
    ],
)
def test_overlay_invalid_alpha(
    image: NDArray[np.uint8],
    heatmap: NDArray[np.float32],
    alpha: float,
) -> None:
    with pytest.raises(
        ValueError,
    ):
        overlay_heatmap(
            image,
            heatmap,
            alpha=alpha,
        )


def test_overlay_heatmap_mismatched_size(
    image: NDArray[np.uint8],
) -> None:
    heatmap = np.zeros(
        (32, 32),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
    ):
        overlay_heatmap(
            image,
            heatmap,
        )


def test_overlay_heatmap_default_alpha(
    image: NDArray[np.uint8],
    heatmap: NDArray[np.float32],
) -> None:
    result = overlay_heatmap(
        image,
        heatmap,
    )

    assert isinstance(
        result,
        np.ndarray,
    )