from __future__ import annotations

from pathlib import Path

from PIL import Image
from PIL import UnidentifiedImageError

import pytest
import torch

from torch import Tensor
from torchvision import transforms

from src.inference.preprocessing import _load_image, _apply_transform, _build_transform, preprocess_image
from src.inference.state import PreprocessingConfig

def test_load_rgb_image(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "rgb.png"

    Image.new(
        mode="RGB",
        size=(32, 32),
        color=(255, 0, 0),
    ).save(image_path)

    image = _load_image(image_path)

    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert image.size == (32, 32)


def test_load_grayscale_image_converts_to_rgb(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "grayscale.png"

    Image.new(
        mode="L",
        size=(16, 16),
        color=128,
    ).save(image_path)

    image = _load_image(image_path)

    assert image.mode == "RGB"
    assert image.size == (16, 16)


def test_load_rgba_image_converts_to_rgb(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "rgba.png"

    Image.new(
        mode="RGBA",
        size=(24, 24),
        color=(255, 0, 0, 128),
    ).save(image_path)

    image = _load_image(image_path)

    assert image.mode == "RGB"
    assert image.size == (24, 24)


def test_load_image_missing_file() -> None:
    image_path = Path("does_not_exist.png")

    with pytest.raises(FileNotFoundError):
        _load_image(image_path)


def test_load_invalid_image_file(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "invalid.png"

    image_path.write_text(
        "this is not an image",
        encoding="utf-8",
    )

    with pytest.raises(UnidentifiedImageError):
        _load_image(image_path)

def test_build_transform_returns_compose() -> None:
    config = PreprocessingConfig(
        image_size=(224, 224),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    transform = _build_transform(config)

    assert isinstance(transform, transforms.Compose)


def test_build_transform_contains_expected_transforms() -> None:
    config = PreprocessingConfig(
        image_size=(224, 224),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    transform = _build_transform(config)

    assert len(transform.transforms) == 3

    assert isinstance(
        transform.transforms[0],
        transforms.Resize,
    )
    assert isinstance(
        transform.transforms[1],
        transforms.ToTensor,
    )
    assert isinstance(
        transform.transforms[2],
        transforms.Normalize,
    )


def test_resize_uses_config_image_size() -> None:
    config = PreprocessingConfig(
        image_size=(256, 192),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )

    transform = _build_transform(config)

    resize = transform.transforms[0]

    assert isinstance(resize, transforms.Resize)
    assert resize.size == config.image_size


def test_normalize_uses_config_statistics() -> None:
    config = PreprocessingConfig(
        image_size=(224, 224),
        mean=(0.1, 0.2, 0.3),
        std=(0.4, 0.5, 0.6),
    )

    transform = _build_transform(config)

    normalize = transform.transforms[2]

    assert isinstance(normalize, transforms.Normalize)
    assert normalize.mean == config.mean
    assert normalize.std == config.std


def test_transform_order_is_stable() -> None:
    config = PreprocessingConfig(
        image_size=(224, 224),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )

    transform = _build_transform(config)

    expected = (
        transforms.Resize,
        transforms.ToTensor,
        transforms.Normalize,
    )

    actual = tuple(type(t) for t in transform.transforms)

    assert actual == expected

def test_apply_transform_returns_tensor() -> None:
    config = PreprocessingConfig(
        image_size=(224, 224),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    image = Image.new(
        mode="RGB",
        size=(32, 32),
        color=(255, 0, 0),
    )

    transform = _build_transform(config)

    tensor = _apply_transform(
        image,
        transform,
    )

    assert isinstance(tensor, Tensor)

def test_apply_transform_returns_float32_tensor() -> None:
    config = PreprocessingConfig(
        image_size=(224, 224),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )

    image = Image.new(
        mode="RGB",
        size=(16, 16),
    )

    tensor = _apply_transform(
        image,
        _build_transform(config),
    )

    assert tensor.dtype == torch.float32

def test_apply_transform_resizes_image() -> None:
    config = PreprocessingConfig(
        image_size=(128, 96),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )

    image = Image.new(
        mode="RGB",
        size=(32, 32),
    )

    tensor = _apply_transform(
        image,
        _build_transform(config),
    )

    assert tensor.shape == (3, 128, 96)

def test_apply_transform_returns_finite_values() -> None:
    config = PreprocessingConfig(
        image_size=(64, 64),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )

    image = Image.new(
        mode="RGB",
        size=(32, 32),
    )

    tensor = _apply_transform(
        image,
        _build_transform(config),
    )

    assert torch.isfinite(tensor).all()

def test_preprocess_image_returns_batched_tensor(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.png"

    Image.new(
        mode="RGB",
        size=(32, 32),
    ).save(image_path)

    config = PreprocessingConfig(
        image_size=(224, 224),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )

    tensor = preprocess_image(
        image_path,
        config,
    )

    assert tensor.shape == (1, 3, 224, 224)

def test_preprocess_image_is_deterministic(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.png"

    Image.new(
        mode="RGB",
        size=(32, 32),
        color=(10, 20, 30),
    ).save(image_path)

    config = PreprocessingConfig(
        image_size=(224, 224),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    first = preprocess_image(
        image_path,
        config,
    )

    second = preprocess_image(
        image_path,
        config,
    )

    assert torch.equal(first, second)

def test_preprocess_image_missing_file() -> None:
    config = PreprocessingConfig(
        image_size=(224, 224),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )

    with pytest.raises(FileNotFoundError):
        preprocess_image(
            Path("missing.png"),
            config,
        )