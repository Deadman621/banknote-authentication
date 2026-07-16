from pathlib import Path

import pytest
import torch
from PIL import Image

from src.augmentation.transforms import (
    build_eval_transforms,
    build_train_transforms,
)
from src.datasets.dataloader import (
    create_dataloader,
    create_train_val_dataloaders,
)
from src.modules.authenticity.dataset import AuthenticityDataset
from src.modules.denomination.dataset import DenominationDataset
from src.modules.quality.dataset import QualityDataset


def create_test_image(path: Path) -> None:
    image = Image.new(
        mode="RGB",
        size=(300, 200),
        color=(120, 100, 80),
    )
    image.save(path)


@pytest.fixture
def denomination_directory(tmp_path: Path) -> Path:
    filenames = [
        "2 (1).png",
        "2 (2).png",
        "5 (1).png",
        "5 (2).png",
        "10 (1).png",
        "10 (2).png",
    ]

    for filename in filenames:
        create_test_image(tmp_path / filename)

    return tmp_path


def test_denomination_extraction() -> None:
    assert (
        DenominationDataset.extract_denomination("2 (1).png")
        == "2"
    )

    assert (
        DenominationDataset.extract_denomination("100 (25).jpg")
        == "100"
    )


def test_invalid_denomination_filename() -> None:
    with pytest.raises(ValueError):
        DenominationDataset.extract_denomination("unknown.png")


def test_dataset_getitem_returns_tensor_and_int(
    denomination_directory: Path,
) -> None:
    dataset = DenominationDataset(
        root=denomination_directory,
        transform=build_eval_transforms(image_size=224),
    )

    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert isinstance(label, int)
    assert image.shape == (3, 224, 224)


def test_class_mapping_is_stable(
    denomination_directory: Path,
) -> None:
    dataset = DenominationDataset(
        root=denomination_directory,
        transform=build_eval_transforms(),
    )

    assert dataset.classes == ["2", "5", "10"]
    assert dataset.class_to_index == {
        "2": 0,
        "5": 1,
        "10": 2,
    }


def test_train_transform_returns_tensor() -> None:
    image = Image.new("RGB", (300, 200))

    transformed = build_train_transforms(224)(image)

    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == (3, 224, 224)


def test_eval_transform_returns_tensor() -> None:
    image = Image.new("RGB", (300, 200))

    transformed = build_eval_transforms(224)(image)

    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == (3, 224, 224)


def test_dataloader_batch_shapes(
    denomination_directory: Path,
) -> None:
    dataset = DenominationDataset(
        root=denomination_directory,
        transform=build_eval_transforms(),
    )

    loader = create_dataloader(
        dataset=dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    images, labels = next(iter(loader))

    assert images.shape == (2, 3, 224, 224)
    assert labels.shape == (2,)
    assert labels.dtype == torch.int64


def test_train_validation_split(
    denomination_directory: Path,
) -> None:
    dataset = DenominationDataset(
        root=denomination_directory,
        transform=build_eval_transforms(),
    )

    train_loader, val_loader = create_train_val_dataloaders(
        dataset=dataset,
        batch_size=2,
        train_split=0.67,
        val_split=0.33,
        num_workers=0,
        seed=42,
    )

    assert len(train_loader.dataset) == 4
    assert len(val_loader.dataset) == 2


def test_authenticity_dataset_returns_tensor_and_int(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "note.png"
    create_test_image(image_path)

    dataset = AuthenticityDataset(
        root=tmp_path,
        samples=[("note.png", 1)],
        transform=build_eval_transforms(),
    )

    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert isinstance(label, int)
    assert label == 1


def test_quality_dataset_returns_tensor_and_int(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "note.png"
    create_test_image(image_path)

    dataset = QualityDataset(
        root=tmp_path,
        samples=[("note.png", 2)],
        transform=build_eval_transforms(),
    )

    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert isinstance(label, int)
    assert label == 2