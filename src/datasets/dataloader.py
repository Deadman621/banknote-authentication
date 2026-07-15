from math import floor
from torch.utils.data import DataLoader, random_split

from src.datasets.dataset import CurrencyDataset


def create_dataloaders(
    batch_size: int = 32,
    image_size: int = 224,
    num_classes: int = 2,
    dataset_size: int = 640,
    train_split: float = 0.8,
    val_split: float = 0.2,
    num_workers: int = 0,
):
    if train_split + val_split != 1.0:
        raise ValueError("train_split + val_split must equal 1.0")

    dataset = CurrencyDataset(
        image_size=image_size,
        num_classes=num_classes,
        length=dataset_size,
    )

    train_length = floor(dataset_size * train_split)
    val_length = dataset_size - train_length

    train_dataset, val_dataset = random_split(
        dataset,
        [train_length, val_length],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader