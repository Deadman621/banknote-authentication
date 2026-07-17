"""
Generic DataLoader factories for currency-recognition datasets.
"""

from collections.abc import Sized
from math import isclose
from typing import Any, cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split


DatasetType = Dataset[tuple[Tensor, Any]]
LoaderType = DataLoader[tuple[Tensor, Any]]


def create_dataloader(
    dataset: DatasetType,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    drop_last: bool = False,
) -> LoaderType:
    """
    Create a configured PyTorch DataLoader.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


def create_train_val_dataloaders(
    dataset: DatasetType,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[LoaderType, LoaderType]:
    """
    Split one dataset reproducibly and create train/validation loaders.
    """

    if not isclose(
        train_split + val_split,
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "train_split + val_split must equal 1.0."
        )

    if train_split <= 0 or val_split <= 0:
        raise ValueError(
            "train_split and val_split must both be greater than 0."
        )

    dataset_size = len(cast(Sized, dataset))

    if dataset_size < 2:
        raise ValueError(
            "At least two samples are required for a train/validation split."
        )

    train_length = int(dataset_size * train_split)
    val_length = dataset_size - train_length

    if train_length == 0 or val_length == 0:
        raise ValueError(
            "The requested split produced an empty dataset."
        )

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_length, val_length],
        generator=generator,
    )

    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )

    val_loader = create_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, val_loader


def create_dataloaders(
    dataset: DatasetType | None = None,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[LoaderType, LoaderType]:
    """
    Backward-compatible alias for train/validation DataLoader creation.

    The project historically imported ``create_dataloaders`` from this module.
    """

    if dataset is None:
        dataset = cast(
            DatasetType,
            TensorDataset(
                torch.rand(8, 3, 224, 224),
                torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
            ),
        )

    resolved_dataset = cast(DatasetType, dataset)

    return create_train_val_dataloaders(
        dataset=resolved_dataset,
        batch_size=batch_size,
        train_split=train_split,
        val_split=val_split,
        num_workers=num_workers,
        seed=seed,
    )