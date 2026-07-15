"""
Generic dataset implementation.

Currently generates synthetic samples so the training pipeline can be
tested before the real datasets are available.

Later, replace the synthetic data generation in `__getitem__` with
actual image loading and label retrieval.
"""

from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset


class CurrencyDataset(Dataset[tuple[Tensor, int]]):
    """Generic dataset for currency image classification."""

    def __init__(
        self,
        root: Path | None = None,
        transform: Callable[[Tensor], Tensor] | None = None,
        image_size: int = 224,
        num_classes: int = 2,
        length: int = 512,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            root: Root directory of the dataset.
            transform: Optional transform applied to each image.
            image_size: Height and width of generated images.
            num_classes: Number of output classes.
            length: Number of synthetic samples.
        """
        self.root = root
        self.transform = transform
        self.image_size = image_size
        self.num_classes = num_classes
        self.length = length

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.length

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        """
        Return one sample.

        Args:
            index: Dataset index (currently unused).

        Returns:
            A tuple containing:
                - image tensor of shape (3, H, W)
                - integer class label
        """
        # Silence unused-variable warnings until real indexing is added.
        _ = index

        image = torch.rand(
            3,
            self.image_size,
            self.image_size,
            dtype=torch.float32,
        )

        label = int(
            torch.randint(
                low=0,
                high=self.num_classes,
                size=(),
            ).item()
        )

        if self.transform is not None:
            image = self.transform(image)

        return image, label