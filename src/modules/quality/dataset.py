"""
Dataset implementation for currency-image quality classification.
"""

from collections.abc import Callable
from pathlib import Path

from PIL import Image
from torch import Tensor

from src.datasets.dataset import CurrencyDataset


class QualityDataset(CurrencyDataset):
    """
    Dataset for quality classification.

    Expected directory structure:

    root/
    ├── class_1/
    ├── class_2/
    └── ...
    """

    def __init__(
        self,
        root: str | Path,
        transform: Callable[[Image.Image], Tensor] | None = None,
    ) -> None:
        self._class_to_index: dict[str, int] = {}

        super().__init__(root=root, transform=transform)

    def build_samples(self) -> list[tuple[Path, int]]:
        class_dirs = sorted(
            directory
            for directory in self.root.iterdir()
            if directory.is_dir()
        )

        self._class_to_index = {
            directory.name: index
            for index, directory in enumerate(class_dirs)
        }

        samples: list[tuple[Path, int]] = []

        for class_dir in class_dirs:
            label = self._class_to_index[class_dir.name]

            for image_path in sorted(class_dir.iterdir()):
                if self.is_supported_image(image_path):
                    samples.append((image_path, label))

        return samples

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(self._class_to_index.keys())


Dataset = QualityDataset