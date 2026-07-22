"""
Dataset implementation for currency-denomination recognition.
"""
from src.datasets.dataset import CurrencyDataset
import re
from collections.abc import Callable
from pathlib import Path

from PIL import Image
from torch import Tensor

class DenominationDataset(CurrencyDataset):
    """
    Dataset for denomination classification.

    Expected directory structure:

    root/
    ├── 2/
    ├── 5/
    ├── 10/
    ├── 20/
    ├── 50/
    ├── 100/
    ├── 200/
    ├── 500/
    └── 1000/
    """

    def __init__(self,root: str | Path,transform: Callable[[Image.Image], Tensor] | None = None) -> None:
        self.class_to_index: dict[str, int] = {}
        self.index_to_class: dict[int, str] = {}

        super().__init__(
            root=root,
            transform=transform,
        )

        self.num_classes = len(self.class_to_index)

    def build_samples(self) -> list[tuple[Path, int]]:
        """
        Discover images from denomination folders and create integer labels.
        """

        if not self.root.exists():
            return []

        class_dirs = sorted(
            [d for d in self.root.iterdir() if d.is_dir()],
            key=lambda d: int(d.name),
        )

        self.class_to_index = {
            class_dir.name: idx
            for idx, class_dir in enumerate(class_dirs)
        }

        self.index_to_class = {
            idx: class_name
            for class_name, idx in self.class_to_index.items()
        }

        samples: list[tuple[Path, int]] = []

        for class_dir in class_dirs:
            label = self.class_to_index[class_dir.name]

            for image_path in sorted(class_dir.iterdir()):
                if self.is_supported_image(image_path):
                    samples.append((image_path, label))

        return samples

    @property
    def classes(self) -> list[str]:
        """Return denomination names ordered by class index."""

        return [
            self.index_to_class[idx]
            for idx in range(len(self.index_to_class))
        ]

class DenominationDataset_OLD(CurrencyDataset):
    """
    Dataset for denomination classification.

    The denomination is extracted from the beginning of each filename.

    Examples:
        2 (1).png   -> denomination 2
        10 (4).jpg  -> denomination 10
    """

    def __init__(self, root: str | Path, transform: Callable[[Image.Image], Tensor] | None = None) -> None:
        self.class_to_index: dict[str, int] = {}
        self.index_to_class: dict[int, str] = {}

        super().__init__(
            root=root,
            transform=transform,
        )
        self.num_classes = len(self.class_to_index)

    @staticmethod
    def extract_denomination(filename: str) -> str:
        """Extract the denomination appearing at the start of a filename."""

        match = re.match(r"^\s*(\d+)", filename)

        if match is None:
            raise ValueError(
                f"Cannot extract denomination from filename: {filename}"
            )

        return match.group(1)

    def build_samples(self) -> list[tuple[Path, int]]:
        """
        Discover images and create integer class labels.
        """

        image_paths = sorted(
            path
            for path in self.root.iterdir()
            if self.is_supported_image(path)
        )

        if not image_paths:
            return []

        labeled_paths: list[tuple[Path, str]] = []

        for image_path in image_paths:
            denomination = self.extract_denomination(image_path.name)
            labeled_paths.append((image_path, denomination))

        classes = sorted(
            {label for _, label in labeled_paths},
            key=int,
        )

        self.class_to_index = {
            class_name: index
            for index, class_name in enumerate(classes)
        }

        self.index_to_class = {
            index: class_name
            for class_name, index in self.class_to_index.items()
        }

        return [
            (
                image_path,
                self.class_to_index[denomination],
            )
            for image_path, denomination in labeled_paths
        ]

    @property
    def classes(self) -> list[str]:
        """Return denomination names ordered by class index."""

        return [
            self.index_to_class[index]
            for index in sorted(self.index_to_class)
        ]
