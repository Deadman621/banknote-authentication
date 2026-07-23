"""
Shared base dataset for currency-recognition modules.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


class CurrencyDataset(Dataset[tuple[Tensor, int]], ABC):
    def __init__(self, root: str | Path, transform: Callable[[Image.Image], Tensor] | None = None) -> None:
        self.root = Path(root)
        self.transform = transform

        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset directory does not exist: {self.root}"
            )

        if not self.root.is_dir():
            raise NotADirectoryError(
                f"Dataset path is not a directory: {self.root}"
            )

        self.samples = self.build_samples()

        if not self.samples:
            raise ValueError(
                f"No valid samples were found in: {self.root}"
            )

    @abstractmethod
    def build_samples(self) -> list[tuple[Path, int]]:
        """
        Return image-path and integer-label pairs.

        Example:
            [
                (Path("image_1.png"), 0),
                (Path("image_2.png"), 1),
            ]
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the total number of dataset samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        """
        Load one image and return the trainer-compatible sample format.

        Returns:
            tuple[Tensor, int]: image tensor and integer class index.
        """
        image_path, label = self.samples[index]

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")

                if self.transform is None:
                    raise RuntimeError(
                        "A transform is required to convert the image "
                        "to a torch.Tensor."
                    )

                image_tensor = self.transform(image)

        except Exception as exc:
            raise RuntimeError(
                f"Failed to load image: {image_path}"
            ) from exc

        if not isinstance(image_tensor, Tensor):
            raise TypeError(
                "Dataset transform must return a torch.Tensor."
            )

        return image_tensor, int(label)

    @staticmethod
    def is_supported_image(path: Path) -> bool:
        """Return True when the path is a supported image file."""
        return (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )
    
    @property
    def class_names(self) -> tuple[str, ...]:
        """
        Return ordered class names.

        Child datasets should override this.
        """
        labels = sorted(
            {label for _, label in self.samples}
        )

        return tuple(str(label) for label in labels)