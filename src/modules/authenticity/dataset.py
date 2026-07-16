"""
Dataset implementation for currency-authenticity classification.
"""

from collections.abc import Callable, Sequence
from pathlib import Path

from PIL import Image
from torch import Tensor

from src.datasets.dataset import CurrencyDataset


class AuthenticityDataset(CurrencyDataset):
    """
    Dataset for genuine-versus-counterfeit classification.

    Samples are supplied explicitly as image-path and integer-label pairs.
    """

    def __init__(
        self,
        root: str | Path,
        samples: Sequence[tuple[str | Path, int]],
        transform: Callable[[Image.Image], Tensor] | None = None,
    ) -> None:
        self._provided_samples = [
            (Path(image_path), int(label))
            for image_path, label in samples
        ]

        super().__init__(
            root=root,
            transform=transform,
        )

    def build_samples(self) -> list[tuple[Path, int]]:
        """
        Validate and return authenticity samples.
        """

        validated_samples: list[tuple[Path, int]] = []

        for image_path, label in self._provided_samples:
            full_path = (
                image_path
                if image_path.is_absolute()
                else self.root / image_path
            )

            if not self.is_supported_image(full_path):
                raise ValueError(
                    f"Invalid authenticity image: {full_path}"
                )

            validated_samples.append(
                (full_path, int(label))
            )

        return validated_samples