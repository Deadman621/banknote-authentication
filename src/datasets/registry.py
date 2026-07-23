from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image
from torch import Tensor

from src.datasets.dataset import CurrencyDataset
from src.modules.authenticity.dataset import AuthenticityDataset
from src.modules.denomination.dataset import DenominationDataset
from src.modules.quality.dataset import QualityDataset


class DatasetRegistry:
    """
    Registry for experiment datasets.
    """

    _datasets: dict[str, type[CurrencyDataset]] = {
        "denomination": DenominationDataset,
        "authenticity": AuthenticityDataset,
        "quality": QualityDataset,
    }


    @classmethod
    def get_dataset(cls, name: str, **kwargs: Any) -> CurrencyDataset:
        """
        Create dataset instance from registered name.
        """

        name = name.lower().strip()

        dataset_cls = cls._datasets.get(name)

        if dataset_cls is None:
            raise ValueError(
                f"Unknown dataset module '{name}'."
            )

        return dataset_cls(**kwargs)


    @classmethod
    def register(cls, name: str, dataset_cls: type[CurrencyDataset]) -> None:
        """
        Register a new dataset.
        """

        cls._datasets[name] = dataset_cls