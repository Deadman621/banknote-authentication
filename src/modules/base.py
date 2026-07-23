from __future__ import annotations

from dataclasses import dataclass

from src.datasets.dataset import CurrencyDataset


@dataclass(frozen=True, slots=True)
class ModuleDefinition:
    name: str

    dataset: type[CurrencyDataset]

    preprocessing: object | None = None
    inference: object | None = None
    metrics: object | None = None