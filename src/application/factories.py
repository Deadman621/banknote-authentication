from __future__ import annotations

from importlib import import_module
from typing import Any

from src.datasets.dataset import CurrencyDataset


class ModuleFactory:
    """
    Resolves module-specific implementations.

    Dataset is mandatory.
    Other components are optional.
    """

    @staticmethod
    def dataset(module: str) -> type[CurrencyDataset]:
        return ModuleFactory._required(
            module,
            "dataset",
            "Dataset",
        )

    @staticmethod
    def inference(module: str) -> Any:
        return ModuleFactory._optional(
            module,
            "inference",
            "Predictor",
            "src.inference.predictor",
        )

    @staticmethod
    def metrics(module: str) -> Any:
        return ModuleFactory._optional(
            module,
            "metrics",
            "Metrics",
            "src.evaluation.metrics",
        )

    @staticmethod
    def preprocessing(module: str) -> Any:
        return ModuleFactory._optional(
            module,
            "preprocessing",
            "Preprocessor",
            "src.preprocessing.preprocess",
        )

    @staticmethod
    def _required(module: str, file: str, symbol: str) -> Any:
        implementation = import_module(
            f"src.modules.{module}.{file}"
        )

        try:
            return getattr(implementation, symbol)
        except AttributeError as exc:
            raise ImportError(
                f"Module '{module}' does not export '{symbol}'."
            ) from exc

    @staticmethod
    def _optional(module: str, file: str, symbol: str, fallback_module: str) -> Any:
        try:
            implementation = import_module(
                f"src.modules.{module}.{file}"
            )
        except ModuleNotFoundError:
            implementation = import_module(
                fallback_module
            )

        return getattr(
            implementation,
            symbol,
            None,
        )