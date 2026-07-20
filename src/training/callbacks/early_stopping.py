# src/training/callbacks/early_stopping.py

from __future__ import annotations

from src.core.config import EarlyStoppingConfig
from src.training.callbacks.base import Callback
from src.training.state import TrainState


class EarlyStoppingCallback(Callback):
    """
    Stops training when monitored metric
    stops improving.
    """

    def __init__(self, config: EarlyStoppingConfig) -> None:
        self.config = config

        self.best_metric: float | None = None
        self.wait: int = 0

    def on_epoch_end(self, state: TrainState) -> None:
        metric = self._get_metric(state)

        if self._is_improvement(metric):
            self.best_metric = metric
            self.wait = 0
            return

        self.wait += 1

        if self.wait >= self.config.patience:
            state.should_stop = True

    def _get_metric(self, state: TrainState) -> float:
        value = getattr(state, self.config.monitor, None)

        if value is None:
            raise ValueError(
                f"Unknown early stopping metric "
                f"'{self.config.monitor}'."
            )

        return float(value)

    def _is_improvement(self, value: float) -> bool:
        if self.best_metric is None:
            return True

        # Losses normally decrease.
        if self.config.monitor.endswith("loss"):
            return value < self.best_metric

        # Accuracy-style metrics increase.
        return value > self.best_metric