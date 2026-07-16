# src/training/state.py

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TrainState:
    """
    Mutable state container for an active training run.

    This object is owned by the training engine and shared
    with callbacks.

    It intentionally contains only runtime state.
    Configuration belongs to ExperimentConfig.
    """

    epoch: int = 0
    global_step: int = 0

    train_loss: float = 0.0
    validation_loss: float = 0.0

    train_accuracy: float = 0.0
    validation_accuracy: float = 0.0

    best_metric: float = 0.0

    should_stop: bool = False