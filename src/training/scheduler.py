# src/training/scheduler.py

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
    LRScheduler,
)

from src.core.config import SchedulerConfig


def build_scheduler(optimizer: Optimizer, config: SchedulerConfig) -> LRScheduler | ReduceLROnPlateau | None:
    """
    Build learning rate scheduler from configuration.

    Parameters
    ----------
    optimizer:
        Optimizer instance.

    config:
        Scheduler configuration.

    Returns
    -------
    Scheduler instance or None.

    Raises
    ------
    ValueError
        If scheduler name is unsupported.
    """

    name = config.name.lower().strip()

    params = config.params

    if name == "none":
        return None

    if name == "step_lr":
        return StepLR(
            optimizer,
            **params,
        )

    if name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            **params,
        )

    if name == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            **params,
        )

    raise ValueError(
        f"Unsupported scheduler '{config.name}'."
    )