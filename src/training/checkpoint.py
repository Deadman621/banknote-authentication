# src/training/checkpoint.py

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch
from torch import nn
from torch.optim import Optimizer


class SchedulerProtocol(Protocol):
    """
    Minimal scheduler interface required for checkpoints.
    """

    def state_dict(self) -> dict[str, object]:
        ...


    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        ...


def save_checkpoint(path: Path, model: nn.Module, optimizer: Optimizer, epoch: int, global_step: int, scheduler: SchedulerProtocol | None = None) -> None:
    """
    Save training checkpoint.
    """

    checkpoint: dict[str, object] = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = (
            scheduler.state_dict()
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        checkpoint,
        path,
    )


def load_checkpoint(path: Path, model: nn.Module, optimizer: Optimizer | None = None, scheduler: SchedulerProtocol | None = None) -> tuple[int, int]:
    """
    Load training checkpoint.
    """

    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"],
        )

    if (
        scheduler is not None
        and "scheduler_state_dict" in checkpoint
    ):
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"],
        )

    return (
        int(checkpoint["epoch"]),
        int(checkpoint["global_step"]),
    )