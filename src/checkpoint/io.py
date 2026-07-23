# src/checkpoint/io.py

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer

from src.utils.protocols import SchedulerProtocol
from src.training.state import TrainState

def save_checkpoint(path: Path, model: nn.Module, optimizer: Optimizer, epoch: int, global_step: int, scheduler: SchedulerProtocol | None = None, best_metric: float | None = None,) -> None:
    """
    Save training checkpoint.
    """

    checkpoint: dict[str, object] = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer: Optimizer | None = None, scheduler: SchedulerProtocol | None = None, *, load_optimizer: bool = True, load_scheduler: bool = True) -> TrainState:
    """
    Load training checkpoint.
    """

    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if load_optimizer and optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if (load_scheduler and scheduler is not None and "scheduler_state_dict" in checkpoint):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return TrainState(
        epoch=int(checkpoint["epoch"]),
        global_step=int(checkpoint["global_step"]),
        best_metric=float(checkpoint.get("best_metric") or 0.0),
    )