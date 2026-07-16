# src/training/optimizer.py

from __future__ import annotations

from torch import nn
from torch.optim import (
    Adam,
    AdamW,
    SGD,
    Optimizer,
)

from src.core.config import OptimizerConfig


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> Optimizer:
    """
    Build optimizer from configuration.

    Parameters
    ----------
    model:
        Model whose parameters will be optimized.

    config:
        Optimizer configuration.

    Returns
    -------
    torch.optim.Optimizer

    Raises
    ------
    ValueError
        If optimizer name is unsupported.
    """

    name = config.name.lower().strip()

    parameters = model.parameters()

    if name == "adam":
        return Adam(
            parameters,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    if name == "adamw":
        return AdamW(
            parameters,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    if name == "sgd":
        return SGD(
            parameters,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    raise ValueError(
        f"Unsupported optimizer '{config.name}'."
    )