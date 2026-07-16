# src/training/losses.py

from __future__ import annotations
import torch.nn as nn
from src.core.config import LossConfig

def build_loss(config: LossConfig) -> nn.Module:
    """
    Build loss function from configuration.

    Parameters
    ----------
    config:
        Loss configuration.

    Returns
    -------
    torch.nn.Module
        Instantiated loss function.

    Raises
    ------
    ValueError
        If the loss name is unsupported.
    """

    name = config.name.lower().strip()

    if name == "cross_entropy":
        return nn.CrossEntropyLoss()

    raise ValueError(
        f"Unsupported loss '{config.name}'."
    )