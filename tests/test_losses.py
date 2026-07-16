from __future__ import annotations

import pytest
import torch.nn as nn

from src.core.config import LossConfig
from src.training.losses import build_loss


def test_cross_entropy_loss() -> None:
    config = LossConfig(
        name="cross_entropy"
    )

    loss = build_loss(config)

    assert isinstance(
        loss,
        nn.CrossEntropyLoss,
    )


def test_loss_name_is_case_insensitive() -> None:
    config = LossConfig(
        name="Cross_Entropy"
    )

    loss = build_loss(config)

    assert isinstance(
        loss,
        nn.CrossEntropyLoss,
    )


def test_invalid_loss() -> None:
    config = LossConfig(
        name="invalid"
    )

    with pytest.raises(ValueError):
        build_loss(config)