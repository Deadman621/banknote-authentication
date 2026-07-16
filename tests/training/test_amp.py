from __future__ import annotations

import torch
import torch.nn as nn

from src.training.amp import AMPContext


def test_amp_disabled_on_cpu() -> None:
    amp = AMPContext(
        enabled=True,
        device=torch.device("cpu"),
    )

    assert amp.enabled is False


def test_amp_disabled_backward() -> None:
    model = nn.Linear(
        10,
        2,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    inputs = torch.randn(
        4,
        10,
    )

    output = model(inputs)

    loss = output.mean()

    amp = AMPContext(
        enabled=False,
        device=torch.device("cpu"),
    )

    amp.zero_grad(optimizer)

    amp.backward(
        loss,
        optimizer,
    )

    amp.step(
        optimizer,
    )

    assert True