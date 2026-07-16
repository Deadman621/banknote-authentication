from __future__ import annotations

import torch
import torch.nn as nn

from src.training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)


def create_components() -> tuple[
    nn.Module,
    torch.optim.Optimizer,
]:
    model = nn.Linear(
        10,
        2,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    return model, optimizer


def test_save_and_load_checkpoint(
    tmp_path,
) -> None:
    model, optimizer = create_components()

    checkpoint_path = (
        tmp_path / "checkpoint.pt"
    )

    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=5,
        global_step=100,
    )

    assert checkpoint_path.exists()

    new_model, new_optimizer = (
        create_components()
    )

    epoch, step = load_checkpoint(
        checkpoint_path,
        new_model,
        new_optimizer,
    )

    assert epoch == 5
    assert step == 100


def test_checkpoint_restores_weights(
    tmp_path,
) -> None:
    model, optimizer = create_components()

    path = tmp_path / "model.pt"

    original = (
        model.weight.detach() #type: ignore 
        .clone()
    )

    save_checkpoint(
        path,
        model,
        optimizer,
        epoch=1,
        global_step=1,
    )

    new_model, new_optimizer = (
        create_components()
    )

    load_checkpoint(
        path,
        new_model,
        new_optimizer,
    )

    restored = (
        new_model.weight.detach() #type: ignore
        .clone()
    )

    assert torch.equal(
        original,
        restored,
    )