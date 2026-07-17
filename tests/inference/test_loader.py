from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.optim import SGD

from src.checkpoint.io import save_checkpoint
from src.inference.loader import prepare_model


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.linear = nn.Linear(
            in_features=4,
            out_features=2,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.linear(x)
    
def test_prepare_model_sets_eval_mode(
    tmp_path: Path,
) -> None:
    model = ToyModel()

    optimizer = SGD(
        model.parameters(),
        lr=0.1,
    )

    checkpoint = tmp_path / "model.pt"

    save_checkpoint(
        path=checkpoint,
        model=model,
        optimizer=optimizer,
        epoch=1,
        global_step=5,
    )

    loaded = ToyModel()

    assert loaded.training

    prepare_model(
        checkpoint_path=checkpoint,
        model=loaded,
        device=torch.device("cpu"),
    )

    assert not loaded.training

def test_prepare_model_returns_same_instance(
    tmp_path: Path,
) -> None:
    model = ToyModel()

    optimizer = SGD(
        model.parameters(),
        lr=0.1,
    )

    checkpoint = tmp_path / "model.pt"

    save_checkpoint(
        path=checkpoint,
        model=model,
        optimizer=optimizer,
        epoch=0,
        global_step=0,
    )

    loaded = ToyModel()

    result = prepare_model(
        checkpoint_path=checkpoint,
        model=loaded,
        device=torch.device("cpu"),
    )

    assert result is loaded

def test_prepare_model_loads_weights(
    tmp_path: Path,
) -> None:
    model = ToyModel()

    with torch.no_grad():
        model.linear.weight.fill_(2.0)
        model.linear.bias.fill_(3.0)

    optimizer = SGD(
        model.parameters(),
        lr=0.1,
    )

    checkpoint = tmp_path / "model.pt"

    save_checkpoint(
        path=checkpoint,
        model=model,
        optimizer=optimizer,
        epoch=0,
        global_step=0,
    )

    loaded = ToyModel()

    prepare_model(
        checkpoint_path=checkpoint,
        model=loaded,
        device=torch.device("cpu"),
    )

    assert torch.equal(
        loaded.linear.weight,
        model.linear.weight,
    )

    assert torch.equal(
        loaded.linear.bias,
        model.linear.bias,
    )

def test_prepare_model_moves_model_to_device(
    tmp_path: Path,
) -> None:
    device = torch.device("cpu")

    model = ToyModel()

    optimizer = SGD(
        model.parameters(),
        lr=0.1,
    )

    checkpoint = tmp_path / "model.pt"

    save_checkpoint(
        path=checkpoint,
        model=model,
        optimizer=optimizer,
        epoch=0,
        global_step=0,
    )

    loaded = ToyModel()

    prepare_model(
        checkpoint_path=checkpoint,
        model=loaded,
        device=device,
    )

    for parameter in loaded.parameters():
        assert parameter.device == device
