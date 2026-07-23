from __future__ import annotations

import torch
import torch.nn as nn

from src.training.amp import AMPContext
from src.training.engine import TrainingEngine

from torch.utils.data import DataLoader, TensorDataset

def test_train_step_updates_model() -> None:
    model = nn.Linear(
        10,
        3,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )

    engine = TrainingEngine(
        model=model,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        amp=AMPContext(
            enabled=False,
            device=torch.device("cpu"),
        ),
    )

    images = torch.randn(
        8,
        10,
    )

    labels = torch.randint(
        0,
        3,
        (8,),
    )

    before = (
        model.weight.detach()
        .clone()
    )

    loss, correct, batch_size = engine.train_step(
        images,
        labels,
    )

    after = (
        model.weight.detach()
        .clone()
    )

    assert loss > 0
    assert 0 <= correct <= batch_size

    assert not torch.equal(
        before,
        after,
    )

def test_train_epoch() -> None:
    model = nn.Linear(
        10,
        3,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )

    engine = TrainingEngine(
        model=model,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        amp=AMPContext(
            enabled=False,
            device=torch.device("cpu"),
        ),
    )

    dataset = TensorDataset(
        torch.randn(20, 10),
        torch.randint(0, 3, (20,)),
    )

    loader = DataLoader(
        dataset,
        batch_size=5,
    )

    engine.train_epoch(loader)

    assert engine.state.global_step == 4

    assert engine.state.train_loss > 0

    assert (
        0.0
        <= engine.state.train_accuracy
        <= 1.0
    )

def test_validate_epoch() -> None:
    model = nn.Linear(
        10,
        3,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )

    engine = TrainingEngine(
        model=model,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        amp=AMPContext(
            enabled=False,
            device=torch.device("cpu"),
        ),
    )

    dataset = TensorDataset(
        torch.randn(20, 10),
        torch.randint(0, 3, (20,)),
    )

    loader = DataLoader(
        dataset,
        batch_size=5,
    )

    engine.validate_epoch(
        loader
    )

    assert engine.state.validation_loss > 0

    assert (
        0.0
        <= engine.state.validation_accuracy
        <= 1.0
    )

def test_fit() -> None:

    model = nn.Linear(
        10,
        3,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )

    engine = TrainingEngine(
        model=model,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        amp=AMPContext(
            enabled=False,
            device=torch.device("cpu"),
        ),
    )

    dataset = TensorDataset(
        torch.randn(20, 10),
        torch.randint(0, 3, (20,)),
    )

    loader = DataLoader(
        dataset,
        batch_size=5,
    )

    state = engine.fit(
        loader,
        loader,
        epochs=2,
    )

    assert state.epoch == 2
    assert state.global_step == 8