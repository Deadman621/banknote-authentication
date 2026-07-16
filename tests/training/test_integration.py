from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from src.core.config import (
    CheckpointConfig,
    DatasetConfig,
    EarlyStoppingConfig,
    ExperimentConfig,
    ExperimentSettings,
    LoggingConfig,
    LossConfig,
    ModelConfig,
    OptimizerConfig,
    OutputConfig,
    SchedulerConfig,
    TrainerConfig,
)

from src.training.callbacks.checkpoint import (
    CheckpointCallback,
)

from src.training.callbacks.early_stopping import (
    EarlyStoppingCallback,
)

from src.training.callbacks.logging import (
    LoggingCallback,
)

from src.training.callbacks.progress import (
    ProgressCallback,
)

from src.training.trainer import Trainer


def create_config(
    tmp_path: Path,
) -> ExperimentConfig:

    return ExperimentConfig(
        seed=42,
        device="cpu",

        experiment=ExperimentSettings(
            name="integration_test",
        ),

        dataset=DatasetConfig(
            root=tmp_path,
            image_size=32,
            batch_size=4,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            num_classes=3,
            class_names=(
                "class_0",
                "class_1",
                "class_2",
            ),
        ),

        trainer=TrainerConfig(
            epochs=3,
            mixed_precision=False,
            gradient_clip=None,

            early_stopping=EarlyStoppingConfig(
                patience=5,
                monitor="validation_loss",
            ),

            checkpoint=CheckpointConfig(
                monitor="validation_loss",
                mode="min",
                save_best_only=True,
            ),
        ),

        optimizer=OptimizerConfig(
            name="adam",
            lr=0.001,
            weight_decay=0.0,
        ),

        scheduler=SchedulerConfig(
            name="none",
        ),

        loss=LossConfig(
            name="cross_entropy",
        ),

        model=ModelConfig(
            name="linear",
            pretrained=False,
        ),

        logging=LoggingConfig(
            level="INFO",
        ),

        output=OutputConfig(
            save_dir=tmp_path,
        ),
    )


def test_phase5_training_integration(
    tmp_path: Path,
) -> None:

    config = create_config(
        tmp_path
    )

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(
            32 * 32,
            3,
        ),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    checkpoint_path = (
        tmp_path
        / "best.pt"
    )

    logger = __import__(
        "logging"
    ).getLogger(
        "integration"
    )


    callbacks = [
        LoggingCallback(
            logger
        ),

        ProgressCallback(),

        CheckpointCallback(
            config=config.trainer.checkpoint,
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
        ),

        EarlyStoppingCallback(
            config=config.trainer.early_stopping
        ),
    ]


    trainer = Trainer(
        config=config,
        model=model,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        callbacks=callbacks,
    )


    dataset = TensorDataset(
        torch.randn(
            32,
            1,
            32,
            32,
        ),

        torch.randint(
            0,
            3,
            (32,),
        ),
    )


    loader = DataLoader(
        dataset,
        batch_size=4,
    )


    state = trainer.fit(
        loader,
        loader,
    )


    # Training completed

    assert state.epoch > 0


    # State updated

    assert state.global_step > 0


    assert (
        state.train_loss
        > 0
    )


    assert (
        state.validation_loss
        > 0
    )


    # Checkpoint callback worked

    assert checkpoint_path.exists()