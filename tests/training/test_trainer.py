from __future__ import annotations

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from src.core.config import (
    ExperimentConfig,
)
from src.core.config import (
    AlignedOutputConfig,
    CannyConfig,
    CheckpointConfig,
    ContourConfig,
    EarlyStoppingConfig,
    ExperimentSettings,
    DatasetConfig,
    TrainerConfig,
    OptimizerConfig,
    SchedulerConfig,
    LossConfig,
    ModelConfig,
    LoggingConfig,
    OutputConfig,
    PreprocessingConfig,
)

from src.training.trainer import Trainer


def create_config() -> ExperimentConfig:
    return ExperimentConfig(
        seed=1,
        device="cpu",
        experiment=ExperimentSettings(
            name="test",
        ),
        dataset=DatasetConfig(
            root=None,  # replace if Path required
            image_size=224,
            batch_size=4,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            num_classes=3,
            class_names=(
                "a",
                "b",
                "c",
            ),
        ),
        trainer=TrainerConfig(
            epochs=1,
            mixed_precision=False,
            gradient_clip=None,
            early_stopping=EarlyStoppingConfig(
                patience=5,
                monitor="validation_loss",
            ),
            checkpoint=CheckpointConfig(
                monitor="validation_loss",
                mode="min",
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
            name="test",
            pretrained=False,
        ),
        logging=LoggingConfig(
            level="INFO",
        ),
        output=OutputConfig(
            save_dir=".",
        ),
        preprocessing=PreprocessingConfig(
            gaussian_kernel=(5, 5),
            gaussian_sigma=1.0,
            canny=CannyConfig(adaptive=True, lower=50, upper=150),
            contour=ContourConfig(min_area_ratio=0.1, approx_epsilon=0.02),
            aligned_output=AlignedOutputConfig(width=256, height=117),
            resize_max_dim=1024,
        ),
    )


def test_trainer_fit() -> None:

    model = nn.Linear(
        10,
        3,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
    )

    trainer = Trainer(
        config=create_config(),
        model=model,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    dataset = TensorDataset(
        torch.randn(8,10),
        torch.randint(0,3,(8,)),
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
    )

    state = trainer.fit(
        loader,
        loader,
    )

    assert state.epoch == 1