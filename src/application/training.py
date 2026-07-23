# src/application/training.py

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence, cast

import torch

from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Subset

from src.application.experiment import ExperimentContext
from src.application.factories import ModuleFactory
from src.augmentation.transforms import build_eval_transforms, build_train_transforms
from src.datasets.dataloader import create_dataloader
from src.datasets.dataset import CurrencyDataset
from src.models.registry import ModelRegistry

from src.training.callbacks.checkpoint import CheckpointCallback
from src.training.callbacks.early_stopping import EarlyStoppingCallback
from src.training.callbacks.logging import LoggingCallback
from src.training.callbacks.progress import ProgressCallback

from src.training.losses import build_loss
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.training.trainer import Trainer

from src.utils.protocols import SchedulerProtocol


@dataclass(slots=True)
class TrainingData:
    train_loader: DataLoader
    validation_loader: DataLoader
    class_names: tuple[str, ...]
    class_weights: torch.Tensor


@dataclass(slots=True)
class TrainingComponents:
    trainer: Trainer
    optimizer: Optimizer
    scheduler: SchedulerProtocol | None


class TrainingApplication:
    """
    Application layer responsible for constructing
    the training pipeline.

    It does not perform training itself.
    """

    def build_data(self, context: ExperimentContext, train_split: float = 0.8) -> TrainingData:
        train_dataset = self._create_dataset(context, train=True)
        validation_dataset = self._create_dataset(context, train=False)

        labels = [
            label
            for _, label in train_dataset.samples
        ]

        train_indices, validation_indices = self._split_indices(
            labels=labels,
            train_split=train_split,
            seed=context.config.seed,
        )

        train_loader = create_dataloader(
            dataset=Subset(train_dataset, train_indices),
            batch_size=context.config.dataset.batch_size,
            shuffle=True,
            num_workers=context.config.dataset.num_workers,
            pin_memory=context.config.dataset.pin_memory,
        )

        validation_loader = create_dataloader(
            dataset=Subset(validation_dataset, validation_indices),
            batch_size=context.config.dataset.batch_size,
            shuffle=False,
            num_workers=context.config.dataset.num_workers,
            pin_memory=context.config.dataset.pin_memory,
        )

        return TrainingData(
            train_loader=train_loader,
            validation_loader=validation_loader,
            class_names=train_dataset.class_names,
            class_weights=self._calculate_class_weights(
                labels=labels,
                num_classes=len(train_dataset.class_names),
            ),
        )

    def build_model(self, context: ExperimentContext) -> nn.Module:
        config = context.config

        return ModelRegistry.get_model(
            model_name=config.model.name,
            num_classes=config.dataset.num_classes,
            pretrained=config.model.pretrained,
            **config.model.params,
        )

    def build_components(
        self,
        context: ExperimentContext,
        model: nn.Module,
        class_weights: torch.Tensor | None = None,
    ) -> TrainingComponents:

        config = context.config

        if class_weights is not None:
            class_weights = class_weights.to(context.device)

        loss_fn = build_loss(
            config.loss,
            class_weights,
        )

        optimizer = build_optimizer(
            model,
            config.optimizer,
        )

        scheduler = build_scheduler(
            optimizer,
            config.scheduler,
        )

        callbacks = [
            ProgressCallback(),
            LoggingCallback(context.logger),
            EarlyStoppingCallback(
                config.trainer.early_stopping,
            ),
            CheckpointCallback(
                config=config.trainer.checkpoint,
                directory=context.paths.checkpoints,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            ),
        ]

        trainer = Trainer(
            config=config,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=context.device,
            callbacks=callbacks,
        )

        return TrainingComponents(
            trainer=trainer,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    def _create_dataset(self, context: ExperimentContext, train: bool) -> CurrencyDataset:
        transform = (
            build_train_transforms(
                context.config.dataset.image_size,
            )
            if train
            else build_eval_transforms(
                context.config.dataset.image_size,
            )
        )

        Dataset = cast(
            type[CurrencyDataset],
            ModuleFactory.dataset(
                context.module,
            ),
        )

        return Dataset(
            root=context.config.dataset.root,
            transform=transform,
        )

    def _calculate_class_weights(self, labels: Sequence[int], num_classes: int) -> torch.Tensor:
        counts = Counter(labels)
        total = len(labels)

        return torch.tensor(
            [
                total / (num_classes * counts[index])
                for index in range(num_classes)
            ],
            dtype=torch.float32,
        )

    def _split_indices(self, labels: Sequence[int], train_split: float, seed: int) -> tuple[list[int], list[int]]:
        if not 0 < train_split < 1:
            raise ValueError(
                "train_split must be between 0 and 1."
            )

        generator = torch.Generator()
        generator.manual_seed(seed)

        indices = torch.randperm(
            len(labels),
            generator=generator,
        ).tolist()

        split = int(
            len(indices) * train_split
        )

        return (
            indices[:split],
            indices[split:],
        )