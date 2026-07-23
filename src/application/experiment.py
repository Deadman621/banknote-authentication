from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from src.core.config import ExperimentConfig
from src.core.experiment import Experiment
from src.checkpoint.io import load_checkpoint
from src.training.state import TrainState
from src.training.trainer import Trainer
from src.models.registry import ModelRegistry
from src.training.losses import build_loss
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.utils.io import load_yaml, save_json
from src.utils.logger import get_logger
from src.utils.seed import seed_everything


@dataclass(frozen=True, slots=True)
class ExperimentContext:
    """
    Runtime context for an experiment.

    This object is passed around application workflows.
    """

    module: str
    model_name: str
    raw_config: dict[str, Any]
    config: ExperimentConfig
    paths: Any
    device: torch.device
    logger: logging.Logger


class ExperimentApplication:
    """
    Application service responsible for experiment lifecycle.

    Responsibilities:
    - create experiment context
    - restore experiments
    - build models
    - build training components
    - resume checkpoints
    """

    def prepare(self, module: str | None = None, model: str | None = None, experiment_name: str | None = None, existing_run: Path | None = None) -> ExperimentContext:
        if existing_run is None and (module is None or model is None):
            raise ValueError("module and model are required when creating a new experiment.")

        experiment = Experiment(
            module=module,
            model=model,
            experiment_name=experiment_name,
            existing_run=existing_run,
        )

        device = self._resolve_device(experiment.config.device)

        logger = get_logger(
            name=(
                f"banknote."
                f"{experiment.module}."
                f"{experiment.model}"
            ),
            log_file=experiment.paths.log_file,
            level=getattr(
                logging,
                experiment.config.logging.level.upper(),
                logging.INFO,
            ),
        )

        seed_everything(experiment.config.seed)

        return ExperimentContext(
            module=experiment.module,
            model_name=experiment.model,
            raw_config=experiment.raw_config,
            config=experiment.config,
            paths=experiment.paths,
            device=device,
            logger=logger,
        )


    def build_model(self, context: ExperimentContext) -> nn.Module:

        return ModelRegistry.get_model(
            model_name=context.config.model.name,
            num_classes=context.config.dataset.num_classes,
            pretrained=context.config.model.pretrained,
            **context.config.model.params,
        )


    def build_training_components(
        self,
        context: ExperimentContext,
        model: nn.Module,
        class_weights: torch.Tensor | None = None,
    ) -> tuple[
        Trainer,
        torch.optim.Optimizer,
        torch.optim.lr_scheduler.LRScheduler | None,
    ]:

        if class_weights is not None:
            class_weights = class_weights.to(context.device)

        loss_fn = build_loss(context.config.loss, class_weights=class_weights,)
        optimizer = build_optimizer(model, context.config.optimizer)
        scheduler = build_scheduler(optimizer, context.config.scheduler)

        trainer = Trainer(
            config=context.config,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=context.device,
            callbacks=[],
        )

        return trainer, optimizer, scheduler


    def resume_training(
        self,
        context: ExperimentContext,
        trainer: Trainer,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
    ) -> TrainState:

        checkpoint_path = (
            context.paths.checkpoints /
            "last.pt"
        )

        checkpoint = load_checkpoint(
            path=checkpoint_path,
            model=trainer.engine.model,
            optimizer=optimizer,
            scheduler=scheduler,
            load_optimizer=True,
            load_scheduler=True,
        )

        trainer.restore_checkpoint(checkpoint)

        context.logger.info(
            "Resumed training from %s "
            "(epoch=%d, step=%d)",
            checkpoint_path,
            checkpoint.epoch,
            checkpoint.global_step,
        )

        return TrainState(
            epoch=checkpoint.epoch,
            global_step=checkpoint.global_step,
        )


    def load_checkpoint_model(self, context: ExperimentContext, checkpoint: Path) -> nn.Module:
        model = self.build_model(context)

        checkpoint_data = load_checkpoint(
            path=checkpoint,
            model=model,
            optimizer=None,
            scheduler=None,
            load_optimizer=False,
            load_scheduler=False,
        )

        return model


    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        if device_name.lower() == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        return torch.device(device_name)