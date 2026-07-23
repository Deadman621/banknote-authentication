from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.application.experiment import ExperimentContext
from src.application.factories import ModuleFactory
from src.augmentation.transforms import build_eval_transforms
from src.datasets.dataloader import create_dataloader
from src.evaluation.evaluator import Evaluator
from src.inference.loader import prepare_model
from src.models.registry import ModelRegistry
from src.training.losses import build_loss


class EvaluationApplication:
    """
    Application layer responsible for model evaluation.
    """

    def build_dataloader(self, context: ExperimentContext) -> tuple[DataLoader, tuple[str, ...]]:
        Dataset = ModuleFactory.dataset(
            context.module
        )

        dataset = Dataset(
            root=context.config.dataset.root,
            transform=build_eval_transforms(
                context.config.dataset.image_size,
            ),
        )

        loader = create_dataloader(
            dataset=dataset,
            batch_size=context.config.dataset.batch_size,
            shuffle=False,
            num_workers=context.config.dataset.num_workers,
            pin_memory=context.config.dataset.pin_memory,
        )

        return loader, dataset.class_names


    def build_model(self, context: ExperimentContext):
        config = context.config

        return ModelRegistry.get_model(
            model_name=config.model.name,
            num_classes=config.dataset.num_classes,
            pretrained=False,
            **config.model.params,
        )


    def evaluate(self, context: ExperimentContext, checkpoint_name: str = "best.pt"):
        model = self.build_model(context)

        checkpoint = (
            context.paths.checkpoints
            / checkpoint_name
        )

        model = prepare_model(
            checkpoint_path=checkpoint,
            model=model,
            device=context.device,
        )

        loader, _ = self.build_dataloader(context)

        loss_fn = build_loss(
            context.config.loss,
            None,
        )

        evaluator = Evaluator(
            model=model,
            loss_fn=loss_fn,
            device=context.device,
        )

        return evaluator.evaluate(loader)


    def save_result(self, result, directory: Path) -> None:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            result,
            directory / "evaluation.pt",
        )