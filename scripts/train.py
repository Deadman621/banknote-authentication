from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from scripts._common import (
    build_model_from_config,
    build_training_loaders,
    build_trainer,
    evaluate_model,
    prepare_experiment,
    save_evaluation_result,
)

from src.checkpoint.io import load_checkpoint
from src.training.state import TrainState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a banknote model."
    )

    parser.add_argument(
        "--module",
        default=None,
        help="Module name for a new experiment.",
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Model name for a new experiment.",
    )

    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Override the experiment name from config.",
    )

    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of samples used for training.",
    )

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume an existing training run.",
    )

    group.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Initialize a new training run from an existing checkpoint.",
    )

    args = parser.parse_args()

    # Validation rules
    if args.resume is None:
        if args.module is None or args.model is None:
            parser.error(
                "--module and --model are required "
                "unless --resume is provided."
            )

    if args.resume is not None and args.experiment_name is not None:
        parser.error(
            "--experiment-name cannot be used with --resume."
        )

    return args

def main() -> None:
    args = parse_args()
    bundle = prepare_experiment(
        module=args.module,
        model=args.model,
        experiment_name=args.experiment_name,
        existing_run=args.resume,
    )

    train_loader, val_loader, class_names, class_weights = build_training_loaders(
        bundle.module,
        bundle,
    )

    model = build_model_from_config(bundle.config)
    trainer, optimizer, scheduler = build_trainer(bundle, model, class_names, class_weights)
    resume_state: Optional[TrainState] = None

    if args.resume is not None:
        checkpoint = load_checkpoint(
            path=bundle.paths.checkpoints / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            load_optimizer=True,
            load_scheduler=True,
        )

        trainer.restore_checkpoint(checkpoint)

        resume_state = TrainState(
            epoch=checkpoint.epoch,
            global_step=checkpoint.global_step,
        )

        bundle.logger.info(
            "Resumed training from %s (epoch=%d, step=%d)",
            args.resume,
            checkpoint.epoch,
            checkpoint.global_step,
        )

    elif args.checkpoint is not None:
        checkpoint = load_checkpoint(
            path=args.checkpoint,
            model=model,
            optimizer=None,
            scheduler=None,
            load_optimizer=False,
            load_scheduler=False,
        )

        bundle.logger.info(
            "Initialized model from checkpoint %s",
            args.checkpoint,
        )

    bundle.logger.info(
        "Starting training for module=%s model=%s on %s",
        bundle.module,
        bundle.model_name,
        bundle.device,
    )

    state = trainer.fit(train_loader=train_loader, validation_loader=val_loader, initial_state=resume_state)

    evaluation = evaluate_model(bundle, model, val_loader)
    save_evaluation_result(evaluation, bundle.paths.evaluation, class_names)

    bundle.logger.info(
        "Training complete | best checkpoint=%s | last checkpoint=%s",
        bundle.paths.checkpoints / "best.pt",
        bundle.paths.checkpoints / "last.pt",
    )


if __name__ == "__main__":
    main()