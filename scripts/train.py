from __future__ import annotations

import argparse
from pathlib import Path

from src.application.experiment import ExperimentApplication
from src.application.training import TrainingApplication


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a banknote model.",
    )

    parser.add_argument("--module", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--train-split", type=float, default=0.8)

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--resume",
        type=Path,
        help="Resume an existing experiment.",
    )

    group.add_argument(
        "--checkpoint",
        type=Path,
        help="Initialize a new experiment from an existing checkpoint.",
    )

    args = parser.parse_args()

    if args.resume is None:
        if args.module is None or args.model is None:
            parser.error(
                "--module and --model are required unless --resume is provided."
            )

    if args.resume is not None and args.experiment_name is not None:
        parser.error(
            "--experiment-name cannot be used with --resume."
        )

    return args


def main() -> None:
    args = parse_args()

    experiment_app = ExperimentApplication()
    training_app = TrainingApplication()

    # Create or resume experiment
    context = experiment_app.prepare(
        module=args.module,
        model=args.model,
        experiment_name=args.experiment_name,
        existing_run=args.resume,
    )

    data = training_app.build_data(
        context=context,
        train_split=args.train_split,
    )

    model = experiment_app.build_model(
        context=context,
    )

    components = training_app.build_components(
        context=context,
        model=model,
        class_weights=data.class_weights,
    )

    initial_state = None

    if args.resume is not None:
        initial_state = experiment_app.resume_training(
            context=context,
            trainer=components.trainer,
            optimizer=components.optimizer,
            scheduler=components.scheduler,
        )

    elif args.checkpoint is not None:
        model = experiment_app.load_checkpoint_model(
            context=context,
            checkpoint=args.checkpoint,
        )

        components = training_app.build_components(
            context=context,
            model=model,
            class_weights=data.class_weights,
        )

    components.trainer.fit(
        train_loader=data.train_loader,
        validation_loader=data.validation_loader,
        initial_state=initial_state,
    )


if __name__ == "__main__":
    main()