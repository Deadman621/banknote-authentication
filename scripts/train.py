from __future__ import annotations

import argparse
from pathlib import Path

from scripts._common import (
    build_model_from_config,
    build_training_loaders,
    build_trainer,
    evaluate_model,
    prepare_experiment,
    save_evaluation_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a banknote model.")
    parser.add_argument("--module", required=True, help="Module name, e.g. denomination.")
    parser.add_argument("--model", required=True, help="Model name, e.g. cnn.")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = prepare_experiment(args.module, args.model, args.experiment_name)

    train_loader, val_loader, class_names = build_training_loaders(  # type: ignore[name-defined]
        args.module,
        bundle,
        train_split=args.train_split,
    )

    model = build_model_from_config(bundle.config)
    trainer, optimizer, scheduler = build_trainer(bundle, model, class_names)

    bundle.logger.info(
        "Starting training for module=%s model=%s on %s",
        args.module,
        args.model,
        bundle.device,
    )

    state = trainer.fit(train_loader=train_loader, validation_loader=val_loader)

    final_checkpoint = bundle.paths.checkpoints / "final.pt"
    from src.checkpoint.io import save_checkpoint

    save_checkpoint(
        path=final_checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=state.epoch,
        global_step=state.global_step,
    )

    evaluation = evaluate_model(bundle, model, val_loader)
    save_evaluation_result(evaluation, bundle.paths.evaluation, class_names)

    bundle.logger.info(
        "Training complete | best checkpoint=%s | final checkpoint=%s",
        bundle.paths.checkpoints / "best.pt",
        final_checkpoint,
    )


if __name__ == "__main__":
    main()