from __future__ import annotations

import argparse
from pathlib import Path

from scripts._common import (
    build_evaluation_loaders,
    evaluate_model,
    load_experiment_bundle,
    save_evaluation_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained experiment.")
    parser.add_argument("--experiment-root", required=True, help="Path to an experiment directory.")
    parser.add_argument(
        "--checkpoint-name",
        default="best.pt",
        help="Checkpoint filename inside the experiment checkpoint directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_experiment_bundle(Path(args.experiment_root), args.checkpoint_name)

    dataloader, class_names = build_evaluation_loaders(bundle.config.experiment.name, bundle)
    evaluation = evaluate_model(bundle, bundle.model, dataloader)

    save_evaluation_result(evaluation, Path(args.experiment_root) / "evaluation", class_names)
    print(bundle.checkpoint_path)


if __name__ == "__main__":
    main()