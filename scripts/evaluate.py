from __future__ import annotations

import argparse
from pathlib import Path

from src.application.evaluation import EvaluationApplication
from src.application.experiment import ExperimentApplication


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained experiment.",
    )

    parser.add_argument(
        "--experiment-root",
        type=Path,
        required=True,
        help="Path to an experiment directory.",
    )

    parser.add_argument(
        "--checkpoint-name",
        default="best.pt",
        help="Checkpoint filename.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    experiment_app = ExperimentApplication()

    context = experiment_app.prepare(
        existing_run=args.experiment_root,
    )

    evaluation_app = EvaluationApplication()

    result = evaluation_app.evaluate(
        context=context,
        checkpoint_name=args.checkpoint_name,
    )

    output_dir = context.paths.evaluation

    evaluation_app.save_result(
        result=result,
        directory=output_dir,
    )

    print(f"Evaluation complete: {output_dir}")


if __name__ == "__main__":
    main()