# scripts/compare_experiments.py

from __future__ import annotations

import argparse
import csv

from pathlib import Path

from src.application.experiment import ExperimentApplication
from src.application.evaluation import EvaluationApplication


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare multiple trained experiments."
    )

    parser.add_argument(
        "--experiment-roots",
        nargs="+",
        required=True,
        type=Path,
        help="Experiment directories to compare.",
    )

    parser.add_argument(
        "--checkpoint-name",
        default="best.pt",
        help="Checkpoint filename.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("comparison.csv"),
        help="CSV output path.",
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    experiment_app = ExperimentApplication()
    evaluation_app = EvaluationApplication()

    rows: list[dict[str, object]] = []


    for experiment_root in args.experiment_roots:

        context = experiment_app.prepare(
            existing_run=experiment_root,
        )

        result = evaluation_app.evaluate(
            context=context,
            checkpoint_name=args.checkpoint_name,
        )

        rows.append(
            {
                "experiment_root": str(experiment_root),
                "module": context.module,
                "model": context.config.model.name,
                "accuracy": result.metrics.accuracy,
                "precision": result.metrics.precision,
                "recall": result.metrics.recall,
                "f1": result.metrics.f1,
                "roc_auc": result.metrics.roc_auc,
                "loss": result.loss,
            }
        )


    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with args.output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


    for row in rows:
        print(
            f"{row['experiment_root']}: "
            f"accuracy={row['accuracy']:.4f} "
            f"f1={row['f1']:.4f}"
        )


if __name__ == "__main__":
    main()