from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scripts._common import build_evaluation_loaders, evaluate_model, load_experiment_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple trained experiments.")
    parser.add_argument(
        "--experiment-roots",
        nargs="+",
        required=True,
        help="One or more experiment root directories.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="best.pt",
        help="Checkpoint filename inside each experiment checkpoint directory.",
    )
    parser.add_argument(
        "--output",
        default="comparison.csv",
        help="Path to the CSV summary output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []

    for root_name in args.experiment_roots:
        bundle = load_experiment_bundle(Path(root_name), args.checkpoint_name)
        dataloader, class_names = build_evaluation_loaders(bundle.config.experiment.name, bundle)
        evaluation = evaluate_model(bundle, bundle.model, dataloader)

        rows.append(
            {
                "experiment_root": root_name,
                "model": bundle.config.model.name,
                "module": bundle.config.experiment.name,
                "accuracy": evaluation.metrics.accuracy,
                "precision": evaluation.metrics.precision,
                "recall": evaluation.metrics.recall,
                "f1": evaluation.metrics.f1,
                "roc_auc": evaluation.metrics.roc_auc,
                "loss": evaluation.loss,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['experiment_root']}: accuracy={row['accuracy']:.4f} f1={row['f1']:.4f}"
        )


if __name__ == "__main__":
    main()