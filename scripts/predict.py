from __future__ import annotations

import argparse
from pathlib import Path

from scripts._common import load_experiment_bundle, predict_image, save_prediction_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on one or more images.")
    parser.add_argument("--experiment-root", required=True, help="Path to an experiment directory.")
    parser.add_argument("--images", nargs="+", required=True, help="Image paths to classify.")
    parser.add_argument(
        "--checkpoint-name",
        default="best.pt",
        help="Checkpoint filename inside the experiment checkpoint directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_experiment_bundle(Path(args.experiment_root), args.checkpoint_name)

    results: list[dict[str, object]] = []

    for image_name in args.images:
        result, _ = predict_image(
            bundle.model,
            Path(image_name),
            bundle.config.dataset.image_size,
        )

        predicted_name = (
            bundle.class_names[result.predicted_class]
            if result.predicted_class < len(bundle.class_names)
            else str(result.predicted_class)
        )

        print(f"{image_name}: {predicted_name} ({result.confidence:.4f})")
        results.append(
            {
                "image": image_name,
                "predicted_class": result.predicted_class,
                "predicted_label": predicted_name,
                "confidence": result.confidence,
                "probabilities": result.probabilities.tolist(),
            }
        )

    save_prediction_results(Path(args.experiment_root) / "predictions" / "predictions.json", results)


if __name__ == "__main__":
    main()