from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from scripts._common import generate_gradcam_overlay, load_experiment_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and save a Grad-CAM overlay.")
    parser.add_argument("--experiment-root", required=True, help="Path to an experiment directory.")
    parser.add_argument("--image", required=True, help="Image path to explain.")
    parser.add_argument(
        "--checkpoint-name",
        default="best.pt",
        help="Checkpoint filename inside the experiment checkpoint directory.",
    )
    parser.add_argument(
        "--target-layer",
        default=None,
        help="Optional dotted path to the target layer.",
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="Optional class index to explain.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Defaults to the experiment gradcam directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_experiment_bundle(Path(args.experiment_root), args.checkpoint_name)

    result, overlay = generate_gradcam_overlay(
        bundle.model,
        Path(args.image),
        bundle.config.dataset.image_size,
        target_layer=args.target_layer,
        target_class=args.target_class,
    )

    output_path = (
        Path(args.output)
        if args.output is not None
        else Path(args.experiment_root) / "gradcam" / f"{Path(args.image).stem}_gradcam.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(overlay, dtype=np.uint8)).save(output_path)

    print(output_path)
    print(f"predicted_class={result.predicted_class} target_class={result.target_class}")


if __name__ == "__main__":
    main()