from __future__ import annotations

import csv
import json
import logging
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from src.augmentation.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_eval_transforms,
    build_train_transforms,
)
from src.checkpoint.io import load_checkpoint, save_checkpoint
from src.core.config import ExperimentConfig
from src.core.experiment import (
    create_experiment_paths,
    load_config_dict,
    load_preprocessing_configs,
    parse_config,
)
from src.datasets.dataloader import create_dataloader
from src.datasets.dataset import CurrencyDataset
from src.evaluation.evaluator import Evaluator
from src.evaluation.serialization import serialize as serialize_eval_result
from src.evaluation.state import EvaluationResult
from src.explainability.gradcam import GradCAM
from src.explainability.visualization import overlay_heatmap
from src.inference.loader import prepare_model
from src.inference.predictor import Predictor
from src.inference.preprocessing import preprocess_image as preprocess_inference_image
from src.inference.state import PreprocessingConfig as InferencePreprocessingConfig
from src.models.registry import ModelRegistry
from src.modules.authenticity.dataset import AuthenticityDataset
from src.modules.denomination.dataset import DenominationDataset
from src.modules.quality.dataset import QualityDataset
from src.preprocessing.preprocess import Preprocessor
from src.training.callbacks.checkpoint import CheckpointCallback
from src.training.callbacks.early_stopping import EarlyStoppingCallback
from src.training.callbacks.logging import LoggingCallback
from src.training.callbacks.progress import ProgressCallback
from src.training.losses import build_loss
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.training.trainer import Trainer
from src.utils.io import load_yaml, save_json, save_yaml
from src.utils.logger import get_logger
from src.utils.seed import seed_everything

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


@dataclass(frozen=True, slots=True)
class ExperimentBundle:
    module: str
    model_name: str
    raw_config: dict[str, Any]
    config: ExperimentConfig
    paths: Any
    device: torch.device
    logger: logging.Logger


@dataclass(frozen=True, slots=True)
class LoadedExperiment:
    root: Path
    module: str
    model_name: str
    raw_config: dict[str, Any]
    config: ExperimentConfig
    model: nn.Module
    checkpoint_path: Path
    device: torch.device
    class_names: tuple[str, ...]


class ImageSampleDataset(CurrencyDataset):
    """
    Generic fallback dataset implementing CurrencyDataset.
    """

    def __init__(
        self,
        root: str | Path,
        samples: Sequence[tuple[str | Path, int]],
        transform: Callable[[Image.Image], Tensor] | None = None,
    ) -> None:
        self._provided_samples = [
            (Path(image_path), int(label)) for image_path, label in samples
        ]
        super().__init__(root=root, transform=transform)

    def build_samples(self) -> list[tuple[Path, int]]:
        validated_samples: list[tuple[Path, int]] = []
        for image_path, label in self._provided_samples:
            full_path = (
                image_path
                if image_path.is_absolute()
                else self.root / image_path
            )
            validated_samples.append((full_path, int(label)))
        return validated_samples


def resolve_path(path: str | Path, base: Path = WORKSPACE_ROOT) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base / candidate


def resolve_device(device_name: str) -> torch.device:
    if device_name.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device_name)


def load_raw_config(module: str, model: str) -> dict[str, Any]:
    raw_config = load_config_dict(module, model)

    if not isinstance(raw_config, dict):
        raise TypeError("Loaded configuration must be a mapping.")

    return deepcopy(raw_config)


def prepare_experiment(
    module: str, model: str, experiment_name: str | None = None
) -> ExperimentBundle:
    raw_config = load_raw_config(module, model)

    if experiment_name is not None:
        raw_config.setdefault("experiment", {})["name"] = experiment_name

    config = parse_config(raw_config)

    paths = create_experiment_paths(
        save_dir=config.output.save_dir,
        module=module,
        model=model,
        experiment_name=config.experiment.name,
    )

    save_yaml(raw_config, paths.config_file)

    device = resolve_device(config.device)
    logger = get_logger(
        name=f"banknote.{module}.{model}",
        log_file=paths.log_file,
        level=getattr(logging, config.logging.level.upper(), logging.INFO),
    )

    seed_everything(config.seed)

    return ExperimentBundle(
        module=module,
        model_name=model,
        raw_config=raw_config,
        config=config,
        paths=paths,
        device=device,
        logger=logger,
    )


def _require_mapping(value: object, key: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected '{key}' to be a mapping.")

    return value


def _get_dataset_config(raw_config: Mapping[str, Any]) -> Mapping[str, Any]:
    return _require_mapping(raw_config.get("dataset"), "dataset")


def _collect_denomination_samples(
    dataset_config: Mapping[str, Any]
) -> list[tuple[Path, str]]:
    root = resolve_path(dataset_config["root"])

    if not root.exists():
        raise FileNotFoundError(root)

    samples: list[tuple[Path, str]] = []

    for image_path in sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ):
        filename = image_path.name
        label = filename.lstrip().split(" ", maxsplit=1)[0]
        if not label:
            raise ValueError(f"Could not infer denomination from '{filename}'.")
        samples.append((image_path, label))

    if not samples:
        raise ValueError(f"No denomination images were found in {root}.")

    return samples


def _collect_tabular_samples(
    dataset_config: Mapping[str, Any]
) -> list[tuple[Path, str]]:
    metadata_value = dataset_config.get("metadata_path")
    if metadata_value in (None, ""):
        raise ValueError("metadata_path must be provided for this module.")

    metadata_path = resolve_path(metadata_value)
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    root_value = dataset_config.get("root", WORKSPACE_ROOT)
    root = resolve_path(root_value)

    image_column = str(dataset_config.get("image_column", "image_path"))
    label_column = str(dataset_config.get("label_column", "label"))

    samples: list[tuple[Path, str]] = []

    with metadata_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            image_value = row.get(image_column)
            label_value = row.get(label_column)

            if not image_value:
                raise ValueError(f"Missing '{image_column}' in {metadata_path}.")

            if label_value is None:
                raise ValueError(f"Missing '{label_column}' in {metadata_path}.")

            image_path = Path(image_value)
            if not image_path.is_absolute():
                candidate = root / image_path
                image_path = candidate if candidate.exists() else metadata_path.parent / image_path

            samples.append((image_path, str(label_value)))

    if not samples:
        raise ValueError(f"No samples were found in {metadata_path}.")

    return samples


def collect_raw_samples(
    module: str, raw_config: Mapping[str, Any]
) -> list[tuple[Path, str]]:
    dataset_config = _get_dataset_config(raw_config)

    if module == "denomination":
        return _collect_denomination_samples(dataset_config)

    return _collect_tabular_samples(dataset_config)


def encode_samples(
    raw_samples: Sequence[tuple[Path, str]], dataset_config: Mapping[str, Any]
) -> tuple[list[tuple[Path, int]], tuple[str, ...]]:
    label_map = dataset_config.get("label_map")
    class_names = dataset_config.get("class_names")

    if isinstance(label_map, Mapping):
        encoded_lookup = {str(key): int(value) for key, value in label_map.items()}
    elif isinstance(class_names, Sequence):
        encoded_lookup = {str(name): index for index, name in enumerate(class_names)}
    else:
        unique_labels = sorted({str(label) for _, label in raw_samples})
        encoded_lookup = {
            label: index for index, label in enumerate(unique_labels)
        }

    encoded_samples: list[tuple[Path, int]] = []

    for image_path, label in raw_samples:
        key = str(label)
        if key not in encoded_lookup:
            raise KeyError(f"Unknown label '{label}'.")
        encoded_samples.append((image_path, encoded_lookup[key]))

    ordered_class_names = tuple(
        name
        for name, _ in sorted(encoded_lookup.items(), key=lambda item: item[1])
    )

    return encoded_samples, ordered_class_names


def split_indices(
    labels: Sequence[int], train_split: float, seed: int
) -> tuple[list[int], list[int]]:
    if not 0.0 < train_split < 1.0:
        raise ValueError("train_split must be between 0 and 1.")

    rng = np.random.default_rng(seed)

    class_indices: dict[int, list[int]] = defaultdict(list)

    for index, label in enumerate(labels):
        class_indices[label].append(index)

    train_indices: list[int] = []
    val_indices: list[int] = []

    for indices in class_indices.values():
        arr_indices = np.array(indices)
        rng.shuffle(arr_indices)
        split = int(len(arr_indices) * train_split)

        train_indices.extend(arr_indices[:split].tolist())
        val_indices.extend(arr_indices[split:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return train_indices, val_indices


def build_module_dataset(
    module: str,
    raw_config: Mapping[str, Any],
    transform: Callable[[Image.Image], Tensor] | None,
) -> tuple[CurrencyDataset, tuple[str, ...]]:
    """
    Construct module-specific CurrencyDataset instance from src.modules.
    """
    dataset_config = _get_dataset_config(raw_config)
    root = resolve_path(dataset_config.get("root", WORKSPACE_ROOT))

    if module == "denomination":
        if root.exists() and root.is_dir():
            try:
                ds = DenominationDataset(root=root, transform=transform)
                return ds, tuple(ds.classes)
            except Exception:
                pass

    raw_samples = collect_raw_samples(module, raw_config)
    encoded_samples, class_names = encode_samples(raw_samples, dataset_config)

    if module == "authenticity":
        ds = AuthenticityDataset(root=root, samples=encoded_samples, transform=transform)
    elif module == "quality":
        ds = QualityDataset(root=root, samples=encoded_samples, transform=transform)
    else:
        ds = ImageSampleDataset(root=root, samples=encoded_samples, transform=transform)

    return ds, class_names


def build_training_loaders(
    module: str, bundle: Any, train_split: float = 0.8
) -> tuple[DataLoader, DataLoader, tuple[str, ...], torch.Tensor]:

    train_dataset, class_names = build_module_dataset(
        module,
        bundle.raw_config,
        transform=build_train_transforms(bundle.config.dataset.image_size),
    )

    val_dataset, _ = build_module_dataset(
        module,
        bundle.raw_config,
        transform=build_eval_transforms(bundle.config.dataset.image_size),
    )

    labels = [label for _, label in train_dataset.samples]
    counts = Counter(labels)

    num_samples = len(labels)
    num_classes = len(class_names)

    class_weights = torch.tensor(
        [
            num_samples / (num_classes * counts[index])
            for index in range(num_classes)
        ],
        dtype=torch.float32,
    )

    train_indices, val_indices = split_indices(
        labels=labels,
        train_split=train_split,
        seed=bundle.config.seed,
    )

    train_loader = create_dataloader(
        dataset=Subset(train_dataset, train_indices),
        batch_size=bundle.config.dataset.batch_size,
        shuffle=True,
        num_workers=bundle.config.dataset.num_workers,
        pin_memory=bundle.config.dataset.pin_memory,
    )

    val_loader = create_dataloader(
        dataset=Subset(val_dataset, val_indices),
        batch_size=bundle.config.dataset.batch_size,
        shuffle=False,
        num_workers=bundle.config.dataset.num_workers,
        pin_memory=bundle.config.dataset.pin_memory,
    )

    return train_loader, val_loader, class_names, class_weights


def build_evaluation_loader(
    module: str, bundle: Any
) -> tuple[DataLoader, tuple[str, ...]]:
    dataset, class_names = build_module_dataset(
        module,
        bundle.raw_config,
        transform=build_eval_transforms(bundle.config.dataset.image_size),
    )

    loader = create_dataloader(
        dataset=dataset,
        batch_size=bundle.config.dataset.batch_size,
        shuffle=False,
        num_workers=bundle.config.dataset.num_workers,
        pin_memory=bundle.config.dataset.pin_memory,
    )

    return loader, class_names


def build_evaluation_loaders(
    module: str, bundle: Any
) -> tuple[DataLoader, tuple[str, ...]]:
    """Compatibility wrapper for callers that use the pluralized helper name."""
    return build_evaluation_loader(module, bundle)


def build_model_from_config(config: ExperimentConfig) -> nn.Module:
    return ModelRegistry.get_model(
        model_name=config.model.name,
        num_classes=config.dataset.num_classes,
        pretrained=config.model.pretrained,
        **config.model.params,
    )


def build_trainer(
    bundle: ExperimentBundle,
    model: nn.Module,
    class_names: Sequence[str],
    class_weights: torch.Tensor | None = None,
) -> tuple[
    Trainer,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler
    | torch.optim.lr_scheduler.ReduceLROnPlateau
    | None,
]:
    if class_weights is not None:
        class_weights = class_weights.to(bundle.device)

    loss_fn = build_loss(bundle.config.loss, class_weights=class_weights)
    optimizer = build_optimizer(model, bundle.config.optimizer)
    scheduler = build_scheduler(optimizer, bundle.config.scheduler)

    callbacks = [
        ProgressCallback(),
        LoggingCallback(bundle.logger),
        EarlyStoppingCallback(bundle.config.trainer.early_stopping),
        CheckpointCallback(
            config=bundle.config.trainer.checkpoint,
            directory=bundle.paths.checkpoints,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        ),
    ]

    trainer = Trainer(
        config=bundle.config,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=bundle.device,
        callbacks=callbacks,
    )

    return trainer, optimizer, scheduler


def evaluate_model(
    bundle: Any, model: nn.Module, dataloader: DataLoader
) -> EvaluationResult:
    evaluator = Evaluator(
        model=model,
        loss_fn=build_loss(bundle.config.loss),
        device=bundle.device,
    )

    return evaluator.evaluate(dataloader)


def serialize_evaluation_result(
    result: EvaluationResult, class_names: Sequence[str]
) -> dict[str, Any]:
    serialized = serialize_eval_result(result)
    serialized["class_names"] = list(class_names)
    return serialized


def save_evaluation_result(
    result: EvaluationResult, output_dir: Path, class_names: Sequence[str]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        serialize_evaluation_result(result, class_names),
        output_dir / "metrics.json",
    )

    with (output_dir / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["class"] + list(class_names))
        for index, row in enumerate(result.confusion_matrix.tolist()):
            writer.writerow(
                [
                    class_names[index] if index < len(class_names) else index,
                    *row,
                ]
            )


def resolve_named_module(root: nn.Module, path: str) -> nn.Module:
    module: nn.Module = root

    for part in path.split("."):
        if part.isdigit():
            module = module[int(part)]  # type: ignore[index]
            continue

        module = getattr(module, part)

    if not isinstance(module, nn.Module):
        raise TypeError(f"'{path}' did not resolve to a module.")

    return module


def resolve_gradcam_target_layer(
    model: nn.Module, target_layer: str | None = None
) -> nn.Module:
    if target_layer:
        return resolve_named_module(model, target_layer)

    last_conv: nn.Module | None = None

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module

    if last_conv is None:
        raise ValueError(
            "Could not infer a Grad-CAM target layer. "
            "Pass --target-layer explicitly."
        )

    return last_conv


def preprocess_pil_image(image: Image.Image, image_size: int) -> Tensor:
    transform = build_eval_transforms(image_size)
    tensor = transform(image.convert("RGB"))

    if not isinstance(tensor, Tensor):
        raise TypeError("Preprocessing must return a tensor.")

    return tensor.unsqueeze(0)


def load_image_tensor(image_path: Path, image_size: int) -> Tensor:
    config = InferencePreprocessingConfig(
        image_size=(image_size, image_size),
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )
    return preprocess_inference_image(image_path, config)


def predict_image(
    model: nn.Module, image: Path | Image.Image, image_size: int
) -> tuple[Any, Tensor]:
    predictor = Predictor(model)

    if isinstance(image, Path):
        tensor = load_image_tensor(image, image_size)
    else:
        tensor = preprocess_pil_image(image, image_size)

    result = predictor.predict(tensor)
    return result, tensor


def generate_gradcam_overlay(
    model: nn.Module,
    image: Path | Image.Image,
    image_size: int,
    target_layer: str | None = None,
    target_class: int | None = None,
) -> tuple[Any, np.ndarray]:
    if isinstance(image, Path):
        with Image.open(image) as loaded:
            pil_image = loaded.convert("RGB")
    else:
        pil_image = image.convert("RGB")

    tensor = preprocess_pil_image(pil_image, image_size).squeeze(0)
    layer = resolve_gradcam_target_layer(model, target_layer)
    gradcam = GradCAM(model, layer)
    result = gradcam.generate(tensor, target_class=target_class)

    resized = pil_image.resize(result.image_size)
    overlay = overlay_heatmap(
        np.asarray(resized, dtype=np.uint8),
        result.heatmap,
    )

    return result, overlay


def load_experiment_bundle(
    experiment_root: Path,
    checkpoint_name: str = "best.pt",
    device_name: str = "auto",
) -> LoadedExperiment:
    raw_config = load_yaml(experiment_root / "config.yaml")
    if not isinstance(raw_config, dict):
        raise TypeError("Experiment config must be a mapping.")

    config = parse_config(raw_config)
    device = resolve_device(device_name if device_name else config.device)

    model = build_model_from_config(config)
    module = str(raw_config["module"]["name"])
    model_name = str(raw_config["model"]["name"])

    checkpoint_path = experiment_root / "checkpoints" / checkpoint_name

    if not checkpoint_path.exists():
        checkpoint_path = experiment_root / "checkpoints" / "last.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    prepare_model(checkpoint_path=checkpoint_path, model=model, device=device)

    class_names = tuple(
        str(name) for name in raw_config.get("dataset", {}).get("class_names", [])
    )

    return LoadedExperiment(
        root=experiment_root,
        module=module,
        model_name=model_name,
        raw_config=raw_config,
        config=config,
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
        class_names=class_names,
    )


def save_prediction_results(
    output_path: Path, predictions: Sequence[dict[str, Any]]
) -> None:
    save_json({"predictions": list(predictions)}, output_path)


def run_streamlit_app(app_path: Path, args: Sequence[str] = ()) -> int:
    command = [sys.executable, "-m", "streamlit", "run", str(app_path), *args]
    completed = subprocess.run(command, check=False)
    return completed.returncode