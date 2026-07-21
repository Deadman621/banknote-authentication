from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any, TypeVar, cast
from datetime import datetime
from typing import TypeAlias
from copy import deepcopy
from pathlib import Path

from src.utils.io import load_yaml, save_yaml
from src.utils.paths import CONFIG_DIR

from src.core.config import (
    DatasetConfig,
    ExperimentConfig,
    ExperimentSettings,
    LoggingConfig,
    LossConfig,
    ModelConfig,
    OptimizerConfig,
    OutputConfig,
    SchedulerConfig,
    TrainerConfig,
    EarlyStoppingConfig,
    CheckpointConfig,
    CannyConfig,
    ContourConfig,
    AlignedOutputConfig,
    PreprocessingConfig
)

T = TypeVar("T")

ConfigValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | list["ConfigValue"]
    | dict[str, "ConfigValue"]
)

ConfigDict: TypeAlias = dict[str, ConfigValue]

@dataclass(frozen=True, slots=True)
class ExperimentPaths:
    root: Path

    checkpoints: Path
    predictions: Path
    figures: Path
    gradcam: Path
    evaluation: Path

    config_file: Path
    log_file: Path
    metrics_file: Path

def deep_merge(base: ConfigDict, update: Mapping[str, ConfigValue]) -> ConfigDict:
    """
    Recursively merge two mappings.

    Values from ``update`` override values in ``base``.
    Neither input mapping is modified.
    """

    merged: ConfigDict = deepcopy(dict(base))

    for key, value in update.items():
        existing = merged.get(key)

        if (
            isinstance(existing, MutableMapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = deepcopy(value)

    return merged

def load_config_dict(module: str, model: str) -> ConfigDict:
    """
    Load and merge configuration files.

    Precedence (later overrides earlier):

    common.yaml
    trainer.yaml
    modules/<module>.yaml
    models/<model>.yaml
    """

    config: dict[str, Any] = {}

    files = (
        CONFIG_DIR / "common.yaml",
        CONFIG_DIR / "trainer.yaml",
        CONFIG_DIR / "modules" / f"{module}.yaml",
        CONFIG_DIR / "models" / f"{model}.yaml",
    )

    for file in files:

        if not file.exists():
            raise FileNotFoundError(file)

        loaded = load_yaml(file)

        if not isinstance(loaded, dict):
            raise TypeError(
                f"{file} did not contain a mapping."
            )

        config = deep_merge(config, loaded)

    return config

def load_preprocessing_configs():
    cfg = load_yaml(CONFIG_DIR / "common.yaml")

    preprocessing = parse_preprocessing_config(require_mapping(cfg, "preprocessing"))
    logging = parse_logging_config(require_mapping(cfg, "logging"))

    return preprocessing, logging

def require(mapping: Mapping[str, object], key: str, expected_type: type[T] | tuple[type[Any], ...]) -> T:
    value = mapping.get(key)

    if not isinstance(value, expected_type):
        if isinstance(expected_type, tuple):
            expected = " or ".join(t.__name__ for t in expected_type)
        else:
            expected = expected_type.__name__

        raise TypeError(
            f"Expected '{key}' to be {expected}."
        )

    return cast(T, value)


def require_mapping(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = mapping.get(key)

    if not isinstance(value, Mapping):
        raise TypeError(
            f"Expected '{key}' to be a mapping."
        )

    return value

def parse_experiment_settings(cfg: Mapping[str, object]) -> ExperimentSettings:
    return ExperimentSettings(name=require(cfg, "name", str))

def parse_loss_config(cfg: Mapping[str, object]) -> LossConfig:
    return LossConfig(name=require(cfg, "name", str))

def parse_optimizer_config(cfg: Mapping[str, object]) -> OptimizerConfig:
    return OptimizerConfig(
        name=require(cfg, "name", str),
        lr=require(cfg, "lr", float),
        weight_decay=require(cfg, "weight_decay", float),
    )

def parse_logging_config(cfg: Mapping[str, object]) -> LoggingConfig:
    return LoggingConfig(
        level=require(cfg, "level", str),
    )


def parse_output_config(cfg: Mapping[str, object]) -> OutputConfig:
    return OutputConfig(
        save_dir=Path(require(cfg, "save_dir", str)),
    )


def parse_scheduler_config(cfg: Mapping[str, object]) -> SchedulerConfig:
    params = cfg.get("params", {})

    if not isinstance(params, dict):
        raise TypeError("'params' must be a mapping.")

    return SchedulerConfig(
        name=require(cfg, "name", str),
        params=dict(params),
    )


def parse_model_config(cfg: Mapping[str, object]) -> ModelConfig:
    params = {
        key: value
        for key, value in cfg.items()
        if key not in {"name", "pretrained"}
    }

    return ModelConfig(
        name=require(cfg, "name", str),
        pretrained=require(cfg, "pretrained", bool),
        params=params,
    )


def parse_dataset_config(cfg: Mapping[str, object]) -> DatasetConfig:
    class_names = require(cfg, "class_names", list)

    return DatasetConfig(
        root=Path(require(cfg, "root", str)),
        image_size=require(cfg, "image_size", int),
        batch_size=require(cfg, "batch_size", int),
        num_workers=require(cfg, "num_workers", int),
        pin_memory=require(cfg, "pin_memory", bool),
        persistent_workers=require(cfg, "persistent_workers", bool),
        num_classes=require(cfg, "num_classes", int),
        class_names=tuple(str(name) for name in class_names),
    )


def parse_trainer_config(cfg: Mapping[str, object]) -> TrainerConfig:
    early_cfg = require_mapping(cfg, "early_stopping")
    checkpoint_cfg = require_mapping(cfg, "checkpoint")

    gradient_clip = cfg.get("gradient_clip")

    if gradient_clip is not None and not isinstance(gradient_clip, (int, float)):
        raise TypeError("'gradient_clip' must be numeric or None.")

    return TrainerConfig(
        epochs=require(cfg, "epochs", int),
        mixed_precision=require(cfg, "mixed_precision", bool),
        gradient_clip=None if gradient_clip is None else float(gradient_clip),
        early_stopping=EarlyStoppingConfig(
            patience=require(early_cfg, "patience", int),
            monitor=require(early_cfg, "monitor", str),
        ),
        checkpoint=CheckpointConfig(
            monitor=require(checkpoint_cfg, "monitor", str),
            mode=require(checkpoint_cfg, "mode", str)
        ),
    )

def parse_preprocessing_config(cfg: Mapping[str, object]) -> PreprocessingConfig:
    canny_cfg = require_mapping(cfg, "canny")
    contour_cfg = require_mapping(cfg, "contour")
    output_cfg = require_mapping(cfg, "aligned_output")

    kernel = require(cfg, "gaussian_kernel", list)

    if len(kernel) != 2:
        raise ValueError("'gaussian_kernel' must contain exactly two integers.")

    return PreprocessingConfig(
        gaussian_kernel=(int(kernel[0]), int(kernel[1])),
        gaussian_sigma=float(require(cfg, "gaussian_sigma", (int, float))),
        resize_max_dim=require(cfg, "resize_max_dim", int),
        debug=require(cfg, "debug", bool),
        debug_output_dir=require(cfg, "debug_output_dir", str),
        canny=CannyConfig(
            adaptive=require(canny_cfg, "adaptive", bool),
            lower=require(canny_cfg, "lower", int),
            upper=require(canny_cfg, "upper", int),
        ),
        contour=ContourConfig(
            min_area_ratio=float(require(contour_cfg, "min_area_ratio", (int, float))),
            approx_epsilon=float(require(contour_cfg, "approx_epsilon", (int, float))),
        ),
        aligned_output=AlignedOutputConfig(
            width=require(output_cfg, "width", int),
            height=require(output_cfg, "height", int),
        ),
    )

def parse_config(cfg: Mapping[str, object]) -> ExperimentConfig:
    return ExperimentConfig(
        seed=require(cfg, "seed", int),
        device=require(cfg, "device", str),

        experiment=parse_experiment_settings(
            require_mapping(cfg, "experiment")
        ),

        dataset=parse_dataset_config(
            require_mapping(cfg, "dataset")
        ),

        trainer=parse_trainer_config(
            require_mapping(cfg, "trainer")
        ),

        optimizer=parse_optimizer_config(
            require_mapping(cfg, "optimizer")
        ),

        scheduler=parse_scheduler_config(
            require_mapping(cfg, "scheduler")
        ),

        loss=parse_loss_config(
            require_mapping(cfg, "loss")
        ),

        model=parse_model_config(
            require_mapping(cfg, "model")
        ),

        logging=parse_logging_config(
            require_mapping(cfg, "logging")
        ),

        output=parse_output_config(
            require_mapping(cfg, "output")
        ),

        preprocessing=parse_preprocessing_config(
            require_mapping(cfg, "preprocessing")
        ),
    )

def create_experiment_paths(save_dir: Path, module: str, model: str, experiment_name: str) -> ExperimentPaths:
    """
    Create the directory structure for a training experiment.

    Layout
    ------
    experiments/
        <module>/
            <model>/
                <experiment_name>/
                    <timestamp>/
                        checkpoints/
                        predictions/
                        figures/
                        gradcam/
                        config.yaml
                        train.log
                        metrics.json
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    root = (
        save_dir
        / module
        / model
        / experiment_name
        / timestamp
    )

    evaluation = root / "evaluation"
    checkpoints = root / "checkpoints"
    predictions = root / "predictions"
    figures = root / "figures"
    gradcam = root / "gradcam"

    for directory in (
        checkpoints,
        predictions,
        figures,
        gradcam,
        evaluation,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    return ExperimentPaths(
        root=root,
        checkpoints=checkpoints,
        predictions=predictions,
        figures=figures,
        gradcam=gradcam,
        evaluation=evaluation,
        config_file=root / "config.yaml",
        log_file=root / "train.log",
        metrics_file=root / "metrics.json",
    )

class Experiment:
    def __init__(self, module: str, model: str) -> None:
        raw_config = load_config_dict(
            module=module,
            model=model,
        )

        self.config = parse_config(raw_config)

        self.paths = create_experiment_paths(
            save_dir=self.config.output.save_dir,
            module=module,
            model=model,
            experiment_name=self.config.experiment.name,
        )

        save_yaml(raw_config, self.paths.config_file)