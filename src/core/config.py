# src/core/config.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ExperimentSettings:
    name: str


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    root: Path
    image_size: int
    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    num_classes: int
    class_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EarlyStoppingConfig:
    patience: int
    monitor: str


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    monitor: str
    mode: str
    save_best_only: bool


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    epochs: int
    batch_size: int
    mixed_precision: bool
    gradient_clip: float |None

    early_stopping: EarlyStoppingConfig
    checkpoint: CheckpointConfig


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    name: str
    lr: float
    weight_decay: float


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LossConfig:
    name: str


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str
    pretrained: bool
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    level: str


@dataclass(frozen=True, slots=True)
class OutputConfig:
    save_dir: Path


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    seed: int
    device: str

    experiment: ExperimentSettings

    dataset: DatasetConfig
    trainer: TrainerConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    loss: LossConfig
    model: ModelConfig

    logging: LoggingConfig
    output: OutputConfig