from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from src.core.config import ExperimentConfig
from src.core.experiment import Experiment, ExperimentPaths


@dataclass(slots=True)
class ApplicationContext:
    experiment: Experiment
    device: torch.device
    logger: logging.Logger

    @property
    def config(self) -> ExperimentConfig:
        return self.experiment.config

    @property
    def module(self) -> str:
        return self.experiment.module

    @property
    def model(self) -> str:
        return self.experiment.model

    @property
    def paths(self) -> ExperimentPaths:
        return self.experiment.paths