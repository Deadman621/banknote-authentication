# Currency Recognition Project — Development Context Export (Post Phase 5)

## Project Overview

The project is a modular deep-learning framework for Bangladeshi currency recognition.

Development philosophy:

- Bottom-up implementation
- Strong typing (Pylance Standard)
- Small focused modules
- Immutable configuration
- High unit test coverage
- Separation of responsibilities

Current repository status:

```
████████████████████  Phase 1  Repository Structure
████████████████████  Phase 2  Configuration & Experiment Management
(implemented by another developer)
████████████████████  Phase 3  Data Pipeline
(implemented by another developer)
████████████████████  Phase 4  Model Registry
(implemented by another developer)
████████████████████  Phase 5  Training Infrastructure
□□□□□□□□□□□□□□□  Phase 6  Evaluation
□□□□□□□□□□□□□□□  Phase 7  Explainability
□□□□□□□□□□□□□□□  Phase 8  Inference
□□□□□□□□□□□□□□□  Phase 9  Dashboard
```

---

# Phase 2 (Previously Completed)

Configuration system is fully implemented.

Important dataclasses:

- ExperimentConfig
- ExperimentSettings
- DatasetConfig
- TrainerConfig
- OptimizerConfig
- SchedulerConfig
- LossConfig
- ModelConfig
- LoggingConfig
- OutputConfig
- EarlyStoppingConfig
- CheckpointConfig

Configuration philosophy:

```
YAML
    ↓
ConfigDict
    ↓
parse_config()
    ↓
ExperimentConfig
```

No raw dictionaries beyond parsing.

Experiment responsibilities:

- Load config
- Parse config
- Create experiment directories
- Save merged config

Experiment does NOT:

- create datasets
- create models
- train
- evaluate

---

# Development Contracts

Dataset API:

```python
def __getitem__(self, index: int) -> tuple[Tensor, int]:
```

Model API:

```python
class ModelXYZ(nn.Module):

    def __init__(
        self,
        num_classes: int,
        pretrained: bool,
        **kwargs,
    ) -> None:
        ...

    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        ...
```

---

# Phase 5 — Training Infrastructure

The entire training layer has been implemented.

Architecture:

```
Trainer
    │
    ▼
TrainingEngine
    │
    ├── AMP
    ├── Optimizer
    ├── Scheduler
    ├── Loss
    ├── TrainState
    └── Callbacks
```

---

# src/training/device.py

Implemented:

- device selection
- CPU/CUDA handling
- unit tested

---

# src/training/state.py

Implemented:

TrainState dataclass.

Contains training state including:

- epoch
- global_step
- train_loss
- validation_loss
- train_accuracy
- validation_accuracy
- should_stop

This object is shared across callbacks and engine.

---

# src/training/metrics.py

Implemented:

Training metrics.

Currently includes:

- accuracy()

---

# src/training/losses.py

Implemented:

Loss factory.

Creates loss functions from configuration.

---

# src/training/optimizer.py

Implemented:

Optimizer factory.

Supports configuration-driven optimizer creation.

---

# src/training/scheduler.py

Implemented:

Scheduler factory.

Configuration-driven scheduler creation.

---

# src/training/protocols.py

Created to satisfy strict typing.

Contains:

```python
class SchedulerProtocol(Protocol):

    def step(self) -> None:
        ...

    def state_dict(self) -> dict[str, object]:
        ...

    def load_state_dict(
        self,
        state_dict: dict[str, object],
    ) -> None:
        ...
```

This protocol is shared by:

- checkpoint.py
- callbacks/checkpoint.py
- engine.py
- trainer.py

Avoids:

```python
scheduler: object
```

Avoids:

```python
Any
```

Maintains Pylance Standard compatibility.

---

# src/training/amp.py

Implemented:

AMPContext

Responsibilities:

- autocast()
- backward()
- step()
- zero_grad()

Uses:

- GradScaler
- autocast
- nullcontext

Resolved typing issue by returning:

```python
ContextManager[object]
```

instead of

```python
ContextManager[None]
```

---

# src/training/checkpoint.py

Implemented:

```
save_checkpoint()
load_checkpoint()
```

Responsibilities:

- serialize model
- serialize optimizer
- serialize scheduler
- restore training state

Contains no decision logic.

Decision logic belongs to callback.

---

# Callback System

Implemented completely.

---

## callbacks/base.py

Lifecycle hooks:

```
on_train_begin()

on_epoch_begin()

on_batch_begin()

on_batch_end()

on_validation_end()

on_epoch_end()

on_train_end()
```

No-op implementation.

Subclass overrides only required hooks.

---

## callbacks/progress.py

Responsibilities:

- tqdm progress bar
- update batch progress
- display:

    loss

    accuracy

No training logic.

---

## callbacks/logging.py

Responsibilities:

Write lifecycle events.

Uses externally provided logger.

Does NOT configure logging.

---

## callbacks/checkpoint.py

Responsibilities:

Monitor metric.

Compare best metric.

Call:

```
save_checkpoint()
```

Supports:

```
mode=min
mode=max
```

Supports:

```
save_best_only
```

Uses SchedulerProtocol.

---

## callbacks/early_stopping.py

Responsibilities:

Monitor validation metric.

Maintain:

- best_metric
- wait counter

Sets:

```
state.should_stop = True
```

Engine decides when to exit.

---

# src/training/engine.py

Implemented completely.

Constructor accepts:

- model
- loss function
- optimizer
- scheduler
- AMPContext
- callbacks
- device

Engine responsibilities:

- forward pass
- backward pass
- optimizer step
- validation
- callback execution

Engine does NOT:

- create models
- parse config
- create datasets
- create optimizer

---

Implemented methods:

## train_step()

Performs:

```
forward

loss

backward

optimizer step

accuracy
```

Returns:

```
(loss, accuracy)
```

---

## train_epoch()

Responsibilities:

- iterate DataLoader
- call callbacks
- update TrainState
- aggregate metrics

---

## validate_epoch()

Responsibilities:

- model.eval()
- torch.no_grad()
- validation metrics
- update TrainState

No optimizer operations.

---

## fit()

Responsibilities:

```
for epoch

    train_epoch()

    validate_epoch()

    scheduler.step()

    callbacks

    early stopping
```

Returns final TrainState.

---

# src/training/trainer.py

Implemented.

Trainer is intentionally thin.

Responsibilities:

Create TrainingEngine.

Pass:

- model
- optimizer
- scheduler
- callbacks
- AMP
- loss
- config

Expose:

```
fit()
```

Contains no training logic.

---

# Testing

All implemented modules have dedicated unit tests.

Tests completed:

- device
- state
- metrics
- losses
- optimizer
- scheduler
- AMP
- checkpoint
- callback base
- progress callback
- logging callback
- checkpoint callback
- early stopping callback
- engine constructor
- train_step
- train_epoch
- validate_epoch
- fit
- trainer

All tests passed during development.

---

# Integration Test

Created:

```
tests/test_training_integration.py
```

Purpose:

End-to-end verification of Phase 5.

Flow:

Trainer

↓

TrainingEngine

↓

Callbacks

↓

Checkpoint

↓

TrainState

Verifies:

- trainer creation
- engine execution
- callbacks execute
- checkpoint written
- state updated
- end-to-end compatibility

---

# __init__.py Philosophy

Keep package initializers minimal.

Preferred:

```python
"""
Training package.

Contains:
- engine
- trainer
- callbacks
- optimization
"""
```

Avoid importing all submodules from package root.

Avoid wildcard exports.

Avoid introducing circular imports.

---

# Typing Philosophy

Project uses:

Pylance Standard.

Rules:

- complete type hints
- avoid Any
- use Protocol where appropriate
- immutable dataclasses
- typed public APIs
- explicit interfaces

Protocols preferred over:

```python
object
```

or

```python
Any
```

---

# Current Architecture

```
Experiment
      │
      ▼
Trainer
      │
      ▼
TrainingEngine
      │
      ├── AMP
      ├── Optimizer
      ├── Scheduler
      ├── Loss
      ├── Callbacks
      ├── TrainState
      └── Device
```

Callbacks:

```
TrainingEngine
      │
      ├── ProgressCallback
      ├── LoggingCallback
      ├── CheckpointCallback
      └── EarlyStoppingCallback
```

---

# Remaining Work

## Phase 6

Evaluation

Implement:

```
src/evaluation/

metrics.py

evaluator.py

confusion_matrix.py
```

Responsibilities:

- inference on validation/test sets
- classification metrics
- confusion matrix
- ROC/AUC where applicable
- prediction export
- error analysis support

---

## Phase 7

Explainability

Implement:

```
Grad-CAM

visualization

heatmap generation
```

---

## Phase 8

Inference

Implement:

- prediction pipeline
- checkpoint loading
- batch inference
- CLI interface

---

## Phase 9

Dashboard

Integrate:

- training metrics
- evaluation metrics
- confusion matrix
- Grad-CAM visualizations
- model comparison

---

# Overall Status

The project now has a complete, modular, strongly typed training infrastructure that is decoupled from configuration, datasets, and model implementations.

The training layer is designed around:

- immutable configuration
- dependency injection
- protocol-based typing
- callback extensibility
- isolated responsibilities
- comprehensive unit testing
- end-to-end integration testing

The next logical milestone is implementing the evaluation layer while keeping the same architectural principles.