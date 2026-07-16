# Currency Recognition Project — Development Context Export (Post Phase 6)

## Project Overview

The project is a modular deep-learning framework for Bangladeshi currency recognition.

Development philosophy:

* Bottom-up implementation
* Strong typing (Pylance Standard)
* Immutable configuration
* Small focused modules
* High unit test coverage
* Separation of responsibilities
* Dependency injection
* Protocol-based interfaces where appropriate

Current repository status:

```
████████████████████  Phase 1  Repository Structure
████████████████████  Phase 2  Configuration & Experiment Management
████████████████████  Phase 3  Data Pipeline
████████████████████  Phase 4  Model Registry
████████████████████  Phase 5  Training Infrastructure
████████████████████  Phase 6  Evaluation
□□□□□□□□□□□□□□□  Phase 7  Explainability
□□□□□□□□□□□□□□□  Phase 8  Inference
□□□□□□□□□□□□□□□  Phase 9  Dashboard
```

---

# Phase 2

Configuration is fully implemented.

Configuration flow:

```
YAML
    ↓
ConfigDict
    ↓
parse_config()
    ↓
ExperimentConfig
```

No raw dictionaries are used after parsing.

Experiment responsibilities:

* load configuration
* merge configuration
* parse immutable dataclasses
* create experiment directory structure
* save merged configuration

Experiment does **not**:

* build models
* create datasets
* train
* evaluate

## Experiment Paths

The experiment directory structure was refined during Phase 6.

Instead of exposing a single evaluation file path, the experiment now exposes an evaluation directory.

Example layout:

```
experiment/
│
├── checkpoints/
├── predictions/
├── figures/
├── gradcam/
├── evaluation/
│
├── config.yaml
└── train.log
```

Evaluation artifacts are stored inside:

```
evaluation/
```

instead of directly under the experiment root.

This keeps ExperimentPaths stable for future phases.

---

# Dataset Contract

Dataset API:

```python
def __getitem__(
    self,
    index: int,
) -> tuple[Tensor, int]:
```

---

# Model Contract

Every model implements:

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

Training layer is complete.

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
    ├── Callbacks
    ├── Device
    └── TrainState
```

Implemented components:

```
device.py
state.py
metrics.py
losses.py
optimizer.py
scheduler.py
protocols.py
amp.py
checkpoint.py
trainer.py
engine.py
```

Callback system:

```
base.py
progress.py
logging.py
checkpoint.py
early_stopping.py
```

TrainingEngine responsibilities:

* forward pass
* backward pass
* optimizer step
* validation
* scheduler stepping
* callback execution

TrainingEngine intentionally does **not**:

* create models
* create datasets
* parse configuration
* save checkpoints directly

Checkpoint decisions belong to callbacks.

---

## Shared Utility

A shared helper now exists:

```
move_batch_to_device()
```

TrainingEngine uses this helper instead of manually calling `.to(device)` on tensors.

This utility is intended to be reused by future evaluation and inference pipelines.

---

# Phase 6 — Evaluation

Phase 6 has been fully implemented.

Directory:

```
src/evaluation/
│
├── __init__.py
├── state.py
├── metrics.py
├── confusion_matrix.py
├── evaluator.py
└── serialization.py
```

The evaluation package mirrors the design philosophy of the training package.

---

## state.py

Implemented immutable dataclasses.

### EvaluationMetrics

Contains:

* accuracy
* precision
* recall
* f1
* roc_auc

### EvaluationResult

Contains:

* loss
* EvaluationMetrics
* predictions
* targets
* probabilities
* confusion_matrix

Both dataclasses are:

* frozen
* slotted
* fully typed

---

## metrics.py

Pure metric functions.

Implemented:

```
accuracy()
precision()
recall()
f1()
roc_auc()
classification_metrics()
```

Implementation uses:

```
scikit-learn
```

Characteristics:

* macro precision
* macro recall
* macro F1
* binary ROC-AUC
* multiclass ROC-AUC (One-vs-Rest)

No state.

No filesystem operations.

No plotting.

---

## confusion_matrix.py

Implemented:

```
compute_confusion_matrix()
```

Responsibilities:

* compute confusion matrix
* return NumPy array

No visualization.

No plotting.

---

## evaluator.py

Core evaluation engine.

Responsibilities:

* model inference
* forward pass
* loss computation
* probability computation
* prediction computation
* aggregate outputs
* compute metrics
* compute confusion matrix

Evaluator intentionally does **not**:

* load checkpoints
* create datasets
* create models
* save files
* write JSON
* generate figures

It mirrors TrainingEngine in philosophy.

---

## serialization.py

Implemented:

```
serialize()
```

Responsibilities:

Convert an immutable EvaluationResult into a JSON-serializable dictionary.

Serialization converts:

* tensors → lists
* NumPy arrays → lists

Serialization is intentionally separated from EvaluationResult to preserve single responsibility.

---

# Evaluation Architecture

```
DataLoader
      │
      ▼
Evaluator
      │
      ├── move_batch_to_device()
      ├── forward
      ├── loss
      ├── probabilities
      ├── predictions
      │
      ▼
classification_metrics()
      │
      ▼
compute_confusion_matrix()
      │
      ▼
EvaluationResult
      │
      ▼
serialize()
```

---

# Testing

Unit tests implemented:

```
test_state.py
test_metrics.py
test_confusion_matrix.py
test_evaluator.py
test_serialization.py
```

Integration test:

```
test_evaluation_integration.py
```

Integration verifies:

```
DataLoader
      │
      ▼
Evaluator
      │
      ▼
metrics
      │
      ▼
confusion matrix
      │
      ▼
EvaluationResult
```

All evaluation tests pass.

---

# Typing Philosophy

Project continues using Pylance Standard.

Rules:

* explicit type hints
* avoid Any
* immutable dataclasses
* Protocol where appropriate
* dependency injection
* typed public APIs

---

# Package Philosophy

Package initializers remain minimal.

Example:

```python
"""
Evaluation package.

Provides:
- evaluator
- metrics
- confusion matrix
- serialization
"""
```

Avoid:

* wildcard imports
* package-level re-exports
* circular imports

---

# Current Overall Architecture

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


Experiment
      │
      ▼
Evaluator
      │
      ├── forward
      ├── probabilities
      ├── predictions
      ├── metrics
      ├── confusion matrix
      └── EvaluationResult
```

Training and evaluation now follow the same architectural style.

---

# Scripts

CLI scripts are intentionally postponed.

The following remain to be implemented after earlier project phases are finalized:

```
scripts/train.py
scripts/evaluate.py
scripts/predict.py
scripts/generate_gradcam.py
scripts/dashboard.py
```

These scripts should remain thin orchestration layers and must not contain business logic.

---

# Remaining Work

## Phase 7

Explainability

Implement:

```
Grad-CAM
visualization
heatmap generation
```

Responsibilities:

* Grad-CAM generation
* overlay visualization
* figure export

---

## Phase 8

Inference

Implement:

* checkpoint loading
* prediction pipeline
* batch inference
* CLI interface

---

## Phase 9

Dashboard

Integrate:

* training metrics
* evaluation metrics
* confusion matrix
* Grad-CAM visualizations
* model comparison

---

# Overall Status

The project now contains complete, modular implementations of:

* configuration management
* experiment management
* dataset pipeline
* model registry
* training infrastructure
* evaluation infrastructure

The codebase consistently follows:

* immutable configuration
* dependency injection
* protocol-based typing
* strong separation of concerns
* reusable utilities
* comprehensive unit testing
* end-to-end integration testing

The next development milestone is Phase 7 (Explainability), which will build on the completed evaluation pipeline while preserving the same architectural principles.
