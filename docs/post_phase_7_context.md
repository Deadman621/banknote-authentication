# Currency Recognition Project — Development Context Export (Post Phase 7)

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
████████████████████  Phase 7  Explainability
□□□□□□□□□□□□□□□  Phase 8  Inference
□□□□□□□□□□□□□□□  Phase 9  Dashboard

```

---

# Recent Architectural Change

Checkpointing logic has been decoupled from training.

Previous:

```

TrainingEngine
|
└── checkpoint saving

```

Current:

```

TrainingEngine
|
▼
Callbacks
|
▼
Checkpoint Module

```

Checkpoint responsibilities are isolated.

TrainingEngine does not:

* save checkpoints
* manage checkpoint paths
* decide checkpoint policy

Checkpoint decisions belong to callbacks.

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

Experiment does not:

* build models
* create datasets
* train
* evaluate

---

# Experiment Paths

Current experiment structure:

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

Grad-CAM artifacts are stored inside:

```

gradcam/

````

---

# Dataset Contract

Dataset API:

```python
def __getitem__(
    self,
    index: int,
) -> tuple[Tensor, int]:
````

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

Implemented:

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

Shared utility:

```
move_batch_to_device()
```

---

# Phase 6 — Evaluation

Complete.

Directory:

```
src/evaluation/

├── __init__.py
├── state.py
├── metrics.py
├── confusion_matrix.py
├── evaluator.py
└── serialization.py
```

Implemented:

## state.py

Immutable dataclasses:

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

Properties:

* frozen
* slotted
* fully typed

## metrics.py

Pure functions:

```
accuracy()
precision()
recall()
f1()
roc_auc()
classification_metrics()
```

Uses:

```
scikit-learn
```

No state.

No filesystem.

No plotting.

## confusion_matrix.py

Implemented:

```
compute_confusion_matrix()
```

Returns NumPy array.

No visualization.

## evaluator.py

Responsibilities:

* model inference
* forward pass
* loss computation
* probability generation
* predictions
* metrics
* confusion matrix

Does not:

* load checkpoints
* create datasets
* create models
* save files

## serialization.py

Implemented:

```
serialize()
```

Converts:

* tensors → lists
* NumPy arrays → lists

---

# Phase 7 — Explainability

Complete.

Directory:

```
src/explainability/

├── __init__.py
├── state.py
├── hooks.py
├── gradcam.py
└── visualization.py
```

---

# state.py

Contains immutable Grad-CAM result object.

Example:

```python
GradCAMResult
```

Contains:

* heatmap
* predicted_class
* target_class
* class_probabilities
* image_size

Properties:

* frozen
* slotted
* typed

---

# hooks.py

Responsible for:

* activation capture
* gradient capture
* hook registration
* hook removal

No Grad-CAM calculations.

---

# gradcam.py

Complete Grad-CAM implementation.

Responsibilities:

* forward pass
* target class selection
* backward pass
* gradient extraction
* channel weight computation
* heatmap generation
* normalization
* resizing

Does not:

* save images
* write files
* load models

Flow:

```
Image Tensor
      |
      ▼
GradCAM
      |
      ▼
GradCAMResult
```

---

# visualization.py

Pure visualization utilities.

No PyTorch dependency.

Responsibilities:

* image validation
* heatmap validation
* colormap generation
* heatmap overlay

Implemented functions:

```
validate_image()
validate_heatmap()
apply_colormap()
overlay_heatmap()
```

Uses:

```
OpenCV
NumPy
```

Returns:

```
NDArray[np.uint8]
```

---

# Explainability Testing

Complete.

Tests:

```
tests/explainability/

├── test_state.py
├── test_hooks.py
├── test_gradcam.py
├── test_visualization.py
└── test_integration.py
```

Validated:

* GradCAM generation
* hook lifecycle
* target class selection
* heatmap generation
* normalization
* visualization
* full pipeline integration

Final integration flow:

```
ToyModel
    |
    ▼
GradCAM.generate()
    |
    ▼
GradCAMResult
    |
    ▼
overlay_heatmap()
    |
    ▼
RGB image
```

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

OpenCV typing handled using:

```python
typing.cast()
```

because OpenCV returns `MatLike`.

---

# Current Architecture

```
Experiment
      |
      ▼
Trainer
      |
      ▼
TrainingEngine
      |
      ├── AMP
      ├── Optimizer
      ├── Scheduler
      ├── Loss
      ├── Callbacks
      ├── TrainState
      └── Checkpoint Module


Experiment
      |
      ▼
Evaluator
      |
      ├── forward
      ├── probabilities
      ├── predictions
      ├── metrics
      ├── confusion matrix
      └── EvaluationResult


Experiment
      |
      ▼
GradCAM
      |
      ├── Hooks
      ├── GradCAMResult
      └── Visualization
```

---

# Scripts

Current scripts:

```
scripts/

├── train.py
├── evaluate.py
├── predict.py
├── generate_gradcam.py
├── dashboard.py
└── compare_models.py
```

Scripts remain thin orchestration layers.

No business logic belongs inside scripts.

---

# Testing Status

Completed:

```
Training tests        ✅
Evaluation tests      ✅
Explainability tests  ✅
Integration tests     ✅
```

---

# Remaining Work

## Phase 8 — Inference

Implement:

* checkpoint loading
* prediction pipeline
* batch inference
* CLI interface

Expected architecture:

```
src/inference/

├── __init__.py
├── state.py
├── loader.py
├── predictor.py
└── preprocessing.py
```

Expected flow:

```
Input Image
      |
      ▼
Preprocessing
      |
      ▼
Checkpoint Loader
      |
      ▼
Model Registry
      |
      ▼
Predictor
      |
      ▼
PredictionResult
```

---

## Phase 9 — Dashboard

Integrate:

* training metrics
* evaluation metrics
* confusion matrix
* Grad-CAM visualizations
* model comparison

---

# Overall Status

The project now contains complete modular implementations of:

* configuration management
* experiment management
* dataset pipeline
* model registry
* training infrastructure
* evaluation infrastructure
* explainability infrastructure

The codebase consistently follows:

* immutable configuration
* dependency injection
* protocol-based typing
* strong separation of concerns
* reusable utilities
* comprehensive unit testing
* end-to-end integration testing

Next milestone:
Phase 8 — Inference