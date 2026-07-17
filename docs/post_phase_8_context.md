# Currency Recognition Project — Development Context Export (Post Phase 8)

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

Repository status:

```text
████████████████████  Phase 1  Repository Structure
████████████████████  Phase 2  Configuration & Experiment Management
████████████████████  Phase 3  Data Pipeline
████████████████████  Phase 4  Model Registry
████████████████████  Phase 5  Training Infrastructure
████████████████████  Phase 6  Evaluation
████████████████████  Phase 7  Explainability
████████████████████  Phase 8  Inference
□□□□□□□□□□□□□□□  Phase 9  Dashboard
```

---

# Architectural Principles

The project consistently follows:

* Immutable dataclasses for state objects
* Dependency injection
* Protocol-oriented interfaces where appropriate
* Strong separation of concerns
* Thin orchestration layers
* Reusable utility functions
* Comprehensive unit testing
* End-to-end integration testing

---

# Experiment

Configuration flow:

```text
YAML
 ↓
ConfigDict
 ↓
parse_config()
 ↓
ExperimentConfig
```

After parsing, no raw dictionaries are used.

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
* perform inference

---

# Training

Training architecture:

```text
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

Checkpoint saving has been completely decoupled from the training engine.

Current ownership:

```text
TrainingEngine
        │
        ▼
Callbacks
        │
        ▼
Checkpoint Module
```

TrainingEngine never saves checkpoints directly.

---

# Evaluation

Directory:

```text
src/evaluation/

├── state.py
├── metrics.py
├── confusion_matrix.py
├── evaluator.py
└── serialization.py
```

Evaluation responsibilities:

* forward pass
* loss computation
* prediction generation
* probability computation
* classification metrics
* confusion matrix generation

Evaluation does **not**:

* load checkpoints
* create datasets
* instantiate models
* save files

---

# Explainability

Directory:

```text
src/explainability/

├── state.py
├── hooks.py
├── gradcam.py
└── visualization.py
```

Responsibilities:

* Grad-CAM generation
* hook lifecycle management
* activation capture
* gradient capture
* heatmap generation
* visualization

Visualization remains completely independent of PyTorch.

---

# Phase 8 — Inference

Directory:

```text
src/inference/

├── __init__.py
├── state.py
├── preprocessing.py
├── loader.py
└── predictor.py
```

---

## state.py

Contains immutable inference state.

### PredictionResult

Represents a single prediction.

Fields:

* predicted_class
* confidence
* probabilities (`NDArray[np.float32]`)

Properties:

* frozen
* slotted
* fully typed

### BatchPredictionResult

Contains:

* tuple of `PredictionResult`

Properties:

* frozen
* slotted
* typed

State objects intentionally contain no validation or business logic.

---

## preprocessing.py

Responsible for image preprocessing only.

Responsibilities:

* image loading
* RGB conversion
* transform construction
* transform application
* tensor generation

Public API:

```python
preprocess_image(...)
```

Private helpers:

* `_load_image()`
* `_build_transform()`
* `_apply_transform()`

Outputs batched tensors ready for inference.

No model interaction.

No filesystem writes.

---

## loader.py

Responsible only for preparing an existing model for inference.

Responsibilities:

* load checkpoint weights
* move model to device
* switch model to evaluation mode

Implementation delegates checkpoint loading to the existing checkpoint module.

Model construction remains outside the loader.

Public API:

```python
prepare_model(
    checkpoint_path,
    model,
    device,
)
```

This design avoids coupling checkpoint serialization with model instantiation.

---

## predictor.py

Responsible for model inference.

Public interface:

```python
Predictor

predict(...)
predict_batch(...)
```

Responsibilities:

* forward pass
* softmax computation
* class selection
* confidence computation
* immutable result construction

Returns:

* `PredictionResult`
* `BatchPredictionResult`

Predictor has no knowledge of:

* checkpoints
* image loading
* preprocessing
* experiment configuration
* filesystem

---

# Inference Flow

```text
Image
    │
    ▼
preprocess_image()
    │
    ▼
Tensor
    │
    ▼
prepare_model()
    │
    ▼
Predictor
    │
    ├── predict()
    └── predict_batch()
    │
    ▼
PredictionResult
```

---

# Checkpoint Architecture

Checkpoint module remains training-oriented.

Stored state:

* epoch
* global_step
* model_state_dict
* optimizer_state_dict
* scheduler_state_dict (optional)

Checkpoint module does **not** store:

* architecture name
* model metadata
* number of classes
* configuration

Therefore inference receives an already-constructed model and loads weights into it.

---

# Typing Philosophy

The project continues using Pylance Standard.

Rules:

* explicit type hints
* avoid `Any`
* immutable dataclasses
* Protocol where appropriate
* dependency injection
* typed public APIs

Known third-party typing limitations are handled using `typing.cast()` where appropriate (e.g., OpenCV and torchvision transform outputs).

---

# Testing Status

Training tests:

* ✅ complete

Evaluation tests:

* ✅ complete

Explainability tests:

* ✅ complete

Inference tests:

```text
test_state.py            ✅
test_preprocessing.py    ✅
test_loader.py           ✅
test_predictor.py        ✅
test_integration.py      ✅
```

Inference coverage includes:

* immutable state
* preprocessing pipeline
* checkpoint loading
* model preparation
* prediction
* batch prediction
* probability generation
* deterministic inference
* complete end-to-end integration

All inference tests pass.

---

# Scripts

Current scripts:

```text
scripts/

├── train.py
├── evaluate.py
├── predict.py
├── generate_gradcam.py
├── dashboard.py
└── compare_models.py
```

Scripts remain thin orchestration layers.

Business logic belongs only inside `src/`.

---

# Current Project Architecture

```text
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
      └── Checkpoint
```

```text
Experiment
      │
      ▼
Evaluator
      │
      ├── Forward
      ├── Metrics
      ├── Confusion Matrix
      └── EvaluationResult
```

```text
Experiment
      │
      ▼
GradCAM
      │
      ├── Hooks
      ├── Heatmap
      └── Visualization
```

```text
Image
      │
      ▼
Preprocessing
      │
      ▼
Prepared Model
      │
      ▼
Predictor
      │
      ├── PredictionResult
      └── BatchPredictionResult
```

---

# Remaining Work

## Phase 9 — Dashboard

Integrate:

* training metrics
* evaluation metrics
* confusion matrices
* Grad-CAM visualizations
* prediction visualization
* experiment comparison
* model comparison

The dashboard should act as a presentation layer over the existing modular infrastructure and should not duplicate training, evaluation, explainability, or inference logic.

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
* inference infrastructure

The architecture consistently follows:

* immutable configuration
* immutable state objects
* dependency injection
* protocol-based typing
* strong separation of concerns
* reusable utilities
* comprehensive unit testing
* end-to-end integration testing

**Next milestone:** **Phase 9 — Dashboard**
