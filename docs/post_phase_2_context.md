# Currency Recognition Project — Development Context (End of Phase 2)

## Project Status

Completed **Phase 1** (project structure) and **Phase 2** (configuration & experiment management).

Current codebase includes:

* Fully structured repository
* Typed configuration system
* Experiment configuration loader
* Recursive YAML merging
* Configuration parser
* Experiment path generation
* Unit tests
* Pylance Standard compatibility

---

# Configuration Architecture

## `src/core/config.py`

Contains immutable dataclasses for:

* ExperimentConfig
* ExperimentSettings
* DatasetConfig
* TrainerConfig
* OptimizerConfig
* SchedulerConfig
* LossConfig
* ModelConfig
* LoggingConfig
* OutputConfig
* EarlyStoppingConfig
* CheckpointConfig

All dataclasses use

```python
@dataclass(frozen=True, slots=True)
```

---

# `src/core/experiment.py`

Implemented:

### Type aliases

```python
ConfigValue
ConfigDict
```

### Dataclass

```python
ExperimentPaths
```

contains

* root
* checkpoints
* predictions
* figures
* gradcam
* config_file
* log_file
* metrics_file

---

### Functions

Implemented:

```
deep_merge()

load_config_dict()

require()

require_mapping()

parse_experiment_settings()

parse_loss_config()

parse_optimizer_config()

parse_logging_config()

parse_output_config()

parse_scheduler_config()

parse_model_config()

parse_dataset_config()

parse_trainer_config()

parse_config()

create_experiment_paths()
```

Configuration loading order:

```
common.yaml
↓
trainer.yaml
↓
modules/<module>.yaml
↓
models/<model>.yaml
```

Later files override earlier ones.

---

# Testing

Implemented and passing:

* deep_merge
* load_config_dict
* require
* require_mapping
* parse_experiment_settings
* parse_loss_config
* parse_optimizer_config
* parse_logging_config
* parse_output_config
* parse_scheduler_config
* parse_model_config
* parse_dataset_config
* parse_trainer_config
* parse_config
* create_experiment_paths

Current status:

```
15 passed
```

---

# Design Decisions

## Strong typing

Pylance type checking is set to **Standard**.

All public APIs should include complete type hints.

Avoid `Any` except at external I/O boundaries.

---

## Configuration philosophy

YAML is treated as an untyped input.

Immediately after loading:

```
YAML
    ↓
ConfigDict
    ↓
parse_config()
    ↓
ExperimentConfig
```

The remainder of the application should use typed dataclasses only.

No raw dictionaries beyond the parsing layer.

---

## Experiment philosophy

`Experiment` is intentionally thin.

Responsibilities:

* load configuration
* parse configuration
* create experiment directory structure
* save merged configuration

Responsibilities it should **not** have:

* create datasets
* create models
* instantiate optimizers
* training logic
* inference logic

---

## Experiment directory layout

```
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
```

---

# Development Philosophy

Proceed bottom-up.

Configuration first.

Then data.

Then models.

Then training.

Then evaluation.

Avoid writing large classes.

Prefer many small helper functions with focused responsibilities.

---

# Remaining Work

## Phase 3 — Data Pipeline

Planned order:

1. Base dataset

```
src/datasets/dataset.py
```

2. Module datasets

```
src/modules/authenticity/dataset.py
src/modules/denomination/dataset.py
src/modules/quality/dataset.py
```

3. Transform pipeline

```
src/augmentation/transforms.py
```

4. DataLoader factory

```
src/datasets/dataloader.py
```

5. Dataset unit tests

---

## Phase 4

Model registry

Model builders

---

## Phase 5

Trainer

Callbacks

Checkpointing

Early stopping

Mixed precision

---

## Phase 6

Evaluation

Metrics

Grad-CAM

Error analysis

Inference

Dashboard integration

---

# Current Progress

Phase 1 — Project Structure

Completed.

Phase 2 — Configuration & Experiment Management

Completed.

Next milestone:

Begin implementing the dataset layer (`src/datasets/dataset.py`) as the foundation of the training pipeline.
