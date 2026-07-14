# Project Architecture

## Overview

This project implements an end-to-end computer vision pipeline for automated banknote analysis. The architecture is designed around a shared processing pipeline so that multiple deep learning models can be evaluated under identical conditions.

The project supports:

* Denomination recognition
* Authenticity / anomaly detection
* Damage and quality assessment
* Explainable AI (Grad-CAM)
* Interactive Streamlit dashboard

---

# System Architecture

```text
                    Public Datasets
                           │
                           ▼
                   Dataset Preparation
                           │
                           ▼
                 Preprocessing Pipeline
                           │
                           ▼
                Data Augmentation Pipeline
                           │
                           ▼
                PyTorch Dataset & DataLoader
                           │
                           ▼
                   Deep Learning Model
         ┌──────────┬──────────┬──────────┬──────────┬──────────┐
         │          │          │          │          │
         ▼          ▼          ▼          ▼          ▼
        CNN     ResNet50  EfficientNet  MobileNetV3   ViT
         └──────────┴──────────┴──────────┴──────────┴──────────┘
                           │
                           ▼
                    Shared Trainer
                           │
                           ▼
                 Checkpoint & Logging
                           │
                           ▼
                      Evaluation
                           │
                           ▼
                     Explainability
                     (Grad-CAM)
                           │
                           ▼
                 Streamlit Dashboard
```

---

# Repository Layers

The repository is divided into three logical layers.

## Layer 1 — Data

Responsible for loading and preparing data.

```
data/
src/datasets/
src/preprocessing/
src/augmentation/
```

Responsibilities:

* Load datasets
* Clean images
* Segment banknotes
* Apply augmentations
* Create DataLoaders

---

## Layer 2 — Models

Responsible for defining neural network architectures.

```
src/models/
```

Contains:

* CNN
* ResNet50
* EfficientNet-B3
* MobileNetV3
* Vision Transformer

Each model is implemented independently but follows a common interface.

---

## Layer 3 — Shared Pipeline

Responsible for all operations independent of the model architecture.

```
src/training/
src/evaluation/
src/explainability/
src/dashboard/
src/utils/
```

Responsibilities:

* Training loop
* Validation
* Metrics
* Checkpointing
* Explainability
* Dashboard integration

---

# Training Pipeline

Each training script follows the same sequence:

```
Load configuration
        │
        ▼
Load dataset
        │
        ▼
Preprocess images
        │
        ▼
Apply augmentation
        │
        ▼
Instantiate model
        │
        ▼
Initialize Trainer
        │
        ▼
Train
        │
        ▼
Validate
        │
        ▼
Save checkpoint
        │
        ▼
Evaluate
```

Only the **model implementation** changes between experiments.

---

# Configuration Management

Configuration is separated from implementation.

Shared project configuration:

```
src/configs/
```

Examples:

* dataset.yaml
* augmentation.yaml
* trainer.yaml

Model-specific configuration:

```
src/configs/models/
```

Examples:

* cnn.yaml
* resnet50.yaml
* efficientnet_b3.yaml
* mobilenet_v3.yaml
* vit.yaml

This separation allows fair comparisons while keeping each model configurable.

---

# Team Responsibilities

Each team member owns one model.

| Member   | Ownership                        |
| -------- | -------------------------------- |
| Member 1 | CNN                              |
| Member 2 | ResNet50                         |
| Member 3 | EfficientNet-B3                  |
| Member 4 | MobileNetV3                      |
| Member 5 | Vision Transformer & Integration |

Each owner is responsible for:

* Model implementation
* Model configuration
* Training launcher
* Experiment results

Shared modules are maintained collaboratively.

---

# Experiment Outputs

Each model stores artifacts independently.

```
checkpoints/
    cnn/
    resnet50/
    efficientnet_b3/
    mobilenet_v3/
    vit/

experiments/
    cnn/
    resnet50/
    efficientnet_b3/
    mobilenet_v3/
    vit/
```

Typical outputs include:

* Best checkpoint
* Final checkpoint
* Training history
* Confusion matrix
* Classification report
* Metrics
* Grad-CAM examples

---

# Development Workflow

1. Clone the repository.
2. Open the project in VS Code.
3. Reopen in the Dev Container.
4. Build the Docker image (first run only).
5. Develop or train the assigned model.
6. Save checkpoints and experiment outputs in the model-specific directories.
7. Merge code changes through Pull Requests after review.

---

# Design Principles

* **Separation of concerns:** Data processing, models, training, evaluation, and visualization are independent modules.
* **Reproducibility:** Configurations are stored separately from code, and experiments use consistent settings where comparisons require fairness.
* **Modularity:** Adding a new model requires only a new model implementation, a configuration file, and a training launcher.
* **Collaboration:** Model ownership minimizes merge conflicts while shared infrastructure ensures consistent behavior.
* **Transparency:** The pipeline is explicit and easy to follow, making it suitable for academic evaluation and future extension.

---

# Future Extensions

The architecture is designed so that additional models or deployment targets can be added with minimal changes, such as:

* Support for additional currencies
* REST API deployment
* Mobile inference
* Currency sorting machine integration
* Distributed or cloud-based training

```
```
