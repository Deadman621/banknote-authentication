# Banknote Authentication Project — Development Context

## Project Goal

Build a research-oriented computer vision pipeline for banknote analysis using public datasets.

The system should support:

* Banknote denomination recognition
* Authenticity / suspicious note classification
* Damage and fitness assessment
* Explainability using Grad-CAM
* Streamlit dashboard
* Comparison of multiple deep learning architectures

---

# Technology Stack

* Python 3.12+
* PyTorch
* OpenCV
* Albumentations
* scikit-learn
* Streamlit
* Plotly
* TensorBoard
* Jupyter
* Docker
* VS Code Dev Containers

---

# Development Environment

Development is performed inside a VS Code Dev Container.

Base image:

```
pytorch/pytorch:2.13.0-cuda13.2-cudnn9-devel
```

Project uses:

* Docker Compose
* Dev Containers
* GPU passthrough
* CUDA-enabled PyTorch

The Dev Container is the primary development environment.

---

# Repository Philosophy

The project is organized around the machine learning pipeline.

Shared components should exist only once.

Only the model implementation changes between experiments.

Training, evaluation, preprocessing, augmentation, explainability and dashboard are shared across all models.

---

# Final Folder Structure

The repository contains:

* data/
* notebooks/
* scripts/
* checkpoints/
* experiments/
* src/

Inside src:

* configs/
* datasets/
* preprocessing/
* augmentation/
* models/
* training/
* evaluation/
* explainability/
* dashboard/
* utils/

---

# Model Organization

Each architecture has:

```
src/models/
```

containing

* cnn.py
* resnet50.py
* efficientnet_b3.py
* mobilenet_v3.py
* vit.py

Each model has a separate configuration:

```
src/configs/models/
```

containing

* cnn.yaml
* resnet50.yaml
* efficientnet_b3.yaml
* mobilenet_v3.yaml
* vit.yaml

Implementation and configuration are intentionally separated.

---

# Shared Configurations

Project-wide settings live in

```
src/configs/
```

Examples:

* dataset.yaml
* augmentation.yaml
* trainer.yaml

These define shared behavior across all models to ensure fair comparison.

---

# Training Philosophy

Each architecture has its own launcher.

```
scripts/

train_cnn.py
train_resnet50.py
train_efficientnet_b3.py
train_mobilenet_v3.py
train_vit.py
```

Each launcher:

* loads the model
* loads the model configuration
* creates the shared Trainer
* starts training

All architectures use the same training pipeline.

---

# Shared Pipeline

Dataset

↓

Preprocessing

↓

Augmentation

↓

Model

↓

Trainer

↓

Evaluation

↓

Grad-CAM

↓

Checkpoint

↓

Dashboard

Only the Model stage changes.

---

# Team Ownership

One developer owns one model.

Example:

Person A

* cnn.py
* cnn.yaml
* train_cnn.py

Person B

* resnet50.py
* resnet50.yaml
* train_resnet50.py

etc.

This minimizes merge conflicts while keeping the pipeline obvious.

---

# Checkpoints

Checkpoints are organized by model.

```
checkpoints/

cnn/

resnet50/

efficientnet_b3/

mobilenet_v3/

vit/
```

Each launcher saves only into its own directory.

---

# Experiments

Experiments are separated similarly.

```
experiments/

cnn/

resnet50/

efficientnet_b3/

mobilenet_v3/

vit/
```

Metrics, logs and plots for each model are stored independently.

---

# Notebooks

Notebooks are for research only.

They are not part of the production pipeline.

Typical uses:

* Dataset exploration
* Augmentation testing
* Error analysis
* Grad-CAM visualization

Final training always runs from scripts/.

---

# Dev Workflow

1. Clone repository.
2. Open in VS Code.
3. Reopen in Dev Container.
4. VS Code builds the Docker image.
5. Container starts.
6. Extensions install automatically.
7. Development begins.

---

# Future Goals

After the baseline pipeline is complete:

* Implement CNN baseline
* Implement ResNet50
* Implement EfficientNet-B3
* Implement MobileNetV3
* Implement Vision Transformer
* Compare all models
* Add Grad-CAM
* Build Streamlit dashboard
* Prepare final presentation and report