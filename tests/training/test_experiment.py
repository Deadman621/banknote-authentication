from src.core.experiment import (
    ConfigDict,
    ConfigValue,
    deep_merge,
    load_config_dict,
    parse_experiment_settings,
    parse_loss_config,
    parse_optimizer_config,
    require,
    require_mapping,
    parse_config,
    parse_dataset_config,
    parse_experiment_settings,
    parse_logging_config,
    parse_loss_config,
    parse_model_config,
    parse_optimizer_config,
    parse_output_config,
    parse_scheduler_config,
    parse_trainer_config,
    create_experiment_paths
)

from pathlib import Path
from typing import cast

def test_deep_merge() -> None:
    base: ConfigDict = {
        "trainer": {
            "epochs": 10,
            "device": "cuda",
        }
    }

    update: ConfigDict = {
        "trainer": {
            "epochs": 50,
        }
    }

    merged = deep_merge(base, update)

    assert merged == {
        "trainer": {
            "epochs": 50,
            "device": "cuda",
        }
    }

    trainer = cast(dict[str, ConfigValue], base["trainer"])
    assert trainer["epochs"] == 10

def test_load_config_dict():
    cfg = load_config_dict(module="denomination", model="resnet50")
    assert isinstance(cfg, dict)

    assert "dataset" in cfg
    assert "trainer" in cfg
    assert "model" in cfg

def test_require_str() -> None:
    cfg = {"name": "resnet50"}

    value = require(cfg, "name", str)

    assert value == "resnet50"


def test_require_wrong_type() -> None:
    cfg = {"epochs": "50"}

    try:
        require(cfg, "epochs", int)
        assert False
    except TypeError:
        pass

def test_require_mapping() -> None:
    cfg = {
        "trainer": {
            "epochs": 50,
        }
    }

    trainer = require_mapping(cfg, "trainer")

    assert trainer["epochs"] == 50

def test_parse_experiment_settings() -> None:
    cfg = {
        "name": "denomination_resnet50",
    }

    settings = parse_experiment_settings(cfg)

    assert settings.name == "denomination_resnet50"

def test_parse_loss_config() -> None:
    cfg = {
        "name": "cross_entropy",
    }

    loss = parse_loss_config(cfg)

    assert loss.name == "cross_entropy"

def test_parse_optimizer_config() -> None:
    cfg = {
        "name": "adamw",
        "lr": 1e-3,
        "weight_decay": 1e-4,
    }

    optimizer = parse_optimizer_config(cfg)

    assert optimizer.name == "adamw"
    assert optimizer.lr == 1e-3
    assert optimizer.weight_decay == 1e-4

def test_parse_logging_config() -> None:
    cfg = {
        "level": "INFO",
    }

    logging = parse_logging_config(cfg)

    assert logging.level == "INFO"

def test_parse_output_config() -> None:
    cfg = {
        "save_dir": "experiments",
    }

    output = parse_output_config(cfg)

    assert output.save_dir == Path("experiments")

def test_parse_scheduler_config() -> None:
    cfg = {
        "name": "cosine",
        "params": {
            "T_max": 100,
        },
    }

    scheduler = parse_scheduler_config(cfg)

    assert scheduler.name == "cosine"
    assert scheduler.params["T_max"] == 100

def test_parse_model_config() -> None:
    cfg = {
        "name": "resnet50",
        "pretrained": True,
        "dropout": 0.2,
    }

    model = parse_model_config(cfg)

    assert model.name == "resnet50"
    assert model.pretrained is True
    assert model.params["dropout"] == 0.2

def test_parse_dataset_config() -> None:
    cfg = {
        "root": "data/processed",
        "image_size": 224,
        "batch_size": 32,
        "num_workers": 8,
        "pin_memory": True,
        "persistent_workers": True,
        "num_classes": 4,
        "class_names": [
            "10",
            "20",
            "50",
            "100",
        ],
    }

    dataset = parse_dataset_config(cfg)

    assert dataset.root == Path("data/processed")
    assert dataset.image_size == 224
    assert dataset.batch_size == 32
    assert dataset.num_workers == 8
    assert dataset.pin_memory is True
    assert dataset.persistent_workers is True
    assert dataset.num_classes == 4
    assert dataset.class_names == (
        "10",
        "20",
        "50",
        "100",
    )

def test_parse_trainer_config() -> None:
    cfg = {
        "epochs": 100,
        "mixed_precision": True,
        "gradient_clip": 1.0,
        "early_stopping": {
            "patience": 10,
            "monitor": "val_loss",
        },
        "checkpoint": {
            "monitor": "val_loss",
            "mode": "min"
        },
    }

    trainer = parse_trainer_config(cfg)

    assert trainer.epochs == 100
    assert trainer.mixed_precision is True
    assert trainer.gradient_clip == 1.0

    assert trainer.early_stopping.patience == 10
    assert trainer.early_stopping.monitor == "val_loss"

    assert trainer.checkpoint.monitor == "val_loss"
    assert trainer.checkpoint.mode == "min"

def test_parse_config() -> None:
    cfg = load_config_dict(
        module="denomination",
        model="resnet50",
    )

    experiment = parse_config(cfg)

    assert experiment.seed >= 0
    assert experiment.device == "auto"

    assert experiment.model.name == "resnet50"
    assert experiment.dataset.batch_size > 0
    assert experiment.trainer.epochs > 0

def test_create_experiment_paths(tmp_path: Path) -> None:
    paths = create_experiment_paths(
        save_dir=tmp_path,
        module="denomination",
        model="resnet50",
        experiment_name="baseline",
    )

    assert paths.root.exists()

    assert paths.checkpoints.exists()
    assert paths.predictions.exists()
    assert paths.figures.exists()
    assert paths.gradcam.exists()

    assert paths.config_file.parent == paths.root
    assert paths.log_file.parent == paths.root
    assert paths.metrics_file.parent == paths.root