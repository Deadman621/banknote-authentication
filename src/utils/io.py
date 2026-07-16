# src/utils/io.py

from pathlib import Path
from typing import Any

import json
import torch
import yaml

def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_yaml(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.safe_dump(data, f)

def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def save_checkpoint(state: dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path: Path, map_location="cpu"):
    return torch.load(path, map_location=map_location)