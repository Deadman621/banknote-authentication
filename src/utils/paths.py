from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

CONFIG_DIR = ROOT / "configs"

CHECKPOINT_DIR = ROOT / "checkpoints"
EXPERIMENT_DIR = ROOT / "experiments"
LOG_DIR = ROOT / "logs"
REPORT_DIR = ROOT / "reports"

DOCS_DIR = ROOT / "docs"

NOTEBOOK_DIR = ROOT / "notebooks"

SRC_DIR = ROOT / "src"


def create_directories() -> None:
    """
    Create directories used during training if they don't exist.
    """
    directories = [
        PROCESSED_DATA_DIR,
        CHECKPOINT_DIR,
        EXPERIMENT_DIR,
        LOG_DIR,
        REPORT_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)