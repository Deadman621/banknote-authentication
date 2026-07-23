from src.modules.base import ModuleDefinition
from .dataset import QualityDataset

MODULE = ModuleDefinition(
    name="quality",
    dataset=QualityDataset,
)