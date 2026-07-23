from src.modules.base import ModuleDefinition
from .dataset import AuthenticityDataset

MODULE = ModuleDefinition(
    name="authenticity",
    dataset=AuthenticityDataset,
)