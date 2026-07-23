from src.modules.base import ModuleDefinition

from .dataset import DenominationDataset


MODULE = ModuleDefinition(
    name="denomination",
    dataset=DenominationDataset,
)