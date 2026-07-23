from src.modules.authenticity import MODULE as AUTHENTICITY
from src.modules.denomination import MODULE as DENOMINATION
from src.modules.quality import MODULE as QUALITY

class ModuleRegistry:

    _modules = {
        "authenticity": AUTHENTICITY,
        "denomination": DENOMINATION,
        "quality": QUALITY
    }

    @classmethod
    def get(cls, name: str):
        return cls._modules[name]