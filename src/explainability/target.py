from __future__ import annotations

from torch import nn
from torch.nn import Module


def resolve_target_layer(model: Module, target_layer: str | None = None) -> Module:

    if target_layer is not None:

        layer: Module = model

        for name in target_layer.split("."):
            if name.isdigit():
                if isinstance(
                    layer, (nn.Sequential, nn.ModuleList)):
                    layer = layer[int(name)]
                else:
                    raise TypeError(f"{name} cannot be indexed on {type(layer).__name__}")

            else:
                layer = getattr(layer, name)

        return layer


    # CNN fallback
    last_conv: Module | None = None

    for module in model.modules():

        if isinstance(module, nn.Conv2d):
            last_conv = module


    if last_conv is None:
        raise RuntimeError(
            "No Conv2d layer found."
        )

    return last_conv