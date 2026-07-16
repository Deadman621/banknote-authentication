import torch
from torch import Tensor

def move_batch_to_device(images: Tensor, labels: Tensor, device: torch.device) -> tuple[Tensor, Tensor]:
    return images.to(device), labels.to(device)