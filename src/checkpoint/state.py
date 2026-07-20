from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True, slots=True)
class CheckpointState:
    epoch: int
    global_step: int
    best_metric: Optional[float] = None