from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class CheckpointState:
    epoch: int
    global_step: int