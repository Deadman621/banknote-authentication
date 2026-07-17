from __future__ import annotations

from typing import Protocol

class SchedulerProtocol(Protocol):
    """
    Minimal scheduler interface.
    """

    def step(self) -> None:
        ...

    def state_dict(self) -> dict[str, object]:
        ...

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        ...