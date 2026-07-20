from __future__ import annotations

from typing import Any, Protocol

class SchedulerProtocol(Protocol):
    """
    Minimal scheduler interface.
    """

    def step(self, *args: Any, **kwargs: Any) -> None:
        ...

    def state_dict(self) -> dict[str, object]:
        ...

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        ...