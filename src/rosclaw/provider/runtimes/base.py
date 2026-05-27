"""ROSClaw Provider - RuntimeAdapter ABC.

A RuntimeAdapter is the bridge between a Provider and its execution backend.
"""

from abc import ABC, abstractmethod
from typing import Any


class RuntimeAdapter(ABC):
    """Abstract base for provider runtime adapters."""

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        self.name = name
        self.config = config or {}
        self._started = False

    @abstractmethod
    async def start(self) -> None:
        ...

    @abstractmethod
    async def stop(self) -> None:
        ...

    @abstractmethod
    async def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...

    def ensure_started(self) -> None:
        if not self._started:
            raise RuntimeError(
                f"RuntimeAdapter '{self.name}' is not started. Call start() first."
            )
