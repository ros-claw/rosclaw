"""Fail-closed backend registry."""

from __future__ import annotations

from collections.abc import Callable

from rosclaw.sandbox.backends.base import SandboxBackend


class SandboxBackendRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., SandboxBackend]] = {}

    def register(self, name: str, factory: Callable[..., SandboxBackend]) -> None:
        normalized = name.strip().lower()
        if not normalized or not callable(factory):
            raise ValueError("A backend name and callable factory are required")
        self._factories[normalized] = factory

    def create(self, name: str, **kwargs: object) -> SandboxBackend:
        normalized = name.strip().lower()
        factory = self._factories.get(normalized)
        if factory is None:
            raise KeyError(f"SANDBOX_BACKEND_NOT_REGISTERED: {normalized}")
        return factory(**kwargs)

    def available(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


__all__ = ["SandboxBackendRegistry"]
