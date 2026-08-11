"""Task and sandbox-backend registries for downstream SimForge extensions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from rosclaw.extension_discovery import (
    EntryPointDiscoveryReport,
    discover_entry_point_objects,
)
from rosclaw.sandbox.backends.base import SandboxBackend
from rosclaw.simforge.models import SimForgeTaskSpec

SIMFORGE_TASK_GROUP = "rosclaw.simforge.tasks"
SIMFORGE_BACKEND_GROUP = "rosclaw.simforge.backends"
_DISABLE_ENV = "ROSCLAW_DISABLE_SIMFORGE_EXTENSIONS"
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")


def _require_identifier(label: str, value: str) -> None:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} must be a normalized identifier")


def _unique_identifiers(values: tuple[str, ...], *, label: str) -> tuple[str, ...]:
    normalized = tuple(values)
    if not normalized or len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} must be non-empty and unique")
    for value in normalized:
        _require_identifier(label, value)
    return normalized


@runtime_checkable
class SimForgeTaskProvider(Protocol):
    """Describe domain tasks without importing them into Core."""

    provider_id: str
    task_ids: tuple[str, ...]

    def task_spec(self, task_id: str) -> SimForgeTaskSpec: ...


@runtime_checkable
class SimForgeBackendFactory(Protocol):
    """Deferred factory for a sandbox backend; discovery never instantiates it."""

    backend_id: str

    def create(self) -> SandboxBackend: ...


class SimForgeTaskRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, SimForgeTaskProvider] = {}
        self._task_providers: dict[str, SimForgeTaskProvider] = {}

    def register(self, provider: Any) -> str:
        if not isinstance(provider, SimForgeTaskProvider):
            raise TypeError("task entry point does not satisfy SimForgeTaskProvider")
        _require_identifier("provider_id", provider.provider_id)
        task_ids = _unique_identifiers(tuple(provider.task_ids), label="task_ids")
        if tuple(provider.task_ids) != task_ids:
            raise ValueError("provider task_ids must be a stable tuple")
        if provider.provider_id in self._providers:
            raise ValueError(f"duplicate SimForge task provider: {provider.provider_id}")
        duplicates = sorted(set(task_ids).intersection(self._task_providers))
        if duplicates:
            raise ValueError(f"duplicate SimForge task ids: {duplicates}")
        self._providers[provider.provider_id] = provider
        self._task_providers.update(dict.fromkeys(task_ids, provider))
        return provider.provider_id

    def task_spec(self, task_id: str) -> SimForgeTaskSpec:
        _require_identifier("task_id", task_id)
        try:
            provider = self._task_providers[task_id]
        except KeyError as exc:
            raise KeyError(f"unknown SimForge task: {task_id}") from exc
        spec = provider.task_spec(task_id)
        if not isinstance(spec, SimForgeTaskSpec):
            raise TypeError("task provider returned a non-SimForgeTaskSpec")
        if spec.task_id != task_id:
            raise ValueError("task provider returned a spec for a different task_id")
        return spec

    @property
    def provider_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._task_providers))


class SimForgeBackendRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, SimForgeBackendFactory] = {}

    def register(self, factory: Any) -> str:
        if not isinstance(factory, SimForgeBackendFactory):
            raise TypeError("backend entry point does not satisfy SimForgeBackendFactory")
        _require_identifier("backend_id", factory.backend_id)
        if factory.backend_id in self._factories:
            raise ValueError(f"duplicate SimForge backend: {factory.backend_id}")
        self._factories[factory.backend_id] = factory
        return factory.backend_id

    def create(self, backend_id: str) -> SandboxBackend:
        _require_identifier("backend_id", backend_id)
        try:
            factory = self._factories[backend_id]
        except KeyError as exc:
            raise KeyError(f"unknown SimForge backend: {backend_id}") from exc
        backend = factory.create()
        if not isinstance(backend, SandboxBackend):
            raise TypeError("backend factory returned a non-SandboxBackend")
        capabilities = backend.capabilities()
        if capabilities.name != backend_id:
            backend.close()
            raise ValueError("backend capability name does not match its registered id")
        return backend

    @property
    def backend_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


class SimForgeExtensionRegistry:
    """Aggregate registry whose discovery path has no execution authority."""

    def __init__(self) -> None:
        self.tasks = SimForgeTaskRegistry()
        self.backends = SimForgeBackendRegistry()

    def discover(self) -> SimForgeDiscoveryReport:
        tasks = discover_entry_point_objects(
            group=SIMFORGE_TASK_GROUP,
            disable_env=_DISABLE_ENV,
            register=self.tasks.register,
        )
        backends = discover_entry_point_objects(
            group=SIMFORGE_BACKEND_GROUP,
            disable_env=_DISABLE_ENV,
            register=self.backends.register,
        )
        return SimForgeDiscoveryReport(tasks=tasks, backends=backends)


@dataclass(frozen=True)
class SimForgeDiscoveryReport:
    tasks: EntryPointDiscoveryReport
    backends: EntryPointDiscoveryReport


__all__ = [
    "SIMFORGE_BACKEND_GROUP",
    "SIMFORGE_TASK_GROUP",
    "SimForgeBackendFactory",
    "SimForgeBackendRegistry",
    "SimForgeDiscoveryReport",
    "SimForgeExtensionRegistry",
    "SimForgeTaskProvider",
    "SimForgeTaskRegistry",
]
