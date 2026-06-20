"""Runtime subsystem health map for ROSClaw v1.0 Sprint 1.

Provides a unified view of the eight subsystems required by the Sprint 1
runtime skeleton::

    runtime, event_bus, seekdb, registry, sandbox, provider, memory, practice

The checks are intentionally lightweight (import/liveness only) so that
``rosclaw status`` and ``rosclaw doctor`` stay fast and do not need a running
runtime process.
"""

from __future__ import annotations

import importlib
from typing import Any

Subsystem = tuple[str, str, str]

SUBSYSTEMS: list[Subsystem] = [
    ("runtime", "rosclaw.core.runtime", "Runtime"),
    ("event_bus", "rosclaw.core.event_bus", "EventBus"),
    ("seekdb", "rosclaw.memory.seekdb_client", "SeekDBClient"),
    ("registry", "rosclaw.runtime.eurdf_loader", "RobotRegistry"),
    ("sandbox", "rosclaw.sandbox.runtime_adapter", "SandboxRuntimeAdapter"),
    ("provider", "rosclaw.provider.core.registry", "ProviderRegistry"),
    ("memory", "rosclaw.memory.interface", "MemoryInterface"),
    ("practice", "rosclaw.practice.episode_recorder", "EpisodeRecorder"),
]


def _class_importable(module_name: str, attr: str) -> bool:
    try:
        mod = importlib.import_module(module_name)
        getattr(mod, attr)
        return True
    except Exception:
        return False


def _mujoco_available() -> bool:
    try:
        importlib.import_module("mujoco")
        return True
    except Exception:
        return False


def _provider_count() -> int | None:
    try:
        from rosclaw.provider.core.registry import ProviderRegistry

        reg = ProviderRegistry()
        providers = reg.list_providers()
        return len(providers)
    except Exception:
        return None


def _robot_registry_count() -> int | None:
    try:
        from rosclaw.runtime.eurdf_loader import RobotRegistry

        reg = RobotRegistry()
        return len(reg.list())
    except Exception:
        return None


def subsystem_health() -> dict[str, dict[str, Any]]:
    """Return a health map for each Sprint 1 subsystem.

    Status values:
      * ``healthy``   – module importable and ready.
      * ``loaded``    – registry importable (used for ``registry``).
      * ``disabled``  – module importable but an optional backend is missing
                        (e.g. MuJoCo for sandbox).
      * ``degraded``  – module cannot be imported or initialized.
    """
    health: dict[str, dict[str, Any]] = {}

    for name, module_name, attr in SUBSYSTEMS:
        available = _class_importable(module_name, attr)

        if name == "registry":
            count = _robot_registry_count() if available else None
            status = "loaded" if available else "degraded"
            health[name] = {
                "status": status,
                "available": available,
                "robots_registered": count if count is not None else 0,
            }
            continue

        if name == "provider":
            count = _provider_count() if available else None
            if not available:
                status = "degraded"
            elif count == 0:
                status = "disabled"
            else:
                status = "healthy"
            health[name] = {
                "status": status,
                "available": available,
                "providers_registered": count if count is not None else 0,
            }
            continue

        if name == "sandbox":
            if not available:
                status = "degraded"
            elif not _mujoco_available():
                status = "disabled"
            else:
                status = "healthy"
            health[name] = {
                "status": status,
                "available": available,
                "mujoco": _mujoco_available(),
            }
            continue

        status = "healthy" if available else "degraded"
        health[name] = {"status": status, "available": available}

    return health


def overall_status(health: dict[str, dict[str, Any]] | None = None) -> str:
    """Return ``HEALTHY`` unless any subsystem is ``degraded``."""
    if health is None:
        health = subsystem_health()

    degraded = [
        name
        for name, info in health.items()
        if info.get("status") == "degraded"
    ]
    return "HEALTHY" if not degraded else "DEGRADED"


__all__ = ["SUBSYSTEMS", "overall_status", "subsystem_health"]
