"""Minimal entry-point discovery with deterministic, recoverable failure isolation."""

from __future__ import annotations

import importlib.metadata
import logging
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("rosclaw.extension_discovery")
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True)
class EntryPointDiscoveryReport:
    group: str
    loaded: tuple[str, ...]
    errors: tuple[str, ...]
    disabled: bool = False


def discover_entry_point_objects(
    *,
    group: str,
    disable_env: str,
    register: Callable[[Any], str],
) -> EntryPointDiscoveryReport:
    """Load isolated extension objects and register each validated object.

    Only the loaded object is passed to ``register``. The discovery layer does
    not provide a runtime, driver, executor, transport, or actuator handle.
    One broken optional package is reported without hiding healthy packages.
    """

    if os.environ.get(disable_env, "").strip().lower() in _TRUE_VALUES:
        return EntryPointDiscoveryReport(group=group, loaded=(), errors=(), disabled=True)
    try:
        entry_points = importlib.metadata.entry_points()
        selected: Iterable[Any]
        if hasattr(entry_points, "select"):
            selected = entry_points.select(group=group)
        else:  # pragma: no cover - compatibility with older metadata providers
            selected = entry_points.get(group, ())
    except Exception as exc:
        message = f"discovery failed: {exc}"
        logger.warning("ROSClaw extension group %s %s", group, message)
        return EntryPointDiscoveryReport(group=group, loaded=(), errors=(message,))

    loaded: list[str] = []
    errors: list[str] = []
    ordered = sorted(
        selected,
        key=lambda item: (
            str(getattr(item, "name", "")),
            str(getattr(item, "value", "")),
        ),
    )
    for entry_point in ordered:
        name = str(getattr(entry_point, "name", getattr(entry_point, "value", "unknown")))
        try:
            identifier = register(entry_point.load())
        except Exception as exc:
            message = f"{name}: {exc}"
            errors.append(message)
            logger.warning("Failed to register ROSClaw extension %s in %s", name, group)
            continue
        loaded.append(identifier)
    return EntryPointDiscoveryReport(group=group, loaded=tuple(loaded), errors=tuple(errors))


__all__ = ["EntryPointDiscoveryReport", "discover_entry_point_objects"]
