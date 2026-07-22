"""Adapter registry: task_id/event-shape -> adapter (数据库优化v3 §4.1)."""

from __future__ import annotations

from typing import Any

from .base import TaskDistillationAdapter

_ADAPTERS: list[TaskDistillationAdapter] = []


def register_adapter(adapter: TaskDistillationAdapter) -> None:
    _ADAPTERS.append(adapter)


def adapter_for(context: Any, events: list[dict[str, Any]]) -> TaskDistillationAdapter | None:
    """Pick the first adapter whose task_ids match the session context.

    Falls back to event-shape sniffing when the context carries no
    task_id (e.g. older sessions): an adapter may declare
    ``matches_events(events)``; otherwise task_id match only.
    """
    task_id = getattr(context, "task_id", None)
    for adapter in _ADAPTERS:
        if task_id and task_id in adapter.task_ids:
            return adapter
    for adapter in _ADAPTERS:
        matcher = getattr(adapter, "matches_events", None)
        if callable(matcher) and matcher(events):
            return adapter
    return None


def _register_builtin() -> None:
    from .rh56_rps import Rh56RpsAdapter

    register_adapter(Rh56RpsAdapter())


_register_builtin()
