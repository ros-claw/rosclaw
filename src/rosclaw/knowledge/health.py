"""Health snapshots safe for Runtime and Dashboard display."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ComponentHealth:
    name: str
    mode: str
    status: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KnowledgeHealth:
    status: str
    mode: str
    know: ComponentHealth
    how: ComponentHealth
    memory_boundary: str = "isolated"
    advisory_only: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def probe_component(name: str, mode: str, client: Any) -> ComponentHealth:
    try:
        detail = client.health()
        status = str(detail.get("status", "unknown"))
    except Exception as exc:  # noqa: BLE001 - health must never break Core
        status = "unavailable"
        detail = {"error": type(exc).__name__}
    return ComponentHealth(name=name, mode=mode, status=status, detail=detail)


def combine_health(mode: str, know_client: Any, how_client: Any) -> KnowledgeHealth:
    know = probe_component("know", mode, know_client)
    how = probe_component("how", mode, how_client)
    statuses = {know.status, how.status}
    if statuses <= {"ok"}:
        status = "ok"
    elif statuses <= {"disabled"}:
        status = "disabled"
    else:
        status = "degraded"
    return KnowledgeHealth(status=status, mode=mode, know=know, how=how)
