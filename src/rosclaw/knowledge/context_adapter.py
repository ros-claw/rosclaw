"""Build minimum, explicitly separated context for How advice."""

from __future__ import annotations

import platform
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any

from .contracts import (
    BodyContextV2,
    HowContextV2,
    MemoryEvidenceV2,
    RuntimeContextV2,
    SoftwareContextV2,
)

_FORBIDDEN_MEMORY_KEYS = {
    "trajectory",
    "frames",
    "video",
    "sensor_data",
    "time_series",
    "raw_content",
    "embedding",
}


def _strings(values: Iterable[Any], *, limit: int = 100) -> list[str]:
    return [str(value) for value in list(values)[:limit] if str(value)]


def build_body_context(
    *,
    robot_model: str | None,
    body: Any | None = None,
    safety_limits: Iterable[Any] = (),
) -> BodyContextV2:
    """Read descriptive metadata only; no handles, permits, or commands cross."""

    sensors = getattr(body, "sensors", ()) if body is not None else ()
    actuators = getattr(body, "actuators", ()) if body is not None else ()
    robot_type = getattr(body, "robot_type", None) if body is not None else None
    return BodyContextV2(
        robot_model=robot_model,
        robot_type=str(robot_type) if robot_type else None,
        sensors=_strings(sensors),
        actuators=_strings(actuators),
        safety_limits=_strings(safety_limits),
    )


def build_memory_evidence(items: Iterable[Mapping[str, Any]]) -> list[MemoryEvidenceV2]:
    """Accept summaries and opaque refs only; reject raw Practice/Memory payloads."""

    evidence = []
    for item in list(items)[:20]:
        forbidden = _FORBIDDEN_MEMORY_KEYS.intersection(item)
        if forbidden:
            raise ValueError(f"raw Memory/Practice fields are forbidden: {sorted(forbidden)}")
        created_at = item.get("created_at") or datetime.now(UTC)
        evidence.append(
            MemoryEvidenceV2(
                memory_id=str(item["memory_id"]),
                summary=str(item["summary"]),
                confidence=float(item.get("confidence", 0.0)),
                receipt_ref=str(item["receipt_ref"]) if item.get("receipt_ref") else None,
                practice_ref=str(item["practice_ref"]) if item.get("practice_ref") else None,
                created_at=created_at,
            )
        )
    return evidence


def build_how_context(
    *,
    task: str,
    robot_model: str | None = None,
    body: Any | None = None,
    safety_limits: Iterable[Any] = (),
    ros_distro: str | None = None,
    simulator: str | None = None,
    software_versions: Mapping[str, str] | None = None,
    current_stage: str | None = None,
    current_failure: str | None = None,
    error_log: str | None = None,
    verifier_signals: Iterable[Any] = (),
    memory_evidence: Iterable[Mapping[str, Any]] = (),
) -> HowContextV2:
    return HowContextV2(
        body=build_body_context(robot_model=robot_model, body=body, safety_limits=safety_limits),
        software=SoftwareContextV2(
            ros_distro=ros_distro,
            simulator=simulator,
            versions=dict(software_versions or {}),
            hardware_architecture=platform.machine() or None,
        ),
        runtime=RuntimeContextV2(
            task=task,
            current_stage=current_stage,
            current_failure=current_failure,
            error_log=error_log,
            verifier_signals=_strings(verifier_signals),
        ),
        memory_evidence=build_memory_evidence(memory_evidence),
    )
