"""Provider body binder — binds a robot provider to the current EffectiveBody."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from rosclaw.body.schema import EffectiveBody


@dataclass
class ProviderInterface:
    """Descriptor for a provider interface required or optional for a body."""

    name: str
    required: bool
    category: str = "unknown"  # state, command, sensor, actuator, telemetry
    status: str = "unknown"    # available, unavailable, degraded, unknown
    error: str | None = None
    topic: str = ""
    provider_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "required": self.required,
            "category": self.category,
            "status": self.status,
            "error": self.error,
            "topic": self.topic,
            "provider_ref": self.provider_ref,
            "metadata": self.metadata,
        }


@dataclass
class ProviderBodyDiagnosis:
    """Result of diagnosing a provider against the current body."""

    body_instance_id: str
    effective_body_hash: str
    interfaces: dict[str, dict[str, Any]]
    status: str = "unknown"  # nominal, degraded, blocked, unknown
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    summary: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "body_instance_id": self.body_instance_id,
            "effective_body_hash": self.effective_body_hash,
            "timestamp": self.timestamp,
            "status": self.status,
            "interfaces": self.interfaces,
            "summary": self.summary,
        }


class ProviderBodyBinder:
    """Binds a robot provider to the compiled Effective Body Model.

    The binder consumes ``EffectiveBody`` directly (never ``body.yaml``) so that
    provider diagnosis, required interface discovery, and health reporting all
    share the same body truth as the rest of ROSClaw.
    """

    def __init__(self, body: EffectiveBody):
        self._body = body
        self.effective_body_hash = body.effective_body_hash
        self.eurdf_uri = body.eurdf_uri
        self.body_instance_id = body.body_instance_id
        self._interfaces: dict[str, ProviderInterface] | None = None

    @classmethod
    def from_effective_body(cls, body: EffectiveBody) -> ProviderBodyBinder:
        return cls(body)

    def _component_status(self, interface_id: str) -> tuple[str, str]:
        """Infer interface status from effective body installed components."""
        sensors = self._body.sensors or {}
        actuators = self._body.actuators or {}

        if interface_id in sensors:
            status = sensors[interface_id].get("status", "unknown")
            return (status, "" if status == "available" else f"{interface_id} not available")
        if interface_id in actuators:
            status = actuators[interface_id].get("status", "unknown")
            return (status, "" if status == "available" else f"{interface_id} not available")

        # Interfaces such as joint_states / joint_trajectory do not map to a
        # single component. Fall back to overall body readiness.
        readiness = self._body.readiness or {}
        overall = readiness.get("status", "unknown")
        if overall == "blocked":
            return ("unavailable", "body readiness is blocked")
        if overall == "degraded":
            return ("degraded", "body readiness is degraded")
        return ("available", "")

    def _build_interfaces(self) -> dict[str, ProviderInterface]:
        """Build the full interface map from provider_interfaces + components."""
        interfaces: dict[str, ProviderInterface] = {}
        provider_interfaces = self._body.provider_interfaces or {}

        # Declared groups (state, command, sensor, actuator, telemetry)
        for group, group_def in provider_interfaces.items():
            if not isinstance(group_def, dict):
                continue
            for name in group_def.get("required", []):
                interfaces[str(name)] = ProviderInterface(
                    name=str(name), required=True, category=group
                )
            for name in group_def.get("optional", []):
                interfaces[str(name)] = ProviderInterface(
                    name=str(name), required=False, category=group
                )

        # Also expose every installed sensor/actuator as a provider interface
        # so diagnose() can report component-level health.
        for sensor_id, sensor in (self._body.sensors or {}).items():
            if sensor_id not in interfaces:
                interfaces[sensor_id] = ProviderInterface(
                    name=sensor_id,
                    required=False,
                    category="sensor",
                    provider_ref=sensor.get("provider_ref"),
                )
        for actuator_id, actuator in (self._body.actuators or {}).items():
            if actuator_id not in interfaces:
                interfaces[actuator_id] = ProviderInterface(
                    name=actuator_id,
                    required=False,
                    category="actuator",
                    provider_ref=actuator.get("provider_ref"),
                )

        # Resolve status for each interface.
        for iface in interfaces.values():
            status, error = self._component_status(iface.name)
            iface.status = status
            iface.error = error

        return interfaces

    def _ensure_interfaces(self) -> dict[str, ProviderInterface]:
        if self._interfaces is None:
            self._interfaces = self._build_interfaces()
        return self._interfaces

    def required_interfaces(self) -> list[ProviderInterface]:
        """Return interfaces that are required by the body's provider profile."""
        return [iface for iface in self._ensure_interfaces().values() if iface.required]

    def optional_interfaces(self) -> list[ProviderInterface]:
        """Return interfaces that are optional for the body's provider profile."""
        return [iface for iface in self._ensure_interfaces().values() if not iface.required]

    def diagnose(self, *, available: set[str] | None = None) -> ProviderBodyDiagnosis:
        """Diagnose provider interfaces against the current body.

        If ``available`` is provided, those interface names are treated as
        reported available by the runtime/provider; interfaces not in the set
        are reported unavailable. If ``available`` is ``None``, status is
        derived from the Effective Body installed-component state.

        Missing required interfaces make the diagnosis ``blocked``; missing
        optional interfaces make it ``degraded``.
        """
        interfaces = self._ensure_interfaces()
        runtime_reported = available is not None
        available = available or set()

        result: dict[str, dict[str, Any]] = {}
        counts: dict[str, int] = {"available": 0, "unavailable": 0, "degraded": 0, "unknown": 0}
        required_unavailable = 0

        for name, iface in interfaces.items():
            if runtime_reported:
                if name in available:
                    status = "available"
                    error = None
                else:
                    status = "unavailable"
                    error = "interface not reported available"
            elif name in available:
                status = "available"
                error = None
            else:
                status = iface.status
                error = iface.error

            counts[status] = counts.get(status, 0) + 1
            if iface.required and status in ("unavailable", "unknown") or iface.required and status == "degraded":
                required_unavailable += 1

            result[name] = {
                "required": iface.required,
                "status": status,
                "error": error,
                "category": iface.category,
                "topic": iface.topic,
                "provider_ref": iface.provider_ref,
            }

        if required_unavailable:
            diagnosis_status = "blocked"
        elif counts.get("degraded", 0) or counts.get("unavailable", 0):
            diagnosis_status = "degraded"
        elif counts.get("unknown", 0):
            diagnosis_status = "unknown"
        else:
            diagnosis_status = "nominal"

        return ProviderBodyDiagnosis(
            body_instance_id=self.body_instance_id,
            effective_body_hash=self.effective_body_hash,
            interfaces=result,
            status=diagnosis_status,
            summary=counts,
        )
