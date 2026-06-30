"""Effective Body Compiler — merges Physical DNA + instance state + calibration + maintenance."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rosclaw.body.safety import SafetyInvariantEngine
from rosclaw.body.schema import (
    BodyYaml,
    CalibrationYaml,
    EffectiveBody,
    EurdfProfile,
    MaintenanceEvent,
)


class EffectiveBodyCompiler:
    """Compile an Effective Body Model from all body-definition sources."""

    def compile(
        self,
        eurdf: EurdfProfile,
        body: BodyYaml,
        calibration: CalibrationYaml | None = None,
        maintenance_events: list[MaintenanceEvent] | None = None,
        runtime: dict[str, Any] | None = None,
        previous: EffectiveBody | None = None,
    ) -> EffectiveBody:
        """Compile effective body according to dependency priority."""
        calibration = calibration or CalibrationYaml()
        maintenance_events = maintenance_events or []
        runtime = runtime or {}

        body_id = body.body_instance.get("id") or f"body-{eurdf.profile_id}-default"
        eurdf_uri = (
            body.model_ref.get("eurdf_uri")
            or f"rosclaw://eurdf/{eurdf.profile_id}@{eurdf.profile_version}"
        )

        # 1. Start with e-URDF structural model
        joints = _index_by_name(eurdf.joints, "name")
        sensors = _index_by_name(eurdf.sensors, "name")
        actuators = _index_by_name(eurdf.actuators, "name")

        # 2. Merge instance component state
        installed = body.installed_components
        sensors = _merge_component_status(sensors, installed.get("sensors", {}))
        actuators = _merge_component_status(actuators, installed.get("actuators", {}))

        # 3. Merge calibration (numerical offsets only — no topology changes)
        if calibration.joint_offsets:
            for j_name, offset_data in calibration.joint_offsets.items():
                if j_name in joints:
                    joints[j_name]["calibration"] = offset_data
        if calibration.sensor_extrinsics:
            for s_name, extrinsic in calibration.sensor_extrinsics.items():
                if s_name in sensors:
                    sensors[s_name]["extrinsics"] = extrinsic

        # 4. Apply safety overrides (only tighten)
        safety = copy.deepcopy(eurdf.safety)
        safety = _apply_safety_overrides(safety, body.safety_overrides)

        # 5. Derive capabilities from e-URDF hints + instance state + prohibitions + safety invariants
        capabilities = self._derive_capabilities(eurdf, body, maintenance_events, calibration)

        # 6. Apply maintenance-derived constraints
        for event in maintenance_events:
            if event.type in ("incident", "safety") and event.severity in ("warning", "critical"):
                for affect in event.affects:
                    if affect in actuators:
                        actuators[affect]["status"] = "unavailable"
                    if affect in sensors:
                        sensors[affect]["status"] = "unavailable"

        # 7. Runtime availability overlay
        for s_name, sensor in sensors.items():
            if s_name in runtime:
                sensor["runtime"] = runtime[s_name]

        # Build source trace
        source_trace = {
            "eurdf": "refs/eurdf.profile.yaml",
            "body": "body.yaml",
            "calibration": "calibration.yaml",
            "calibration_status": calibration.overall_status(),
            "maintenance": "maintenance.log",
        }

        # Generation: increment from previous effective body if present.
        generation = self._next_generation(body, source_trace, previous)

        effective = EffectiveBody(
            body_instance_id=body_id,
            eurdf_uri=eurdf_uri,
            effective_body_hash="",  # computed below
            compiled_at=_utc_now(),
            schema_version="rosclaw.effective_body.v1",
            generation=generation,
            frames=copy.deepcopy(eurdf.frames),
            identity=copy.deepcopy(eurdf.identity),
            joints=joints,
            sensors=sensors,
            actuators=actuators,
            capabilities=capabilities,
            forbidden_capabilities=body.forbidden_capabilities or [],
            safety=safety,
            provider_interfaces=copy.deepcopy(eurdf.provider_interfaces),
            sandbox=copy.deepcopy(eurdf.sandbox),
            source_trace=source_trace,
            known_faults=self._derive_known_faults(body, maintenance_events),
            known_successes=body.known_successes or [],
            known_failures=body.known_failures or [],
        )
        effective.runtime_state = (
            copy.deepcopy(runtime) if runtime else copy.deepcopy(body.runtime_state)
        )
        effective.effective_body_hash = effective.compute_hash()
        return effective

    def _derive_capabilities(
        self,
        eurdf: EurdfProfile,
        body: BodyYaml,
        maintenance_events: list[MaintenanceEvent] | None,
        calibration: CalibrationYaml | None = None,
    ) -> dict[str, list[str]]:
        """Derive enabled/degraded/blocked capabilities."""
        enabled: set[str] = set()
        degraded: set[str] = set()
        blocked: set[str] = set()

        # Start from e-URDF capability hints
        for cap_list in eurdf.capability_hints.values():
            enabled.update(cap_list)

        # Instance-level explicit capability state
        body_caps = body.capabilities
        enabled.update(body_caps.get("enabled", []))
        degraded.update(body_caps.get("degraded", []))
        blocked.update(body_caps.get("disabled", []))

        # Prohibited capabilities become blocked
        for prohibited in body.prohibited_capabilities:
            cap = prohibited.get("capability")
            if cap:
                blocked.add(cap)
                enabled.discard(cap)
                degraded.discard(cap)

        # Sensor availability can block capabilities
        installed = body.installed_components
        sensor_status = {
            name: data.get("status", "available")
            for name, data in installed.get("sensors", {}).items()
        }
        if (
            sensor_status.get("head_rgb_camera") == "unavailable"
            or sensor_status.get("head_camera") == "unavailable"
        ):
            blocked.add("visual_navigation")
            blocked.add("scan_workspace")
            enabled.discard("visual_navigation")
            enabled.discard("scan_workspace")

        # Actuator availability can block capabilities
        actuator_status = {
            name: data.get("status", "available")
            for name, data in installed.get("actuators", {}).items()
        }
        if (
            actuator_status.get("right_arm_actuator_group") == "unavailable"
            or actuator_status.get("right_arm") == "unavailable"
        ):
            blocked.add("dual_arm_coordination")
            enabled.discard("dual_arm_coordination")

        # Maintenance events may degrade/block
        for event in maintenance_events or []:
            if event.type == "incident" and event.severity in ("warning", "critical"):
                for affect in event.affects:
                    if "arm" in affect:
                        degraded.add("reach")
                    if "camera" in affect or "vision" in affect:
                        blocked.add("visual_navigation")
                        enabled.discard("visual_navigation")

        # Calibration status can degrade precision skills and dexterous gestures
        cal_status = body.calibration.get("status", "uncalibrated")
        if cal_status != "validated":
            degraded.add("precision_grasp")
            for cap in list(enabled):
                if "gesture" in cap.lower():
                    degraded.add(cap)
                    enabled.discard(cap)

        # Experimental / real-robot-blocked assets keep read-only / perception
        # capabilities enabled; only motion/actuation capabilities are degraded
        # to simulation-only.
        real_allowed = eurdf.safety.get("environment", {}).get("real_robot_execution_allowed", True)
        if not real_allowed:
            read_only_caps = {
                "get_state",
                "read_state",
                "list_joints",
                "report_faults",
                # Generic sensor/perception capabilities remain enabled for
                # perception-only bodies (e.g. RealSense cameras).
                "rgb_camera",
                "depth_camera",
                "stereo_infrared",
                "imu",
                "lidar",
                "microphone",
                "speaker",
            }
            for cap in list(enabled):
                if cap.lower() in read_only_caps:
                    continue
                degraded.add(cap)
                enabled.discard(cap)

        base = {
            "enabled": sorted(enabled - blocked),
            "degraded": sorted(degraded),
            "blocked": sorted(blocked),
        }

        # Apply mandatory safety invariants (only restrict)
        engine = SafetyInvariantEngine()
        mods = engine.apply(body, maintenance_events, calibration, base)
        for cap in mods["disabled"]:
            blocked.add(cap)
            enabled.discard(cap)
            degraded.discard(cap)
        for cap in mods["degraded"]:
            if cap not in blocked:
                degraded.add(cap)

        return {
            "enabled": sorted(enabled - blocked),
            "degraded": sorted(degraded),
            "blocked": sorted(blocked),
        }

    def _next_generation(
        self,
        body: BodyYaml,
        source_trace: dict[str, Any],
        previous: EffectiveBody | None,
    ) -> int:
        """Increment generation whenever source inputs change."""
        prior = previous.generation if previous else 0
        return prior + 1

    def _derive_known_faults(
        self,
        body: BodyYaml,
        maintenance_events: list[MaintenanceEvent] | None,
    ) -> list[dict[str, Any]]:
        """Derive known faults from body state plus maintenance incidents.

        Later resolution/repair events close matching fault IDs.
        """
        faults: dict[str, dict[str, Any]] = {}
        if body.known_faults:
            raw = body.known_faults.get("faults", body.known_faults)
            if isinstance(raw, list):
                for fault in raw:
                    fid = fault.get("id") or f"fault-{_utc_now()}"
                    faults[fid] = dict(fault, id=fid)

        for event in maintenance_events or []:
            if event.type in ("incident", "fault", "safety") and event.severity in (
                "warning",
                "critical",
                "high",
            ):
                fault_id = event.result.get("fault_id") or event.event_id or f"fault-{_utc_now()}"
                faults[fault_id] = {
                    "id": fault_id,
                    "component": event.component
                    or (event.affects[0] if event.affects else "unknown"),
                    "severity": event.severity,
                    "status": "open" if event.type in ("incident", "fault") else "resolved",
                    "summary": event.summary or event.message,
                    "time": event.time or event.ts,
                }
            if event.type in ("repair", "resolution") and event.result.get("fault_id"):
                fault_id = event.result["fault_id"]
                if fault_id in faults:
                    faults[fault_id]["status"] = "resolved"
                    if event.summary:
                        faults[fault_id]["summary"] = (
                            f"{faults[fault_id].get('summary', '')} → resolved: {event.summary}"
                        )

        return list(faults.values())


def _index_by_name(items: list[dict[str, Any]], key: str = "name") -> dict[str, dict[str, Any]]:
    """Convert a list of dicts into a name-indexed dict, preserving mutable copies."""
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        name = item.get(key)
        if name:
            result[name] = copy.deepcopy(item)
    return result


def _merge_component_status(
    eurdf_components: dict[str, dict[str, Any]],
    instance_components: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Overlay instance installation/availability status onto e-URDF components."""
    result = copy.deepcopy(eurdf_components)
    for name, data in instance_components.items():
        if name in result:
            if isinstance(data, dict):
                result[name]["installed"] = data.get("installed", True)
                result[name]["status"] = data.get("status", "available")
                if "provider_ref" in data:
                    result[name]["provider_ref"] = data["provider_ref"]
            else:
                result[name]["status"] = str(data)
    return result


def _apply_safety_overrides(
    eurdf_safety: dict[str, Any],
    body_overrides: dict[str, Any],
) -> dict[str, Any]:
    """Apply instance safety overrides, only tightening numeric limits."""
    safety = copy.deepcopy(eurdf_safety)
    for key, value in body_overrides.items():
        if key == "source":
            continue
        if isinstance(value, (int, float)) and isinstance(safety.get(key), (int, float)):
            # For limits where lower is safer, tighten; here we assume max limits
            safety[key] = min(safety[key], value)
        else:
            safety[key] = value
    return safety


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def compute_checksum(path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
