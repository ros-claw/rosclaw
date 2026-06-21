"""Body module schemas — Physical DNA, instance ledger, and effective body model."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from rosclaw.runtime.eurdf_loader import RobotCompleteProfile

# ── e-URDF Profile (normalized from RobotCompleteProfile) ──

@dataclass
class EurdfProfile:
    """Normalized e-URDF profile — model-level Physical DNA."""

    profile_id: str
    profile_version: str
    vendor: str
    model: str
    display_name: str
    description: str = ""
    schema_version: str = "eurdf.profile.v1"
    assets: dict[str, str] = field(default_factory=dict)
    identity: dict[str, str] = field(default_factory=dict)
    frames: dict[str, str] = field(default_factory=dict)
    joints: list[dict[str, Any]] = field(default_factory=list)
    sensors: list[dict[str, Any]] = field(default_factory=list)
    actuators: list[dict[str, Any]] = field(default_factory=list)
    capability_hints: dict[str, list[str]] = field(default_factory=dict)
    safety: dict[str, Any] = field(default_factory=dict)
    provider_interfaces: dict[str, Any] = field(default_factory=dict)
    sandbox: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_robot_complete_profile(cls, profile: RobotCompleteProfile) -> EurdfProfile:
        """Normalize existing RobotCompleteProfile into EurdfProfile."""
        embodiment = profile.embodiment
        capability = profile.capability
        simulation = profile.simulation
        safety = profile.safety

        capability_hints: dict[str, list[str]] = {}
        if capability.capabilities:
            capability_hints["all"] = sorted({
                c.get("name", "") for c in capability.capabilities if c.get("name")
            })

        provider_interfaces: dict[str, Any] = {
            "state": {"required": ["joint_states"], "optional": []},
            "command": {"required": ["joint_trajectory"], "optional": []},
        }

        sandbox: dict[str, Any] = {"compatible_engines": sorted(simulation.backends.keys())}
        if simulation.backends:
            sandbox["preferred_engine"] = next(iter(simulation.backends.keys()))

        return cls(
            profile_id=profile.robot_id,
            profile_version=str(profile.version),
            vendor=profile.vendor,
            model=profile.robot_id,
            display_name=profile.name,
            description=profile.description.strip(),
            assets={"urdf": "robot.urdf", "mjcf": "robot.mjcf.xml"},
            identity={
                "robot_class": _infer_robot_class(profile.robot_id, embodiment.dof, capability.capabilities),
            },
            frames={"root": "base_link", "world": "world"},
            joints=embodiment.joints,
            sensors=embodiment.sensors,
            actuators=embodiment.actuators,
            capability_hints=capability_hints,
            safety={
                "safety_level": safety.safety_level,
                "joint_soft_limits": safety.joint_soft_limits,
                "safety_limits": safety.safety_limits,
                "workspace_boundaries": safety.workspace_boundaries,
                "pfl": safety.pfl,
                "collision_detection": safety.collision_detection,
                "emergency_stop": safety.emergency_stop,
                "interaction": safety.interaction,
                "environment": safety.environment,
            },
            provider_interfaces=provider_interfaces,
            sandbox=sandbox,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EurdfProfile:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_yaml(cls, path: Path) -> EurdfProfile:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)


# ── Body Registry ──

@dataclass
class BodyRegistryEntry:
    """One registered body in a workspace registry."""

    body_id: str
    profile_id: str
    nickname: str = ""
    profile_version: str = "latest"
    created_at: str = field(default_factory=lambda: _utc_now())
    updated_at: str = field(default_factory=lambda: _utc_now())
    path: str = ""
    source: str = "builtin"
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BodyRegistryEntry:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class BodyRegistry:
    """Workspace-level body registry — tracks bodies and the active body."""

    schema: str = "rosclaw.body_registry.v1"
    current_body_id: str = "default"
    bodies: dict[str, BodyRegistryEntry] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "current_body_id": self.current_body_id,
            "bodies": {k: v.to_dict() for k, v in self.bodies.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BodyRegistry:
        bodies = {
            k: BodyRegistryEntry.from_dict(v) if isinstance(v, dict) else v
            for k, v in data.get("bodies", {}).items()
        }
        return cls(
            schema=data.get("schema", "rosclaw.body_registry.v1"),
            current_body_id=data.get("current_body_id", "default"),
            bodies=bodies,
        )


# ── Body Instance Ledger ──

@dataclass
class BodyYaml:
    """Current robot instance ledger — what this specific body is right now.

    This dataclass stores both the legacy flat fields (kept for backward
    compatibility) and the richer spec-aligned fields defined in
    rosclaw_body.md. New code should prefer the spec fields; legacy consumers
    can continue reading the flat fields.
    """

    # Legacy flat fields (preserved for backward compatibility)
    schema_version: str = "rosclaw.body.v1"
    body_instance: dict[str, Any] = field(default_factory=dict)
    model_ref: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    maintenance: dict[str, Any] = field(default_factory=dict)
    installed_components: dict[str, Any] = field(default_factory=dict)
    capabilities: dict[str, Any] = field(default_factory=dict)
    prohibited_capabilities: list[dict[str, Any]] = field(default_factory=list)
    safety_overrides: dict[str, Any] = field(default_factory=dict)
    runtime_state: dict[str, Any] = field(default_factory=dict)
    fingerprint: dict[str, Any] = field(default_factory=dict)
    compatibility_summary: dict[str, Any] = field(default_factory=dict)

    # Spec-aligned fields (rosclaw_body.md)
    schema: str = "rosclaw.body.v1"
    metadata: dict[str, Any] = field(default_factory=dict)
    identity: dict[str, Any] = field(default_factory=dict)
    eurdf: dict[str, Any] = field(default_factory=dict)
    body_structure: dict[str, Any] = field(default_factory=dict)
    installed_sensors: list[dict[str, Any]] = field(default_factory=list)
    installed_actuators: list[dict[str, Any]] = field(default_factory=list)
    installed_tools: list[dict[str, Any]] = field(default_factory=list)
    forbidden_capabilities: list[dict[str, Any]] = field(default_factory=list)
    safety: dict[str, Any] = field(default_factory=dict)
    known_faults: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    known_successes: list[dict[str, Any]] = field(default_factory=list)
    known_failures: list[dict[str, Any]] = field(default_factory=list)
    runtime_overlay: dict[str, Any] = field(default_factory=dict)
    agent_policy: dict[str, Any] = field(default_factory=dict)
    rendering: dict[str, Any] = field(default_factory=dict)
    mcp_bindings: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BodyYaml:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def get_identity(self) -> dict[str, Any]:
        """Return spec identity, falling back to legacy body_instance."""
        if self.identity:
            return self.identity
        return {
            "robot_instance_id": self.body_instance.get("id"),
            "robot_model": self.body_instance.get("robot_model"),
            "robot_vendor": None,
            "nickname": self.body_instance.get("nickname"),
            "serial_number": self.body_instance.get("serial_number"),
            "site": self.body_instance.get("deployment_site"),
            "operator": self.body_instance.get("owner"),
        }

    def get_capabilities_spec(self) -> dict[str, list[Any]]:
        """Return capabilities grouped as enabled/degraded/disabled.

        Prefers spec fields, falls back to legacy flat fields.
        """
        if self.capabilities:
            return {
                "enabled": list(self.capabilities.get("enabled", [])),
                "degraded": list(self.capabilities.get("degraded", [])),
                "disabled": list(self.capabilities.get("disabled", [])),
            }
        return {"enabled": [], "degraded": [], "disabled": []}

    def get_safety_status(self) -> str:
        """Return overall safety status: nominal|degraded|blocked|unknown."""
        if self.safety:
            return str(self.safety.get("status", "unknown"))
        return str(self.runtime_state.get("health", "unknown"))


@dataclass
class CalibrationYaml:
    """Calibration parameters for this body instance.

    Stores legacy flat fields plus the richer spec-aligned calibration schema.
    """

    # Legacy flat fields
    body_instance_id: str = ""
    model_ref: str = ""
    schema_version: str = "rosclaw.calibration.v1"
    joint_offsets: dict[str, Any] = field(default_factory=dict)
    sensor_extrinsics: dict[str, Any] = field(default_factory=dict)
    sensor_intrinsics: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)

    # Spec-aligned fields
    schema: str = "rosclaw.calibration.v1"
    metadata: dict[str, Any] = field(default_factory=dict)
    validity: dict[str, Any] = field(default_factory=dict)
    frames: dict[str, Any] = field(default_factory=dict)
    joints: dict[str, Any] = field(default_factory=dict)
    sensors: dict[str, Any] = field(default_factory=dict)
    time_sync: dict[str, Any] = field(default_factory=dict)
    calibration_history: list[dict[str, Any]] = field(default_factory=list)
    agent_policy: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationYaml:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def overall_status(self) -> str:
        """Return overall calibration status, preferring spec field."""
        if self.validity:
            return str(self.validity.get("overall_status", "unknown"))
        return str(self.validation.get("status", "uncalibrated"))


# ── Maintenance / Notes ──

@dataclass
class MaintenanceEvent:
    """Single structured maintenance/incident/note event (JSONL line).

    Maintains legacy fields (message, ts, author, severity, affects, tags,
    requires_skill_recheck) and adds spec-aligned fields (event_id, time,
    robot_model, component, summary, operator, before, after, result,
    requires_render).
    """

    message: str = ""
    ts: str = field(default_factory=lambda: _utc_now())
    type: str = "note"  # note, maintenance, calibration, incident, repair, inspection, safety, fault, capability_update, render, validation, init
    severity: str = "info"  # info, warning, critical
    author: str = "human"
    body_instance_id: str = ""
    affects: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    requires_skill_recheck: bool = False

    # Spec-aligned fields
    event_id: str = ""
    time: str = ""
    robot_model: str = ""
    component: str = ""
    summary: str = ""
    operator: str = ""
    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    requires_render: bool = True

    def __post_init__(self) -> None:
        if not self.event_id:
            self.event_id = f"evt-{_utc_now_compact()}-{uuid.uuid4().hex[:8]}"
        if not self.time:
            self.time = self.ts
        if not self.summary:
            self.summary = self.message
        if not self.operator:
            self.operator = self.author
        if not self.message:
            self.message = self.summary

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MaintenanceEvent:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ── Effective Body Model ──

@dataclass
class EffectiveBody:
    """Compiled effective body model — single source of truth for consumers."""

    body_instance_id: str
    eurdf_uri: str
    effective_body_hash: str
    compiled_at: str
    schema_version: str = "rosclaw.effective_body.v1"
    generation: int = 0
    frames: dict[str, Any] = field(default_factory=dict)
    identity: dict[str, Any] = field(default_factory=dict)
    joints: dict[str, Any] = field(default_factory=dict)
    sensors: dict[str, Any] = field(default_factory=dict)
    actuators: dict[str, Any] = field(default_factory=dict)
    capabilities: dict[str, list[str]] = field(default_factory=dict)
    forbidden_capabilities: list[dict[str, Any]] = field(default_factory=list)
    safety: dict[str, Any] = field(default_factory=dict)
    provider_interfaces: dict[str, Any] = field(default_factory=dict)
    sandbox: dict[str, Any] = field(default_factory=dict)
    source_trace: dict[str, str] = field(default_factory=dict)
    readiness: dict[str, Any] = field(default_factory=dict)
    known_faults: list[dict[str, Any]] = field(default_factory=list)
    known_successes: list[dict[str, Any]] = field(default_factory=list)
    known_failures: list[dict[str, Any]] = field(default_factory=list)
    runtime_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_canonical_json(self) -> str:
        """Deterministic JSON for hash computation."""
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    def compute_hash(self) -> str:
        """Recompute SHA-256 hash of canonical representation.

        Excludes compile-time metadata (the hash field itself, ``compiled_at``,
        and ``generation``) so the hash is stable for identical source inputs.
        """
        data = self.to_dict()
        data["effective_body_hash"] = ""
        data["compiled_at"] = ""
        data["generation"] = 0
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EffectiveBody:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ── Validation ──

@dataclass
class ValidationResult:
    """One validation check result."""

    check_id: str
    status: str  # pass, warn, fail, block
    message: str = ""
    category: str = "general"  # schema, safety, calibration, maintenance, render, checksum

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BodyValidationReport:
    """Aggregated body validation report."""

    result: str  # PASS, PASS_WITH_WARNINGS, FAIL, BLOCKED
    checks: list[ValidationResult] = field(default_factory=list)
    summary: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "result": self.result,
            "checks": [c.to_dict() for c in self.checks],
            "summary": self.summary,
        }


# ── Skill Manifest ──

@dataclass
class SkillManifest:
    """Skill manifest with requirements against a body."""

    skill_id: str
    skill_version: str = "1.0.0"
    display_name: str = ""
    schema_version: str = "rosclaw.skill.v1"
    requires: dict[str, Any] = field(default_factory=dict)
    degradation_policy: dict[str, bool] = field(default_factory=dict)
    runtime_policy: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillManifest:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def requirement_ids(self) -> set[str]:
        """Return the set of body component/capability IDs this skill depends on."""
        ids: set[str] = set()
        requires = self.requires

        for cap in requires.get("capabilities", {}).get("all_of", []):
            ids.add(cap)
        for sensor in requires.get("sensors", {}).get("all_of", []):
            sid = sensor.get("id") or sensor.get("name")
            if sid:
                ids.add(sid)
        for actuator in requires.get("actuators", {}).get("all_of", []):
            group = actuator.get("group")
            if group:
                ids.add(group)
        for frame in requires.get("frames", {}).get("all_of", []):
            ids.add(frame)
        if requires.get("calibration"):
            ids.add("calibration")
        if requires.get("safety", {}).get("max_base_speed_mps_at_least") is not None:
            ids.add("safety")
        robot_class = requires.get("robot_class")
        if robot_class:
            if isinstance(robot_class, list):
                ids.update(robot_class)
            else:
                ids.add(robot_class)
        eurdf = requires.get("eurdf")
        if eurdf:
            ids.add("eurdf")
        return ids

    @classmethod
    def from_yaml(cls, path: Path) -> SkillManifest:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)


@dataclass
class SkillCompatibilityResult:
    """Result of checking one skill against the effective body."""

    skill_id: str
    skill_version: str
    status: str  # compatible, degraded, blocked, unknown
    reason: str = ""
    missing_requirements: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_against: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SkillCompatibilityReport:
    """Full skill compatibility report for the current body."""

    body_instance_id: str
    effective_body_hash: str
    checked_at: str = field(default_factory=lambda: _utc_now())
    schema_version: str = "rosclaw.skill_compatibility.v1"
    skills: dict[str, SkillCompatibilityResult] = field(default_factory=dict)
    summary: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "body_instance_id": self.body_instance_id,
            "effective_body_hash": self.effective_body_hash,
            "checked_at": self.checked_at,
            "skills": {k: v.to_dict() for k, v in self.skills.items()},
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillCompatibilityReport:
        skills = {
            k: SkillCompatibilityResult(**v)
            for k, v in data.get("skills", {}).items()
        }
        return cls(
            body_instance_id=data.get("body_instance_id", ""),
            effective_body_hash=data.get("effective_body_hash", ""),
            checked_at=data.get("checked_at", _utc_now()),
            schema_version=data.get("schema_version", "rosclaw.skill_compatibility.v1"),
            skills=skills,
            summary=data.get("summary", {}),
        )


@dataclass
class FleetCompatibilityReport:
    """Aggregated skill compatibility report across all bodies in a workspace."""

    workspace: str
    checked_at: str = field(default_factory=lambda: _utc_now())
    schema_version: str = "rosclaw.fleet_compatibility.v1"
    per_body: dict[str, SkillCompatibilityReport] = field(default_factory=dict)
    fleet_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "workspace": self.workspace,
            "checked_at": self.checked_at,
            "per_body": {k: v.to_dict() for k, v in self.per_body.items()},
            "fleet_summary": self.fleet_summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FleetCompatibilityReport:
        per_body = {
            k: SkillCompatibilityReport.from_dict(v)
            for k, v in data.get("per_body", {}).items()
        }
        return cls(
            workspace=data.get("workspace", ""),
            checked_at=data.get("checked_at", _utc_now()),
            schema_version=data.get("schema_version", "rosclaw.fleet_compatibility.v1"),
            per_body=per_body,
            fleet_summary=data.get("fleet_summary", {}),
        )


# ── Diff ──

@dataclass
class BodyChange:
    """One detected body change."""

    path: str
    old: Any
    new: Any
    category: str  # structural, installed_component, actuator_status, calibration, safety, capability, runtime, note_only
    severity: str = "info"  # info, warning, critical
    requires_skill_recheck: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BodyDiff:
    """Diff result between two body representations."""

    changes: list[BodyChange] = field(default_factory=list)
    summary: dict[str, int] = field(default_factory=dict)
    requires_skill_recheck: bool = False
    affected_categories: list[str] = field(default_factory=list)
    affected_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "changes": [c.to_dict() for c in self.changes],
            "summary": self.summary,
            "requires_skill_recheck": self.requires_skill_recheck,
            "affected_categories": self.affected_categories,
            "affected_ids": self.affected_ids,
        }


# ── Helpers ──


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _utc_now_compact() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def _infer_robot_class(robot_id: str, dof: int, capabilities: list[dict[str, Any]]) -> str:
    """Best-effort robot_class inference from existing profile data."""
    rid = robot_id.lower()
    if "humanoid" in rid or any("walk" in str(c.get("name", "")).lower() for c in capabilities if dof >= 10):
        return "humanoid"
    if "quad" in rid or "go2" in rid or "dog" in rid:
        return "quadruped"
    if "drone" in rid or "crazyflie" in rid or "skydio" in rid:
        return "uav"
    if "mobile" in rid:
        return "mobile_base"
    if dof >= 4:
        return "manipulator"
    return "generic"
