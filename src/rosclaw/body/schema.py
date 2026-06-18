"""Body module schemas — Physical DNA, instance ledger, and effective body model."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
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
    def from_robot_complete_profile(cls, profile: RobotCompleteProfile) -> "EurdfProfile":
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
    def from_dict(cls, data: dict[str, Any]) -> "EurdfProfile":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_yaml(cls, path: Path) -> "EurdfProfile":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)


# ── Body Instance Ledger ──

@dataclass
class BodyYaml:
    """Current robot instance ledger — what this specific body is right now."""

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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BodyYaml":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CalibrationYaml:
    """Calibration parameters for this body instance."""

    body_instance_id: str = ""
    model_ref: str = ""
    schema_version: str = "rosclaw.calibration.v1"
    joint_offsets: dict[str, Any] = field(default_factory=dict)
    sensor_extrinsics: dict[str, Any] = field(default_factory=dict)
    sensor_intrinsics: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibrationYaml":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Maintenance / Notes ──

@dataclass
class MaintenanceEvent:
    """Single structured maintenance/incident/note event (JSONL line)."""

    message: str
    ts: str = field(default_factory=lambda: _utc_now())
    type: str = "note"  # note, maintenance, calibration, incident, repair, inspection, safety
    severity: str = "info"  # info, warning, critical
    author: str = "human"
    body_instance_id: str = ""
    affects: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    requires_skill_recheck: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MaintenanceEvent":
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
    frames: dict[str, Any] = field(default_factory=dict)
    joints: dict[str, Any] = field(default_factory=dict)
    sensors: dict[str, Any] = field(default_factory=dict)
    actuators: dict[str, Any] = field(default_factory=dict)
    capabilities: dict[str, list[str]] = field(default_factory=dict)
    safety: dict[str, Any] = field(default_factory=dict)
    provider_interfaces: dict[str, Any] = field(default_factory=dict)
    sandbox: dict[str, Any] = field(default_factory=dict)
    source_trace: dict[str, str] = field(default_factory=dict)

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
        """Recompute SHA-256 hash of canonical representation."""
        return hashlib.sha256(self.to_canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EffectiveBody":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


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
    def from_dict(cls, data: dict[str, Any]) -> "SkillManifest":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: Path) -> "SkillManifest":
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
    def from_dict(cls, data: dict[str, Any]) -> "SkillCompatibilityReport":
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "changes": [c.to_dict() for c in self.changes],
            "summary": self.summary,
            "requires_skill_recheck": self.requires_skill_recheck,
            "affected_categories": self.affected_categories,
        }


# ── Helpers ──


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
