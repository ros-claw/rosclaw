"""BodyResolver — unified API for reading body state."""

from __future__ import annotations

import contextlib
import json
from datetime import UTC
from pathlib import Path
from typing import Any

import yaml

from rosclaw.body.compatibility import SkillCompatibilityChecker, SkillCompatibilityStore
from rosclaw.body.compiler import EffectiveBodyCompiler
from rosclaw.body.diff import BodyDiffer
from rosclaw.body.notes import MaintenanceLog
from rosclaw.body.patch_validator import apply_nested_update
from rosclaw.body.references import RosclawURI
from rosclaw.body.registry import BodyRegistryError, BodyRegistryManager
from rosclaw.body.renderer import EmbodimentRenderer
from rosclaw.body.schema import (
    BodyYaml,
    CalibrationYaml,
    EffectiveBody,
    EurdfProfile,
    MaintenanceEvent,
    SkillCompatibilityReport,
    SkillManifest,
)
from rosclaw.eurdf.registry import RobotRegistry
from rosclaw.firstboot.workspace import get_rosclaw_home


class BodyResolver:
    """Resolve rosclaw://body/... URIs and load effective body state."""

    def __init__(self, workspace: Path | None = None, body_id: str | None = None):
        self.workspace = workspace or get_rosclaw_home()
        self.registry_manager = BodyRegistryManager(self.workspace)
        self.body_id, self.body_dir, self.is_legacy_single_body = self._resolve_body(body_id)
        self.compiler = EffectiveBodyCompiler()
        self.differ = BodyDiffer()
        self.renderer = EmbodimentRenderer()

    def _resolve_body(self, body_id: str | None) -> tuple[str, Path, bool]:
        """Resolve body_id and body_dir from registry, legacy layout, or defaults."""
        registry = self.registry_manager.load()
        has_registry = self.registry_manager.registry_path.exists()
        has_legacy = (self.workspace / "body").exists() and (self.workspace / "body" / "body.yaml").exists()

        if body_id:
            normalized = body_id.strip().lower()
            entry = registry.bodies.get(normalized)
            if entry is None:
                # Fallback: legacy single-body workspace whose body.yaml id matches.
                if has_legacy:
                    body_yaml_path = self.workspace / "body" / "body.yaml"
                    try:
                        body_data = yaml.safe_load(body_yaml_path.read_text(encoding="utf-8"))
                        legacy_id = (body_data.get("body_instance", {}).get("id") or "").strip().lower()
                        if legacy_id == normalized:
                            return body_id, self.workspace / "body", True
                    except Exception:
                        pass
                raise BodyRegistryError(f"Body not found: {body_id}")
            if entry.path == "body":
                return entry.body_id, self.workspace / "body", not has_registry
            return entry.body_id, self.workspace / entry.path, False

        if has_registry and registry.bodies:
            current_id = self.registry_manager.get_current_body_id()
            entry = registry.bodies[current_id]
            if entry.path == "body":
                return entry.body_id, self.workspace / "body", False
            return entry.body_id, self.workspace / entry.path, False

        # Legacy single-body workspace or empty default.
        return "default", self.workspace / "body", has_legacy or not has_registry

    @classmethod
    def list_workspace_bodies(cls, workspace: Path) -> list[Any]:
        """Return registry entries for bodies in the workspace."""
        manager = BodyRegistryManager(workspace)
        return manager.list_bodies()

    @property
    def body_yaml_path(self) -> Path:
        return self.body_dir / "body.yaml"

    @property
    def calibration_yaml_path(self) -> Path:
        return self.body_dir / "calibration.yaml"

    @property
    def maintenance_log_path(self) -> Path:
        return self.body_dir / "maintenance.log"

    @property
    def effective_body_path(self) -> Path:
        return self.body_dir / "refs" / "effective_body.json"

    @property
    def eurdf_profile_path(self) -> Path:
        return self.body_dir / "refs" / "eurdf.profile.yaml"

    @property
    def eurdf_lock_path(self) -> Path:
        return self.body_dir / "refs" / "eurdf.lock"

    @property
    def skill_compatibility_path(self) -> Path:
        return self.body_dir / "skill_compatibility.yaml"

    @property
    def embodiment_md_path(self) -> Path:
        return self.body_dir / "EMBODIMENT.md"

    @property
    def body_md_path(self) -> Path:
        return self.body_dir / "BODY.md"

    @property
    def snapshots_dir(self) -> Path:
        return self.body_dir / "snapshots"

    @property
    def generated_dir(self) -> Path:
        return self.body_dir / "refs" / "generated"

    def ensure_body_dir(self) -> None:
        self.body_dir.mkdir(parents=True, exist_ok=True)
        (self.body_dir / "refs").mkdir(exist_ok=True)
        self.snapshots_dir.mkdir(exist_ok=True)
        self.generated_dir.mkdir(exist_ok=True)

    def is_linked(self) -> bool:
        return self.body_yaml_path.exists()

    def resolve(self, uri: str) -> Any:
        """Resolve a rosclaw:// URI to a loaded object."""
        parsed = RosclawURI(uri)
        if parsed.resource_type == "body":
            if parsed.path == "current":
                if parsed.qualifier == "effective":
                    return self.get_effective_body()
                if parsed.qualifier == "calibration":
                    return self.get_calibration()
                if parsed.qualifier == "maintenance":
                    return self.get_maintenance_events()
                if parsed.qualifier == "capabilities":
                    return self.get_effective_body().capabilities
                return self.get_current_body_yaml()
            # Named body instance: in P0 only current is supported
            return self.get_current_body_yaml()
        if parsed.resource_type == "eurdf":
            return self.get_current_eurdf_profile()
        raise ValueError(f"Unsupported rosclaw URI resource type: {parsed.resource_type}")

    def get_current_body_yaml(self) -> BodyYaml:
        if not self.body_yaml_path.exists():
            raise BodyNotLinkedError("No body linked. Run: rosclaw body link-eurdf <profile_id>")
        with open(self.body_yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return BodyYaml.from_dict(data)

    def get_current_eurdf_profile(self) -> EurdfProfile:
        if self.eurdf_profile_path.exists():
            return EurdfProfile.from_yaml(self.eurdf_profile_path)
        # Fallback: derive from RobotRegistry if lock exists
        if self.eurdf_lock_path.exists():
            with open(self.eurdf_lock_path, encoding="utf-8") as f:
                lock = yaml.safe_load(f) or {}
            profile_id = lock.get("profile_id")
            if profile_id:
                profile = RobotRegistry().get(profile_id)
                if profile:
                    return EurdfProfile.from_robot_complete_profile(profile)
        raise BodyNotLinkedError("No e-URDF profile linked.")

    def get_calibration(self) -> CalibrationYaml:
        if not self.calibration_yaml_path.exists():
            return CalibrationYaml()
        with open(self.calibration_yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return CalibrationYaml.from_dict(data)

    def get_maintenance_events(self) -> list[MaintenanceEvent]:
        return MaintenanceLog(self.maintenance_log_path).read_events()

    def get_effective_body(self, recompile_if_stale: bool = True) -> EffectiveBody:
        if not recompile_if_stale and self.effective_body_path.exists():
            with open(self.effective_body_path, encoding="utf-8") as f:
                data = json.load(f)
            return EffectiveBody.from_dict(data)
        return self.recompile_effective_body()

    def recompile_effective_body(self) -> EffectiveBody:
        body = self.get_current_body_yaml()
        eurdf = self.get_current_eurdf_profile()
        calibration = self.get_calibration()
        maintenance = self.get_maintenance_events()
        previous = None
        if self.effective_body_path.exists():
            try:
                with open(self.effective_body_path, encoding="utf-8") as f:
                    previous = EffectiveBody.from_dict(json.load(f))
            except Exception:
                previous = None
        effective = self.compiler.compile(eurdf, body, calibration, maintenance, previous=previous)
        self.effective_body_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.effective_body_path, "w", encoding="utf-8") as f:
            json.dump(effective.to_dict(), f, indent=2, default=str)
        return effective

    def get_effective_body_hash(self) -> str:
        return self.get_effective_body().effective_body_hash

    def get_skill_compatibility(self) -> SkillCompatibilityReport:
        if not self.skill_compatibility_path.exists():
            return SkillCompatibilityReport(
                body_instance_id=self.get_current_body_yaml().body_instance.get("id", ""),
                effective_body_hash=self.get_effective_body_hash(),
            )
        with open(self.skill_compatibility_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return SkillCompatibilityReport.from_dict(data)

    def has_sensor(self, sensor_id: str) -> bool:
        return sensor_id in self.get_effective_body().sensors

    def get_sensor(self, sensor_id: str) -> dict[str, Any] | None:
        return self.get_effective_body().sensors.get(sensor_id)

    def get_provider_ref(self, component_id: str) -> str | None:
        components = self.get_current_body_yaml().installed_components
        for category in ("sensors", "actuators"):
            data = components.get(category, {}).get(component_id)
            if isinstance(data, dict):
                return data.get("provider_ref")
        return None

    def check_skill_compatibility(
        self,
        skill_id: str,
        skill_version: str | None = None,
        skill_manifest_path: Path | None = None,
    ) -> Any:
        """Check a single skill against the current effective body."""
        from rosclaw.body.compatibility import SkillCompatibilityChecker
        body = self.get_effective_body()
        manifest: SkillManifest | None
        if skill_manifest_path:
            manifest = SkillManifest.from_yaml(skill_manifest_path)
        else:
            manifest = self._load_skill_manifest(skill_id, skill_version)
        if manifest is None:
            from rosclaw.body.schema import SkillCompatibilityResult
            return SkillCompatibilityResult(
                skill_id=skill_id,
                skill_version=skill_version or "1.0.0",
                status="unknown",
                reason="Skill manifest not found",
            )
        return SkillCompatibilityChecker().check_one(manifest, body)

    def _load_skill_manifest(self, skill_id: str, skill_version: str | None = None) -> SkillManifest | None:
        """Locate skill manifest in workspace skills/ or builtin paths."""
        candidates = [
            self.workspace / "skills" / f"{skill_id}.skill.yaml",
            self.workspace / "skills" / skill_id / "skill.yaml",
            self.workspace / "skills" / f"{skill_id}.yaml",
        ]
        for path in candidates:
            if path.exists():
                return SkillManifest.from_yaml(path)
        # Builtin skill manifests shipped with rosclaw.
        builtin_dir = Path(__file__).parent.parent / "skill" / "builtins" / skill_id
        builtin_manifest = builtin_dir / "skill.yaml"
        if builtin_manifest.exists():
            return SkillManifest.from_yaml(builtin_manifest)
        return None

    def refresh_all_artifacts(
        self,
        skill_manifests: list[Path] | None = None,
        reason: str = "",
        no_body_alias: bool = False,
        affected_only: set[str] | None = None,
    ) -> tuple[EffectiveBody, SkillCompatibilityReport]:
        """Recompile effective body, run skill checks, render EMBODIMENT.md."""
        effective = self.recompile_effective_body()
        body = self.get_current_body_yaml()
        calibration = self.get_calibration()
        maintenance = self.get_maintenance_events()

        # Skill compatibility check
        checker = SkillCompatibilityChecker()
        if skill_manifests:
            manifests = [SkillManifest.from_yaml(p) for p in skill_manifests]
        else:
            manifests = self._discover_skill_manifests()
        if affected_only:
            existing = SkillCompatibilityStore(self.skill_compatibility_path).load()
            report = checker.check_incremental(existing, manifests, effective, affected_only)
        else:
            report = checker.check_all(manifests, effective)
        store = SkillCompatibilityStore(self.skill_compatibility_path)
        store.save(report)

        # Render agent view artifacts through the dedicated Agent View layer.
        from rosclaw.body.agent_view import BodyAgentViewRenderer

        agent_renderer = BodyAgentViewRenderer(workspace=self.workspace, body_id=self.body_id)
        agent_renderer.render_all(
            effective=effective,
            body=body,
            calibration=calibration,
            maintenance=maintenance,
            report=report,
            reason=reason,
        )

        return effective, report

    def _refresh_body_md_alias(self) -> None:
        """Create or refresh BODY.md as a pointer to EMBODIMENT.md."""
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        body_md_content = (
            "# BODY.md\n\n"
            "This file is an **alias** for `EMBODIMENT.md`.\n\n"
            "To read the canonical body context, open `EMBODIMENT.md` instead.\n\n"
            f"See: [{self.embodiment_md_path.name}]({self.embodiment_md_path.name})\n"
        )
        self.body_md_path.write_text(body_md_content, encoding="utf-8")

    def write_generated_summaries(
        self,
        effective: EffectiveBody,
        body: BodyYaml,
        calibration: CalibrationYaml,
        report: SkillCompatibilityReport,
        maintenance: list[MaintenanceEvent],
    ) -> None:
        """Write machine-readable summary files to generated/."""
        self.generated_dir.mkdir(parents=True, exist_ok=True)

        identity = body.get_identity()
        forbidden = [
            item.get("id", item.get("capability", "unknown"))
            for item in (effective.forbidden_capabilities or body.forbidden_capabilities or [])
        ]
        open_faults = [f.get("id", "unknown") for f in effective.known_faults if f.get("status") == "open"]

        body_summary = {
            "schema": "rosclaw.generated.body_summary.v1",
            "generated_at": _utc_now(),
            "robot_instance_id": effective.body_instance_id,
            "robot_model": identity.get("robot_model") or body.body_instance.get("robot_model", "unknown"),
            "eurdf_profile": body.model_ref.get("profile_id", "unknown"),
            "safety_status": body.get_safety_status(),
            "calibration_status": calibration.overall_status(),
            "capabilities": effective.capabilities,
            "forbidden": forbidden,
            "known_faults_open": open_faults,
            "effective_body_hash": effective.effective_body_hash,
            "generation": effective.generation,
        }
        (self.generated_dir / "body.summary.json").write_text(
            json.dumps(body_summary, indent=2, default=str), encoding="utf-8"
        )

        agent_summary = {
            "schema": "rosclaw.generated.embodiment_agent.v1",
            "generated_at": _utc_now(),
            "identity": identity,
            "capabilities": effective.capabilities,
            "forbidden": forbidden,
            "open_faults": open_faults,
            "agent_policy": {
                "physical_execution_requires_sandbox": True,
                "direct_real_robot_execution_allowed": False,
                "human_approval_required_for_high_risk": True,
            },
        }
        (self.generated_dir / "embodiment.agent.json").write_text(
            json.dumps(agent_summary, indent=2, default=str), encoding="utf-8"
        )

        safety_summary = {
            "schema": "rosclaw.generated.safety_summary.v1",
            "generated_at": _utc_now(),
            "safety_status": body.get_safety_status(),
            "global_limits": effective.safety.get("safety_limits") or effective.safety.get("global_limits") or {},
            "workspace_limits": effective.safety.get("workspace_boundaries") or effective.safety.get("workspace_limits") or [],
            "contact_limits": effective.safety.get("contact_limits") or [],
            "open_faults": open_faults,
            "calibration_status": calibration.overall_status(),
        }
        (self.generated_dir / "safety.summary.json").write_text(
            json.dumps(safety_summary, indent=2, default=str), encoding="utf-8"
        )

    def _discover_skill_manifests(self) -> list[SkillManifest]:
        skills_dir = self.workspace / "skills"
        manifests: list[SkillManifest] = []
        if skills_dir.exists():
            for path in skills_dir.rglob("*.skill.yaml"):
                with contextlib.suppress(Exception):
                    manifests.append(SkillManifest.from_yaml(path))
        # Always include builtin skill manifests shipped with rosclaw.
        builtin_dir = Path(__file__).parent.parent / "skill" / "builtins"
        if builtin_dir.exists():
            for path in builtin_dir.rglob("*/skill.yaml"):
                with contextlib.suppress(Exception):
                    manifests.append(SkillManifest.from_yaml(path))
        return manifests

    def update_body_yaml(self, patch: dict[str, Any]) -> BodyYaml:
        """Load body.yaml, apply patch dict, validate, save, return new body."""
        body = self.get_current_body_yaml()
        data = body.to_dict()
        for key, value in patch.items():
            apply_nested_update(data, key, value)
        # Update timestamps
        data.setdefault("body_instance", {})["updated_at"] = _utc_now()
        new_body = BodyYaml.from_dict(data)
        with open(self.body_yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(new_body.to_dict(), f, sort_keys=False, allow_unicode=True)
        return new_body

    def create_snapshot(self, effective: EffectiveBody) -> Path:
        from datetime import datetime
        ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S.%f")
        snap_path = self.snapshots_dir / f"body-{ts}.yaml"
        fingerprint_path = self.snapshots_dir / f"body-{ts}.fingerprint"
        with open(snap_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(effective.to_dict(), f, sort_keys=False, allow_unicode=True)
        fingerprint_path.write_text(effective.effective_body_hash, encoding="utf-8")
        return snap_path


class BodyNotLinkedError(RuntimeError):
    """Raised when body operations are requested before link-eurdf."""


def _utc_now() -> str:
    from datetime import datetime
    return datetime.now(UTC).isoformat()
