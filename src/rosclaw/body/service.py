"""Body instance lifecycle service.

Unifies `rosclaw body init`, `rosclaw body create`, and `rosclaw body link-eurdf`
so they all go through a single internal API.

The service is responsible for:
- resolving the target body directory (single-body legacy or multi-body registry)
- loading the e-URDF profile
- generating body.yaml, calibration.yaml, maintenance.log
- writing the e-URDF lock and normalized profile
- compiling the Effective Body and rendering EMBODIMENT.md / BODY.md / summaries
- updating the body registry when in registry mode

All CLI commands should delegate file generation here instead of reimplementing it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from rosclaw.body.compiler import compute_checksum
from rosclaw.body.notes import MaintenanceLog
from rosclaw.body.registry import BodyRegistryError, BodyRegistryManager
from rosclaw.body.resolver import BodyResolver
from rosclaw.body.schema import (
    BodyYaml,
    CalibrationYaml,
    EurdfProfile,
)
from rosclaw.eurdf.registry import RobotRegistry
from rosclaw.eurdf.zoo_client import EurdfZooClient, EurdfZooClientError


@dataclass
class BodyCreateResult:
    """Result of a successful create_or_init call."""

    body_id: str
    profile_id: str
    profile_version: str
    workspace: Path
    body_dir: Path
    eurdf_uri: str
    checksum: str
    effective_body_hash: str | None
    created_files: list[Path]


def _resolve_profile_alias(profile_id: str) -> str:
    """Map common user-facing profile IDs to registry names."""
    aliases = {
        "unitree-g1": "g1",
        "unitree-go2": "unitree_go2",
        "franka-panda": "franka_panda",
        "fetch": "fetch_robot",
        "realsense-d405": "realsense_d405",
        "realsense_d405": "realsense_d405",
        "d405": "realsense_d405",
        "realsense-d435i": "realsense_d435i",
        "realsense_d435i": "realsense_d435i",
        "d435i": "realsense_d435i",
        "realsense-dual": "realsense_dual",
        "realsense_dual": "realsense_dual",
        "dual": "realsense_dual",
    }
    return aliases.get(profile_id, profile_id)


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp string."""
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def _default_installed_components(eurdf: EurdfProfile) -> dict[str, Any]:
    """Build default installed_components from e-URDF sensors/actuators."""
    sensors = {}
    for sensor in eurdf.sensors:
        name = sensor.get("name")
        if name:
            sensors[name] = {
                "installed": True,
                "status": "available",
                "provider_ref": None,
                "notes": [],
            }
    actuators = {}
    for actuator in eurdf.actuators:
        name = actuator.get("name")
        if name:
            actuators[name] = {"installed": True, "status": "available", "notes": []}
    return {"sensors": sensors, "actuators": actuators}


class BodyInstanceService:
    """Unified service for creating or re-initializing a body instance."""

    def __init__(self, workspace: Path | None = None):
        self.workspace = workspace or Path.home() / ".rosclaw"
        self.registry_manager = BodyRegistryManager(self.workspace)

    def create_or_init(
        self,
        robot: str,
        profile: str | None = None,
        name: str | None = None,
        nickname: str | None = None,
        mode: Literal["single", "registry"] = "single",
        version: str = "latest",
        from_eurdf: Path | None = None,
        from_zoo: bool = False,
        zoo_path: Path | None = None,
        force: bool = False,
        update_registry: bool = False,
        switch_active: bool = False,
        render_agent_view: bool = True,
    ) -> BodyCreateResult:
        """Create or re-initialize a body instance from an e-URDF profile.

        Args:
            robot: User-facing robot profile ID (e.g. ``unitree-g1`` or
                ``dexhands/inspire_hand/right``).
            profile: Optional alias override; ignored if ``robot`` is given.
            name: Body instance ID. Auto-generated if omitted.
            nickname: Human-readable nickname. Defaults to ``name``.
            mode: ``single`` for legacy/active body layout, ``registry`` for
                multi-body registry layout under ``bodies/<id>/``.
            version: e-URDF profile version. ``latest`` resolves to ``1.0.0``.
            from_eurdf: Optional path to an external e-URDF file.
            from_zoo: If True, resolve ``robot`` as a manifest-driven zoo asset
                ID (e.g. ``dexhands/inspire_hand/right``).
            zoo_path: Optional explicit e-URDF-Zoo robots directory.
            force: Allow overwriting an existing body link.
            update_registry: Write the body into ``body_registry.yaml``.
            switch_active: Set the new body as the active body pointer.
            render_agent_view: Render EMBODIMENT.md and generated summaries.

        Returns:
            ``BodyCreateResult`` with paths and hashes.

        Raises:
            BodyRegistryError: If the body already exists and ``force`` is False,
                or if the profile cannot be found.
        """
        profile_id = _resolve_profile_alias(robot or profile or "unknown")
        profile_version = version if version != "latest" else "1.0.0"

        if mode == "registry":
            body_id = (name or f"{profile_id.replace('/', '-')}-001").strip().lower()
            body_dir = self.workspace / "bodies" / body_id
            if update_registry:
                self.registry_manager.create_body(
                    body_id=body_id,
                    profile_id=robot,
                    nickname=nickname or body_id,
                    profile_version=profile_version,
                    force=force,
                )
            if switch_active:
                self.registry_manager.set_current_body_id(body_id)
        else:
            body_id = (name or f"body-{profile_id.replace('/', '-')}-001").strip().lower()
            body_dir = self.workspace / "body"

        # Load e-URDF profile.
        if from_eurdf is not None:
            raise NotImplementedError("External e-URDF file loading is not yet supported.")

        if from_zoo or "/" in profile_id:
            try:
                normalized = EurdfZooClient(zoo_path=zoo_path).get_eurdf_profile(
                    profile_id, version=profile_version
                )
            except EurdfZooClientError as exc:
                raise BodyRegistryError(str(exc)) from exc
            profile_source = "zoo"
        else:
            registry = RobotRegistry()
            profile_obj = registry.get(profile_id)
            if profile_obj is None:
                raise BodyRegistryError(f"e-URDF profile not found: {robot}")
            normalized = EurdfProfile.from_robot_complete_profile(profile_obj)
            profile_source = "builtin"

        now = _utc_now()

        body_dir.mkdir(parents=True, exist_ok=True)

        if mode == "single":
            # For single mode we must also manage the registry if it exists.
            if update_registry and self.registry_manager.registry_path.exists():
                try:
                    self.registry_manager.create_body(
                        body_id=body_id,
                        profile_id=robot,
                        nickname=nickname or body_id,
                        profile_version=profile_version,
                        force=force,
                    )
                except BodyRegistryError:
                    if not force:
                        raise
            if switch_active and self.registry_manager.registry_path.exists():
                self.registry_manager.set_current_body_id(body_id)

        resolver = BodyResolver(
            workspace=self.workspace, body_id=body_id if mode == "registry" else None
        )
        resolver.ensure_body_dir()

        if mode == "single" and resolver.is_linked() and not force:
            raise BodyRegistryError(
                f"Body already linked at {resolver.body_dir}. Use force=True to overwrite."
            )

        # Write normalized profile reference.
        with open(resolver.eurdf_profile_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(normalized.to_dict(), f, sort_keys=False, allow_unicode=True)

        checksum = compute_checksum(resolver.eurdf_profile_path)
        eurdf_uri = f"rosclaw://eurdf/{robot or profile_id}@{profile_version}"

        # Write lock file.
        lock = {
            "schema_version": "rosclaw.eurdf_lock.v1",
            "profile_id": profile_id,
            "profile_version": profile_version,
            "uri": eurdf_uri,
            "source": profile_source,
            "zoo_source": profile_source == "zoo",
            "checksum": checksum,
            "locked_at": now,
        }
        if zoo_path is not None and profile_source == "zoo":
            lock["zoo_path"] = str(zoo_path)
        with open(resolver.eurdf_lock_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(lock, f, sort_keys=False, allow_unicode=True)

        # Write body.yaml.
        body_yaml = BodyYaml(
            body_instance={
                "id": body_id,
                "nickname": nickname or body_id,
                "robot_model": robot or profile_id,
                "serial_number": "UNKNOWN",
                "owner": "local",
                "deployment_site": "lab",
                "created_at": now,
                "updated_at": now,
            },
            model_ref={
                "eurdf_uri": eurdf_uri,
                "profile_id": robot or profile_id,
                "profile_version": profile_version,
                "profile_checksum": checksum,
                "lock_file": "refs/eurdf.lock",
            },
            calibration={
                "file": "calibration.yaml",
                "checksum": "sha256:uninitialized",
                "last_calibrated_at": None,
                "status": "factory_default",
            },
            maintenance={
                "log_file": "maintenance.log",
                "last_event_at": None,
                "safety_relevant_open_items": [],
            },
            installed_components=_default_installed_components(normalized),
            capabilities={"enabled": [], "disabled": [], "degraded": []},
            prohibited_capabilities=[],
            forbidden_capabilities=normalized.forbidden_capabilities or [],
            safety_overrides={},
            agent_policy={
                "physical_execution_requires_sandbox": True,
                "direct_real_robot_execution_allowed": bool(
                    normalized.safety.get("environment", {}).get(
                        "real_robot_execution_allowed", False
                    )
                ),
                "human_approval_required_for_high_risk": True,
            },
            metadata=normalized.metadata or {},
            runtime_state={
                "battery_percent": None,
                "last_seen_at": None,
                "health": "unknown",
                "online": False,
            },
            fingerprint={
                "effective_body_hash": None,
                "last_compiled_at": None,
                "last_skill_check_at": None,
            },
            compatibility_summary={
                "compatible_skills": 0,
                "degraded_skills": 0,
                "blocked_skills": 0,
                "unknown_skills": 0,
            },
        )
        with open(resolver.body_yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(body_yaml.to_dict(), f, sort_keys=False, allow_unicode=True)

        # Write calibration.yaml.
        calibration = CalibrationYaml(
            body_instance_id=body_id,
            model_ref=eurdf_uri,
            validation={
                "status": "factory_default",
                "last_validated_at": None,
                "errors": [],
                "warnings": [],
            },
        )
        with open(resolver.calibration_yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(calibration.to_dict(), f, sort_keys=False, allow_unicode=True)

        # Write initial maintenance log.
        MaintenanceLog(resolver.maintenance_log_path).write_init_event(body_id, eurdf_uri)

        # Compile effective body and render agent view.
        effective_body_hash: str | None = None
        created_files = [
            resolver.eurdf_profile_path,
            resolver.eurdf_lock_path,
            resolver.body_yaml_path,
            resolver.calibration_yaml_path,
            resolver.maintenance_log_path,
        ]

        if render_agent_view:
            try:
                effective, report = resolver.refresh_all_artifacts(reason="create_or_init")
                effective_body_hash = effective.effective_body_hash
            except Exception as exc:
                from rosclaw.body.agent_view import BodyAgentViewRenderer

                effective = resolver.recompile_effective_body()
                effective_body_hash = effective.effective_body_hash
                agent_renderer = BodyAgentViewRenderer(
                    workspace=self.workspace, body_id=body_id if mode == "registry" else None
                )
                agent_renderer.render_all(
                    effective=effective,
                    reason=f"create_or_init fallback after {exc}",
                )

            resolver.create_snapshot(effective)

            created_files.extend(
                [
                    resolver.effective_body_path,
                    resolver.embodiment_md_path,
                    resolver.body_md_path,
                ]
            )
            if resolver.generated_dir.exists():
                created_files.extend(resolver.generated_dir.glob("*.json"))

        return BodyCreateResult(
            body_id=body_id,
            profile_id=profile_id,
            profile_version=profile_version,
            workspace=self.workspace,
            body_dir=body_dir,
            eurdf_uri=eurdf_uri,
            checksum=checksum,
            effective_body_hash=effective_body_hash,
            created_files=[Path(p) for p in created_files],
        )
