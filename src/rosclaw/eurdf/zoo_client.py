"""ROSClaw client for the new manifest-driven e-URDF-Zoo.

This client bridges the new ``e_urdf_zoo`` asset library (manifest-driven
bundles under ``robots/<category>/...``) with the existing ROSClaw runtime
that expects ``RobotCompleteProfile`` / ``EurdfProfile`` objects.

Resolution order:
1. Explicit local path passed by the caller.
2. ``~/.rosclaw/cache/e-urdf-zoo/<asset_id>/``
3. Project-root ``e-urdf-zoo`` checkout (or pip-installed package data).
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rosclaw.body.schema import EurdfProfile
from rosclaw.firstboot.workspace import get_rosclaw_home
from rosclaw.runtime.eurdf_loader import (
    RobotBenchmarkProfile,
    RobotCapabilityProfile,
    RobotCompleteProfile,
    RobotEmbodimentProfile,
    RobotSafetyProfile,
    RobotSemanticProfile,
    RobotSimulationProfile,
)

try:
    from e_urdf_zoo import AssetLoader, AssetSummary, EmbodimentAsset
    from e_urdf_zoo.schemas import (
        CapabilitiesSchema,
        ManifestSchema,
        SafetySchema,
        SandboxSchema,
        SemanticSchema,
    )

    E_URDF_ZOO_AVAILABLE = True
except Exception:  # pragma: no cover - degraded gracefully
    E_URDF_ZOO_AVAILABLE = False
    AssetLoader = None  # type: ignore[misc,assignment]
    EmbodimentAsset = None  # type: ignore[misc,assignment]
    ManifestSchema = None  # type: ignore[misc,assignment]


@dataclass
class ResolvedAssetSource:
    """Where an asset was resolved from."""

    asset_id: str
    version: str
    path: Path
    source: str  # explicit, cache, project_root, package


class EurdfZooClientError(Exception):
    """Raised when a zoo client operation fails."""


class EurdfZooClient:
    """Discover, resolve, pull, and convert manifest-driven e-URDF assets."""

    def __init__(
        self,
        zoo_path: Path | str | None = None,
        cache_dir: Path | str | None = None,
    ):
        if not E_URDF_ZOO_AVAILABLE:
            raise EurdfZooClientError(
                "e_urdf_zoo package is not available. Install e-urdf-zoo first."
            )

        self.zoo_path = Path(zoo_path) if zoo_path else None
        if cache_dir is None:
            cache_dir = get_rosclaw_home() / "cache" / "e-urdf-zoo"
        self.cache_dir = Path(cache_dir)
        self._loader: AssetLoader | None = None

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------
    def _get_loader(self, source_path: Path | None = None) -> AssetLoader:
        """Return an AssetLoader rooted at the requested or default path."""
        if source_path is not None:
            return AssetLoader(zoo_path=source_path)
        if self._loader is None:
            self._loader = AssetLoader(zoo_path=self.zoo_path if self.zoo_path else None)
        return self._loader

    def resolve(
        self,
        asset_id: str,
        version: str = "latest",
        source_path: Path | str | None = None,
    ) -> ResolvedAssetSource:
        """Resolve an asset ID to an existing directory.

        Resolution order:
        1. ``source_path`` if provided.
        2. ``~/.rosclaw/cache/e-urdf-zoo/<asset_id>/``
        3. Default zoo path (project-root or pip package).
        """
        if source_path is not None:
            path = Path(source_path)
            if not path.is_dir():
                raise EurdfZooClientError(f"Explicit source path not found: {path}")
            return ResolvedAssetSource(
                asset_id=asset_id, version=version, path=path, source="explicit"
            )

        # Cache
        cache_path = self._cache_path(asset_id)
        if cache_path.is_dir() and (cache_path / "manifest.yaml").exists():
            return ResolvedAssetSource(
                asset_id=asset_id, version=version, path=cache_path, source="cache"
            )

        # Default loader (project-root / pip package)
        loader = self._get_loader()
        try:
            asset = loader.load_asset(asset_id)
        except FileNotFoundError as exc:
            raise EurdfZooClientError(str(exc)) from exc

        return ResolvedAssetSource(
            asset_id=asset_id,
            version=version,
            path=asset.base_path,
            source="project_root" if self.zoo_path else "package",
        )

    def load(
        self,
        asset_id: str,
        version: str = "latest",
        source_path: Path | str | None = None,
    ) -> EmbodimentAsset:
        """Load a manifest asset bundle."""
        resolved = self.resolve(asset_id, version=version, source_path=source_path)
        return EmbodimentAsset(asset_id, resolved.path)

    def list_assets(
        self,
        category: str | None = None,
        source_path: Path | str | None = None,
    ) -> list[AssetSummary]:
        """List available assets, optionally filtered by category."""
        loader = self._get_loader(source_path if source_path else self.zoo_path)
        return loader.list_assets(category=category)

    def search_assets(
        self,
        query: str,
        source_path: Path | str | None = None,
    ) -> list[AssetSummary]:
        """Search asset IDs, names, and categories."""
        loader = self._get_loader(source_path if source_path else self.zoo_path)
        return loader.search_assets(query)

    def validate(
        self,
        asset_id: str,
        version: str = "latest",
        source_path: Path | str | None = None,
    ) -> dict[str, Any]:
        """Validate a manifest asset bundle and return a serializable report."""
        resolved = self.resolve(asset_id, version=version, source_path=source_path)
        loader = self._loader_for_resolved(asset_id, resolved)
        report = loader.validate_asset(asset_id)
        return report.to_dict()

    # ------------------------------------------------------------------
    # Pull / cache
    # ------------------------------------------------------------------
    def pull(
        self,
        asset_id: str,
        version: str = "latest",
        source_path: Path | str | None = None,
    ) -> Path:
        """Copy an asset bundle into the local cache.

        Returns the cached directory path.
        """
        resolved = self.resolve(asset_id, version=version, source_path=source_path)
        dest = self._cache_path(asset_id)

        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(resolved.path, dest)

        # Write a small provenance file so consumers know where it came from.
        provenance = {
            "asset_id": asset_id,
            "version": version,
            "source": resolved.source,
            "source_path": str(resolved.path),
        }
        (dest / ".rosclaw_source.yaml").write_text(
            yaml.safe_dump(provenance, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return dest

    def cache_list(self) -> list[dict[str, Any]]:
        """List assets currently in the local cache."""
        results: list[dict[str, Any]] = []
        if not self.cache_dir.exists():
            return results

        def _walk(current: Path, prefix: str) -> None:
            manifest = current / "manifest.yaml"
            if manifest.exists():
                name = current.name
                version = "unknown"
                provenance = current / ".rosclaw_source.yaml"
                if provenance.exists():
                    try:
                        prov = yaml.safe_load(provenance.read_text(encoding="utf-8")) or {}
                        version = prov.get("version", version)
                    except Exception:
                        pass
                try:
                    data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
                    name = data.get("asset", {}).get("name", name)
                except Exception:
                    pass
                results.append(
                    {
                        "asset_id": prefix,
                        "version": version,
                        "path": str(current),
                        "name": name,
                    }
                )
                return
            for child in sorted(current.iterdir()):
                if child.is_dir():
                    child_prefix = f"{prefix}/{child.name}" if prefix else child.name
                    _walk(child, child_prefix)

        _walk(self.cache_dir, "")
        return results

    def _cache_path(self, asset_id: str) -> Path:
        """Return the cache directory for an asset ID.

        Forward slashes in asset IDs create nested directories so the cached
        layout mirrors the source zoo layout.
        """
        return self.cache_dir / asset_id

    def _loader_for_resolved(self, asset_id: str, resolved: ResolvedAssetSource) -> AssetLoader:
        """Return an AssetLoader that can resolve ``asset_id`` to ``resolved.path``."""
        if resolved.source == "cache":
            return AssetLoader(zoo_path=self.cache_dir)
        parts = asset_id.split("/")
        root = resolved.path.parent if len(parts) == 1 else resolved.path.parents[len(parts) - 1]
        return AssetLoader(zoo_path=root)

    # ------------------------------------------------------------------
    # Conversion to ROSClaw profiles
    # ------------------------------------------------------------------
    def get_profile(
        self,
        asset_id: str,
        version: str = "latest",
        source_path: Path | str | None = None,
    ) -> RobotCompleteProfile:
        """Convert a manifest asset to a ``RobotCompleteProfile``."""
        asset = self.load(asset_id, version=version, source_path=source_path)
        if asset.manifest is None:
            raise EurdfZooClientError(f"Asset '{asset_id}' is not a manifest-driven bundle")
        manifest = asset.manifest
        safety = asset.safety
        capabilities = asset.capabilities
        semantic = asset.semantic
        sandbox = asset.sandbox

        embodiment = self._build_embodiment(asset, manifest)
        safety_profile = self._build_safety(asset_id, safety)
        capability_profile = self._build_capability(asset_id, capabilities)
        simulation_profile = self._build_simulation(asset_id, sandbox)
        semantic_profile = self._build_semantic(asset_id, semantic)
        benchmark_profile = RobotBenchmarkProfile(robot_id=asset_id)

        return RobotCompleteProfile(
            robot_id=asset_id,
            name=manifest.asset.name,
            vendor=manifest.asset.vendor,
            version=manifest.asset.version,
            description=manifest.asset.description,
            embodiment=embodiment,
            safety=safety_profile,
            capability=capability_profile,
            simulation=simulation_profile,
            semantic=semantic_profile,
            benchmark=benchmark_profile,
        )

    def get_eurdf_profile(
        self,
        asset_id: str,
        version: str = "latest",
        source_path: Path | str | None = None,
    ) -> EurdfProfile:
        """Convert a manifest asset directly to a normalized ``EurdfProfile``."""
        complete = self.get_profile(asset_id, version=version, source_path=source_path)
        return EurdfProfile.from_robot_complete_profile(complete)

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------
    def _build_embodiment(
        self, asset: EmbodimentAsset, manifest: ManifestSchema
    ) -> RobotEmbodimentProfile:
        """Build embodiment profile from URDF and manifest metadata."""
        urdf_path = asset.model_urdf
        links: list[dict[str, Any]] = []
        joints: list[dict[str, Any]] = []

        if urdf_path and urdf_path.exists():
            try:
                tree = ET.parse(urdf_path)
                root = tree.getroot()
                links = [_parse_urdf_link(el) for el in root.findall("link")]
                joints = [_parse_urdf_joint(el) for el in root.findall("joint")]
            except Exception as exc:  # pragma: no cover - defensive
                links = [{"name": f"parse_error:{exc}", "type": "error"}]
                joints = []

        dof = manifest.robot.dof
        if dof is None:
            dof = sum(1 for j in joints if j.get("type") not in {"fixed", "floating"})

        return RobotEmbodimentProfile(
            robot_id=asset.asset_id,
            name=manifest.asset.name,
            vendor=manifest.asset.vendor,
            version=manifest.asset.version,
            description=manifest.asset.description,
            dof=dof,
            links=links,
            joints=joints,
            sensors=[],
            actuators=[
                {"name": j["name"], "type": j.get("type", "revolute")}
                for j in joints
                if j.get("type") not in {"fixed", "floating"}
            ],
            metadata={
                "category": manifest.asset.category,
                "variant": manifest.asset.variant,
                "model": manifest.asset.model,
                "urdf_path": str(urdf_path) if urdf_path else None,
            },
        )

    def _build_safety(self, asset_id: str, safety: SafetySchema | None) -> RobotSafetyProfile:
        """Build safety profile from manifest safety.yaml."""
        if safety is None:
            return RobotSafetyProfile(
                robot_id=asset_id,
                safety_level="STRICT",
            )

        policy = safety.global_policy
        limits = safety.limits
        trajectory = safety.trajectory_policy

        safety_level = safety.safety_status.upper()
        if safety_level not in {"STRICT", "MODERATE", "RELAXED"}:
            safety_level = "STRICT" if not policy.real_robot_execution_allowed else "MODERATE"

        return RobotSafetyProfile(
            robot_id=asset_id,
            safety_level=safety_level,
            safety_limits={
                "max_joint_speed_scale": limits.max_joint_speed_scale,
                "max_joint_torque_scale": limits.max_joint_torque_scale,
                "max_position_step_scale": limits.max_position_step_scale,
                "joint_limit_margin_ratio": limits.joint_limit_margin_ratio,
                "require_joint_limit_margin": limits.require_joint_limit_margin,
            },
            joint_soft_limits={
                "margin_ratio": limits.joint_limit_margin_ratio,
            },
            pfl={
                "current_limit_required": policy.current_limit_required,
                "fault_monitor_required": policy.fault_monitor_required,
            },
            collision_detection={
                "require_self_collision_check": trajectory.require_self_collision_check,
            },
            emergency_stop={"enabled": True},
            workspace_boundaries={},
            interaction={
                "manual_enable_required_after_validation": policy.manual_enable_required_after_validation,
            },
            environment={
                "sandbox_required": policy.sandbox_required,
                "real_robot_execution_allowed": policy.real_robot_execution_allowed,
            },
        )

    def _build_capability(
        self, asset_id: str, capabilities: CapabilitiesSchema | None
    ) -> RobotCapabilityProfile:
        """Build capability profile from manifest capabilities.yaml."""
        caps: list[dict[str, Any]] = []
        forbidden: list[dict[str, Any]] = []
        skill_registry: dict[str, Any] = {}

        if capabilities is not None:
            for cap in capabilities.capabilities:
                caps.append(
                    {
                        "id": cap.id,
                        "name": cap.name,
                        "scope": cap.scope,
                        "risk": cap.risk,
                        "body_parts": cap.body_parts,
                        "sandbox_required": cap.sandbox_required,
                        "real_robot_execution_allowed": cap.real_robot_execution_allowed,
                        "required_runtime_monitors": cap.required_runtime_monitors,
                        "required_calibration": cap.required_calibration,
                        "safety_notes": cap.safety_notes,
                    }
                )
            forbidden = [
                {
                    "id": fc.id,
                    "description": fc.description,
                    "reason": fc.reason,
                    "severity": fc.severity,
                }
                for fc in capabilities.forbidden_capabilities
            ]
            skill_registry["forbidden_capabilities"] = forbidden

        return RobotCapabilityProfile(
            robot_id=asset_id,
            capabilities=caps,
            skill_registry=skill_registry,
            precondition_checks={
                "real_robot_execution_allowed": not bool(forbidden),
            },
        )

    def _build_simulation(
        self, asset_id: str, sandbox: SandboxSchema | None
    ) -> RobotSimulationProfile:
        """Build simulation profile from manifest sandbox.yaml."""
        backends: dict[str, Any] = {}
        if sandbox is not None and sandbox.engines:
            for engine, config in sandbox.engines.items():
                backends[engine] = {
                    "supported": config.supported,
                    "model_path": config.model_path,
                    "status": config.status,
                }
        if not backends:
            backends["mock"] = {"supported": True, "status": "fallback"}
        return RobotSimulationProfile(robot_id=asset_id, backends=backends)

    def _build_semantic(
        self, asset_id: str, semantic: SemanticSchema | None
    ) -> RobotSemanticProfile:
        """Build semantic profile from manifest semantic.yaml."""
        functional_regions: list[dict[str, Any]] = []
        semantic_tags: list[str] = []
        if semantic is not None:
            if semantic.frames:
                if semantic.frames.root:
                    functional_regions.append({"name": "root", "frame": semantic.frames.root})
                if semantic.frames.tcp:
                    functional_regions.append({"name": "tcp", "frame": semantic.frames.tcp})
                if semantic.frames.mounting:
                    functional_regions.append(
                        {"name": "mounting", "frame": semantic.frames.mounting}
                    )
            for name, group in semantic.groups.items():
                functional_regions.append(
                    {
                        "name": name,
                        "type": group.type,
                        "links": group.links,
                        "joints": group.joints,
                        "side": group.side,
                        "roles": group.roles,
                        "source": group.source,
                        "confidence": group.confidence,
                    }
                )
            semantic_tags = [f"category:{group.type}" for group in semantic.groups.values()]

        return RobotSemanticProfile(
            robot_id=asset_id,
            functional_regions=functional_regions,
            semantic_tags=sorted(set(semantic_tags)),
        )


# ------------------------------------------------------------------
# URDF helpers
# ------------------------------------------------------------------
def _parse_floats(text: str | None) -> list[float]:
    if not text:
        return []
    try:
        return [float(x) for x in text.strip().split()]
    except ValueError:
        return []


def _parse_urdf_link(element: ET.Element) -> dict[str, Any]:
    name = element.get("name", "unknown")
    mass = 0.0
    inertial = element.find("inertial")
    if inertial is not None:
        mass_el = inertial.find("mass")
        if mass_el is not None:
            try:
                mass = float(mass_el.get("value", "0"))
            except ValueError:
                mass = 0.0
    return {"name": name, "type": "link", "mass": mass}


def _parse_urdf_joint(element: ET.Element) -> dict[str, Any]:
    name = element.get("name", "unknown")
    jtype = element.get("type", "fixed")
    parent = child = None
    parent_el = element.find("parent")
    if parent_el is not None:
        parent = parent_el.get("link")
    child_el = element.find("child")
    if child_el is not None:
        child = child_el.get("link")

    axis = [1.0, 0.0, 0.0]
    axis_el = element.find("axis")
    if axis_el is not None:
        axis = _parse_floats(axis_el.get("xyz")) or axis

    limits: dict[str, Any] = {}
    limit_el = element.find("limit")
    if limit_el is not None:
        for key in ("lower", "upper", "effort", "velocity"):
            val = limit_el.get(key)
            if val is not None:
                try:
                    limits[key] = float(val)
                except ValueError:
                    limits[key] = val

    origin: dict[str, Any] = {}
    origin_el = element.find("origin")
    if origin_el is not None:
        xyz = _parse_floats(origin_el.get("xyz"))
        rpy = _parse_floats(origin_el.get("rpy"))
        if xyz:
            origin["xyz"] = xyz
        if rpy:
            origin["rpy"] = rpy

    return {
        "name": name,
        "type": jtype,
        "parent": parent,
        "child": child,
        "axis": axis,
        "limits": limits,
        "origin": origin,
    }
