"""Permission policy analysis for ROSClaw Hub assets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rosclaw.hub.schema import AssetManifest, load_manifest


@dataclass
class PermissionCheckResult:
    """Result of checking an asset's permission requirements."""

    allowed: bool = True
    requires_human_approval: bool = False
    dangerous_permissions: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def block(self, reason: str) -> None:
        self.allowed = False
        self.issues.append(reason)


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def check_permissions(
    manifest: AssetManifest | str | Path,
    *,
    allow_real_robot: bool | None = None,
    allow_safety_config_changes: bool = False,
    allow_network_inbound: bool = False,
) -> PermissionCheckResult:
    """Check whether an asset's requested permissions are acceptable.

    Args:
        manifest: Loaded manifest, path, or path string.
        allow_real_robot: If False, reject assets that request real robot
            execution. If None, allow but flag for human approval.
        allow_safety_config_changes: If False, reject modifications to safety
            configuration.
        allow_network_inbound: If False, reject non-local inbound network
            access.

    Returns:
        :class:`PermissionCheckResult` describing allowed/denied status and
        any dangerous permissions that require explicit operator approval.
    """
    if isinstance(manifest, (str, Path)):
        manifest = load_manifest(manifest)

    result = PermissionCheckResult()
    perms = manifest.permissions
    hardware = perms.get("hardware", {})
    ros = perms.get("ros", {})
    fs = perms.get("filesystem", {})
    network = perms.get("network", {})
    modifies = perms.get("modifies", {})
    requires_approval = _as_list(perms.get("requires_human_approval", []))

    # Real robot execution.
    if hardware.get("real_robot_execution", False):
        if allow_real_robot is False:
            result.block("Asset requests real robot execution, which is denied by policy")
        else:
            result.requires_human_approval = True
            result.dangerous_permissions.append("hardware.real_robot_execution")

    # Safety configuration changes are always treated as dangerous.
    if modifies.get("safety_config", False):
        if not allow_safety_config_changes:
            result.block("Asset modifies safety configuration")
        result.dangerous_permissions.append("modifies.safety_config")

    # Core runtime configuration changes.
    if modifies.get("rosclaw_yaml", False):
        result.dangerous_permissions.append("modifies.rosclaw_yaml")

    # Inbound network access beyond localhost.
    inbound = _as_list(network.get("inbound", []))
    non_local_inbound = [h for h in inbound if h != "localhost"]
    if non_local_inbound:
        if not allow_network_inbound:
            result.block(f"Asset requests non-local inbound network access: {non_local_inbound}")
        result.dangerous_permissions.append("network.inbound")

    # Filesystem writes to sensitive areas.
    writes = _as_list(fs.get("write", []))
    for path in writes:
        path_str = str(path).lower()
        if "safety" in path_str or "rosclaw.yaml" in path_str:
            result.dangerous_permissions.append(f"filesystem.write:{path}")

    # ROS topics / services that imply motion.
    motion_topics = {"/cmd_vel", "/joint_command", "/position_command"}
    ros_writes = set(_as_list(ros.get("topics_write", [])))
    requested_motion = ros_writes & motion_topics
    if requested_motion:
        result.dangerous_permissions.append(f"ros.topics_write:{requested_motion}")

    # Human approval list.
    if requires_approval:
        result.requires_human_approval = True
        for item in requires_approval:
            label = f"requires_human_approval:{item}"
            if label not in result.dangerous_permissions:
                result.dangerous_permissions.append(label)

    return result
