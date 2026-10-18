"""Original installed-manifest comparison, never version or runtime guessing."""

import hashlib
import re
from xml.etree import ElementTree as ET

COMPONENTS = {
    "navigation": ("nav2_bringup", "nav2_controller", "nav2_planner", "nav2_bt_navigator"),
    "localization": ("nav2_amcl", "nav2_map_server"),
    "obstacle_guard": ("nav2_collision_monitor",),
    "coverage": ("opennav_coverage",),
}


def review_component_compatibility(model, policies, original_manifests):
    """Compare pinned original XML and explicit operator requirements.

    An inventory entry proves only package presence. Matching a declared exact
    version and distro proves only this comparison, never ABI, launch, BT plugin
    availability or successful execution. Original file/source authentication
    remains an independent owner/SDK operation.
    """
    if model.compute_snapshot_hash() != model.snapshot_hash:
        raise ValueError("unchanged sealed original model required")
    if type(policies) is not list or not 1 <= len(policies) <= 32:
        raise ValueError("1..32 explicit reusable component policies required")
    if type(original_manifests) is not dict or len(original_manifests) > 32:
        raise ValueError("bounded original manifest mapping required")
    if any(type(p) is not dict or type(p.get("package")) is not str for p in policies):
        raise ValueError("explicit literal package policies required")
    if len({p.get("package") for p in policies}) != len(policies):
        raise ValueError("unique explicit package policies required")
    inventory = model.observations.get("package_inventory")
    rows = []
    for policy in policies:
        if type(policy) is not dict or set(policy) != {
            "package",
            "supported_ros",
            "supported_distros",
            "required_version",
            "policy_source_sha256",
        }:
            raise ValueError("explicit package/version/distro and pinned policy source required")
        package = policy["package"]
        if type(package) is not str or not re.fullmatch(r"[a-z][a-z0-9_]{0,127}", package):
            raise ValueError("literal ROS package name required")
        for key in ("supported_ros", "supported_distros"):
            if (
                type(policy[key]) is not list
                or not 1 <= len(policy[key]) <= 16
                or any(type(v) is not str or not 1 <= len(v) <= 64 for v in policy[key])
            ):
                raise ValueError("bounded explicit component compatibility policy required")
        version = policy["required_version"]
        if version is not None and (
            type(version) is not str or not re.fullmatch(r"[0-9][A-Za-z0-9.+_-]{0,63}", version)
        ):
            raise ValueError("exact literal required version or UNKNOWN required")
        if type(policy["policy_source_sha256"]) is not str or not re.fullmatch(
            r"[0-9a-f]{64}", policy["policy_source_sha256"]
        ):
            raise ValueError("original compatibility policy SHA256 required")
        entry = {
            "package": package,
            "status": "UNKNOWN",
            "installed_version": None,
            "required_version": version,
            "manifest_sha256": None,
            "policy_source_sha256": policy["policy_source_sha256"],
            "runtime_compatibility": "NOT_VERIFIED",
            "source_authentication": "NOT_ESTABLISHED_BY_HASH",
        }
        if type(inventory) is not list:
            entry["reason"] = "original installed package inventory missing"
        elif package not in inventory:
            entry.update(
                status="MISSING_DEPENDENCY", reason="package not observed in original inventory"
            )
        elif not model.environment.get("ros_generation") or not model.environment.get("distro"):
            entry["reason"] = "original ROS generation and distro missing"
        elif (
            model.environment.get("ros_generation") not in policy["supported_ros"]
            or model.environment.get("distro") not in policy["supported_distros"]
        ):
            entry.update(
                status="UNSUPPORTED_DECLARED_ENVIRONMENT",
                reason="original ROS/distro outside pinned reusable policy",
            )
        elif package not in original_manifests:
            entry["reason"] = (
                "actual installed package.xml original required; package presence is not a version"
            )
        else:
            original = original_manifests[package]
            if type(original) is not dict or set(original) != {"xml", "sha256"}:
                raise ValueError("original manifest XML and SHA256 required")
            xml = original["xml"]
            if (
                type(xml) is not str
                or not 0 < len(xml.encode()) <= 65536
                or re.search(r"<!\s*(DOCTYPE|ENTITY)", xml, re.I)
            ):
                raise ValueError("bounded original XML without DTD/entities required")
            if hashlib.sha256(xml.encode()).hexdigest() != original["sha256"]:
                raise ValueError("installed manifest original SHA256 mismatch")
            try:
                root = ET.fromstring(xml)
            except ET.ParseError as exc:
                raise ValueError("invalid original package manifest") from exc
            names, versions = root.findall("name"), root.findall("version")
            if (
                root.tag != "package"
                or len(names) != 1
                or len(versions) != 1
                or names[0].text != package
            ):
                raise ValueError("original package identity/version must be unambiguous")
            actual = versions[0].text
            if type(actual) is not str or not re.fullmatch(r"[0-9][A-Za-z0-9.+_-]{0,63}", actual):
                raise ValueError("literal installed package version required")
            entry.update(installed_version=actual, manifest_sha256=original["sha256"])
            if version is None:
                entry["reason"] = (
                    "installed version observed but compatible version policy remains UNKNOWN"
                )
            elif version != actual:
                entry.update(
                    status="VERSION_MISMATCH",
                    reason="installed version differs from pinned reuse policy",
                )
            else:
                entry.update(
                    status="DECLARED_VERSION_AND_DISTRO_MATCH",
                    reason="exact declared comparison only; new independent World validation remains required",
                )
        rows.append(entry)
    return {
        "schema_version": "rosclaw.component_compatibility_review.v1",
        "snapshot_hash": model.snapshot_hash,
        "components": rows,
        "all_declared_requirements_match": all(
            r["status"] == "DECLARED_VERSION_AND_DISTRO_MATCH" for r in rows
        ),
        "runtime_compatibility": "NOT_VERIFIED",
        "execution_authority": "NONE",
    }
