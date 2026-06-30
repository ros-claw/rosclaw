"""Skill upload logic for the ROSClaw Hub."""

from __future__ import annotations

import os
import urllib.request
from datetime import UTC, datetime
from typing import Any

import semver

from rosclaw.skill.hub_client import SkillHubClient
from rosclaw.skill.models import SkillPackage


def _github_repo_url(pkg: SkillPackage) -> str:
    """Return the public-facing GitHub URL for this skill.

    For official catalog skills the canonical source lives in a subdir of
    ``ros-claw/skills``. The Hub only stores one ``github_repo_url`` field, so
    we point it at the actual subdir path to avoid 404 links.
    """
    if pkg.skill is None:
        raise RuntimeError("skill.yaml not loaded")

    identity = pkg.skill.identity
    base = (identity.git_repo or "").rstrip("/")
    if not base:
        return ""

    ref = identity.git_ref or "main"
    subdir = identity.source_subdir or ""
    if subdir:
        return f"{base}/tree/{ref}/{subdir}"
    return base


def _verify_repo_url(url: str) -> bool:
    """Do a lightweight HEAD check against a GitHub URL.

    Returns ``True`` if the URL responds with 2xx/3xx, ``False`` otherwise.
    """
    try:
        request = urllib.request.Request(
            url,
            method="HEAD",
            headers={"User-Agent": "Mozilla/5.0 (compatible; rosclaw-cli/1.0)"},
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            return 200 <= response.getcode() < 400
    except Exception:  # noqa: BLE001
        return False


def build_hub_payload(pkg: SkillPackage, visibility: str = "private") -> dict[str, Any]:
    if pkg.skill is None:
        raise RuntimeError("skill.yaml not loaded")

    readme_path = pkg.root / "README.md"
    long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

    author_name = ""
    author_url = ""
    if pkg.skill.metadata.authors:
        author_name = pkg.skill.metadata.authors[0].name
        author_url = pkg.skill.metadata.authors[0].url or ""

    compatible_robots: list[str] = []
    robot_types: list[str] = []
    if pkg.eurdf_compat:
        for robot in pkg.eurdf_compat.compatible_robots:
            compatible_robots.append(robot.robot)
            rt = _infer_robot_type(robot.robot)
            if rt and rt not in robot_types:
                robot_types.append(rt)

    # Dependencies from lock.
    lock_path = pkg.root / ".rosclaw" / "lock.yaml"
    dependencies: list[str] = []
    if lock_path.exists():
        import yaml

        lock = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
        dependencies = lock.get("dependencies", [])

    name = pkg.skill.identity.package_name or pkg.skill_id
    github_url = _github_repo_url(pkg)

    return {
        "name": name,
        "display_name": pkg.skill.metadata.display_name or pkg.skill.metadata.name,
        "description": pkg.skill.metadata.description,
        "long_description": long_description,
        "github_repo_url": github_url,
        "author_name": author_name,
        "author_url": author_url,
        "category": pkg.skill.metadata.category or "general",
        "robot_types": robot_types,
        "compatible_robots": compatible_robots,
        "tags": pkg.skill.metadata.tags,
        "version": pkg.skill.metadata.version,
        "dependencies": dependencies,
    }


def _infer_robot_type(robot_name: str) -> str | None:
    name = robot_name.lower()
    if "g1" in name or "humanoid" in name:
        return "humanoid"
    if "ur5" in name or "franka" in name or "panda" in name:
        return "manipulator"
    if "go2" in name or "a1" in name or "spot" in name or "quadruped" in name:
        return "legged"
    if "turtle" in name or "mobile" in name:
        return "mobile"
    if "drone" in name or "uav" in name:
        return "drone"
    return None


def upload_skill(
    pkg: SkillPackage,
    visibility: str = "private",
    hub_base_url: str = "https://www.rosclaw.io",
    api_key_env: str = "ROSCLAW_ADMIN_API_KEY",
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing API key in environment variable {api_key_env}. "
            "Metadata upload to rosclaw.io is reserved for administrators. "
            "To contribute a Skill without an admin key, use `rosclaw skill submit-catalog`."
        )

    payload = build_hub_payload(pkg, visibility=visibility)
    github_url = payload.get("github_repo_url", "")

    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "payload": payload,
            "receipt": None,
        }

    if not github_url:
        raise RuntimeError(
            "skill.yaml identity.git_repo is empty. "
            "Set a valid GitHub repository URL before uploading."
        )

    if not _verify_repo_url(github_url):
        raise RuntimeError(
            f"GitHub source is not reachable: {github_url}\n"
            "The Hub record would point to a 404 page. "
            "For official Skills, submit the source via `rosclaw skill submit-catalog`. "
            "For external/community Skills, ensure the repository exists and is public."
        )

    client = SkillHubClient(hub_base_url, api_key)
    response = client.create_skill(payload)

    if response.get("status_code") == 409:
        remote = client.get_skill(payload["name"])
        local_version = payload["version"]
        remote_version = remote.get("version") if remote else None
        if remote_version and _version_higher(remote_version, local_version):
            raise RuntimeError(
                f"Remote version {remote_version} is newer than local {local_version}; "
                "use --force to overwrite or bump version."
            )
        if remote_version == local_version and not force:
            raise RuntimeError(
                f"Skill {payload['name']}@{local_version} already exists with same version. "
                "Use --force to update."
            )
        response = client.update_skill(payload["name"], payload)

    if response.get("status_code") not in (200, 201):
        raise RuntimeError(
            f"Upload failed ({response.get('status_code')}): {response.get('error')} {response.get('body', '')}"
        )

    receipt = {
        "uploaded_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "hub_base_url": hub_base_url,
        "skill_name": payload["name"],
        "version": payload["version"],
        "visibility": visibility,
        "api_response": response,
    }
    pkg.write_upload_receipt(receipt)

    # Update manifest with visibility.
    manifest_path = pkg.root / ".rosclaw" / "manifest.json"
    if manifest_path.exists():
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("hub", {})["visibility"] = visibility
        pkg.write_manifest_json(manifest)

    return {
        "ok": True,
        "dry_run": False,
        "payload": payload,
        "receipt": receipt,
    }


def _version_higher(a: str, b: str) -> bool:
    try:
        return semver.VersionInfo.parse(a) > semver.VersionInfo.parse(b)
    except Exception:  # noqa: BLE001
        return a > b
