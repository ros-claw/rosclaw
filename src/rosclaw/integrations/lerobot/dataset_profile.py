"""Export profiles for ROSClaw-rich LeRobotDatasets.

A profile is a named selection of feature groups.  The default profile is
``minimal`` which keeps P2 behavior unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PROFILE_NAMES = {"minimal", "safety", "physical", "safety-rich"}
AVAILABLE_PROFILE_NAMES = {"minimal", "safety", "physical", "safety-rich"}
PLANNED_PROFILE_NAMES: set[str] = set()


@dataclass
class DatasetExportProfile:
    """Named profile describing which feature groups to export."""

    name: str = "minimal"
    feature_groups: set[str] = field(default_factory=set)
    include_body_snapshot: bool = False
    body_snapshot_mode: str = "sanitized"
    available: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "feature_groups": sorted(self.feature_groups),
            "include_body_snapshot": self.include_body_snapshot,
            "body_snapshot_mode": self.body_snapshot_mode,
            "available": self.available,
        }


def _base(name: str, groups: set[str]) -> DatasetExportProfile:
    return DatasetExportProfile(name=name, feature_groups=groups)


PROFILE_DEFINITIONS: dict[str, DatasetExportProfile] = {
    "minimal": _base("minimal", set()),
    "safety": _base("safety", {"safety"}),
    "physical": DatasetExportProfile(
        name="physical",
        feature_groups={"safety", "action", "physical_telemetry"},
        available=True,
    ),
    "safety-rich": _base("safety-rich", {"safety", "failure", "intervention", "action", "outcome"}),
}


def resolve_profile(
    name: str | None,
    *,
    include_groups: list[str] | None = None,
    exclude_groups: list[str] | None = None,
    include_body_snapshot: bool = False,
    body_snapshot_mode: str = "sanitized",
    allow_unavailable: bool = False,
) -> DatasetExportProfile:
    """Resolve a profile name and optional include/exclude overrides."""
    name = name or "minimal"
    if name not in PROFILE_NAMES:
        raise ValueError(f"Unknown export profile: {name}. Choose from {sorted(PROFILE_NAMES)}.")

    profile = PROFILE_DEFINITIONS[name]
    if not profile.available and not allow_unavailable:
        raise ValueError(
            f"Profile '{name}' is planned for a future gate and not available yet. "
            f"Available profiles: {sorted(AVAILABLE_PROFILE_NAMES)}."
        )

    groups: set[str] = set(profile.feature_groups)

    if include_groups:
        for g in include_groups:
            groups.add(g)
    if exclude_groups:
        groups.difference_update(exclude_groups)

    return DatasetExportProfile(
        name=name,
        feature_groups=groups,
        include_body_snapshot=include_body_snapshot,
        body_snapshot_mode=body_snapshot_mode,
        available=profile.available,
    )


def profile_from_request(request: dict[str, Any]) -> DatasetExportProfile:
    """Build a profile from a dataset worker request dict (v2)."""
    profile_name = request.get("profile") or "minimal"
    features = request.get("features", {})
    include_groups: list[str] | None = None
    if isinstance(features, dict) and features.get("include_groups"):
        include_groups = list(features["include_groups"])
    return resolve_profile(
        profile_name,
        include_groups=include_groups,
        include_body_snapshot=bool(features.get("include_body_snapshot", False)),
        body_snapshot_mode=str(features.get("body_snapshot_mode", "sanitized")),
        allow_unavailable=bool(features.get("allow_unavailable_profile", False)),
    )


def resolve_feature_groups_for_profile(profile_name: str) -> list[str]:
    """Return the worker feature groups for a profile name."""
    return sorted(resolve_profile(profile_name).feature_groups)


def get_profile_availability(profile_name: str) -> dict[str, Any]:
    """Return availability metadata for a profile."""
    profile = PROFILE_DEFINITIONS.get(profile_name)
    if profile is None:
        return {"available": False, "reason": "unknown profile"}
    if profile_name in PLANNED_PROFILE_NAMES:
        return {
            "available": False,
            "reason": f"Profile '{profile_name}' is planned for a future gate.",
        }
    return {"available": True, "feature_groups": sorted(profile.feature_groups)}


__all__ = [
    "PROFILE_NAMES",
    "AVAILABLE_PROFILE_NAMES",
    "PLANNED_PROFILE_NAMES",
    "DatasetExportProfile",
    "resolve_profile",
    "resolve_feature_groups_for_profile",
    "profile_from_request",
    "get_profile_availability",
]
