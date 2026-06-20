"""Body diff — detect and categorize changes between body representations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from rosclaw.body.schema import BodyChange, BodyDiff, EffectiveBody, EurdfProfile


class BodyDiffer:
    """Compare body representations and categorize changes."""

    RECHECK_CATEGORIES = {
        "structural",
        "installed_component",
        "actuator_status",
        "sensor_status",
        "calibration",
        "safety",
        "capability",
        "incident",
    }

    def diff_against_eurdf(self, effective: EffectiveBody, eurdf: EurdfProfile) -> BodyDiff:
        """Default diff: e-URDF base vs current effective body."""
        changes: list[BodyChange] = []

        # Safety overrides
        for key, eurdf_value in eurdf.safety.items():
            effective_value = effective.safety.get(key)
            if effective_value != eurdf_value:
                changes.append(BodyChange(
                    path=f"safety.{key}",
                    old=eurdf_value,
                    new=effective_value,
                    category="safety",
                    severity="warning" if key == "safety_level" else "info",
                    requires_skill_recheck=True,
                    reason="body.yaml safety override",
                ))

        # Component status changes
        for name, sensor in effective.sensors.items():
            status = sensor.get("status", "unknown")
            if status != "available":
                changes.append(BodyChange(
                    path=f"installed_components.sensors.{name}.status",
                    old="available",
                    new=status,
                    category="sensor_status",
                    severity="warning",
                    requires_skill_recheck=True,
                    reason="sensor not available",
                ))

        for name, actuator in effective.actuators.items():
            status = actuator.get("status", "unknown")
            if status != "available":
                changes.append(BodyChange(
                    path=f"installed_components.actuators.{name}.status",
                    old="available",
                    new=status,
                    category="actuator_status",
                    severity="warning",
                    requires_skill_recheck=True,
                    reason="actuator not available",
                ))

        # Capability changes
        for cap in effective.capabilities.get("blocked", []):
            if cap in (eurdf.capability_hints.get("all") or []):
                changes.append(BodyChange(
                    path=f"capabilities.{cap}",
                    old="enabled",
                    new="blocked",
                    category="capability",
                    severity="warning",
                    requires_skill_recheck=True,
                    reason="capability disabled or prohibited",
                ))

        return self._summarize(changes)

    def diff_effective_bodies(
        self,
        old: EffectiveBody,
        new: EffectiveBody,
    ) -> BodyDiff:
        """Diff two effective body models (e.g., before/after update-state)."""
        changes: list[BodyChange] = []

        # Capabilities
        for category in ("enabled", "degraded", "blocked"):
            old_set = set(old.capabilities.get(category, []))
            new_set = set(new.capabilities.get(category, []))
            for cap in new_set - old_set:
                changes.append(BodyChange(
                    path=f"capabilities.{cap}",
                    old=f"not in {category}",
                    new=category,
                    category="capability",
                    severity="warning" if category in ("degraded", "blocked") else "info",
                    requires_skill_recheck=True,
                ))
            for cap in old_set - new_set:
                changes.append(BodyChange(
                    path=f"capabilities.{cap}",
                    old=category,
                    new=f"not in {category}",
                    category="capability",
                    severity="info",
                    requires_skill_recheck=True,
                ))

        # Sensor status
        for name, sensor in new.sensors.items():
            old_status = old.sensors.get(name, {}).get("status", "available")
            new_status = sensor.get("status", "available")
            if old_status != new_status:
                changes.append(BodyChange(
                    path=f"installed_components.sensors.{name}.status",
                    old=old_status,
                    new=new_status,
                    category="sensor_status",
                    severity="warning" if new_status != "available" else "info",
                    requires_skill_recheck=True,
                ))

        # Actuator status
        for name, actuator in new.actuators.items():
            old_status = old.actuators.get(name, {}).get("status", "available")
            new_status = actuator.get("status", "available")
            if old_status != new_status:
                changes.append(BodyChange(
                    path=f"installed_components.actuators.{name}.status",
                    old=old_status,
                    new=new_status,
                    category="actuator_status",
                    severity="warning" if new_status != "available" else "info",
                    requires_skill_recheck=True,
                ))

        # Safety overrides
        for key, new_value in new.safety.items():
            old_value = old.safety.get(key)
            if old_value != new_value:
                changes.append(BodyChange(
                    path=f"safety.{key}",
                    old=old_value,
                    new=new_value,
                    category="safety",
                    severity="info",
                    requires_skill_recheck=True,
                ))

        # Structural changes (joints, frames, identity)
        changes.extend(self._detect_structural_changes(old, new))

        return self._summarize(changes)

    def _detect_structural_changes(
        self, old: EffectiveBody, new: EffectiveBody
    ) -> list[BodyChange]:
        """Detect added/removed/changed joints, frames, and identity metadata."""
        changes: list[BodyChange] = []

        for field, label in (("joints", "joint"), ("frames", "frame")):
            old_items = getattr(old, field) or {}
            new_items = getattr(new, field) or {}
            old_names = set(old_items.keys())
            new_names = set(new_items.keys())

            for name in new_names - old_names:
                changes.append(BodyChange(
                    path=f"{field}.{name}",
                    old=None,
                    new="present",
                    category="structural",
                    severity="critical",
                    requires_skill_recheck=True,
                    reason=f"{label} added",
                ))
            for name in old_names - new_names:
                changes.append(BodyChange(
                    path=f"{field}.{name}",
                    old="present",
                    new=None,
                    category="structural",
                    severity="critical",
                    requires_skill_recheck=True,
                    reason=f"{label} removed",
                ))
            for name in old_names & new_names:
                if old_items[name] != new_items[name]:
                    changes.append(BodyChange(
                        path=f"{field}.{name}",
                        old=old_items[name],
                        new=new_items[name],
                        category="structural",
                        severity="warning",
                        requires_skill_recheck=True,
                        reason=f"{label} topology changed",
                    ))

        old_identity = old.identity or {}
        new_identity = new.identity or {}
        for key in set(old_identity) | set(new_identity):
            if old_identity.get(key) != new_identity.get(key):
                changes.append(BodyChange(
                    path=f"identity.{key}",
                    old=old_identity.get(key),
                    new=new_identity.get(key),
                    category="structural",
                    severity="warning",
                    requires_skill_recheck=True,
                    reason="identity metadata changed",
                ))

        return changes

    def diff_against_snapshot(self, effective: EffectiveBody, snapshot_path: Path) -> BodyDiff:
        """Diff current effective body against a historical snapshot."""
        if not snapshot_path.exists():
            return BodyDiff(changes=[BodyChange(
                path="snapshot",
                old=None,
                new=str(snapshot_path),
                category="note_only",
                severity="info",
                requires_skill_recheck=False,
                reason="snapshot not found",
            )])
        with open(snapshot_path, encoding="utf-8") as f:
            old_data = yaml.safe_load(f) or {}
        old = EffectiveBody.from_dict(old_data)
        return self.diff_effective_bodies(old, effective)

    def _summarize(self, changes: list[BodyChange]) -> BodyDiff:
        summary: dict[str, int] = {}
        affected_ids: set[str] = set()
        for change in changes:
            summary[change.category] = summary.get(change.category, 0) + 1
            if (
                change.path.startswith("installed_components.sensors.")
                and ".status" in change.path
            ) or (
                change.path.startswith("installed_components.actuators.")
                and ".status" in change.path
            ):
                affected_ids.add(change.path.split(".")[2])
            elif change.path.startswith("capabilities."):
                affected_ids.add(change.path.split(".")[1])
            elif change.path.startswith("safety."):
                affected_ids.add("safety")
            elif change.path.startswith("joints.") or change.path.startswith("frames."):
                affected_ids.add(change.path.split(".")[1])
            elif change.path.startswith("identity."):
                affected_ids.add("identity")
        affected = sorted({c.category for c in changes})
        return BodyDiff(
            changes=changes,
            summary=summary,
            requires_skill_recheck=any(c.requires_skill_recheck for c in changes),
            affected_categories=affected,
            affected_ids=sorted(affected_ids),
        )


def _format_value(value: Any) -> str:
    """Human-friendly value formatter."""
    if value is None:
        return "null"
    return str(value)
