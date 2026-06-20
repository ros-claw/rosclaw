"""Fleet-wide body operations and cross-body compatibility aggregation."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from rosclaw.body.compatibility import SkillCompatibilityChecker
from rosclaw.body.registry import BodyRegistryManager
from rosclaw.body.resolver import BodyResolver
from rosclaw.body.schema import (
    FleetCompatibilityReport,
    SkillCompatibilityReport,
    SkillManifest,
)


def discover_skill_manifests(workspace: Path | str) -> list[SkillManifest]:
    """Discover skill manifests under workspace/skills."""
    skills_dir = Path(workspace) / "skills"
    if not skills_dir.exists():
        return []
    manifests: list[SkillManifest] = []
    for path in skills_dir.rglob("*.skill.yaml"):
        with contextlib.suppress(Exception):
            manifests.append(SkillManifest.from_yaml(path))
    return manifests


class FleetCompatibilityError(RuntimeError):
    """Raised when fleet aggregation cannot complete."""


class FleetCompatibilityAggregator:
    """Aggregate skill compatibility across all bodies in a workspace."""

    def __init__(self, workspace: Path | str) -> None:
        self.workspace = Path(workspace)
        self.registry = BodyRegistryManager(self.workspace)
        self.checker = SkillCompatibilityChecker()

    def aggregate(
        self,
        skill_manifests: list[SkillManifest] | None = None,
    ) -> FleetCompatibilityReport:
        """Check every registered body against the supplied skill manifests.

        Bodies that cannot be resolved or compiled are recorded as empty
        compatibility reports with an ``error`` note in the fleet summary.
        """
        manifests = skill_manifests or []
        entries = self.registry.list_bodies()
        per_body: dict[str, SkillCompatibilityReport] = {}
        errors: list[str] = []

        for entry in entries:
            try:
                resolver = BodyResolver(self.workspace, body_id=entry.body_id)
                effective = resolver.recompile_effective_body()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{entry.body_id}: {exc}")
                per_body[entry.body_id] = SkillCompatibilityReport(
                    body_instance_id=entry.body_id,
                    effective_body_hash="",
                )
                continue

            report = self.checker.check_all(manifests, effective)
            per_body[entry.body_id] = report

        return FleetCompatibilityReport(
            workspace=str(self.workspace),
            per_body=per_body,
            fleet_summary=self._summarize(per_body, errors),
        )

    def _summarize(
        self,
        per_body: dict[str, SkillCompatibilityReport],
        errors: list[str],
    ) -> dict[str, Any]:
        """Build fleet-level compatibility summary."""
        status_priority = {"blocked": 0, "unknown": 1, "degraded": 2, "compatible": 3}
        skill_status: dict[str, str] = {}

        for report in per_body.values():
            for skill_key, result in report.skills.items():
                current = skill_status.get(skill_key)
                candidate = result.status
                if current is None or status_priority.get(candidate, 1) < status_priority.get(current, 1):
                    skill_status[skill_key] = candidate

        summary = {
            status: sum(1 for s in skill_status.values() if s == status)
            for status in ("compatible", "degraded", "blocked", "unknown")
        }

        compatible_with_all = [
            k for k, s in skill_status.items() if s == "compatible"
        ]
        blocked_on_any = [k for k, s in skill_status.items() if s == "blocked"]
        degraded_on_any = [k for k, s in skill_status.items() if s == "degraded"]
        unknown_on_any = [k for k, s in skill_status.items() if s == "unknown"]

        return {
            "total_bodies": len(per_body),
            "total_skills": len(skill_status),
            "compatible_skills": summary["compatible"],
            "degraded_skills": summary["degraded"],
            "blocked_skills": summary["blocked"],
            "unknown_skills": summary["unknown"],
            "compatible_with_all": compatible_with_all,
            "blocked_on_any": blocked_on_any,
            "degraded_on_any": degraded_on_any,
            "unknown_on_any": unknown_on_any,
            "errors": errors,
        }
