"""Skill compatibility checking against the effective body."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import yaml

from rosclaw.body.schema import (
    EffectiveBody,
    SkillCompatibilityReport,
    SkillCompatibilityResult,
    SkillManifest,
)


class SkillCompatibilityChecker:
    """Check skill manifests against effective body model."""

    def check_one(self, skill: SkillManifest, body: EffectiveBody) -> SkillCompatibilityResult:
        missing: list[str] = []
        warnings: list[str] = []

        requires = skill.requires

        # Robot class
        robot_class_req = requires.get("robot_class")
        if robot_class_req:
            body_class = body.identity.get("robot_class", "")
            if isinstance(robot_class_req, list):
                if body_class not in robot_class_req:
                    return self._blocked(skill, body, f"robot_class mismatch: {body_class} not in {robot_class_req}")
            elif body_class != robot_class_req:
                return self._blocked(skill, body, f"robot_class mismatch: {body_class}")

        # e-URDF profile compatibility
        eurdf_req = requires.get("eurdf")
        if eurdf_req:
            profile_id = eurdf_req.get("compatible_profiles", [])
            if profile_id and body.eurdf_uri.split("/")[-1].split("@")[0] not in profile_id:
                return self._blocked(skill, body, "e-URDF profile not supported")

        # Capabilities
        caps = body.capabilities
        for cap in requires.get("capabilities", {}).get("all_of", []):
            if cap in caps.get("blocked", []):
                missing.append(f"capability:{cap}")
            elif cap in caps.get("degraded", []):
                warnings.append(f"capability degraded:{cap}")
            elif cap not in caps.get("enabled", []):
                missing.append(f"capability:{cap}")

        # Sensors
        for sensor_req in requires.get("sensors", {}).get("all_of", []):
            sensor_id = sensor_req.get("id") or sensor_req.get("name")
            sensor = body.sensors.get(sensor_id) if sensor_id else None
            if not sensor:
                missing.append(f"sensor:{sensor_id}")
            elif sensor.get("status", "available") != sensor_req.get("status", "available"):
                missing.append(f"sensor:{sensor_id}:{sensor.get('status')}")

        # Actuators
        for actuator_req in requires.get("actuators", {}).get("all_of", []):
            group = actuator_req.get("group")
            actuator = body.actuators.get(group) if group else None
            if not actuator:
                missing.append(f"actuator_group:{group}")
            elif actuator.get("status", "available") != actuator_req.get("status", "available"):
                missing.append(f"actuator_group:{group}:{actuator.get('status')}")

        # Frames
        for frame in requires.get("frames", {}).get("all_of", []):
            if frame not in body.frames.values() and frame not in body.frames:
                missing.append(f"frame:{frame}")

        # Calibration
        cal_req = requires.get("calibration", {})
        if cal_req:
            cal_status = body.source_trace.get("calibration_status", "uncalibrated")
            allowed = cal_req.get("required_status", ["validated"])
            if cal_status not in allowed:
                if skill.degradation_policy.get("allow_uncalibrated_camera"):
                    warnings.append("calibration not ideal")
                else:
                    missing.append("calibration_status")

        # Safety speed limit
        speed_req = requires.get("safety", {}).get("max_base_speed_mps_at_least")
        if speed_req is not None:
            body_speed = body.safety.get("max_base_speed_mps")
            if body_speed is not None and body_speed < speed_req:
                if skill.degradation_policy.get("allow_lower_speed"):
                    warnings.append("lower max speed; skill execution should be slowed")
                else:
                    missing.append("max_base_speed_mps")

        if missing:
            return self._blocked(skill, body, missing=missing, warnings=warnings)
        if warnings:
            return self._degraded(skill, body, warnings)
        return self._compatible(skill, body)

    def check_all(
        self,
        skills: list[SkillManifest],
        body: EffectiveBody,
    ) -> SkillCompatibilityReport:
        report = SkillCompatibilityReport(
            body_instance_id=body.body_instance_id,
            effective_body_hash=body.effective_body_hash,
        )
        for skill in skills:
            result = self.check_one(skill, body)
            report.skills[f"{skill.skill_id}@{skill.skill_version}"] = result
        report.summary = {
            "compatible": sum(1 for r in report.skills.values() if r.status == "compatible"),
            "degraded": sum(1 for r in report.skills.values() if r.status == "degraded"),
            "blocked": sum(1 for r in report.skills.values() if r.status == "blocked"),
            "unknown": sum(1 for r in report.skills.values() if r.status == "unknown"),
        }
        return report

    def check_incremental(
        self,
        previous: SkillCompatibilityReport | None,
        skills: list[SkillManifest],
        body: EffectiveBody,
        affected_ids: set[str],
    ) -> SkillCompatibilityReport:
        """Re-check only skills whose requirements intersect the affected IDs.

        Unaffected skills are copied from the previous report when available,
        avoiding redundant compatibility checks on every body state change.
        """
        report = SkillCompatibilityReport(
            body_instance_id=body.body_instance_id,
            effective_body_hash=body.effective_body_hash,
        )
        previous_map = previous.skills if previous else {}
        for skill in skills:
            key = f"{skill.skill_id}@{skill.skill_version}"
            req_ids = skill.requirement_ids()
            result: SkillCompatibilityResult | None
            if req_ids & affected_ids:
                result = self.check_one(skill, body)
            else:
                result = previous_map.get(key)
                if result is None:
                    result = self.check_one(skill, body)
                else:
                    result = replace(result)
                    result.checked_against = {
                        "body_hash": body.effective_body_hash,
                        "eurdf_uri": body.eurdf_uri,
                    }
            report.skills[key] = result
        report.summary = {
            "compatible": sum(1 for r in report.skills.values() if r.status == "compatible"),
            "degraded": sum(1 for r in report.skills.values() if r.status == "degraded"),
            "blocked": sum(1 for r in report.skills.values() if r.status == "blocked"),
            "unknown": sum(1 for r in report.skills.values() if r.status == "unknown"),
        }
        return report

    def _compatible(self, skill: SkillManifest, body: EffectiveBody) -> SkillCompatibilityResult:
        return SkillCompatibilityResult(
            skill_id=skill.skill_id,
            skill_version=skill.skill_version,
            status="compatible",
            reason="All requirements satisfied.",
            checked_against={"body_hash": body.effective_body_hash, "eurdf_uri": body.eurdf_uri},
        )

    def _degraded(
        self,
        skill: SkillManifest,
        body: EffectiveBody,
        warnings: list[str],
    ) -> SkillCompatibilityResult:
        return SkillCompatibilityResult(
            skill_id=skill.skill_id,
            skill_version=skill.skill_version,
            status="degraded",
            reason="; ".join(warnings),
            warnings=warnings,
            checked_against={"body_hash": body.effective_body_hash, "eurdf_uri": body.eurdf_uri},
        )

    def _blocked(
        self,
        skill: SkillManifest,
        body: EffectiveBody,
        reason: str = "",
        missing: list[str] | None = None,
        warnings: list[str] | None = None,
    ) -> SkillCompatibilityResult:
        return SkillCompatibilityResult(
            skill_id=skill.skill_id,
            skill_version=skill.skill_version,
            status="blocked",
            reason=reason or "Missing requirements",
            missing_requirements=missing or [],
            warnings=warnings or [],
            checked_against={"body_hash": body.effective_body_hash, "eurdf_uri": body.eurdf_uri},
        )


class SkillCompatibilityStore:
    """Persist skill compatibility report to YAML."""

    def __init__(self, store_path: Path):
        self.store_path = store_path

    def load(self) -> SkillCompatibilityReport:
        if not self.store_path.exists():
            return SkillCompatibilityReport(body_instance_id="", effective_body_hash="")
        with open(self.store_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return SkillCompatibilityReport.from_dict(data)

    def save(self, report: SkillCompatibilityReport) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(report.to_dict(), f, sort_keys=False, allow_unicode=True)

    def get(self, skill_id: str, skill_version: str, body_hash: str) -> SkillCompatibilityResult | None:
        report = self.load()
        if report.effective_body_hash != body_hash:
            return None
        return report.skills.get(f"{skill_id}@{skill_version}")

    def invalidate(self, body_hash: str) -> None:
        report = self.load()
        if report.effective_body_hash == body_hash:
            report.skills = {}
            self.save(report)
