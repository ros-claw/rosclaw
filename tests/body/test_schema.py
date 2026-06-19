"""Tests for body module schemas."""

from pathlib import Path

from rosclaw.body.schema import (
    BodyYaml,
    CalibrationYaml,
    EffectiveBody,
    EurdfProfile,
    MaintenanceEvent,
    SkillCompatibilityReport,
    SkillCompatibilityResult,
    SkillManifest,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_eurdf_profile_from_yaml():
    profile = EurdfProfile.from_yaml(FIXTURES / "eurdf" / "unitree-g1" / "profile.yaml")
    assert profile.profile_id == "unitree-g1"
    assert profile.profile_version == "1.0.0"
    assert "head_rgb_camera" in [s["name"] for s in profile.sensors]


def test_body_yaml_roundtrip():
    body = BodyYaml(body_instance={"id": "test"})
    data = body.to_dict()
    restored = BodyYaml.from_dict(data)
    assert restored.body_instance["id"] == "test"


def test_calibration_yaml_roundtrip():
    cal = CalibrationYaml(body_instance_id="b1", model_ref="rosclaw://eurdf/unitree-g1@1.0.0")
    data = cal.to_dict()
    restored = CalibrationYaml.from_dict(data)
    assert restored.body_instance_id == "b1"


def test_effective_body_hash_stable():
    body = EffectiveBody(
        body_instance_id="b1",
        eurdf_uri="rosclaw://eurdf/unitree-g1@1.0.0",
        effective_body_hash="",
        compiled_at="2026-06-18T10:00:00+00:00",
    )
    body.effective_body_hash = body.compute_hash()
    assert body.effective_body_hash
    # Same canonical JSON yields same hash
    assert body.compute_hash() == body.effective_body_hash


def test_maintenance_event_from_dict():
    event = MaintenanceEvent(
        message="test",
        type="incident",
        severity="warning",
        affects=["camera"],
    )
    restored = MaintenanceEvent.from_dict(event.to_dict())
    assert restored.type == "incident"


def test_skill_manifest_from_yaml():
    manifest = SkillManifest.from_yaml(FIXTURES / "skills" / "walk_forward.skill.yaml")
    assert manifest.skill_id == "walk_forward"
    assert "walk" in manifest.requires["capabilities"]["all_of"]


def test_skill_compatibility_report_roundtrip():
    result = SkillCompatibilityResult(
        skill_id="s1",
        skill_version="1.0.0",
        status="blocked",
        reason="missing camera",
    )
    report = SkillCompatibilityReport(
        body_instance_id="b1",
        effective_body_hash="h1",
        skills={"s1@1.0.0": result},
        summary={"blocked": 1},
    )
    restored = SkillCompatibilityReport.from_dict(report.to_dict())
    assert restored.skills["s1@1.0.0"].status == "blocked"
