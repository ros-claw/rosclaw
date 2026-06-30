"""Tests for RealSense skill runtime execution (dispatch and result schema)."""
from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.skill.skill_runner import SkillRunner, SkillRunResult
from rosclaw.skill.runtime_handlers import (
    realsense_capture_rgbd,
    realsense_depth_health_check,
    realsense_imu_check,
    scene_risk_scan,
)


def test_skill_runner_list() -> None:
    runner = SkillRunner()
    skills = runner.list_skills()
    assert "realsense_capture_rgbd" in skills
    assert "realsense_depth_health_check" in skills
    assert "realsense_imu_check" in skills
    assert "scene_risk_scan" in skills


def test_skill_runner_run_unknown() -> None:
    runner = SkillRunner()
    result = runner.run("unknown_skill")
    assert result.skill_id == "unknown_skill"
    assert result.success is False
    assert "error" in result.payload


def test_skill_runner_run_realsense_capture_rgbd() -> None:
    runner = SkillRunner()
    result = runner.run("realsense_capture_rgbd", body="d405_lab_01", output_dir="/tmp")
    assert isinstance(result, SkillRunResult)
    assert result.skill_id == "realsense_capture_rgbd"
    # Without ROS2 the payload contains an error, but the runner itself succeeds
    assert "body" in result.payload
    assert result.payload["body"] == "d405_lab_01"


def test_skill_runner_run_realsense_depth_health_check() -> None:
    runner = SkillRunner()
    result = runner.run("realsense_depth_health_check", body="d405_lab_01", duration_sec=30)
    assert result.skill_id == "realsense_depth_health_check"
    assert "body" in result.payload
    assert result.payload["duration_sec"] == 30


def test_skill_runner_run_realsense_imu_check() -> None:
    runner = SkillRunner()
    result = runner.run("realsense_imu_check", body="d435i_lab_01", duration_sec=5)
    assert result.skill_id == "realsense_imu_check"
    assert "body" in result.payload


def test_skill_runner_run_scene_risk_scan(monkeypatch, tmp_path: Path) -> None:
    image = tmp_path / "scene.jpg"
    image.write_bytes(b"fake image")

    async def _fake_call(*args, **kwargs):
        return {
            "status": "ok",
            "request_id": "req-1",
            "latency_ms": 123,
            "scene": "lab bench",
            "normalized_risk": 0.2,
            "executable": True,
            "requires_guard": False,
            "risks": [],
            "input_frame_uri": f"file://{image}",
        }

    monkeypatch.setattr("rosclaw.skill.runtime_handlers.call_provider", _fake_call)

    runner = SkillRunner()
    result = runner.run("scene_risk_scan", body="d435i_lab_01", provider="cosmos-reason2-lan", image_path=str(image))
    assert result.skill_id == "scene_risk_scan"
    assert result.payload["provider"] == "cosmos-reason2-lan"
    assert result.payload["image_path"] == str(image)
    assert result.payload["scene"] == "lab bench"
    assert result.payload["status"] == "ok"


def test_handler_realsense_capture_rgbd() -> None:
    result = realsense_capture_rgbd(body="d405_lab_01", output_dir="/tmp")
    assert "body" in result
    assert "rgb_path" in result
    assert "depth_path" in result


def test_handler_realsense_depth_health_check() -> None:
    result = realsense_depth_health_check(body="d405_lab_01", duration_sec=10)
    assert "body" in result
    assert "duration_sec" in result


def test_handler_realsense_imu_check() -> None:
    result = realsense_imu_check(body="d435i_lab_01", duration_sec=2)
    assert "body" in result
    assert "duration_sec" in result


def test_handler_scene_risk_scan_no_image() -> None:
    result = scene_risk_scan(body="d435i_lab_01", provider=None, image_path=None)
    assert "body" in result
    assert "image_path" in result
    assert result["status"] == "failed"
    assert "error" in result


def test_handler_scene_risk_scan_missing_image() -> None:
    result = scene_risk_scan(body="d435i_lab_01", provider="cosmos-reason2-lan", image_path="/tmp/does_not_exist.jpg")
    assert result["status"] == "failed"
    assert "error" in result
