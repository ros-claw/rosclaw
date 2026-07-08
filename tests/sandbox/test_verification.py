"""Tests for deterministic MuJoCo sandbox verification cases."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

import pytest

from rosclaw.sandbox.verification import run_ur5e_joint_preview, run_verification_case


def test_run_verification_case_rejects_unknown_case() -> None:
    with pytest.raises(ValueError):
        run_verification_case("unknown-case")


def test_ur5e_joint_preview_reports_no_physics(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = MagicMock()
    fake.has_physics = False
    fake.close.return_value = None
    monkeypatch.setattr(
        "rosclaw.sandbox.verification.Sandbox.create",
        lambda **_kwargs: fake,
    )

    result = run_ur5e_joint_preview()

    assert result.passed is False
    assert result.has_physics is False
    assert "without physics" in result.reason
    fake.close.assert_called_once()


def test_ur5e_joint_preview_passes_with_advancing_state(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = MagicMock()
    fake.has_physics = True
    fake.get_state.return_value = {"time": 0.0}
    fake.step.return_value = {"qpos": [0.1] * 6, "qvel": [0.0] * 6, "time": 0.01}
    fake.get_observation.return_value = {
        "contacts": [],
        "body_positions": {"base": [0.0, 0.0, 0.0]},
    }
    monkeypatch.setattr(
        "rosclaw.sandbox.verification.Sandbox.create",
        lambda **_kwargs: fake,
    )

    result = run_ur5e_joint_preview(steps=3)

    assert result.passed is True
    assert result.qpos_size == 6
    assert result.qvel_size == 6
    assert result.final_time == 0.01
    assert result.details["body_count"] == 1
    assert fake.step.call_count == 3


def test_sandbox_verify_cli_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rosclaw.cli import main
    from rosclaw.sandbox.verification import SandboxVerificationResult

    def fake_case(*_args: object, **_kwargs: object) -> SandboxVerificationResult:
        return SandboxVerificationResult(
            case="ur5e-joint-preview",
            robot_id="universal_robots_ur5e",
            world_id="empty",
            task="preview",
            passed=True,
            has_physics=True,
            steps=5,
            qpos_size=6,
            qvel_size=6,
            final_time=0.01,
        )

    monkeypatch.setattr("rosclaw.sandbox.verification.run_verification_case", fake_case)
    monkeypatch.setattr(
        sys,
        "argv",
        ["rosclaw", "sandbox", "verify", "--json"],
    )

    assert main() == 0
    data = json.loads(capsys.readouterr().out)
    assert data["passed"] is True
    assert data["case"] == "ur5e-joint-preview"


def test_real_ur5e_joint_preview_if_mujoco_available() -> None:
    result = run_ur5e_joint_preview(steps=2)
    if not result.has_physics:
        pytest.skip(result.reason)
    assert result.passed is True
    assert result.qpos_size > 0
    assert result.qvel_size > 0
    assert result.final_time > 0.0
