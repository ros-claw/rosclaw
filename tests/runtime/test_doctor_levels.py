"""Progressive execution-readiness doctor tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rosclaw.runtime.doctor_levels import DoctorLevel, LevelDoctor


def _configured_home(tmp_path: Path) -> Path:
    home = tmp_path / ".rosclaw"
    config = home / "config" / "rosclaw.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("runtime:\n  robot_id: sim_ur5e\n", encoding="utf-8")
    return home


def test_package_level_does_not_claim_runtime_or_robot_readiness(tmp_path: Path) -> None:
    result = LevelDoctor(tmp_path / ".rosclaw").run(DoctorLevel.PACKAGE)

    assert result.passed is True
    assert result.runtime_initialized is False
    assert result.robot_connected is False
    assert result.real_execution_ready is False


def test_configured_level_fails_without_configuration(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = LevelDoctor(tmp_path / ".rosclaw").run(DoctorLevel.CONFIGURED)

    assert result.passed is False
    assert result.configured is False


def test_verified_level_runs_real_mujoco_and_preserves_hardware_boundary(tmp_path: Path) -> None:
    result = LevelDoctor(_configured_home(tmp_path)).run(DoctorLevel.VERIFIED)

    assert result.passed is True
    assert result.southbound_connected is True
    assert result.action_dry_run is True
    assert result.verified_action_path is True
    assert result.verified_execution_mode == "SIMULATION"
    assert result.robot_connected is False
    assert result.real_execution_ready is False


def test_cli_doctor_level_json_exposes_readiness_fields(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    home = _configured_home(tmp_path)
    monkeypatch.setenv("ROSCLAW_HOME", str(home))
    from rosclaw.cli import main

    sys.argv = ["rosclaw", "doctor", "--level", "package", "--json"]
    code = main()
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["requested_level"] == "package"
    assert payload["package_healthy"] is True
    assert payload["robot_connected"] is False
    assert payload["real_execution_ready"] is False
