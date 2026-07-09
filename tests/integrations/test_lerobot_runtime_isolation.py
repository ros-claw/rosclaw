"""Tests for LeRobot runtime isolation and installer modes."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rosclaw.integrations.lerobot.config import migrate_v0_config_to_v1
from rosclaw.integrations.lerobot.installer import LeRobotInstaller
from rosclaw.integrations.lerobot.runtime import RuntimeMode


# ------------------------------------------------------------------
# Mode resolution
# ------------------------------------------------------------------
def test_auto_mode_uses_current_env_when_python312(monkeypatch):
    installer = LeRobotInstaller()
    fake_current = MagicMock()
    fake_current.major = 3
    fake_current.minor = 12
    assert installer._resolve_mode("auto", fake_current) == "current-env"


def test_auto_mode_uses_isolated_when_python311(monkeypatch):
    installer = LeRobotInstaller()
    fake_current = MagicMock()
    fake_current.major = 3
    fake_current.minor = 11
    assert installer._resolve_mode("auto", fake_current) == "isolated"


# ------------------------------------------------------------------
# current-env rejects Python < 3.12
# ------------------------------------------------------------------
def test_current_env_rejects_python311(monkeypatch, tmp_path: Path):
    installer = LeRobotInstaller()
    fake_current = MagicMock()
    fake_current.major = 3
    fake_current.minor = 11
    fake_current.version = "3.11.15"
    fake_current.executable = "/usr/bin/python3.11"

    report = installer._install_current_env(
        "core",
        MagicMock(pip=["lerobot"], checks=["lerobot-info"], name="core"),
        fake_current,
        upgrade=False,
        index_url=None,
        extra_index_url=None,
        env={},
    )

    assert report.ok is False
    assert report.error_code == "python_too_old"
    assert "--mode isolated" in report.message


# ------------------------------------------------------------------
# isolated fails cleanly when python3.12 missing
# ------------------------------------------------------------------
def test_isolated_fails_cleanly_when_python312_missing(monkeypatch, tmp_path: Path):
    installer = LeRobotInstaller()
    fake_current = MagicMock()
    fake_current.major = 3
    fake_current.minor = 11
    fake_current.version = "3.11.15"
    fake_current.executable = "/usr/bin/python3.11"

    monkeypatch.setattr(
        "rosclaw.integrations.lerobot.installer.find_python312",
        lambda preferred=None: None,
    )

    report = installer._install_isolated(
        "core",
        MagicMock(pip=["lerobot"], checks=["lerobot-info"], name="core"),
        fake_current,
        python=None,
        runtime_path=str(tmp_path / "lerobot"),
        upgrade=False,
        force=False,
        index_url=None,
        extra_index_url=None,
        env={},
    )

    assert report.ok is False
    assert report.error_code == "python312_not_found"
    assert "--mode external --python" in report.message


# ------------------------------------------------------------------
# isolated dry-run plan
# ------------------------------------------------------------------
def test_isolated_dry_run_outputs_expected_plan(tmp_path: Path):
    installer = LeRobotInstaller()
    fake_current = MagicMock()
    fake_current.major = 3
    fake_current.minor = 11
    fake_current.version = "3.11.15"
    fake_current.executable = "/usr/bin/python3.11"

    report = installer.install(
        "core",
        mode="isolated",
        runtime_path=str(tmp_path / "lerobot"),
        dry_run=True,
    )

    assert report.ok is True
    assert report.dry_run is True
    assert report.mode == "isolated"
    plan = report.details.get("plan", [])
    plan_text = "\n".join(plan)
    assert "Find Python 3.12" in plan_text or "python3.12" in plan_text
    assert "venv" in plan_text.lower()
    assert "pip install" in plan_text
    assert "lerobot-info" in plan_text
    assert "config" in plan_text.lower()


# ------------------------------------------------------------------
# external mode validation
# ------------------------------------------------------------------
def test_external_mode_rejects_missing_python():
    installer = LeRobotInstaller()
    fake_current = MagicMock()
    fake_current.major = 3
    fake_current.minor = 11
    fake_current.version = "3.11.15"
    fake_current.executable = "/usr/bin/python3.11"

    report = installer._register_external(
        "core",
        MagicMock(pip=["lerobot"], checks=[], name="core"),
        fake_current,
        python=None,
        env={},
    )

    assert report.ok is False
    assert report.error_code == "external_python_not_found"


def test_external_mode_rejects_python311(tmp_path: Path, monkeypatch):
    installer = LeRobotInstaller()
    fake_python = tmp_path / "fake_python3.11"
    fake_python.write_text("#!/bin/sh\nexit 0")
    fake_python.chmod(0o755)

    fake_current = MagicMock()
    fake_current.major = 3
    fake_current.minor = 11
    fake_current.version = "3.11.15"
    fake_current.executable = "/usr/bin/python3.11"

    report = installer._register_external(
        "core",
        MagicMock(pip=["lerobot"], checks=[], name="core"),
        fake_current,
        python=str(fake_python),
        env={},
    )

    assert report.ok is False
    assert report.error_code == "external_python_too_old"


# ------------------------------------------------------------------
# Config migration
# ------------------------------------------------------------------
def test_lerobot_config_v0_migrates_to_v1():
    v0 = {
        "enabled": True,
        "profile": "core",
        "python": "/home/user/.venv-lerobot/bin/python",
        "pip": "/home/user/.venv-lerobot/bin/pip",
        "lerobot_version": "0.6.1",
        "capabilities": {"provider_type_lerobot_policy": True},
    }
    v1 = migrate_v0_config_to_v1(v0)

    assert "lerobot_runtime" in v1
    assert v1["lerobot_runtime"]["python_executable"] == v0["python"]
    assert v1["lerobot_runtime"]["pip_executable"] == v0["pip"]
    assert v1["lerobot_runtime"]["lerobot_version"] == v0["lerobot_version"]
    assert v1["install_mode"] == "external"
    assert v1["capabilities"] == v0["capabilities"]


def test_lerobot_config_v0_current_env_inference():
    v0 = {
        "enabled": True,
        "profile": "core",
        "python": sys.executable,
        "pip": sys.executable.replace("python", "pip"),
        "lerobot_version": "0.6.1",
    }
    v1 = migrate_v0_config_to_v1(v0)
    assert v1["install_mode"] == "current-env"
    assert v1["lerobot_runtime"]["mode"] == "current-env"


# ------------------------------------------------------------------
# Runtime helpers smoke
# ------------------------------------------------------------------
def test_inspect_python_reports_version(tmp_path: Path):
    from rosclaw.integrations.lerobot.runtime import inspect_python

    info = inspect_python(sys.executable)
    assert info.ok is True
    assert info.major is not None
    assert info.minor is not None
