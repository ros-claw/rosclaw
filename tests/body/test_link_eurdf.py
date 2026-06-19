"""Tests for rosclaw body link-eurdf."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from rosclaw.cli import main as rosclaw_main


@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    yield


def test_link_eurdf_generates_artifacts(capsys):
    """End-to-end CLI test: link-eurdf creates required files."""
    with patch.object(sys, "argv", ["rosclaw", "body", "link-eurdf", "unitree-g1"]):
        rc = rosclaw_main()
    assert rc == 0

    body_dir = Path.home() / ".rosclaw" / "body"
    assert (body_dir / "body.yaml").exists()
    assert (body_dir / "calibration.yaml").exists()
    assert (body_dir / "maintenance.log").exists()
    assert (body_dir / "EMBODIMENT.md").exists()
    assert (body_dir / "refs" / "eurdf.lock").exists()
    assert (body_dir / "refs" / "effective_body.json").exists()

    captured = capsys.readouterr()
    assert "Linked e-URDF profile" in captured.out
    assert "unitree-g1" in captured.out


def test_link_eurdf_respects_force_flag(isolated_workspace, capsys):
    with patch.object(sys, "argv", ["rosclaw", "body", "link-eurdf", "unitree-g1"]):
        assert rosclaw_main() == 0
    # Without --force should refuse
    with patch.object(sys, "argv", ["rosclaw", "body", "link-eurdf", "unitree-g1"]):
        assert rosclaw_main() == 1
    # With --force should succeed
    with patch.object(sys, "argv", ["rosclaw", "body", "link-eurdf", "unitree-g1", "--force"]):
        assert rosclaw_main() == 0


def test_link_eurdf_unknown_profile():
    with patch.object(sys, "argv", ["rosclaw", "body", "link-eurdf", "nonexistent_robot_xyz"]):
        assert rosclaw_main() == 1
