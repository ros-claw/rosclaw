"""Test LeRobot CLI commands."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rosclaw.cli import cmd_practice_export, main


@pytest.fixture
def sample_manifest(tmp_path: Path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "name: lerobot_policy_sample\n"
        "version: 0.1.0\n"
        "type: skill\n"
        "capabilities:\n"
        "  - lerobot.policy.infer\n"
        "embodiment:\n"
        "  supported_robots:\n"
        "    - ur5e_lab_01\n"
        "  action_space:\n"
        "    - j0\n"
        "safety:\n"
        "  max_action_norm: 0.5\n",
        encoding="utf-8",
    )
    return manifest


@pytest.fixture
def sample_input(tmp_path: Path):
    inp = tmp_path / "input.json"
    inp.write_text(json.dumps({"observation": {"state": [0.0]}}), encoding="utf-8")
    return inp


def test_setup_lerobot_dry_run(capsys):
    """Dry-run setup should succeed and not install anything."""
    with patch.object(sys, "argv", ["rosclaw", "setup", "lerobot", "--profile", "core", "--dry-run"]):
        rc = main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "Dry-run" in out


def test_setup_lerobot_json_is_single_machine_readable_document(capsys):
    with patch.object(
        sys,
        "argv",
        [
            "rosclaw",
            "setup",
            "lerobot",
            "--profile",
            "core",
            "--dry-run",
            "--json",
        ],
    ):
        rc = main()
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["profile"] == "core"
    assert payload["dry_run"] is True
    assert payload["details"]["plan"]


def test_lerobot_doctor(capsys):
    """Doctor command should not crash."""
    with patch.object(sys, "argv", ["rosclaw", "lerobot", "doctor"]):
        rc = main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "lerobot" in out.lower()


def test_lerobot_capabilities(capsys):
    """Capabilities command should list LeRobot capabilities."""
    with patch.object(sys, "argv", ["rosclaw", "lerobot", "capabilities"]):
        rc = main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "provider_type_lerobot_policy" in out


def test_capability_list(capsys):
    """Capability list should include LeRobot entries."""
    with patch.object(sys, "argv", ["rosclaw", "capability", "list"]):
        rc = main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "lerobot" in out.lower()


def test_lerobot_compatibility(capsys):
    """Compatibility command should print the matrix."""
    with patch.object(sys, "argv", ["rosclaw", "lerobot", "compatibility"]):
        rc = main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "act" in out.lower()
    assert "LeRobot Policy Compatibility Matrix" in out


def test_lerobot_compatibility_json(capsys):
    """Compatibility --json should return a parseable report."""
    with patch.object(sys, "argv", ["rosclaw", "lerobot", "compatibility", "--json"]):
        rc = main()
    assert rc == 0
    out = capsys.readouterr().out
    report = json.loads(out)
    assert "matrix" in report
    assert "levels" in report


def test_provider_infer_dry_run(capsys, sample_manifest, sample_input):
    """Provider infer dry-run should return sample action."""
    with patch.object(
        sys,
        "argv",
        [
            "rosclaw",
            "provider",
            "infer",
            "--type",
            "lerobot_policy",
            "--manifest",
            str(sample_manifest),
            "--input",
            str(sample_input),
            "--dry-run",
        ],
    ):
        rc = main()
    assert rc == 0
    out = capsys.readouterr().out
    result = json.loads(out)
    assert result["result"]["dry_run"] is True
    assert result["result"]["safety"]["requires_guard"] is True


def test_provider_infer_missing_manifest(sample_input):
    """Provider infer should fail gracefully with missing manifest."""
    with patch.object(
        sys,
        "argv",
        [
            "rosclaw",
            "provider",
            "infer",
            "--type",
            "lerobot_policy",
            "--manifest",
            "/does/not/exist.yaml",
            "--input",
            str(sample_input),
            "--dry-run",
        ],
    ):
        rc = main()
    assert rc == 1


def test_practice_id_keeps_real_lerobot_export_path(tmp_path: Path):
    practice_id = "practice-001"
    (tmp_path / practice_id).mkdir()
    real_exporter = MagicMock()
    real_exporter.export.return_value = tmp_path / "real-export"
    skeleton_exporter = MagicMock()
    args = Namespace(
        episode_dir=None,
        episode_id=practice_id,
        practice_id=None,
        format="lerobot",
        data_root=str(tmp_path),
        output=None,
    )

    with (
        patch(
            "rosclaw.practice.exporters.LeRobotExporter",
            return_value=real_exporter,
        ),
        patch(
            "rosclaw.practice.exporters.LeRobotSkeletonExporter",
            return_value=skeleton_exporter,
        ),
    ):
        rc = cmd_practice_export(args)

    assert rc == 0
    real_exporter.export.assert_called_once_with(practice_id, output_path=None)
    skeleton_exporter.export.assert_not_called()
