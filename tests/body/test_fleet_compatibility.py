"""Tests for fleet-wide skill compatibility aggregation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from rosclaw.body.fleet import FleetCompatibilityAggregator
from rosclaw.body.registry import BodyRegistryManager
from rosclaw.body.schema import SkillManifest
from rosclaw.cli import main as rosclaw_main


@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))


def _run_with_capture(*argv: str) -> str:
    from io import StringIO

    out = StringIO()
    with patch.object(sys, "argv", ["rosclaw", *argv]), patch("sys.stdout", new=out):
        rosclaw_main()
    return out.getvalue()


def _run(*argv: str) -> int:
    with patch.object(sys, "argv", ["rosclaw", *argv]):
        return rosclaw_main()


def _write_skill(workspace: Path, skill_id: str, requires: dict) -> None:
    skills_dir = workspace / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "rosclaw.skill.v1",
        "skill_id": skill_id,
        "skill_version": "1.0.0",
        "requires": requires,
    }
    (skills_dir / f"{skill_id}.skill.yaml").write_text(
        yaml.safe_dump(manifest), encoding="utf-8"
    )


def test_fleet_aggregator_per_body_reports(tmp_path: Path) -> None:
    """Aggregator returns one compatibility report per body."""
    registry = BodyRegistryManager(tmp_path)
    registry.create_body("g1", "unitree-g1")
    registry.create_body("arm", "ur5e")

    manifests = [
        SkillManifest(
            skill_id="needs_camera",
            requires={"sensors": {"all_of": [{"id": "head_camera"}]}},
        ),
    ]

    aggregator = FleetCompatibilityAggregator(tmp_path)
    report = aggregator.aggregate(manifests)

    assert set(report.per_body.keys()) == {"g1", "arm"}
    assert report.fleet_summary["total_bodies"] == 2


def test_fleet_aggregator_blocked_on_any(tmp_path: Path) -> None:
    """Skills blocked on any body are listed in fleet_summary."""
    registry = BodyRegistryManager(tmp_path)
    registry.create_body("g1", "unitree-g1")
    registry.create_body("arm", "ur5e")

    manifests = [
        SkillManifest(
            skill_id="needs_camera",
            requires={"sensors": {"all_of": [{"id": "head_camera"}]}},
        ),
    ]

    report = FleetCompatibilityAggregator(tmp_path).aggregate(manifests)
    assert report.fleet_summary["blocked_on_any"] == ["needs_camera@1.0.0"]
    assert report.fleet_summary["blocked_skills"] == 1


def test_fleet_aggregator_compatible_with_all(tmp_path: Path) -> None:
    """Skills satisfied by every body are listed as compatible_with_all."""
    registry = BodyRegistryManager(tmp_path)
    registry.create_body("g1", "unitree-g1")
    registry.create_body("g1b", "unitree-g1")

    manifests = [
        SkillManifest(
            skill_id="needs_head_camera",
            requires={"sensors": {"all_of": [{"id": "head_camera"}]}},
        ),
    ]

    report = FleetCompatibilityAggregator(tmp_path).aggregate(manifests)
    assert report.fleet_summary["compatible_with_all"] == ["needs_head_camera@1.0.0"]
    assert report.fleet_summary["compatible_skills"] == 1


def test_fleet_compat_cli_json_output(tmp_path: Path) -> None:
    """rosclaw body fleet-compat --json returns a valid report."""
    ws = tmp_path / "ws"
    ws.mkdir()
    assert _run("body", "create", "--robot", "unitree-g1", "--name", "g1", "--workspace", str(ws)) == 0
    assert _run("body", "create", "--robot", "ur5e", "--name", "arm", "--workspace", str(ws)) == 0

    _write_skill(ws, "needs_head_camera", {"sensors": {"all_of": [{"id": "head_camera"}]}})

    assert _run("body", "fleet-compat", "--workspace", str(ws), "--json") == 0

    captured = _run_with_capture("body", "fleet-compat", "--workspace", str(ws), "--json")
    data = json.loads(captured)
    assert data["schema_version"] == "rosclaw.fleet_compatibility.v1"
    assert data["fleet_summary"]["total_bodies"] == 2
    assert data["per_body"]["g1"]["summary"]["blocked"] >= 0


def test_fleet_status_cli_json_output(tmp_path: Path) -> None:
    """rosclaw fleet status --json lists all bodies."""
    ws = tmp_path / "ws"
    ws.mkdir()
    assert _run("body", "create", "--robot", "unitree-g1", "--name", "g1", "--workspace", str(ws)) == 0
    assert _run("body", "create", "--robot", "ur5e", "--name", "arm", "--workspace", str(ws)) == 0

    captured = _run_with_capture("fleet", "status", "--workspace", str(ws), "--json")
    data = json.loads(captured)
    assert data["current"] in {"g1", "arm"}
    assert len(data["bodies"]) == 2
    assert {b["body_id"] for b in data["bodies"]} == {"g1", "arm"}


def test_fleet_stop_cli_broadcasts_events(tmp_path: Path) -> None:
    """rosclaw fleet stop succeeds and reports an event per body."""
    ws = tmp_path / "ws"
    ws.mkdir()
    assert _run("body", "create", "--robot", "unitree-g1", "--name", "g1", "--workspace", str(ws)) == 0
    assert _run("body", "create", "--robot", "ur5e", "--name", "arm", "--workspace", str(ws)) == 0

    captured = _run_with_capture("fleet", "stop", "--workspace", str(ws), "--reason", "test")
    assert "Emergency stop event published for body 'g1'" in captured
    assert "Emergency stop event published for body 'arm'" in captured
    assert "Fleet stop broadcast complete" in captured
