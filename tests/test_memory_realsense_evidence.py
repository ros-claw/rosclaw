"""Tests for Memory/How/Know evidence closed loop from RealSense practice episodes."""
from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from rosclaw.cli import cmd_bench_realsense, cmd_how_advise, cmd_know_compile, cmd_memory_ingest
from rosclaw.memory.interface import MemoryInterface
from rosclaw.memory.seekdb_client import SeekDBMemoryClient
from rosclaw.practice.storage.layout import PracticeLayout


@pytest.fixture
def shared_seekdb(monkeypatch: Any) -> SeekDBMemoryClient:
    """Patch MemoryInterface to use a shared in-memory SeekDB."""
    client = SeekDBMemoryClient()
    client.connect()
    monkeypatch.setattr("rosclaw.memory.interface.SeekDBMemoryClient", lambda: client)
    return client


@pytest.fixture
def fake_episode(tmp_path: Path) -> tuple[str, Path]:
    """Create a minimal RealSense practice session directory."""
    episode_id = "prac_20260626T120000Z_abc123"
    data_root = tmp_path / "practice"
    layout = PracticeLayout(data_root)
    session_dir = layout.create_session_dirs(episode_id)

    manifest = {
        "schema_version": "practice.manifest.v1",
        "practice_id": episode_id,
        "session_id": episode_id,
        "robot_id": "dual_lab_01",
        "robot_type": "perception_only",
        "task": {"task_id": "scene_risk_scan", "task_name": "Scene risk scan", "skill_id": "scene_risk_scan"},
        "start_time": "2026-06-26T12:00:00Z",
        "end_time": "2026-06-26T12:00:30Z",
        "duration_ms": 30000.0,
        "sources": {"camera": True, "ros2": True, "provider": True, "sandbox": True},
        "status": {"outcome": "success", "failure_labels": ["depth_invalid"], "reward": 1.0},
    }
    (session_dir / "manifest.yaml").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    # Frame artifacts
    (session_dir / "frames" / "d405" / "color_0001.jpg").write_bytes(b"fake_rgb")
    (session_dir / "frames" / "d405" / "depth_0001.png").write_bytes(b"fake_depth")
    (session_dir / "frames" / "d435i" / "color_0001.jpg").write_bytes(b"fake_rgb")
    (session_dir / "frames" / "d435i" / "depth_0001.png").write_bytes(b"fake_depth")

    # IMU sample
    (session_dir / "imu" / "d435i_imu.jsonl").write_text(
        json.dumps({"timestamp": 0.0, "accel": [0.0, 0.0, 9.8], "gyro": [0.0, 0.0, 0.0]}),
        encoding="utf-8",
    )

    # Provider trace
    (session_dir / "provider" / "trace.jsonl").write_text(
        json.dumps({
            "request_id": "req-1",
            "normalized_risk": 0.3,
            "result": {
                "scene": "lab bench",
                "obstacles": ["cable"],
                "risks": [{"category": "cable_trip", "severity": "warning", "description": "Cable on floor"}],
            },
        }),
        encoding="utf-8",
    )

    # Sandbox trace
    (session_dir / "sandbox" / "decisions.jsonl").write_text(
        json.dumps({"trace_id": "t1", "decision": "ALLOW", "action_type": "sensor_read"}),
        encoding="utf-8",
    )

    return episode_id, data_root


def _args(**kwargs: Any) -> Namespace:
    return Namespace(**kwargs)


def test_memory_ingest_stores_artifacts_and_experience(
    fake_episode: tuple[str, Path],
    shared_seekdb: SeekDBMemoryClient,
) -> None:
    episode_id, data_root = fake_episode

    args = _args(
        episode=episode_id,
        data_root=str(data_root),
        robot="dual_lab_01",
        json=True,
    )
    rc = cmd_memory_ingest(args)
    assert rc == 0

    mem = MemoryInterface("dual_lab_01", seekdb_client=shared_seekdb)
    mem._do_initialize()
    artifacts = mem.find_artifacts_by_episode(episode_id)
    assert len(artifacts) == 7

    by_type: dict[str, int] = {}
    for a in artifacts:
        by_type[a.artifact_type] = by_type.get(a.artifact_type, 0) + 1
    assert by_type.get("rgb") == 2
    assert by_type.get("depth") == 2
    assert by_type.get("imu") == 1
    assert by_type.get("provider_trace") == 1
    assert by_type.get("sandbox_trace") == 1

    exp = mem.get_experience(episode_id)
    assert exp is not None
    assert exp["event_type"] == "practice_episode"
    assert exp["outcome"] == "success"
    meta = exp.get("metadata", {})
    assert meta.get("artifact_count") == 7


def test_how_advise_uses_real_evidence(
    fake_episode: tuple[str, Path],
    shared_seekdb: SeekDBMemoryClient,
    capsys: Any,
) -> None:
    episode_id, data_root = fake_episode
    cmd_memory_ingest(_args(episode=episode_id, data_root=str(data_root), robot="dual_lab_01", json=False))

    args = _args(
        body="dual_lab_01",
        failure="depth_invalid",
        episode=episode_id,
        json=False,
    )
    rc = cmd_how_advise(args)
    assert rc == 0

    out = capsys.readouterr().out
    assert "ROSClaw HOW" in out
    assert episode_id in out
    assert "Evidence:" in out


def test_know_compile_generates_task_card(
    fake_episode: tuple[str, Path],
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    episode_id, data_root = fake_episode

    cards_dir = tmp_path / "task_cards"

    args = _args(
        from_episode=episode_id,
        task="scene_risk_scan",
        body="dual_lab_01",
        task_family="perception",
        data_root=str(data_root),
        output=str(cards_dir),
        json=True,
    )
    rc = cmd_know_compile(args)
    assert rc == 0

    card_path = cards_dir / "scene_risk_scan.json"
    assert card_path.exists()
    card = json.loads(card_path.read_text(encoding="utf-8"))
    assert card["task_id"] == "scene_risk_scan"
    assert "realsense_d405" in card["prerequisites"]
    assert len(card["common_failures"]) >= 1


def test_bench_realsense_skips_without_hardware(capsys: Any) -> None:
    args = _args(duration=1.0, robot="dual_lab_01", data_root="/tmp/rosclaw_practice", output=None, json=True)
    rc = cmd_bench_realsense(args)
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["status"] == "skip"
    assert data["errors"]
