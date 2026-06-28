"""Tests for ``rosclaw memory ingest`` and ``MemoryInterface.ingest_episode``."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from rosclaw.cli import cmd_memory_ingest
from rosclaw.memory.interface import MemoryInterface
from rosclaw.memory.seekdb_client import SeekDBMemoryClient


class _SpyMemoryInterface:
    """Capture ingest calls for CLI tests without touching real storage."""

    def __init__(self, robot_id: str, *args, **kwargs):
        self.robot_id = robot_id
        self.calls = []

    def _do_initialize(self):
        pass

    def ingest_episode(self, episode_id: str, data_root: str | None = None) -> dict:
        self.calls.append({"episode_id": episode_id, "data_root": data_root})
        return {
            "status": "success",
            "experience_id": "exp-123",
            "event_count": 3,
            "outcome": "success",
        }


def _make_episode(data_root: Path, episode_id: str) -> Path:
    """Create a minimal practice episode on disk."""
    root = Path(data_root)
    session_dir = root / "sessions" / episode_id
    raw_dir = session_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    episode = {
        "episode_id": episode_id,
        "robot_id": "d405_lab_01",
        "robot_type": "realsense_d405",
        "outcome": "SUCCESS",
        "duration_ms": 1234,
        "task": {"task_id": "realsense_inspection"},
    }
    (session_dir / "episode.json").write_text(json.dumps(episode), encoding="utf-8")

    events = [
        {
            "event_type": "rgbd_frame",
            "source": "camera",
            "payload": {"rgb_ref": str(session_dir / "color.png")},
        },
        {
            "event_type": "provider_result",
            "source": "provider",
            "payload": {"provider_id": "cosmos-reason2-lan"},
        },
        {
            "event_type": "sandbox_decision",
            "source": "sandbox",
            "payload": {"decision": "ALLOW"},
        },
    ]
    (raw_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(ev) for ev in events), encoding="utf-8"
    )

    provider = {
        "provider_id": "cosmos-reason2-lan",
        "capability": "vlm.risk_assessment",
        "normalized": {"risk_score": 0.1, "requires_guard": False},
    }
    (session_dir / "provider").mkdir(parents=True, exist_ok=True)
    (session_dir / "provider" / "provider_result.json").write_text(
        json.dumps(provider), encoding="utf-8"
    )

    (session_dir / "color.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    return session_dir


class TestMemoryIngestPracticeEpisode:
    """Phase H memory ingestion tests."""

    def test_ingest_episode_stores_experience_and_artifact(self, tmp_path):
        episode_id = "ep_001"
        data_root = tmp_path / "practice"
        _make_episode(data_root, episode_id)

        client = SeekDBMemoryClient()
        mem = MemoryInterface(robot_id="d405_lab_01", seekdb_client=client)
        mem._do_initialize()

        result = mem.ingest_episode(episode_id, data_root=str(data_root))

        assert result["status"] == "success"
        assert result["event_count"] == 3
        assert result["outcome"] == "success"
        assert result["experience_id"]

        experiences = client.query("experience_graph")
        assert len(experiences) == 1
        assert experiences[0]["event_type"] == "practice_episode"
        assert "realsense_inspection" in experiences[0]["instruction"]

        artifacts = client.query("artifacts")
        assert len(artifacts) == 1
        assert artifacts[0]["artifact_type"] == "rgb_frame"
        assert artifacts[0]["episode_id"] == episode_id

    def test_ingest_episode_missing_session_returns_error(self, tmp_path):
        client = SeekDBMemoryClient()
        mem = MemoryInterface(robot_id="d405_lab_01", seekdb_client=client)
        mem._do_initialize()

        result = mem.ingest_episode("missing", data_root=str(tmp_path))

        assert result["status"] == "error"
        assert "session not found" in result["reason"]

    def test_cli_memory_ingest(self, monkeypatch, tmp_path):
        episode_id = "ep_cli_001"
        data_root = tmp_path / "practice"
        _make_episode(data_root, episode_id)

        spy = _SpyMemoryInterface("cli")
        monkeypatch.setattr("rosclaw.memory.interface.MemoryInterface", lambda _rid, **kwargs: spy)

        args = SimpleNamespace(
            episode_id=episode_id,
            data_root=str(data_root),
        )
        assert cmd_memory_ingest(args) == 0
        assert len(spy.calls) == 1
        assert spy.calls[0]["episode_id"] == episode_id
        assert spy.calls[0]["data_root"] == str(data_root)
