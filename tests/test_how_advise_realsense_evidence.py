"""Tests for ``rosclaw how advise`` and ``HeuristicEngine.advise``."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from rosclaw.cli import cmd_how_advise
from rosclaw.how.engine import HeuristicEngine
from rosclaw.memory.seekdb_client import SeekDBMemoryClient


def _make_advise_episode(data_root: Path, episode_id: str) -> Path:
    """Create a practice episode with a low_fps camera event."""
    root = Path(data_root)
    session_dir = root / "sessions" / episode_id
    raw_dir = session_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    episode = {
        "episode_id": episode_id,
        "robot_id": "d405_lab_01",
        "robot_type": "realsense_d405",
        "outcome": "DEGRADED",
        "duration_ms": 5000,
        "task": {"task_id": "realsense_inspection"},
    }
    (session_dir / "episode.json").write_text(json.dumps(episode), encoding="utf-8")

    events = [
        {
            "event_type": "rgbd_frame",
            "source": "camera",
            "payload": {"fps": 5.0, "usb_mode": "USB2", "degraded": True},
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
    return session_dir


class _FakeHeuristicEngine:
    """Stub HOW engine for CLI tests."""

    def __init__(self, *args, **kwargs):
        pass

    async def initialize(self):
        pass

    async def advise(self, body_id: str, failure: str, episode_id: str, data_root: str | None = None):
        return {
            "body_id": body_id,
            "failure": failure,
            "episode_id": episode_id,
            "intervention": {
                "rule_id": "rule_low_fps",
                "action": "Switch to USB3 cable/port and reduce resolution.",
                "priority": 2,
            },
            "evidence": {"event_count": 2, "sources": ["camera", "sandbox"]},
        }


class TestHowAdviseRealSenseEvidence:
    """Phase H HOW evidence-backed advice tests."""

    def test_advise_loads_episode_events(self, tmp_path):
        episode_id = "ep_low_fps_01"
        data_root = tmp_path / "practice"
        _make_advise_episode(data_root, episode_id)

        engine = HeuristicEngine(seekdb_client=SeekDBMemoryClient())
        result = asyncio.run(
            engine.advise(
                body_id="d405_lab_01",
                failure="low_fps",
                episode_id=episode_id,
                data_root=str(data_root),
            )
        )

        assert result["body_id"] == "d405_lab_01"
        assert result["failure"] == "low_fps"
        assert result["episode_id"] == episode_id
        assert result["evidence"]["event_count"] == 2
        assert "camera" in result["evidence"]["sources"]
        assert "intervention" in result
        assert result["intervention"].get("action")

    def test_advise_falls_back_when_no_events(self, tmp_path):
        engine = HeuristicEngine(seekdb_client=SeekDBMemoryClient())
        result = asyncio.run(
            engine.advise(
                body_id="d405_lab_01",
                failure="low_fps",
                episode_id="missing_episode",
                data_root=str(tmp_path),
            )
        )

        assert result["evidence"]["event_count"] == 0
        assert result["intervention"]["rule_id"] == "fallback"

    def test_cli_how_advise(self, monkeypatch, capsys, tmp_path):
        episode_id = "ep_cli_low_fps"
        data_root = tmp_path / "practice"
        _make_advise_episode(data_root, episode_id)

        monkeypatch.setattr("rosclaw.how.engine.HeuristicEngine", _FakeHeuristicEngine)

        args = SimpleNamespace(
            body="d405_lab_01",
            failure="low_fps",
            episode_id=episode_id,
            data_root=str(data_root),
            json=False,
        )
        assert cmd_how_advise(args) == 0

        captured = capsys.readouterr().out
        assert "Evidence-backed Intervention" in captured
        assert "Switch to USB3" in captured

    def test_cli_how_advise_json(self, monkeypatch, capsys, tmp_path):
        episode_id = "ep_cli_low_fps_json"
        data_root = tmp_path / "practice"
        _make_advise_episode(data_root, episode_id)

        monkeypatch.setattr("rosclaw.how.engine.HeuristicEngine", _FakeHeuristicEngine)

        args = SimpleNamespace(
            body="d405_lab_01",
            failure="low_fps",
            episode_id=episode_id,
            data_root=str(data_root),
            json=True,
        )
        assert cmd_how_advise(args) == 0

        captured = capsys.readouterr().out
        result = json.loads(captured)
        assert result["body_id"] == "d405_lab_01"
        assert result["intervention"]["rule_id"] == "rule_low_fps"
