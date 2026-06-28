"""Tests for ``rosclaw know compile`` and ``KnowledgeInterface.compile_task_card``."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from rosclaw.cli import cmd_know_compile
from rosclaw.know.interface import KnowledgeInterface


def _make_episode_for_know(data_root: Path, episode_id: str) -> Path:
    """Create a minimal practice episode for task-card compilation."""
    root = Path(data_root)
    session_dir = root / "sessions" / episode_id
    raw_dir = session_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    episode = {
        "episode_id": episode_id,
        "robot_id": "d405_lab_01",
        "robot_type": "realsense_d405",
        "outcome": "SUCCESS",
        "duration_ms": 2000,
        "task": {"task_id": "inspect surface"},
    }
    (session_dir / "episode.json").write_text(json.dumps(episode), encoding="utf-8")

    events = [
        {"event_type": "rgbd_frame", "source": "camera", "payload": {}},
        {"event_type": "provider_result", "source": "provider", "payload": {}},
    ]
    (raw_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(ev) for ev in events), encoding="utf-8"
    )
    return session_dir


class TestKnowCompileRealSenseTaskCard:
    """Phase H KNOW task-card tests."""

    def test_compile_task_card_reads_episode_evidence(self, tmp_path):
        episode_id = "ep_inspect_01"
        data_root = tmp_path / "practice"
        _make_episode_for_know(data_root, episode_id)

        know = KnowledgeInterface(robot_id="d405_lab_01")
        know._do_initialize()

        card = know.compile_task_card(
            task="inspect surface",
            episode_id=episode_id,
            data_root=str(data_root),
        )

        assert card["schema_version"] == "rosclaw.task_card.v1"
        assert card["task"] == "inspect surface"
        assert card["episode_id"] == episode_id
        assert card["robot_id"] == "d405_lab_01"
        assert card["outcome"] == "SUCCESS"
        assert card["evidence"]["event_count"] == 2
        assert "camera" in card["evidence"]["sources"]
        assert "position_sensor" in card["steps"]

    def test_compile_task_card_missing_episode_returns_empty_evidence(self, tmp_path):
        know = KnowledgeInterface(robot_id="d405_lab_01")
        know._do_initialize()

        card = know.compile_task_card(
            task="inspect surface",
            episode_id="missing",
            data_root=str(tmp_path),
        )

        assert card["evidence"]["event_count"] == 0
        assert card["steps"]  # canonical decomposition is still returned
        assert "position_sensor" in card["steps"]

    def test_cli_know_compile(self, capsys, tmp_path):
        episode_id = "ep_cli_inspect"
        data_root = tmp_path / "practice"
        _make_episode_for_know(data_root, episode_id)

        args = SimpleNamespace(
            task="inspect surface",
            episode_id=episode_id,
            data_root=str(data_root),
            json=False,
        )
        assert cmd_know_compile(args) == 0

        captured = capsys.readouterr().out
        assert "Grounded Task Card" in captured
        assert "position_sensor" in captured

    def test_cli_know_compile_json(self, capsys, tmp_path):
        episode_id = "ep_cli_inspect_json"
        data_root = tmp_path / "practice"
        _make_episode_for_know(data_root, episode_id)

        args = SimpleNamespace(
            task="inspect surface",
            episode_id=episode_id,
            data_root=str(data_root),
            json=True,
        )
        assert cmd_know_compile(args) == 0

        captured = capsys.readouterr().out
        card = json.loads(captured)
        assert card["task"] == "inspect surface"
        assert card["robot_id"] == "d405_lab_01"
