"""Tests for the ROSClaw Practice LeRobot exporter structure."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from rosclaw.practice.config import PracticeConfig, SourceConfig
from rosclaw.practice.coordinator import PracticeCoordinator
from rosclaw.practice.exporters.lerobot_exporter import LeRobotExporter
from rosclaw.practice.recorder import PracticeRecorder
from rosclaw.practice.schemas import PhysicalFeedbackPayload, PracticeEventEnvelope
from rosclaw.runtime.bus import RuntimeBus

pytest.importorskip("pyarrow")


def _run_feedback_session(tmp: str) -> str:
    bus = RuntimeBus()
    recorder = PracticeRecorder(bus, data_root=tmp, publish_to_event_bus=False)
    recorder.initialize()
    recorder.start()
    cfg = PracticeConfig(
        robot_id="test_bot",
        task_name="ok_contact",
        data_root=tmp,
        sources=SourceConfig(agent=True, runtime=True),
        mock=True,
        publish_to_event_bus=False,
    )
    coord = PracticeCoordinator(cfg, runtime_bus=bus, recorder=recorder)
    coord.initialize()
    coord.start()
    practice_id = coord._session.practice_id

    for i in range(4):
        coord.emit_event(
            PracticeEventEnvelope(
                practice_id=practice_id,
                robot_id="test_bot",
                body_id="body_rh56_left",
                source="runtime",
                event_type="physical_feedback_event",
                payload=PhysicalFeedbackPayload(
                    frame_id=f"f{i}",
                    body_id="body_rh56_left",
                    timestamp=float(i) * 0.05,
                    target={"thumb": float(i), "index": float(i + 1)},
                    actual={"thumb": float(i) + 0.5, "index": float(i) + 1.5},
                    force_net={"thumb": 100.0 + i, "index": 110.0 + i},
                    primary_event="desired_contact",
                ).model_dump(),
            )
        )

    coord.stop()
    recorder.stop()
    return practice_id


def test_lerobot_export_structure():
    with tempfile.TemporaryDirectory() as tmp:
        practice_id = _run_feedback_session(tmp)
        exporter = LeRobotExporter(tmp)
        out = exporter.export(practice_id)

        assert out.exists()
        assert (out / "data" / "observation.state.parquet").exists()
        assert (out / "data" / "action.parquet").exists()
        assert (out / "data" / "episode_index.parquet").exists()
        assert (out / "data" / "timestamp.parquet").exists()
        assert (out / "meta" / "info.json").exists()
        assert (out / "meta" / "tasks.jsonl").exists()
        assert (out / "rosclaw_extra.jsonl").exists()


def test_lerobot_export_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        practice_id = _run_feedback_session(tmp)
        exporter = LeRobotExporter(tmp)
        out = exporter.export(practice_id)

        with open(out / "meta" / "info.json", encoding="utf-8") as f:
            info = json.load(f)

        assert info["total_frames"] == 4
        assert info["total_episodes"] == 1
        assert info["body_id"] == "body_rh56_left"
        assert "observation.state" in info["features"]
        assert "action" in info["features"]
        assert info["features"]["observation.state"]["shape"] == [2]
        assert set(info["features"]["observation.state"]["names"]) == {"thumb", "index"}


def test_lerobot_export_cli(capsys, monkeypatch, tmp_path):
    with tempfile.TemporaryDirectory() as tmp:
        practice_id = _run_feedback_session(tmp)
        monkeypatch.setattr(
            "sys.argv",
            [
                "rosclaw",
                "practice",
                "export",
                practice_id,
                "--format",
                "lerobot",
                "--data-root",
                tmp,
            ],
        )
        from rosclaw.cli import main

        rc = main()
        assert rc == 0
        captured = capsys.readouterr()
        assert "Exported LeRobot dataset" in captured.out

        out = Path(tmp) / "datasets" / "lerobot" / practice_id
        assert (out / "data" / "observation.state.parquet").exists()
