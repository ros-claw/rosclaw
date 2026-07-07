"""Tests for the ROSClaw Practice Parquet exporter."""

from __future__ import annotations

import tempfile

import pytest

from rosclaw.practice.config import PracticeConfig, SourceConfig
from rosclaw.practice.coordinator import PracticeCoordinator
from rosclaw.practice.exporters.parquet_exporter import ParquetExporter
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

    for i in range(3):
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
                    timestamp=float(i) * 0.1,
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


def test_parquet_export_creates_file():
    with tempfile.TemporaryDirectory() as tmp:
        practice_id = _run_feedback_session(tmp)
        exporter = ParquetExporter(tmp)
        out = exporter.export(practice_id)
        assert out.exists()


def test_parquet_export_schema_and_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        practice_id = _run_feedback_session(tmp)
        exporter = ParquetExporter(tmp)
        out = exporter.export(practice_id)

        import pyarrow.parquet as pq

        table = pq.read_table(out)
        columns = set(table.column_names)
        assert "observation_state" in columns
        assert "action" in columns
        assert "event_type" in columns
        assert "timestamp_ns" in columns

        meta = table.schema.metadata or {}
        assert meta.get(b"body_id") == b"body_rh56_left"
        assert meta.get(b"practice_id") == practice_id.encode()
        assert int(meta.get(b"event_count", b"0")) == 7


def test_parquet_export_cli(capsys, monkeypatch, tmp_path):
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
                "parquet",
                "--data-root",
                tmp,
            ],
        )
        from rosclaw.cli import main

        rc = main()
        assert rc == 0
        captured = capsys.readouterr()
        assert "Exported Parquet" in captured.out
