"""CLI integration tests for `rosclaw practice query`."""

from __future__ import annotations

import tempfile

from rosclaw.cli import main
from rosclaw.practice.config import PracticeConfig, SourceConfig
from rosclaw.practice.coordinator import PracticeCoordinator
from rosclaw.practice.recorder import PracticeRecorder
from rosclaw.practice.schemas import (
    FailureEventPayload,
    HowInterventionPayload,
    PhysicalFeedbackPayload,
    PracticeEventEnvelope,
)
from rosclaw.practice.seekdb_ingestor import SeekDBIngestor
from rosclaw.runtime.bus import RuntimeBus


def _ingest_session(tmp: str, seekdb_path: str) -> str:
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

    coord.emit_event(
        PracticeEventEnvelope(
            practice_id=practice_id,
            robot_id="test_bot",
            body_id="body_rh56_left",
            source="runtime",
            event_type="physical_feedback_event",
            payload=PhysicalFeedbackPayload(
                frame_id="f1",
                body_id="body_rh56_left",
                timestamp=1.0,
                force_net={"thumb": 100.0},
                primary_event="desired_contact",
            ).model_dump(),
        )
    )
    coord.emit_event(
        PracticeEventEnvelope(
            practice_id=practice_id,
            robot_id="test_bot",
            source="runtime",
            event_type="failure_event",
            payload=FailureEventPayload(
                failure_id="fail_1",
                failure_type="over_contact",
                severity="high",
                source="sandbox",
                description="too much force",
            ).model_dump(),
        )
    )
    coord.emit_event(
        PracticeEventEnvelope(
            practice_id=practice_id,
            robot_id="test_bot",
            source="runtime",
            event_type="how_intervention_event",
            payload=HowInterventionPayload(
                intervention_id="how_1",
                failure_id="fail_1",
                description="back off",
                action_taken={"delta": -10.0},
                outcome="resolved",
            ).model_dump(),
        )
    )
    coord.stop()
    recorder.stop()

    from rosclaw.memory.seekdb_client import SeekDBSQLiteClient

    client = SeekDBSQLiteClient(seekdb_path)
    ingestor = SeekDBIngestor(tmp, seekdb_client=client)
    ingestor.ingest_practice(practice_id)
    ingestor.close()
    return practice_id


def test_cli_query_failures(capsys, monkeypatch, tmp_path):
    seekdb_path = str(tmp_path / "seekdb.sqlite")
    with tempfile.TemporaryDirectory() as tmp:
        _ingest_session(tmp, seekdb_path)
        monkeypatch.setattr(
            "sys.argv",
            [
                "rosclaw",
                "practice",
                "query",
                "failures",
                "--body-id",
                "body_rh56_left",
                "--failure-type",
                "over_contact",
                "--data-root",
                tmp,
                "--seekdb-path",
                seekdb_path,
            ],
        )
        rc = main()
        assert rc == 0
        captured = capsys.readouterr()
        assert "Query results: failures" in captured.out
        assert "over_contact" in captured.out


def test_cli_query_interventions(capsys, monkeypatch, tmp_path):
    seekdb_path = str(tmp_path / "seekdb.sqlite")
    with tempfile.TemporaryDirectory() as tmp:
        _ingest_session(tmp, seekdb_path)
        monkeypatch.setattr(
            "sys.argv",
            [
                "rosclaw",
                "practice",
                "query",
                "interventions",
                "--failure-type",
                "over_contact",
                "--data-root",
                tmp,
                "--seekdb-path",
                seekdb_path,
                "--json",
            ],
        )
        rc = main()
        assert rc == 0
        captured = capsys.readouterr()
        assert '"outcome": "resolved"' in captured.out


def test_cli_query_body_cognition(capsys, monkeypatch, tmp_path):
    seekdb_path = str(tmp_path / "seekdb.sqlite")
    with tempfile.TemporaryDirectory() as tmp:
        _ingest_session(tmp, seekdb_path)
        monkeypatch.setattr(
            "sys.argv",
            [
                "rosclaw",
                "practice",
                "query",
                "body-cognition",
                "--body-id",
                "body_rh56_left",
                "--data-root",
                tmp,
                "--seekdb-path",
                seekdb_path,
            ],
        )
        rc = main()
        assert rc == 0
        captured = capsys.readouterr()
        assert "Query results: body-cognition" in captured.out


def test_cli_query_episodes(capsys, monkeypatch, tmp_path):
    seekdb_path = str(tmp_path / "seekdb.sqlite")
    with tempfile.TemporaryDirectory() as tmp:
        _ingest_session(tmp, seekdb_path)
        monkeypatch.setattr(
            "sys.argv",
            [
                "rosclaw",
                "practice",
                "query",
                "episodes",
                "--body-id",
                "body_rh56_left",
                "--data-root",
                tmp,
            ],
        )
        rc = main()
        assert rc == 0
        captured = capsys.readouterr()
        assert "Query results: episodes" in captured.out


def test_cli_query_backend_error_is_clear(capsys, monkeypatch, tmp_path):
    seekdb_path = tmp_path / "seekdb_dir"
    seekdb_path.mkdir()
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(
            "sys.argv",
            [
                "rosclaw",
                "practice",
                "query",
                "failures",
                "--data-root",
                tmp,
                "--seekdb-path",
                str(seekdb_path),
            ],
        )
        rc = main()
        assert rc == 1
        captured = capsys.readouterr()
        assert "Query backend unavailable" in captured.err
