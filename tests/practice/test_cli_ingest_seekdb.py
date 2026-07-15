"""CLI integration tests for `rosclaw practice ingest-seekdb`."""

from __future__ import annotations

import tempfile

from rosclaw.cli import main
from rosclaw.practice.config import PracticeConfig, SourceConfig
from rosclaw.practice.coordinator import PracticeCoordinator
from rosclaw.practice.recorder import PracticeRecorder
from rosclaw.practice.schemas import (
    FailureEventPayload,
    HowInterventionPayload,
    PracticeEventEnvelope,
)
from rosclaw.runtime.bus import RuntimeBus


def _run_session(tmp: str):
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
    return practice_id


def test_cli_practice_ingest_seekdb(capsys, monkeypatch, tmp_path):
    seekdb_path = tmp_path / "seekdb.sqlite"
    with tempfile.TemporaryDirectory() as tmp:
        practice_id = _run_session(tmp)
        monkeypatch.setattr(
            "sys.argv",
            [
                "rosclaw",
                "practice",
                "ingest-seekdb",
                practice_id,
                "--data-root",
                tmp,
                "--seekdb-path",
                str(seekdb_path),
            ],
        )
        rc = main()
        assert rc == 0
        captured = capsys.readouterr()
        assert "SeekDB Ingestion Report" in captured.out
        assert "failures: 1" in captured.out


def test_cli_practice_ingest_seekdb_json(capsys, monkeypatch, tmp_path):
    seekdb_path = tmp_path / "seekdb.sqlite"
    with tempfile.TemporaryDirectory() as tmp:
        practice_id = _run_session(tmp)
        monkeypatch.setattr(
            "sys.argv",
            [
                "rosclaw",
                "practice",
                "ingest-seekdb",
                practice_id,
                "--data-root",
                tmp,
                "--seekdb-path",
                str(seekdb_path),
                "--json",
            ],
        )
        rc = main()
        assert rc == 0
        captured = capsys.readouterr()
        assert '"success": true' in captured.out
        assert '"failures": 1' in captured.out


def test_cli_practice_ingest_seekdb_missing(capsys, monkeypatch, tmp_path):
    seekdb_path = tmp_path / "seekdb.sqlite"
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(
            "sys.argv",
            [
                "rosclaw",
                "practice",
                "ingest-seekdb",
                "does_not_exist",
                "--data-root",
                tmp,
                "--seekdb-path",
                str(seekdb_path),
            ],
        )
        rc = main()
        assert rc == 1
        captured = capsys.readouterr()
        assert "Ingest failed" in captured.err


def test_cli_practice_ingest_seekdb_connection_error_is_clear(capsys, monkeypatch, tmp_path):
    seekdb_path = tmp_path / "seekdb_dir"
    seekdb_path.mkdir()
    with tempfile.TemporaryDirectory() as tmp:
        practice_id = _run_session(tmp)
        monkeypatch.setattr(
            "sys.argv",
            [
                "rosclaw",
                "practice",
                "ingest-seekdb",
                practice_id,
                "--data-root",
                tmp,
                "--seekdb-path",
                str(seekdb_path),
            ],
        )
        rc = main()
        assert rc == 1
        captured = capsys.readouterr()
        assert "SeekDB connection failed" in captured.err


def test_cli_practice_ingest_seekdb_url_backend(capsys, monkeypatch):
    from rosclaw.memory.seekdb_client import SeekDBMemoryClient

    client = SeekDBMemoryClient()
    captured_url = {}

    def make_client(url: str):
        captured_url["url"] = url
        return client

    monkeypatch.setattr("rosclaw.memory.seekdb_client.SeekDBMySQLClient", make_client)
    with tempfile.TemporaryDirectory() as tmp:
        practice_id = _run_session(tmp)
        monkeypatch.setattr(
            "sys.argv",
            [
                "rosclaw",
                "practice",
                "ingest-seekdb",
                practice_id,
                "--data-root",
                tmp,
                "--seekdb-url",
                "mysql://root@127.0.0.1:2881/rosclaw_test",
                "--json",
            ],
        )

        assert main() == 0

    assert captured_url["url"] == "mysql://root@127.0.0.1:2881/rosclaw_test"
    assert client.count("failures") == 1
    assert '"success": true' in capsys.readouterr().out
