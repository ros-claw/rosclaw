"""CLI tests for recording a Practice fixture."""

from __future__ import annotations

from pathlib import Path

from rosclaw.cli import main
from rosclaw.practice.verifier import PracticeVerifier

FIXTURE = Path("tests/fixtures/practice/rh56_minimal_loop.json")


def test_cli_record_fixture_verifies_strict(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "rosclaw",
            "practice",
            "record",
            "--fixture",
            str(FIXTURE),
            "--out",
            str(tmp_path),
        ],
    )
    rc = main()
    assert rc == 0
    captured = capsys.readouterr()
    assert "practice_rh56_minimal_loop" in captured.out

    report = PracticeVerifier(tmp_path).verify("practice_rh56_minimal_loop", strict=True)
    assert report.passed, report.issues

    events_path = tmp_path / "sessions" / "practice_rh56_minimal_loop" / "raw" / "events.jsonl"
    assert events_path.exists()


def test_cli_record_fixture_requires_envelope_fields(tmp_path, monkeypatch, capsys):
    bad_fixture = tmp_path / "bad_fixture.json"
    bad_fixture.write_text(
        '{"practice_id":"practice_bad","events":[{"event_type":"physical_feedback_event"}]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "rosclaw",
            "practice",
            "record",
            "--fixture",
            str(bad_fixture),
            "--out",
            str(tmp_path / "out"),
        ],
    )
    rc = main()
    assert rc == 1
    captured = capsys.readouterr()
    assert "missing required fields" in captured.err
