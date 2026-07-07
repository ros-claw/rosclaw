"""CLI integration tests for `rosclaw practice verify`."""

from __future__ import annotations

import tempfile

from rosclaw.cli import main
from rosclaw.practice.config import PracticeConfig, SourceConfig
from rosclaw.practice.coordinator import PracticeCoordinator
from rosclaw.practice.recorder import PracticeRecorder
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
    coord.stop()
    recorder.stop()
    return coord.summary.practice_id


def test_cli_practice_verify_passes(capsys, monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        practice_id = _run_session(tmp)
        monkeypatch.setattr(
            "sys.argv", ["rosclaw", "practice", "verify", practice_id, "--data-root", tmp]
        )
        rc = main()
        assert rc == 0
        captured = capsys.readouterr()
        assert "Passed: True" in captured.out


def test_cli_practice_verify_strict(capsys, monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        practice_id = _run_session(tmp)
        monkeypatch.setattr(
            "sys.argv",
            ["rosclaw", "practice", "verify", practice_id, "--data-root", tmp, "--strict"],
        )
        main()
        captured = capsys.readouterr()
        assert "Passed:" in captured.out


def test_cli_practice_verify_json(capsys, monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        practice_id = _run_session(tmp)
        monkeypatch.setattr(
            "sys.argv",
            ["rosclaw", "practice", "verify", practice_id, "--data-root", tmp, "--json"],
        )
        rc = main()
        assert rc == 0
        captured = capsys.readouterr()
        assert '"passed": true' in captured.out.lower()


def test_cli_practice_verify_missing_practice(capsys, monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(
            "sys.argv",
            ["rosclaw", "practice", "verify", "does_not_exist", "--data-root", tmp],
        )
        rc = main()
        assert rc == 1
        captured = capsys.readouterr()
        assert "Passed: False" in captured.out
