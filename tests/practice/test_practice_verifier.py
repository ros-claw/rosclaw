"""Tests for PracticeVerifier."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from rosclaw.practice.config import PracticeConfig, SourceConfig
from rosclaw.practice.coordinator import PracticeCoordinator
from rosclaw.practice.recorder import PracticeRecorder
from rosclaw.practice.storage.catalog import PracticeCatalog
from rosclaw.practice.storage.layout import PracticeLayout
from rosclaw.practice.verifier import PracticeVerifier
from rosclaw.runtime.bus import RuntimeBus


def _run_coordinator_with_recorder(tmp: str):
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
    return coord


def test_verifier_passes_for_clean_session():
    with tempfile.TemporaryDirectory() as tmp:
        coord = _run_coordinator_with_recorder(tmp)
        practice_id = coord.summary.practice_id

        verifier = PracticeVerifier(tmp)
        report = verifier.verify(practice_id)
        assert report.passed, report.issues
        assert "catalog_exists" in report.checked


def test_verifier_fails_for_missing_practice():
    with tempfile.TemporaryDirectory() as tmp:
        # Create an empty catalog so the verifier can look inside it.
        catalog = PracticeCatalog(Path(tmp) / "indexes" / "practice_catalog.sqlite")
        catalog.close()

        verifier = PracticeVerifier(tmp)
        report = verifier.verify("does_not_exist")
        assert not report.passed
        assert any(i.scope == "practice" for i in report.issues)


def test_verifier_detects_tampered_artifact():
    with tempfile.TemporaryDirectory() as tmp:
        coord = _run_coordinator_with_recorder(tmp)
        practice_id = coord.summary.practice_id

        # Tamper with the summary YAML artifact
        catalog = PracticeCatalog(Path(tmp) / "indexes" / "practice_catalog.sqlite")
        practice = catalog.get_practice(practice_id)
        session_id = practice["session_id"]
        episode_id = practice["episode_id"]
        summary_path = (
            Path(tmp)
            / "sessions"
            / session_id
            / "episodes"
            / episode_id
            / "artifacts"
            / "summary"
            / f"summary_{episode_id}.yaml"
        )
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write("# tamper\n")
        catalog.close()

        verifier = PracticeVerifier(tmp)
        report = verifier.verify(practice_id)
        assert not report.passed
        assert any("sha256 mismatch" in i.message for i in report.issues)


def test_verifier_required_event_types():
    with tempfile.TemporaryDirectory() as tmp:
        coord = _run_coordinator_with_recorder(tmp)
        practice_id = coord.summary.practice_id

        verifier = PracticeVerifier(tmp)
        report = verifier.verify(practice_id, required_event_types=["runtime.start", "runtime.stop"])
        assert report.passed

        report2 = verifier.verify(practice_id, required_event_types=["nonexistent_event"])
        assert not report2.passed
        assert any("missing required event type" in i.message for i in report2.issues)


def test_verifier_event_count_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        coord = _run_coordinator_with_recorder(tmp)
        practice_id = coord.summary.practice_id

        # Append an extra line to events.jsonl but don't update catalog
        layout = PracticeLayout(tmp)
        events_path = layout.events_jsonl_path(practice_id)
        with open(events_path, "a", encoding="utf-8") as f:
            f.write('{"event_type":"extra"}\n')

        verifier = PracticeVerifier(tmp)
        report = verifier.verify(practice_id)
        assert not report.passed
        assert any("event count" in i.message for i in report.issues)


def test_verifier_strict_detects_missing_envelope_fields():
    with tempfile.TemporaryDirectory() as tmp:
        coord = _run_coordinator_with_recorder(tmp)
        practice_id = coord.summary.practice_id
        layout = PracticeLayout(tmp)
        events_path = layout.events_jsonl_path(practice_id)

        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
        events[0].pop("event_id", None)
        events[0].pop("trace_id", None)
        events[0].pop("timestamp_ns", None)
        events_path.write_text(
            "\n".join(json.dumps(event) for event in events) + "\n",
            encoding="utf-8",
        )

        verifier = PracticeVerifier(tmp)
        normal_report = verifier.verify(practice_id)
        assert normal_report.passed
        assert any(i.level == "warning" and "event_id" in i.message for i in normal_report.issues)

        strict_report = verifier.verify(practice_id, strict=True)
        assert not strict_report.passed
        assert any(i.level == "error" and "event_id" in i.message for i in strict_report.issues)
        assert any(i.level == "error" and "trace_id" in i.message for i in strict_report.issues)
        assert any(i.level == "error" and "timestamp_ns" in i.message for i in strict_report.issues)
