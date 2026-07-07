"""Tests for PracticeDistiller."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from rosclaw.practice.config import PracticeConfig, SourceConfig
from rosclaw.practice.coordinator import PracticeCoordinator
from rosclaw.practice.distiller import PracticeDistiller
from rosclaw.practice.recorder import PracticeRecorder
from rosclaw.practice.schemas import (
    FailureEventPayload,
    HowInterventionPayload,
    PhysicalFeedbackPayload,
    PracticeEventEnvelope,
)
from rosclaw.runtime.bus import RuntimeBus


def _run_session_with_events(tmp: str, events: list[dict[str, Any]]):
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

    for ev in events:
        coord.emit_event(PracticeEventEnvelope(**ev))

    coord.stop()
    recorder.stop()
    return coord.summary.practice_id


def test_distill_body_cognition_from_feedback_events():
    with tempfile.TemporaryDirectory() as tmp:
        events = [
            PracticeEventEnvelope(
                practice_id="prac_1",
                robot_id="test_bot",
                body_id="body_rh56_left",
                source="runtime",
                event_type="physical_feedback_event",
                payload=PhysicalFeedbackPayload(
                    frame_id="f1",
                    body_id="body_rh56_left",
                    timestamp=1.0,
                    force_net={"thumb": 100.0, "index": 120.0},
                    primary_event="desired_contact",
                ).model_dump(),
            ).model_dump(mode="json"),
            PracticeEventEnvelope(
                practice_id="prac_1",
                robot_id="test_bot",
                body_id="body_rh56_left",
                source="runtime",
                event_type="physical_feedback_event",
                payload=PhysicalFeedbackPayload(
                    frame_id="f2",
                    body_id="body_rh56_left",
                    timestamp=2.0,
                    force_net={"thumb": 110.0, "index": 130.0},
                    primary_event="desired_contact",
                ).model_dump(),
            ).model_dump(mode="json"),
        ]
        practice_id = _run_session_with_events(tmp, events)

        distiller = PracticeDistiller(tmp)
        result = distiller.distill(practice_id)

        assert result.body_cognition["body_id"] == "body_rh56_left"
        assert "desired_contact_region" in result.body_cognition["known_traits"]
        assert result.body_cognition["force_model"]["thumb"]["count"] == 2


def test_distill_failure_and_how_events():
    with tempfile.TemporaryDirectory() as tmp:
        events = [
            PracticeEventEnvelope(
                practice_id="prac_1",
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
            ).model_dump(mode="json"),
            PracticeEventEnvelope(
                practice_id="prac_1",
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
            ).model_dump(mode="json"),
        ]
        practice_id = _run_session_with_events(tmp, events)

        distiller = PracticeDistiller(tmp)
        result = distiller.distill(practice_id)

        assert len(result.failures) == 1
        assert len(result.how_interventions) == 1


def test_distill_writes_artifacts():
    with tempfile.TemporaryDirectory() as tmp:
        events = [
            PracticeEventEnvelope(
                practice_id="prac_1",
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
            ).model_dump(mode="json"),
        ]
        practice_id = _run_session_with_events(tmp, events)

        distiller = PracticeDistiller(tmp)
        result = distiller.distill(practice_id)

        assert "failures" in result.artifact_refs
        assert Path(result.artifact_refs["failures"]).exists()


def test_distill_no_events_returns_empty():
    with tempfile.TemporaryDirectory() as tmp:
        practice_id = _run_session_with_events(tmp, [])
        distiller = PracticeDistiller(tmp)
        result = distiller.distill(practice_id)
        assert result.failures == []
        assert result.body_cognition.get("known_traits") == []
