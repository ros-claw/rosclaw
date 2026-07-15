"""Tests for SeekDBIngestor."""

from __future__ import annotations

import tempfile
from typing import Any

from rosclaw.memory.seekdb_client import SeekDBMemoryClient
from rosclaw.practice.config import PracticeConfig, SourceConfig
from rosclaw.practice.coordinator import PracticeCoordinator
from rosclaw.practice.recorder import PracticeRecorder
from rosclaw.practice.schemas import (
    CandidatePolicyPayload,
    FailureEventPayload,
    HowInterventionPayload,
    PhysicalFeedbackPayload,
    PracticeEventEnvelope,
    PromotionResultPayload,
    Sim2RealDeltaPayload,
)
from rosclaw.practice.seekdb_ingestor import SeekDBIngestor
from rosclaw.runtime.bus import RuntimeBus


def _run_session_with_events(tmp: str, events: list[dict[str, Any]]) -> str:
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


def test_ingest_practice_writes_episode_and_failures():
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

        client = SeekDBMemoryClient()
        ingestor = SeekDBIngestor(tmp, seekdb_client=client)
        report = ingestor.ingest_practice(practice_id)
        ingestor.close()

        assert report.success
        assert report.table_counts.get("episodes") == 1
        assert report.table_counts.get("failures") == 1
        assert report.table_counts.get("how_interventions") == 1

        failures = client.query("failures", filters={"failure_type": "over_contact"})
        assert len(failures) == 1
        assert failures[0]["root_cause"] == "too much force"

        how = client.query("how_interventions", filters={"outcome": "resolved"})
        assert len(how) == 1
        assert how[0]["failure_id"] == "fail_1"


def test_ingest_practice_is_idempotent():
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
            ).model_dump(mode="json"),
        ]
        practice_id = _run_session_with_events(tmp, events)

        client = SeekDBMemoryClient()
        ingestor = SeekDBIngestor(tmp, seekdb_client=client)
        report1 = ingestor.ingest_practice(practice_id)
        report2 = ingestor.ingest_practice(practice_id)
        ingestor.close()

        assert report1.table_counts["failures"] == report2.table_counts["failures"]
        assert client.count("failures") == 1
        assert client.count("body_cognition") == 1


def test_ingest_body_cognition_and_sim2real_deltas():
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
                event_type="sim2real_delta_event",
                payload=Sim2RealDeltaPayload(
                    delta_id="delta_1",
                    body_id="body_rh56_left",
                    dofs=["thumb"],
                    sim_value={"thumb": 650.0},
                    real_value={"thumb": 992.0},
                    delta={"thumb": 342.0},
                    unit="force_net",
                ).model_dump(),
            ).model_dump(mode="json"),
        ]
        practice_id = _run_session_with_events(tmp, events)

        client = SeekDBMemoryClient()
        ingestor = SeekDBIngestor(tmp, seekdb_client=client)
        report = ingestor.ingest_practice(practice_id)
        ingestor.close()

        assert report.table_counts.get("body_cognition") == 1
        assert report.table_counts.get("sim2real_deltas") == 1

        cog = client.query("body_cognition", filters={"body_id": "body_rh56_left"})
        assert len(cog) == 1
        assert "desired_contact_region" in cog[0]["data"]["known_traits"]

        deltas = client.query("sim2real_deltas", filters={"body_id": "body_rh56_left"})
        assert len(deltas) == 1
        assert deltas[0]["delta"]["thumb"] == 342.0


def test_ingest_candidates_and_promotions():
    with tempfile.TemporaryDirectory() as tmp:
        events = [
            PracticeEventEnvelope(
                practice_id="prac_1",
                robot_id="test_bot",
                source="runtime",
                event_type="candidate_policy_event",
                payload=CandidatePolicyPayload(
                    candidate_id="cand_1",
                    policy_id="pol_1",
                    skill_id="skill_ok_contact",
                    policy_type="ok_pose",
                    policy_params={"thumb": 100.0, "index": 120.0},
                    metrics={"contact_rate": 0.9},
                ).model_dump(),
            ).model_dump(mode="json"),
            PracticeEventEnvelope(
                practice_id="prac_1",
                robot_id="test_bot",
                source="runtime",
                event_type="promotion_result_event",
                payload=PromotionResultPayload(
                    promotion_id="promo_1",
                    candidate_id="cand_1",
                    policy_id="pol_1",
                    passed=True,
                    gate_name="ok_promotion_gate",
                    metrics={"thumb_mean": 100.0},
                ).model_dump(),
            ).model_dump(mode="json"),
        ]
        practice_id = _run_session_with_events(tmp, events)

        client = SeekDBMemoryClient()
        ingestor = SeekDBIngestor(tmp, seekdb_client=client)
        report = ingestor.ingest_practice(practice_id)
        ingestor.close()

        assert report.table_counts.get("skill_candidates") == 1
        assert report.table_counts.get("promotion_results") == 1

        candidates = client.query("skill_candidates", filters={"id": "cand_1"})
        assert len(candidates) == 1
        assert candidates[0]["status"] == "promoted"

        promotions = client.query("promotion_results", filters={"id": "promo_1"})
        assert len(promotions) == 1
        assert promotions[0]["passed"] == 1


def test_ingest_practice_missing_raises():
    with tempfile.TemporaryDirectory() as tmp:
        ingestor = SeekDBIngestor(tmp)
        try:
            ingestor.ingest_practice("does_not_exist")
            raise AssertionError("expected ValueError")
        except ValueError as e:
            assert "not found" in str(e)
        finally:
            ingestor.close()
