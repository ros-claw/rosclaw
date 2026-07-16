"""Tests for PracticeQuery backend."""

from __future__ import annotations

import tempfile
from typing import Any

from rosclaw.memory.seekdb_client import InMemoryKnowledgeStore
from rosclaw.practice.config import PracticeConfig, SourceConfig
from rosclaw.practice.coordinator import PracticeCoordinator
from rosclaw.practice.query import PracticeQuery
from rosclaw.practice.recorder import PracticeRecorder
from rosclaw.practice.schemas import (
    CandidatePolicyPayload,
    FailureEventPayload,
    HowInterventionPayload,
    PhysicalFeedbackPayload,
    PracticeEventEnvelope,
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
    practice_id = coord._session.practice_id

    for ev in events:
        coord.emit_event(PracticeEventEnvelope(**ev))

    coord.stop()
    recorder.stop()
    return practice_id


def _ingest_and_query(tmp: str, events: list[dict[str, Any]]) -> tuple[str, PracticeQuery]:
    practice_id = _run_session_with_events(tmp, events)
    client = InMemoryKnowledgeStore()
    ingestor = SeekDBIngestor(tmp, seekdb_client=client)
    ingestor.ingest_practice(practice_id)
    ingestor.close()
    query = PracticeQuery(tmp, seekdb_client=client)
    return practice_id, query


def test_list_episodes_by_body():
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
                    force_net={"thumb": 100.0},
                    primary_event="desired_contact",
                ).model_dump(),
            ).model_dump(mode="json"),
        ]
        practice_id, query = _ingest_and_query(tmp, events)
        practice = query.list_episodes(body_id="body_rh56_left")
        assert len(practice) == 1
        assert practice[0]["session_id"] is not None
        query.close()


def test_list_failures_by_body_and_type():
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
                    force_net={"thumb": 100.0},
                    primary_event="desired_contact",
                ).model_dump(),
            ).model_dump(mode="json"),
            PracticeEventEnvelope(
                practice_id="prac_1",
                robot_id="test_bot",
                body_id="body_rh56_left",
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
        _, query = _ingest_and_query(tmp, events)

        failures = query.list_failures(body_id="body_rh56_left", failure_type="over_contact")
        assert len(failures) == 1
        assert failures[0]["failure_type"] == "over_contact"
        query.close()


def test_list_how_interventions_by_failure_type():
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
        _, query = _ingest_and_query(tmp, events)

        interventions = query.list_how_interventions(failure_type="over_contact")
        assert len(interventions) == 1
        assert interventions[0]["outcome"] == "resolved"
        query.close()


def test_list_body_cognition_by_body():
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
                    force_net={"thumb": 100.0},
                    primary_event="desired_contact",
                ).model_dump(),
            ).model_dump(mode="json"),
        ]
        _, query = _ingest_and_query(tmp, events)

        cogs = query.list_body_cognition(body_id="body_rh56_left")
        assert len(cogs) == 1
        assert "desired_contact_region" in cogs[0]["data"]["known_traits"]
        query.close()


def test_list_sim2real_deltas_by_body():
    with tempfile.TemporaryDirectory() as tmp:
        events = [
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
        _, query = _ingest_and_query(tmp, events)

        deltas = query.list_sim2real_deltas(body_id="body_rh56_left")
        assert len(deltas) == 1
        assert deltas[0]["delta"]["thumb"] == 342.0
        query.close()


def test_list_candidates_by_skill():
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
                    policy_params={"thumb": 100.0},
                    metrics={"contact_rate": 0.9},
                ).model_dump(),
            ).model_dump(mode="json"),
        ]
        _, query = _ingest_and_query(tmp, events)

        candidates = query.list_candidates(skill_id="skill_ok_contact")
        assert len(candidates) == 1
        assert candidates[0]["policy_id"] == "pol_1"
        query.close()


def test_explain_failure_includes_interventions():
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
        _, query = _ingest_and_query(tmp, events)

        explanation = query.explain_failure("fail_1")
        assert explanation is not None
        assert explanation["id"] == "fail_1"
        assert len(explanation["interventions"]) == 1
        query.close()


def test_explain_episode_snapshot():
    with tempfile.TemporaryDirectory() as tmp:
        events = [
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
        practice_id, query = _ingest_and_query(tmp, events)
        # episode_id comes from catalog; fetch it via list_episodes
        episodes = query.list_episodes()
        episode_id = episodes[0]["episode_id"]

        snapshot = query.explain_episode(episode_id)
        assert snapshot["episode_id"] == episode_id
        assert len(snapshot["sim2real_deltas"]) == 1
        query.close()
