"""Tests for rosclaw.practice schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rosclaw.practice.schemas import (
    AgentPlanPayload,
    ExecutedActionPayload,
    FootContactPayload,
    HumanFeedbackPayload,
    IMUPayload,
    JointStatePayload,
    PracticeEventEnvelope,
    ToolCallPayload,
)


def test_envelope_defaults():
    env = PracticeEventEnvelope(
        practice_id="prac_001",
        robot_id="r1",
        source="agent",
        event_type="agent.task_received",
    )
    assert env.schema_version == "practice.event.v1"
    assert env.timestamp_ns is not None
    assert env.timestamp_utc is not None
    assert env.event_id


def test_envelope_payload_roundtrip():
    payload = AgentPlanPayload(
        task="pick cup",
        plan_id="plan_1",
        planner="mock",
        plan_steps=[{"step": 1, "name": "perceive"}],
    )
    env = PracticeEventEnvelope(
        practice_id="prac_001",
        robot_id="r1",
        source="agent",
        event_type="agent.plan_created",
        payload=payload.model_dump(),
    )
    data = env.model_dump(mode="json")
    assert data["payload"]["task"] == "pick cup"
    restored = PracticeEventEnvelope.model_validate(data)
    assert restored.payload["task"] == "pick cup"


def test_envelope_source_must_be_valid():
    with pytest.raises(ValidationError):
        PracticeEventEnvelope(
            practice_id="prac_001",
            robot_id="r1",
            source="not_a_source",  # type: ignore[arg-type]
            event_type="x",
        )


def test_joint_state_payload_validation():
    p = JointStatePayload(
        joint_names=["j1", "j2"],
        position=[0.1, 0.2],
        velocity=[0.0, 0.0],
    )
    assert p.joint_names == ["j1", "j2"]


def test_imu_payload_requires_angular_and_linear():
    with pytest.raises(ValidationError):
        IMUPayload(angular_velocity_xyz=[0.0, 0.0, 0.0])  # missing linear_acceleration


def test_foot_contact_payload():
    p = FootContactPayload(left_contact=True, right_contact=False)
    assert p.left_contact is True


def test_tool_call_payload_status():
    p = ToolCallPayload(
        tool_call_id="tc1",
        tool_name="grasp",
        arguments={"object": "cup"},
        status="started",
    )
    assert p.status == "started"


def test_executed_action_payload_status():
    p = ExecutedActionPayload(
        action_id="a1",
        action_type="move",
        command={"target": [0.0, 0.0]},
        controller="mock",
        start_time_ns=123,
        status="completed",
        reward=0.8,
    )
    assert p.reward == 0.8


def test_human_feedback_payload():
    p = HumanFeedbackPayload(feedback_id="f1", feedback_type="thumbs_up", rating=5)
    assert p.rating == 5
