"""Keep exact blocking observations visible without changing their verdict."""

from datetime import UTC, datetime
from types import SimpleNamespace

from rosclaw.connectors.ros.context.compiler import compile_agent_summary
from rosclaw.connectors.ros.intelligence.system_model import RosSystemModel, Signal


def stale_model():
    now = datetime.now(UTC)
    return RosSystemModel(
        robot_id="diagnostic",
        snapshot_id="actual-fields",
        captured_at=now,
        graph={"topics": [{"name": "/scan", "msg_type": "sensor_msgs/msg/LaserScan"}]},
        signals=[
            Signal(
                topic="/scan",
                source="fixture",
                captured_at=now,
                publisher_count=1,
                last_message_age_ms=4500,
                max_age_ms=1000,
            )
        ],
    )


def test_summary_exposes_exact_stale_signal_and_original_limits():
    model = stale_model()
    summary = compile_agent_summary(model, now=model.captured_at)
    assert "health=BLOCKED" in summary
    assert '"topic": "/scan"' in summary
    assert '"last_message_age_ms": 4500.0' in summary
    assert '"max_age_ms": 1000.0' in summary
    assert "request_action → rosclawd → verified receipt" in summary


def test_native_log_projection_preserves_structured_issue_evidence(monkeypatch):
    from rosclaw.connectors.ros.context import native

    model = stale_model()
    monkeypatch.setattr(native, "inspect_system", lambda **kwargs: model)
    service = SimpleNamespace(
        _config=SimpleNamespace(raw={"agent": {"ros_expert": {"enabled": True}}})
    )
    mission = SimpleNamespace(body_binding=SimpleNamespace(body_id="diagnostic"))
    result = native.native_ros_observations(service, mission, None)
    assert result["authorization"] is False
    assert result["diagnosis"]["status"] == "BLOCKED"
    issue = next(i for i in result["diagnosis"]["issues"] if i["issue_code"] == "ROS_TOPIC_002")
    assert issue["evidence"][0]["observation"]["topic"] == "/scan"
    assert issue["evidence"][0]["observation"]["last_message_age_ms"] == 4500
    assert "ROS_TOPIC_002" in result["admission_facts"]["blocking_issues"]


async def test_diagnostic_detail_does_not_change_admission_hash(tmp_path):
    from rosclaw.agentd.pi_bridge.context import build_embodied_context
    from rosclaw.agentd.pi_bridge.context_lease import context_hash_of
    from tests.agentd.test_pi_tool_bridge import _setup

    service, mission = await _setup(tmp_path)
    try:
        envelope = build_embodied_context(service, mission.mission_id)
        envelope.self_state["ros_observations"] = {
            "status": "OBSERVED",
            "admission_facts": {"blocking_issues": ["ROS_TOPIC_002"]},
            "summary": "original",
            "diagnosis": {"issues": []},
        }
        before = context_hash_of(envelope)
        envelope.self_state["ros_observations"]["summary"] = "bounded topic/age details"
        envelope.self_state["ros_observations"]["diagnosis"] = {
            "issues": [{"topic": "/scan", "age": 4500}]
        }
        assert context_hash_of(envelope) == before
        envelope.self_state["ros_observations"]["admission_facts"]["blocking_issues"] = []
        assert context_hash_of(envelope) != before
    finally:
        await service.close()
