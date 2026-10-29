"""No telemetry, explicit synthetic examples, and historical replay stay distinct."""

import json

import pytest

from rosclaw.sense.collectors.file_replay_collector import FileReplayCollector
from rosclaw.sense.config import SenseConfig
from rosclaw.sense.estimators.readiness import ReadinessEvaluator
from rosclaw.sense.explain import SenseExplainer
from rosclaw.sense.interface import SenseInterface
from rosclaw.sense.schemas import BodyReadiness, BodyRiskSummary, BodyState


def test_unconfigured_interface_does_not_invent_healthy_telemetry():
    from rosclaw.core.runtime import RuntimeConfig

    assert RuntimeConfig().sense_collector == "unconfigured"
    assert SenseConfig().collector == "unconfigured"
    assert SenseConfig.from_dict({}).collector == "unconfigured"
    interface = SenseInterface(robot_id="unconfigured_robot")
    interface.initialize()
    try:
        state = interface.get_body_state()
        assert state.energy.battery_percent is None
        assert state.joints == {}
        assert state.communication.heartbeat_ok is None
        sense = interface.tick()
        assert sense.overall_status != "ready"
        assert sense.evidence["collector_source"] == "unconfigured"
        assert sense.evidence["observation_kind"] == "unavailable"
    finally:
        interface.stop()


@pytest.mark.parametrize("command", ["now", "state", "readiness"])
@pytest.mark.parametrize("explicit_mock", [False, True])
def test_real_cli_requires_explicit_mock(command, explicit_mock, monkeypatch, tmp_path, capsys):
    from rosclaw.cli import main

    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path / "home"))
    args = ["rosclaw", "sense", command, "--robot-id", "test_robot", "--json"]
    if command == "readiness":
        args += ["--task", "observe_scene"]
    if explicit_mock:
        args += ["--mock", "normal"]
    monkeypatch.setattr("sys.argv", args)
    assert main() == 0
    data = json.loads(capsys.readouterr().out)
    if command == "state":
        assert data["energy"]["battery_percent"] == (80.0 if explicit_mock else None)
        assert data["source"] == ("mock:normal" if explicit_mock else "unconfigured")
    else:
        assert (data["overall_status"] == "ready") is explicit_mock
        if command == "readiness":
            assert data["source"] == ("mock:normal" if explicit_mock else "unconfigured")
        if command == "now":
            assert data["evidence"]["observation_kind"] == (
                "synthetic" if explicit_mock else "unavailable"
            )


@pytest.mark.parametrize("risk", ["unknown", "low"])
def test_empty_unknown_readiness_does_not_say_all_capabilities_ready(risk):
    state = BodyState(robot_id="no_observations", timestamp=1.0)
    sense = SenseExplainer().summarize(
        state, BodyRiskSummary(overall_risk=risk), BodyReadiness("no_observations", 1.0)
    )
    assert sense.overall_status == "unknown"
    assert "ready for all" not in sense.natural_language_summary
    assert "unknown" in sense.natural_language_summary


def test_empty_replay_is_unavailable():
    state = FileReplayCollector().collect()
    sense = SenseExplainer().summarize(
        state, BodyRiskSummary(), BodyReadiness(state.robot_id, state.timestamp)
    )
    assert state.energy.battery_percent is None
    assert sense.evidence["observation_kind"] == "unavailable"


def test_replay_preserves_original_time_and_source_attribution(tmp_path):
    original = BodyState(robot_id="recorded_robot", timestamp=123.0, source="recorded:odom")
    original.energy.battery_percent = 42.0
    path = tmp_path / "history.jsonl"
    path.write_text(json.dumps(original.to_dict()) + "\n")
    collector = FileReplayCollector(str(path))
    first = collector.collect()
    second = collector.collect()
    for state in [first, second]:
        assert state.timestamp == 123.0
        assert state.source == "file_replay:recorded:odom"
        assert state.raw["replay"]["original_source"] == "recorded:odom"
        assert state.raw["replay"]["replayed_at"] > state.timestamp
        assert state.energy.battery_percent == 42.0
    assert json.loads(path.read_text()) == original.to_dict()
    sense = SenseExplainer().summarize(
        first, BodyRiskSummary(), BodyReadiness(first.robot_id, first.timestamp)
    )
    assert sense.evidence["observation_kind"] == "historical_replay"
    readiness = ReadinessEvaluator().evaluate(first, BodyRiskSummary(), task="observe_scene")
    assert readiness.timestamp == 123.0
    assert readiness.source == "file_replay:recorded:odom"
    restored = BodyReadiness.from_dict(readiness.to_dict())
    assert restored.timestamp == 123.0
    assert restored.source == readiness.source


def test_explicit_unknown_mock_keeps_its_synthetic_identity():
    interface = SenseInterface(collector="mock", scenario="unknown_partial")
    interface.initialize()
    try:
        sense = interface.tick()
        assert sense.overall_status != "ready"
        assert sense.evidence["collector_source"] == "mock:unknown_partial"
        assert sense.evidence["observation_kind"] == "synthetic"
    finally:
        interface.stop()


@pytest.mark.parametrize("collector", ["unconfigured", "file_replay"])
async def test_daemon_projection_refuses_unavailable_collectors_even_if_transport_is_fresh(
    collector, tmp_path
):
    from rosclaw.core.event_bus import EventBus
    from rosclaw.sense.observation import SenseObservationError, project_robot_state
    from rosclaw.sense.runtime import SenseRuntime

    runtime = SenseRuntime(SenseConfig(collector=collector, update_hz=0), EventBus())
    runtime.initialize()
    try:
        runtime.tick()
        # Transport freshness cannot turn missing telemetry into live evidence.
        runtime.is_stale = False
        with pytest.raises(SenseObservationError) as error:
            project_robot_state(runtime, "no_sensor")
        assert error.value.code == "STATE_PROVENANCE_UNAVAILABLE"
    finally:
        runtime.stop()


async def test_daemon_projection_keeps_real_file_replay_historical(tmp_path):
    from rosclaw.core.event_bus import EventBus
    from rosclaw.sense.observation import project_robot_state
    from rosclaw.sense.runtime import SenseRuntime

    path = tmp_path / "recording.jsonl"
    path.write_text(json.dumps(BodyState("recorded", 123.0, source="recorded:odom").to_dict()))
    runtime = SenseRuntime(
        SenseConfig(collector="file_replay", replay_path=str(path), update_hz=0), EventBus()
    )
    runtime.initialize()
    try:
        runtime.tick()
        runtime.is_stale = False
        result = project_robot_state(runtime, "recorded")
        assert result["mode"] == "replay"
        assert result["execution_mode"] == "REPLAY"
        assert result["trust_level"] == "RECORDED"
        assert result["usable_for_real_execution"] is False
        assert result["body_state"]["timestamp"] == result["readiness"]["timestamp"] == 123.0
    finally:
        runtime.stop()
