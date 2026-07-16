"""Unit tests for proposal-only and shadow rollout loops."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rosclaw.integrations.lerobot.rollout.loop import (
    RolloutConfig,
    _run_loop,
)
from rosclaw.integrations.lerobot.rollout.metrics import RolloutMetrics, StepTimer
from rosclaw.integrations.lerobot.rollout.observation_source import FixtureObservationSource
from rosclaw.integrations.lerobot.rollout.recorder import RolloutRecorder
from rosclaw.integrations.lerobot.rollout.state import RolloutMode, RolloutStopReason


@pytest.fixture
def observation_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "observations.json"
    path.write_text(
        json.dumps(
            [
                {"observation.state": [0.1, 0.2, 0.3], "task": "test"},
                {"observation.state": [0.2, 0.3, 0.4], "task": "test"},
            ]
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def mock_body() -> Any:
    return type(
        "EffectiveBody",
        (),
        {
            "body_instance_id": "mock_body",
            "joints": {
                "j1": {"type": "revolute", "limits": {"lower": -1, "upper": 1}},
                "j2": {"type": "revolute", "limits": {"lower": -1, "upper": 1}},
                "j3": {"type": "revolute", "limits": {"lower": -1, "upper": 1}},
            },
        },
    )()


def _make_runtime_manager_mock() -> MagicMock:
    runtime = MagicMock()
    runtime.start.return_value = type("State", (), {"state": "ready", "error": None})()

    def _call(method: str, params: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        if method == "LOAD_POLICY":
            return {
                "status": "ok",
                "policy_metadata": {
                    "output_features": {
                        "action": {
                            "shape": [3],
                            "names": ["j1", "j2", "j3"],
                            "unit": "rad",
                            "representation": "joint_position",
                        }
                    }
                },
            }
        if method == "CREATE_SESSION":
            return {"status": "ok", "session_id": params.get("session_id", "unknown")}
        if method == "INFER":
            return {
                "status": "ok",
                "processed_action": {
                    "values": [0.11, 0.21, 0.31],
                    "shape": [3],
                    "dtype": "float32",
                },
                "raw_action": {"values": [0.1, 0.2, 0.3], "shape": [3], "dtype": "float32"},
            }
        if method == "CLOSE_SESSION":
            return {"status": "ok"}
        return {"status": "ok"}

    runtime.call.side_effect = _call
    return runtime


def test_fixture_observation_source(observation_fixture: Path) -> None:
    source = FixtureObservationSource(observation_fixture)
    assert not source.is_exhausted(0)
    obs = source.get_observation(0)
    assert obs["observation.state"] == [0.1, 0.2, 0.3]
    assert source.is_exhausted(2)


def test_rollout_metrics_summary() -> None:
    metrics = RolloutMetrics()
    metrics.record_step(10.0)
    metrics.record_step(20.0)
    metrics.record_inference(5.0)
    summary = metrics.to_dict()
    assert summary["step_latency_ms"]["count"] == 2
    assert summary["step_latency_ms"]["mean_ms"] == 15.0


def test_step_timer() -> None:
    with StepTimer() as timer:
        pass
    assert timer.elapsed_ms >= 0.0


def test_rollout_recorder(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    recorder = RolloutRecorder(trace, robot_id="r1")
    event_id = recorder.record_runtime_started({"policy_path": "p1"})
    trace_events = recorder.read_trace()
    assert len(trace_events) == 1
    assert trace_events[0]["event_type"] == "rollout.runtime.started"
    assert trace_events[0]["robot_id"] == "r1"
    assert trace_events[0]["event_id"] == event_id


@patch("rosclaw.integrations.lerobot.rollout.loop.PersistentRuntimeManager")
@patch("rosclaw.integrations.lerobot.rollout.loop._load_body")
def test_proposal_only_loop_completes(
    mock_load_body,
    mock_runtime_cls,
    observation_fixture: Path,
    mock_body: Any,
    tmp_path: Path,
) -> None:
    mock_load_body.return_value = mock_body
    runtime = _make_runtime_manager_mock()
    mock_runtime_cls.return_value = runtime

    config = RolloutConfig(
        mode=RolloutMode.PROPOSAL_ONLY,
        policy_path="mock/policy",
        observation_fixture=observation_fixture,
        steps=2,
        run_sandbox_preflight=False,
        trace_path=tmp_path / "trace.jsonl",
        python_executable="python3",
    )
    result = _run_loop(config, FixtureObservationSource(observation_fixture))

    assert result.stop_reason == RolloutStopReason.COMPLETED
    assert result.steps_completed == 2
    assert result.hardware_actions_executed == 0
    assert len(result.proposals) == 2
    assert Path(result.trace_path).exists()


@patch("rosclaw.integrations.lerobot.rollout.loop.PersistentRuntimeManager")
@patch("rosclaw.integrations.lerobot.rollout.loop._load_body")
def test_execute_flag_blocks(
    mock_load_body,
    mock_runtime_cls,
    observation_fixture: Path,
    mock_body: Any,
) -> None:
    mock_load_body.return_value = mock_body
    config = RolloutConfig(
        mode=RolloutMode.PROPOSAL_ONLY,
        policy_path="mock/policy",
        observation_fixture=observation_fixture,
        steps=2,
        execute=True,
        run_sandbox_preflight=False,
        python_executable="python3",
    )
    result = _run_loop(config, FixtureObservationSource(observation_fixture))
    assert result.stop_reason == RolloutStopReason.RUNTIME_FAILURE
    assert "--execute is not allowed" in result.errors[0]


@patch("rosclaw.integrations.lerobot.rollout.loop.PersistentRuntimeManager")
@patch("rosclaw.integrations.lerobot.rollout.loop._load_body")
def test_policy_load_failure(
    mock_load_body,
    mock_runtime_cls,
    observation_fixture: Path,
    mock_body: Any,
) -> None:
    mock_load_body.return_value = mock_body
    runtime = _make_runtime_manager_mock()
    runtime.call.side_effect = lambda method, params, **kwargs: {
        "LOAD_POLICY": {"status": "error", "error": {"message": "boom"}},
    }.get(method, {"status": "ok"})
    mock_runtime_cls.return_value = runtime

    config = RolloutConfig(
        mode=RolloutMode.PROPOSAL_ONLY,
        policy_path="mock/policy",
        observation_fixture=observation_fixture,
        steps=2,
        run_sandbox_preflight=False,
        python_executable="python3",
    )
    result = _run_loop(config, FixtureObservationSource(observation_fixture))
    assert result.stop_reason == RolloutStopReason.POLICY_LOAD_FAILED


@patch("rosclaw.integrations.lerobot.rollout.loop.PersistentRuntimeManager")
@patch("rosclaw.integrations.lerobot.rollout.loop._load_body")
def test_runtime_start_failure(
    mock_load_body,
    mock_runtime_cls,
    observation_fixture: Path,
    mock_body: Any,
) -> None:
    mock_load_body.return_value = mock_body
    runtime = _make_runtime_manager_mock()
    runtime.start.return_value = type("State", (), {"state": "error", "error": "no python"})()
    mock_runtime_cls.return_value = runtime

    config = RolloutConfig(
        mode=RolloutMode.PROPOSAL_ONLY,
        policy_path="mock/policy",
        observation_fixture=observation_fixture,
        steps=2,
        run_sandbox_preflight=False,
        python_executable="python3",
    )
    result = _run_loop(config, FixtureObservationSource(observation_fixture))
    assert result.stop_reason == RolloutStopReason.RUNTIME_FAILURE
    assert "no python" in result.errors[0]
