"""P4-E mock RH56 body mapping + shadow rollout smoke test.

This test verifies exact body mapping and end-to-end shadow rollout without any
real policy loading or robot motion.  A mock persistent runtime returns a 6-DoF
joint_position action with names matching the mock RH56 legs; the shadow loop:

- reads observations from the mock body (SenseInterface / MockCollector),
- maps the policy action to the body action space with ``exact`` compatibility,
- runs sandbox preflight,
- records a Practice-compatible JSONL trace,
- never executes any hardware action.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rosclaw.integrations.lerobot.rollout import RolloutConfig, run_shadow_loop
from rosclaw.integrations.lerobot.rollout.state import RolloutMode, RolloutStopReason


RH56_JOINTS = [
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]


def _make_rh56_body() -> Any:
    return type(
        "EffectiveBody",
        (),
        {
            "body_instance_id": "rh56_mock",
            "joints": {
                name: {
                    "type": "revolute",
                    "limits": {"lower": -6.28, "upper": 6.28},
                }
                for name in RH56_JOINTS
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
                            "shape": [6],
                            "names": RH56_JOINTS,
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
                    "values": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                    "shape": [6],
                    "dtype": "float32",
                },
                "raw_action": {
                    "values": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
                    "shape": [6],
                    "dtype": "float32",
                },
                "runtime_id": "mock_runtime",
                "timing": {"infer_ms": 1.0},
            }
        if method == "CLOSE_SESSION":
            return {"status": "ok"}
        return {"status": "ok"}

    runtime.call.side_effect = _call
    return runtime


@patch("rosclaw.integrations.lerobot.rollout.loop.PersistentRuntimeManager")
@patch("rosclaw.integrations.lerobot.rollout.loop._load_body")
def test_rh56_shadow_rollout_exact_mapping_and_trace(
    mock_load_body,
    mock_runtime_cls,
    tmp_path: Path,
) -> None:
    """Shadow rollout on a mock RH56 body achieves exact mapping and records trace."""
    mock_load_body.return_value = _make_rh56_body()
    runtime = _make_runtime_manager_mock()
    mock_runtime_cls.return_value = runtime

    trace_path = tmp_path / "rh56_shadow_trace.jsonl"
    config = RolloutConfig(
        mode=RolloutMode.SHADOW,
        policy_path="mock/rh56_joint_position_policy",
        robot_id="rh56_mock",
        steps=3,
        control_hz=10.0,
        run_sandbox_preflight=True,
        trace_path=trace_path,
        python_executable="python3",
    )

    result = run_shadow_loop(config)

    assert result.stop_reason == RolloutStopReason.COMPLETED
    assert result.steps_completed == 3
    assert result.hardware_actions_executed == 0
    assert len(result.proposals) == 3
    assert len(result.mapped_actions) == 3
    assert len(result.sandbox_decisions) == 3
    assert Path(result.trace_path).exists()

    # Read trace and check for the expected Practice event types.
    trace = json.loads(trace_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert trace["event_type"] == "episode.summary"
    assert trace["payload"]["hardware_actions_executed"] == 0
    assert trace["payload"]["outcome"] == "completed"

    # Every step should have produced an action mapping event and a sandbox decision.
    lines = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    event_types = [line["event_type"] for line in lines]
    assert event_types.count("rollout.action.mapped") == 3
    assert event_types.count("rollout.sandbox.decision") == 3

    # The first proposal must carry the RH56 joint names and exact-mapping safety.
    first_proposal = result.proposals[0]
    assert first_proposal.get("schema_version") == "rosclaw.action_proposal.v2"
    assert first_proposal.get("representation") == "joint_position"
    assert first_proposal.get("action", {}).get("names") == RH56_JOINTS
    assert first_proposal.get("safety", {}).get("executable") is False

    # Sandbox preflight should have allowed the small nominal actions.
    first_sandbox = result.sandbox_decisions[0]
    assert first_sandbox.get("is_safe") is True
    assert first_sandbox.get("decision") == "ALLOW"


@patch("rosclaw.integrations.lerobot.rollout.loop.PersistentRuntimeManager")
@patch("rosclaw.integrations.lerobot.rollout.loop._load_body")
def test_rh56_shadow_rollout_blocks_incompatible_mapping(
    mock_load_body,
    mock_runtime_cls,
    tmp_path: Path,
) -> None:
    """A policy whose action names do not match the body joints must be blocked."""
    mock_load_body.return_value = _make_rh56_body()
    runtime = MagicMock()
    runtime.start.return_value = type("State", (), {"state": "ready", "error": None})()

    def _call(method: str, params: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        if method == "LOAD_POLICY":
            return {
                "status": "ok",
                "policy_metadata": {
                    "output_features": {
                        "action": {
                            "shape": [6],
                            "names": ["unknown_joint_" + str(i) for i in range(6)],
                            "unit": "rad",
                            "representation": "joint_position",
                        }
                    }
                },
            }
        if method == "CREATE_SESSION":
            return {"status": "ok"}
        if method == "INFER":
            return {
                "status": "ok",
                "processed_action": {
                    "values": [0.1] * 6,
                    "shape": [6],
                    "dtype": "float32",
                },
            }
        if method == "CLOSE_SESSION":
            return {"status": "ok"}
        return {"status": "ok"}

    runtime.call.side_effect = _call
    mock_runtime_cls.return_value = runtime

    config = RolloutConfig(
        mode=RolloutMode.SHADOW,
        policy_path="mock/bad_policy",
        robot_id="rh56_mock",
        steps=3,
        run_sandbox_preflight=True,
        trace_path=tmp_path / "bad_trace.jsonl",
        python_executable="python3",
    )

    result = run_shadow_loop(config)

    assert result.stop_reason == RolloutStopReason.INCOMPATIBLE_MAPPING
    assert result.hardware_actions_executed == 0
    assert result.steps_completed == 0
    assert any("No policy action names match body joint names" in e for e in result.errors)
