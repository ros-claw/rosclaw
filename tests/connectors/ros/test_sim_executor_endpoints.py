"""Frozen namespace/frame configuration; fake dispatch is not SIM acceptance."""

from types import SimpleNamespace

import pytest

from rosclaw.connectors.ros.mission.executor import RosCoverageSimulationExecutor
from rosclaw.connectors.ros.mission.sim_endpoints import (
    DEFAULT_ENDPOINTS,
    absolute_endpoint,
    freeze_sim_endpoints,
)


def configured(tmp_path, *, endpoints=None, spawn=(0.0, 0.0, 0.0)):
    return RosCoverageSimulationExecutor(
        owner="daemon_test",
        client=None,
        control=None,
        witness=SimpleNamespace(fresh=lambda: {"x": 0, "y": 0}),
        output=tmp_path,
        body_id="body",
        body_snapshot_hash="hash",
        grid={"frame_id": "world"},
        recovery_centers=[(-1, -1), (-1, 1), (1, -1), (1, 1)],
        boundary_pass=True,
        endpoints=endpoints,
        configured_spawn=spawn,
    )


def test_boundary_dispatch_uses_frozen_namespaced_action_and_observed_frame(tmp_path):
    endpoints = {k: "/isolated/robot" + name for k, name in DEFAULT_ENDPOINTS.items()}
    executor = configured(tmp_path, endpoints=endpoints)
    calls = []
    executor._run_goal = lambda *a, **kw: calls.append((a, kw)) or {"status": 4}
    endpoints["navigate_through_poses"] = "/other_unobserved"
    result = executor._boundary({"status": 4}, "root", 100)
    assert result["status"] == "SUCCEEDED"
    args = calls[0][0]
    assert args[0] == "/isolated/robot/navigate_through_poses"
    assert all(p["header"]["frame_id"] == "world" for p in args[2]["poses"])
    with pytest.raises(TypeError):
        executor.endpoints["cleaning"] = "/agent_override"


def test_original_endpoint_defaults_remain_unchanged(tmp_path):
    executor = configured(tmp_path)
    assert dict(executor.endpoints) == dict(DEFAULT_ENDPOINTS)


@pytest.mark.parametrize(
    "bad",
    [
        "relative",
        "/",
        "/bad//name",
        "/bad/",
        "/9robot/action",
        "/robot/$action",
        "/" + "x" * 257,
        None,
        1,
    ],
)
def test_bad_endpoint_refused(bad):
    with pytest.raises(ValueError):
        absolute_endpoint(bad)


@pytest.mark.parametrize("fault", ["missing", "extra", "duplicate", "relative"])
def test_incomplete_or_ambiguous_set_refused(fault):
    endpoints = dict(DEFAULT_ENDPOINTS)
    if fault == "missing":
        endpoints.pop("lease")
    elif fault == "extra":
        endpoints["cmd_vel"] = "/velocity"
    elif fault == "duplicate":
        endpoints["lease"] = endpoints["cleaning"]
    else:
        endpoints["cleaning"] = "brush"
    with pytest.raises(ValueError):
        freeze_sim_endpoints(endpoints)


@pytest.mark.parametrize("spawn", [(0.7, -0.4, 0.8), (-0.6, 0.3, -2.1)])
def test_localization_uses_frozen_shifted_spawn_and_namespace(tmp_path, monkeypatch, spawn):
    import math
    from datetime import UTC, datetime

    from rosclaw.connectors.ros.mission import executor as module
    from rosclaw.kernel import ActionState

    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    endpoints = {k: "/renamed" + name for k, name in DEFAULT_ENDPOINTS.items()}
    executor = configured(tmp_path, endpoints=endpoints, spawn=spawn)
    calls = []
    executor.control = SimpleNamespace(
        call_service=lambda *a, **kw: calls.append((a, kw)) or SimpleNamespace(ok=True)
    )

    def observed():
        return {
            "x": spawn[0],
            "y": spawn[1],
            "yaw": spawn[2],
            "cleaning_enabled": False,
            "lease_remaining_sec": 0,
            "localization": {
                "x": spawn[0],
                "y": spawn[1],
                "frame_id": "world",
                "captured_at": datetime.now(UTC).isoformat(),
                "covariance": [0.0] * 36,
            },
        }

    executor.witness = SimpleNamespace(fresh=observed)
    result = executor._localize(
        SimpleNamespace(arguments={}, verification_policy=SimpleNamespace(timeout_sec=2))
    )
    assert result.final_state is ActionState.COMPLETED
    assert result.verification_result["localization_error_m"] == 0
    assert result.verification_result["configured_spawn"] == list(spawn)
    name, request = calls[0][0]
    assert name == "/renamed/set_initial_pose"
    assert request["pose"]["header"]["frame_id"] == "world"
    pose = request["pose"]["pose"]["pose"]
    assert pose["position"] == {"x": spawn[0], "y": spawn[1], "z": 0}
    assert pose["orientation"]["z"] == pytest.approx(math.sin(spawn[2] / 2))
    assert pose["orientation"]["w"] == pytest.approx(math.cos(spawn[2] / 2))
    # Action arguments cannot rewrite the configured startup position.
    calls.clear()
    rejected = executor._localize(SimpleNamespace(arguments={"x": 0, "y": 0}))
    assert rejected.final_state is ActionState.FAILED and calls == []


@pytest.mark.parametrize(
    "bad",
    [
        [],
        [0, 0],
        [0, 0, 0, 0],
        [True, 0, 0],
        [float("nan"), 0, 0],
        [float("inf"), 0, 0],
        [10**400, 0, 0],
    ],
)
def test_spawn_numeric_bounds_refuse_unknown_configuration(bad):
    from rosclaw.connectors.ros.mission.sim_endpoints import freeze_sim_spawn

    with pytest.raises(ValueError):
        freeze_sim_spawn(bad)


def test_dynamic_wait_uses_frozen_namespaced_hold_service(tmp_path):
    endpoints = {k: "/different/robot" + value for k, value in DEFAULT_ENDPOINTS.items()}
    executor = configured(tmp_path, endpoints=endpoints)
    calls = []
    executor.lease_control = SimpleNamespace(
        call_service=lambda name, values, **kwargs: (
            calls.append((name, values))
            or SimpleNamespace(ok=True, data={"values": {"success": True}})
        )
    )
    executor._set_obstacle_wait(True)
    executor._set_obstacle_wait(False)
    assert calls == [
        ("/different/robot/rosclaw_sim/hold", {"data": True}),
        ("/different/robot/rosclaw_sim/hold", {"data": False}),
    ]


def test_legacy_six_endpoints_remain_static_compatible_but_dynamic_requires_hold(tmp_path):
    endpoints = {k: "/different" + value for k, value in DEFAULT_ENDPOINTS.items() if k != "hold"}
    assert freeze_sim_endpoints(endpoints)["navigate_to_pose"] == "/different/navigate_to_pose"
    with pytest.raises(ValueError, match="explicit hold"):
        RosCoverageSimulationExecutor(
            owner="daemon_test",
            client=None,
            control=None,
            witness=None,
            output=tmp_path,
            body_id="body",
            body_snapshot_hash="hash",
            grid={},
            endpoints=endpoints,
            occupancy_binding={"run_id": "fresh", "geometry_hash": "source"},
        )
