"""Regression coverage for faults discovered during installed/native acceptance."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from rosclaw.connectors.ros.context.native import native_ros_observations
from rosclaw.connectors.ros.intelligence import RosSystemModel
from rosclaw.connectors.ros.practice import RosPracticeAdapter
from rosclaw.connectors.ros.verification.reachable import cleanable_cells
from rosclaw.core.event_bus import Event, EventBus
from rosclaw.memory.interface import MemoryInterface


@pytest.mark.parametrize("mode, body_hash", [("REAL", "configured"), ("SIMULATION", "different")])
def test_simulation_executor_rejects_wrong_mode_or_body_before_dispatch(tmp_path, mode, body_hash):
    from rosclaw.connectors.ros.mission.executor import RosCoverageSimulationExecutor
    from rosclaw.kernel import ActionState, ExecutionMode

    executor = RosCoverageSimulationExecutor(
        owner="daemon_test",
        client=None,
        control=None,
        witness=None,
        output=tmp_path,
        body_id="robot",
        body_snapshot_hash="configured",
        grid={},
    )
    result = executor(
        SimpleNamespace(
            execution_mode=ExecutionMode(mode),
            body_id="robot",
            body_snapshot_hash=body_hash,
            capability_id="coverage.execute",
        )
    )
    assert result.final_state is ActionState.BLOCKED
    assert not result.dispatch_result["accepted"]


def test_memory_executor_rejects_changed_body_without_reading_artifact(tmp_path):
    from rosclaw.connectors.ros.mission.remember import VerifiedMissionMemoryExecutor
    from rosclaw.kernel import ActionState, ExecutionMode

    receipt = SimpleNamespace(to_dict=lambda: {"body_id": "robot", "body_snapshot_hash": "old"})
    runtime = SimpleNamespace(action_gateway=SimpleNamespace(get_receipt=lambda _: receipt))
    executor = VerifiedMissionMemoryExecutor(runtime, evidence_directory=tmp_path)
    result = executor(
        SimpleNamespace(
            execution_mode=ExecutionMode.SIMULATION,
            arguments={"coverage_action_id": "a"},
            body_id="robot",
            body_snapshot_hash="new",
        )
    )
    assert result.final_state is ActionState.BLOCKED
    assert result.errors[0]["message"] == "receipt Body mismatch"


def test_simulation_executor_serializes_motion_for_one_body(tmp_path):
    from rosclaw.connectors.ros.mission.executor import RosCoverageSimulationExecutor
    from rosclaw.kernel import ActionState, ExecutionMode

    executor = RosCoverageSimulationExecutor(
        owner="daemon_test",
        client=None,
        control=None,
        witness=None,
        output=tmp_path,
        body_id="robot",
        body_snapshot_hash="body",
        grid={},
    )
    executor.execution_lock.acquire()
    try:
        result = executor(
            SimpleNamespace(
                execution_mode=ExecutionMode.SIMULATION,
                body_id="robot",
                body_snapshot_hash="body",
                capability_id="coverage.execute",
            )
        )
        assert result.final_state is ActionState.BLOCKED
        assert result.errors[0]["code"] == "ROS_SIMULATION_BUSY"
        assert not result.dispatch_result["accepted"]
    finally:
        executor.execution_lock.release()


@pytest.mark.parametrize("complete, collisions", [(False, 0), (True, 1)])
def test_fresh_independent_witness_rejects_incomplete_or_colliding_observation(
    complete, collisions
):
    import threading
    import time

    from rosclaw.connectors.ros.mission.executor import SimulationWitness

    witness = SimulationWitness.__new__(SimulationWitness)
    witness.lock = threading.Lock()
    witness.action_fault = None
    witness.latest = (
        time.monotonic(),
        {"observation_complete": complete, "collision_count": collisions},
    )
    with pytest.raises(RuntimeError):
        witness.fresh()


def test_diagnostic_event_is_observed_in_actual_memory():
    bus = EventBus()
    memory = MemoryInterface("regression", event_bus=bus)
    adapter = RosPracticeAdapter(bus)
    memory.initialize()
    adapter.initialize()
    try:
        bus.publish(
            Event(
                topic="rosclaw.ros.diagnosis.created",
                payload={"snapshot_id": "diagnostic-only", "status": "HEALTHY"},
            )
        )
        stored = memory.get_experience("diagnostic-only")
        assert stored is not None
        assert stored["outcome"] == "observed"
        assert memory.get_statistics()["success_count"] == 0
    finally:
        adapter.stop()
        memory.stop()


def test_native_context_disabled_never_connects(monkeypatch):
    def forbidden(**_):
        raise AssertionError("disabled feature opened ROS connection")

    monkeypatch.setattr("rosclaw.connectors.ros.context.native.inspect_system", forbidden)
    assert native_ros_observations(SimpleNamespace(), None, None) is None


@pytest.mark.parametrize(
    "age_seconds, expected", [(0, "OBSERVED"), (10, "UNKNOWN"), (-10, "UNKNOWN")]
)
def test_native_context_freshness_is_enforced(monkeypatch, age_seconds, expected):
    model = RosSystemModel(
        robot_id="test",
        snapshot_id="test",
        captured_at=datetime.now(UTC) - timedelta(seconds=age_seconds),
    ).seal()
    monkeypatch.setattr("rosclaw.connectors.ros.context.native.inspect_system", lambda **_: model)
    service = SimpleNamespace(
        _config=SimpleNamespace(raw={"agent": {"ros_expert": {"enabled": True}}})
    )
    mission = SimpleNamespace(body_binding=SimpleNamespace(body_id="test"))
    observations = native_ros_observations(service, mission, None)
    assert observations["status"] == expected
    assert observations["authorization"] is False


def test_map_denominator_excludes_disconnected_permanent_component():
    occupancy = [100 if x == 5 else 0 for y in range(12) for x in range(12)]
    cells = cleanable_cells(
        width=12,
        height=12,
        resolution=1,
        occupancy=occupancy,
        start_cell=2 + 6 * 12,
        robot_radius=0.5,
        cleaning_radius=0.5,
    )
    assert cells
    assert all(i % 12 < 5 for i in cells)
    assert occupancy[8 + 6 * 12] == 0  # Free but physically disconnected stays excluded.


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_invalid_physical_geometry_rejected(bad):
    with pytest.raises(ValueError):
        cleanable_cells(
            width=5,
            height=5,
            resolution=1,
            occupancy=[0] * 25,
            start_cell=12,
            robot_radius=0.5,
            cleaning_radius=bad,
        )


def test_current_action_envelope_and_malformed_status_keep_listener_alive():
    import queue
    import threading

    from rosclaw.connectors.ros.action_client import Ros2ActionClient
    from rosclaw.connectors.ros.transport.base import RosTransportResult

    class Transport:
        def __init__(self):
            self.messages = queue.Queue()
            self.sent = []

        def send(self, message):
            self.sent.append(message)
            return RosTransportResult(ok=True)

        def receive(self, **_):
            try:
                return RosTransportResult(ok=True, data=self.messages.get(timeout=0.05))
            except queue.Empty:
                return RosTransportResult(ok=False)

        def close(self):
            pass

    transport = Transport()
    client = Ros2ActionClient(transport)
    observed, done = [], threading.Event()

    def completed(status, payload):
        observed.append((status, payload))
        if status == 4:
            done.set()

    try:
        for goal_id in ["bad", "good"]:
            client.send_goal(
                action="/navigate_to_pose",
                action_type="nav2_msgs/action/NavigateToPose",
                args={},
                goal_id=goal_id,
                on_feedback=lambda _: None,
                on_result=completed,
            )
        transport.messages.put(
            {"op": "action_result", "id": "bad", "status": "invalid", "values": {}}
        )
        transport.messages.put(
            {
                "op": "action_result",
                "id": "good",
                "status": 4,
                "result": True,
                "values": {"error_code": 0},
            }
        )
        assert done.wait(2)
        assert observed == [(6, {}), (4, {"error_code": 0})]
        assert all(message["op"] == "send_action_goal" for message in transport.sent)
    finally:
        client.close()


def test_simulation_request_is_opt_in_and_fixture_remains_forbidden(monkeypatch, tmp_path):
    from rosclaw.mcp.adapters.runtime_client import RuntimeClient
    from rosclaw.mcp.schemas.common import MCPError

    client = RuntimeClient(
        project_root=tmp_path, robot_id="test", runtime_profile={}, daemon_client=object()
    )
    kwargs = {
        "capability_id": "coverage.execute",
        "arguments": {},
        "execution_mode": "SIMULATION",
        "body_snapshot_hash": "body-test",
    }
    monkeypatch.delenv("ROSCLAW_ROS_EXPERT", raising=False)
    with pytest.raises(MCPError) as blocked:
        client._build_action(**kwargs)
    assert blocked.value.code == "INVALID_EXECUTION_MODE"
    monkeypatch.setenv("ROSCLAW_ROS_EXPERT", "1")
    action, mode = client._build_action(**kwargs)
    assert mode.value == "SIMULATION"
    assert action.authorization.approved is False
    client.fixture_mode = True
    with pytest.raises(MCPError) as blocked:
        client._build_action(**kwargs)
    assert blocked.value.code == "FIXTURE_ACTION_FORBIDDEN"


@pytest.mark.parametrize(
    "sample,arguments",
    [
        ({"x": 0.1, "y": 0, "yaw": 0, "cleaning_enabled": False}, {}),
        ({"x": 0, "y": 0, "yaw": 0.3, "cleaning_enabled": False}, {}),
        ({"x": 0, "y": 0, "yaw": 0, "cleaning_enabled": True}, {}),
        ({"x": 0, "y": 0, "yaw": 0, "cleaning_enabled": False, "lease_remaining_sec": 1}, {}),
        ({"x": 0, "y": 0, "yaw": 0, "cleaning_enabled": False}, {"x": 2}),
    ],
)
def test_fixture_initial_localization_refuses_unconfigured_or_active_pose(
    tmp_path, sample, arguments
):
    from rosclaw.connectors.ros.mission.executor import RosCoverageSimulationExecutor
    from rosclaw.kernel import ActionState

    executor = RosCoverageSimulationExecutor(
        owner="daemon_test",
        client=None,
        control=None,
        witness=SimpleNamespace(fresh=lambda: sample),
        output=tmp_path,
        body_id="robot",
        body_snapshot_hash="configured",
        grid={},
    )
    result = executor._localize(SimpleNamespace(arguments=arguments))
    assert result.final_state is ActionState.FAILED
    assert not result.dispatch_result["accepted"]


@pytest.mark.parametrize(
    "fault, expected",
    [
        (None, "AVAILABLE"),
        ("unbound", "MISSING"),
        ("wrong_service", "BLOCKED"),
        ("stale", "BLOCKED"),
        ("missing_state", "UNKNOWN"),
        ("future", "BLOCKED"),
    ],
)
def test_cleaner_requires_explicit_body_and_fresh_typed_state(fault, expected):
    from rosclaw.connectors.ros.intelligence.system_model import Signal
    from rosclaw.connectors.ros.resolver.capability_resolver import resolve_capabilities

    now = datetime.now(UTC)
    binding = {
        "name": "/clean",
        "srv_type": "std_srvs/srv/SetBool",
        "data": True,
        "state_topic": "/clean_state",
        "state_type": "std_msgs/msg/Bool",
    }
    model = RosSystemModel(
        robot_id="test",
        snapshot_id="test",
        captured_at=now,
        body={
            "effective_body_hash": "bound",
            "ros_capability_bindings": {"cleaning.enable": binding},
        },
        graph={
            "services": [{"name": "/clean", "srv_type": "std_srvs/srv/SetBool"}],
            "topics": [{"name": "/clean_state", "msg_type": "std_msgs/msg/Bool"}],
        },
        signals=[
            Signal(
                topic="/clean_state",
                source="native",
                captured_at=now,
                publisher_count=1,
                last_message_age_ms=10,
            )
        ],
    )
    if fault == "unbound":
        model.body = {}
    elif fault == "wrong_service":
        model.graph["services"][0]["srv_type"] = "std_srvs/srv/Trigger"
    elif fault == "stale":
        model.signals[0].last_message_age_ms = 2000
    elif fault == "missing_state":
        model.graph["topics"] = []
    elif fault == "future":
        model.signals[0].captured_at = now + timedelta(seconds=1)
    row = next(
        r for r in resolve_capabilities(model, now=now) if r["semantic_id"] == "cleaning.enable"
    )
    assert row["status"] == expected
    assert row["usable_for_real_execution"] is False


def test_native_context_rejects_mismatched_compiled_body_before_probe(tmp_path, monkeypatch):
    import json

    from rosclaw.body.schema import EffectiveBody
    from rosclaw.connectors.ros.context.body import configured_ros_body

    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
    (tmp_path / "body" / "refs").mkdir(parents=True)
    (tmp_path / "body" / "body.yaml").write_text("body_instance:\n  id: test\n")
    body = EffectiveBody(
        body_instance_id="test", eurdf_uri="fixture", effective_body_hash="", compiled_at=""
    )
    body.effective_body_hash = body.compute_hash()
    (tmp_path / "body" / "refs" / "effective_body.json").write_text(json.dumps(body.to_dict()))
    assert (
        configured_ros_body("test", body.compute_hash())["effective_body_hash"]
        == body.compute_hash()
    )
    with pytest.raises(ValueError, match="mission binding"):
        configured_ros_body("test", "different")
    body.frames["base"] = "tampered"
    (tmp_path / "body" / "refs" / "effective_body.json").write_text(json.dumps(body.to_dict()))
    with pytest.raises(ValueError, match="mission binding"):
        configured_ros_body("test")


@pytest.mark.parametrize(
    "polygon, expected",
    [
        ([[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]], "AVAILABLE"),
        ([[0, 0], [1, 1], [0, 1], [1, 0]], "BLOCKED"),
        ([], "BLOCKED"),
    ],
)
def test_bound_verifier_checks_actual_cleaning_geometry(polygon, expected):
    from rosclaw.connectors.ros.resolver.capability_resolver import resolve_capabilities

    now = datetime.now(UTC)
    model = RosSystemModel(
        robot_id="test",
        snapshot_id="test",
        captured_at=now,
        body={
            "effective_body_hash": "bound",
            "ros_capability_bindings": {
                "coverage.verify": {
                    "name": "rosclaw.coverage_verifier.v1",
                    "cleaning_polygon": polygon,
                }
            },
        },
    )
    row = next(
        r for r in resolve_capabilities(model, now=now) if r["semantic_id"] == "coverage.verify"
    )
    assert row["status"] == expected
    assert row["usable_for_real_execution"] is False


@pytest.mark.parametrize("acknowledged", [True, False])
def test_repair_goal_budget_requires_canonical_cancellation_ack(tmp_path, acknowledged):
    import time

    from rosclaw.connectors.ros.mission.executor import RosCoverageSimulationExecutor

    class Client:
        def __init__(self):
            self.cancelled = []
            self.callback = None

        def send_goal(self, **args):
            self.callback = args["on_result"]

        def cancel_goal(self, goal_id):
            self.cancelled.append(goal_id)
            if acknowledged:
                self.callback(5, {"error_code": 0})

    client = Client()
    executor = RosCoverageSimulationExecutor(
        owner="daemon_test",
        client=client,
        control=None,
        witness=SimpleNamespace(
            fresh=lambda: {
                "observation_complete": True,
                "cleaning_enabled": True,
                "lease_remaining_sec": 1.0,
            }
        ),
        output=tmp_path,
        body_id="base",
        body_snapshot_hash="bound",
        grid={},
    )
    executor._service = lambda *args: SimpleNamespace(ok=True, data={"values": {"success": True}})
    if acknowledged:
        result = executor._run_goal(
            "/navigate",
            "nav2_msgs/action/NavigateToPose",
            {},
            "repair",
            time.monotonic() + 10,
            goal_timeout_sec=0.01,
        )
        assert result["status"] == 5
        assert result["timed_out"] is True
    else:
        with pytest.raises(RuntimeError, match="cancellation was not acknowledged"):
            executor._run_goal(
                "/navigate",
                "nav2_msgs/action/NavigateToPose",
                {},
                "repair",
                time.monotonic() + 10,
                goal_timeout_sec=0.01,
            )
    assert client.cancelled == ["repair"]


def test_repair_reuses_independently_observed_active_cleaning_lease(tmp_path):
    import time

    from rosclaw.connectors.ros.mission.executor import RosCoverageSimulationExecutor

    sent = []

    def send_goal(**args):
        sent.append(args["goal_id"])
        args["on_result"](4, {"error_code": 0})

    control = SimpleNamespace(
        call_service=lambda *args, **kwargs: pytest.fail(
            "active state must not be redundantly toggled"
        )
    )
    executor = RosCoverageSimulationExecutor(
        owner="daemon_test",
        client=SimpleNamespace(send_goal=send_goal),
        control=control,
        witness=SimpleNamespace(
            fresh=lambda: {"cleaning_enabled": True, "lease_remaining_sec": 1.2}
        ),
        output=tmp_path,
        body_id="base",
        body_snapshot_hash="bound",
        grid={},
    )
    result = executor._run_goal("/navigate", "type", {}, "repair", time.monotonic() + 5)
    assert sent == ["repair"]
    assert result["status"] == 4
    assert executor.lease_lock is executor.control_lock


@pytest.mark.parametrize("remaining", [None, False, float("nan"), -1.0, 0.5])
def test_repair_missing_or_expiring_lease_requires_positive_service_ack(tmp_path, remaining):
    import time

    from rosclaw.connectors.ros.mission.executor import RosCoverageSimulationExecutor

    control = SimpleNamespace(
        call_service=lambda *args, **kwargs: SimpleNamespace(
            ok=True, data={"values": {"success": False}}
        )
    )
    executor = RosCoverageSimulationExecutor(
        owner="daemon_test",
        client=SimpleNamespace(send_goal=lambda **kwargs: pytest.fail("unleased motion")),
        control=control,
        witness=SimpleNamespace(
            fresh=lambda: {"cleaning_enabled": True, "lease_remaining_sec": remaining}
        ),
        output=tmp_path,
        body_id="base",
        body_snapshot_hash="bound",
        grid={},
    )
    with pytest.raises(RuntimeError, match="did not acknowledge enable"):
        executor._run_goal("/navigate", "type", {}, "repair", time.monotonic() + 5)


def scope_executor(tmp_path):
    from rosclaw.connectors.ros.mission.executor import RosCoverageSimulationExecutor

    def fail(*args, **kwargs):
        pytest.fail("scope rejection must precede every effect and observation")

    return RosCoverageSimulationExecutor(
        owner="daemon_test",
        client=SimpleNamespace(send_goal=fail),
        control=SimpleNamespace(call_service=fail),
        witness=SimpleNamespace(mark=fail, fresh=fail),
        output=tmp_path,
        body_id="base",
        body_snapshot_hash="bound",
        grid={
            "frame_id": "map",
            "width": 2,
            "height": 2,
            "resolution": 1.0,
            "origin": [0.0, 0.0],
            "accessible_cells": [0, 1, 2, 3],
        },
    )


def scope_arguments(coords, frame="map"):
    return {
        "frame_id": frame,
        "polygons": [{"points": [{"x": x, "y": y, "z": 0.0} for x, y in coords]}],
    }


@pytest.mark.parametrize(
    "coords",
    [
        [(0, 0), (2, 0), (2, 2), (0, 2)],
        [(2, 2), (2, 0), (0, 0), (0, 2)],
        [(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)],
    ],
)
def test_coverage_scope_accepts_equivalent_whole_room_polygons(tmp_path, coords):
    goal = scope_executor(tmp_path)._coverage_goal(scope_arguments(coords))
    assert goal["frame_id"] == "map"
    assert goal["polygons"][0]["points"][0] == goal["polygons"][0]["points"][-1]
    assert len(goal["polygons"][0]["points"]) == 5


@pytest.mark.parametrize(
    "case",
    ["smaller", "larger", "frame", "bowtie", "duplicate", "height", "nan", "boolean", "empty"],
)
def test_coverage_scope_rejects_before_observation_lease_or_cleaner_enable(tmp_path, case):
    from rosclaw.kernel import ActionState

    args = scope_arguments([(0, 0), (2, 0), (2, 2), (0, 2)])
    points = args["polygons"][0]["points"]
    if case == "smaller":
        points[1]["x"] = 1.0
    elif case == "larger":
        points[1]["x"] = 3.0
    elif case == "frame":
        args["frame_id"] = "odom"
    elif case == "bowtie":
        points[1], points[2] = points[2], points[1]
    elif case == "duplicate":
        points[1] = dict(points[0])
    elif case == "height":
        points[1]["z"] = 1.0
    elif case == "nan":
        points[1]["x"] = float("nan")
    elif case == "boolean":
        points[1]["x"] = True
    else:
        args["polygons"] = []
    result = scope_executor(tmp_path)._execute(
        SimpleNamespace(
            capability_id="coverage.execute",
            arguments=args,
        )
    )
    assert result.final_state is ActionState.BLOCKED
    assert result.policy_decision["allowed"] is False
    assert result.dispatch_result["accepted"] is False
    assert result.errors[0]["code"] == "COVERAGE_SCOPE_REJECTED"


@pytest.mark.parametrize(
    "bad",
    [
        {"observation_complete": False, "collision_count": 0},
        {"observation_complete": True, "collision_count": 1},
    ],
)
def test_brief_physics_fault_remains_terminal_after_a_new_good_sample(bad):
    import threading

    from rosclaw.connectors.ros.mission.executor import SimulationWitness

    witness = SimulationWitness.__new__(SimulationWitness)
    witness.lock = threading.Lock()
    witness.latest, witness.samples, witness.action_fault = None, [], None
    witness.tracking = False
    good = {"observation_complete": True, "collision_count": 0}
    witness._record(bad)
    witness._record(good)
    assert witness.fresh() == good
    assert witness.mark() == 2
    witness._record(bad)
    witness._record(good)
    with pytest.raises(RuntimeError, match="during action"):
        witness.fresh()
    assert witness.since(2) == [bad, good]


@pytest.mark.parametrize("half_width, expected_far_retries", [(0.175, 0), (0.275, 3)])
def test_repair_retry_bookkeeping_uses_configured_cleaner_footprint(
    tmp_path, monkeypatch, half_width, expected_far_retries
):
    """A Burger-size cleaner must not exhaust cells only a Waffle brush reaches."""
    from rosclaw.connectors.ros.mission import executor as module
    from rosclaw.connectors.ros.verification.coverage import CoverageVerifier

    grid = {"width": 21, "resolution": 0.05, "origin": [-0.525, -0.525]}
    verifier = CoverageVerifier(
        width=21,
        height=21,
        resolution=0.05,
        accessible_cells=list(range(441)),
        origin=(-0.525, -0.525),
        cleaning_polygon=[
            (-half_width, -half_width),
            (half_width, -half_width),
            (half_width, half_width),
            (-half_width, half_width),
        ],
    )
    observed = []
    original = module.MissedRegionRecovery

    class ThreeAttemptsDoneError(Exception):
        pass

    class CapturedRecovery(original):
        def record_attempt(self, cells, *, action_id):
            super().record_attempt(cells, action_id=action_id)
            observed.append(dict(self.attempts))
            if len(observed) == 3:
                raise ThreeAttemptsDoneError

    monkeypatch.setattr(module, "MissedRegionRecovery", CapturedRecovery)
    executor = module.RosCoverageSimulationExecutor(
        owner="daemon_test",
        client=None,
        control=None,
        witness=SimpleNamespace(since=lambda _: [], fresh=lambda: {"x": 0, "y": 0}),
        output=tmp_path,
        body_id="robot",
        body_snapshot_hash="body",
        grid=grid,
        recovery_centers=[(0, 0)],
    )
    monkeypatch.setattr(executor, "_run_goal", lambda *a, **k: {"status": 6})
    with pytest.raises(ThreeAttemptsDoneError):
        executor._repair(verifier, 0, "coverage", 1)
    # This cell is 0.30 m from the target: inside the old fixed 0.39 m retry
    # radius, but outside the smaller body's rotated cleaning polygon.
    assert observed[-1].get(10 * 21 + 16, 0) == expected_far_retries
    assert observed[-1][10 * 21 + 10] == 3
    assert verifier.result()["coverage_ratio"] == 0


@pytest.mark.parametrize(
    "half_width, cell_count, expected_goals", [(0.275, 100, 60), (0.01, 100, 200), (0.01, 200, 240)]
)
def test_repair_budget_scales_for_narrow_cleaner_without_crediting_plans(
    tmp_path, monkeypatch, half_width, cell_count, expected_goals
):
    from rosclaw.connectors.ros.mission import executor as module
    from rosclaw.connectors.ros.verification.coverage import CoverageVerifier

    verifier = CoverageVerifier(
        width=10,
        height=cell_count // 10,
        resolution=0.1,
        accessible_cells=list(range(cell_count)),
        cleaning_polygon=[
            (-half_width, -half_width),
            (half_width, -half_width),
            (half_width, half_width),
            (-half_width, half_width),
        ],
    )

    class PendingRecovery:
        attempts = {}

        def __init__(self, verifier):
            pass

        def propose(self):
            return {"ready": [{"cells": list(range(cell_count))}]}

        def record_attempt(self, cells, *, action_id):
            pass

    monkeypatch.setattr(module, "MissedRegionRecovery", PendingRecovery)
    executor = module.RosCoverageSimulationExecutor(
        owner="daemon_test",
        client=None,
        control=None,
        witness=SimpleNamespace(since=lambda _: [], fresh=lambda: {"x": 0, "y": 0}),
        output=tmp_path,
        body_id="robot",
        body_snapshot_hash="body",
        grid={"width": 10, "resolution": 0.1, "origin": [0, 0]},
        recovery_centers=[(0.05, 0.05)],
    )
    monkeypatch.setattr(executor, "_run_goal", lambda *a, **k: {"status": 6})
    records = executor._repair(verifier, 0, "coverage", 1)
    assert len(records) == expected_goals
    assert verifier.result()["coverage_ratio"] == 0
