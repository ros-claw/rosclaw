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
