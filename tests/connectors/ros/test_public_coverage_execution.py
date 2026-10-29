"""Software and authenticated UDS fixtures only; no ROS, model or physics."""

import threading
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw.app.runner import AppRunner
from rosclaw.app.store import AppStore
from rosclaw.body.compiler import EffectiveBodyCompiler
from rosclaw.body.schema import BodyYaml, EurdfProfile
from rosclaw.connectors.ros.mission.public_coverage import (
    CoverageSource,
    CoverageSourceFact,
    PublicCoverageBinding,
    PublicCoverageExecutor,
    PublicCoverageSnapshot,
)
from rosclaw.connectors.ros.public_coverage.graph_ledger import GraphAnchoredCoverage
from rosclaw.connectors.ros.public_coverage.grid import Grid
from rosclaw.kernel import (
    ActionEnvelope,
    ActionExecutionResult,
    ActionState,
    EvidenceDomain,
    EvidenceLevel,
    ExecutionMode,
)

pytest_plugins = ("tests.daemon.test_server",)


@pytest.mark.parametrize("size", [16, 24])
def test_original_rmw_gid_width_is_preserved(size):
    gid = bytes(range(size)).hex()
    source = CoverageSource(
        "odom", "/odom", "nav_msgs/msg/Odometry", "odom", 1.5, 1.5, publisher_gid=gid
    )
    assert source.publisher_gid == gid


@pytest.mark.parametrize(
    "gid", ["", "0" * 32, "0" * 48, "f" * 30, "f" * 34, "f" * 46, "f" * 50, "g" * 32]
)
def test_missing_zero_malformed_or_unsupported_rmw_gid_is_not_a_source(gid):
    with pytest.raises(ValueError, match="publisher endpoint GID"):
        CoverageSource(
            "odom", "/odom", "nav_msgs/msg/Odometry", "odom", 1.5, 1.5, publisher_gid=gid
        )


def compiled_binding():
    types = {
        "odom": "nav_msgs/msg/Odometry",
        "cleaner": "std_msgs/msg/Bool",
        "occupancy": "nav_msgs/msg/OccupancyGrid",
        "global_costmap": "nav_msgs/msg/OccupancyGrid",
        "local_costmap": "nav_msgs/msg/OccupancyGrid",
        "graph": "visualization_msgs/msg/MarkerArray",
        "map_tf": "tf2_msgs/msg/TFMessage",
        "base_tf": "tf2_msgs/msg/TFMessage",
        "scan": "sensor_msgs/msg/LaserScan",
        "depth": "sensor_msgs/msg/PointCloud2",
        "battery": "sensor_msgs/msg/BatteryState",
        "dock": "irobot_create_msgs/msg/DockStatus",
    }
    sources = [
        CoverageSource(
            k,
            "/fixture/" + k,
            v,
            None if k == "cleaner" else "map",
            1.5,
            None if k in {"cleaner", "graph"} else 1.5,
            "odom" if k == "map_tf" else "base_link" if k == "base_tf" else None,
            publisher_gid="1" * 48,
        )
        for k, v in types.items()
    ]
    profile = EurdfProfile(
        profile_id="owned_coverage_fixture",
        profile_version="1",
        vendor="synthetic software fixture",
        model="fixture",
        display_name="Fixture only",
        frames={"map": "map"},
        capability_hints={
            "all": [
                "coverage.execute_slice",
                "navigation.follow_path",
                "cleaning.enable",
                "cleaning.disable",
            ]
        },
        safety={"environment": {"real_robot_execution_allowed": False}},
        metadata={
            "driver_observation_contract_sha256": "a" * 64,
            "public_coverage_geometry": {
                "robot_radius_m": 0.20,
                "planning_tracking_margin_m": 0.03,
                "brush_radius_m": 0.15,
            },
        },
        provider_interfaces={
            "ros_capability_bindings": {
                "navigation.follow_path": {
                    "name": "/fixture/follow",
                    "action_type": "nav2_msgs/action/FollowPath",
                },
                "cleaning.enable": {"state_type": "std_msgs/msg/Bool"},
                "cleaning.disable": {"state_type": "std_msgs/msg/Bool"},
            },
            "public_coverage": {
                "schema_version": "rosclaw.public_coverage.v1",
                "odom_epoch": "fixture-odom",
                "graph_epoch": "fixture-graph",
                "execution_mode": "FIXTURE",
                "maximum_work": 6,
                "max_duration_sec": 20.0,
                "minimum_work_soc": 0.25,
                "driver_sha256": "a" * 64,
                "geometry": {
                    "robot_radius_m": 0.20,
                    "planning_tracking_margin_m": 0.03,
                    "brush_radius_m": 0.15,
                },
                "sources": [asdict(s) for s in sources],
            },
        },
    )
    body = EffectiveBodyCompiler().compile(profile, BodyYaml(body_instance={"id": "rh56-test"}))
    return body, PublicCoverageBinding.from_body(body)


class SoftwarePort:
    """Commanded state is synthetic; never presented as independent physics."""

    def __init__(self, binding):
        self.binding = binding
        origin = SimpleNamespace(
            position=SimpleNamespace(x=0.0, y=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        message = SimpleNamespace(
            info=SimpleNamespace(width=40, height=40, resolution=0.1, origin=origin),
            data=[0] * 1600,
        )
        cost = SimpleNamespace(
            metadata=SimpleNamespace(size_x=40, size_y=40, resolution=0.1, origin=origin),
            data=[0] * 1600,
        )
        self.grid = Grid(
            message,
            (2.0, 2.0),
            clearance=0.20,
            tracking_margin=0.03,
            navigation=cost,
            local_navigation=cost,
            map_from_local=(0.0, 0.0, 0.0),
        )
        self.xy, self.brush, self.native = (2.0, 2.0), False, 1.0
        self.buffer, self.calls = [], []
        self.enabled, self.release = threading.Event(), threading.Event()
        self.hold_after_enable = False
        self.fault = None
        self.stop_calls = 0
        self.motion_settled = True
        self._sample()

    def snapshot(self, *, work=True):
        b = self.binding
        facts = tuple(
            CoverageSourceFact(
                s.role,
                s.topic,
                s.message_type,
                s.frame_id,
                None if s.max_header_age_s is None else self.native,
                time.monotonic(),
                s.child_frame_id,
                s.publisher_gid,
            )
            for s in b.sources
        )
        result = PublicCoverageSnapshot(
            b.body_id,
            b.body_snapshot_hash,
            b.driver_sha256,
            b.odom_epoch,
            b.graph_epoch,
            self.native,
            time.monotonic(),
            self.xy,
            self.xy,
            self.brush,
            0.0,
            0.0,
            self.motion_settled,
            1.0,
            False,
            ((0, (0.0, 0.0)), (1, (4.0, 0.0))),
            facts,
            self.grid if work else None,
        )
        if self.fault == "epoch":
            result = replace(result, odom_epoch="different")
        elif self.fault == "body":
            result = replace(result, body_snapshot_hash="b" * 64)
        elif self.fault == "low_soc":
            result = replace(result, soc=0.249)
        elif self.fault == "docked":
            result = replace(result, docked=True)
        elif self.fault == "publisher":
            result = replace(
                result,
                sources=tuple(
                    replace(f, publisher_gid="2" * 48) if f.role == "depth" else f for f in facts
                ),
            )
        elif self.fault == "depth":
            result = replace(
                result,
                sources=tuple(
                    replace(f, header_native_s=self.native - 3.0) if f.role == "depth" else f
                    for f in facts
                ),
            )
        return result

    def _sample(self, *, work=True):
        self.native += 0.05
        self.buffer.append(self.snapshot(work=work))

    def drain(self):
        result, self.buffer = self.buffer, []
        return result

    def execute(
        self, capability, arguments, *, parent, primitive_id, stop_requested, deadline_monotonic
    ):
        assert time.monotonic() < deadline_monotonic
        self.calls.append((capability, parent.action_id, parent.session_id, primitive_id))
        if capability.startswith("cleaning."):
            self.brush = capability == "cleaning.enable"
            self._sample()
            if self.brush:
                self.enabled.set()
                if self.hold_after_enable:
                    while not stop_requested.is_set() and not self.release.wait(0.01):
                        pass
        else:
            assert capability == "navigation.follow_path"
            points = np.asarray(arguments["path_xy"])
            for a, b in zip(points, points[1:], strict=False):
                for point in np.linspace(a, b, max(2, int(np.linalg.norm(b - a) / 0.025) + 1)):
                    self.xy = tuple(map(float, point))
                    self._sample()
        return ActionExecutionResult(
            final_state=ActionState.COMPLETED,
            evidence_level=EvidenceLevel.DRIVER_CONFIRMED,
            evidence_domain=EvidenceDomain.FIXTURE,
            dispatch_result={
                "parent_action_id": parent.action_id,
                "owned_primitive_id": "wrong" if self.fault == "receipt" else primitive_id,
            },
            verification_result={"passed": True},
        )

    def emergency_stop(self):
        self.stop_calls += 1
        self.brush = False
        self._sample(work=False)
        self.release.set()
        return {"acknowledged": True}


def setup_executor():
    body, binding = compiled_binding()
    port = SoftwarePort(binding)
    ledger = GraphAnchoredCoverage(
        odom_epoch=binding.odom_epoch,
        graph_epoch=binding.graph_epoch,
        brush_radius=binding.brush_radius_m,
    )
    executor = PublicCoverageExecutor(binding=binding, port=port, ledger=ledger)
    action = ActionEnvelope(
        action_id="fixture-owned-coverage",
        actor_id="fixture",
        agent_framework="fixture",
        session_id="coverage-session",
        body_id=binding.body_id,
        body_snapshot_hash=binding.body_snapshot_hash,
        capability_id="coverage.execute_slice",
        execution_mode=ExecutionMode.FIXTURE,
        arguments={"maximum_work": 6},
        deadline_at=datetime.now(UTC) + timedelta(seconds=20),
    )
    return body, executor, port, action


def test_real_library_work_runs_inside_one_parent_and_keeps_fixture_scope():
    _, executor, port, action = setup_executor()
    result = executor(action)
    assert result.final_state is ActionState.COMPLETED, result.errors
    assert result.evidence_domain is EvidenceDomain.FIXTURE
    assert result.verification_result["whole_house_success"] is None
    assert result.dispatch_result["completed_work"] >= 1
    assert result.dispatch_result["completed_continuous_strokes"] >= 1
    assert port.calls and all(c[1:3] == (action.action_id, action.session_id) for c in port.calls)
    assert len({c[3] for c in port.calls}) == len(port.calls)
    assert not port.brush and port.stop_calls == 0
    assert executor.ledger.anchor.cleaned


class OneLiveCancellationPort(SoftwarePort):
    """One explicitly synthetic original terminal; never physical evidence."""

    def __init__(self, binding, defect=None):
        super().__init__(binding)
        self.cancelled = False
        self.defect = defect

    def execute(self, capability, arguments, **kwargs):
        if capability == "navigation.follow_path" and self.brush and not self.cancelled:
            self.cancelled = True
            parent, primitive = kwargs["parent"], kwargs["primitive_id"]
            self.calls.append((capability, parent.action_id, parent.session_id, primitive))
            path = np.asarray(arguments["path_xy"], float)
            self.xy = tuple(map(float, path[0] + 0.25 * (path[1] - path[0])))
            self.motion_settled = self.defect != "unknown_original_goal"
            self._sample()
            dispatch = {
                "parent_action_id": parent.action_id,
                "owned_primitive_id": primitive,
                "action": "/follow_path",
                "action_type": "nav2_msgs/action/FollowPath",
                "status": 5,
                "cancel_result_received": True,
                "live_path_invalidation": "Fixture fresh geometry changed",
                "live_path_evidence": "fixture-only-public-geometry",
            }
            if self.defect == "ack_only":
                dispatch["status"] = None
            if self.defect == "not_terminal":
                dispatch["status"] = 2
            if self.defect == "no_cancel_ack":
                dispatch["cancel_result_received"] = False
            if self.defect == "wrong_parent":
                dispatch["parent_action_id"] = "other"
            if self.defect == "wrong_primitive":
                dispatch["owned_primitive_id"] = "other"
            return ActionExecutionResult(
                final_state=ActionState.FAILED,
                evidence_level=EvidenceLevel.REQUESTED,
                evidence_domain=EvidenceDomain.REAL
                if self.defect == "wrong_domain"
                else EvidenceDomain.FIXTURE,
                dispatch_result=dispatch,
                verification_result={"passed": False},
            )
        return super().execute(capability, arguments, **kwargs)


def test_known_original_live_cancellation_replans_only_after_stopped_off():
    _, executor, old_port, action = setup_executor()
    port = OneLiveCancellationPort(old_port.binding)
    executor.port = port
    result = executor(action)
    assert result.final_state is ActionState.COMPLETED, result.errors
    recovery = result.dispatch_result["observed_live_path_recoveries"]
    assert len(recovery) == 1 and recovery[0]["fresh_stopped_off"] is True
    assert recovery[0]["original_native_terminal_known"] is True
    assert recovery[0]["replayed_native_goal"] is False
    assert result.dispatch_result["completed_continuous_strokes"] >= 1
    assert len({call[3] for call in port.calls}) == len(port.calls)
    assert not port.brush and port.stop_calls == 0


@pytest.mark.parametrize(
    "defect",
    [
        "ack_only",
        "not_terminal",
        "no_cancel_ack",
        "wrong_parent",
        "wrong_primitive",
        "unknown_original_goal",
        "wrong_domain",
    ],
)
def test_uncertain_or_unowned_cancellation_cannot_enter_replanning(defect):
    _, executor, old_port, action = setup_executor()
    port = OneLiveCancellationPort(old_port.binding, defect)
    executor.port = port
    result = executor(action)
    assert result.final_state is ActionState.FAILED
    assert result.dispatch_result["observed_live_path_recoveries"] == []
    assert executor.stopping.is_set() and port.stop_calls > 0
    assert executor(action).final_state is ActionState.BLOCKED


def test_stopped_off_recovery_cannot_extend_five_second_window(monkeypatch):
    _, executor, port, _ = setup_executor()
    clock = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    executor._deadline = 160.0

    def delayed_cleanup(*_args):
        clock[0] += 6.0
        return port.snapshot(work=False)

    monkeypatch.setattr(executor, "_primitive", delayed_cleanup)
    with pytest.raises(RuntimeError, match="five-second"):
        executor._recover_observed_live_path({}, SimpleNamespace())


def test_rejected_new_owner_receives_no_prior_recovery_evidence():
    _, executor, _, action = setup_executor()
    executor._recoveries = [{"original_primitive_id": "previous-owner"}]
    result = executor(replace(action, body_snapshot_hash="wrong"))
    assert result.final_state is ActionState.BLOCKED
    assert result.dispatch_result["observed_live_path_recoveries"] == []


@pytest.mark.parametrize(
    "fault", ["epoch", "body", "depth", "publisher", "receipt", "low_soc", "docked"]
)
def test_bad_bound_observations_or_wrong_primitive_receipt_fail_closed(fault):
    _, executor, port, action = setup_executor()
    port.fault = fault
    result = executor(action)
    assert result.final_state is ActionState.FAILED
    assert executor.stopping.is_set() and not port.brush
    assert not any(c[0] == "navigation.follow_path" for c in port.calls)
    assert executor(action).final_state is ActionState.BLOCKED


def test_unknown_old_motion_never_enters_new_work():
    _, executor, port, action = setup_executor()
    port.motion_settled = False
    assert executor(action).final_state is ActionState.FAILED
    assert not port.calls and port.stop_calls == 1


def test_body_binding_cannot_be_changed_or_promoted_to_real():
    body, binding = compiled_binding()
    with pytest.raises(ValueError):
        replace(binding, execution_mode=ExecutionMode.REAL)
    body.provider_interfaces["public_coverage"]["driver_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="hash differs"):
        PublicCoverageBinding.from_body(body)


@pytest.mark.parametrize("loss", ["session", "emergency_stop"])
def test_actual_daemon_loss_between_brush_enable_and_next_native_goal(daemon_client, loss):
    client, runtime, _permits, daemon = daemon_client
    _, executor, port, action = setup_executor()
    port.hold_after_enable = True
    runtime.action_gateway.register_executor(
        "coverage.execute_slice", ExecutionMode.FIXTURE, executor
    )
    runtime.register_driver("owned_fixture_coverage", executor)
    client.create_session(
        session_id=action.session_id,
        actor_id=action.actor_id,
        agent_framework=action.agent_framework,
        body_scope=[action.body_id],
        capability_scope=[action.capability_id],
    )
    try:
        ticket = client.request_action(action)
        assert port.enabled.wait(3), client.get_action_status(ticket["action_id"])
        assert port.brush
        assert client.get_action_status(ticket["action_id"])["scheduler_state"] == "RUNNING"
        prior_calls = len(port.calls)
        if loss == "session":
            client.close_session(
                action.session_id, reason="fixture owner disappeared between native primitives"
            )
        else:
            cancelled = client.cancel_action(ticket["action_id"])
            assert cancelled["cancelled"] is False
            assert cancelled["code"] == "ACTIVE_ACTION_REQUIRES_EMERGENCY_STOP"
            assert port.brush
            client.emergency_stop("fixture stop between coverage primitives", source="fixture")
        status = client.wait_for_action(ticket["action_id"], timeout_sec=3)
        assert status["final_state"] == ("ORPHANED" if loss == "session" else "FAILED")
        assert not port.brush and executor.stopping.is_set()
        assert len(port.calls) == prior_calls
        assert client.get_runtime_status()["hardware_actions_executed"] == 0
    finally:
        port.release.set()


def test_stop_ack_without_new_observations_is_not_stop_verification(monkeypatch):
    _, executor, port, _action = setup_executor()
    old = port.snapshot()
    monkeypatch.setattr(port, "snapshot", lambda **_kwargs: old)
    monkeypatch.setattr(port, "emergency_stop", lambda: {"acknowledged": True})
    result = executor.emergency_stop()
    assert result["acknowledged"] is True
    assert result["physical_stop_observed"] is False


def test_historical_credit_uses_original_odom_stamp_and_capture_freshness():
    _, executor, port, _action = setup_executor()
    sample = port.buffer[0]
    captured = time.monotonic() - 10.0
    old = replace(
        sample,
        captured_monotonic_s=captured,
        sources=tuple(replace(f, received_monotonic_s=captured) for f in sample.sources),
    )
    port.buffer = [old]
    executor._record()
    assert len(executor.ledger.evidence_samples) == 1
    port.buffer = [replace(old, native_s=old.native_s + 0.1)]
    with pytest.raises(ValueError, match="out-of-order"):
        executor._record()
    assert len(executor.ledger.evidence_samples) == 1


def test_past_outer_deadline_stops_before_dispatch():
    _, executor, port, action = setup_executor()
    action.deadline_at = datetime.now(UTC) - timedelta(seconds=1)
    result = executor(action)
    assert result.final_state is ActionState.FAILED
    assert not port.calls and port.stop_calls == 1


def test_headerless_brush_state_never_invents_header_provenance():
    _, binding = compiled_binding()
    source = next(s for s in binding.sources if s.role == "cleaner")
    assert source.frame_id is None and source.max_header_age_s is None
    with pytest.raises(ValueError, match="Headerless"):
        replace(source, max_header_age_s=1.5)


@pytest.mark.parametrize("case", ["receive_future", "header_future", "fake_brush_header"])
def test_original_clock_domains_and_header_contracts_are_not_relaxed(case, monkeypatch):
    _, executor, port, _ = setup_executor()
    sample = port.snapshot()
    monotonic = sample.captured_monotonic_s + 0.01
    monkeypatch.setattr(time, "monotonic", lambda: monotonic)
    facts = []
    for fact in sample.sources:
        if case == "receive_future" and fact.role == "depth":
            fact = replace(fact, received_monotonic_s=time.monotonic() + 0.01)
        elif case == "header_future" and fact.role == "depth":
            fact = replace(fact, header_native_s=sample.native_s + 0.101)
        elif case == "fake_brush_header" and fact.role == "cleaner":
            fact = replace(fact, header_native_s=sample.native_s)
        facts.append(fact)
    with pytest.raises(ValueError, match="freshness"):
        executor._validate(replace(sample, sources=tuple(facts)))


@pytest.mark.parametrize("future", [0.5, 1.001])
def test_declared_tf_future_bound_preserves_original_native_absolute_guard(future):
    _, executor, port, _ = setup_executor()
    sources = tuple(
        replace(s, max_header_future_s=1.0) if s.role == "map_tf" else s
        for s in executor.binding.sources
    )
    executor.binding = replace(executor.binding, sources=sources)
    port.binding = executor.binding
    sample = port.snapshot()
    facts = tuple(
        replace(f, header_native_s=sample.native_s + future) if f.role == "map_tf" else f
        for f in sample.sources
    )
    changed = replace(sample, sources=facts)
    if future <= 1.0:
        executor._validate(changed, work=False, required_roles={"map_tf"})
    else:
        with pytest.raises(ValueError, match="freshness"):
            executor._validate(changed, work=False, required_roles={"map_tf"})


@pytest.mark.parametrize(
    "role,kind,limit",
    [
        ("odom", "nav_msgs/msg/Odometry", 0.101),
        ("map_tf", "tf2_msgs/msg/TFMessage", 1.001),
        ("map_tf", "tf2_msgs/msg/TFMessage", float("nan")),
    ],
)
def test_compiled_future_bound_cannot_exceed_original_role_guard(role, kind, limit):
    with pytest.raises(ValueError, match="future tolerance"):
        CoverageSource(
            role,
            "/fixture/source",
            kind,
            "map",
            1.0,
            1.0,
            publisher_gid="1" * 32,
            max_header_future_s=limit,
        )


def test_fresh_stop_proof_survives_unavailable_map_registration(monkeypatch):
    _, executor, port, _ = setup_executor()
    original = port.snapshot
    monkeypatch.setattr(
        port,
        "snapshot",
        lambda **_kwargs: replace(original(), map_xy=None, graph_epoch="unavailable", grid=None),
    )
    result = executor.emergency_stop()
    assert result["physical_stop_observed"] is True


def test_bundled_app_runs_through_actual_daemon_with_one_owned_action(daemon_client, tmp_path):
    client, runtime, _permits, _daemon = daemon_client
    _body, executor, port, action = setup_executor()
    runtime.action_gateway.register_executor(
        "coverage.execute_slice", ExecutionMode.FIXTURE, executor
    )
    runtime.register_driver("owned_fixture_coverage", executor)
    store = AppStore(tmp_path / "isolated-app-store")
    store.install("public-coverage-slice")
    _record, manifest = store.resolve("public-coverage-slice")
    result = AppRunner(client).run(
        manifest,
        body_id=action.body_id,
        body_snapshot_hash=action.body_snapshot_hash,
        execution_mode=ExecutionMode.FIXTURE,
    )
    assert result.status == "success", [v for v in result.verification if not v["passed"]]
    assert len(result.receipts) == 1
    receipt = result.receipts[0]
    assert receipt["evidence_domain"] == "FIXTURE"
    assert (
        receipt["final_state"] == "DEGRADED"
    )  # Preserve canonical limited-evidence classification.
    assert receipt["verification_result"]["whole_house_success"] is None
    assert receipt["dispatch_result"]["completed_continuous_strokes"] >= 1
    assert all(
        call[1] == receipt["action_id"] and call[2] == result.session_id for call in port.calls
    )
    assert client.get_session(result.session_id)["session"]["state"] == "CLOSED"
    assert not port.brush


def test_stop_observation_does_not_request_work_geometry(monkeypatch):
    _, executor, port, _ = setup_executor()
    original = port.snapshot
    requests = []

    def observe(*, work=True):
        requests.append(work)
        if work:
            raise ValueError("Work geometry unavailable")
        return replace(original(), grid=None)

    monkeypatch.setattr(port, "snapshot", observe)
    result = executor.emergency_stop()
    assert result["physical_stop_observed"] is True
    assert set(requests) == {False}


def test_planning_service_skips_grid_rebuild_but_requires_all_live_sources(monkeypatch):
    _, executor, port, _ = setup_executor()
    executor._deadline = time.monotonic() + 10.0
    original = port.snapshot
    requests = []

    def observe(*, work=True):
        requests.append(work)
        assert work is False
        return replace(original(), grid=None)

    monkeypatch.setattr(port, "snapshot", observe)
    assert executor._planning_service().grid is None
    port.fault = "depth"
    with pytest.raises(ValueError, match="freshness"):
        executor._planning_service()
    assert requests == [False, False]


def test_planning_service_keeps_original_deadline_and_energy_guard():
    _, executor, port, _ = setup_executor()
    executor._deadline = time.monotonic() - 1.0
    with pytest.raises(TimeoutError, match="Original"):
        executor._planning_service()
    executor._deadline = time.monotonic() + 10.0
    original = port.snapshot
    port.snapshot = lambda **_kwargs: replace(original(), soc=0.1)
    with pytest.raises(RuntimeError, match="energy"):
        executor._planning_service()


@pytest.mark.parametrize("entry_x", [2.0, 2.4])
@pytest.mark.parametrize("live_cancellation", [False, True])
def test_fresh_post_entry_obstacle_replans_before_enabling_brush(
    monkeypatch, entry_x, live_cancellation
):
    _, executor, port, action = setup_executor()
    if live_cancellation:
        port = OneLiveCancellationPort(port.binding)
        executor.port = port
    selected = []

    def proposals(*_args, **_kwargs):
        selected.append(port.xy)
        if len(selected) == 1:
            # The proposal was safe when selected. A newly occupied cell far
            # from the actual pose invalidates its end, not the entry transfer.
            path = [[entry_x, 2.0], [3.2, 2.0]]
            assert port.grid.motion_path_safe(path)
            port.grid.free[20, 32] = False
        else:
            path = [list(port.xy), [port.xy[0], port.xy[1] + 0.7]]
        return {"kind": "WORK", "path_xy": path}

    monkeypatch.setattr(executor.ledger, "next_work", proposals)
    result = executor(action)
    assert result.final_state is ActionState.COMPLETED, result.errors
    assert len(selected) == (3 if live_cancellation else 2)
    assert result.dispatch_result["completed_work"] == 1
    assert result.dispatch_result["completed_continuous_strokes"] == 1
    recovery = result.dispatch_result["observed_post_entry_recoveries"]
    assert len(recovery) == 1 and recovery[0]["fresh_stopped_off"] is True
    assert recovery[0]["original_native_terminal_known"] is True
    assert recovery[0]["original_deadline_preserved"] is True
    assert recovery[0]["replayed_native_goal"] is False
    assert len(result.dispatch_result["observed_live_path_recoveries"]) == int(live_cancellation)
    assert sum(c[0] == "cleaning.enable" for c in port.calls) == (2 if live_cancellation else 1)
    assert not port.brush and port.stop_calls == 0
    assert all(c[1:3] == (action.action_id, action.session_id) for c in port.calls)
    assert len({c[3] for c in port.calls}) == len(port.calls)


def test_repeated_post_entry_changes_consume_original_work_attempts(monkeypatch):
    _, executor, port, action = setup_executor()
    selected = []

    def proposals(*_args, **_kwargs):
        selected.append(1)
        port.grid.free[20, 32] = False
        return {"kind": "WORK", "path_xy": [[2.0, 2.0], [3.2, 2.0]]}

    monkeypatch.setattr(executor.ledger, "next_work", proposals)
    result = executor(replace(action, arguments={"maximum_work": 3}))
    assert result.final_state is ActionState.FAILED
    assert len(selected) == 3
    assert len(result.dispatch_result["observed_post_entry_recoveries"]) == 3
    assert result.dispatch_result["completed_work"] == 0
    assert not any(c[0] == "cleaning.enable" for c in port.calls)
    assert executor.stopping.is_set() and not port.brush and port.stop_calls == 1
    assert executor(action).final_state is ActionState.BLOCKED


@pytest.mark.parametrize("fault", ["body", "epoch", "depth", "publisher", "unknown_native"])
def test_fresh_post_entry_change_does_not_swallow_other_guard_failures(monkeypatch, fault):
    _, executor, port, action = setup_executor()
    selected = []

    def proposals(*_args, **_kwargs):
        selected.append(1)
        port.grid.free[20, 32] = False
        if fault == "unknown_native":
            port.motion_settled = False
        else:
            port.fault = fault
        return {"kind": "WORK", "path_xy": [[2.0, 2.0], [3.2, 2.0]]}

    monkeypatch.setattr(executor.ledger, "next_work", proposals)
    result = executor(action)
    assert result.final_state is ActionState.FAILED
    assert len(selected) == 1
    assert result.dispatch_result["observed_post_entry_recoveries"] == []
    assert not any(c[0] == "cleaning.enable" for c in port.calls)
    assert executor.stopping.is_set() and port.stop_calls == 1


def test_generic_geometry_exception_is_not_a_replanning_permission(monkeypatch):
    _, executor, port, action = setup_executor()

    def unavailable(*_args, **_kwargs):
        raise ValueError("Original unknown geometry source")

    monkeypatch.setattr(port.grid, "motion_path_safe", unavailable)
    result = executor(action)
    assert result.final_state is ActionState.FAILED
    assert result.dispatch_result["observed_post_entry_recoveries"] == []
    assert executor.stopping.is_set() and port.stop_calls == 1
