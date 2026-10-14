"""Synthetic software contracts only; no World, robot or physical receipt."""

import math
import threading
import time
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from rosclaw.body.compiler import EffectiveBodyCompiler
from rosclaw.body.schema import BodyYaml, EurdfProfile
from rosclaw.connectors.ros.mission.executor import RosCoverageSimulationExecutor, SimulationWitness
from rosclaw.connectors.ros.mission.navigation_binding import SimulationNavigationBinding
from rosclaw.connectors.ros.mission.navigation_supervision import NavigationExecutionSupervision
from rosclaw.connectors.ros.transport.base import RosTransportResult
from rosclaw.kernel import ActionState, ExecutionMode


def compiled_body(namespace="/robot", frame="world"):
    profile = EurdfProfile(
        profile_id="synthetic_navigation_only",
        profile_version="1",
        vendor="synthetic contract",
        display_name="Synthetic navigation contract",
        model="synthetic",
        frames={"map": frame},
        capability_hints={"all": ["navigation.navigate_to_pose"]},
        safety={
            "environment": {"real_robot_execution_allowed": False},
            "navigation_simulation": {
                "max_goal_distance_m": 1.0,
                "max_duration_sec": 3.0,
                "goal_position_tolerance_m": 0.05,
                "goal_yaw_tolerance_rad": 0.1,
                "max_speed_mps": 0.2,
                "max_yaw_rate_rps": 0.5,
                "stop_speed_mps": 0.005,
                "stop_yaw_rate_rps": 0.01,
            },
        },
        provider_interfaces={
            "ros_capability_bindings": {
                "navigation.navigate_to_pose": {
                    "name": namespace + "/go",
                    "action_type": "nav2_msgs/action/NavigateToPose",
                },
                "safety.simulation_lease": {
                    "name": namespace + "/deadman",
                    "srv_type": "std_srvs/srv/SetBool",
                },
                "safety.simulation_observation": {
                    "name": namespace + "/truth",
                    "msg_type": "std_msgs/msg/String",
                    "evidence_domain": "GAZEBO_PHYSICS",
                    "run_id": "synthetic-contract",
                },
            }
        },
    )
    return EffectiveBodyCompiler().compile(profile, BodyYaml(body_instance={"id": "mobile"}))


class Witness:
    def __init__(self, binding):
        self.navigation_binding = binding
        self.samples = []
        self.x, self.y, self.yaw = 0.0, 0.0, 0.0
        self.lease = False
        self.fault = None

    def fresh(self):
        b = self.navigation_binding
        row = {
            "body_id": b.body_id,
            "body_snapshot_hash": b.body_snapshot_hash,
            "run_id": b.run_id,
            "frame_id": b.frame_id,
            "lease_enabled": self.lease,
            "x": self.x,
            "y": self.y,
            "yaw": self.yaw,
            "time_sec": time.monotonic(),
            "captured_at": datetime.now(UTC).isoformat(),
            "collision_count": 0,
            "observation_complete": True,
            "evidence_domain": "GAZEBO_PHYSICS",
            "linear_speed_mps": 0.0,
            "angular_speed_rps": 0.0,
        }
        if self.fault == "stale":
            raise RuntimeError("independent simulation witness is stale")
        if self.fault == "collision":
            raise RuntimeError("independent simulation witness detected collision")
        if self.fault == "wrong_body":
            row["body_snapshot_hash"] = "different"
        if self.fault == "overspeed":
            row["linear_speed_mps"] = 0.3
        self.samples.append(row)
        return dict(row)

    def mark(self):
        return len(self.samples)

    def since(self, offset):
        rows = [dict(s) for s in self.samples[offset:]]
        if self.fault == "transient_overspeed":
            rows[0]["linear_speed_mps"] = 0.3
        return rows


class Control:
    def __init__(self, witness):
        self.witness, self.calls = witness, []
        self.reject_enable = False
        self.ignore_stop = False

    def call_service(self, name, args, **kwargs):
        assert name == self.witness.navigation_binding.lease_service, "must never call a cleaner"
        self.calls.append((name, dict(args)))
        accepted = not (self.reject_enable and args["data"])
        if accepted and not (self.ignore_stop and args["data"] is False):
            self.witness.lease = args["data"]
        return RosTransportResult(ok=True, data={"values": {"success": accepted}})


class Client:
    def __init__(self, witness):
        self.witness, self.sent, self.cancelled = witness, [], []
        self.fault = None

    def send_goal(self, **kwargs):
        self.sent.append(kwargs)
        position = kwargs["args"]["pose"]["pose"]["position"]
        self.witness.x, self.witness.y = position["x"], position["y"]
        if self.fault == "wrong_position":
            self.witness.x += 0.1
        elif self.fault == "wrong_yaw":
            self.witness.yaw = 0.2
        elif self.fault not in (None, "aborted", "never_finishes"):
            self.witness.fault = self.fault
        if self.fault != "never_finishes":
            kwargs["on_result"](6 if self.fault == "aborted" else 4, {"error_code": 0})

    def cancel_goal(self, goal_id):
        self.cancelled.append(goal_id)
        return True


def make_executor(tmp_path, namespace="/robot", frame="world", supervision_factory=None):
    body = compiled_body(namespace, frame)
    binding = SimulationNavigationBinding.from_body(body)
    witness = Witness(binding)
    control, client = Control(witness), Client(witness)
    executor = RosCoverageSimulationExecutor(
        owner="daemon_unit_fixture",
        client=client,
        control=control,
        witness=witness,
        output=tmp_path,
        body_id=body.body_instance_id,
        body_snapshot_hash=body.effective_body_hash,
        grid=None,
        navigation_binding=binding,
        navigation_supervision=supervision_factory(binding) if supervision_factory else None,
    )
    action = SimpleNamespace(
        action_id="synthetic-navigation",
        execution_mode=ExecutionMode.SIMULATION,
        body_id=body.body_instance_id,
        body_snapshot_hash=body.effective_body_hash,
        capability_id="navigation.navigate_to_pose",
        verification_policy=SimpleNamespace(timeout_sec=3),
        arguments={
            "pose": {
                "header": {"frame_id": frame},
                "pose": {
                    "position": {"x": 0.2, "y": 0.1, "z": 0.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
            }
        },
    )
    return executor, action, witness, control, client


class SyntheticSupervision(NavigationExecutionSupervision):
    """Fault injection only; never an implementation for physical execution."""

    def __init__(self, binding):
        super().__init__(binding)
        self.fail = False
        self.admitted = None
        self.duration = None

    def begin(self, action_id, *, duration_sec):
        if self.fail:
            raise RuntimeError("synthetic source capacity or admission failure")
        self.admitted, self.duration = action_id, duration_sec

    def check(self, action_id):
        if self.fail or action_id != self.admitted:
            raise RuntimeError("synthetic current source/ROS/generation loss")


def test_supervision_admission_loss_blocks_enable_and_latches_future_action(tmp_path):
    executor, action, witness, control, client = make_executor(
        tmp_path, supervision_factory=SyntheticSupervision
    )
    executor.navigation_supervision.fail = True
    result = executor(action)
    assert result.final_state is ActionState.FAILED
    assert not client.sent and not witness.lease
    assert all(args["data"] is False for _, args in control.calls)
    executor.navigation_supervision.fail = False
    assert executor(action).final_state is ActionState.BLOCKED


def test_wrong_body_hash_has_precise_refusal_before_supervision_or_dispatch(tmp_path):
    executor, action, witness, control, client = make_executor(
        tmp_path, supervision_factory=SyntheticSupervision
    )
    action.body_snapshot_hash = "0" * 64
    result = executor(action)
    assert result.final_state is ActionState.BLOCKED
    assert result.errors[0]["code"] == "BODY_SNAPSHOT_MISMATCH"
    assert executor.navigation_supervision.admitted is None
    assert not control.calls and not client.sent and not witness.lease


def test_source_loss_between_enable_and_send_never_dispatches_goal(tmp_path):
    executor, action, witness, control, client = make_executor(
        tmp_path, supervision_factory=SyntheticSupervision
    )
    original = control.call_service

    def inject_after_enable(name, args, **kwargs):
        result = original(name, args, **kwargs)
        if args["data"]:
            executor.navigation_supervision.fail = True
        return result

    control.call_service = inject_after_enable
    result = executor(action)
    assert result.final_state is ActionState.FAILED
    assert not client.sent and not witness.lease and executor.stopping.is_set()
    assert control.calls[-1][1] == {"data": False}


@pytest.mark.parametrize("immediate_result", [True, False])
def test_source_loss_after_send_cancels_revokes_and_never_verifies_success(
    tmp_path, immediate_result
):
    executor, action, witness, control, client = make_executor(
        tmp_path, supervision_factory=SyntheticSupervision
    )
    if not immediate_result:
        client.fault = "never_finishes"
    original = client.send_goal

    def inject_after_dispatch(**kwargs):
        original(**kwargs)
        executor.navigation_supervision.fail = True

    client.send_goal = inject_after_dispatch
    result = executor(action)
    assert result.final_state is ActionState.FAILED
    assert client.sent and client.cancelled and executor.stopping.is_set()
    assert not witness.lease and control.calls[-1][1] == {"data": False}
    assert result.verification_result["physical_stop_verified"] is False


def test_healthy_supervision_preserves_independent_goal_and_stop_verification(tmp_path):
    executor, action, witness, _, _ = make_executor(
        tmp_path, supervision_factory=SyntheticSupervision
    )
    result = executor(action)
    assert result.final_state is ActionState.COMPLETED, result.errors
    assert executor.navigation_supervision.admitted == action.action_id
    assert (
        0 < executor.navigation_supervision.duration <= executor.navigation_binding.max_duration_sec
    )
    assert result.verification_result["physical_stop_verified"] is True
    assert not witness.lease and executor._supervised_action_id is None


@pytest.mark.parametrize(
    "namespace,frame", [("/robot", "world"), ("/other/renamed", "floor_frame")]
)
def test_navigation_completes_without_grid_or_cleaner_and_independently_stops(
    tmp_path, namespace, frame
):
    executor, action, witness, control, client = make_executor(tmp_path, namespace, frame)
    result = executor(action)
    assert result.final_state is ActionState.COMPLETED, result.errors
    assert result.verification_result["physical_stop_verified"] is True
    assert result.verification_result["goal_yaw_error_rad"] == 0
    assert client.sent[0]["action"] == namespace + "/go"
    assert all(name == namespace + "/deadman" for name, _ in control.calls)
    assert not witness.lease and not executor.grid
    assert "cleaning_enabled" not in witness.fresh()


@pytest.mark.parametrize(
    "fault",
    [
        "REAL",
        "SHADOW",
        "coverage",
        "localize",
        "hash",
        "far",
        "frame",
        "orientation",
        "endpoint_argument",
        "nan",
    ],
)
def test_invalid_intent_is_blocked_before_any_service_or_goal(tmp_path, fault):
    executor, action, _, control, client = make_executor(tmp_path)
    if fault in {"REAL", "SHADOW"}:
        action.execution_mode = ExecutionMode(fault)
    elif fault == "coverage":
        action.capability_id = "coverage.execute"
    elif fault == "localize":
        action.capability_id = "localization.set_initial_pose"
    elif fault == "hash":
        action.body_snapshot_hash = "different"
    elif fault == "far":
        action.arguments["pose"]["pose"]["position"]["x"] = 1.1
    elif fault == "frame":
        action.arguments["pose"]["header"]["frame_id"] = "other"
    elif fault == "orientation":
        action.arguments["pose"]["pose"]["orientation"]["w"] = 0.5
    elif fault == "endpoint_argument":
        action.arguments["action_name"] = "/bypass"
    else:
        action.arguments["pose"]["pose"]["position"]["x"] = math.nan
    result = executor(action)
    assert result.final_state is ActionState.BLOCKED
    assert not client.sent and not control.calls


@pytest.mark.parametrize(
    "fault",
    [
        "aborted",
        "stale",
        "collision",
        "wrong_body",
        "overspeed",
        "transient_overspeed",
        "wrong_position",
        "wrong_yaw",
        "never_finishes",
    ],
)
def test_success_status_cannot_hide_bad_independent_observations_or_timeout(tmp_path, fault):
    executor, action, witness, control, client = make_executor(tmp_path)
    client.fault = fault
    if fault == "never_finishes":
        action.verification_policy.timeout_sec = 0.01
    result = executor(action)
    assert result.final_state is ActionState.FAILED, (fault, result)
    assert not witness.lease
    assert control.calls[-1][1] == {"data": False}
    assert client.cancelled


def test_lease_rejection_is_not_dispatch_acceptance(tmp_path):
    executor, action, witness, control, client = make_executor(tmp_path)
    control.reject_enable = True
    result = executor(action)
    assert result.final_state is ActionState.FAILED
    assert not result.dispatch_result["accepted"] and not client.sent and not witness.lease


def test_stop_service_ack_without_measured_stop_cannot_complete(tmp_path):
    executor, action, _, control, _ = make_executor(tmp_path)
    control.ignore_stop = True
    result = executor(action)
    assert result.final_state is ActionState.FAILED
    assert "stop" in result.errors[0]["message"]


def test_cancel_transport_failure_still_revokes_bound_lease(tmp_path):
    executor, _, witness, control, client = make_executor(tmp_path)
    executor.goal_id = "active"
    witness.lease = True

    def fail(_):
        raise RuntimeError("cancel disconnected")

    client.cancel_goal = fail
    result = executor.emergency_stop()
    assert control.calls == [(executor.navigation_binding.lease_service, {"data": False})]
    assert not result["acknowledged"] and not witness.lease
    assert result["physical_stop_verified"] is True


@pytest.mark.parametrize(
    "fault",
    ["tampered", "real_allowed", "blocked", "missing_limit", "bad_type", "relative_endpoint"],
)
def test_compiled_body_binding_refuses_missing_or_conflicting_truth(fault):
    body = compiled_body()
    if fault == "real_allowed":
        body.safety["environment"]["real_robot_execution_allowed"] = True
    elif fault == "blocked":
        body.capabilities.setdefault("blocked", []).append("navigation.navigate_to_pose")
    elif fault == "missing_limit":
        body.safety["navigation_simulation"].pop("max_duration_sec")
    elif fault == "bad_type":
        body.provider_interfaces["ros_capability_bindings"]["navigation.navigate_to_pose"][
            "action_type"
        ] = "std_msgs/msg/String"
    elif fault == "relative_endpoint":
        body.provider_interfaces["ros_capability_bindings"]["safety.simulation_lease"]["name"] = (
            "lease"
        )
    body.effective_body_hash = body.compute_hash()
    if fault == "tampered":
        body.frames["map"] = "tampered"
    with pytest.raises(ValueError):
        SimulationNavigationBinding.from_body(body)


def test_binding_copies_compiled_body_and_remains_immutable():
    body = compiled_body()
    assert "navigation.navigate_to_pose" in body.capabilities["degraded"]
    binding = SimulationNavigationBinding.from_body(body)
    body.provider_interfaces["ros_capability_bindings"]["navigation.navigate_to_pose"]["name"] = (
        "/mutated"
    )
    assert binding.action_name == "/robot/go"
    with pytest.raises(FrozenInstanceError):
        binding.action_name = "/mutated"


def test_transient_witness_violation_is_latched_until_action_ends(tmp_path):
    _, _, fake, _, _ = make_executor(tmp_path)
    witness = SimulationWitness.__new__(SimulationWitness)
    witness.navigation_binding = fake.navigation_binding
    witness.latest, witness.samples = None, []
    witness.lock = threading.Lock()
    witness.tracking, witness.action_fault = True, None
    bad = fake.fresh()
    bad["linear_speed_mps"] = 0.3
    witness._record(bad)
    witness._record(fake.fresh())
    with pytest.raises(RuntimeError, match="excessive speed"):
        witness.fresh()


def test_navigation_stopping_state_is_not_cleared_by_a_new_action(tmp_path):
    executor, action, _, control, client = make_executor(tmp_path)
    executor.stopping.set()
    result = executor(action)
    assert result.final_state is ActionState.BLOCKED
    assert executor.stopping.is_set() and not control.calls and not client.sent


@pytest.mark.parametrize("stop_observed", [True, False])
def test_real_runtime_stop_aggregation_preserves_sim_witness_evidence(tmp_path, stop_observed):
    from rosclaw.core.runtime import Runtime, RuntimeConfig

    executor, _, witness, control, _ = make_executor(tmp_path)
    witness.lease = True
    control.ignore_stop = not stop_observed
    runtime = Runtime(
        RuntimeConfig(
            enable_event_persistence=False,
            enable_tracing=False,
            enable_firewall=False,
            enable_memory=False,
            enable_practice=False,
            enable_skill_manager=False,
            enable_knowledge=False,
            enable_how=False,
            enable_auto=False,
            enable_provider=False,
            enable_sense=False,
        )
    )
    runtime.register_driver("navigation_sim", executor)
    receipt = runtime.request_emergency_stop("synthetic contract", timeout_sec=3)
    assert receipt.driver_acknowledged is True
    assert receipt.physical_stop_observed is stop_observed
    assert receipt.stopped is stop_observed
    assert receipt.execution_mode == "SIMULATION"
    assert receipt.verification_source == (
        "gazebo_ground_truth_and_revoked_sim_lease" if stop_observed else None
    )


def test_emergency_revoke_follows_inflight_renewal_and_refuses_later_enable(tmp_path):
    executor, _, witness, control, _ = make_executor(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = control.call_service
    errors = []

    def delayed(name, args, **kwargs):
        if args["data"]:
            entered.set()
            assert release.wait(2)
        return original(name, args, **kwargs)

    control.call_service = delayed

    def renew():
        try:
            executor._renew_service(executor.lease_service, timeout_sec=0.5)
        except BaseException as exc:
            errors.append(exc)

    renewal = threading.Thread(target=renew)
    stopping = threading.Thread(target=executor.emergency_stop)
    renewal.start()
    assert entered.wait(2)
    stopping.start()
    assert executor.stopping.wait(2)
    release.set()
    renewal.join(3)
    stopping.join(3)
    assert not renewal.is_alive() and not stopping.is_alive() and not errors
    assert [args["data"] for _, args in control.calls] == [True, False]
    assert not witness.lease
    with pytest.raises(RuntimeError, match="stopping"):
        executor._renew_service(executor.lease_service, timeout_sec=0.5)
    with pytest.raises(RuntimeError, match="stopped before goal"):
        executor._send_navigation_goal()


@pytest.mark.parametrize("fault", ["unknown_kind", "hash", "grid", "tampered_body", "long_socket"])
def test_actual_daemon_refuses_invalid_navigation_configuration_before_runtime(tmp_path, fault):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    body = compiled_body()
    config = {
        "executor_kind": "navigation_only",
        "body_id": body.body_instance_id,
        "body_snapshot_hash": body.effective_body_hash,
    }
    if fault == "unknown_kind":
        config["executor_kind"] = "unknown"
    elif fault == "hash":
        config["body_snapshot_hash"] = "wrong"
    elif fault == "grid":
        config["grid"] = {}
    elif fault == "tampered_body":
        body.frames["map"] = "tampered"
    elif fault == "long_socket":
        tmp_path = tmp_path / ("long-navigation-evidence-directory-" * 4)
        tmp_path.mkdir()
    folder = tmp_path / "home/body/refs"
    folder.mkdir(parents=True)
    (folder / "effective_body.json").write_text(json.dumps(body.to_dict()))
    (folder.parent / "body.yaml").write_text("body_instance:\n  id: mobile\n")
    (tmp_path / "execution_config.json").write_text(json.dumps(config))
    repository = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable,
            str(repository / "integrations/ros_probe/acceptance/daemon.py"),
            "--directory",
            str(tmp_path),
        ],
        cwd=repository,
        env=dict(
            os.environ,
            PYTHONPATH=str(repository / "src"),
            ROSCLAW_HOME=str(tmp_path / "private-home"),
        ),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 1 and "ValueError" in result.stderr
    expected = {
        "unknown_kind": "unsupported simulation executor kind",
        "hash": "configuration Body hash mismatch",
        "grid": "must not contain coverage configuration",
        "tampered_body": "compiled Body hash mismatch",
        "long_socket": "shorter isolated directory",
    }[fault]
    assert expected in result.stderr
    assert (
        not (tmp_path / "memory.sqlite").exists() and not (tmp_path / "run/rosclawd.sock").exists()
    )


@pytest.mark.parametrize(
    "fault",
    [
        "no_ledger",
        "missing_policy",
        "extra_field",
        "wrong_schema",
        "duplicate_actions",
        "too_many_actions",
    ],
)
def test_governed_role_cannot_start_runtime_without_closed_operator_policy(fault):
    import json
    import os
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    body = compiled_body()
    with tempfile.TemporaryDirectory(prefix="reh-governed-contract-") as directory:
        root = Path(directory)
        refs = root / "home/body/refs"
        refs.mkdir(parents=True)
        (refs / "effective_body.json").write_text(json.dumps(body.to_dict()))
        (refs.parent / "body.yaml").write_text("body_instance:\n  id: mobile\n")
        config = {
            "schema_version": "rosclaw.governed_navigation_fixture.v1",
            "executor_kind": "governed_navigation",
            "body_id": body.body_instance_id,
            "body_snapshot_hash": body.effective_body_hash,
            "supervision": {
                "schema_version": "rosclaw.governed_navigation_supervision.v1",
                "facts_enrollment_sha256": "a" * 64,
                "P03_review_sha256": "b" * 64,
                "allowed_action_ids": ["position", "yaw"],
            },
        }
        if fault == "missing_policy":
            config.pop("supervision")
        elif fault == "extra_field":
            config["approved"] = True
        elif fault == "wrong_schema":
            config["supervision"]["schema_version"] = "unknown"
        elif fault == "duplicate_actions":
            config["supervision"]["allowed_action_ids"] = ["position", "position"]
        elif fault == "too_many_actions":
            config["supervision"]["allowed_action_ids"] = ["position", "yaw", "extra"]
        (root / "execution_config.json").write_text(json.dumps(config))
        repository = Path(__file__).resolve().parents[3]
        argv = [
            sys.executable,
            str(repository / "integrations/ros_probe/acceptance/daemon.py"),
            "--directory",
            str(root),
        ]
        if fault != "no_ledger":
            argv.append("--persistent-ledger")
        result = subprocess.run(
            argv,
            cwd=repository,
            env={
                **os.environ,
                "PYTHONPATH": str(repository / "src"),
                "ROSCLAW_HOME": str(root / "home"),
            },
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 1 and "ValueError" in result.stderr
        assert "Initializing ROSClaw Runtime" not in result.stderr
        assert not (root / "memory.sqlite").exists() and not (root / "run/rosclawd.sock").exists()
