from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from rosclaw.body.schema import EffectiveBody
from rosclaw.core.runtime import Runtime, RuntimeConfig
from rosclaw.daemon.client import DaemonClient
from rosclaw.daemon.server import RosclawDaemon
from rosclaw.daemon.service import DaemonControlPlane
from rosclaw.kernel import (
    AcknowledgementStage,
    ActionEnvelope,
    ActionGateway,
    ActionState,
    EvidenceLevel,
    ExecutionMode,
)
from rosclaw.mcp.adapters.runtime_client import RuntimeClient
from rosclaw.simforge.tasks.body_mutation import (
    apply_and_validate_mutation,
    generate_body_mutations,
)
from rosclaw.simforge.tasks.contact_push import run_contact_push
from rosclaw.simforge.tasks.guarded_base import (
    GenericMobileBaseSimulationExecutor,
    MobileBaseObservation,
    RosbridgeMobileBaseSink,
)
from rosclaw.simforge.tasks.ros2_chaos import (
    Ros2ChaosObservation,
    Ros2Fault,
    classify_ros2_evidence,
    generate_ros2_chaos_matrix,
)


@dataclass
class _DaemonSink:
    daemon_owner_id: str
    observation: MobileBaseObservation | None
    commands: int = 0
    stops: int = 0
    raise_observation: bool = False
    fresh_observation: bool = True

    def publish_velocity(
        self,
        linear_x_mps: float,
        angular_z_radps: float,
        duration_sec: float,
    ) -> bool:
        self.commands += 1
        return linear_x_mps == 0.2 and angular_z_radps == 0.0 and duration_sec == 1.0

    def observe_effect(self, timeout_sec: float) -> MobileBaseObservation | None:
        assert timeout_sec > 0
        if self.raise_observation:
            raise RuntimeError("injected observation failure")
        if self.observation is None:
            return None
        return MobileBaseObservation(
            start_x_m=self.observation.start_x_m,
            final_x_m=self.observation.final_x_m,
            final_velocity_mps=self.observation.final_velocity_mps,
            timestamp_monotonic=(
                time.monotonic() if self.fresh_observation else self.observation.timestamp_monotonic
            ),
        )

    def stop(self) -> bool:
        self.stops += 1
        return True


def _guarded_action(action_id: str) -> ActionEnvelope:
    return ActionEnvelope(
        action_id=action_id,
        actor_id="rosclaw-mcp",
        agent_framework="mcp",
        session_id="guarded-base-session",
        body_id="sim_mobile_base",
        body_snapshot_hash="sha256:" + "a" * 64,
        capability_id="mobile_base.guarded_move",
        arguments={"linear_x_mps": 0.2, "angular_z_radps": 0.0, "duration_sec": 1.0},
        execution_mode=ExecutionMode.SHADOW,
    )


def test_guarded_base_executor_is_gateway_owned_and_requires_observation() -> None:
    sink = _DaemonSink(
        daemon_owner_id="daemon_test",
        observation=MobileBaseObservation(0.0, 0.2, 0.0, 1.0),
    )
    gateway = ActionGateway()
    gateway.register_executor(
        "mobile_base.guarded_move",
        ExecutionMode.SHADOW,
        GenericMobileBaseSimulationExecutor(sink, daemon_instance_id="daemon_test"),
    )
    receipt = gateway.submit(_guarded_action("guarded-observed"))

    assert receipt.final_state is ActionState.COMPLETED
    assert receipt.evidence_level is EvidenceLevel.TASK_VERIFIED
    assert receipt.acknowledgement_stage is AcknowledgementStage.TASK_VERIFIED
    assert receipt.dispatch_result["owner"] == "daemon_test"
    assert sink.commands == sink.stops == 1

    lost_sink = _DaemonSink(daemon_owner_id="daemon_test", observation=None)
    lost_gateway = ActionGateway()
    lost_gateway.register_executor(
        "mobile_base.guarded_move",
        ExecutionMode.SHADOW,
        GenericMobileBaseSimulationExecutor(lost_sink, daemon_instance_id="daemon_test"),
    )
    lost = lost_gateway.submit(_guarded_action("guarded-observation-lost"))
    assert lost.final_state is ActionState.DEGRADED
    assert lost.evidence_level is EvidenceLevel.DISPATCH_CONFIRMED
    assert lost.acknowledgement_stage is AcknowledgementStage.COMMAND_DISPATCHED
    assert lost.verification_result["success"] is False
    assert lost.verification_result["stop_confirmed"] is True

    failing_sink = _DaemonSink(
        daemon_owner_id="daemon_test",
        observation=None,
        raise_observation=True,
    )
    failing = GenericMobileBaseSimulationExecutor(
        failing_sink,
        daemon_instance_id="daemon_test",
    )(_guarded_action("guarded-observation-exception"))
    assert failing.final_state is ActionState.DEGRADED
    assert failing.errors[0]["code"] == "OBSERVATION_FAILED"
    assert failing_sink.commands == failing_sink.stops == 1

    stale_sink = _DaemonSink(
        daemon_owner_id="daemon_test",
        observation=MobileBaseObservation(0.0, 0.2, 0.0, 0.0),
        fresh_observation=False,
    )
    stale = GenericMobileBaseSimulationExecutor(
        stale_sink,
        daemon_instance_id="daemon_test",
    )(_guarded_action("guarded-stale-observation"))
    assert stale.evidence_level is EvidenceLevel.DISPATCH_CONFIRMED
    assert stale.errors[0]["code"] == "STALE_OBSERVATION"

    angular_action = _guarded_action("guarded-angular-unverified")
    angular_action.arguments["angular_z_radps"] = 0.2
    angular = GenericMobileBaseSimulationExecutor(
        sink,
        daemon_instance_id="daemon_test",
    )(angular_action)
    assert angular.final_state is ActionState.BLOCKED
    assert angular.errors[0]["code"] == "ANGULAR_MOTION_UNVERIFIED"

    with pytest.raises(ValueError, match="finite and positive"):
        GenericMobileBaseSimulationExecutor(
            sink,
            daemon_instance_id="daemon_test",
            max_linear_speed_mps=float("nan"),
        )

    with pytest.raises(ValueError, match="simulation sink"):
        RosbridgeMobileBaseSink(
            object(),  # type: ignore[arg-type]
            daemon_owner_id="daemon_test",
            command_topic="/cmd_vel",
            pose_topic="/odom",
        )


async def test_guarded_base_canonical_mcp_daemon_gateway_executor_receipt_chain(
    tmp_path: Path,
) -> None:
    sink = _DaemonSink(
        daemon_owner_id="daemon_simforge",
        observation=MobileBaseObservation(0.0, 0.2, 0.0, 1.0),
    )
    runtime = Runtime(
        RuntimeConfig(
            robot_id="sim_mobile_base",
            enable_firewall=False,
            enable_memory=False,
            enable_practice=False,
            enable_skill_manager=False,
            enable_knowledge=False,
            enable_how=False,
            enable_auto=False,
            enable_provider=False,
            enable_sense=False,
            enable_event_persistence=False,
            enable_tracing=False,
        )
    )
    runtime.action_gateway.register_executor(
        "mobile_base.guarded_move",
        ExecutionMode.SHADOW,
        GenericMobileBaseSimulationExecutor(sink, daemon_instance_id="daemon_simforge"),
    )
    service = DaemonControlPlane(runtime=runtime)
    daemon = RosclawDaemon(service=service, socket_path=tmp_path / "rosclawd.sock")
    daemon.start()
    try:
        mcp = RuntimeClient(
            project_root=tmp_path,
            robot_id="sim_mobile_base",
            runtime_profile={},
            daemon_client=DaemonClient(socket_path=daemon.socket_path, timeout_sec=5),
        )
        result = await mcp.request_action(
            capability_id="mobile_base.guarded_move",
            arguments={"linear_x_mps": 0.2, "angular_z_radps": 0.0, "duration_sec": 1.0},
            execution_mode="SHADOW",
            body_snapshot_hash="sha256:" + "a" * 64,
            body_id="sim_mobile_base",
            action_id="mcp-daemon-guarded-base",
            required_evidence="TASK_VERIFIED",
            wait_timeout_sec=5,
        )
    finally:
        daemon.stop()

    receipt = result["receipt"]
    assert result["state"] == "FINISHED"
    assert receipt["final_state"] == "COMPLETED"
    assert receipt["evidence_level"] == "TASK_VERIFIED"
    assert receipt["dispatch_result"]["owner"] == "daemon_simforge"
    assert receipt["evidence_domain"] == "SHADOW"


def test_contact_push_observes_real_contact_force_and_displacement() -> None:
    evidence = run_contact_push()
    assert evidence.physics_executed
    assert evidence.contact_observed
    assert evidence.peak_contact_force_n > 0
    assert evidence.object_displacement_m >= 0.08
    assert evidence.success


def test_ros2_chaos_never_upgrades_dispatch_without_observation() -> None:
    explicit = classify_ros2_evidence(
        Ros2ChaosObservation(
            fault=Ros2Fault.OBSERVATION_DROP,
            request_accepted=True,
            command_sent=True,
            protocol_acknowledged=False,
            effect_observed=False,
            task_predicate_verified=False,
        )
    )
    assert explicit.evidence_level is EvidenceLevel.DISPATCH_CONFIRMED
    assert explicit.acknowledgement_stage is AcknowledgementStage.COMMAND_DISPATCHED
    assert not explicit.task_verified

    inconsistent = classify_ros2_evidence(
        Ros2ChaosObservation(
            fault=Ros2Fault.NONE,
            request_accepted=False,
            command_sent=False,
            protocol_acknowledged=True,
            effect_observed=True,
            task_predicate_verified=True,
        )
    )
    assert inconsistent.evidence_level is EvidenceLevel.REQUESTED
    assert inconsistent.fail_closed

    matrix = generate_ros2_chaos_matrix(count=1000, seed=20260723)
    assert len(matrix) == 1000
    for observation in matrix:
        verdict = classify_ros2_evidence(observation)
        if not (
            observation.effect_observed
            and observation.observation_fresh
            and observation.observation_valid
            and observation.task_predicate_verified
        ):
            assert verdict.evidence_level is not EvidenceLevel.TASK_VERIFIED
    duplicate = next(item for item in matrix if item.fault is Ros2Fault.DUPLICATE_COMMAND)
    duplicate_verdict = classify_ros2_evidence(duplicate)
    assert not duplicate_verdict.task_verified
    assert duplicate_verdict.fail_closed


def test_body_mutation_1000_preserves_or_rejects_every_invariant() -> None:
    base = EffectiveBody(
        body_instance_id="mutation-base",
        eurdf_uri="rosclaw://eurdf/mutation@1.0.0",
        effective_body_hash="sha256:" + "b" * 64,
        compiled_at="2026-07-23T00:00:00Z",
        joints={"joint_1": {"lower": -1.0, "upper": 1.0, "velocity": 2.0, "effort": 10.0}},
        actuators={"motor_1": {"status": "available"}},
        capabilities={"enabled": ["joint.move"], "degraded": [], "blocked": []},
        runtime_state={"calibration": {"joint_offsets": {"joint_1": 0.0}}},
    )
    mutations = generate_body_mutations(count=1000, seed=20260723)
    results = [apply_and_validate_mutation(base, mutation) for mutation in mutations]

    assert all(result.accepted is result.mutation.expected_valid for result in results)
    assert all(
        result.effective_config_hash is not None if result.accepted else result.invariant_errors
        for result in results
    )
