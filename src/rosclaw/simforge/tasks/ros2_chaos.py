"""ROS 2 fault model and truthful acknowledgement/evidence classification."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import StrEnum

from rosclaw.kernel import AcknowledgementStage, EvidenceLevel


class Ros2Fault(StrEnum):
    NONE = "none"
    BRIDGE_DISCONNECT = "bridge_disconnect"
    COMMAND_DROP = "command_drop"
    ACK_DROP = "ack_drop"
    OBSERVATION_DROP = "observation_drop"
    OBSERVATION_DELAY = "observation_delay"
    STALE_OBSERVATION = "stale_observation"
    MALFORMED_OBSERVATION = "malformed_observation"
    DUPLICATE_COMMAND = "duplicate_command"
    DAEMON_RESTART = "daemon_restart"


@dataclass(frozen=True)
class Ros2ChaosObservation:
    fault: Ros2Fault
    request_accepted: bool
    command_sent: bool
    protocol_acknowledged: bool
    effect_observed: bool
    task_predicate_verified: bool
    observation_fresh: bool = True
    observation_valid: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.fault, Ros2Fault):
            raise ValueError("ROS2Chaos fault must be a Ros2Fault")
        for name in (
            "request_accepted",
            "command_sent",
            "protocol_acknowledged",
            "effect_observed",
            "task_predicate_verified",
            "observation_fresh",
            "observation_valid",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"ROS2Chaos {name} must be boolean")


@dataclass(frozen=True)
class Ros2ChaosVerdict:
    evidence_level: EvidenceLevel
    acknowledgement_stage: AcknowledgementStage
    task_verified: bool
    fail_closed: bool


def classify_ros2_evidence(observation: Ros2ChaosObservation) -> Ros2ChaosVerdict:
    dispatched = observation.request_accepted and observation.command_sent
    acknowledged = dispatched and observation.protocol_acknowledged
    valid_effect = bool(
        dispatched
        and observation.effect_observed
        and observation.observation_fresh
        and observation.observation_valid
    )
    task_verified = valid_effect and observation.task_predicate_verified
    if task_verified:
        evidence = EvidenceLevel.TASK_VERIFIED
        stage = AcknowledgementStage.TASK_VERIFIED
    elif valid_effect:
        evidence = EvidenceLevel.PHYSICALLY_OBSERVED
        stage = AcknowledgementStage.EFFECT_OBSERVED
    elif acknowledged:
        evidence = EvidenceLevel.DRIVER_CONFIRMED
        stage = AcknowledgementStage.PROTOCOL_ACKNOWLEDGED
    elif dispatched:
        evidence = EvidenceLevel.DISPATCH_CONFIRMED
        stage = AcknowledgementStage.COMMAND_DISPATCHED
    else:
        evidence = EvidenceLevel.REQUESTED
        stage = AcknowledgementStage.REQUEST_ACCEPTED
    consistent = bool(
        (not observation.command_sent or observation.request_accepted)
        and (not observation.protocol_acknowledged or dispatched)
        and (not observation.effect_observed or dispatched)
        and (not observation.task_predicate_verified or valid_effect)
    )
    fail_closed = not task_verified and (observation.fault is not Ros2Fault.NONE or not consistent)
    return Ros2ChaosVerdict(evidence, stage, task_verified, fail_closed)


def generate_ros2_chaos_matrix(*, count: int, seed: int) -> tuple[Ros2ChaosObservation, ...]:
    if count < 1 or count > 1_000_000:
        raise ValueError("ROS2Chaos case count must be in [1, 1000000]")
    rng = random.Random(seed)
    faults = list(Ros2Fault)
    result = []
    for index in range(count):
        fault = faults[index % len(faults)]
        command_sent = fault not in {Ros2Fault.BRIDGE_DISCONNECT, Ros2Fault.COMMAND_DROP}
        acknowledged = command_sent and fault not in {
            Ros2Fault.ACK_DROP,
            Ros2Fault.DAEMON_RESTART,
        }
        effect = command_sent and fault not in {
            Ros2Fault.OBSERVATION_DROP,
            Ros2Fault.DAEMON_RESTART,
        }
        fresh = fault not in {Ros2Fault.OBSERVATION_DELAY, Ros2Fault.STALE_OBSERVATION}
        valid = fault is not Ros2Fault.MALFORMED_OBSERVATION
        predicate = effect and fresh and valid and rng.random() > 0.05
        result.append(
            Ros2ChaosObservation(
                fault=fault,
                request_accepted=True,
                command_sent=command_sent,
                protocol_acknowledged=acknowledged,
                effect_observed=effect,
                task_predicate_verified=predicate,
                observation_fresh=fresh,
                observation_valid=valid,
            )
        )
    return tuple(result)


__all__ = [
    "Ros2ChaosObservation",
    "Ros2ChaosVerdict",
    "Ros2Fault",
    "classify_ros2_evidence",
    "generate_ros2_chaos_matrix",
]
