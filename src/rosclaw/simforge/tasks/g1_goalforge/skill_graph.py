"""Executable, inspectable G1 penalty-kick Skill Graph."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rosclaw.simforge.tasks.g1_goalforge.concepts import hash_json


@dataclass(frozen=True)
class SkillGraphNode:
    name: str
    preconditions: tuple[str, ...]
    executor: str
    sandbox_policy: tuple[str, ...]
    success_verifier: tuple[str, ...]
    failure_signatures: tuple[str, ...]
    retry_budget: int
    safe_reset_state: str
    evidence_requirements: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.executor:
            raise ValueError("Skill Graph nodes require name and executor")
        if not 0 <= self.retry_budget <= 2:
            raise ValueError("Skill Graph node retry budget must be in [0, 2]")
        required = (
            self.sandbox_policy,
            self.success_verifier,
            self.failure_signatures,
            self.evidence_requirements,
        )
        if any(not values for values in required):
            raise ValueError("Skill Graph nodes require safety, verification, and evidence")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class G1PenaltyKickSkillGraph:
    STATES = (
        "TASK_START",
        "OBSERVE_BALL_AND_GOAL",
        "SELECT_TARGET_ZONE",
        "SELECT_KICK_FOOT",
        "ALIGN_TO_BALL",
        "SHIFT_CENTER_OF_MASS",
        "LIFT_KICK_FOOT",
        "SWING_LEG",
        "BALL_CONTACT",
        "RECOVERY_STEP",
        "STABILIZE",
        "VERIFY_GOAL",
        "TASK_SUCCESS",
    )

    def __init__(self) -> None:
        self.nodes = tuple(_node(name) for name in self.STATES[1:-1])
        self._by_name = {node.name: node for node in self.nodes}
        self.current_state = "TASK_START"
        self._attempts: dict[str, int] = {}

    @property
    def graph_hash(self) -> str:
        return hash_json(
            {
                "schema_version": "rosclaw.g1_goalforge.skill_graph.v1",
                "states": self.STATES,
                "nodes": [node.to_dict() for node in self.nodes],
            }
        )

    def transition(self, next_state: str, *, verified: set[str]) -> None:
        current_index = self.STATES.index(self.current_state)
        if current_index + 1 >= len(self.STATES) or self.STATES[current_index + 1] != next_state:
            raise ValueError(f"invalid GoalForge transition {self.current_state}->{next_state}")
        node = self._by_name.get(next_state)
        if node is not None:
            missing = set(node.preconditions) - verified
            if missing:
                raise PermissionError(
                    f"GoalForge transition {next_state} missing preconditions: {sorted(missing)}"
                )
        self.current_state = next_state

    def record_failure(self, failure_type: str) -> str:
        node = self._by_name.get(self.current_state)
        if node is None or failure_type not in node.failure_signatures:
            return "STANDING_READY"
        attempts = self._attempts.get(node.name, 0) + 1
        self._attempts[node.name] = attempts
        return node.safe_reset_state if attempts <= node.retry_budget else "STOPPED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rosclaw.g1_goalforge.skill_graph.v1",
            "graph_hash": self.graph_hash,
            "current_state": self.current_state,
            "states": list(self.STATES),
            "nodes": [node.to_dict() for node in self.nodes],
        }


@dataclass(frozen=True)
class SkillGraphCandidate:
    """Bounded graph patch that can only add re-observation before a kick."""

    parent_graph_hash: str
    insert_micro_step: bool
    insert_reobserve: bool
    retry_budget: int = 1
    schema_version: str = "rosclaw.g1_goalforge.skill_graph_candidate.v1"

    def __post_init__(self) -> None:
        if not self.parent_graph_hash.startswith("sha256:"):
            raise ValueError("Skill Graph Candidate requires a parent graph hash")
        if not (self.insert_micro_step and self.insert_reobserve):
            raise ValueError("GoalForge graph patch must pair MICRO_STEP with REOBSERVE")
        if self.retry_budget != 1:
            raise ValueError("GoalForge graph patch permits exactly one bounded retry")

    @property
    def candidate_hash(self) -> str:
        return hash_json(asdict(self))

    @property
    def patched_route(self) -> tuple[str, ...]:
        return (
            "ALIGN_TO_BALL",
            "MICRO_STEP",
            "REOBSERVE",
            "SHIFT_CENTER_OF_MASS",
            "LIFT_KICK_FOOT",
            "SWING_LEG",
        )


def _node(name: str) -> SkillGraphNode:
    common_evidence = (
        "action_envelope",
        "body_hash",
        "joint_imu_timeline",
        "simulation_receipt",
    )
    if name == "SWING_LEG":
        return SkillGraphNode(
            name=name,
            preconditions=(
                "support_foot_contact_verified",
                "com_inside_support_margin",
                "ball_observation_fresh",
                "kick_trigger_authorized",
            ),
            executor="g1.kick_prior.execute",
            sandbox_policy=(
                "joint_limit_margin_min=0.03",
                "support_foot_slip_max_m=0.08",
                "torso_roll_max_rad=0.35",
                "torso_pitch_max_rad=0.40",
                "peak_joint_torque_scale_max=0.95",
            ),
            success_verifier=("ball_contact_detected", "kick_foot_in_recovery_region"),
            failure_signatures=(
                "BALL_NOT_CONTACTED",
                "SUPPORT_FOOT_SLIP",
                "TORQUE_LIMIT_EXCEEDED",
                "BALL_OBSERVATION_STALE",
            ),
            retry_budget=2,
            safe_reset_state="STANDING_READY",
            evidence_requirements=common_evidence + ("foot_ball_contact", "support_polygon_margin"),
        )
    return SkillGraphNode(
        name=name,
        preconditions=("session_live", "permit_valid", "observation_fresh"),
        executor=f"g1.goalforge.{name.lower()}",
        sandbox_policy=("lease_live", "hard_limits_immutable"),
        success_verifier=(f"{name.lower()}_verified",),
        failure_signatures=("STATE_FEEDBACK_STALE", "EXECUTOR_UNAVAILABLE"),
        retry_budget=1,
        safe_reset_state="STANDING_READY",
        evidence_requirements=common_evidence,
    )


__all__ = ["G1PenaltyKickSkillGraph", "SkillGraphCandidate", "SkillGraphNode"]
