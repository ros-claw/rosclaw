"""Memory/Know/How/Auto components for ContactPush Phase 3."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from rosclaw.know.task_card import TaskCard
from rosclaw.memory.seekdb_client import InMemoryKnowledgeStore
from rosclaw.memory.v2.models import MemoryItem, MemoryStatus, MemoryType
from rosclaw.memory.v2.repository import MemoryRepository
from rosclaw.memory.v2.retrieval import MemoryQuery, MemoryRetriever
from rosclaw.simforge.candidates import (
    CandidateCompiler,
    CandidateGenerator,
    CandidatePatch,
    ParameterBound,
)
from rosclaw.simforge.dataset_snapshot import PracticeEpisodeRecord
from rosclaw.simforge.failure_router_v2 import FailureSignatureV2
from rosclaw.simforge.models import HumanInvolvement
from rosclaw.simforge.tasks.contact_push_v3 import (
    CONTACT_PUSH_BODY_HASH,
    CONTACT_PUSH_TASK_ID,
    ContactPushEpisodeEvidence,
    ContactPushPhysics,
    ContactPushPolicy,
    ContactPushResult,
    ContactPushScenario,
    ContactPushStatus,
)


class ContactPushCandidateType(StrEnum):
    PARAMETER = "parameter"
    TRAJECTORY = "trajectory"
    SKILL_GRAPH = "skill_graph"
    LEARNED_POLICY = "learned_policy"


@dataclass(frozen=True)
class ContactPushTaskKnowledge:
    """TaskCard plus machine-enforced candidate and safety constraints."""

    task_card: TaskCard
    maximum_force_n: float = 30.0
    maximum_duration_sec: float = 1.5
    minimum_duration_sec: float = 0.25
    deadline_sec: float = 2.5

    @classmethod
    def default(cls) -> ContactPushTaskKnowledge:
        return cls(
            task_card=TaskCard(
                task_id=CONTACT_PUSH_TASK_ID,
                task_family="contact_manipulation",
                domain="simulation",
                embodiment_type="one_axis_pusher",
                objective_direction="maximize",
                metric_name="task_success_with_force_guard",
                prerequisites=["fresh_object_pose", "body_hash_match", "force_verifier"],
                common_failures=[
                    {
                        "failure_type": "OBJECT_OVERSHOT",
                        "causes": ["low_surface_friction", "push_velocity_too_high"],
                    },
                    {
                        "failure_type": "FORCE_LIMIT_EXCEEDED",
                        "causes": ["push_velocity_too_high", "object_mass_high"],
                    },
                ],
                verified_patterns=[
                    "decelerate_before_target",
                    "bounded_micro_push_only_for_undershoot",
                    "never_modify_force_limit_from_candidate",
                ],
                source_manifest={
                    "schema_version": "rosclaw.contact_push_task_card.v1",
                    "source": "physics_and_verified_practice",
                },
            )
        )

    @property
    def card_hash(self) -> str:
        return _hash_json(
            {
                "task_card": self.task_card.to_dict(),
                "constraints": {
                    "maximum_force_n": self.maximum_force_n,
                    "maximum_duration_sec": self.maximum_duration_sec,
                    "minimum_duration_sec": self.minimum_duration_sec,
                    "deadline_sec": self.deadline_sec,
                    "immutable": ["force_limit_n", "deadline_sec", "evidence_domain"],
                },
            }
        )

    def compiler(self, parent: ContactPushPolicy) -> CandidateCompiler:
        return CandidateCompiler(
            parent_policy=_policy_pointer_dict(parent),
            allowed_bounds={
                "/controller/push_velocity_mps": ParameterBound(minimum=0.08, maximum=0.5),
                "/controller/contact_duration_sec": ParameterBound(
                    minimum=self.minimum_duration_sec,
                    maximum=self.maximum_duration_sec,
                ),
                "/controller/contact_offset_y_m": ParameterBound(
                    minimum=-0.035,
                    maximum=0.035,
                ),
                "/trajectory/deceleration_fraction": ParameterBound(
                    minimum=0.45,
                    maximum=1.0,
                ),
                "/skill_graph/micro_push": ParameterBound(choices=(False, True)),
            },
        )

    def validate_policy(self, policy: ContactPushPolicy) -> tuple[bool, tuple[str, ...]]:
        errors = []
        if policy.contact_duration_sec > self.maximum_duration_sec:
            errors.append("duration_exceeds_task_card")
        if policy.contact_duration_sec < self.minimum_duration_sec:
            errors.append("duration_below_task_card")
        if policy.push_velocity_mps > 0.5:
            errors.append("velocity_exceeds_body_limit")
        if abs(policy.contact_offset_y_m) > 0.035:
            errors.append("contact_offset_exceeds_geometry")
        return not errors, tuple(errors)

    def filter_proposals(
        self,
        proposals: tuple[dict[str, Any], ...],
    ) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
        accepted = []
        rejected = []
        compiler = self.compiler(ContactPushPolicy.baseline())
        immutable = {"/safety/force_limit_n", "/runtime/deadline_sec", "/evidence/domain"}
        for proposal in proposals:
            if set(proposal) & immutable:
                rejected.append({"proposal": proposal, "reason": "immutable_safety_path"})
                continue
            try:
                compiler.compile(
                    proposal,
                    failure_signature_id="KNOW_FILTER:contact_push",
                    generator=CandidateGenerator(type="know_filter", algorithm="task_card"),
                )
            except (TypeError, ValueError) as exc:
                rejected.append({"proposal": proposal, "reason": str(exc)})
            else:
                accepted.append(proposal)
        return tuple(accepted), tuple(rejected)


@dataclass(frozen=True)
class MemorySuggestion:
    memory_id: str
    policy: ContactPushPolicy
    source_features: tuple[tuple[str, float], ...]
    feature_distance: float
    retrieval_score: float
    evidence_refs: tuple[str, ...]


class ContactPushMemory:
    """Use ROSClaw Memory v2 for body-scoped, evidence-backed warm starts."""

    def __init__(self) -> None:
        client = InMemoryKnowledgeStore()
        client.connect()
        self.repository = MemoryRepository(client)
        self.retriever = MemoryRetriever(self.repository)

    def ingest_recovery(
        self,
        evidence: ContactPushEpisodeEvidence,
        *,
        body_hash: str = CONTACT_PUSH_BODY_HASH,
        status: MemoryStatus = MemoryStatus.ACTIVE,
    ) -> str:
        if not evidence.result.success:
            raise ValueError("only verified successful recoveries can warm-start ContactPush")
        features = evidence.scenario.feature_dict()
        policy = evidence.policy.to_dict()
        item = MemoryItem(
            memory_type=MemoryType.INTERVENTION.value,
            robot_id="sim_contact_pusher",
            body_id=body_hash,
            practice_id=evidence.practice_id,
            episode_id=evidence.episode_id,
            task_id=CONTACT_PUSH_TASK_ID,
            skill_id="contact_push",
            policy_id=evidence.policy.policy_hash,
            failure_type=ContactPushStatus.OVERSHOOT.value,
            title=(
                "ContactPush low-friction overshoot recovery "
                f"{evidence.scenario.scenario_commitment.removeprefix('sha256:')[:12]}"
            ),
            document=(
                "OBJECT_OVERSHOT low surface friction recovery using bounded velocity, "
                "deceleration, and optional micro push; "
                f"scenario={evidence.scenario.scenario_commitment}; "
                f"policy={evidence.policy.policy_hash}"
            ),
            summary="Verified same-physics recovery policy",
            outcome="success",
            confidence=1.0,
            quality_score=1.0,
            evidence_refs=[
                f"episode://{evidence.episode_id}",
                f"receipt://{evidence.receipt_hash.removeprefix('sha256:')}",
            ],
            artifact_refs=[
                f"artifact://{evidence.request_hash.removeprefix('sha256:')}",
                f"artifact://{evidence.state_hash.removeprefix('sha256:')}",
            ],
            tags=["contact_push", "low_friction", "overshoot", "strict_replay"],
            metadata={
                "features": features,
                "policy": policy,
                "strict_replay": evidence.strict_replay,
                "body_snapshot_hash": body_hash,
            },
            status=status.value,
        )
        return self.repository.store(item)

    def retrieve(
        self,
        scenario: ContactPushScenario,
        *,
        body_hash: str = CONTACT_PUSH_BODY_HASH,
    ) -> MemorySuggestion | None:
        results = self.retriever.retrieve(
            MemoryQuery(
                text="OBJECT_OVERSHOT low friction contact push recovery",
                memory_types=[MemoryType.INTERVENTION.value],
                body_id=body_hash,
                task_id=CONTACT_PUSH_TASK_ID,
                outcome="success",
                minimum_confidence=0.8,
                limit=100,
            )
        )
        target = scenario.feature_dict()
        ranked = []
        for result in results:
            item = result.memory
            if item.failure_type != ContactPushStatus.OVERSHOOT.value:
                continue
            metadata = item.metadata
            features = metadata.get("features")
            policy = metadata.get("policy")
            if not isinstance(features, dict) or not isinstance(policy, dict):
                continue
            if metadata.get("strict_replay") is not True:
                continue
            try:
                source_features = {str(key): float(value) for key, value in features.items()}
                parsed_policy = _policy_from_dict(policy, policy_type="parameter")
                distance = _feature_distance(target, source_features)
            except (KeyError, TypeError, ValueError):
                continue
            ranked.append(
                MemorySuggestion(
                    memory_id=item.memory_id,
                    policy=parsed_policy,
                    source_features=tuple(sorted(source_features.items())),
                    feature_distance=distance,
                    retrieval_score=result.fusion_score,
                    evidence_refs=tuple(item.evidence_refs + item.artifact_refs),
                )
            )
        if not ranked:
            return None
        return min(ranked, key=lambda item: (item.feature_distance, -item.retrieval_score))


@dataclass(frozen=True)
class ExecutableRecoveryHint:
    failure_id: str
    memory_id: str
    task_card_hash: str
    changes: tuple[tuple[str, float | bool], ...]
    confidence: float
    evidence_refs: tuple[str, ...]
    schema_version: str = "rosclaw.executable_recovery_hint.v1"

    def __post_init__(self) -> None:
        if not self.failure_id.startswith("failure_") or not self.memory_id:
            raise ValueError("executable recovery hint identity is invalid")
        if not 0 <= self.confidence <= 1:
            raise ValueError("recovery confidence must be in [0, 1]")
        if not self.changes or not self.evidence_refs:
            raise ValueError("executable recovery hint requires changes and evidence")

    def compile(
        self,
        *,
        knowledge: ContactPushTaskKnowledge,
        parent: ContactPushPolicy,
    ) -> tuple[CandidatePatch, ContactPushPolicy]:
        compiler = knowledge.compiler(parent)
        patch = compiler.compile(
            dict(self.changes),
            failure_signature_id=self.failure_id,
            generator=CandidateGenerator(type="how", algorithm="memory_context_adaptation"),
            human_involvement=HumanInvolvement(),
        )
        return patch, apply_candidate_patch(parent, patch, policy_type="parameter")


class ContactPushHow:
    def generate(
        self,
        *,
        failure: FailureSignatureV2,
        scenario: ContactPushScenario,
        suggestion: MemorySuggestion,
        task_card_hash: str,
    ) -> ExecutableRecoveryHint:
        source = dict(suggestion.source_features)
        target = scenario.feature_dict()
        target_ratio = target["target_distance_m"] / max(source["target_distance_m"], 1e-9)
        friction_delta = target["estimated_friction"] - source["estimated_friction"]
        mass_ratio = target["object_mass_kg"] / max(source["object_mass_kg"], 1e-9)
        velocity = suggestion.policy.push_velocity_mps
        velocity *= min(1.20, max(0.80, target_ratio**0.72))
        velocity *= min(1.08, max(0.92, mass_ratio**0.08))
        velocity *= min(1.06, max(0.94, 1.0 + friction_delta * 0.08))
        duration = 0.72 + 1.30 * target["target_distance_m"]
        deceleration = suggestion.policy.deceleration_fraction + friction_delta * 0.12
        changes: dict[str, float | bool] = {
            "/controller/push_velocity_mps": min(0.48, max(0.10, velocity)),
            "/controller/contact_duration_sec": min(1.25, max(0.75, duration)),
            "/controller/contact_offset_y_m": min(
                0.035,
                max(-0.035, target["initial_offset_y_m"]),
            ),
            "/trajectory/deceleration_fraction": min(
                0.85,
                max(0.60, deceleration),
            ),
            "/skill_graph/micro_push": True,
        }
        return ExecutableRecoveryHint(
            failure_id=failure.failure_id,
            memory_id=suggestion.memory_id,
            task_card_hash=task_card_hash,
            changes=tuple(sorted(changes.items())),
            confidence=max(0.5, 1.0 - suggestion.feature_distance),
            evidence_refs=suggestion.evidence_refs,
        )


@dataclass(frozen=True)
class ExpertSearchResult:
    policy: ContactPushPolicy
    outcome: ContactPushResult
    attempts: int
    evaluated_policy_hashes: tuple[str, ...]


class ContactPushExpertSearch:
    """Bounded monotonic search used to label Practice training rows."""

    def __init__(self, physics: ContactPushPhysics) -> None:
        self.physics = physics

    def optimize(self, scenario: ContactPushScenario, *, budget: int = 9) -> ExpertSearchResult:
        if not 2 <= budget <= 20:
            raise ValueError("expert search budget must be in [2, 20]")
        duration = min(1.25, max(0.75, 0.72 + 1.30 * scenario.target_distance_m))
        deceleration = min(0.82, max(0.62, 0.68 + 0.12 * scenario.observed_friction))
        lower, upper = 0.08, 0.50
        best: tuple[ContactPushPolicy, ContactPushResult] | None = None
        hashes = []
        for attempt in range(1, budget + 1):
            velocity = (lower + upper) / 2.0
            policy = ContactPushPolicy(
                push_velocity_mps=velocity,
                contact_duration_sec=duration,
                contact_offset_y_m=scenario.initial_offset_y_m,
                deceleration_fraction=deceleration,
                micro_push=True,
                policy_type="parameter",
            )
            outcome = self.physics.run(scenario, policy)
            hashes.append(policy.policy_hash)
            if best is None or abs(outcome.final_error_m) < abs(best[1].final_error_m):
                best = (policy, outcome)
            if outcome.success:
                return ExpertSearchResult(
                    policy=policy,
                    outcome=outcome,
                    attempts=attempt,
                    evaluated_policy_hashes=tuple(hashes),
                )
            if outcome.status is ContactPushStatus.OVERSHOOT:
                upper = velocity
            else:
                lower = velocity
        if best is None:
            raise RuntimeError("expert search produced no policy")
        return ExpertSearchResult(
            policy=best[0],
            outcome=best[1],
            attempts=budget,
            evaluated_policy_hashes=tuple(hashes),
        )


_MODEL_FEATURES = (
    "bias",
    "object_mass_kg",
    "estimated_friction",
    "target_distance_m",
    "initial_offset_y_m",
    "control_delay_sec",
    "target_over_friction",
    "target_times_mass",
    "target_squared",
    "friction_squared",
    "mass_times_friction",
)
_MODEL_OUTPUTS = (
    "push_velocity_mps",
    "contact_duration_sec",
    "contact_offset_y_m",
    "deceleration_fraction",
)


@dataclass(frozen=True)
class ContextualPolicyModel:
    dataset_snapshot_hash: str
    feature_names: tuple[str, ...]
    output_names: tuple[str, ...]
    coefficients: tuple[tuple[float, ...], ...]
    velocity_calibration: float
    training_rows: int
    validation_rows: int
    schema_version: str = "rosclaw.contact_push_contextual_policy.v1"

    def __post_init__(self) -> None:
        if not self.dataset_snapshot_hash.startswith("sha256:"):
            raise ValueError("learned policy must reference a dataset snapshot")
        if self.feature_names != _MODEL_FEATURES or self.output_names != _MODEL_OUTPUTS:
            raise ValueError("contextual policy feature/output contract mismatch")
        if len(self.coefficients) != len(self.feature_names) or any(
            len(row) != len(self.output_names) for row in self.coefficients
        ):
            raise ValueError("contextual policy coefficient dimensions are invalid")
        if self.training_rows < 1 or self.validation_rows < 1:
            raise ValueError("contextual policy requires train and validation rows")
        values = [value for row in self.coefficients for value in row]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("contextual policy coefficients must be finite")
        if not 0.8 <= self.velocity_calibration <= 1.0:
            raise ValueError("velocity calibration must be in [0.8, 1.0]")

    def predict(self, scenario: ContactPushScenario) -> ContactPushPolicy:
        features = np.asarray(_model_features(scenario.feature_dict()), dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        values = features @ coefficients
        return ContactPushPolicy(
            push_velocity_mps=float(np.clip(values[0] * self.velocity_calibration, 0.08, 0.50)),
            contact_duration_sec=float(np.clip(values[1], 0.25, 1.50)),
            contact_offset_y_m=float(np.clip(values[2], -0.035, 0.035)),
            deceleration_fraction=float(np.clip(values[3], 0.45, 1.0)),
            micro_push=True,
            policy_type=ContactPushCandidateType.LEARNED_POLICY.value,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def artifact_hash(self) -> str:
        return _hash_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ContextualPolicyModel:
        return cls(
            dataset_snapshot_hash=str(value["dataset_snapshot_hash"]),
            feature_names=tuple(map(str, value["feature_names"])),
            output_names=tuple(map(str, value["output_names"])),
            coefficients=tuple(tuple(float(item) for item in row) for row in value["coefficients"]),
            velocity_calibration=float(value["velocity_calibration"]),
            training_rows=int(value["training_rows"]),
            validation_rows=int(value["validation_rows"]),
            schema_version=str(
                value.get(
                    "schema_version",
                    "rosclaw.contact_push_contextual_policy.v1",
                )
            ),
        )


class ContextualPolicyTrainer:
    def train(
        self,
        *,
        development: tuple[PracticeEpisodeRecord, ...],
        validation: tuple[PracticeEpisodeRecord, ...],
        dataset_snapshot_hash: str,
        ridge: float = 1e-4,
    ) -> ContextualPolicyModel:
        if not development or not validation:
            raise ValueError("policy training requires development and validation rows")
        if not math.isfinite(ridge) or not 0 < ridge <= 1:
            raise ValueError("ridge must be in (0, 1]")
        rows = development + validation
        if any(record.task_id != CONTACT_PUSH_TASK_ID for record in rows):
            raise ValueError("policy training dataset contains the wrong task")
        if any(
            dict(record.labels).get("success") is not True
            or not record.complete
            or not record.independently_verified
            or not record.strict_replay
            for record in rows
        ):
            raise ValueError("policy training requires successful verified strict-replay rows")
        x = np.asarray(
            [_model_features(dict(record.features)) for record in development],
            dtype=np.float64,
        )
        y = np.asarray(
            [
                [float(dict(record.policy)[name]) for name in _MODEL_OUTPUTS]
                for record in development
            ],
            dtype=np.float64,
        )
        regularizer = ridge * np.eye(x.shape[1], dtype=np.float64)
        coefficients = np.linalg.solve(x.T @ x + regularizer, x.T @ y)
        if not np.isfinite(coefficients).all():
            raise RuntimeError("contextual policy training produced non-finite coefficients")
        return ContextualPolicyModel(
            dataset_snapshot_hash=dataset_snapshot_hash,
            feature_names=_MODEL_FEATURES,
            output_names=_MODEL_OUTPUTS,
            coefficients=tuple(tuple(float(value) for value in row) for row in coefficients),
            velocity_calibration=0.99,
            training_rows=len(development),
            validation_rows=len(validation),
        )


@dataclass(frozen=True)
class ContactPushCandidate:
    candidate_id: str
    candidate_type: ContactPushCandidateType
    parent_policy_hash: str
    failure_signature_id: str
    task_card_hash: str
    dataset_snapshot_hash: str | None
    static_policy: ContactPushPolicy | None
    learned_policy: ContextualPolicyModel | None
    lineage_refs: tuple[str, ...]
    human_involvement: HumanInvolvement = HumanInvolvement()
    schema_version: str = "rosclaw.contact_push_candidate.v1"

    def __post_init__(self) -> None:
        if not self.candidate_id.startswith("candidate_"):
            raise ValueError("contact-push candidate id must start with candidate_")
        if self.candidate_type is ContactPushCandidateType.LEARNED_POLICY:
            if self.learned_policy is None or self.static_policy is not None:
                raise ValueError("learned candidate requires only a learned policy")
            if self.dataset_snapshot_hash != self.learned_policy.dataset_snapshot_hash:
                raise ValueError("learned candidate dataset binding mismatch")
        elif self.static_policy is None or self.learned_policy is not None:
            raise ValueError("non-learned candidates require only a static policy")
        if not self.lineage_refs:
            raise ValueError("candidate lineage cannot be empty")

    def policy_for(self, scenario: ContactPushScenario) -> ContactPushPolicy:
        if self.learned_policy is not None:
            return self.learned_policy.predict(scenario)
        if self.static_policy is None:
            raise RuntimeError("candidate policy is missing")
        return self.static_policy

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "candidate_type": self.candidate_type.value,
            "parent_policy_hash": self.parent_policy_hash,
            "failure_signature_id": self.failure_signature_id,
            "task_card_hash": self.task_card_hash,
            "dataset_snapshot_hash": self.dataset_snapshot_hash,
            "static_policy": self.static_policy.to_dict() if self.static_policy else None,
            "learned_policy": self.learned_policy.to_dict() if self.learned_policy else None,
            "lineage_refs": list(self.lineage_refs),
            "human_involvement": asdict(self.human_involvement),
            "constraints": {
                "candidate_whitelist_only": True,
                "safety_limits_immutable": True,
                "hardware_authority": False,
            },
        }

    @property
    def candidate_hash(self) -> str:
        return _hash_json(self.to_dict())

    @classmethod
    def learned(
        cls,
        *,
        model: ContextualPolicyModel,
        parent: ContactPushPolicy,
        failure_signature_id: str,
        task_card_hash: str,
        lineage_refs: tuple[str, ...],
    ) -> ContactPushCandidate:
        identity = _hash_json(
            {
                "model": model.artifact_hash,
                "failure": failure_signature_id,
                "task_card": task_card_hash,
            }
        )
        return cls(
            candidate_id="candidate_" + identity.removeprefix("sha256:")[:24],
            candidate_type=ContactPushCandidateType.LEARNED_POLICY,
            parent_policy_hash=parent.policy_hash,
            failure_signature_id=failure_signature_id,
            task_card_hash=task_card_hash,
            dataset_snapshot_hash=model.dataset_snapshot_hash,
            static_policy=None,
            learned_policy=model,
            lineage_refs=lineage_refs,
        )

    @classmethod
    def static(
        cls,
        *,
        candidate_type: ContactPushCandidateType,
        policy: ContactPushPolicy,
        parent: ContactPushPolicy,
        failure_signature_id: str,
        task_card_hash: str,
        dataset_snapshot_hash: str | None,
        lineage_refs: tuple[str, ...],
    ) -> ContactPushCandidate:
        if candidate_type is ContactPushCandidateType.LEARNED_POLICY:
            raise ValueError("use ContactPushCandidate.learned for a learned policy")
        expected_policy_type = {
            ContactPushCandidateType.PARAMETER: "parameter",
            ContactPushCandidateType.TRAJECTORY: "trajectory",
            ContactPushCandidateType.SKILL_GRAPH: "skill_graph",
        }[candidate_type]
        if policy.policy_type != expected_policy_type:
            raise ValueError(
                f"{candidate_type.value} candidate requires a {expected_policy_type} policy"
            )
        identity = _hash_json(
            {
                "candidate_type": candidate_type.value,
                "policy": policy.policy_hash,
                "failure": failure_signature_id,
                "task_card": task_card_hash,
                "dataset": dataset_snapshot_hash,
            }
        )
        return cls(
            candidate_id="candidate_" + identity.removeprefix("sha256:")[:24],
            candidate_type=candidate_type,
            parent_policy_hash=parent.policy_hash,
            failure_signature_id=failure_signature_id,
            task_card_hash=task_card_hash,
            dataset_snapshot_hash=dataset_snapshot_hash,
            static_policy=policy,
            learned_policy=None,
            lineage_refs=lineage_refs,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ContactPushCandidate:
        static_value = value.get("static_policy")
        learned_value = value.get("learned_policy")
        involvement = value.get("human_involvement") or {}
        return cls(
            candidate_id=str(value["candidate_id"]),
            candidate_type=ContactPushCandidateType(str(value["candidate_type"])),
            parent_policy_hash=str(value["parent_policy_hash"]),
            failure_signature_id=str(value["failure_signature_id"]),
            task_card_hash=str(value["task_card_hash"]),
            dataset_snapshot_hash=(
                str(value["dataset_snapshot_hash"])
                if value.get("dataset_snapshot_hash") is not None
                else None
            ),
            static_policy=(
                _policy_from_dict(static_value, policy_type=str(static_value["policy_type"]))
                if isinstance(static_value, dict)
                else None
            ),
            learned_policy=(
                ContextualPolicyModel.from_dict(learned_value)
                if isinstance(learned_value, dict)
                else None
            ),
            lineage_refs=tuple(map(str, value.get("lineage_refs") or ())),
            human_involvement=HumanInvolvement(**involvement),
            schema_version=str(value.get("schema_version", "rosclaw.contact_push_candidate.v1")),
        )


@dataclass(frozen=True)
class CausalSearchResult:
    memory_enabled: bool
    attempts: int
    success: bool
    selected_policy: ContactPushPolicy | None
    outcomes: tuple[tuple[str, str], ...]
    memory_id: str | None


class CausalContactPushSearch:
    """Matched Memory ON/OFF search with an identical physical scenario."""

    def __init__(
        self,
        *,
        physics: ContactPushPhysics,
        knowledge: ContactPushTaskKnowledge,
    ) -> None:
        self.physics = physics
        self.knowledge = knowledge

    def run(
        self,
        scenario: ContactPushScenario,
        *,
        memory: ContactPushMemory | None,
    ) -> CausalSearchResult:
        suggestion = memory.retrieve(scenario) if memory is not None else None
        policies = []
        if suggestion is not None:
            adapted = ContactPushHow().generate(
                failure=_synthetic_overshoot_failure(scenario),
                scenario=scenario,
                suggestion=suggestion,
                task_card_hash=self.knowledge.card_hash,
            )
            _patch, policy = adapted.compile(
                knowledge=self.knowledge,
                parent=ContactPushPolicy.baseline(),
            )
            policies.append(policy)
        duration = min(1.25, max(0.75, 0.72 + 1.30 * scenario.target_distance_m))
        deceleration = min(0.82, max(0.62, 0.68 + 0.12 * scenario.observed_friction))
        for velocity in (0.46, 0.42, 0.38, 0.34, 0.30, 0.26, 0.22, 0.18, 0.14, 0.10):
            policies.append(
                ContactPushPolicy(
                    push_velocity_mps=velocity,
                    contact_duration_sec=duration,
                    contact_offset_y_m=scenario.initial_offset_y_m,
                    deceleration_fraction=deceleration,
                    micro_push=True,
                    policy_type="parameter",
                )
            )
        unique: dict[str, ContactPushPolicy] = {}
        for policy in policies:
            unique.setdefault(policy.policy_hash, policy)
        outcomes = []
        for attempt, policy in enumerate(unique.values(), start=1):
            valid, _errors = self.knowledge.validate_policy(policy)
            if not valid:
                continue
            result = self.physics.run(scenario, policy)
            outcomes.append((policy.policy_hash, result.status.value))
            if result.success:
                return CausalSearchResult(
                    memory_enabled=memory is not None,
                    attempts=attempt,
                    success=True,
                    selected_policy=policy,
                    outcomes=tuple(outcomes),
                    memory_id=suggestion.memory_id if suggestion else None,
                )
        return CausalSearchResult(
            memory_enabled=memory is not None,
            attempts=len(outcomes),
            success=False,
            selected_policy=None,
            outcomes=tuple(outcomes),
            memory_id=suggestion.memory_id if suggestion else None,
        )


def apply_candidate_patch(
    parent: ContactPushPolicy,
    patch: CandidatePatch,
    *,
    policy_type: str,
) -> ContactPushPolicy:
    values = _policy_pointer_dict(parent)
    for change in patch.changes:
        values[change.path] = change.new
    return ContactPushPolicy(
        push_velocity_mps=float(values["/controller/push_velocity_mps"]),
        contact_duration_sec=float(values["/controller/contact_duration_sec"]),
        contact_offset_y_m=float(values["/controller/contact_offset_y_m"]),
        deceleration_fraction=float(values["/trajectory/deceleration_fraction"]),
        micro_push=bool(values["/skill_graph/micro_push"]),
        policy_type=policy_type,
    )


def _policy_pointer_dict(policy: ContactPushPolicy) -> dict[str, float | bool]:
    return {
        "/controller/push_velocity_mps": policy.push_velocity_mps,
        "/controller/contact_duration_sec": policy.contact_duration_sec,
        "/controller/contact_offset_y_m": policy.contact_offset_y_m,
        "/trajectory/deceleration_fraction": policy.deceleration_fraction,
        "/skill_graph/micro_push": policy.micro_push,
    }


def _policy_from_dict(value: dict[str, Any], *, policy_type: str) -> ContactPushPolicy:
    return ContactPushPolicy(
        push_velocity_mps=float(value["push_velocity_mps"]),
        contact_duration_sec=float(value["contact_duration_sec"]),
        contact_offset_y_m=float(value["contact_offset_y_m"]),
        deceleration_fraction=float(value["deceleration_fraction"]),
        micro_push=bool(value["micro_push"]),
        policy_type=policy_type,
    )


def _feature_distance(left: dict[str, float], right: dict[str, float]) -> float:
    scales = {
        "object_mass_kg": 0.6,
        "estimated_friction": 0.8,
        "target_distance_m": 0.25,
        "initial_offset_y_m": 0.04,
        "control_delay_sec": 0.08,
    }
    squared = [((left[key] - right[key]) / scale) ** 2 for key, scale in scales.items()]
    return math.sqrt(sum(squared) / len(squared))


def _model_features(features: dict[str, float]) -> tuple[float, ...]:
    mass = float(features["object_mass_kg"])
    friction = float(features["estimated_friction"])
    target = float(features["target_distance_m"])
    offset = float(features["initial_offset_y_m"])
    delay = float(features["control_delay_sec"])
    return (
        1.0,
        mass,
        friction,
        target,
        offset,
        delay,
        target / max(friction, 1e-9),
        target * mass,
        target * target,
        friction * friction,
        mass * friction,
    )


def _synthetic_overshoot_failure(scenario: ContactPushScenario) -> FailureSignatureV2:
    from rosclaw.simforge.failure_router_v2 import FailureObservation, FailureRouterV2

    return FailureRouterV2().route(
        FailureObservation(
            task_id=CONTACT_PUSH_TASK_ID,
            body_id="sim_contact_pusher",
            expected_body_hash=CONTACT_PUSH_BODY_HASH,
            observed_body_hash=CONTACT_PUSH_BODY_HASH,
            action_id=f"action_{scenario.scenario_id}",
            evidence_refs=(f"scenario://{scenario.scenario_commitment.removeprefix('sha256:')}",),
            task_success=False,
            target_error_m=-0.1,
            target_tolerance_m=0.035,
            object_overshot=True,
            estimated_friction=scenario.observed_friction,
            peak_force_n=10.0,
            force_limit_n=30.0,
        )
    )


def _hash_json(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


__all__ = [
    "CausalContactPushSearch",
    "CausalSearchResult",
    "ContactPushCandidate",
    "ContactPushCandidateType",
    "ContactPushExpertSearch",
    "ContactPushHow",
    "ContactPushMemory",
    "ContactPushTaskKnowledge",
    "ContextualPolicyModel",
    "ContextualPolicyTrainer",
    "ExecutableRecoveryHint",
    "ExpertSearchResult",
    "MemorySuggestion",
    "apply_candidate_patch",
]
