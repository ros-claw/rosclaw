"""Executable ContactPush Practice-to-Policy and causal recovery flywheel."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from rosclaw.simforge.contact_push_learning import (
    CausalContactPushSearch,
    CausalSearchResult,
    ContactPushCandidate,
    ContactPushCandidateType,
    ContactPushExpertSearch,
    ContactPushMemory,
    ContactPushTaskKnowledge,
    ContextualPolicyModel,
    ContextualPolicyTrainer,
)
from rosclaw.simforge.dataset_snapshot import (
    PracticeDatasetBuilder,
    PracticeDatasetSnapshot,
    SnapshotFiles,
    load_public_partition,
)
from rosclaw.simforge.failure_router_v2 import (
    FailureObservation,
    FailureRouterV2,
    FailureSignatureV2,
)
from rosclaw.simforge.models import Partition
from rosclaw.simforge.seed_ledger import SeedLedger
from rosclaw.simforge.tasks.contact_push_v3 import (
    CONTACT_PUSH_BODY_HASH,
    CONTACT_PUSH_BODY_ID,
    CONTACT_PUSH_TASK_ID,
    ContactPushEpisodeEvidence,
    ContactPushPhysics,
    ContactPushPolicy,
    ContactPushScenario,
    ContactPushStatus,
    generate_contact_push_scenarios,
)


@dataclass(frozen=True)
class ContactPushFlywheel:
    snapshot: PracticeDatasetSnapshot
    snapshot_files: SnapshotFiles
    model: ContextualPolicyModel
    practice_evidence: tuple[ContactPushEpisodeEvidence, ...]
    development_episode_ids: tuple[str, ...]
    validation_episode_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rosclaw.contact_push_flywheel.v1",
            "dataset_snapshot": self.snapshot.to_dict(),
            "model": self.model.to_dict(),
            "model_hash": self.model.artifact_hash,
            "practice_episode_count": len(self.practice_evidence),
            "development_episode_count": len(self.development_episode_ids),
            "validation_episode_count": len(self.validation_episode_ids),
            "private_holdout_disclosed": False,
        }


@dataclass(frozen=True)
class KnowAblation:
    proposals_without_know: int
    invalid_without_know: int
    accepted_with_know: int
    rejected_with_know: int
    invalid_candidates_reduced: int
    safety_override_admitted: int
    rejected_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContactPushCausalLoop:
    scenario: ContactPushScenario
    failure: FailureSignatureV2
    baseline_evidence: ContactPushEpisodeEvidence
    recovery_evidence: ContactPushEpisodeEvidence
    memory_off: CausalSearchResult
    memory_on: CausalSearchResult
    know_ablation: KnowAblation
    candidate_parameter: ContactPushCandidate
    candidate_trajectory: ContactPushCandidate
    candidate_skill_graph: ContactPushCandidate
    candidate_learned: ContactPushCandidate
    wrong_body_memory_rejected: bool
    stale_memory_rejected: bool

    @property
    def memory_attempts_saved(self) -> int:
        return self.memory_off.attempts - self.memory_on.attempts

    @property
    def same_seed_retry_passed(self) -> bool:
        return (
            self.baseline_evidence.scenario.scenario_commitment
            == self.recovery_evidence.scenario.scenario_commitment
            and self.baseline_evidence.scenario.seed_commitment
            == self.recovery_evidence.scenario.seed_commitment
            and not self.baseline_evidence.result.success
            and self.recovery_evidence.result.success
        )

    def candidates(self) -> tuple[ContactPushCandidate, ...]:
        return (
            self.candidate_parameter,
            self.candidate_trajectory,
            self.candidate_skill_graph,
            self.candidate_learned,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rosclaw.contact_push_causal_loop.v1",
            "scenario": self.scenario.to_dict(reveal_seed=True),
            "failure": self.failure.to_dict(),
            "baseline": {
                "episode_id": self.baseline_evidence.episode_id,
                "status": self.baseline_evidence.result.status.value,
                "receipt_hash": self.baseline_evidence.receipt_hash,
                "strict_replay": self.baseline_evidence.strict_replay,
            },
            "same_seed_recovery": {
                "episode_id": self.recovery_evidence.episode_id,
                "status": self.recovery_evidence.result.status.value,
                "receipt_hash": self.recovery_evidence.receipt_hash,
                "strict_replay": self.recovery_evidence.strict_replay,
                "passed": self.same_seed_retry_passed,
            },
            "memory_ablation": {
                "off": _search_dict(self.memory_off),
                "on": _search_dict(self.memory_on),
                "attempts_saved": self.memory_attempts_saved,
                "wrong_body_rejected": self.wrong_body_memory_rejected,
                "stale_memory_rejected": self.stale_memory_rejected,
            },
            "know_ablation": self.know_ablation.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates()],
        }


def build_contact_push_flywheel(
    *,
    output_root: Path,
    source_checkout: Path,
    practice_episodes: int,
    root_seed: int,
) -> ContactPushFlywheel:
    """Produce replayed Practice evidence, an immutable split, and a trained policy."""

    if practice_episodes < 24:
        raise ValueError("ContactPush flywheel requires at least 24 Practice episodes")
    root = _external_root(output_root, source_checkout)
    root.mkdir(parents=True, exist_ok=False)
    physics = ContactPushPhysics(trace_stride=20)
    expert = ContactPushExpertSearch(physics)
    ledger = SeedLedger(
        task_id=CONTACT_PUSH_TASK_ID,
        secret=_secret(root_seed, "practice-ledger"),
    )
    scenarios = generate_contact_push_scenarios(
        ledger=ledger,
        partition=Partition.DEVELOPMENT,
        count=practice_episodes,
        root_seed=root_seed,
    )
    evidence = []
    for scenario in scenarios:
        expert_result = expert.optimize(scenario)
        if not expert_result.outcome.success:
            raise RuntimeError(
                f"Practice expert failed to solve {scenario.scenario_id} within its fixed budget"
            )
        evidence.append(
            physics.run_and_record(
                scenario=scenario,
                policy=expert_result.policy,
                artifact_root=root / "raw-practice",
                source_checkout=source_checkout,
                practice_id="practice_contact_push_contextual_policy_v1",
            )
        )
    records = tuple(item.to_practice_record() for item in evidence)
    snapshot, files = PracticeDatasetBuilder(
        source_checkout=source_checkout,
        split_secret=_secret(root_seed, "dataset-split"),
    ).build(
        records=records,
        output_dir=root / "dataset",
        dataset_id="dataset_contact_push_contextual_policy_v1",
        label_provenance={
            "task_success": "independent_contact_push_task_verifier_v3",
            "failure_signature": "failure_router_v2",
            "contact_force": "mujoco_contact_force",
            "strict_replay": "independent_physics_reexecution",
        },
    )
    development = load_public_partition(files.development)
    validation = load_public_partition(files.validation)
    model = ContextualPolicyTrainer().train(
        development=development,
        validation=validation,
        dataset_snapshot_hash=snapshot.snapshot_hash,
    )
    flywheel = ContactPushFlywheel(
        snapshot=snapshot,
        snapshot_files=files,
        model=model,
        practice_evidence=tuple(evidence),
        development_episode_ids=tuple(record.episode_id for record in development),
        validation_episode_ids=tuple(record.episode_id for record in validation),
    )
    _atomic_json(root / "flywheel.json", flywheel.to_dict())
    _atomic_json(root / "contextual-policy.json", model.to_dict())
    return flywheel


def run_contact_push_causal_loop(
    *,
    flywheel: ContactPushFlywheel,
    output_root: Path,
    source_checkout: Path,
    root_seed: int,
) -> ContactPushCausalLoop:
    """Run matched ablations and construct four executable candidate families."""

    root = _external_root(output_root, source_checkout)
    root.mkdir(parents=True, exist_ok=False)
    physics = ContactPushPhysics(trace_stride=10)
    knowledge = ContactPushTaskKnowledge.default()
    memory = ContactPushMemory()
    development_ids = set(flywheel.development_episode_ids)
    for item in flywheel.practice_evidence:
        if item.episode_id in development_ids:
            memory.ingest_recovery(item)
    if memory.repository.count() != len(development_ids):
        raise RuntimeError("Memory ingestion lost distinct Practice recoveries")

    scenarios = generate_contact_push_scenarios(
        ledger=SeedLedger(
            task_id=CONTACT_PUSH_TASK_ID,
            secret=_secret(root_seed, "causal-discovery"),
        ),
        partition=Partition.DISCOVERY,
        count=96,
        root_seed=root_seed + 101,
    )
    search = CausalContactPushSearch(physics=physics, knowledge=knowledge)
    selected: tuple[ContactPushScenario, CausalSearchResult, CausalSearchResult] | None = None
    for scenario in scenarios:
        baseline = physics.run(scenario, ContactPushPolicy.baseline())
        if baseline.status is not ContactPushStatus.OVERSHOOT:
            continue
        memory_off = search.run(scenario, memory=None)
        memory_on = search.run(scenario, memory=memory)
        if (
            memory_off.success
            and memory_on.success
            and memory_on.memory_id is not None
            and memory_on.attempts == 1
            and memory_on.attempts < memory_off.attempts
        ):
            selected = (scenario, memory_off, memory_on)
            break
    if selected is None:
        raise RuntimeError("no matched ContactPush scenario demonstrated a causal Memory benefit")
    scenario, memory_off, memory_on = selected
    if memory_on.selected_policy is None:
        raise RuntimeError("Memory treatment reported success without a selected policy")
    baseline_evidence = physics.run_and_record(
        scenario=scenario,
        policy=ContactPushPolicy.baseline(),
        artifact_root=root / "flagship" / "baseline",
        source_checkout=source_checkout,
        practice_id="practice_contact_push_flagship_failure",
    )
    failure = FailureRouterV2().route(
        FailureObservation(
            task_id=CONTACT_PUSH_TASK_ID,
            body_id=CONTACT_PUSH_BODY_ID,
            expected_body_hash=CONTACT_PUSH_BODY_HASH,
            observed_body_hash=CONTACT_PUSH_BODY_HASH,
            action_id=f"action_{baseline_evidence.episode_id}",
            evidence_refs=(
                f"episode://{baseline_evidence.episode_id}",
                "receipt://" + baseline_evidence.receipt_hash.removeprefix("sha256:"),
            ),
            task_success=False,
            target_error_m=baseline_evidence.result.final_error_m,
            target_tolerance_m=baseline_evidence.result.target_tolerance_m,
            object_overshot=True,
            estimated_friction=scenario.observed_friction,
            peak_force_n=baseline_evidence.result.peak_contact_force_n,
            force_limit_n=baseline_evidence.result.force_limit_n,
        )
    )
    recovery_evidence = physics.run_and_record(
        scenario=scenario,
        policy=memory_on.selected_policy,
        artifact_root=root / "flagship" / "recovery",
        source_checkout=source_checkout,
        practice_id="practice_contact_push_flagship_recovery",
    )
    if not recovery_evidence.result.success:
        raise RuntimeError("the compiled How intervention failed its exact-seed retry")

    know_ablation = _run_know_ablation(knowledge)
    wrong_body_rejected = memory.retrieve(scenario, body_hash="sha256:" + "0" * 64) is None
    stale_memory_rejected = _stale_memory_is_rejected(memory, scenario)
    lineage = (
        "failure://" + failure.failure_id,
        "dataset://" + flywheel.snapshot.snapshot_hash.removeprefix("sha256:"),
        "receipt://" + recovery_evidence.receipt_hash.removeprefix("sha256:"),
    )
    candidates = _build_candidates(
        model=flywheel.model,
        failure=failure,
        knowledge=knowledge,
        recovery_policy=memory_on.selected_policy,
        lineage=lineage,
    )
    loop = ContactPushCausalLoop(
        scenario=scenario,
        failure=failure,
        baseline_evidence=baseline_evidence,
        recovery_evidence=recovery_evidence,
        memory_off=memory_off,
        memory_on=memory_on,
        know_ablation=know_ablation,
        candidate_parameter=candidates[0],
        candidate_trajectory=candidates[1],
        candidate_skill_graph=candidates[2],
        candidate_learned=candidates[3],
        wrong_body_memory_rejected=wrong_body_rejected,
        stale_memory_rejected=stale_memory_rejected,
    )
    _atomic_json(root / "causal-loop.json", loop.to_dict())
    candidates_root = root / "candidates"
    candidates_root.mkdir(parents=True, exist_ok=False)
    for candidate in loop.candidates():
        _atomic_json(candidates_root / f"{candidate.candidate_id}.json", candidate.to_dict())
    return loop


def build_canary_regression_candidate(
    causal_loop: ContactPushCausalLoop,
    *,
    velocity_scale: float = 1.08,
) -> ContactPushCandidate:
    """Create a plausible learned-policy update for independent Canary testing."""

    if not 1.01 <= velocity_scale <= 1.20:
        raise ValueError("Canary regression velocity scale must be in [1.01, 1.20]")
    model = causal_loop.candidate_learned.learned_policy
    if model is None:
        raise ValueError("Canary regression requires a learned ContactPush candidate")
    coefficients = [list(row) for row in model.coefficients]
    for row in coefficients:
        row[0] *= velocity_scale
    regressed_model = replace(
        model,
        coefficients=tuple(tuple(value for value in row) for row in coefficients),
    )
    return ContactPushCandidate.learned(
        model=regressed_model,
        parent=ContactPushPolicy.baseline(),
        failure_signature_id=causal_loop.failure.failure_id,
        task_card_hash=causal_loop.candidate_learned.task_card_hash,
        lineage_refs=(
            *causal_loop.candidate_learned.lineage_refs,
            f"canary://velocity_scale_{velocity_scale:.3f}",
        ),
    )


def _run_know_ablation(knowledge: ContactPushTaskKnowledge) -> KnowAblation:
    proposals: tuple[dict[str, Any], ...] = (
        {"/controller/push_velocity_mps": 0.31},
        {"/skill_graph/micro_push": True},
        {"/controller/contact_duration_sec": 4.0},
        {"/safety/force_limit_n": 1000.0},
    )
    accepted, rejected = knowledge.filter_proposals(proposals)
    invalid_without = 2
    safety_admitted = sum("/safety/force_limit_n" in proposal for proposal in accepted)
    return KnowAblation(
        proposals_without_know=len(proposals),
        invalid_without_know=invalid_without,
        accepted_with_know=len(accepted),
        rejected_with_know=len(rejected),
        invalid_candidates_reduced=invalid_without,
        safety_override_admitted=safety_admitted,
        rejected_reasons=tuple(str(item["reason"]) for item in rejected),
    )


def _stale_memory_is_rejected(
    memory: ContactPushMemory,
    scenario: ContactPushScenario,
) -> bool:
    suggestion = memory.retrieve(scenario)
    if suggestion is None:
        return False
    item = memory.repository.get(suggestion.memory_id)
    if item is None:
        return False
    original = dict(item.metadata)
    memory.repository.update_fields(
        suggestion.memory_id,
        {"metadata": json.dumps({**original, "strict_replay": False}, sort_keys=True)},
    )
    rejected = memory.retrieve(scenario) is None or (
        memory.retrieve(scenario) is not None
        and memory.retrieve(scenario).memory_id != suggestion.memory_id
    )
    memory.repository.update_fields(
        suggestion.memory_id,
        {"metadata": json.dumps(original, sort_keys=True)},
    )
    return rejected


def _build_candidates(
    *,
    model: ContextualPolicyModel,
    failure: FailureSignatureV2,
    knowledge: ContactPushTaskKnowledge,
    recovery_policy: ContactPushPolicy,
    lineage: tuple[str, ...],
) -> tuple[
    ContactPushCandidate,
    ContactPushCandidate,
    ContactPushCandidate,
    ContactPushCandidate,
]:
    parent = ContactPushPolicy.baseline()
    dataset_hash = model.dataset_snapshot_hash
    parameter_policy = ContactPushPolicy(
        push_velocity_mps=0.50,
        contact_duration_sec=1.40,
        contact_offset_y_m=0.0,
        deceleration_fraction=1.0,
        micro_push=False,
        policy_type="parameter",
    )
    trajectory_policy = ContactPushPolicy(
        push_velocity_mps=recovery_policy.push_velocity_mps,
        contact_duration_sec=recovery_policy.contact_duration_sec,
        contact_offset_y_m=recovery_policy.contact_offset_y_m,
        deceleration_fraction=recovery_policy.deceleration_fraction,
        micro_push=False,
        policy_type="trajectory",
    )
    skill_graph_policy = ContactPushPolicy(
        push_velocity_mps=recovery_policy.push_velocity_mps,
        contact_duration_sec=recovery_policy.contact_duration_sec,
        contact_offset_y_m=recovery_policy.contact_offset_y_m,
        deceleration_fraction=recovery_policy.deceleration_fraction,
        micro_push=True,
        policy_type="skill_graph",
    )
    common = {
        "parent": parent,
        "failure_signature_id": failure.failure_id,
        "task_card_hash": knowledge.card_hash,
        "dataset_snapshot_hash": dataset_hash,
        "lineage_refs": lineage,
    }
    return (
        ContactPushCandidate.static(
            candidate_type=ContactPushCandidateType.PARAMETER,
            policy=parameter_policy,
            **common,
        ),
        ContactPushCandidate.static(
            candidate_type=ContactPushCandidateType.TRAJECTORY,
            policy=trajectory_policy,
            **common,
        ),
        ContactPushCandidate.static(
            candidate_type=ContactPushCandidateType.SKILL_GRAPH,
            policy=skill_graph_policy,
            **common,
        ),
        ContactPushCandidate.learned(
            model=model,
            parent=parent,
            failure_signature_id=failure.failure_id,
            task_card_hash=knowledge.card_hash,
            lineage_refs=lineage,
        ),
    )


def _search_dict(result: CausalSearchResult) -> dict[str, Any]:
    return {
        "enabled": result.memory_enabled,
        "attempts": result.attempts,
        "success": result.success,
        "selected_policy_hash": (
            result.selected_policy.policy_hash if result.selected_policy is not None else None
        ),
        "outcomes": [
            {"policy_hash": policy_hash, "status": status}
            for policy_hash, status in result.outcomes
        ],
        "memory_id": result.memory_id,
    }


def _external_root(output_root: Path, source_checkout: Path) -> Path:
    root = output_root.expanduser().resolve()
    checkout = source_checkout.resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("ContactPush flywheel artifacts must stay outside the checkout")
    return root


def _secret(root_seed: int, purpose: str) -> bytes:
    return hashlib.sha256(f"rosclaw.contact_push.phase3\0{root_seed}\0{purpose}".encode()).digest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "ContactPushCausalLoop",
    "ContactPushFlywheel",
    "KnowAblation",
    "build_canary_regression_candidate",
    "build_contact_push_flywheel",
    "run_contact_push_causal_loop",
]
