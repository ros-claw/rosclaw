"""End-to-end DreamForge adapter for SIM-only G1 recovery learning.

This module is deliberately task specific.  It translates GoalForge evidence
into the generic Growth and Dream contracts without teaching Dream Core what a
football, a G1, or a joint torque is.  The active controller is never changed.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.collective.sources.motiondecode.motion_prior import (
    load_g1_motion_prior_artifact,
)
from rosclaw.continual.services.persistence import atomic_write_json
from rosclaw.dream.contracts import DreamBudget, DreamType
from rosclaw.dream.control import DreamPlanner, DreamPlanRequest, DreamScheduler
from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.contracts import (
    ConsolidationDecision,
    ConsolidationManifest,
    GateName,
    GateResult,
    GateStatus,
    GrowthMetricSpec,
    MetricDirection,
    SkillGrowthSpec,
)
from rosclaw.simforge.evolution import EvolutionRun, EvolutionState
from rosclaw.simforge.g1_neural_torque import (
    G1_NEURAL_TORQUE_ACTIONS,
    G1_NEURAL_TORQUE_OBSERVATIONS,
    load_g1_neural_torque_artifact,
)
from rosclaw.simforge.g1_recovery_awr_validation import run_g1_recovery_awr_validation
from rosclaw.simforge.tasks.g1_goalforge.concepts import hash_bytes

_CONTEXT_FEATURES = (
    "policy_phase",
    "pelvis_height_m",
    "projected_gravity_x",
    "projected_gravity_y",
    "projected_gravity_z",
    "base_linear_velocity_x_mps",
    "base_linear_velocity_y_mps",
    "base_linear_velocity_z_mps",
    "base_angular_velocity_x_rps",
    "base_angular_velocity_y_rps",
    "base_angular_velocity_z_rps",
    "ball_speed_mps",
    "ball_direction_x",
    "ball_direction_y",
    "ball_direction_z",
    "left_contact",
    "right_contact",
)


def _file_hash(path: Path) -> str:
    return hash_bytes(path.expanduser().resolve().read_bytes())


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


@dataclass(frozen=True)
class G1RecoveryContextCluster:
    cluster_id: str
    center: tuple[float, ...]
    normalized_radius: float
    scenario_ids: tuple[str, ...]
    sample_count: int
    elite_count: int
    elite_fraction: float
    mean_naturalness_reduction: float
    worst_naturalness_reduction: float
    task_failure_count: int
    priority_score: float
    schema_version: str = "rosclaw.simforge.g1_recovery_context_cluster.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class G1RecoveryContextCurriculum:
    source_report_hash: str
    feature_names: tuple[str, ...]
    normalization_center: tuple[float, ...]
    normalization_scale: tuple[float, ...]
    valid_context_count: int
    missing_context_count: int
    scenario_count: int
    clusters: tuple[G1RecoveryContextCluster, ...]
    routing_ready: bool
    routing_blockers: tuple[str, ...]
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.simforge.g1_recovery_context_curriculum.v1"

    @property
    def curriculum_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_report_hash": self.source_report_hash,
            "feature_names": list(self.feature_names),
            "normalization_center": list(self.normalization_center),
            "normalization_scale": list(self.normalization_scale),
            "valid_context_count": self.valid_context_count,
            "missing_context_count": self.missing_context_count,
            "scenario_count": self.scenario_count,
            "clusters": [item.to_dict() for item in self.clusters],
            "routing_ready": self.routing_ready,
            "routing_blockers": list(self.routing_blockers),
            "activation_ceiling": self.activation_ceiling,
            "hardware_authorized": self.hardware_authorized,
        }


@dataclass(frozen=True)
class G1RecoveryContextRoute:
    context_hash: str
    expert_cluster_id: str | None
    normalized_distance: float | None
    eligible: bool
    fallback_reason: str | None
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.simforge.g1_recovery_context_route.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_g1_recovery_context_curriculum(
    report: Mapping[str, Any],
    *,
    source_report_hash: str,
    maximum_clusters: int = 3,
) -> G1RecoveryContextCurriculum:
    """Cluster contact-time self states without looking at holdout rows."""

    if report.get("schema_version") != "rosclaw.simforge.g1_recovery_awr_validation.v1":
        raise ValueError("unsupported G1 recovery AWR report schema")
    if report.get("hardware_authorized") is not False:
        raise ValueError("G1 recovery Dream input must be explicitly hardware-inactive")
    if not 1 <= maximum_clusters <= 8:
        raise ValueError("maximum context clusters must be in [1, 8]")
    collection = report.get("collection")
    if not isinstance(collection, list) or not collection:
        raise ValueError("G1 recovery Dream requires non-empty collection evidence")

    rows: list[np.ndarray] = []
    metadata: list[Mapping[str, Any]] = []
    missing = 0
    for raw in collection:
        item = _mapping(raw, label="collection row")
        context = item.get("contact_context")
        if not isinstance(context, Mapping):
            missing += 1
            continue
        try:
            row = np.asarray(
                [float(context[name]) for name in _CONTEXT_FEATURES],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("contact context does not match the v1 feature contract") from exc
        if not np.all(np.isfinite(row)):
            raise ValueError("contact context contains non-finite values")
        rows.append(row)
        metadata.append(item)

    blockers: list[str] = []
    if not rows:
        blockers.append("no_contact_context_snapshots")
        return G1RecoveryContextCurriculum(
            source_report_hash=source_report_hash,
            feature_names=_CONTEXT_FEATURES,
            normalization_center=tuple(0.0 for _ in _CONTEXT_FEATURES),
            normalization_scale=tuple(1.0 for _ in _CONTEXT_FEATURES),
            valid_context_count=0,
            missing_context_count=missing,
            scenario_count=0,
            clusters=(),
            routing_ready=False,
            routing_blockers=tuple(blockers),
        )

    matrix = np.stack(rows)
    center = np.median(matrix, axis=0)
    scale = np.quantile(np.abs(matrix - center), 0.75, axis=0)
    scale = np.maximum(scale, np.asarray([1e-3] * len(_CONTEXT_FEATURES)))
    normalized = (matrix - center) / scale
    scenario_count = len({str(item.get("scenario_id", "")) for item in metadata})
    cluster_count = min(maximum_clusters, scenario_count, len(rows))
    labels, centers = _deterministic_kmeans(normalized, cluster_count)
    clusters: list[G1RecoveryContextCluster] = []
    for index in range(cluster_count):
        selected = np.flatnonzero(labels == index)
        distances = np.linalg.norm(normalized[selected] - centers[index], axis=1)
        items = [metadata[int(value)] for value in selected]
        reductions = [float(item.get("naturalness_reduction", -math.inf)) for item in items]
        if any(not math.isfinite(value) for value in reductions):
            raise ValueError("context outcomes must contain finite naturalness reductions")
        elite_count = sum(bool(item.get("elite", False)) for item in items)
        failures = sum(not bool(item.get("task_preserved", False)) for item in items)
        mean_reduction = float(np.mean(reductions))
        priority = float(failures / len(items) + max(0.0, -mean_reduction))
        clusters.append(
            G1RecoveryContextCluster(
                cluster_id=f"recovery-context-{index:02d}",
                center=tuple(float(value) for value in centers[index]),
                normalized_radius=float(np.max(distances) + 1e-6),
                scenario_ids=tuple(sorted({str(item["scenario_id"]) for item in items})),
                sample_count=len(items),
                elite_count=elite_count,
                elite_fraction=elite_count / len(items),
                mean_naturalness_reduction=mean_reduction,
                worst_naturalness_reduction=min(reductions),
                task_failure_count=failures,
                priority_score=priority,
            )
        )
    clusters.sort(key=lambda item: (-item.priority_score, item.cluster_id))
    if missing:
        blockers.append("some_collection_rows_lack_contact_context")
    if scenario_count < 3:
        blockers.append("fewer_than_three_context_scenarios")
    if any(item.sample_count < 2 for item in clusters):
        blockers.append("context_cluster_has_fewer_than_two_samples")
    if not any(item.elite_count for item in clusters):
        blockers.append("no_elite_context_available")
    return G1RecoveryContextCurriculum(
        source_report_hash=source_report_hash,
        feature_names=_CONTEXT_FEATURES,
        normalization_center=tuple(float(value) for value in center),
        normalization_scale=tuple(float(value) for value in scale),
        valid_context_count=len(rows),
        missing_context_count=missing,
        scenario_count=scenario_count,
        clusters=tuple(clusters),
        routing_ready=not blockers,
        routing_blockers=tuple(blockers),
    )


def route_g1_recovery_context(
    curriculum: G1RecoveryContextCurriculum,
    context: Mapping[str, Any],
) -> G1RecoveryContextRoute:
    """Select an applicable training domain or return an exact parent fallback."""

    context_value = {name: context.get(name) for name in _CONTEXT_FEATURES}
    context_hash = canonical_hash(
        {
            "schema_version": "rosclaw.simforge.g1_recovery_context_route_input.v1",
            "context": context_value,
        }
    )
    if not curriculum.routing_ready:
        return G1RecoveryContextRoute(
            context_hash=context_hash,
            expert_cluster_id=None,
            normalized_distance=None,
            eligible=False,
            fallback_reason="context_curriculum_not_qualified",
        )
    try:
        row = np.asarray([float(context[name]) for name in _CONTEXT_FEATURES], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return G1RecoveryContextRoute(
            context_hash=context_hash,
            expert_cluster_id=None,
            normalized_distance=None,
            eligible=False,
            fallback_reason="context_contract_mismatch",
        )
    if not np.all(np.isfinite(row)):
        return G1RecoveryContextRoute(
            context_hash=context_hash,
            expert_cluster_id=None,
            normalized_distance=None,
            eligible=False,
            fallback_reason="context_nonfinite",
        )
    center = np.asarray(curriculum.normalization_center, dtype=np.float64)
    scale = np.asarray(curriculum.normalization_scale, dtype=np.float64)
    normalized = (row - center) / scale
    distances = tuple(
        float(np.linalg.norm(normalized - np.asarray(cluster.center, dtype=np.float64)))
        for cluster in curriculum.clusters
    )
    index = int(np.argmin(distances))
    cluster = curriculum.clusters[index]
    distance = distances[index]
    if distance > cluster.normalized_radius:
        return G1RecoveryContextRoute(
            context_hash=context_hash,
            expert_cluster_id=None,
            normalized_distance=distance,
            eligible=False,
            fallback_reason="outside_sealed_context_envelope",
        )
    if cluster.elite_count == 0:
        return G1RecoveryContextRoute(
            context_hash=context_hash,
            expert_cluster_id=None,
            normalized_distance=distance,
            eligible=False,
            fallback_reason="cluster_has_no_qualified_expert_data",
        )
    return G1RecoveryContextRoute(
        context_hash=context_hash,
        expert_cluster_id=cluster.cluster_id,
        normalized_distance=distance,
        eligible=True,
        fallback_reason=None,
    )


def _deterministic_kmeans(matrix: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    if matrix.ndim != 2 or not len(matrix) or not 1 <= count <= len(matrix):
        raise ValueError("context clustering input is invalid")
    chosen = [0]
    while len(chosen) < count:
        distance = np.min(
            np.stack(
                [np.sum(np.square(matrix - matrix[index]), axis=1) for index in chosen]
            ),
            axis=0,
        )
        distance[np.asarray(chosen)] = -1.0
        chosen.append(int(np.argmax(distance)))
    centers = matrix[np.asarray(chosen)].copy()
    labels = np.zeros(len(matrix), dtype=np.int64)
    for _ in range(32):
        distances = np.stack(
            [np.sum(np.square(matrix - value), axis=1) for value in centers],
            axis=1,
        )
        updated = np.argmin(distances, axis=1)
        next_centers = centers.copy()
        for index in range(count):
            selected = matrix[updated == index]
            if len(selected):
                next_centers[index] = np.mean(selected, axis=0)
        if np.array_equal(updated, labels) and np.allclose(next_centers, centers):
            labels = updated
            centers = next_centers
            break
        labels = updated
        centers = next_centers
    return labels, centers


def run_g1_recovery_dream_cycle(
    *,
    asset_root: Path,
    motion_prior_path: Path,
    motiondecode_pilot_report_path: Path,
    stable_artifact_path: Path,
    recovery_artifact_path: Path,
    output_dir: Path,
    source_checkout: Path,
    prior_report_path: Path | None = None,
    device: str = "cuda:1",
    seed: int = 8920,
    exploration_replicates: int = 6,
    value_updates: int = 64,
    actor_updates: int = 4,
    validation_round: int = 1,
    validation_runner: Callable[..., dict[str, Any]] = run_g1_recovery_awr_validation,
) -> dict[str, Any]:
    """Run planning, learning, independent replay, gates, and terminal journaling."""

    root = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("G1 recovery Dream evidence must be outside the source checkout")
    root.mkdir(parents=True, exist_ok=False)
    stable = load_g1_neural_torque_artifact(stable_artifact_path)
    retained = load_g1_neural_torque_artifact(
        recovery_artifact_path,
        expected_body_hash=stable.body_hash,
    )
    prior = load_g1_motion_prior_artifact(motion_prior_path)
    if prior.body_hash != stable.body_hash:
        raise ValueError("G1 recovery Dream inputs target different bodies")
    if stable.parent_policy_hash != retained.parent_policy_hash:
        raise ValueError("G1 recovery Dream inputs have different controller parents")
    prior_report_hash: str | None = None
    prior_replay_hash: str | None = None
    if prior_report_path is not None:
        prior_report_hash = _file_hash(prior_report_path)
        import json

        prior_report = json.loads(prior_report_path.read_text(encoding="utf-8"))
        if not isinstance(prior_report, dict):
            raise ValueError("prior G1 recovery report must be a JSON object")
        value = prior_report.get("online_replay_hash")
        if isinstance(value, str):
            prior_replay_hash = value

    boundary_hash = canonical_hash(
        {
            "suite": "g1.recovery.awr.independent",
            "validation_round": validation_round,
            "strict_replay": True,
        }
    )
    spec = SkillGrowthSpec(
        skill_id="g1.football.recovery",
        adapter_id="simforge.g1.recovery_awr",
        body_hashes=(stable.body_hash,),
        capability_ids=("stable_kick", "post_kick_recovery", "natural_motion"),
        observation_contract_hash=canonical_hash(
            {"torque_observations": G1_NEURAL_TORQUE_OBSERVATIONS, "context": _CONTEXT_FEATURES}
        ),
        action_contract_hash=canonical_hash(
            {"direct_torque_actions": G1_NEURAL_TORQUE_ACTIONS, "sim_only": True}
        ),
        reward_contract_hash=canonical_hash(
            {"recovery_naturalness": "g1_recovery_naturalness_cost.v1"}
        ),
        cost_contract_hash=canonical_hash(
            {"costs": ["fall", "joint_limit", "torque", "slip", "nonfinite"]}
        ),
        practice_source_ids=("simforge.goalforge.strict_replay",),
        collective_source_ids=("motiondecode.motion_prior",),
        allowed_dream_types=("replay", "self", "social", "nightmare"),
        allowed_learner_ids=("g1.recovery.awr_in_sample.v1",),
        historical_anchor_hashes=(stable.artifact_hash, retained.artifact_hash),
        boundary_suite_hash=boundary_hash,
        metrics=(
            GrowthMetricSpec(
                metric_id="recovery.naturalness",
                direction=MetricDirection.MAXIMIZE,
                primary=True,
                minimum_relative_improvement=0.05,
            ),
        ),
        promotion_profile_hash=canonical_hash(
            {"minimum_learned_gain": 0.05, "strict_replay": True, "no_regressions": True}
        ),
        rollback_policy_hash=retained.artifact_hash,
    )
    trigger_hash = prior_report_hash or _file_hash(motiondecode_pilot_report_path)
    request = DreamPlanRequest(
        body_hash=stable.body_hash,
        parent_policy_hash=retained.artifact_hash,
        trigger_kind="post_practice_recovery_instability",
        trigger_evidence_hashes=(trigger_hash,),
        objectives=("improve_post_kick_naturalness", "retain_kick_and_balance"),
        constraint_hashes=(spec.cost_contract_hash, spec.action_contract_hash),
        practice_snapshot_hashes=((prior_replay_hash,) if prior_replay_hash else ()),
        collective_capsule_hashes=(prior.artifact_hash,),
        historical_anchor_hashes=spec.historical_anchor_hashes,
        boundary_suite_hashes=(boundary_hash,),
        private_holdout_commitment=canonical_hash(
            {"suite": boundary_hash, "round": validation_round, "rows_revealed": False}
        ),
        dream_types=(DreamType.REPLAY, DreamType.SELF, DreamType.SOCIAL, DreamType.NIGHTMARE),
        learner_ids=("g1.recovery.awr_in_sample.v1",),
        budget=DreamBudget(
            max_gpu_seconds=3600.0,
            max_cpu_rollouts=1000,
            max_candidates=6,
            max_wall_seconds=7200.0,
            max_policy_change=0.05,
            max_anchor_drift=0.02,
        ),
    )
    planned = DreamPlanner("simforge.g1_recovery_dream.v1").plan(spec, request)
    atomic_write_json(root / "skill-growth-spec.json", spec.to_dict())
    atomic_write_json(root / "dream-plan-receipt.json", planned.to_dict())

    started = time.monotonic()
    state_root = root / "dream-state"
    with DreamScheduler(
        state_root,
        source_checkout=checkout,
        max_lease_seconds=3600.0,
    ) as scheduler:
        scheduler.submit(planned.campaign)
        lease = scheduler.acquire(
            worker_id="simforge-g1-recovery-awr-0",
            lease_seconds=3600.0,
            campaign_hash=planned.campaign.campaign_hash,
        )
        try:
            report = validation_runner(
                asset_root=asset_root,
                motion_prior_path=motion_prior_path,
                motiondecode_pilot_report_path=motiondecode_pilot_report_path,
                stable_artifact_path=stable_artifact_path,
                recovery_artifact_path=recovery_artifact_path,
                output_dir=root / "worker-evidence",
                source_checkout=checkout,
                device=device,
                seed=seed,
                exploration_replicates=exploration_replicates,
                value_updates=value_updates,
                actor_updates=actor_updates,
                validation_round=validation_round,
            )
            report_path = root / "worker-evidence" / "g1-recovery-awr-report.json"
            report_hash = _file_hash(report_path)
            curriculum = build_g1_recovery_context_curriculum(
                report,
                source_report_hash=report_hash,
            )
            atomic_write_json(root / "context-curriculum.json", curriculum.to_dict())
            gates = _growth_gates(report, curriculum)
            disposition, candidate_hashes = _disposition(
                spec=spec,
                campaign_hash=planned.campaign.campaign_hash,
                report=report,
                report_hash=report_hash,
                curriculum=curriculum,
                gates=gates,
            )
            atomic_write_json(root / "growth-disposition.json", disposition)
            elapsed = time.monotonic() - started
            cpu_rollouts = _reported_rollout_count(report)
            scheduler.record_usage(
                planned.campaign.campaign_hash,
                lease_token=lease.lease_token,
                gpu_seconds=elapsed,
                cpu_rollouts=cpu_rollouts,
                candidates=len(candidate_hashes),
            )
            terminal = scheduler.complete(
                planned.campaign.campaign_hash,
                lease_token=lease.lease_token,
                result_manifest_hash=canonical_hash(disposition),
                candidate_artifact_hashes=candidate_hashes,
            )
        except BaseException:
            current = scheduler.status(planned.campaign.campaign_hash)
            if not current.terminal:
                scheduler.fail(
                    planned.campaign.campaign_hash,
                    lease_token=lease.lease_token,
                    reason="G1 recovery Dream failed closed before terminal evidence",
                )
            raise

    result = {
        "schema_version": "rosclaw.simforge.g1_recovery_dream_cycle.v1",
        "campaign_hash": planned.campaign.campaign_hash,
        "growth_spec_hash": spec.spec_hash,
        "source_report_hash": report_hash,
        "context_curriculum_hash": curriculum.curriculum_hash,
        "result_manifest_hash": canonical_hash(disposition),
        "decision": disposition["decision"],
        "scheduler_status": terminal.to_dict(),
        "gate_results": [item.to_dict() for item in gates],
        "evidence_root": str(root),
        "activation_authorized": False,
        "hardware_authorized": False,
    }
    atomic_write_json(root / "g1-recovery-dream-cycle.json", result)
    return result


def _growth_gates(
    report: Mapping[str, Any],
    curriculum: G1RecoveryContextCurriculum,
) -> tuple[GateResult, ...]:
    checks = _mapping(report.get("checks"), label="AWR checks")
    validation = _mapping(report.get("validation_gate"), label="validation gate")
    candidate_unique = report.get("candidate_artifact_hash") != report.get(
        "retained_recovery_artifact_hash"
    )
    learned_gain = float(validation.get("mean_naturalness_reduction", -math.inf))
    states = {
        GateName.LEARNING: bool(
            checks.get("in_sample_value_converged")
            and checks.get("actor_updated_only_from_elites")
            and checks.get("development_candidate_found")
        ),
        GateName.RETENTION: bool(
            checks.get("retained_parent_exact_before_contact")
            and not any("regress" in str(value) or "lost" in str(value) for value in report.get("blockers", []))
        ),
        GateName.SAFETY: bool(
            checks.get("sim_only_boundary_preserved")
            and report.get("hardware_authorized") is False
        ),
        GateName.APPLICABILITY: curriculum.routing_ready,
        GateName.DARWIN: bool(
            candidate_unique
            and checks.get("independent_strict_validation_passed")
            and math.isfinite(learned_gain)
            and learned_gain >= 0.05
        ),
    }
    values = []
    for name in GateName:
        status = GateStatus.PASS if states[name] else GateStatus.FAIL
        detail = {
            "gate": name.value,
            "passed": states[name],
            "source_report_hash": curriculum.source_report_hash,
            "curriculum_hash": curriculum.curriculum_hash,
            "candidate_unique": candidate_unique,
            "learned_gain": learned_gain if math.isfinite(learned_gain) else None,
        }
        values.append(
            GateResult(
                name=name,
                status=status,
                report_hash=canonical_hash(detail),
                detail=str(detail),
            )
        )
    return tuple(values)


def _disposition(
    *,
    spec: SkillGrowthSpec,
    campaign_hash: str,
    report: Mapping[str, Any],
    report_hash: str,
    curriculum: G1RecoveryContextCurriculum,
    gates: tuple[GateResult, ...],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    retained_hash = str(report["retained_recovery_artifact_hash"])
    candidate_hash = str(report["candidate_artifact_hash"])
    evolution = _evolution(report, gates)
    if candidate_hash != retained_hash:
        darwin = next(item for item in gates if item.name is GateName.DARWIN)
        manifest = ConsolidationManifest(
            skill_growth_spec_hash=spec.spec_hash,
            candidate_artifact_hash=candidate_hash,
            parent_artifact_hash=retained_hash,
            rollback_artifact_hash=retained_hash,
            learned_changes={
                "direct_torque_recovery_actor": candidate_hash,
                "contact_context_curriculum": curriculum.curriculum_hash,
            },
            new_capability_ids=(),
            retained_capability_ids=(
                tuple(spec.capability_ids)
                if all(item.status is GateStatus.PASS for item in gates)
                else ()
            ),
            forgotten_capability_ids=(),
            gate_results=gates,
            darwin_report_hash=str(darwin.report_hash),
            decision=(
                ConsolidationDecision.CONSOLIDATE_SIM
                if all(item.status is GateStatus.PASS for item in gates)
                else ConsolidationDecision.REJECT
            ),
        )
        value = {
            **manifest.to_dict(),
            "campaign_hash": campaign_hash,
            "source_report_hash": report_hash,
            "context_curriculum_hash": curriculum.curriculum_hash,
            "evolution": {"state": evolution.state.value, "history": evolution.history},
            "activation_authorized": False,
        }
        return value, (candidate_hash,)
    value = {
        "schema_version": "rosclaw.simforge.g1_recovery_no_candidate_manifest.v1",
        "campaign_hash": campaign_hash,
        "skill_growth_spec_hash": spec.spec_hash,
        "source_report_hash": report_hash,
        "parent_artifact_hash": retained_hash,
        "candidate_artifact_hash": None,
        "context_curriculum_hash": curriculum.curriculum_hash,
        "gate_results": [item.to_dict() for item in gates],
        "decision": ConsolidationDecision.REJECT.value,
        "reason": "learner produced no development-qualified successor distinct from parent",
        "evolution": {"state": evolution.state.value, "history": evolution.history},
        "active_policy_unchanged": True,
        "activation_authorized": False,
        "hardware_authorized": False,
    }
    return value, ()


def _evolution(report: Mapping[str, Any], gates: tuple[GateResult, ...]) -> EvolutionRun:
    run = EvolutionRun(
        run_id=str(report.get("online_replay_hash", "g1-recovery-dream"))[-64:],
        task_id="g1.football.recovery",
    )
    for state, reason in (
        (EvolutionState.FAILURE_CLUSTERED, "matched recovery outcomes grouped by contact context"),
        (EvolutionState.DIAGNOSED, "mixed-context interference tested with in-sample AWR"),
        (EvolutionState.CANDIDATES_GENERATED, "bounded direct-torque successor search completed"),
        (EvolutionState.SAME_SEED_RETRY, "exact parent and matched exploration replay checked"),
        (EvolutionState.DEVELOPMENT_EVAL, "development trust-region search completed"),
    ):
        run.transition(state, reason=reason)
    if report.get("candidate_artifact_hash") == report.get("retained_recovery_artifact_hash"):
        run.transition(EvolutionState.REJECTED, reason="no development-qualified successor")
        return run
    run.transition(EvolutionState.VALIDATION_EVAL, reason="fresh validation replay completed")
    run.transition(EvolutionState.HOLDOUT_EVAL, reason="sealed holdout replay completed")
    if any(item.status is GateStatus.FAIL for item in gates):
        run.transition(EvolutionState.REJECTED, reason="one or more growth gates failed")
    else:
        run.transition(
            EvolutionState.NEED_MORE_EVIDENCE,
            reason="SIM candidate requires falsification and cross-backend evidence",
        )
    return run


def _reported_rollout_count(report: Mapping[str, Any]) -> int:
    count = len(report.get("collection", []))
    for name in (
        "parent_development",
        "parent_validation",
        "candidate_validation",
    ):
        count += len(report.get(name, []))
    for trust in report.get("trust_runs", []):
        if isinstance(trust, Mapping):
            count += len(trust.get("rollouts", []))
    return max(1, count)


__all__ = [
    "G1RecoveryContextCluster",
    "G1RecoveryContextCurriculum",
    "G1RecoveryContextRoute",
    "build_g1_recovery_context_curriculum",
    "route_g1_recovery_context",
    "run_g1_recovery_dream_cycle",
]
