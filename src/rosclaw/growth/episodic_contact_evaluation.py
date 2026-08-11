"""Fail-closed multi-context evaluation for episodic G1 contact memory."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.episodic_contact_memory import (
    g1_episodic_contact_context_hash,
    load_g1_episodic_contact_memory,
)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _load_evidence(path: Path) -> dict[str, Any]:
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if evidence.get("strict_replay") is not True:
        raise ValueError("episodic contact evaluation requires strict replay evidence")
    trajectory = Path(str(evidence.get("trajectory_path", ""))).resolve()
    if not trajectory.is_file() or evidence.get("trajectory_hash") != _file_hash(trajectory):
        raise ValueError("episodic contact evaluation trajectory binding is invalid")
    return evidence


def _context_hash(evidence: dict[str, Any]) -> str:
    return g1_episodic_contact_context_hash(
        flow_config=dict(evidence.get("flow_config", {})),
        goal_spec=dict(evidence.get("goal_spec", {})),
        runup_config=dict(evidence.get("runup_config", {})),
        sonic_runup_config=(
            None
            if evidence.get("sonic_runup_config") is None
            else dict(evidence["sonic_runup_config"])
        ),
        approach_strike_candidate_hash=evidence.get("approach_strike_candidate_hash"),
    )


def _physically_safe(result: dict[str, Any]) -> bool:
    authority = float(result.get("contact_task_authority_scale_min", 0.0))
    return bool(
        result.get("kick_contact_observed") is True
        and result.get("perceptual_continuity_passed") is True
        and result.get("post_kick_fall") is False
        and result.get("joint_limit_violation") is False
        and result.get("torque_limit_violation") is False
        and result.get("actuator_saturation") is False
        and result.get("torque_authority_projection_qualified") is True
        and math.isfinite(authority)
        and authority >= 0.95
        and float(result.get("post_contact_backward_displacement_m", math.inf)) <= 0.02
    )


@dataclass(frozen=True)
class G1EpisodicContactCaseEvaluation:
    planner_seed_audit_label: int
    baseline_evidence_hash: str
    candidate_evidence_hash: str
    baseline_goal_plane_target_error_m: float
    candidate_goal_plane_target_error_m: float
    absolute_error_improvement_m: float
    candidate_active_frames: int
    selected_context_seed_label: int
    peak_context_distance: float
    candidate_physically_safe: bool
    precision_improved: bool
    schema_version: str = "rosclaw.growth.g1_episodic_contact_case_evaluation.v1"

    def __post_init__(self) -> None:
        if self.planner_seed_audit_label < 0 or self.selected_context_seed_label < 0:
            raise ValueError("episodic contact case seed labels must be non-negative")
        if any(
            not value.startswith("sha256:")
            for value in (self.baseline_evidence_hash, self.candidate_evidence_hash)
        ):
            raise ValueError("episodic contact case hashes must be SHA-256")
        values = (
            self.baseline_goal_plane_target_error_m,
            self.candidate_goal_plane_target_error_m,
            self.absolute_error_improvement_m,
            self.peak_context_distance,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("episodic contact case metrics must be finite")
        if self.candidate_active_frames < 1:
            raise ValueError("episodic contact case did not activate the memory")


@dataclass(frozen=True)
class G1EpisodicContactEvaluation:
    memory_hash: str
    memory_artifact_hash: str
    evaluation_implementation_hash: str
    body_hash: str
    implementation_hash: str
    cases: tuple[G1EpisodicContactCaseEvaluation, ...]
    supported_prototype_count: int
    active_context_coverage_complete: bool
    stability_anchor_evidence_hash: str
    stability_anchor_seed_audit_label: int
    stability_anchor_out_of_support_frames: int
    stability_anchor_preserved: bool
    mean_baseline_error_m: float
    mean_candidate_error_m: float
    mean_absolute_error_improvement_m: float
    improved_context_count: int
    physically_safe_context_count: int
    development_breakthrough: bool
    verdict: str
    activation_ceiling: str = "SIM_ONLY"
    sealed_generalization_evidence: bool = False
    promotion_authorized: bool = False
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.growth.g1_episodic_contact_evaluation.v1"

    def __post_init__(self) -> None:
        hashes = (
            self.memory_hash,
            self.memory_artifact_hash,
            self.evaluation_implementation_hash,
            self.body_hash,
            self.implementation_hash,
            self.stability_anchor_evidence_hash,
        )
        if any(not value.startswith("sha256:") for value in hashes):
            raise ValueError("episodic contact evaluation hashes must be SHA-256")
        if not self.cases:
            raise ValueError("episodic contact evaluation requires an active context")
        if len({item.planner_seed_audit_label for item in self.cases}) != len(self.cases):
            raise ValueError("episodic contact evaluation contexts must be distinct")
        if self.supported_prototype_count < len(self.cases):
            raise ValueError("episodic contact active coverage exceeds memory support")
        values = (
            self.mean_baseline_error_m,
            self.mean_candidate_error_m,
            self.mean_absolute_error_improvement_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("episodic contact aggregate metrics must be finite")
        if self.verdict not in {"DEVELOPMENT", "REJECTED"}:
            raise ValueError("episodic contact evaluation verdict is invalid")
        if self.development_breakthrough != (self.verdict == "DEVELOPMENT"):
            raise ValueError("episodic contact evaluation verdict disagrees with gates")
        if (
            self.activation_ceiling != "SIM_ONLY"
            or self.sealed_generalization_evidence
            or self.promotion_authorized
            or self.hardware_authorized
        ):
            raise ValueError("episodic contact evaluation must remain unpromoted SIM_ONLY")

    @property
    def evaluation_hash(self) -> str:
        return canonical_hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            **asdict(self),
            "cases": [asdict(item) for item in self.cases],
        }
        if include_hash:
            value["evaluation_hash"] = self.evaluation_hash
        return value


def evaluate_g1_episodic_contact_memory(
    *,
    memory_path: Path,
    baseline_evidence_paths: tuple[Path, ...],
    candidate_evidence_paths: tuple[Path, ...],
    stability_anchor_evidence_path: Path,
    output_path: Path,
    source_checkout: Path,
) -> G1EpisodicContactEvaluation:
    """Evaluate every supported state island plus one rejected state anchor."""

    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("episodic contact evaluation must remain outside source checkout")
    if output.exists():
        raise FileExistsError("episodic contact evaluation output already exists")
    if len(baseline_evidence_paths) != len(candidate_evidence_paths):
        raise ValueError("episodic contact baseline/candidate counts disagree")
    memory_file = memory_path.expanduser().resolve()
    memory = load_g1_episodic_contact_memory(memory_file)
    expected_seeds = {item.planner_seed_audit_label for item in memory.prototypes}
    cases: list[G1EpisodicContactCaseEvaluation] = []
    used_paths: set[Path] = set()
    for baseline_raw, candidate_raw in zip(
        baseline_evidence_paths, candidate_evidence_paths, strict=True
    ):
        baseline_path = baseline_raw.expanduser().resolve()
        candidate_path = candidate_raw.expanduser().resolve()
        if (
            baseline_path == candidate_path
            or baseline_path in used_paths
            or candidate_path in used_paths
        ):
            raise ValueError("episodic contact evaluation evidence paths must be distinct")
        used_paths.update((baseline_path, candidate_path))
        baseline = _load_evidence(baseline_path)
        candidate = _load_evidence(candidate_path)
        for evidence in (baseline, candidate):
            if evidence.get("body_hash") != memory.body_hash:
                raise ValueError("episodic contact evaluation Body hash mismatch")
            if evidence.get("implementation_hash") != memory.implementation_hash:
                raise ValueError("episodic contact evaluation implementation hash mismatch")
            if _context_hash(evidence) != memory.experiment_context_hash:
                raise ValueError("episodic contact evaluation context mismatch")
        if baseline.get("goal_spec") != candidate.get("goal_spec"):
            raise ValueError("episodic contact baseline and candidate goals disagree")
        baseline_result = dict(baseline.get("result", {}))
        candidate_result = dict(candidate.get("result", {}))
        if baseline_result.get("episodic_contact_memory_executed") is not False:
            raise ValueError("episodic contact baseline is contaminated")
        seed = int(dict(candidate.get("sonic_runup_config", {})).get("planner_seed", -1))
        selected = int(candidate_result.get("episodic_contact_memory_selected_seed_label", -1))
        active_frames = int(candidate_result.get("episodic_contact_memory_active_frames", 0))
        if (
            seed not in expected_seeds
            or selected != seed
            or active_frames < 1
            or dict(candidate.get("flow_config", {})).get("episodic_contact_memory_hash")
            != memory.memory_hash
        ):
            raise ValueError("episodic contact candidate did not route the bound state island")
        baseline_error = float(baseline_result.get("goal_plane_target_error_m", math.inf))
        candidate_error = float(candidate_result.get("goal_plane_target_error_m", math.inf))
        precision = float(candidate_result.get("precision_radius_m", 0.0))
        improvement = baseline_error - candidate_error
        if not all(math.isfinite(value) for value in (baseline_error, candidate_error, precision)):
            raise ValueError("episodic contact case errors are invalid")
        cases.append(
            G1EpisodicContactCaseEvaluation(
                planner_seed_audit_label=seed,
                baseline_evidence_hash=_file_hash(baseline_path),
                candidate_evidence_hash=_file_hash(candidate_path),
                baseline_goal_plane_target_error_m=baseline_error,
                candidate_goal_plane_target_error_m=candidate_error,
                absolute_error_improvement_m=improvement,
                candidate_active_frames=active_frames,
                selected_context_seed_label=selected,
                peak_context_distance=float(
                    candidate_result.get("episodic_contact_memory_peak_context_distance", math.inf)
                ),
                candidate_physically_safe=_physically_safe(candidate_result),
                precision_improved=bool(candidate_error <= precision and improvement >= 0.001),
            )
        )
    evaluated_seeds = {item.planner_seed_audit_label for item in cases}
    if not evaluated_seeds.issubset(expected_seeds):
        raise ValueError("episodic contact evaluation used an unsupported active state island")

    anchor_path = stability_anchor_evidence_path.expanduser().resolve()
    if anchor_path in used_paths:
        raise ValueError("episodic contact stability anchor must be distinct")
    anchor = _load_evidence(anchor_path)
    if (
        anchor.get("body_hash") != memory.body_hash
        or anchor.get("implementation_hash") != memory.implementation_hash
        or _context_hash(anchor) != memory.experiment_context_hash
    ):
        raise ValueError("episodic contact stability anchor provenance mismatch")
    anchor_result = dict(anchor.get("result", {}))
    anchor_seed = int(dict(anchor.get("sonic_runup_config", {})).get("planner_seed", -1))
    anchor_outside = int(anchor_result.get("episodic_contact_memory_out_of_support_frames", 0))
    anchor_selected = anchor_result.get("episodic_contact_memory_selected_seed_label")
    anchor_membership_valid = bool(
        anchor_seed in memory.rejected_context_seed_labels
        or (anchor_seed in expected_seeds and anchor_selected == anchor_seed)
    )
    anchor_preserved = bool(
        anchor_membership_valid
        and dict(anchor.get("flow_config", {})).get("episodic_contact_memory_hash")
        == memory.memory_hash
        and anchor_result.get("episodic_contact_memory_executed") is True
        and int(anchor_result.get("episodic_contact_memory_active_frames", -1)) == 0
        and anchor_outside > 0
        and _physically_safe(anchor_result)
    )
    mean_baseline = sum(item.baseline_goal_plane_target_error_m for item in cases) / len(cases)
    mean_candidate = sum(item.candidate_goal_plane_target_error_m for item in cases) / len(cases)
    improved_count = sum(item.precision_improved for item in cases)
    safe_count = sum(item.candidate_physically_safe for item in cases)
    development = bool(
        anchor_preserved
        and safe_count == len(cases)
        and improved_count >= 1
        and mean_baseline - mean_candidate >= 0.001
    )
    evaluation = G1EpisodicContactEvaluation(
        memory_hash=memory.memory_hash,
        memory_artifact_hash=_file_hash(memory_file),
        evaluation_implementation_hash=_file_hash(Path(__file__).resolve()),
        body_hash=memory.body_hash,
        implementation_hash=memory.implementation_hash,
        cases=tuple(sorted(cases, key=lambda item: item.planner_seed_audit_label)),
        supported_prototype_count=len(memory.prototypes),
        active_context_coverage_complete=(evaluated_seeds == expected_seeds),
        stability_anchor_evidence_hash=_file_hash(anchor_path),
        stability_anchor_seed_audit_label=anchor_seed,
        stability_anchor_out_of_support_frames=anchor_outside,
        stability_anchor_preserved=anchor_preserved,
        mean_baseline_error_m=mean_baseline,
        mean_candidate_error_m=mean_candidate,
        mean_absolute_error_improvement_m=mean_baseline - mean_candidate,
        improved_context_count=improved_count,
        physically_safe_context_count=safe_count,
        development_breakthrough=development,
        verdict="DEVELOPMENT" if development else "REJECTED",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evaluation.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evaluation


__all__ = [
    "G1EpisodicContactCaseEvaluation",
    "G1EpisodicContactEvaluation",
    "evaluate_g1_episodic_contact_memory",
]
