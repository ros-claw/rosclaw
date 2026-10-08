"""Task-neutral research diagnosis; never a training or promotion permit.

Receipt authenticity and worker execution remain outside these pure contracts.
Execution budgets reuse DreamBudget; this module does not introduce another
scheduler, actuator path, or automatic candidate activation mechanism.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from enum import StrEnum

from rosclaw.dream.contracts import DreamBudget
from rosclaw.dream.control import DreamBudgetUsage
from rosclaw.feedback.contracts import canonical_hash


def _hash(value: str) -> None:
    if type(value) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ValueError("content-bound SHA-256 required")


def _name(value: str) -> None:
    if type(value) is not str or re.fullmatch(r"[a-z][a-z0-9_.-]{0,127}", value) is None:
        raise ValueError("normalized research identity required")


@dataclass(frozen=True)
class ResearchHypothesis:
    hypothesis_id: str
    claim: str
    prediction: str
    fixed_conditions_hash: str
    evaluation_contract_hash: str

    def __post_init__(self) -> None:
        _name(self.hypothesis_id)
        for value in (self.claim, self.prediction):
            if type(value) is not str or not value.strip() or len(value) > 4096:
                raise ValueError("bounded falsifiable claim and prediction required")
        _hash(self.fixed_conditions_hash)
        _hash(self.evaluation_contract_hash)


@dataclass(frozen=True)
class ExperimentFamily:
    family_id: str
    mechanism_contract_hash: str

    def __post_init__(self) -> None:
        _name(self.family_id)
        _hash(self.mechanism_contract_hash)


@dataclass(frozen=True)
class ResearchBudget:
    execution: DreamBudget
    maximum_experiments: int

    def __post_init__(self) -> None:
        if not isinstance(self.execution, DreamBudget):
            raise ValueError("reuse the bounded Dream execution budget")
        self.execution.__post_init__()
        if type(self.maximum_experiments) is not int or not 1 <= self.maximum_experiments <= 10000:
            raise ValueError("bounded experiment count required")


@dataclass(frozen=True)
class ResearchCampaign:
    campaign_id: str
    hypothesis: ResearchHypothesis
    family: ExperimentFamily
    budget: ResearchBudget
    train_snapshot_hash: str | None
    development_snapshot_hash: str | None
    retention_snapshot_hash: str | None
    sealed_commitment: str | None
    feedback_teacher_contract_hash: str | None = None

    def __post_init__(self) -> None:
        _name(self.campaign_id)
        for value, expected in (
            (self.hypothesis, ResearchHypothesis),
            (self.family, ExperimentFamily),
            (self.budget, ResearchBudget),
        ):
            if not isinstance(value, expected):
                raise ValueError("typed research hypothesis, family and budget required")
            value.__post_init__()
        hashes = (
            self.train_snapshot_hash,
            self.development_snapshot_hash,
            self.retention_snapshot_hash,
            self.sealed_commitment,
        )
        known_hashes = tuple(value for value in hashes if value is not None)
        for value in known_hashes:
            _hash(value)
        if len(set(known_hashes)) != len(known_hashes):
            raise ValueError("training, development, retention and sealed identities must differ")
        if self.feedback_teacher_contract_hash is not None:
            _hash(self.feedback_teacher_contract_hash)

    @property
    def banks_bound(self) -> bool:
        """Unknown banks stay unknown; planning must not fabricate commitments."""
        return all(
            value is not None
            for value in (
                self.train_snapshot_hash,
                self.development_snapshot_hash,
                self.retention_snapshot_hash,
                self.sealed_commitment,
            )
        )

    @property
    def campaign_hash(self) -> str:
        payload = asdict(self)
        # Existing campaigns did not require a feedback teacher. Preserve their
        # identities; opting in creates a new, explicitly bound campaign.
        if self.feedback_teacher_contract_hash is None:
            payload.pop("feedback_teacher_contract_hash")
        return canonical_hash(payload)


def research_budget_available(
    campaign: ResearchCampaign,
    *,
    usage: DreamBudgetUsage,
    requested: DreamBudgetUsage,
    experiments_reserved: int,
    elapsed_wall_seconds: float,
    requested_wall_seconds: float,
) -> bool:
    """Pure preflight only; concurrent reservations belong to DreamScheduler.

    Failed experiments must remain accounted. A True return is not a lease,
    training approval, drift acceptance or permission to access a sealed bank.
    """
    campaign.__post_init__()
    if not isinstance(usage, DreamBudgetUsage) or not isinstance(requested, DreamBudgetUsage):
        raise ValueError("typed Dream usage and proposed resource use required")
    usage.__post_init__()
    requested.__post_init__()
    if type(experiments_reserved) is not int or experiments_reserved < 0:
        raise ValueError("actual nonnegative reservation count required")
    for value in (elapsed_wall_seconds, requested_wall_seconds):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("finite nonnegative wall-clock accounting required")
    limits = campaign.budget.execution
    return bool(
        experiments_reserved < campaign.budget.maximum_experiments
        and usage.cpu_rollouts + requested.cpu_rollouts <= limits.max_cpu_rollouts
        and usage.gpu_seconds + requested.gpu_seconds <= limits.max_gpu_seconds
        and usage.candidates + requested.candidates <= limits.max_candidates
        and elapsed_wall_seconds + requested_wall_seconds <= limits.max_wall_seconds
    )


@dataclass(frozen=True)
class ResearchObservation:
    """Authenticated upstream boolean judgments, not inferred pass from absence."""

    oracle_pass: bool | None = None
    imitation_pass: bool | None = None
    closed_loop_pass: bool | None = None
    development_pass: bool | None = None
    blind_pass: bool | None = None
    retention_pass: bool | None = None
    individual_skill_pass: bool | None = None
    evidence_hashes: tuple[str, ...] = ()
    # Independent assay controls: can this declared search/evaluation setup
    # recover known feasible examples? Unknown is not an implicit pass.
    # Kept after evidence_hashes to preserve existing positional construction.
    oracle_assay_pass: bool | None = None
    feedback_teacher_pass: bool | None = None

    def __post_init__(self) -> None:
        values = tuple(v for k, v in asdict(self).items() if k != "evidence_hashes")
        if any(v is not None and type(v) is not bool for v in values):
            raise ValueError("explicit boolean or unknown research judgments required")
        if type(self.evidence_hashes) is not tuple or len(set(self.evidence_hashes)) != len(
            self.evidence_hashes
        ):
            raise ValueError("unique immutable evidence identities required")
        for value in self.evidence_hashes:
            _hash(value)
        if any(v is not None for v in values) and not self.evidence_hashes:
            raise ValueError("research judgments require upstream evidence")


@dataclass(frozen=True)
class FamilyExperiment:
    experiment_hash: str
    mechanism_contract_hash: str
    evaluation_contract_hash: str
    blind_bank_commitment: str | None
    blind_improvement: float | None

    def __post_init__(self) -> None:
        for value in (
            self.experiment_hash,
            self.mechanism_contract_hash,
            self.evaluation_contract_hash,
        ):
            _hash(value)
        if self.blind_bank_commitment is not None:
            _hash(self.blind_bank_commitment)
        if (self.blind_improvement is None) != (self.blind_bank_commitment is None):
            raise ValueError("blind improvement requires its consumed bank commitment")
        if self.blind_improvement is not None and (
            type(self.blind_improvement) not in (int, float)
            or not math.isfinite(self.blind_improvement)
        ):
            raise ValueError("finite actual improvement required")


@dataclass(frozen=True)
class PlateauSignal:
    stopped: bool
    experiment_hashes: tuple[str, ...]
    reason: str


def detect_plateau(
    campaign: ResearchCampaign, experiments: tuple[FamilyExperiment, ...]
) -> PlateauSignal:
    """Three consecutive known non-improvements; missing blind data breaks a streak.

    Input order must be authenticated journal order. Family display-name changes
    do not reset the pinned mechanism identity. This is not semantic detection
    of callers disguising the same algorithm under a different mechanism hash.
    """
    if type(experiments) is not tuple or any(
        not isinstance(r, FamilyExperiment) for r in experiments
    ):
        raise ValueError("ordered immutable experiment history required")
    seen: set[str] = set()
    banks: set[str] = set()
    streak: list[str] = []
    stopped: tuple[str, ...] = ()
    for row in experiments:
        row.__post_init__()
        if row.experiment_hash in seen:
            raise ValueError("duplicate experiment cannot count toward plateau")
        seen.add(row.experiment_hash)
        if row.blind_bank_commitment is not None:
            if row.blind_bank_commitment in banks:
                raise ValueError("a consumed blind bank cannot be reused")
            banks.add(row.blind_bank_commitment)
        if row.mechanism_contract_hash != campaign.family.mechanism_contract_hash:
            continue
        if row.evaluation_contract_hash != campaign.hypothesis.evaluation_contract_hash:
            raise ValueError("same-family evaluation drift cannot reset or establish plateau")
        if stopped:
            continue  # a later claimed gain cannot erase a prior STOP_FAMILY
        if row.blind_improvement is None or row.blind_improvement > 0:
            streak.clear()
        else:
            streak.append(row.experiment_hash)
            if len(streak) == 3:
                stopped = tuple(streak)
    return PlateauSignal(
        bool(stopped), stopped, "three_blind_non_improvements" if stopped else "not_established"
    )


class ResearchRoute(StrEnum):
    NEED_EVIDENCE = "NEED_EVIDENCE"
    STOP_FAMILY = "STOP_FAMILY"
    SEARCH_OR_ASSAY = "SEARCH_OR_ASSAY"
    ACTION_SPACE_OR_ENVIRONMENT = "ACTION_SPACE_OR_ENVIRONMENT"
    TEACHER_OR_CONTROL = "TEACHER_OR_CONTROL"
    REPRESENTATION_OR_DAGGER = "REPRESENTATION_OR_DAGGER"
    CREDIT_OR_ON_POLICY = "CREDIT_OR_ON_POLICY"
    COVERAGE_OR_CURRICULUM = "COVERAGE_OR_CURRICULUM"
    STABILITY_PLASTICITY = "STABILITY_PLASTICITY"
    TEAM_INTEGRATION = "TEAM_INTEGRATION"


@dataclass(frozen=True)
class PivotDecision:
    campaign_hash: str
    route: ResearchRoute
    reason: str
    evidence_hashes: tuple[str, ...]

    @property
    def training_authorized(self) -> bool:
        return False

    @property
    def promotion_authorized(self) -> bool:
        return False


def route_research(
    campaign: ResearchCampaign,
    observation: ResearchObservation,
    experiments: tuple[FamilyExperiment, ...] = (),
) -> PivotDecision:
    campaign.__post_init__()
    observation.__post_init__()
    plateau = detect_plateau(campaign, experiments)
    route, reason = ResearchRoute.NEED_EVIDENCE, "missing_next_stage_evidence"
    if plateau.stopped:
        route, reason = ResearchRoute.STOP_FAMILY, plateau.reason
    elif observation.retention_pass is False:
        route, reason = ResearchRoute.STABILITY_PLASTICITY, "retention_failed"
    elif observation.oracle_assay_pass is False:
        route, reason = ResearchRoute.SEARCH_OR_ASSAY, "oracle_assay_controls_failed"
    elif observation.oracle_pass is False:
        if observation.oracle_assay_pass is True:
            route, reason = (
                ResearchRoute.ACTION_SPACE_OR_ENVIRONMENT,
                "oracle_failed_within_declared_budget",
            )
        else:
            reason = "oracle_failure_requires_assay_controls"
    elif observation.oracle_pass is True:
        if observation.feedback_teacher_pass is False:
            route, reason = ResearchRoute.TEACHER_OR_CONTROL, "feedback_teacher_failed"
        elif (
            campaign.feedback_teacher_contract_hash is not None
            and observation.feedback_teacher_pass is not True
        ):
            reason = "feedback_teacher_evidence_required_before_student"
        elif observation.imitation_pass is False:
            route, reason = ResearchRoute.REPRESENTATION_OR_DAGGER, "imitation_failed"
        elif observation.imitation_pass is True:
            if observation.closed_loop_pass is False:
                route, reason = ResearchRoute.CREDIT_OR_ON_POLICY, "closed_loop_failed"
            elif observation.closed_loop_pass is True:
                if observation.development_pass is True and observation.blind_pass is False:
                    route, reason = ResearchRoute.COVERAGE_OR_CURRICULUM, "development_blind_gap"
                elif campaign.banks_bound and all(
                    v is True
                    for v in (
                        observation.development_pass,
                        observation.blind_pass,
                        observation.retention_pass,
                        observation.individual_skill_pass,
                    )
                ):
                    route, reason = ResearchRoute.TEAM_INTEGRATION, "individual_gates_passed"
    return PivotDecision(
        campaign.campaign_hash,
        route,
        reason,
        observation.evidence_hashes + plateau.experiment_hashes,
    )
