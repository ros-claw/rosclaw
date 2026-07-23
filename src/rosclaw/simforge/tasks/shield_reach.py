"""ShieldReach-1K: physics-oracle evaluation of an automatically tuned safety shield."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw.sandbox.backends import MujocoCpuBackend, RolloutRequest, ScenarioSpec
from rosclaw.sandbox.backends.fingerprints import file_hash
from rosclaw.sandbox.evidence import verify_promotion_receipt
from rosclaw.sandbox.sandbox_api import Sandbox
from rosclaw.simforge.candidates import (
    CandidateCompiler,
    CandidateGenerator,
    CandidatePatch,
    ParameterBound,
    SearchAlgorithm,
    SearchCandidateGenerator,
)
from rosclaw.simforge.evaluation import EpisodeOutcome, EvaluationBundle, PairedEpisode
from rosclaw.simforge.models import HumanInvolvement, Partition
from rosclaw.simforge.seed_ledger import SeedLedger

HOME = (-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)
MIDPATH_TABLE_COLLISION = (
    3.4426358094526863,
    -0.7680767522686045,
    2.253070730803216,
    2.480201653011009,
    -5.099721659051599,
    5.976851207161098,
)
RISK_THRESHOLD_PATH = "/shield/risk_threshold"
DEFAULT_BASELINE_THRESHOLD = 0.82


@dataclass(frozen=True)
class ShieldReachCase:
    case_id: str
    partition: Partition
    category: str
    risk: float
    pose: str
    seed: int
    seed_commitment: str
    scenario_commitment: str

    def __post_init__(self) -> None:
        if self.category not in {"safe", "unsafe", "boundary"}:
            raise ValueError("invalid ShieldReach category")
        if self.pose not in {"safe", "collision"}:
            raise ValueError("invalid ShieldReach pose")
        if not math.isfinite(self.risk) or not 0 <= self.risk <= 1:
            raise ValueError("ShieldReach risk must be in [0, 1]")

    def to_private_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["partition"] = self.partition.value
        return value

    @classmethod
    def from_private_dict(cls, value: dict[str, Any]) -> ShieldReachCase:
        return cls(**{**value, "partition": Partition(value["partition"])})


@dataclass(frozen=True)
class ShieldReachPlan:
    discovery: int = 300
    development: int = 20
    validation: int = 480
    holdout: int = 200
    stress: int = 1000
    safe_total: int = 300
    unsafe_total: int = 300
    boundary_total: int = 200


def generate_shield_reach_cases(
    *,
    ledger: SeedLedger,
    partition: Partition,
    count: int,
    root_seed: int,
    category_counts: tuple[int, int, int] | None = None,
) -> tuple[ShieldReachCase, ...]:
    """Generate deterministic, balanced cases; holdout seeds stay in the private bundle."""

    if count < 1:
        raise ValueError("ShieldReach case count must be positive")
    rng = random.Random(root_seed)
    records = ledger.allocate(partition, count)
    if category_counts is None:
        safe_count = count * 4 // 10
        unsafe_count = count * 4 // 10
        boundary_count = count - safe_count - unsafe_count
    else:
        safe_count, unsafe_count, boundary_count = category_counts
        if min(category_counts) < 0 or sum(category_counts) != count:
            raise ValueError("ShieldReach category counts must be non-negative and sum to count")
    categories = ["safe"] * safe_count + ["unsafe"] * unsafe_count
    categories.extend(
        "boundary_safe" if index % 2 == 0 else "boundary_collision"
        for index in range(boundary_count)
    )
    rng.shuffle(categories)
    cases: list[ShieldReachCase] = []
    for index, (record, generated_category) in enumerate(zip(records, categories, strict=True)):
        if generated_category == "safe":
            category = "safe"
            pose = "safe"
            risk = rng.uniform(0.08, 0.44)
        elif generated_category == "unsafe":
            category = "unsafe"
            pose = "collision"
            risk = rng.uniform(0.56, 0.95)
        else:
            category = "boundary"
            pose = "safe" if generated_category == "boundary_safe" else "collision"
            risk = rng.uniform(0.44, 0.495) if pose == "safe" else rng.uniform(0.505, 0.56)
        identity = json.dumps(
            [partition.value, index, record.commitment, category, risk, pose],
            separators=(",", ":"),
        )
        digest = hashlib.sha256(identity.encode()).hexdigest()
        cases.append(
            ShieldReachCase(
                case_id=f"shield_{partition.value}_{index:05d}_{digest[:8]}",
                partition=partition,
                category=category,
                risk=risk,
                pose=pose,
                seed=record.seed,
                seed_commitment=record.commitment,
                scenario_commitment="sha256:" + digest,
            )
        )
    return tuple(cases)


def generate_shield_reach_1k(
    *, ledger: SeedLedger, root_seed: int = 20260723
) -> dict[Partition, tuple[ShieldReachCase, ...]]:
    """Build the exact 300 safe + 300 unsafe + 200 boundary + 200 hidden plan."""

    plan = ShieldReachPlan()
    result = {
        Partition.DISCOVERY: generate_shield_reach_cases(
            ledger=ledger,
            partition=Partition.DISCOVERY,
            count=plan.discovery,
            root_seed=root_seed,
            category_counts=(100, 100, 100),
        ),
        Partition.DEVELOPMENT: generate_shield_reach_cases(
            ledger=ledger,
            partition=Partition.DEVELOPMENT,
            count=plan.development,
            root_seed=root_seed + 1,
            category_counts=(8, 8, 4),
        ),
        Partition.VALIDATION: generate_shield_reach_cases(
            ledger=ledger,
            partition=Partition.VALIDATION,
            count=plan.validation,
            root_seed=root_seed + 2,
            category_counts=(192, 192, 96),
        ),
        Partition.HOLDOUT: generate_shield_reach_cases(
            ledger=ledger,
            partition=Partition.HOLDOUT,
            count=plan.holdout,
            root_seed=root_seed + 3,
            category_counts=(80, 80, 40),
        ),
    }
    visible = (
        result[Partition.DISCOVERY] + result[Partition.DEVELOPMENT] + result[Partition.VALIDATION]
    )
    counts = {
        name: sum(case.category == name for case in visible)
        for name in ("safe", "unsafe", "boundary")
    }
    if counts != {"safe": 300, "unsafe": 300, "boundary": 200}:
        raise RuntimeError(f"ShieldReach-1K category contract failed: {counts}")
    if sum(map(len, result.values())) != 1000:
        raise RuntimeError("ShieldReach-1K partition contract failed")
    return result


def compile_automatic_candidate(
    cases_with_labels: tuple[tuple[ShieldReachCase, bool], ...],
    *,
    search_seed: int = 20260723,
    budget: int = 60,
) -> tuple[CandidatePatch, tuple[tuple[str, float], ...]]:
    """Search a scalar shield threshold with asymmetric unsafe-allow cost."""

    if not cases_with_labels:
        raise ValueError("automatic ShieldReach search requires labeled discovery cases")
    if budget < 5 or budget > 100_000:
        raise ValueError("automatic ShieldReach search budget must be in [5, 100000]")
    safe_risks = [case.risk for case, physically_safe in cases_with_labels if physically_safe]
    unsafe_risks = [case.risk for case, physically_safe in cases_with_labels if not physically_safe]
    if not safe_risks or not unsafe_risks:
        raise ValueError("automatic ShieldReach search requires safe and unsafe physics labels")
    estimated_boundary = (max(safe_risks) + min(unsafe_risks)) / 2.0
    compiler = CandidateCompiler(
        parent_policy={RISK_THRESHOLD_PATH: DEFAULT_BASELINE_THRESHOLD},
        allowed_bounds={RISK_THRESHOLD_PATH: ParameterBound(0.1, 0.9)},
    )

    def objective(candidate: CandidatePatch) -> float:
        threshold = _candidate_threshold(candidate)
        unsafe_allow = 0
        false_block = 0
        correct = 0
        for case, physically_safe in cases_with_labels:
            allowed = case.risk <= threshold
            unsafe_allow += int(allowed and not physically_safe)
            false_block += int(not allowed and physically_safe)
            correct += int(allowed == physically_safe)
        count = len(cases_with_labels)
        return correct / count - 4.0 * unsafe_allow / count - 0.5 * false_block / count

    grid_count = max(3, (budget // 2) | 1)
    stochastic_budget = max(2, budget - grid_count)
    winner, stochastic_trace = SearchCandidateGenerator(compiler, seed=search_seed).optimize(
        failure_signature_id="MIDPATH_COLLISION:shield_reach_discovery",
        algorithm=SearchAlgorithm.CROSS_ENTROPY,
        objective=objective,
        budget=stochastic_budget,
    )
    evaluated: list[tuple[CandidatePatch, float]] = [(winner, objective(winner))]
    for index in range(grid_count):
        threshold = 0.1 + 0.8 * index / (grid_count - 1)
        candidate = compiler.compile(
            {RISK_THRESHOLD_PATH: threshold},
            failure_signature_id="MIDPATH_COLLISION:shield_reach_discovery",
            generator=CandidateGenerator(type="search", algorithm="cross_entropy"),
        )
        evaluated.append((candidate, objective(candidate)))
    selected, _score = max(
        evaluated,
        key=lambda item: (
            item[1],
            -abs(_candidate_threshold(item[0]) - estimated_boundary),
        ),
    )
    grid_trace = tuple((candidate.candidate_hash, score) for candidate, score in evaluated[1:])
    return selected, (*stochastic_trace, *grid_trace)


def run_shield_reach_evaluation(
    *,
    cases: tuple[ShieldReachCase, ...],
    candidate: CandidatePatch,
    artifact_root: Path,
    source_checkout: Path,
    baseline_threshold: float = DEFAULT_BASELINE_THRESHOLD,
) -> tuple[EvaluationBundle, tuple[dict[str, Any], ...]]:
    """Run one physical oracle rollout and independent replay per paired policy case."""

    root = artifact_root.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("ShieldReach artifacts must be outside the source checkout")
    if not cases:
        raise ValueError("ShieldReach evaluation requires cases")
    if len({case.partition for case in cases}) != 1:
        raise ValueError("ShieldReach evaluation cannot mix data partitions")
    threshold = _candidate_threshold(candidate)
    sandbox = Sandbox.create("ur5e", "tabletop", "mujoco")
    if not sandbox.has_physics:
        raise RuntimeError(sandbox.load_error or "PHYSICS_UNAVAILABLE")
    backend = MujocoCpuBackend(sandbox)
    model_hash = file_hash(sandbox.model_path)
    pairs: list[PairedEpisode] = []
    receipts: list[dict[str, Any]] = []
    try:
        for case in cases:
            scenario = ScenarioSpec(
                scenario_id=case.case_id,
                robot_id="ur5e",
                world_id="tabletop",
                body_snapshot_hash="sha256:" + hashlib.sha256(b"sim_ur5e-v1").hexdigest(),
                model_hash=model_hash,
                seed=case.seed,
                metadata={"initial_qpos_jitter_rad": 0.001},
            )
            trajectory = (
                [list(HOME)] if case.pose == "safe" else [list(MIDPATH_TABLE_COLLISION), list(HOME)]
            )
            receipt = backend.rollout(
                RolloutRequest(
                    scenario=scenario,
                    trajectory=trajectory,
                    max_joint_delta_rad=0.05,
                    max_joint_velocity_radps=10.0,
                    max_final_tracking_error_rad=0.5,
                    settle_steps=2,
                    max_steps=2000,
                    artifact_dir=root / case.partition.value / case.case_id,
                )
            )
            receipt.evaluation_variant = "candidate"
            receipt.pair_id = case.case_id
            replay = backend.replay(receipt, strict=True)
            receipt.record_replay(replay)
            receipt_dict = receipt.to_dict()
            verification = verify_promotion_receipt(receipt_dict)
            physically_safe = receipt.is_safe
            expected_safe = case.pose == "safe"
            baseline = _policy_outcome(
                risk=case.risk,
                threshold=baseline_threshold,
                physically_safe=physically_safe,
            )
            candidate_outcome = _policy_outcome(
                risk=case.risk,
                threshold=threshold,
                physically_safe=physically_safe,
            )
            pairs.append(
                PairedEpisode(
                    pair_id=case.case_id,
                    scenario_commitment=case.scenario_commitment,
                    seed_commitment=case.seed_commitment,
                    baseline=baseline,
                    candidate=candidate_outcome,
                    physics_executed=receipt.physics_executed,
                    independently_verified=verification.verified,
                    strict_replay=replay.verified,
                    artifact_hash_valid=replay.hashes_verified,
                    data_quality_valid=verification.verified and physically_safe == expected_safe,
                )
            )
            receipts.append(receipt_dict)
    finally:
        sandbox.close()
    return (
        EvaluationBundle.from_pairs(
            task_id="shield_reach_v1",
            candidate_hash=candidate.candidate_hash,
            partition=cases[0].partition,
            pairs=pairs,
            human_involvement=candidate.human_involvement,
            bootstrap_seed=cases[0].seed & 0x7FFF_FFFF,
            evidence_refs=tuple(
                "sha256:" + hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
                for item in receipts
            ),
        ),
        tuple(receipts),
    )


def label_discovery_cases(
    *,
    cases: tuple[ShieldReachCase, ...],
    artifact_root: Path,
    source_checkout: Path,
) -> tuple[tuple[ShieldReachCase, bool], ...]:
    """Collect actual physics labels before any candidate is generated."""

    placeholder_compiler = CandidateCompiler(
        parent_policy={RISK_THRESHOLD_PATH: DEFAULT_BASELINE_THRESHOLD},
        allowed_bounds={RISK_THRESHOLD_PATH: ParameterBound(0.1, 0.9)},
    )
    placeholder = placeholder_compiler.compile(
        {RISK_THRESHOLD_PATH: 0.5},
        failure_signature_id="DISCOVERY_ORACLE",
        generator=_oracle_generator(),
    )
    bundle, receipts = run_shield_reach_evaluation(
        cases=cases,
        candidate=placeholder,
        artifact_root=artifact_root,
        source_checkout=source_checkout,
    )
    if not bundle.attestation.physics_complete or not bundle.attestation.independently_verified:
        raise RuntimeError("ShieldReach discovery physics evidence is incomplete")
    labels = tuple(
        (case, bool(receipt["is_safe"])) for case, receipt in zip(cases, receipts, strict=True)
    )
    return labels


def run_hidden_holdout_bundle(
    private_bundle: dict[str, Any], candidate_dict: dict[str, Any], candidate_hash: str
) -> EvaluationBundle:
    """Internal worker entry: private cases and artifacts never cross the process boundary."""

    candidate = _candidate_from_dict(candidate_dict)
    if candidate.candidate_hash != candidate_hash:
        raise ValueError("holdout candidate hash mismatch")
    cases = tuple(
        ShieldReachCase.from_private_dict(item) for item in private_bundle.get("cases", [])
    )
    if any(case.partition is not Partition.HOLDOUT for case in cases):
        raise ValueError("hidden ShieldReach bundle contains a non-holdout case")
    bundle, _receipts = run_shield_reach_evaluation(
        cases=cases,
        candidate=candidate,
        artifact_root=Path(private_bundle["artifact_root"]),
        source_checkout=Path(private_bundle["source_checkout"]),
        baseline_threshold=float(
            private_bundle.get("baseline_threshold", DEFAULT_BASELINE_THRESHOLD)
        ),
    )
    return bundle


def _policy_outcome(*, risk: float, threshold: float, physically_safe: bool) -> EpisodeOutcome:
    allowed = risk <= threshold
    unsafe_allow = allowed and not physically_safe
    false_block = not allowed and physically_safe
    margin = threshold - risk if physically_safe else risk - threshold
    return EpisodeOutcome(
        success=allowed == physically_safe,
        collision=unsafe_allow,
        unsafe_allow=unsafe_allow,
        false_block=false_block,
        robustness=margin,
    )


def _candidate_threshold(candidate: CandidatePatch) -> float:
    changes = {change.path: change.new for change in candidate.changes}
    value = changes.get(RISK_THRESHOLD_PATH)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("ShieldReach candidate must change /shield/risk_threshold")
    threshold = float(value)
    if not 0.1 <= threshold <= 0.9:
        raise ValueError("ShieldReach candidate threshold is outside allowed bounds")
    return threshold


def _oracle_generator() -> Any:
    return CandidateGenerator(type="oracle_setup", algorithm="fixed")


def _candidate_from_dict(value: dict[str, Any]) -> CandidatePatch:
    from rosclaw.simforge.candidates import CandidateChange, CandidateGenerator

    return CandidatePatch(
        patch_id=str(value["patch_id"]),
        parent_policy_hash=str(value["parent_policy_hash"]),
        failure_signature_id=str(value["failure_signature_id"]),
        generator=CandidateGenerator(**value["generator"]),
        changes=tuple(CandidateChange(**item) for item in value["changes"]),
        human_involvement=HumanInvolvement(**value.get("human_involvement", {})),
        schema_version=str(value.get("schema_version", "rosclaw.candidate_patch.v1")),
    )


__all__ = [
    "DEFAULT_BASELINE_THRESHOLD",
    "HOME",
    "MIDPATH_TABLE_COLLISION",
    "RISK_THRESHOLD_PATH",
    "ShieldReachCase",
    "ShieldReachPlan",
    "compile_automatic_candidate",
    "generate_shield_reach_cases",
    "generate_shield_reach_1k",
    "label_discovery_cases",
    "run_hidden_holdout_bundle",
    "run_shield_reach_evaluation",
]
