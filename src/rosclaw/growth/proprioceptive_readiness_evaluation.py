"""Sealed counterfactual evaluation for the G1 strike-readiness gate."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.proprioceptive_expert_router import (
    G1StrikeHandoffFeatures,
    load_g1_proprioceptive_expert_router,
)
from rosclaw.growth.proprioceptive_readiness_gate import (
    load_g1_proprioceptive_readiness_gate,
)

_MISS_PENALTY_M = 2.0


@dataclass(frozen=True)
class G1ReadinessGateEvaluation:
    gate_hash: str
    router_hash: str
    seeds: tuple[int, ...]
    attempted: int
    abstained: int
    attempt_coverage: float
    all_unsafe_states: int
    all_unsafe_abstained: int
    avoidable_abstentions: int
    router_unsafe_attempts: int
    gated_unsafe_attempts: int
    router_precision_hits: int
    gated_precision_hits: int
    gated_mean_penalized_error_m: float
    gated_phase_counts: dict[int, int]
    strict_replay_all: bool
    accepted: bool
    failure_codes: tuple[str, ...]
    report_hash: str
    schema_version: str = "rosclaw.growth.g1_readiness_gate_evaluation.v1"

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            **asdict(self),
            "seeds": list(self.seeds),
            "gated_phase_counts": {
                str(key): item for key, item in sorted(self.gated_phase_counts.items())
            },
            "failure_codes": list(self.failure_codes),
            "evidence_domain": "SIM_ONLY_SEALED_COUNTERFACTUAL_HOLDOUT",
            "abstention_recovery_physically_validated": False,
            "promotion_truth_allowed": False,
            "activation_authorized": False,
            "hardware_authorized": False,
        }
        if not include_hash:
            value.pop("report_hash")
        return value


@dataclass(frozen=True)
class _Outcome:
    features: G1StrikeHandoffFeatures
    safe: bool
    error_m: float
    precision_hit: bool


def evaluate_g1_proprioceptive_readiness_holdout(
    *,
    evidence_paths: tuple[Path, ...],
    router_path: Path,
    gate_path: Path,
    output_path: Path,
    source_checkout: Path,
) -> G1ReadinessGateEvaluation:
    """Evaluate decisions against all three unobserved expert outcomes per seed."""

    router = load_g1_proprioceptive_expert_router(router_path)
    gate = load_g1_proprioceptive_readiness_gate(gate_path)
    if gate.router_hash != router.router_hash or gate.body_hash != router.body_hash:
        raise ValueError("readiness holdout router/gate binding is invalid")
    if not gate.accepted:
        raise ValueError("readiness holdout requires an accepted development gate")
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("readiness evaluation evidence must be outside the source checkout")
    if output.exists():
        raise FileExistsError("readiness evaluation output already exists")
    if len(evidence_paths) < 24 or len(evidence_paths) % len(gate.expert_phases):
        raise ValueError("readiness holdout requires at least eight fully paired seeds")

    outcomes: dict[int, dict[int, _Outcome]] = {}
    body_hashes: set[str] = set()
    context_hashes: set[str] = set()
    replay_values: list[bool] = []
    for raw_path in evidence_paths:
        path = raw_path.expanduser().resolve()
        evidence = json.loads(path.read_text(encoding="utf-8"))
        trajectory = Path(str(evidence.get("trajectory_path", ""))).resolve()
        if not trajectory.is_file() or evidence.get("trajectory_hash") != _file_hash(trajectory):
            raise ValueError("readiness holdout trajectory binding is invalid")
        replay_values.append(evidence.get("strict_replay") is True)
        body_hashes.add(str(evidence.get("body_hash", "")))
        flow = dict(evidence.get("flow_config", {}))
        sonic = dict(evidence.get("sonic_runup_config", {}))
        runup = dict(evidence.get("runup_config", {}))
        goal = dict(evidence.get("goal_spec", {}))
        for mapping in (flow, sonic, runup, goal):
            mapping.pop("schema_version", None)
        seed = int(sonic.pop("planner_seed", -1))
        result = dict(evidence.get("result", {}))
        phase = int(result.get("selected_kick_phase_start_frame", -1))
        if (
            seed < 0
            or seed in gate.development_seeds
            or phase not in gate.expert_phases
            or phase in outcomes.setdefault(seed, {})
        ):
            raise ValueError("readiness holdout seed/phase is invalid, leaked, or duplicated")
        if result.get("proprioceptive_router_executed") is True:
            raise ValueError("readiness counterfactual probes must execute fixed experts")
        if int(flow.get("kick_phase_start_frame", -1)) != phase:
            raise ValueError("readiness counterfactual declared and executed phases disagree")
        for key in (
            "kick_phase_start_frame",
            "contextual_phase_yaw_threshold_rad",
            "contextual_high_yaw_kick_phase_start_frame",
            "contextual_phase_calibration_hash",
            "proprioceptive_router_hash",
        ):
            flow.pop(key, None)
        context_hashes.add(
            canonical_hash(
                {
                    "flow_config": flow,
                    "sonic_runup_config": sonic,
                    "runup_config": runup,
                    "goal_spec": goal,
                }
            )
        )
        safe = _safe(result)
        crossed = result.get("goal_crossed") is True
        raw_error = result.get("goal_plane_target_error_m")
        error = (
            float(raw_error)
            if crossed
            and isinstance(raw_error, (int, float))
            and math.isfinite(float(raw_error))
            else _MISS_PENALTY_M
        )
        outcomes[seed][phase] = _Outcome(
            features=_trajectory_features(trajectory),
            safe=safe,
            error_m=error,
            precision_hit=bool(safe and crossed and error <= float(result["precision_radius_m"])),
        )

    if body_hashes != {gate.body_hash}:
        raise ValueError("readiness holdout Body binding is invalid")
    if len(context_hashes) != 1:
        raise ValueError("readiness holdout experiment contexts disagree")
    phases = gate.expert_phases
    seeds = tuple(sorted(outcomes))
    if len(seeds) < 8 or any(set(outcomes[seed]) != set(phases) for seed in seeds):
        raise ValueError("readiness holdout lacks a complete expert counterfactual set")

    attempted = 0
    abstained = 0
    all_unsafe_states = 0
    all_unsafe_abstained = 0
    avoidable_abstentions = 0
    router_unsafe = 0
    gated_unsafe = 0
    router_hits = 0
    gated_hits = 0
    gated_errors: list[float] = []
    phase_counts = dict.fromkeys(phases, 0)
    for seed in seeds:
        vectors = [outcomes[seed][phase].features for phase in phases]
        if any(vectors[0].vector != item.vector for item in vectors[1:]):
            raise ValueError(f"readiness holdout handoff features differ for seed {seed}")
        router_phase = router.select(vectors[0]).phase_start_frame
        router_outcome = outcomes[seed][router_phase]
        router_unsafe += int(not router_outcome.safe)
        router_hits += int(router_outcome.precision_hit)
        all_unsafe = not any(outcomes[seed][phase].safe for phase in phases)
        all_unsafe_states += int(all_unsafe)
        decision = gate.decide(vectors[0], router)
        if decision.abstained:
            abstained += 1
            all_unsafe_abstained += int(all_unsafe)
            avoidable_abstentions += int(not all_unsafe)
            continue
        attempted += 1
        assert decision.selected_phase_start_frame is not None
        selected = outcomes[seed][decision.selected_phase_start_frame]
        gated_unsafe += int(not selected.safe)
        gated_hits += int(selected.precision_hit)
        gated_errors.append(selected.error_m)
        phase_counts[decision.selected_phase_start_frame] += 1

    strict = all(replay_values)
    coverage = attempted / len(seeds)
    failures: list[str] = []
    if not strict:
        failures.append("STRICT_REPLAY_FAILED")
    if gated_unsafe:
        failures.append("GATED_UNSAFE_ATTEMPT")
    if all_unsafe_abstained != all_unsafe_states:
        failures.append("ALL_UNSAFE_STATE_NOT_ABSTAINED")
    if coverage < gate.minimum_attempt_coverage:
        failures.append("SEALED_ATTEMPT_COVERAGE_TOO_LOW")
    if attempted and gated_hits / attempted < 0.20:
        failures.append("SEALED_PRECISION_YIELD_TOO_LOW")
    accepted = not failures
    unsigned: dict[str, Any] = {
        "schema_version": "rosclaw.growth.g1_readiness_gate_evaluation.v1",
        "gate_hash": gate.gate_hash,
        "router_hash": router.router_hash,
        "seeds": list(seeds),
        "attempted": attempted,
        "abstained": abstained,
        "attempt_coverage": coverage,
        "all_unsafe_states": all_unsafe_states,
        "all_unsafe_abstained": all_unsafe_abstained,
        "avoidable_abstentions": avoidable_abstentions,
        "router_unsafe_attempts": router_unsafe,
        "gated_unsafe_attempts": gated_unsafe,
        "router_precision_hits": router_hits,
        "gated_precision_hits": gated_hits,
        "gated_mean_penalized_error_m": (
            sum(gated_errors) / len(gated_errors) if gated_errors else _MISS_PENALTY_M
        ),
        "gated_phase_counts": {str(key): item for key, item in sorted(phase_counts.items())},
        "strict_replay_all": strict,
        "accepted": accepted,
        "failure_codes": failures,
        "evidence_domain": "SIM_ONLY_SEALED_COUNTERFACTUAL_HOLDOUT",
        "abstention_recovery_physically_validated": False,
        "promotion_truth_allowed": False,
        "activation_authorized": False,
        "hardware_authorized": False,
    }
    report = G1ReadinessGateEvaluation(
        gate_hash=gate.gate_hash,
        router_hash=router.router_hash,
        seeds=seeds,
        attempted=attempted,
        abstained=abstained,
        attempt_coverage=coverage,
        all_unsafe_states=all_unsafe_states,
        all_unsafe_abstained=all_unsafe_abstained,
        avoidable_abstentions=avoidable_abstentions,
        router_unsafe_attempts=router_unsafe,
        gated_unsafe_attempts=gated_unsafe,
        router_precision_hits=router_hits,
        gated_precision_hits=gated_hits,
        gated_mean_penalized_error_m=unsigned["gated_mean_penalized_error_m"],
        gated_phase_counts=phase_counts,
        strict_replay_all=strict,
        accepted=accepted,
        failure_codes=tuple(failures),
        report_hash=canonical_hash(unsigned),
    )
    if report.report_hash != canonical_hash(report.to_dict(include_hash=False)):
        raise RuntimeError("readiness evaluation report hash construction diverged")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _trajectory_features(path: Path) -> G1StrikeHandoffFeatures:
    from rosclaw.growth.proprioceptive_expert_router import _trajectory_features as load

    return load(path)


def _safe(result: dict[str, Any]) -> bool:
    return bool(
        result.get("finite_state") is True
        and result.get("post_kick_fall") is False
        and result.get("joint_limit_violation") is False
        and result.get("torque_limit_violation") is False
    )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


__all__ = ["G1ReadinessGateEvaluation", "evaluate_g1_proprioceptive_readiness_holdout"]
