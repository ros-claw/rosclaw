"""Sealed paired evaluation for a proprioceptive strike-expert router."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.proprioceptive_expert_router import (
    load_g1_proprioceptive_expert_router,
)

_MISS_PENALTY_M = 2.0


@dataclass(frozen=True)
class G1ProprioceptiveRouterEvaluation:
    router_hash: str
    seeds: tuple[int, ...]
    baseline_mean_penalized_error_m: float
    routed_mean_penalized_error_m: float
    baseline_precision_hits: int
    routed_precision_hits: int
    baseline_misses: int
    routed_misses: int
    baseline_unsafe_episodes: int
    routed_unsafe_episodes: int
    baseline_saturation_steps: int
    routed_saturation_steps: int
    routed_phase_counts: dict[int, int]
    routed_fallback_count: int
    strict_replay_all: bool
    measurable_improvement: bool
    accepted: bool
    failure_codes: tuple[str, ...]
    report_hash: str
    schema_version: str = "rosclaw.growth.g1_proprioceptive_router_evaluation.v1"

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            **asdict(self),
            "seeds": list(self.seeds),
            "routed_phase_counts": {
                str(key): value for key, value in sorted(self.routed_phase_counts.items())
            },
            "failure_codes": list(self.failure_codes),
            "evidence_domain": "SIM_ONLY_SEALED_HOLDOUT",
            "sealed_generalization_evidence": True,
            "promotion_truth_allowed": False,
            "activation_authorized": False,
            "hardware_authorized": False,
        }
        if not include_hash:
            value.pop("report_hash")
        return value


@dataclass(frozen=True)
class _Outcome:
    error_m: float
    precision_hit: bool
    missed: bool
    safe: bool
    saturation_steps: int
    phase: int
    fallback: bool


def evaluate_g1_proprioceptive_router_holdout(
    *,
    baseline_paths: tuple[Path, ...],
    routed_paths: tuple[Path, ...],
    router_path: Path,
    output_path: Path,
    source_checkout: Path,
    minimum_mean_improvement_m: float = 0.05,
) -> G1ProprioceptiveRouterEvaluation:
    router = load_g1_proprioceptive_expert_router(router_path)
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("router evaluation evidence must be outside the source checkout")
    if output.exists():
        raise FileExistsError("router evaluation output already exists")
    if len(baseline_paths) != len(routed_paths) or len(baseline_paths) < 8:
        raise ValueError("router evaluation requires at least eight paired episodes")

    baseline: dict[int, _Outcome] = {}
    routed: dict[int, _Outcome] = {}
    body_hashes: set[str] = set()
    implementation_hashes: set[str] = set()
    context_hashes: set[str] = set()
    replay_values: list[bool] = []
    for expected_routed, paths, target in (
        (False, baseline_paths, baseline),
        (True, routed_paths, routed),
    ):
        for raw_path in paths:
            path = raw_path.expanduser().resolve()
            evidence = json.loads(path.read_text(encoding="utf-8"))
            trajectory = Path(str(evidence.get("trajectory_path", ""))).resolve()
            if not trajectory.is_file() or evidence.get("trajectory_hash") != _file_hash(trajectory):
                raise ValueError("router evaluation trajectory binding is invalid")
            replay_values.append(evidence.get("strict_replay") is True)
            body_hashes.add(str(evidence.get("body_hash", "")))
            implementation_hashes.add(str(evidence.get("implementation_hash", "")))
            flow = dict(evidence.get("flow_config", {}))
            sonic = dict(evidence.get("sonic_runup_config", {}))
            result = dict(evidence.get("result", {}))
            seed = int(sonic.pop("planner_seed", -1))
            if seed < 0 or seed in target or seed in router.development_seeds:
                raise ValueError("router holdout seed is invalid, duplicated, or leaked")
            if result.get("proprioceptive_router_executed") is not expected_routed:
                raise ValueError("router evaluation execution claim is invalid")
            if expected_routed:
                if flow.get("proprioceptive_router_hash") != router.router_hash:
                    raise ValueError("routed evidence is not bound to the router")
            elif (
                flow.get("proprioceptive_router_hash") is not None
                or int(result.get("selected_kick_phase_start_frame", -1))
                != router.baseline_phase
            ):
                raise ValueError("baseline evidence executed a router or wrong phase")
            flow.pop("proprioceptive_router_hash", None)
            context_hashes.add(
                canonical_hash(
                    {
                        "flow_config": flow,
                        "sonic_runup_config": sonic,
                        "runup_config": evidence.get("runup_config"),
                        "goal_spec": evidence.get("goal_spec"),
                    }
                )
            )
            target[seed] = _outcome(result)

    if set(baseline) != set(routed):
        raise ValueError("router evaluation seed pairs disagree")
    if len(body_hashes) != 1 or next(iter(body_hashes)) != router.body_hash:
        raise ValueError("router evaluation Body binding is invalid")
    if len(implementation_hashes) != 1 or len(context_hashes) != 1:
        raise ValueError("router evaluation implementation or context differs")
    seeds = tuple(sorted(baseline))
    baseline_values = [baseline[seed] for seed in seeds]
    routed_values = [routed[seed] for seed in seeds]
    baseline_mean = sum(item.error_m for item in baseline_values) / len(seeds)
    routed_mean = sum(item.error_m for item in routed_values) / len(seeds)
    baseline_hits = sum(item.precision_hit for item in baseline_values)
    routed_hits = sum(item.precision_hit for item in routed_values)
    baseline_misses = sum(item.missed for item in baseline_values)
    routed_misses = sum(item.missed for item in routed_values)
    baseline_unsafe = sum(not item.safe for item in baseline_values)
    routed_unsafe = sum(not item.safe for item in routed_values)
    baseline_saturation = sum(item.saturation_steps for item in baseline_values)
    routed_saturation = sum(item.saturation_steps for item in routed_values)
    strict = all(replay_values)
    measurable = bool(
        routed_mean <= baseline_mean - minimum_mean_improvement_m
        and routed_misses <= baseline_misses
        and routed_saturation < baseline_saturation
    )
    failures: list[str] = []
    if not strict:
        failures.append("STRICT_REPLAY_FAILED")
    if not measurable:
        failures.append("NO_MEASURABLE_HOLDOUT_IMPROVEMENT")
    if routed_hits < baseline_hits:
        failures.append("HOLDOUT_PRECISION_REGRESSION")
    if routed_unsafe:
        failures.append("ROUTED_UNSAFE_EPISODE")
    phase_counts = {
        phase: sum(item.phase == phase for item in routed_values) for phase in router.expert_phases
    }
    unsigned = {
        "schema_version": "rosclaw.growth.g1_proprioceptive_router_evaluation.v1",
        "router_hash": router.router_hash,
        "seeds": list(seeds),
        "baseline_mean_penalized_error_m": baseline_mean,
        "routed_mean_penalized_error_m": routed_mean,
        "baseline_precision_hits": baseline_hits,
        "routed_precision_hits": routed_hits,
        "baseline_misses": baseline_misses,
        "routed_misses": routed_misses,
        "baseline_unsafe_episodes": baseline_unsafe,
        "routed_unsafe_episodes": routed_unsafe,
        "baseline_saturation_steps": baseline_saturation,
        "routed_saturation_steps": routed_saturation,
        "routed_phase_counts": {str(key): value for key, value in sorted(phase_counts.items())},
        "routed_fallback_count": sum(item.fallback for item in routed_values),
        "strict_replay_all": strict,
        "measurable_improvement": measurable,
        "accepted": not failures,
        "failure_codes": failures,
        "evidence_domain": "SIM_ONLY_SEALED_HOLDOUT",
        "sealed_generalization_evidence": True,
        "promotion_truth_allowed": False,
        "activation_authorized": False,
        "hardware_authorized": False,
    }
    report = G1ProprioceptiveRouterEvaluation(
        router_hash=router.router_hash,
        seeds=seeds,
        baseline_mean_penalized_error_m=baseline_mean,
        routed_mean_penalized_error_m=routed_mean,
        baseline_precision_hits=baseline_hits,
        routed_precision_hits=routed_hits,
        baseline_misses=baseline_misses,
        routed_misses=routed_misses,
        baseline_unsafe_episodes=baseline_unsafe,
        routed_unsafe_episodes=routed_unsafe,
        baseline_saturation_steps=baseline_saturation,
        routed_saturation_steps=routed_saturation,
        routed_phase_counts=phase_counts,
        routed_fallback_count=sum(item.fallback for item in routed_values),
        strict_replay_all=strict,
        measurable_improvement=measurable,
        accepted=not failures,
        failure_codes=tuple(failures),
        report_hash=canonical_hash(unsigned),
    )
    if report.report_hash != canonical_hash(report.to_dict(include_hash=False)):
        raise RuntimeError("router evaluation report hash construction diverged")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _outcome(result: dict[str, Any]) -> _Outcome:
    safe = bool(
        result.get("finite_state") is True
        and result.get("post_kick_fall") is False
        and result.get("joint_limit_violation") is False
        and result.get("torque_limit_violation") is False
    )
    crossed = result.get("goal_crossed") is True
    raw_error = result.get("goal_plane_target_error_m")
    error = (
        float(raw_error)
        if crossed and isinstance(raw_error, (int, float)) and math.isfinite(float(raw_error))
        else _MISS_PENALTY_M
    )
    radius = float(result["precision_radius_m"])
    return _Outcome(
        error_m=error,
        precision_hit=safe and crossed and error <= radius,
        missed=not crossed,
        safe=safe,
        saturation_steps=int(result.get("actuator_saturation_steps", 0)),
        phase=int(result["selected_kick_phase_start_frame"]),
        fallback=result.get("proprioceptive_router_fallback") is True,
    )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


__all__ = ["G1ProprioceptiveRouterEvaluation", "evaluate_g1_proprioceptive_router_holdout"]
