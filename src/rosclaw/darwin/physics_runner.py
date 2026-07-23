"""Paired, physics-backed Darwin evaluation for trajectory candidates."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from rosclaw.sandbox.backends import MujocoCpuBackend, RolloutRequest, ScenarioSpec
from rosclaw.sandbox.backends.fingerprints import file_hash
from rosclaw.sandbox.sandbox_api import Sandbox


@dataclass(frozen=True)
class PairedTrajectoryCase:
    scenario: ScenarioSpec
    baseline_trajectory: list[list[float]]
    candidate_trajectory: list[list[float]]


@dataclass(frozen=True)
class PhysicsDarwinResult:
    baseline_metrics: dict[str, float]
    candidate_metrics: dict[str, float]
    per_seed: dict[str, dict[str, dict[str, float]]]
    simulation_receipts: list[dict[str, Any]]
    regression_results: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_metrics": self.baseline_metrics,
            "candidate_metrics": self.candidate_metrics,
            "per_seed": self.per_seed,
            "simulation_receipts": self.simulation_receipts,
            "regression_results": self.regression_results,
        }


def _metrics(outcomes: list[dict[str, Any]]) -> dict[str, float]:
    success = [1.0 if item["is_safe"] else 0.0 for item in outcomes]
    collisions = [1.0 if item["collision"] else 0.0 for item in outcomes]
    durations = [float(item["wall_time_sec"]) for item in outcomes]
    replay = [1.0 if item["replay_verified"] else 0.0 for item in outcomes]
    return {
        "success_rate": statistics.mean(success) if success else 0.0,
        "success_rate_std": statistics.stdev(success) if len(success) > 1 else 0.0,
        "collision_rate": statistics.mean(collisions) if collisions else 0.0,
        "completion_time_mean": statistics.mean(durations) if durations else 0.0,
        "worst_seed_success": min(success) if success else 0.0,
        "replay_success_rate": statistics.mean(replay) if replay else 0.0,
        "receipt_completeness": statistics.mean(
            1.0 if item["receipt_complete"] else 0.0 for item in outcomes
        )
        if outcomes
        else 0.0,
    }


class PhysicsDarwinRunner:
    """Evaluate baseline and candidate in identical model/seed conditions."""

    def __init__(
        self,
        sandbox_factory: Callable[[str, str, str], Sandbox] = Sandbox.create,
    ) -> None:
        self._sandbox_factory = sandbox_factory

    def run(
        self,
        cases: list[PairedTrajectoryCase],
        *,
        artifact_root: Path,
    ) -> PhysicsDarwinResult:
        if not 2 <= len(cases) <= 100:
            raise ValueError("PHYSICS_DARWIN_CASE_COUNT_OUT_OF_RANGE")
        seeds = [case.scenario.seed for case in cases]
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63
            for seed in seeds
        ):
            raise ValueError("PHYSICS_DARWIN_INVALID_SEED")
        if len(set(seeds)) < 2:
            raise ValueError("PHYSICS_DARWIN_REQUIRES_AT_LEAST_TWO_SEEDS")
        if len(set(seeds)) != len(seeds):
            raise ValueError("PHYSICS_DARWIN_REQUIRES_UNIQUE_SEED_CASES")
        for case in cases:
            metadata = case.scenario.metadata
            if not isinstance(metadata, dict):
                raise ValueError("PHYSICS_DARWIN_REQUIRES_SEED_RANDOMIZATION")
            jitter = metadata.get("initial_qpos_jitter_rad")
            if (
                isinstance(jitter, bool)
                or not isinstance(jitter, (int, float))
                or not math.isfinite(float(jitter))
                or not 0.0 < float(jitter) <= 0.1
            ):
                raise ValueError("PHYSICS_DARWIN_REQUIRES_SEED_RANDOMIZATION")
        per_seed: dict[str, dict[str, dict[str, float]]] = {}
        all_receipts: list[dict[str, Any]] = []
        baseline_outcomes: list[dict[str, Any]] = []
        candidate_outcomes: list[dict[str, Any]] = []

        for index, case in enumerate(cases):
            order = ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
            pair: dict[str, dict[str, float]] = {}
            for variant in order:
                trajectory = (
                    case.baseline_trajectory if variant == "baseline" else case.candidate_trajectory
                )
                sandbox = self._sandbox_factory(
                    case.scenario.robot_id, case.scenario.world_id, "mujoco"
                )
                try:
                    if not sandbox.has_physics:
                        raise RuntimeError(sandbox.load_error or "PHYSICS_UNAVAILABLE")
                    resolved_model_hash = file_hash(sandbox.model_path)
                    if case.scenario.body_snapshot_hash not in {
                        "resolved-by-runner",
                        resolved_model_hash,
                    }:
                        raise ValueError("BODY_SNAPSHOT_HASH_MISMATCH")
                    scenario = replace(
                        case.scenario,
                        body_snapshot_hash=resolved_model_hash,
                        model_hash=resolved_model_hash,
                    )
                    backend = MujocoCpuBackend(sandbox)
                    receipt = backend.rollout(
                        RolloutRequest(
                            scenario=scenario,
                            trajectory=trajectory,
                            artifact_dir=artifact_root / str(case.scenario.seed) / variant,
                        )
                    )
                    receipt.evaluation_variant = variant
                    receipt.pair_id = scenario.scenario_id
                    replay = backend.replay(receipt, strict=True)
                finally:
                    sandbox.close()
                value = receipt.to_dict()
                all_receipts.append(value)
                outcome = {
                    "is_safe": receipt.is_safe,
                    "collision": bool(receipt.collision_pairs),
                    "wall_time_sec": receipt.metrics.get("wall_time_sec", 0.0),
                    "replay_verified": replay.verified,
                    "receipt_complete": receipt.valid_for_promotion,
                }
                (baseline_outcomes if variant == "baseline" else candidate_outcomes).append(outcome)
                pair[variant] = {
                    "success_rate": 1.0 if receipt.is_safe else 0.0,
                    "collision_rate": 1.0 if receipt.collision_pairs else 0.0,
                }
            per_seed[str(case.scenario.seed)] = pair

        baseline = _metrics(baseline_outcomes)
        candidate = _metrics(candidate_outcomes)
        critical = []
        if candidate["collision_rate"] > baseline["collision_rate"]:
            critical.append("collision_rate_regression")
        if candidate["replay_success_rate"] < 1.0:
            critical.append("strict_replay_failure")
        return PhysicsDarwinResult(
            baseline_metrics=baseline,
            candidate_metrics=candidate,
            per_seed=per_seed,
            simulation_receipts=all_receipts,
            regression_results={
                "passed": not critical,
                "critical_regressions": critical,
                "suite": "physics_counterexample_v1",
                "episodes": len(cases) * 2,
            },
        )


__all__ = ["PairedTrajectoryCase", "PhysicsDarwinResult", "PhysicsDarwinRunner"]
