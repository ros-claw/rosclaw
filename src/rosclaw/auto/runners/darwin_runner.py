"""DarwinRunner — multi-seed benchmark evaluation via rosclaw-darwin."""

import logging
from typing import Any

from .base import BaseRunner, RunnerResult
from .mock_physics import MockPhysicsModel

logger = logging.getLogger("rosclaw.auto.runners.darwin")


class DarwinRunner(BaseRunner):
    """Execute experiments through rosclaw-darwin benchmark suite.

    Responsibilities:
    - Multi-seed evaluation
    - Regression suite
    - Stress scenario testing
    - Leaderboard recording
    """

    name = "darwin"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        config = config or {}
        self._darwin_client = config.get("darwin_client")
        self._backend = config.get("backend") or (
            "mock" if config.get("simulate") is True else "unconfigured"
        )
        self._simulate = self._backend in {"mock", "fixture"}

    def health(self) -> dict:
        return {
            "status": (
                "connected"
                if self._darwin_client is not None
                else ("fixture" if self._simulate else "unconfigured")
            ),
            "runner": self.name,
            "backend": self._backend,
            "darwin_connected": self._darwin_client is not None,
            "simulate": self._simulate,
            "evidence_domain": "FIXTURE" if self._simulate else None,
            "valid_for_promotion": False,
        }

    def run(self, experiment_spec: Any) -> RunnerResult:
        violations = self.validate_safety(experiment_spec)
        if violations:
            return RunnerResult(success=False, error=f"Safety violations: {violations}")

        getattr(experiment_spec, "task", "unknown")
        candidate = getattr(experiment_spec, "candidate_skill_id", "candidate")
        getattr(experiment_spec, "baseline_skill_id", "baseline")
        evaluation = getattr(experiment_spec, "evaluation", {})
        episodes = evaluation.get("episodes", 50)
        seeds = evaluation.get("seeds", [0])
        patch_ctx = getattr(experiment_spec, "patch_context", {})
        patch_changes = patch_ctx.get("changes", [])

        logger.info(
            "DarwinRunner: benchmark %s on seeds %s (%s episodes each)", candidate, seeds, episodes
        )

        if self._simulate:
            seed_results = {}
            for seed in seeds:
                # Isolated random state per seed — no global pollution
                paired_seed = int(seed) * 1000
                baseline_metrics = MockPhysicsModel.evaluate([], seed=paired_seed)
                candidate_metrics = MockPhysicsModel.evaluate(patch_changes, seed=paired_seed)
                seed_results[seed] = {
                    "baseline": baseline_metrics,
                    "candidate": candidate_metrics,
                }

            # Aggregate across seeds
            avg_baseline_sr = sum(
                r["baseline"]["success_rate"] for r in seed_results.values()
            ) / len(seed_results)
            avg_candidate_sr = sum(
                r["candidate"]["success_rate"] for r in seed_results.values()
            ) / len(seed_results)
            avg_baseline_col = sum(
                r["baseline"]["collision_rate"] for r in seed_results.values()
            ) / len(seed_results)
            avg_candidate_col = sum(
                r["candidate"]["collision_rate"] for r in seed_results.values()
            ) / len(seed_results)

            return RunnerResult(
                success=True,
                metrics={
                    "baseline": {
                        "success_rate": round(avg_baseline_sr, 3),
                        "collision_rate": round(avg_baseline_col, 3),
                        "episodes": episodes,
                        "seeds": seeds,
                    },
                    "candidate": {
                        "success_rate": round(avg_candidate_sr, 3),
                        "collision_rate": round(avg_candidate_col, 3),
                        "episodes": episodes,
                        "seeds": seeds,
                    },
                    "per_seed": seed_results,
                },
                logs=[
                    f"Darwin benchmark completed on {len(seeds)} seeds, {episodes} episodes each"
                ],
                evidence_domain="FIXTURE",
                physics_executed=False,
                valid_for_promotion=False,
            )

        if self._darwin_client is None:
            return RunnerResult(
                success=False,
                error=(
                    "Darwin backend is unconfigured; provide a physics Darwin client or "
                    "explicitly select backend=mock for fixtures"
                ),
            )

        try:
            result = self._darwin_client.run_benchmark(experiment_spec)
            return RunnerResult(
                success=result.get("success") is True,
                metrics=result.get("metrics", {}),
                logs=result.get("logs", []),
                safety_violations=result.get("safety_violations", []),
                evidence_domain=result.get("evidence_domain"),
                physics_executed=result.get("physics_executed") is True,
                valid_for_promotion=result.get("valid_for_promotion") is True,
            )
        except Exception as exc:
            logger.exception("Darwin execution failed")
            return RunnerResult(success=False, error=str(exc))
