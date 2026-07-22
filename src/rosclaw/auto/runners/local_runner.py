"""LocalRunner — lightweight local experiment runner (dry-run / smoke test)."""

import logging
from typing import Any

from .base import BaseRunner, RunnerResult
from .mock_physics import MockPhysicsModel

logger = logging.getLogger("rosclaw.auto.runners.local")


class LocalRunner(BaseRunner):
    """Run experiments locally with simulated or placeholder execution.

    Suitable for:
    - Dry-run validation
    - Unit test mocking
    - Smoke tests before expensive sandbox/darwin runs
    """

    name = "local"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        config = config or {}
        self._backend = config.get("backend") or (
            "mock" if config.get("simulate") is True else "unconfigured"
        )
        self._simulate = self._backend in {"mock", "fixture"}
        self._latency_sec = config.get("latency_sec", 0.1) if config else 0.1

    def health(self) -> dict:
        return {
            "status": "fixture" if self._simulate else "unconfigured",
            "runner": self.name,
            "backend": self._backend,
            "simulate": self._simulate,
            "evidence_domain": "FIXTURE" if self._simulate else None,
            "valid_for_promotion": False,
        }

    def run(self, experiment_spec: Any) -> RunnerResult:
        violations = self.validate_safety(experiment_spec)
        if violations:
            return RunnerResult(success=False, error=f"Safety violations: {violations}")

        task = getattr(experiment_spec, "task", "unknown")
        baseline = getattr(experiment_spec, "baseline_skill_id", "baseline")
        candidate = getattr(experiment_spec, "candidate_skill_id", "candidate")
        evaluation = getattr(experiment_spec, "evaluation", {})
        episodes = evaluation.get("episodes", 10)
        patch_ctx = getattr(experiment_spec, "patch_context", {})
        patch_changes = patch_ctx.get("changes", [])

        logger.info("LocalRunner: running %s episodes for %s", episodes, task)

        if self._simulate:
            # Deterministic mock physics: metrics depend on patch parameters
            baseline_metrics = MockPhysicsModel.evaluate([], seed=0)
            candidate_metrics = MockPhysicsModel.evaluate(patch_changes, seed=1)

            return RunnerResult(
                success=True,
                metrics={
                    "baseline": {
                        "success_rate": baseline_metrics["success_rate"],
                        "collision_rate": baseline_metrics["collision_rate"],
                        "episodes": episodes,
                    },
                    "candidate": {
                        "success_rate": candidate_metrics["success_rate"],
                        "collision_rate": candidate_metrics["collision_rate"],
                        "episodes": episodes,
                    },
                },
                logs=[
                    f"Simulated {episodes} episodes for {candidate} vs {baseline} with parametric physics"
                ],
                evidence_domain="FIXTURE",
                physics_executed=False,
                valid_for_promotion=False,
            )

        # Non-simulate mode: placeholder for real local execution
        return RunnerResult(
            success=False,
            error="LocalRunner backend is unconfigured; explicitly select backend=mock for fixtures",
        )
