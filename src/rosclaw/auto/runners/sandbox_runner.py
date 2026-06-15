"""SandboxRunner — safety-gated experiment runner via rosclaw-sandbox."""
import logging
from typing import Any

from .base import BaseRunner, RunnerResult
from .mock_physics import MockPhysicsModel

logger = logging.getLogger("rosclaw.auto.runners.sandbox")


class SandboxRunner(BaseRunner):
    """Execute experiments through rosclaw-sandbox firewall.

    Steps:
    1. Kinematic feasibility check
    2. Collision check
    3. Force/torque limit check
    4. Workspace boundary check
    5. Execute episode in MuJoCo / e-URDF
    6. Return metrics + safety clearance
    """
    name = "sandbox"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self._sandbox_client = config.get("sandbox_client") if config else None
        self._simulate = config.get("simulate", True) if config else True
        self._fixed_seed = config.get("fixed_seed") if config else None

    def health(self) -> dict:
        return {
            "status": "healthy",
            "runner": self.name,
            "sandbox_connected": self._sandbox_client is not None,
            "simulate": self._simulate,
        }

    def run(self, experiment_spec: Any) -> RunnerResult:
        violations = self.validate_safety(experiment_spec)
        if violations:
            return RunnerResult(success=False, error=f"Safety violations: {violations}")

        getattr(experiment_spec, "task", "unknown")
        candidate = getattr(experiment_spec, "candidate_skill_id", "candidate")
        evaluation = getattr(experiment_spec, "evaluation", {})
        episodes = evaluation.get("episodes", 10)
        safety = getattr(experiment_spec, "safety", {})
        max_collision = safety.get("max_collision", 0)
        max_force = safety.get("max_force", 15)
        patch_ctx = getattr(experiment_spec, "patch_context", {})
        patch_changes = patch_ctx.get("changes", [])

        logger.info("SandboxRunner: sandbox dry-run for %s (%s episodes)", candidate, episodes)

        if self._simulate:
            import random
            seed = self._fixed_seed if self._fixed_seed is not None else hash(candidate) % 10000
            rng = random.Random(seed)

            # Deterministic physics based on patch parameters
            physics = MockPhysicsModel.evaluate(patch_changes, seed=seed)
            collision_count = 0
            if physics["collision_rate"] > 0.05:
                collision_count = rng.randint(1, max(1, int(physics["collision_rate"] * 10)))

            # Force exceeded based on whether patch increases force-related params
            force_exceeded = False
            for change in patch_changes:
                key = change.get("path", "").split("/")[-1]
                if key in ("max_torque", "press_force", "gripper_force"):
                    val = change.get("new", 0)
                    if isinstance(val, (int, float)):
                        if val > 40:
                            force_exceeded = True  # deterministic rejection for extreme values
                        elif val > 30:
                            force_exceeded = rng.random() < 0.30

            if collision_count > max_collision:
                return RunnerResult(
                    success=False,
                    safety_violations=[f"Collision count {collision_count} exceeds limit {max_collision}"],
                    error="Sandbox rejected: collision check failed",
                )
            if force_exceeded and max_force < 50:
                return RunnerResult(
                    success=False,
                    safety_violations=[f"Force exceeded limit {max_force}"],
                    error="Sandbox rejected: force limit exceeded",
                )

            baseline_metrics = MockPhysicsModel.evaluate([], seed=seed)

            return RunnerResult(
                success=True,
                metrics={
                    "baseline": {
                        "success_rate": baseline_metrics["success_rate"],
                        "collision_rate": baseline_metrics["collision_rate"],
                        "episodes": episodes,
                    },
                    "candidate": {
                        "success_rate": physics["success_rate"],
                        "collision_rate": physics["collision_rate"],
                        "episodes": episodes,
                    },
                    "sandbox_clearance": True,
                    "collision_count": collision_count,
                },
                logs=[f"Sandbox cleared {candidate} after {episodes} episodes"],
            )

        # Real sandbox execution via client
        if self._sandbox_client is None:
            return RunnerResult(
                success=False,
                error="Sandbox client not configured and simulate=False",
            )

        try:
            result = self._sandbox_client.run_experiment(experiment_spec)
            return RunnerResult(
                success=result.get("success", False),
                metrics=result.get("metrics", {}),
                logs=result.get("logs", []),
                safety_violations=result.get("safety_violations", []),
            )
        except Exception as exc:
            logger.exception("Sandbox execution failed")
            return RunnerResult(success=False, error=str(exc))
