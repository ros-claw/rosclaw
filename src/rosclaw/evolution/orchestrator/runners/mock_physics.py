"""MockPhysicsModel — deterministic parameter-to-metric mapping for testing.

Replaces pure random metrics with a physics-inspired lookup so that:
- Patch changes have predictable effects on success_rate / collision_rate.
- Different runners (local/sandbox/darwin) see consistent candidate behaviour.
- Tests can assert on exact metric values given known patch parameters.
"""

import math


class MockPhysicsModel:
    """Deterministic mock physics for skill parameter patches.

    Each parameter has an optimal range.  Deviations from optimum reduce
    success_rate and increase collision_rate in a deterministic way.
    """

    # Optimal parameter values (from RuleDiagnoser search_space midpoints)
    OPTIMAL = {
        "pregrasp_height": 0.05,
        "approach_speed": 0.10,
        "gripper_force": 12.5,
        "grasp_offset": 0.0,
        "collision_buffer": 0.03,
        "max_planning_time": 17.5,
        "motion_speed": 0.175,
        "force_limit": 30.0,
        "compliance_gain": 0.55,
        "max_torque": 6.0,
        "alignment_tolerance": 0.06,
        "press_force": 5.5,
        "approach_angle": 7.5,
    }

    # Scale factors: how sensitive is each parameter
    SCALE = {
        "pregrasp_height": 0.15,
        "approach_speed": 0.25,
        "gripper_force": 0.10,
        "grasp_offset": 0.20,
        "collision_buffer": 0.30,
        "max_planning_time": 0.05,
        "motion_speed": 0.15,
        "force_limit": 0.08,
        "compliance_gain": 0.12,
        "max_torque": 0.10,
        "alignment_tolerance": 0.15,
        "press_force": 0.10,
        "approach_angle": 0.08,
    }

    @classmethod
    def _extract_params(cls, patch_changes: list[dict]) -> dict[str, float]:
        """Extract numeric parameter values from patch changes."""
        params = {}
        for change in patch_changes:
            path = change.get("path", "")
            new_val = change.get("new")
            if new_val is None:
                continue
            # Extract last component of path as param name
            key = path.split("/")[-1] if "/" in path else path
            if key in cls.OPTIMAL and isinstance(new_val, (int, float)):
                params[key] = float(new_val)
        return params

    @classmethod
    def evaluate(
        cls,
        patch_changes: list[dict],
        baseline_sr: float = 0.40,
        baseline_col: float = 0.10,
        seed: int | None = None,
        noise_scale: float = 0.02,
    ) -> dict:
        """Return deterministic metrics for a given patch.

        Returns {"success_rate": float, "collision_rate": float}
        """
        import random

        rng = random.Random(seed if seed is not None else 42)
        params = cls._extract_params(patch_changes)

        # Baseline skill has no params changed → baseline metrics
        if not params:
            return {
                "success_rate": round(baseline_sr + rng.gauss(0, noise_scale), 3),
                "collision_rate": round(baseline_col + rng.gauss(0, noise_scale * 0.5), 3),
            }

        # Compute deviation penalty for each modified parameter
        penalty = 0.0
        collision_penalty = 0.0
        for key, val in params.items():
            opt = cls.OPTIMAL.get(key, 0.0)
            scale = cls.SCALE.get(key, 0.1)
            delta = (val - opt) / scale if scale else 0.0
            # Gaussian-like penalty: optimum = 0 penalty, edges = high penalty
            param_penalty = math.exp(-0.5 * delta * delta)  # 1.0 at optimum, lower at edges
            penalty += param_penalty

            # Collision penalty: high speed / low buffer / high force → more collisions
            if key in ("approach_speed", "max_torque", "press_force"):
                collision_penalty += max(0.0, (val - opt) / scale) * 0.05
            if key == "collision_buffer":
                collision_penalty += max(0.0, (opt - val) / scale) * 0.08

        # Average penalty across modified params
        avg_penalty = penalty / len(params)
        candidate_sr = baseline_sr + 0.20 * avg_penalty + rng.gauss(0, noise_scale)
        candidate_col = baseline_col + collision_penalty + rng.gauss(0, noise_scale * 0.5)

        # Clamp to [0, 1]
        candidate_sr = max(0.0, min(1.0, candidate_sr))
        candidate_col = max(0.0, min(1.0, candidate_col))

        return {
            "success_rate": round(candidate_sr, 3),
            "collision_rate": round(candidate_col, 3),
        }

    @classmethod
    def evaluate_multi_seed(
        cls,
        patch_changes: list[dict],
        seeds: list[int],
        baseline_sr: float = 0.40,
        baseline_col: float = 0.10,
        noise_scale: float = 0.02,
    ) -> dict[int, dict]:
        """Evaluate across multiple seeds with isolated random state."""
        results = {}
        for seed in seeds:
            results[seed] = cls.evaluate(
                patch_changes, baseline_sr, baseline_col, seed=seed, noise_scale=noise_scale
            )
        return results
