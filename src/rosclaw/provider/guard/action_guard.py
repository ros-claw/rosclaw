"""Action Guard.

Validates that action proposals are within safe bounds.
"""

from typing import Any

from rosclaw.provider.core.request import ProviderRequest
from rosclaw.provider.core.response import ProviderResponse
from rosclaw.provider.guard.base import Guard


class ActionGuard(Guard):
    """Validate action bounds: max delta norm, workspace, etc."""

    name = "action"

    DEFAULT_MAX_DELTA = 0.05
    DEFAULT_MAX_GRIPPER_FORCE = 1.0

    def check(
        self,
        request: ProviderRequest,
        response: ProviderResponse,
    ) -> dict[str, Any]:
        result = response.result
        checks: list[dict[str, str]] = []

        # Extract constraints
        constraints = request.constraints
        max_delta = constraints.get("max_delta", self.DEFAULT_MAX_DELTA)

        # Check pose delta
        actions = result.get("actions", [])
        if actions:
            action = actions[0]
            deltas = [
                action.get("dx", 0.0),
                action.get("dy", 0.0),
                action.get("dz", 0.0),
            ]
            norm = sum(d * d for d in deltas) ** 0.5
            if norm > max_delta:
                checks.append({
                    "name": "action_bound",
                    "status": "fail",
                    "detail": f"Pose delta norm {norm:.4f} > max {max_delta}",
                })
                return {
                    "pass": False,
                    "checks": checks,
                    "reason": f"Action exceeds safe delta bound: {norm:.4f} > {max_delta}",
                    "recommended_action": "replan",
                }
            checks.append({
                "name": "action_bound",
                "status": "pass",
                "detail": f"Pose delta norm {norm:.4f} within bound",
            })

        return {
            "pass": True,
            "checks": checks,
            "reason": "",
            "recommended_action": "",
        }
