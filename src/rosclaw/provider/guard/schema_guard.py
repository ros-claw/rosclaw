"""Schema Guard.

Validates that provider output conforms to expected schema.
"""

from typing import Any

from rosclaw.provider.core.request import ProviderRequest
from rosclaw.provider.core.response import ProviderResponse
from rosclaw.provider.guard.base import Guard


class SchemaGuard(Guard):
    """Validate response.result structure against required fields."""

    name = "schema"

    # Capability -> required top-level result keys
    REQUIRED_KEYS: dict[str, list[str]] = {
        "vlm.object_grounding": ["objects"],
        "vla.action_proposal": ["actions"],
        "skill.grasp": ["status", "execution_trace"],
        "critic.success_detection": ["success", "confidence"],
    }

    def check(
        self,
        request: ProviderRequest,
        response: ProviderResponse,
    ) -> dict[str, Any]:
        capability = request.capability
        result = response.result
        checks: list[dict[str, str]] = []

        required = self.REQUIRED_KEYS.get(capability, [])
        missing = [k for k in required if k not in result]

        if missing:
            checks.append({
                "name": "schema",
                "status": "fail",
                "detail": f"Missing required keys: {missing}",
            })
            return {
                "pass": False,
                "checks": checks,
                "reason": f"Schema validation failed for {capability}: missing {missing}",
                "recommended_action": "retry_or_fallback",
            }

        checks.append({"name": "schema", "status": "pass", "detail": "All required keys present"})
        return {
            "pass": True,
            "checks": checks,
            "reason": "",
            "recommended_action": "",
        }
