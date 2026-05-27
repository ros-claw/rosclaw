"""GuardPipeline - Sequential guard checks for provider outputs.

Sits between CapabilityRouter and Runtime execution.
"""

from typing import Any

from rosclaw.provider.core.request import ProviderRequest
from rosclaw.provider.core.response import ProviderResponse
from rosclaw.provider.core.errors import GuardBlockedError
from rosclaw.provider.guard.base import Guard


class GuardPipeline:
    """Run multiple guards in sequence.

    If any guard fails, raise GuardBlockedError with aggregated checks.
    """

    def __init__(self, guards: list[Guard] | None = None):
        self.guards = guards or []

    def add(self, guard: Guard) -> None:
        self.guards.append(guard)

    def check(
        self,
        request: ProviderRequest,
        response: ProviderResponse,
    ) -> dict[str, Any]:
        """Run all guards. Returns aggregated result or raises GuardBlockedError."""
        all_checks: list[dict[str, Any]] = []

        for guard in self.guards:
            result = guard.check(request, response)
            all_checks.extend(result.get("checks", []))
            if not result.get("pass", True):
                raise GuardBlockedError(
                    message=result.get("reason", f"Guard '{guard.name}' blocked"),
                    provider=response.provider,
                    request_id=request.request_id,
                    checks=all_checks,
                    recommended_action=result.get("recommended_action", "replan"),
                )

        return {
            "pass": True,
            "checks": all_checks,
            "reason": "",
            "recommended_action": "",
        }
