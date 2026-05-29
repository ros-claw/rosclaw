"""Built-in mock Critic provider."""

from rosclaw.provider.core.provider import Provider
from rosclaw.provider.core.request import ProviderRequest
from rosclaw.provider.core.response import ProviderResponse


class MockCriticProvider(Provider):
    """Mock Critic provider for success detection and retry advice."""

    name = "mock_critic"
    capabilities = ["critic.success_detection", "critic.retry_advice"]

    async def infer(self, request: ProviderRequest) -> ProviderResponse:
        return ProviderResponse(
            request_id=request.request_id,
            provider=self.name,
            capability=request.capability,
            result={
                "success": True,
                "confidence": 0.85,
                "reason": "mock evaluation",
                "retry": {"recommended": False},
            },
        )

    async def health(self):
        return {"ok": True}
