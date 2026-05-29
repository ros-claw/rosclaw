"""Built-in mock Skill provider."""

from rosclaw.provider.core.provider import Provider
from rosclaw.provider.core.request import ProviderRequest
from rosclaw.provider.core.response import ProviderResponse


class MockSkillProvider(Provider):
    """Mock Skill provider for grasp, place, and pick-and-place."""

    name = "mock_skill"
    capabilities = ["skill.grasp", "skill.place", "skill.pick_and_place"]

    async def infer(self, request: ProviderRequest) -> ProviderResponse:
        skill = request.capability.split(".", 1)[1]
        return ProviderResponse(
            request_id=request.request_id,
            provider=self.name,
            capability=request.capability,
            result={
                "skill": skill,
                "status": "dispatched",
                "execution_trace": {
                    "controller": self.name,
                    "guard_checks": ["joint_limit", "collision", "workspace"],
                },
            },
        )

    async def health(self):
        return {"ok": True}
