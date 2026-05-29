"""Built-in mock VLM provider."""

from rosclaw.provider.core.provider import Provider
from rosclaw.provider.core.request import ProviderRequest
from rosclaw.provider.core.response import ProviderResponse


class MockVLMProvider(Provider):
    """Mock VLM provider for scene understanding and object grounding."""

    name = "mock_vlm"
    capabilities = ["vlm.object_grounding", "vlm.scene_understanding"]

    async def infer(self, request: ProviderRequest) -> ProviderResponse:
        cap = request.capability
        if cap == "vlm.object_grounding":
            query = request.inputs.get("query", "")
            return ProviderResponse(
                request_id=request.request_id,
                provider=self.name,
                capability=cap,
                result={
                    "objects": [
                        {
                            "id": "obj_001",
                            "label": query or "unknown",
                            "bbox_2d": [120, 80, 230, 200],
                            "confidence": 0.93,
                        }
                    ],
                    "risks": [],
                },
                confidence=0.93,
            )
        return ProviderResponse(
            request_id=request.request_id,
            provider=self.name,
            capability=cap,
            result={"scene": "tabletop", "objects": [], "risks": []},
        )

    async def health(self):
        return {"ok": True}
