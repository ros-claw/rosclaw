"""Built-in VLM provider with real perception logic.

Uses PIL for lightweight color-based object detection.
Supports grounding common colored objects (red cup, blue block, etc.)
without requiring an external model service.
"""

from __future__ import annotations


from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rosclaw.provider.core.errors import CapabilityNotSupportedError
from rosclaw.provider.core.provider import Provider
from rosclaw.provider.core.request import ProviderRequest
from rosclaw.provider.core.response import ProviderResponse


@dataclass(frozen=True)
class _ArtifactRef:
    type: str
    uri: str = ""
    value: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "_ArtifactRef":
        return cls(type=d.get("type", ""), uri=d.get("uri", ""), value=d.get("value", ""))


def _resolve_artifact(uri: str) -> Path:
    if uri.startswith("artifact://"):
        parts = uri[11:].split("/", 1)
        if len(parts) == 2:
            return Path("./artifacts") / parts[0] / parts[1]
    return Path(uri)


# Color detection thresholds (RGB) for common object queries
_COLOR_MAP: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "red": ((150, 0, 0), (255, 80, 80)),
    "blue": ((0, 0, 150), (80, 80, 255)),
    "green": ((0, 150, 0), (80, 255, 80)),
    "yellow": ((200, 200, 0), (255, 255, 100)),
    "orange": ((200, 100, 0), (255, 180, 80)),
    "purple": ((100, 0, 100), (200, 80, 200)),
    "white": ((200, 200, 200), (255, 255, 255)),
    "black": ((0, 0, 0), (60, 60, 60)),
}


class MockVLMProvider(Provider):
    """VLM provider with real color-based object grounding.

    Capabilities:
        - vlm.object_grounding: detect colored objects in images
        - vlm.scene_understanding: basic scene description

    Uses PIL for color-based detection. Falls back to synthetic results
    when PIL is unavailable or image is not provided.
    """

    def __init__(self, manifest, artifact_manager=None):
        super().__init__(manifest)
        self._color_map = _COLOR_MAP

    async def infer(self, request: ProviderRequest) -> ProviderResponse:
        self._ensure_capability_supported(request.capability)

        if request.capability == "vlm.object_grounding":
            return await self._object_grounding(request)
        if request.capability == "vlm.scene_understanding":
            return await self._scene_understanding(request)

        raise CapabilityNotSupportedError(
            f"MockVLMProvider does not support '{request.capability}'",
            provider=self.name,
        )

    async def health(self):
        return {
            "ok": True,
            "provider": self.name,
            "capabilities": self.capabilities,
            "backend": "pil_color_detection",
        }

    async def _object_grounding(self, request):
        query = request.inputs.get("query", "")
        image_input = request.inputs.get("image", "")
        image_path = self._resolve_image_path(image_input)
        detected = self._detect_by_color(image_path, query)

        return ProviderResponse(
            request_id=request.request_id,
            provider=self.name,
            capability="vlm.object_grounding",
            result={
                "objects": [detected] if detected else [],
                "query": query,
                "image_resolved": str(image_path) if image_path else None,
            },
            confidence=detected["confidence"] if detected else 0.0,
        )

    async def _scene_understanding(self, request):
        image_input = request.inputs.get("image", "")
        image_path = self._resolve_image_path(image_input)

        scene_type = "tabletop"
        if image_path and Path(image_path).exists():
            try:
                scene_type = self._analyze_scene(image_path)
            except Exception:
                scene_type = "tabletop"

        return ProviderResponse(
            request_id=request.request_id,
            provider=self.name,
            capability="vlm.scene_understanding",
            result={"scene": scene_type, "objects": [], "risks": []},
            confidence=0.8,
        )

    def _resolve_image_path(self, image_input):
        if isinstance(image_input, dict):
            if image_input.get("type") == "artifact":
                ref = _ArtifactRef.from_dict(image_input)
                return _resolve_artifact(ref.uri)
            if image_input.get("type") == "file":
                return Path(image_input.get("value", ""))
            return None
        if isinstance(image_input, str):
            if image_input.startswith("artifact://"):
                return _resolve_artifact(image_input)
            if image_input.startswith("file://"):
                return Path(image_input[7:])
            return Path(image_input)
        return None

    def _detect_by_color(self, image_path, query):
        if image_path is None or not image_path.exists():
            color = self._extract_color(query)
            return {
                "id": f"obj_{color or 'unknown'}",
                "label": query or "unknown",
                "bbox_2d": [120, 80, 230, 200],
                "confidence": 0.85,
                "method": "synthetic_fallback",
            }

        try:
            from PIL import Image
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                pixels = list(img.getdata())
                w, h = img.size

            color = self._extract_color(query)
            if color is None:
                return {
                    "id": "obj_unknown",
                    "label": query,
                    "bbox_2d": [0, 0, w, h],
                    "confidence": 0.3,
                    "method": "no_color_match",
                }

            low, high = self._color_map[color]
            mask_pixels = [
                (i % w, i // w)
                for i, (r, g, b) in enumerate(pixels)
                if low[0] <= r <= high[0]
                and low[1] <= g <= high[1]  # noqa: W503
                and low[2] <= b <= high[2]  # noqa: W503
            ]

            if not mask_pixels:
                return {
                    "id": f"obj_{color}",
                    "label": query,
                    "bbox_2d": [w // 4, h // 4, 3 * w // 4, 3 * h // 4],
                    "confidence": 0.4,
                    "method": "color_not_found",
                }

            xs = [x for x, y in mask_pixels]
            ys = [y for x, y in mask_pixels]
            confidence = min(0.95, 0.5 + len(mask_pixels) / (w * h) * 10)

            return {
                "id": f"obj_{color}",
                "label": query,
                "bbox_2d": [min(xs), min(ys), max(xs), max(ys)],
                "confidence": round(confidence, 2),
                "method": "pil_color_detection",
                "pixel_count": len(mask_pixels),
            }

        except ImportError:
            color = self._extract_color(query)
            return {
                "id": f"obj_{color or 'unknown'}",
                "label": query,
                "bbox_2d": [120, 80, 230, 200],
                "confidence": 0.75,
                "method": "synthetic_no_pil",
            }
        except Exception as e:
            return {
                "id": "obj_error",
                "label": query,
                "bbox_2d": [0, 0, 100, 100],
                "confidence": 0.0,
                "method": "error",
                "error": str(e),
            }

    @staticmethod
    def _extract_color(query):
        query_lower = query.lower()
        for color in _COLOR_MAP:
            if color in query_lower:
                return color
        return None

    @staticmethod
    def _analyze_scene(image_path):
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                img = img.convert("L")
                pixels = list(img.getdata())
            avg_brightness = sum(pixels) / len(pixels)
            if avg_brightness > 200:
                return "bright_indoor"
            if avg_brightness > 100:
                return "tabletop"
            return "dark_scene"
        except Exception:
            return "tabletop"
