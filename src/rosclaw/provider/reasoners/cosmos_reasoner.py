"""Cosmos physical reasoner backend."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from rosclaw.provider.core.response import ProviderResponse
from rosclaw.provider.reasoners.base import PhysicalReasoner


class CosmosReasoner(PhysicalReasoner):
    """Cosmos-style physical reasoning endpoint.

    Uses a simple HTTP POST with JSON payload. Image input is transmitted as
    base64. If the endpoint is unreachable, returns a safe fallback response.
    """

    def __init__(self, endpoint: str | None = None, name: str = "cosmos") -> None:
        self.name = name
        self.endpoint = endpoint or os.environ.get("COSMOS_ENDPOINT", "http://localhost:8004")

    def reason(
        self,
        question: str,
        image: str | bytes | None = None,
        image_mime: str = "image/png",
        capability: str = "vlm.risk_assessment",
    ) -> ProviderResponse:
        payload: dict[str, Any] = {
            "capability": capability,
            "question": question,
        }
        if image is not None:
            if isinstance(image, bytes):
                import base64
                image = base64.b64encode(image).decode("utf-8")
            payload["image"] = image
            payload["image_mime"] = image_mime

        raw_text, status, errors = self._call_http(payload, capability)
        normalized = self._normalized_result(raw_text, capability)
        return ProviderResponse(
            request_id="",
            provider=self.name,
            capability=capability,
            result={"raw": raw_text, "normalized": normalized},
            status=status,
            errors=errors,
        )

    def plan(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        capability: str = "reasoning.physical",
    ) -> ProviderResponse:
        payload = {
            "capability": capability,
            "task": task,
            "context": context or {},
        }
        raw_text, status, errors = self._call_http(payload, capability)
        return ProviderResponse(
            request_id="",
            provider=self.name,
            capability=capability,
            result={"raw": raw_text},
            status=status,
            errors=errors,
        )

    def analyze(
        self,
        observations: list[dict[str, Any]],
        capability: str = "reasoning.risk_explain",
    ) -> ProviderResponse:
        payload = {
            "capability": capability,
            "observations": observations,
        }
        raw_text, status, errors = self._call_http(payload, capability)
        return ProviderResponse(
            request_id="",
            provider=self.name,
            capability=capability,
            result={"raw": raw_text},
            status=status,
            errors=errors,
        )

    def _call_http(
        self, payload: dict[str, Any], capability: str
    ) -> tuple[str, str, list[str]]:
        """POST JSON to the reasoning endpoint and return (raw_text, status, errors)."""
        url = self.endpoint.rstrip("/") + "/infer"
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30.0) as response:
                raw_text = response.read().decode("utf-8")
            return raw_text, "ok", []
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")[:500]
            return body, "failed", [f"HTTP {exc.code}: {body}"]
        except urllib.error.URLError as exc:
            return str(exc.reason), "failed", [f"Reasoner unreachable: {exc.reason}"]
        except Exception as exc:
            return str(exc), "failed", [str(exc)]
