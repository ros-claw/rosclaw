"""HTTP, in-process and disabled adapters for the optional Know package."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .contracts import (
    KnowledgeUsageFeedbackV1,
    ReferenceContextV2,
    ReferencePackV2,
    ResearchRequestV2,
)


class KnowUnavailableError(RuntimeError):
    """Raised at the optional service boundary; Core remains operational."""


class HttpKnowClient:
    def __init__(self, base_url: str, *, api_key: str = "", timeout: float = 15.0) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("Know URL must use http:// or https://")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> bytes:
        headers = {"Accept": "application/json"}
        body = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(payload, ensure_ascii=False).encode()
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=body, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read(10_000_000)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            raise KnowUnavailableError(
                f"Know request failed: {method} {path}: {type(exc).__name__}"
            ) from exc

    def health(self) -> dict[str, Any]:
        payload = json.loads(self._request("GET", "/know/v2/health"))
        return {"transport": "service", **payload}

    def doctor(self) -> dict[str, Any]:
        return json.loads(self._request("GET", "/know/v2/doctor"))

    def explain(
        self, *, query: str, context: ReferenceContextV2, top_k: int = 10
    ) -> dict[str, Any]:
        return json.loads(
            self._request(
                "POST",
                "/know/v2/explain",
                {
                    "query": query,
                    "context": context.model_dump(mode="json", exclude_none=True),
                    "top_k": top_k,
                    "token_budget": 8_000,
                },
            )
        )

    def project_diff(
        self, *, project_id: str, from_snapshot: str, to_snapshot: str
    ) -> dict[str, Any]:
        path = (
            f"/know/v2/projects/{project_id}/diff?"
            + urllib.parse.urlencode({"from": from_snapshot, "to": to_snapshot})
        )
        return json.loads(self._request("GET", path))

    def refresh_source(self, *, source_id: str, apply: bool = False) -> dict[str, Any]:
        return json.loads(
            self._request("POST", f"/know/v2/sources/{source_id}/refresh", {"apply": apply})
        )

    def freeze(self, *, label: str) -> dict[str, Any]:
        return json.loads(self._request("POST", "/know/v2/freeze", {"label": label}))

    def research(self, request: ResearchRequestV2) -> dict[str, Any]:
        return json.loads(
            self._request("POST", "/know/v2/research", request.model_dump(mode="json"))
        )

    def reference_pack(
        self, *, query: str, context: ReferenceContextV2, top_k: int, token_budget: int
    ) -> ReferencePackV2:
        raw = self._request(
            "POST",
            "/know/v2/reference-packs",
            {
                "query": query,
                "context": context.model_dump(mode="json", exclude_none=True),
                "top_k": top_k,
                "token_budget": token_budget,
            },
        )
        return ReferencePackV2.validate_wire_json(raw)

    def get_reference_pack(self, reference_pack_id: str) -> ReferencePackV2 | None:
        try:
            raw = self._request("GET", f"/know/v2/reference-packs/{reference_pack_id}")
        except KnowUnavailableError:
            return None
        return ReferencePackV2.validate_wire_json(raw)

    def submit_feedback(self, feedback: KnowledgeUsageFeedbackV1) -> bool:
        raw = self._request(
            "POST", "/know/v2/feedback", feedback.model_dump(mode="json", exclude_none=True)
        )
        return bool(json.loads(raw).get("created"))


class InProcessKnowClient:
    """Explicit adapter over public rosclaw-know APIs, never its internals in Core."""

    def __init__(self, store: Any) -> None:
        from rosclaw_know.retrieval import ReferencePackBuilder

        self.store = store
        self.builder = ReferencePackBuilder(store)

    def health(self) -> dict[str, Any]:
        capabilities = self.store.capabilities.model_dump(mode="json")
        return {"status": "ok", "transport": "inprocess", "store": capabilities}

    def doctor(self) -> dict[str, Any]:
        from rosclaw_know.operations import doctor

        return doctor(self.store)

    def explain(
        self, *, query: str, context: ReferenceContextV2, top_k: int = 10
    ) -> dict[str, Any]:
        from rosclaw_know.contracts import ReferenceContextV2 as KnowReferenceContextV2

        external = KnowReferenceContextV2.validate_wire_json(context.to_wire_json())
        return self.builder.explain(query=query, context=external, top_k=top_k).model_dump(
            mode="json"
        )

    def project_diff(
        self, *, project_id: str, from_snapshot: str, to_snapshot: str
    ) -> dict[str, Any]:
        from rosclaw_know.operations import project_diff

        return project_diff(
            self.store,
            project_id=project_id,
            from_snapshot=from_snapshot,
            to_snapshot=to_snapshot,
        )

    def refresh_source(self, *, source_id: str, apply: bool = False) -> dict[str, Any]:
        from rosclaw_know.operations import refresh_source

        return asyncio.run(refresh_source(self.store, source_id=source_id, apply=apply))

    def freeze(self, *, label: str) -> dict[str, Any]:
        from rosclaw_know.operations import freeze

        return freeze(self.store, label=label).model_dump(mode="json")

    def research(self, request: ResearchRequestV2) -> dict[str, Any]:
        from rosclaw_know.contracts import ResearchRequestV2 as KnowResearchRequestV2
        from rosclaw_know.sources import (
            ResearchOrchestrator,
            default_source_registry,
        )

        external = KnowResearchRequestV2.model_validate_json(request.to_wire_json())
        result = asyncio.run(
            ResearchOrchestrator(self.store, default_source_registry()).run(external)
        )
        return result.model_dump(mode="json")

    def reference_pack(
        self, *, query: str, context: ReferenceContextV2, top_k: int, token_budget: int
    ) -> ReferencePackV2:
        from rosclaw_know.contracts import ReferenceContextV2 as KnowReferenceContextV2

        external_context = KnowReferenceContextV2.model_validate_json(context.to_wire_json())
        pack = self.builder.retrieve(
            query=query, context=external_context, top_k=top_k, token_budget=token_budget
        )
        return ReferencePackV2.validate_wire_json(pack.model_dump_json())

    def get_reference_pack(self, reference_pack_id: str) -> ReferencePackV2 | None:
        pack = self.store.get_reference_pack(reference_pack_id)
        return ReferencePackV2.validate_wire_json(pack.model_dump_json()) if pack else None

    def submit_feedback(self, feedback: KnowledgeUsageFeedbackV1) -> bool:
        from rosclaw_know.contracts import KnowledgeUsageFeedbackV1 as KnowFeedbackV1

        external = KnowFeedbackV1.model_validate_json(feedback.to_wire_json())
        return bool(self.store.put_feedback(external))


class DisabledKnowClient:
    def health(self) -> dict[str, Any]:
        return {"status": "disabled", "transport": "disabled"}

    def research(self, request: ResearchRequestV2) -> dict[str, Any]:
        raise KnowUnavailableError("Know is disabled")

    def doctor(self) -> dict[str, Any]:
        return {"status": "disabled", "transport": "disabled"}

    def explain(self, **kwargs: Any) -> dict[str, Any]:
        raise KnowUnavailableError("Know is disabled")

    def project_diff(self, **kwargs: Any) -> dict[str, Any]:
        raise KnowUnavailableError("Know is disabled")

    def refresh_source(self, **kwargs: Any) -> dict[str, Any]:
        raise KnowUnavailableError("Know is disabled")

    def freeze(self, **kwargs: Any) -> dict[str, Any]:
        raise KnowUnavailableError("Know is disabled")

    def reference_pack(self, **kwargs: Any) -> ReferencePackV2:
        raise KnowUnavailableError("Know is disabled")

    def get_reference_pack(self, reference_pack_id: str) -> ReferencePackV2 | None:
        return None

    def submit_feedback(self, feedback: KnowledgeUsageFeedbackV1) -> bool:
        raise KnowUnavailableError("Know is disabled")
