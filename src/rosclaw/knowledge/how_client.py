"""Advisory-only adapters for rosclaw-how v2."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .contracts import HowAdviceBundleV2, HowAdviceRequestV2, KnowledgeUsageFeedbackV1


class HowUnavailableError(RuntimeError):
    pass


class HttpHowClient:
    def __init__(self, base_url: str, *, api_key: str = "", timeout: float = 15.0) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("How URL must use http:// or https://")
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
            raise HowUnavailableError(
                f"How request failed: {method} {path}: {type(exc).__name__}"
            ) from exc

    def health(self) -> dict[str, Any]:
        return json.loads(self._request("GET", "/how/v2/health"))

    def doctor(self) -> dict[str, Any]:
        return json.loads(self._request("GET", "/how/v2/doctor"))

    def explain(self, advice_id: str) -> dict[str, Any]:
        return json.loads(self._request("GET", f"/how/v2/advice/{advice_id}/explain"))

    def advise(self, request: HowAdviceRequestV2) -> HowAdviceBundleV2:
        raw = self._request("POST", "/how/v2/advice", request.model_dump(mode="json"))
        return HowAdviceBundleV2.validate_wire_json(raw)

    def submit_feedback(self, feedback: KnowledgeUsageFeedbackV1) -> bool:
        raw = self._request(
            "POST", "/how/v2/feedback", feedback.model_dump(mode="json", exclude_none=True)
        )
        return bool(json.loads(raw).get("created"))


class InProcessHowClient:
    def __init__(self, know_client: Any) -> None:
        from rosclaw_how.v2 import AdviceEngine
        from rosclaw_how.v2.know_client import InProcessKnowClient

        external_know = InProcessKnowClient(
            lambda **kwargs: _external_reference_pack(know_client, **kwargs),
            feedback_sink=lambda feedback: know_client.submit_feedback(
                KnowledgeUsageFeedbackV1.validate_wire_json(feedback.model_dump_json())
            ),
            health_provider=know_client.health,
        )
        self.engine = AdviceEngine(external_know)
        self._advice: dict[str, HowAdviceBundleV2] = {}

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "transport": "inprocess",
            "advisory_only": True,
            "modes": ["discover", "consult", "diagnose", "catalyze"],
            "know": self.engine.know_client.health(),
        }

    def doctor(self) -> dict[str, Any]:
        return {
            "schema_version": "rosclaw.how.doctor.v1",
            "status": "ok",
            "know_reachable": True,
            "reference_pack_protocol": "rosclaw.knowledge.legacy.reference_pack.v2",
            "advice_store": {"backend": "bounded_process_memory", "durable": False},
            "pack_cache": {},
            "stale_policy": {"stale": "requires_revalidation"},
            "feedback_channel": "forwarded_to_know_governance",
            "memory_separation": True,
            "action_authority": False,
        }

    def advise(self, request: HowAdviceRequestV2) -> HowAdviceBundleV2:
        from rosclaw_how.v2 import HowAdviceRequestV2 as ExternalAdviceRequestV2

        external = ExternalAdviceRequestV2.validate_wire_json(request.to_wire_json())
        advice = self.engine.advise(external)
        validated = HowAdviceBundleV2.validate_wire_json(advice.model_dump_json())
        self._advice[validated.advice_id] = validated
        return validated

    def explain(self, advice_id: str) -> dict[str, Any]:
        advice = self._advice.get(advice_id)
        if advice is None:
            raise HowUnavailableError(f"advice not found: {advice_id}")
        if advice.explanation is None:
            raise HowUnavailableError(f"advice has no structured explanation: {advice_id}")
        return advice.explanation.model_dump(mode="json")

    def submit_feedback(self, feedback: KnowledgeUsageFeedbackV1) -> bool:
        from rosclaw_how.v2 import KnowledgeUsageFeedbackV1 as ExternalFeedbackV1

        return bool(
            self.engine.submit_feedback(
                ExternalFeedbackV1.validate_wire_json(feedback.to_wire_json())
            )
        )


def _external_reference_pack(know_client: Any, **kwargs: Any):
    from rosclaw_how.v2 import ReferenceContextV2, ReferencePackV2

    context = kwargs["context"]
    core_context = __import__(
        "rosclaw.knowledge.contracts", fromlist=["ReferenceContextV2"]
    ).ReferenceContextV2.validate_wire_json(context.model_dump_json())
    pack = know_client.reference_pack(
        query=kwargs["query"],
        context=core_context,
        top_k=kwargs["top_k"],
        token_budget=kwargs["token_budget"],
    )
    # Import above also asserts the How-side contract exists.
    assert ReferenceContextV2 is not None
    return ReferencePackV2.validate_wire_json(pack.model_dump_json())


class DisabledHowClient:
    def health(self) -> dict[str, Any]:
        return {"status": "disabled", "transport": "disabled", "advisory_only": True}

    def doctor(self) -> dict[str, Any]:
        return {
            "status": "disabled",
            "transport": "disabled",
            "action_authority": False,
        }

    def explain(self, advice_id: str) -> dict[str, Any]:
        raise HowUnavailableError("How is disabled")

    def advise(self, request: HowAdviceRequestV2) -> HowAdviceBundleV2:
        raise HowUnavailableError("How is disabled")

    def submit_feedback(self, feedback: KnowledgeUsageFeedbackV1) -> bool:
        raise HowUnavailableError("How is disabled")
