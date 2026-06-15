"""rosclaw.how.client — HTTP client for a private/local rosclaw-how service.

This module is intentionally dependency-light: it uses only ``urllib`` from
the standard library so the open-source ``rosclaw`` package does not need to
declare a private ``rosclaw-how`` dependency.  Users who have access to the
private repo install it separately and run the service locally; this client
connects to it.

If the service is unreachable or unhealthy, Runtime falls back to the local
:class:`rosclaw.how.engine.HeuristicEngine`.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any

from rosclaw.how.intervention import InterventionDecision, InterventionRequest

logger = logging.getLogger("rosclaw.how.client")


class HowClient:
    """Thin HTTP client wrapping the rosclaw-how REST endpoints.

    Args:
        base_url: Base URL of the running rosclaw-how service,
            e.g. ``http://127.0.0.1:8088``.
        api_key: Optional ``X-API-Key`` for authenticated endpoints.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self._headers["X-API-Key"] = api_key

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make a JSON HTTP request and return the parsed response."""
        url = f"{self.base_url}{path}"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload else None
        req = urllib.request.Request(
            url,
            data=data,
            headers=self._headers,
            method=method,
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = resp.read().decode("utf-8")
            if not body:
                return {}
            return json.loads(body)

    async def initialize(self) -> None:
        """Verify the service is healthy; raise on degraded/unreachable."""
        health = self._request("GET", "/healthz")
        status = health.get("status")
        if status != "ok":
            reasons = health.get("degraded_reasons", [])
            raise RuntimeError(f"rosclaw-how health not ok: {status} (reasons={reasons})")
        logger.info("HowClient connected to %s", self.base_url)

    async def shutdown(self) -> None:
        """No persistent connection to close."""
        return None

    async def suggest_recovery(
        self,
        error_log: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Call ``POST /wiki/v1/prompt/build`` and map to a rule dict.

        The returned shape mirrors :meth:`rosclaw.how.engine.HeuristicEngine.suggest_recovery`
        so callers like :class:`rosclaw.how.recovery.RecoveryEngine` can use it
        unchanged.
        """
        if not error_log:
            return None

        body: dict[str, Any] = {
            "error_log": error_log,
            "previous_scores": [],
        }
        if context:
            run_id = context.get("request_id") or context.get("episode_id")
            if run_id:
                body["run_id"] = str(run_id)

        try:
            resp = self._request("POST", "/wiki/v1/prompt/build", body)
        except Exception as exc:  # noqa: BLE001
            logger.warning("HowClient suggest_recovery failed: %s", exc)
            return None

        return self._rule_from_prompt_build(resp)

    def _rule_from_prompt_build(self, resp: dict[str, Any]) -> dict[str, Any]:
        """Convert a /prompt/build response into HeuristicEngine rule shape."""
        strategy = str(resp.get("strategy", "CATALYST"))
        priority_map = {
            "SAFETY": 3,
            "STOP_UNSAFE": 4,
            "RESOURCE_REPAIR": 2,
            "CATALYST": 1,
            "FREE_EXPLORATION": 0,
            "ABSTAIN": 0,
        }
        return {
            "rule_id": resp.get("pattern_id") or strategy,
            "condition": resp.get("matched_symptom") or resp.get("symptom") or "",
            "action": resp.get("prompt_snippet", ""),
            "priority": priority_map.get(strategy, 1),
            "source": f"how_{strategy.lower()}",
            "injected": bool(resp.get("injected", False)),
            "_raw": resp,
        }

    async def generate_recovery_hint(
        self,
        failure_type: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Canonical recovery-hint API used by Runtime failure handlers."""
        rule = await self.suggest_recovery(failure_type, context)
        if rule is None or not rule.get("injected"):
            return None
        return {
            "hint": rule["action"],
            "rule_id": rule["rule_id"],
            "priority": rule["priority"],
            "source": rule["source"],
        }

    async def decide_recovery(
        self,
        request: InterventionRequest,
        *,
        recent_pattern_id: str | None = None,
    ) -> tuple[InterventionDecision, str | None]:
        """Call ``POST /runtime/v1/intervene`` and return a local decision model."""
        body = request.model_dump(exclude_none=True)
        try:
            resp = self._request("POST", "/runtime/v1/intervene", body)
        except Exception as exc:  # noqa: BLE001
            logger.warning("HowClient decide_recovery failed: %s", exc)
            raise

        decision = InterventionDecision(**resp)
        rule_id: str | None = None
        if decision.injected:
            rule_id = decision.pattern_id or recent_pattern_id
        return decision, rule_id

    async def record_outcome(self, rule_id: str, success: bool) -> bool:
        """Outcome tracking is service-side via /prompt/feedback; no-op here."""
        logger.debug(
            "HowClient.record_outcome is a no-op without an injection_id (rule_id=%s success=%s)",
            rule_id,
            success,
        )
        return False

    async def seed_defaults(self) -> int:
        """Default rules live in the service; nothing to seed locally."""
        return 0

    async def get_retry_plan(
        self,
        failure_type: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Delegate to the local RecoveryEngine using service-provided rule."""
        from rosclaw.how.recovery import RecoveryEngine

        rule = await self.suggest_recovery(failure_type, context)
        if rule is None:
            return None
        re = RecoveryEngine(self)
        return re.build_retry_plan(failure_type, rule, context)

    def get_stats(self) -> dict[str, Any]:
        """Return aggregated per-pattern stats from ``GET /wiki/v1/stats``."""
        try:
            return self._request("GET", "/wiki/v1/stats")
        except Exception as exc:  # noqa: BLE001
            logger.warning("HowClient get_stats failed: %s", exc)
            return {}
