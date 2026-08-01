"""Unified orchestration of MCP confirmation and guarded dispatch."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from rosclaw.interaction.adapters import request_form_confirmation
from rosclaw.interaction.capability_detection import (
    detect_interaction_capabilities,
    request_identity,
)
from rosclaw.interaction.client import InteractionClient
from rosclaw.interaction.schemas import ActionDisplay

_PRIVATE_RESULT_KEYS = {
    "action_intent_hash",
    "approval_id",
    "approval_request",
    "authorization",
    "authorized_action",
    "operator_confirmation",
    "permit",
    "permit_id",
    "session_id",
}


def _public_result(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _public_result(item)
            for key, item in value.items()
            if str(key).lower() not in _PRIVATE_RESULT_KEYS and "permit" not in str(key).lower()
        }
    if isinstance(value, list):
        return [_public_result(item) for item in value]
    return value


class InteractionCoordinator:
    """Complete an exact prepared action using the best negotiated MCP channel."""

    def __init__(self, client: InteractionClient) -> None:
        self._client = client

    async def complete_prepared(
        self,
        ctx: Any,
        *,
        prepared_action: Any,
        approval_request: Mapping[str, Any],
        wait_timeout_sec: float = 2.0,
    ) -> dict[str, Any]:
        display = ActionDisplay.from_legacy(
            dict(approval_request.get("display") or {}),
            risk_tier=str(approval_request.get("risk_class") or "HIGH"),
        )
        capabilities = detect_interaction_capabilities(ctx)
        public_card = display.model_dump(mode="json")
        if not capabilities.form_elicitation:
            decision = (
                "APPROVAL_PENDING"
                if capabilities.url_elicitation or capabilities.asynchronous_elicitation
                else "APPROVAL_CHANNEL_UNAVAILABLE"
            )
            return {
                "ok": False,
                "decision": decision,
                "error_code": decision,
                "message": (
                    "Operator approval is pending in a supported external channel."
                    if decision == "APPROVAL_PENDING"
                    else "This MCP client has no negotiated operator-confirmation channel."
                ),
                "action_display": public_card,
                "command_dispatched": False,
                "usable_for_real_execution": False,
            }

        await self._progress(ctx, 0.1, "Waiting for operator confirmation")
        try:
            decision = await request_form_confirmation(ctx, message=display.render_text())
            if decision != "accept":
                return {
                    "ok": False,
                    "decision": "CANCELLED",
                    "error_code": "OPERATOR_CONFIRMATION_DECLINED",
                    "message": "The operator declined or cancelled the physical action.",
                    "action_display": public_card,
                    "command_dispatched": False,
                    "usable_for_real_execution": False,
                }
            await self._progress(ctx, 0.5, "Operator confirmed; submitting to ROSClaw")
            result = await self._client.confirm_action(
                prepared_action,
                principal_id=request_identity(ctx),
                confirmation={
                    "accepted": True,
                    "action_intent_hash": approval_request.get("action_intent_hash"),
                    "reason": "Operator accepted the exact action through MCP elicitation",
                    "channel": "mcp_form",
                    "request_id": str(getattr(ctx, "request_id", "unknown")),
                },
                wait_timeout_sec=wait_timeout_sec,
            )
            await self._progress(ctx, 1.0, "ROSClaw accepted the action request")
        except asyncio.CancelledError:
            action_id = str(approval_request.get("action_id") or "")
            if action_id:
                await self._client.cancel_action(action_id)
            raise
        except Exception as exc:  # noqa: BLE001 - public interaction boundary
            return {
                "ok": False,
                "decision": "FAILED",
                "error_code": str(getattr(exc, "code", "INTERACTION_FAILED")),
                "message": str(exc),
                "action_display": public_card,
                "command_dispatched": False,
                "usable_for_real_execution": False,
            }
        public = _public_result(result)
        return {
            **public,
            "interaction": {
                "schema_version": "rosclaw.interaction-result.v1",
                "decision": "CONFIRMED",
                "channel": "mcp_form",
            },
        }

    @staticmethod
    async def _progress(ctx: Any, progress: float, message: str) -> None:
        reporter = getattr(ctx, "report_progress", None)
        if callable(reporter):
            try:
                await reporter(progress=progress, total=1.0, message=message)
            except Exception:  # noqa: BLE001 - progress is advisory only
                return
