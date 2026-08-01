"""Native MCP form elicitation adapter."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class _NativeDecision(BaseModel):
    """Empty form: the MCP protocol's accept/decline action is the decision."""

    model_config = ConfigDict(extra="forbid")


async def request_form_confirmation(ctx: Any, *, message: str) -> str:
    """Return accept, decline, or cancel across supported FastMCP API versions."""

    high_level = getattr(ctx, "elicit", None)
    if callable(high_level):
        result = await high_level(message=message, schema=_NativeDecision)
    else:
        session = ctx.request_context.session
        result = await session.elicit_form(
            message=message,
            requestedSchema={"type": "object", "properties": {}},
            related_request_id=getattr(ctx, "request_id", None),
        )
    action = getattr(result, "action", "cancel")
    return str(getattr(action, "value", action)).lower()
