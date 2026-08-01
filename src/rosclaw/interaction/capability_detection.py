"""MCP client capability detection kept out of robot-specific servers."""

from __future__ import annotations

from typing import Any

from rosclaw.interaction.schemas import InteractionCapabilities


def _session(ctx: Any) -> Any | None:
    try:
        return ctx.request_context.session
    except (AttributeError, ValueError):
        return getattr(ctx, "session", None)


def detect_interaction_capabilities(ctx: Any) -> InteractionCapabilities:
    """Read negotiated client capabilities without guessing from method presence."""

    session = _session(ctx)
    client_params = getattr(session, "client_params", None)
    capabilities = getattr(client_params, "capabilities", None)
    elicitation = getattr(capabilities, "elicitation", None)
    tasks = getattr(capabilities, "tasks", None)
    task_requests = getattr(tasks, "requests", None)
    task_elicitation = getattr(task_requests, "elicitation", None)
    task_cancel = getattr(tasks, "cancel", None)
    return InteractionCapabilities(
        form_elicitation=getattr(elicitation, "form", None) is not None,
        url_elicitation=getattr(elicitation, "url", None) is not None,
        asynchronous_elicitation=getattr(task_elicitation, "create", None) is not None,
        progress=callable(getattr(ctx, "report_progress", None)),
        cancellation=task_cancel is not None,
    )


def request_identity(ctx: Any) -> str:
    """Build a non-user-controlled principal label from the MCP connection."""

    session = _session(ctx)
    client_params = getattr(session, "client_params", None)
    client_info = getattr(client_params, "clientInfo", None)
    if client_info is None:
        client_info = getattr(client_params, "client_info", None)
    name = str(getattr(client_info, "name", "mcp-client"))
    request_id = str(getattr(ctx, "request_id", "unknown"))
    return f"mcp:{name}:{request_id}"
