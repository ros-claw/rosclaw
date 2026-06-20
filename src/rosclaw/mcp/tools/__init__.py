"""P0 MCP tool registration."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from rosclaw.mcp.adapters.runtime_client import RuntimeClient
from rosclaw.mcp.schemas.common import MCPError, make_error, make_response

ToolFunc = Callable[..., Any]

# Populated by the server after building the RuntimeClient.
_CLIENT: RuntimeClient | None = None


def set_client(client: RuntimeClient) -> None:
    """Inject the shared RuntimeClient before serving requests."""
    global _CLIENT
    _CLIENT = client


def _client() -> RuntimeClient:
    if _CLIENT is None:
        raise MCPError("CLIENT_NOT_INITIALIZED", "RuntimeClient has not been set for MCP tools.")
    return _CLIENT


def _tool_wrapper(name: str, coro_factory: Callable[..., Any]) -> ToolFunc:
    """Wrap a tool coroutine so it returns a JSON envelope and catches errors."""
    import functools
    import uuid

    @functools.wraps(coro_factory)
    async def wrapper(*args: Any, **kwargs: Any) -> str:
        trace_id = str(uuid.uuid4())
        try:
            data = await coro_factory(*args, **kwargs)
            envelope = make_response(data, trace_id=trace_id)
        except MCPError as exc:
            envelope = exc.to_envelope(trace_id=trace_id)
        except Exception as exc:  # noqa: BLE001
            envelope = make_error(
                "RUNTIME_ERROR",
                f"{name} failed: {exc}",
                trace_id=trace_id,
            )
        return json.dumps(envelope, ensure_ascii=False, default=str)

    wrapper.__name__ = name
    wrapper.__doc__ = coro_factory.__doc__
    return wrapper


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def _get_robot_state() -> dict[str, Any]:
    """Return current body state, readiness, and risk summary."""
    return await _client().get_robot_state()


async def _list_skills(skill_type: str | None = None, full_ids: bool = False) -> dict[str, Any]:
    """List skills registered in the runtime."""
    return await _client().list_skills(skill_type=skill_type, full_ids=full_ids)


async def _query_memory(instruction: str, limit: int = 5, outcome_filter: str | None = None) -> dict[str, Any]:
    """Query past experiences similar to the given instruction."""
    return await _client().query_memory(instruction, limit=limit, outcome_filter=outcome_filter)


async def _validate_trajectory(trajectory: list[list[float]], safety_level: str = "MODERATE") -> dict[str, Any]:
    """Validate a trajectory through the firewall gate and sandbox simulation."""
    return await _client().validate_trajectory(trajectory, safety_level=safety_level)


async def _sandbox_run(joint_positions: list[float]) -> dict[str, Any]:
    """Run one MuJoCo simulation step with the given joint positions."""
    return await _client().sandbox_run(joint_positions)


async def _practice_query(episode_id: str | None = None, limit: int = 10) -> dict[str, Any]:
    """List practice episodes or fetch one by ID."""
    return await _client().practice_query(episode_id=episode_id, limit=limit)


async def _emergency_stop(reason: str) -> dict[str, Any]:
    """Trigger an emergency stop. This is the only destructive P0 tool."""
    return await _client().emergency_stop(reason)


async def _list_bodies() -> dict[str, Any]:
    """List all registered bodies in the workspace."""
    return await _client().list_bodies()


async def _get_body(body_id: str) -> dict[str, Any]:
    """Get registry entry and effective body snapshot for a specific body."""
    return await _client().get_body(body_id)


async def _switch_body(body_id: str) -> dict[str, Any]:
    """Switch the active body pointer (registry-only config change)."""
    return await _client().switch_body(body_id)


async def _list_body_history(body_id: str) -> dict[str, Any]:
    """List snapshot history for a specific body."""
    return await _client().list_body_history(body_id)


async def _check_skill_compatibility() -> dict[str, Any]:
    """Check skill compatibility for the current body."""
    return await _client().check_skill_compatibility()


async def _fleet_skill_compatibility() -> dict[str, Any]:
    """Aggregate skill compatibility across all bodies in the workspace."""
    return await _client().fleet_skill_compatibility()


# Expose wrapped tool functions for FastMCP registration.
get_robot_state = _tool_wrapper("get_robot_state", _get_robot_state)
list_skills = _tool_wrapper("list_skills", _list_skills)
query_memory = _tool_wrapper("query_memory", _query_memory)
validate_trajectory = _tool_wrapper("validate_trajectory", _validate_trajectory)
sandbox_run = _tool_wrapper("sandbox_run", _sandbox_run)
practice_query = _tool_wrapper("practice_query", _practice_query)
emergency_stop = _tool_wrapper("emergency_stop", _emergency_stop)
list_bodies = _tool_wrapper("list_bodies", _list_bodies)
get_body = _tool_wrapper("get_body", _get_body)
switch_body = _tool_wrapper("switch_body", _switch_body)
list_body_history = _tool_wrapper("list_body_history", _list_body_history)
check_skill_compatibility = _tool_wrapper("check_skill_compatibility", _check_skill_compatibility)
fleet_skill_compatibility = _tool_wrapper("fleet_skill_compatibility", _fleet_skill_compatibility)

P0_TOOLS: list[ToolFunc] = [
    get_robot_state,
    list_skills,
    query_memory,
    validate_trajectory,
    sandbox_run,
    practice_query,
    emergency_stop,
    list_bodies,
    get_body,
    switch_body,
    list_body_history,
    check_skill_compatibility,
    fleet_skill_compatibility,
]

__all__ = ["P0_TOOLS", "set_client", "ToolFunc"]
