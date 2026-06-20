"""P0 MCP tool registration."""

from __future__ import annotations

import functools
import inspect
import json
import uuid
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
    """Wrap a tool coroutine so it returns a JSON envelope and catches errors.

    The wrapper preserves the original tool signature so that FastMCP can expose
    accurate input schemas, but declares a ``str`` return type so the JSON
    envelope is treated as unstructured text content.
    """

    @functools.wraps(
        coro_factory,
        assigned=("__module__", "__name__", "__qualname__", "__doc__"),
    )
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

    # Preserve the original parameter schema for MCP discovery, but force the
    # return annotation to ``str`` so FastMCP does not try to validate the JSON
    # envelope as structured output. Remove ``__wrapped__`` so introspection does
    # not follow the chain back to the original dict-returning signature.
    original_sig = inspect.signature(coro_factory)
    wrapper.__signature__ = original_sig.replace(return_annotation=str)
    wrapper.__annotations__ = {**coro_factory.__annotations__, "return": str}
    wrapper.__name__ = name
    wrapper.__dict__.pop("__wrapped__", None)
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


# Expose wrapped tool functions for FastMCP registration.
get_robot_state = _tool_wrapper("get_robot_state", _get_robot_state)
list_skills = _tool_wrapper("list_skills", _list_skills)
query_memory = _tool_wrapper("query_memory", _query_memory)
validate_trajectory = _tool_wrapper("validate_trajectory", _validate_trajectory)
sandbox_run = _tool_wrapper("sandbox_run", _sandbox_run)
practice_query = _tool_wrapper("practice_query", _practice_query)
emergency_stop = _tool_wrapper("emergency_stop", _emergency_stop)

P0_TOOLS: list[ToolFunc] = [
    get_robot_state,
    list_skills,
    query_memory,
    validate_trajectory,
    sandbox_run,
    practice_query,
    emergency_stop,
]

__all__ = ["P0_TOOLS", "set_client", "ToolFunc"]
