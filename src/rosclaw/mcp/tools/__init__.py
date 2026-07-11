"""P0 MCP tool registration."""

from __future__ import annotations

import functools
import inspect
import json
import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rosclaw.agent.tool_catalog import MCP_TOOL_SAFETY_LEVELS
from rosclaw.firstboot.workspace import get_rosclaw_home
from rosclaw.mcp.adapters.runtime_client import RuntimeClient
from rosclaw.mcp.schemas.common import MCPError, make_error, make_response

ToolFunc = Callable[..., Any]

# Populated by the server after building the RuntimeClient.
_CLIENT: RuntimeClient | None = None

# Populated by the server so audit logs can include project/runtime context.
_PROJECT_ROOT: str | None = None
_RUNTIME_PROFILE: str = "default"
_AGENT_CLIENT: str = "claude-code"


def set_client(client: RuntimeClient) -> None:
    """Inject the shared RuntimeClient before serving requests."""
    global _CLIENT
    _CLIENT = client


def set_context(
    *,
    project_root: str | None = None,
    runtime_profile: str | None = None,
    agent_client: str | None = None,
) -> None:
    """Inject server-level context used for audit logging."""
    global _PROJECT_ROOT, _RUNTIME_PROFILE, _AGENT_CLIENT
    if project_root is not None:
        _PROJECT_ROOT = project_root
    if runtime_profile is not None:
        _RUNTIME_PROFILE = runtime_profile
    if agent_client is not None:
        _AGENT_CLIENT = agent_client


def _client() -> RuntimeClient:
    if _CLIENT is None:
        raise MCPError("CLIENT_NOT_INITIALIZED", "RuntimeClient has not been set for MCP tools.")
    return _CLIENT


def _redact_for_audit(arguments: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of arguments safe for the audit log."""
    sensitive = {"token", "password", "secret", "api_key", "apikey", "auth", "key"}
    out: dict[str, Any] = {}
    for key, value in arguments.items():
        if isinstance(key, str) and key.lower() in sensitive:
            out[key] = "<REDACTED>"
        else:
            out[key] = value
    return out


def _audit(
    trace_id: str,
    tool: str,
    arguments: dict[str, Any],
    response: dict[str, Any],
    latency_ms: float,
) -> None:
    """Append one JSON line to ~/.rosclaw/logs/mcp/audit.jsonl."""
    home = os.environ.get("ROSCLAW_HOME", str(get_rosclaw_home()))
    log_dir = Path(home) / "logs" / "mcp"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {
                "trace_id": trace_id,
                "timestamp": make_response({})["timestamp"],
                "agent_client": _AGENT_CLIENT,
                "project_root": _PROJECT_ROOT or "",
                "runtime_profile": _RUNTIME_PROFILE,
                "tool": tool,
                "input_redacted": _redact_for_audit(arguments),
                "ok": response.get("ok", False),
                "latency_ms": round(latency_ms, 3),
                "safety_level": MCP_TOOL_SAFETY_LEVELS.get(tool, "UNKNOWN"),
            },
            ensure_ascii=False,
            default=str,
        )
        with (log_dir / "audit.jsonl").open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        # Audit must never break a tool call.
        pass


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
        started = time.perf_counter()
        try:
            data = await coro_factory(*args, **kwargs)
            envelope = make_response(data, trace_id=trace_id, runtime_profile=_RUNTIME_PROFILE)
        except MCPError as exc:
            envelope = exc.to_envelope(trace_id=trace_id, runtime_profile=_RUNTIME_PROFILE)
        except Exception as exc:  # noqa: BLE001
            envelope = make_error(
                "RUNTIME_ERROR",
                f"{name} failed: {exc}",
                trace_id=trace_id,
                runtime_profile=_RUNTIME_PROFILE,
            )
        latency_ms = (time.perf_counter() - started) * 1000
        _audit(trace_id, name, kwargs, envelope, latency_ms)
        return json.dumps(envelope, ensure_ascii=False, default=str)

    # Preserve the original parameter schema for MCP discovery, but force the
    # return annotation to ``str`` so FastMCP does not try to validate the JSON
    # envelope as structured output. Remove ``__wrapped__`` so introspection does
    # not follow the chain back to the original dict-returning signature.
    original_sig = inspect.signature(coro_factory)
    wrapper_any: Any = wrapper
    wrapper_any.__signature__ = original_sig.replace(return_annotation=str)
    wrapper_any.__annotations__ = {**coro_factory.__annotations__, "return": str}
    wrapper_any.__name__ = name
    wrapper_any.__dict__.pop("__wrapped__", None)
    return wrapper


# ---------------------------------------------------------------------------
# P0 tool implementations
# ---------------------------------------------------------------------------


async def _get_robot_state() -> dict[str, Any]:
    """Return current body state, readiness, and risk summary."""
    return await _client().get_robot_state()


async def _list_skills(skill_type: str | None = None, full_ids: bool = False) -> dict[str, Any]:
    """List skills registered in the runtime."""
    return await _client().list_skills(skill_type=skill_type, full_ids=full_ids)


async def _query_memory(
    instruction: str, limit: int = 5, outcome_filter: str | None = None
) -> dict[str, Any]:
    """Query past experiences similar to the given instruction."""
    return await _client().query_memory(instruction, limit=limit, outcome_filter=outcome_filter)


async def _validate_trajectory(
    trajectory: list[list[float]], safety_level: str = "MODERATE"
) -> dict[str, Any]:
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


# ---------------------------------------------------------------------------
# P0 body tools (e-URDF / Body Runtime)
# ---------------------------------------------------------------------------


async def _get_body_profile() -> dict[str, Any]:
    """Return a static profile summary of the current body."""
    return await _client().get_body_profile()


async def _get_body_state(include_runtime: bool = True) -> dict[str, Any]:
    """Return current body safety state and capability matrix."""
    return await _client().get_body_state(include_runtime=include_runtime)


async def _list_body_capabilities(status: str = "all") -> dict[str, Any]:
    """List capabilities grouped by status."""
    return await _client().list_body_capabilities(status=status)


async def _query_body(question: str) -> dict[str, Any]:
    """Answer a natural-language question about the current body."""
    return await _client().query_body(question)


async def _validate_body_action(
    action: str,
    capability_id: str,
    risk: str = "medium",
) -> dict[str, Any]:
    """Validate a proposed physical action against the current body."""
    return await _client().validate_body_action(action, capability_id, risk=risk)


async def _get_calibration_status(component: str | None = None) -> dict[str, Any]:
    """Return calibration status for the body or a named component."""
    return await _client().get_calibration_status(component=component)


# ---------------------------------------------------------------------------
# Body registry tools (P2 / body scope, not part of P0_TOOLS)
# ---------------------------------------------------------------------------


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
get_body_profile = _tool_wrapper("get_body_profile", _get_body_profile)
get_body_state = _tool_wrapper("get_body_state", _get_body_state)
list_body_capabilities = _tool_wrapper("list_body_capabilities", _list_body_capabilities)
query_body = _tool_wrapper("query_body", _query_body)
validate_body_action = _tool_wrapper("validate_body_action", _validate_body_action)
get_calibration_status = _tool_wrapper("get_calibration_status", _get_calibration_status)
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
    get_body_profile,
    get_body_state,
    list_body_capabilities,
    query_body,
    validate_body_action,
    get_calibration_status,
]

BODY_TOOLS: list[ToolFunc] = [
    list_bodies,
    get_body,
    switch_body,
    list_body_history,
    check_skill_compatibility,
    fleet_skill_compatibility,
]

__all__ = [
    "P0_TOOLS",
    "BODY_TOOLS",
    "set_client",
    "set_context",
    "ToolFunc",
    # Expose individual wrapped tools for tests and P2 registration.
    "get_robot_state",
    "list_skills",
    "query_memory",
    "validate_trajectory",
    "sandbox_run",
    "practice_query",
    "emergency_stop",
    "get_body_profile",
    "get_body_state",
    "list_body_capabilities",
    "query_body",
    "validate_body_action",
    "get_calibration_status",
    "list_bodies",
    "get_body",
    "switch_body",
    "list_body_history",
    "check_skill_compatibility",
    "fleet_skill_compatibility",
]
