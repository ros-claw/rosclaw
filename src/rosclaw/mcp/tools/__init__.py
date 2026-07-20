"""P0 MCP tool registration."""

from __future__ import annotations

import asyncio
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
    """Return a recursively redacted copy of arguments safe for audit logs."""

    sensitive = {
        "approval_id",
        "api_key",
        "apikey",
        "auth",
        "authorization",
        "credential",
        "credentials",
        "key",
        "password",
        "permit",
        "permit_id",
        "private_key",
        "secret",
        "token",
    }
    sensitive_suffixes = (
        "_api_key",
        "_apikey",
        "_credential",
        "_password",
        "_permit",
        "_private_key",
        "_secret",
        "_token",
    )

    def redact(value: Any, *, depth: int = 0) -> Any:
        if depth >= 12:
            return "<REDACTED_DEPTH_LIMIT>"
        if isinstance(value, dict):
            redacted: dict[Any, Any] = {}
            for key, item in value.items():
                normalized = key.lower().replace("-", "_") if isinstance(key, str) else ""
                if normalized in sensitive or normalized.endswith(sensitive_suffixes):
                    redacted[key] = "<REDACTED>"
                else:
                    redacted[key] = redact(item, depth=depth + 1)
            return redacted
        if isinstance(value, list):
            return [redact(item, depth=depth + 1) for item in value]
        if isinstance(value, tuple):
            return tuple(redact(item, depth=depth + 1) for item in value)
        return value

    return redact(arguments)


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
        log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        log_dir.chmod(0o700)
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
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(log_dir / "audit.jsonl", flags, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "a", encoding="utf-8") as file:
                descriptor = -1
                file.write(line + "\n")
        finally:
            if descriptor >= 0:
                os.close(descriptor)
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
    """Request daemon E-Stop and return its physical-stop evidence."""
    return await _client().emergency_stop(reason)


# ---------------------------------------------------------------------------
# rosclawd control-plane tools
# ---------------------------------------------------------------------------


async def _get_runtime_status() -> dict[str, Any]:
    """Return rosclawd health, queue, driver, permit, and boundary status."""
    return await _client().get_runtime_status()


async def _request_action(
    capability_id: str,
    arguments: dict[str, Any],
    execution_mode: str = "SHADOW",
    body_snapshot_hash: str = "",
    principal_id: str = "",
    approval_id: str | None = None,
    body_id: str | None = None,
    action_id: str | None = None,
    required_evidence: str = "TASK_VERIFIED",
    timeout_sec: float = 30.0,
    wait_timeout_sec: float = 2.0,
) -> dict[str, Any]:
    """Submit one structured action; rosclawd independently verifies authorization."""
    return await _client().request_action(
        capability_id=capability_id,
        arguments=arguments,
        execution_mode=execution_mode,
        body_snapshot_hash=body_snapshot_hash,
        principal_id=principal_id,
        approval_id=approval_id,
        body_id=body_id,
        action_id=action_id,
        required_evidence=required_evidence,
        timeout_sec=timeout_sec,
        wait_timeout_sec=wait_timeout_sec,
    )


async def _get_action_status(action_id: str) -> dict[str, Any]:
    """Read daemon queue state and any terminal ExecutionReceipt."""
    return await _client().get_action_status(action_id)


async def _cancel_action(action_id: str) -> dict[str, Any]:
    """Cancel queued work without claiming that active hardware stopped."""
    return await _client().cancel_action(action_id)


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
# Product workflow tools (simulation and receipt access only)
# ---------------------------------------------------------------------------


async def _get_product_status() -> dict[str, Any]:
    """Return the canonical release, support tiers, and evidence boundary."""
    from rosclaw.product.status import load_product_status

    return {
        "execution_mode": "NONE",
        "trust_level": "DECLARED_AND_VALIDATED",
        "usable_for_real_execution": False,
        "product_status": load_product_status(),
    }


async def _list_product_demos() -> dict[str, Any]:
    """List official evidence-bearing product demos."""
    from rosclaw.product.demo import list_demos

    return {
        "execution_mode": "NONE",
        "trust_level": "DECLARED_AND_VALIDATED",
        "usable_for_real_execution": False,
        "demos": [demo.to_dict() for demo in list_demos()],
    }


async def _run_product_demo(
    demo_id: str = "ur5e-reach",
    target: list[float] | None = None,
    max_steps: int = 1200,
    tolerance_m: float = 0.008,
    seed: int = 0,
) -> dict[str, Any]:
    """Run an official simulation demo and persist its ExecutionReceipt."""
    from rosclaw.product.demo import DemoConfigurationError, DemoNotFoundError, run_demo

    target_tuple: tuple[float, float, float] | None = None
    if target is not None:
        if len(target) != 3:
            raise MCPError("INVALID_ARGUMENT", "target must contain exactly three coordinates.")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in target):
            raise MCPError(
                "INVALID_ARGUMENT",
                "target coordinates must be finite numbers.",
            )
        target_tuple = (float(target[0]), float(target[1]), float(target[2]))

    try:
        receipt, receipt_path = await asyncio.to_thread(
            run_demo,
            demo_id,
            target=target_tuple,
            max_steps=max_steps,
            tolerance_m=tolerance_m,
            seed=seed,
            actor_id="rosclaw-mcp",
            agent_framework=_AGENT_CLIENT,
        )
    except DemoNotFoundError as exc:
        raise MCPError("DEMO_NOT_FOUND", str(exc)) from exc
    except DemoConfigurationError as exc:
        raise MCPError("INVALID_ARGUMENT", str(exc)) from exc

    payload = receipt.to_dict()
    return {
        "execution_mode": payload.get("execution_mode"),
        "trust_level": payload.get("trust_level"),
        "usable_for_real_execution": False,
        "receipt": payload,
        "receipt_path": str(receipt_path),
    }


async def _get_execution_receipt(run_reference: str = "latest") -> dict[str, Any]:
    """Read and integrity-check a persisted ExecutionReceipt."""
    from rosclaw.product.runs import ProductRunStore, RunNotFoundError, RunStoreError

    try:
        receipt, receipt_path = ProductRunStore().load(run_reference)
    except RunNotFoundError as exc:
        raise MCPError("RUN_NOT_FOUND", str(exc)) from exc
    except RunStoreError as exc:
        raise MCPError("RUN_INTEGRITY_ERROR", str(exc)) from exc
    return {
        "execution_mode": receipt.get("execution_mode"),
        "trust_level": receipt.get("trust_level"),
        "usable_for_real_execution": False,
        "receipt": receipt,
        "receipt_path": str(receipt_path),
    }


async def _explain_execution(run_reference: str = "latest") -> dict[str, Any]:
    """Explain what ROSClaw requested, guarded, executed, and verified."""
    from rosclaw.product.explain import explain_receipt
    from rosclaw.product.runs import ProductRunStore, RunNotFoundError, RunStoreError

    try:
        receipt, receipt_path = ProductRunStore().load(run_reference)
    except RunNotFoundError as exc:
        raise MCPError("RUN_NOT_FOUND", str(exc)) from exc
    except RunStoreError as exc:
        raise MCPError("RUN_INTEGRITY_ERROR", str(exc)) from exc
    return {
        "execution_mode": receipt.get("execution_mode"),
        "trust_level": receipt.get("trust_level"),
        "usable_for_real_execution": False,
        "explanation": explain_receipt(receipt, receipt_path),
    }


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
get_runtime_status = _tool_wrapper("get_runtime_status", _get_runtime_status)
request_action = _tool_wrapper("request_action", _request_action)
get_action_status = _tool_wrapper("get_action_status", _get_action_status)
cancel_action = _tool_wrapper("cancel_action", _cancel_action)
get_body_profile = _tool_wrapper("get_body_profile", _get_body_profile)
get_body_state = _tool_wrapper("get_body_state", _get_body_state)
list_body_capabilities = _tool_wrapper("list_body_capabilities", _list_body_capabilities)
query_body = _tool_wrapper("query_body", _query_body)
validate_body_action = _tool_wrapper("validate_body_action", _validate_body_action)
get_calibration_status = _tool_wrapper("get_calibration_status", _get_calibration_status)
get_product_status = _tool_wrapper("get_product_status", _get_product_status)
list_product_demos = _tool_wrapper("list_product_demos", _list_product_demos)
run_product_demo = _tool_wrapper("run_product_demo", _run_product_demo)
get_execution_receipt = _tool_wrapper("get_execution_receipt", _get_execution_receipt)
explain_execution = _tool_wrapper("explain_execution", _explain_execution)
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
    get_runtime_status,
    request_action,
    get_action_status,
    cancel_action,
    get_product_status,
    list_product_demos,
    run_product_demo,
    get_execution_receipt,
    explain_execution,
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
    "get_runtime_status",
    "request_action",
    "get_action_status",
    "cancel_action",
    "get_body_profile",
    "get_body_state",
    "list_body_capabilities",
    "query_body",
    "validate_body_action",
    "get_calibration_status",
    "get_product_status",
    "list_product_demos",
    "run_product_demo",
    "get_execution_receipt",
    "explain_execution",
    "list_bodies",
    "get_body",
    "switch_body",
    "list_body_history",
    "check_skill_compatibility",
    "fleet_skill_compatibility",
]
