"""MCP tool registration and handlers for rosclaw.sense."""

from __future__ import annotations

from typing import Any


def register_sense_tools(tools: dict[str, dict[str, Any]]) -> None:
    """Register sense MCP tools in the provided tools dictionary."""
    tools["get_body_sense"] = {
        "name": "get_body_sense",
        "description": (
            "Get the current semantic body sense snapshot: overall status, "
            "blocked/degraded capabilities, main risks, and recommended actions."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    }
    tools["get_body_readiness"] = {
        "name": "get_body_readiness",
        "description": (
            "Check whether the robot is ready to perform a specific task or "
            "capability (e.g. g1_kick_ball). Returns status and failed requirements."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Task or capability name to check",
                },
            },
            "required": ["task"],
        },
    }
    tools["explain_body_block"] = {
        "name": "explain_body_block",
        "description": (
            "Explain why a task is blocked or degraded, including current values, "
            "required thresholds, and recommended actions."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Task or capability name to explain",
                },
            },
            "required": ["task"],
        },
    }


def _sense_available(hub: Any) -> bool:
    return (
        hub.runtime is not None
        and getattr(hub.runtime, "sense", None) is not None
    )


def handle_get_body_sense(hub: Any, _arguments: dict[str, Any]) -> dict[str, Any]:
    if not _sense_available(hub):
        return {
            "status": "unavailable",
            "error": "Sense module is not available",
        }
    sense = hub.runtime.sense
    snapshot = sense.get_latest_sense() or sense.tick()
    return {
        "status": "ok",
        "body_sense": snapshot.to_dict(),
    }


def handle_get_body_readiness(hub: Any, arguments: dict[str, Any]) -> dict[str, Any]:
    task = arguments.get("task")
    if not task:
        return {"status": "error", "error": "Missing required argument: task"}
    if not _sense_available(hub):
        return {
            "status": "unavailable",
            "error": "Sense module is not available",
        }
    readiness = hub.runtime.sense.get_readiness(task=task)
    return {
        "status": "ok",
        "task": task,
        "readiness": readiness.to_dict(),
    }


def handle_explain_body_block(hub: Any, arguments: dict[str, Any]) -> dict[str, Any]:
    task = arguments.get("task")
    if not task:
        return {"status": "error", "error": "Missing required argument: task"}
    if not _sense_available(hub):
        return {
            "status": "unavailable",
            "error": "Sense module is not available",
        }
    explanation = hub.runtime.sense.explain_block(task)
    return {
        "status": "ok",
        "task": task,
        "explanation": explanation,
    }
