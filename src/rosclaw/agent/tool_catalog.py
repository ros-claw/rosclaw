"""Canonical MCP tool catalog advertised to agent clients."""

from __future__ import annotations

P0_CORE_TOOLS: tuple[str, ...] = (
    "get_robot_state",
    "list_skills",
    "query_memory",
    "validate_trajectory",
    "sandbox_run",
    "practice_query",
    "emergency_stop",
)

P0_BODY_CONTEXT_TOOLS: tuple[str, ...] = (
    "get_body_profile",
    "get_body_state",
    "list_body_capabilities",
    "query_body",
    "validate_body_action",
    "get_calibration_status",
)

P0_CONTROL_PLANE_TOOLS: tuple[str, ...] = (
    "get_runtime_status",
    "request_action",
    "get_action_status",
    "cancel_action",
)

P0_PRODUCT_TOOLS: tuple[str, ...] = (
    "get_product_status",
    "list_product_demos",
    "run_product_demo",
    "get_execution_receipt",
    "explain_execution",
)

P0_AGENT_MCP_TOOLS: tuple[str, ...] = (
    P0_CORE_TOOLS + P0_BODY_CONTEXT_TOOLS + P0_CONTROL_PLANE_TOOLS + P0_PRODUCT_TOOLS
)

MCP_TOOL_SAFETY_LEVELS: dict[str, str] = {
    "get_robot_state": "S0_READ_ONLY",
    "list_skills": "S0_READ_ONLY",
    "query_memory": "S0_READ_ONLY",
    "practice_query": "S0_READ_ONLY",
    "get_body_profile": "S0_READ_ONLY",
    "get_body_state": "S0_READ_ONLY",
    "list_body_capabilities": "S0_READ_ONLY",
    "query_body": "S0_READ_ONLY",
    "validate_body_action": "S0_READ_ONLY",
    "get_calibration_status": "S0_READ_ONLY",
    "validate_trajectory": "S2_VALIDATED_PLAN",
    "sandbox_run": "S1_SIMULATION_ONLY",
    "emergency_stop": "S4_EMERGENCY",
    "get_runtime_status": "S0_READ_ONLY",
    "request_action": "S3_GUARDED_ACTION",
    "get_action_status": "S0_READ_ONLY",
    "cancel_action": "S3_GUARDED_ACTION",
    "get_product_status": "S0_READ_ONLY",
    "list_product_demos": "S0_READ_ONLY",
    "run_product_demo": "S1_SIMULATION_ONLY",
    "get_execution_receipt": "S0_READ_ONLY",
    "explain_execution": "S0_READ_ONLY",
    # Body registry tools are not registered in the P0 server, but keeping
    # levels here prevents audit drift when they are enabled by a later server.
    "list_bodies": "S0_READ_ONLY",
    "get_body": "S0_READ_ONLY",
    "switch_body": "S0_CONFIG",
    "list_body_history": "S0_READ_ONLY",
    "check_skill_compatibility": "S0_READ_ONLY",
    "fleet_skill_compatibility": "S0_READ_ONLY",
}


def compact_safety_level(tool_name: str) -> str:
    """Return the short S-level used in agent context snapshots."""
    level = MCP_TOOL_SAFETY_LEVELS.get(tool_name, "UNKNOWN")
    return level.split("_", 1)[0]


__all__ = [
    "P0_CORE_TOOLS",
    "P0_BODY_CONTEXT_TOOLS",
    "P0_CONTROL_PLANE_TOOLS",
    "P0_PRODUCT_TOOLS",
    "P0_AGENT_MCP_TOOLS",
    "MCP_TOOL_SAFETY_LEVELS",
    "compact_safety_level",
]
