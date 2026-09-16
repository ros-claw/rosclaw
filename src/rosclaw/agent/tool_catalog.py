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
    "request_guarded_action",
    "get_action_status",
    "get_approval_status",
    "cancel_approval",
    "cancel_action",
)

P0_PRODUCT_TOOLS: tuple[str, ...] = (
    "get_product_status",
    "list_product_demos",
    "run_product_demo",
    "get_execution_receipt",
    "explain_execution",
)

# Skill Runtime 2.0 capability tools (doc §26): agents resolve/invoke
# capabilities without guessing skill names.
P0_CAPABILITY_TOOLS: tuple[str, ...] = (
    "resolve_capability",
    "invoke_capability",
    "get_skill_job",
    "cancel_skill_job",
)

# MuJoCo Simulation Harness tools (PR-MH6, ADR-0014, 规格 §25/§26):
# Agent 原生物理实验面；全部 <= S1，usable_for_real_execution=false。
P0_SIM_TOOLS: tuple[str, ...] = (
    "sim_get_capabilities",
    "sim_load_model",
    "sim_inspect_model",
    "sim_patch_model",
    "sim_snapshot",
    "sim_observe",
    "sim_rollout",
    "sim_audit",
    "sim_compare",
    "sim_render",
    "sim_branch_experiment",
    "sim_compile_world",
)

P0_AGENT_MCP_TOOLS: tuple[str, ...] = (
    P0_CORE_TOOLS
    + P0_BODY_CONTEXT_TOOLS
    + P0_CONTROL_PLANE_TOOLS
    + P0_PRODUCT_TOOLS
    + P0_CAPABILITY_TOOLS
    + P0_SIM_TOOLS
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
    # MuJoCo Simulation Harness（规格 §26）：只读 S0，写/跑/审 S1；
    # 永不生成 REAL permit。
    "sim_get_capabilities": "S0_READ_ONLY",
    "sim_inspect_model": "S0_READ_ONLY",
    "sim_observe": "S0_READ_ONLY",
    "sim_compare": "S0_READ_ONLY",
    "sim_load_model": "S1_SIMULATION_ONLY",
    "sim_patch_model": "S1_SIMULATION_ONLY",
    "sim_snapshot": "S1_SIMULATION_ONLY",
    "sim_rollout": "S1_SIMULATION_ONLY",
    "sim_audit": "S1_SIMULATION_ONLY",
    "sim_render": "S1_SIMULATION_ONLY",
    "sim_branch_experiment": "S1_SIMULATION_ONLY",
    "sim_compile_world": "S1_SIMULATION_ONLY",
    "get_runtime_status": "S0_READ_ONLY",
    "request_action": "S3_GUARDED_ACTION",
    "request_guarded_action": "S3_GUARDED_ACTION",
    "get_action_status": "S0_READ_ONLY",
    "get_approval_status": "S0_READ_ONLY",
    "cancel_approval": "S0_CONFIG",
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
    "rosclaw_know_research": "S0_READ_ONLY",
    "rosclaw_know_build_reference_pack": "S0_READ_ONLY",
    "rosclaw_know_open_reference_pack": "S0_READ_ONLY",
    "rosclaw_how_advice": "S0_ADVISORY",
    # Skill Runtime 2.0 capability tools (doc §26): resolution is read-only;
    # invoke only ever creates an AWAITING_APPROVAL job — execution needs a
    # plan-hash approval plus local-TTY authorization (§21/§23).
    "resolve_capability": "S0_READ_ONLY",
    "get_skill_job": "S0_READ_ONLY",
    "invoke_capability": "S3_GUARDED_ACTION",
    "cancel_skill_job": "S3_GUARDED_ACTION",
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
