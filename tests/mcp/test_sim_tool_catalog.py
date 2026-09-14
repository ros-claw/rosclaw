"""P0_SIM_TOOLS 目录/安全等级/注册测试（PR-MH6，规格 §25/§26/§29，红→绿）。

全部 sim tool ≤ S1，usable_for_real_execution=false，零 REAL permit。
"""

from __future__ import annotations

from rosclaw.agent.tool_catalog import (
    MCP_TOOL_SAFETY_LEVELS,
    P0_AGENT_MCP_TOOLS,
    P0_SIM_TOOLS,
    compact_safety_level,
)
from rosclaw.mcp import tools as mcp_tools

EXPECTED_SIM_TOOLS = (
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
)

S0_TOOLS = {"sim_get_capabilities", "sim_inspect_model", "sim_observe", "sim_compare"}


def test_sim_tools_catalog_complete() -> None:
    assert set(P0_SIM_TOOLS) == set(EXPECTED_SIM_TOOLS)


def test_sim_tools_in_agent_surface() -> None:
    for name in EXPECTED_SIM_TOOLS:
        assert name in P0_AGENT_MCP_TOOLS


def test_sim_tools_safety_levels_bounded() -> None:
    for name in EXPECTED_SIM_TOOLS:
        level = MCP_TOOL_SAFETY_LEVELS.get(name)
        assert level is not None, name
        compact = compact_safety_level(name)
        assert compact in ("S0", "S1"), f"{name} must be <= S1, got {level}"
        if name in S0_TOOLS:
            assert level == "S0_READ_ONLY", name
        else:
            assert level == "S1_SIMULATION_ONLY", name


def test_sim_tools_registered_in_p0_tools() -> None:
    registered = {func.__name__ for func in mcp_tools.P0_TOOLS}
    for name in EXPECTED_SIM_TOOLS:
        assert name in registered, name


def test_sandbox_run_kept_for_compatibility() -> None:
    """规格 §27：sandbox_run 保留（legacy compatibility facade）。"""
    assert "sandbox_run" in P0_AGENT_MCP_TOOLS
    assert MCP_TOOL_SAFETY_LEVELS["sandbox_run"] == "S1_SIMULATION_ONLY"
