"""Native tool adapter — registers the agentd's built-in P0 tools in the catalog.

Wraps ``BuiltinToolRegistry`` so the existing sim tools flow through the same
descriptor/resolver/evidence path as MCP-discovered capabilities.
"""

from __future__ import annotations

from typing import Any

from rosclaw.agentd.tooling.catalog import ToolCatalog
from rosclaw.agentd.tools import (
    SIM_BODY_TOOL,
    SIM_REACH_TOOL,
    SIM_STATE_TOOL,
    BuiltinToolRegistry,
)
from rosclaw.contracts.agent.tool import (
    ExecutionClass,
    ToolDescriptorV2,
    ToolEvidenceClass,
)

NATIVE_SOURCE = "native:agentd"


def register_native_tools(
    catalog: ToolCatalog, registry: BuiltinToolRegistry, *, simulation: bool = True
) -> None:
    descriptors = [
        ToolDescriptorV2(
            tool_id=SIM_BODY_TOOL,
            source=NATIVE_SOURCE,
            execution_class=ExecutionClass.OBSERVE,
            description="Read the bound body's static profile summary.",
            input_schema={
                "type": "object",
                "properties": {"detail": {"type": "boolean"}},
                "additionalProperties": False,
            },
            supported_modes=["SIMULATION", "SHADOW", "REAL"],
            evidence_class=ToolEvidenceClass.CONFIGURED,
            verifier="schema+configured-label",
            reliability=1.0,
            typical_latency_ms=1,
        ),
    ]
    if simulation:
        # The simulated live-state tool only exists for simulated bodies;
        # for real bodies it is never registered (fail closed, not filtered).
        descriptors.insert(
            0,
            ToolDescriptorV2(
                tool_id=SIM_STATE_TOOL,
                source=NATIVE_SOURCE,
                execution_class=ExecutionClass.OBSERVE,
                description=(
                    "Read the SIMULATED body state (joints, health). "
                    "Evidence class: simulated — never usable as REAL proof."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"verbose": {"type": "boolean"}},
                    "additionalProperties": False,
                },
                supported_modes=["SIMULATION", "SHADOW", "REAL"],
                evidence_class=ToolEvidenceClass.SIMULATED,
                verifier="schema+simulated-label",
                reliability=1.0,
                typical_latency_ms=1,
            ),
        )
        # MuJoCo reach: real physics, no physical side effect -> COMPUTE;
        # simulated bodies only, and only in SIMULATION mode.
        descriptors.append(
            ToolDescriptorV2(
                tool_id=SIM_REACH_TOOL,
                source=NATIVE_SOURCE,
                execution_class=ExecutionClass.COMPUTE,
                description=(
                    "Execute one MuJoCo physics reach with the SIMULATED UR5e "
                    "to a tabletop target (x/y/z meters) and return the verified "
                    "receipt: policy check, physics steps, collision check and "
                    "task success. Evidence class: simulated."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "z": {"type": "number"},
                    },
                    "required": ["x", "y", "z"],
                    "additionalProperties": False,
                },
                supported_modes=["SIMULATION"],
                evidence_class=ToolEvidenceClass.SIMULATED,
                verifier="sandbox-receipt+task-predicate",
                reliability=0.9,
                typical_latency_ms=4000,
            ),
        )
        # 十四审 PR-14.6：SIM 动力学闭环四能力（COMPUTE——确定性仿真/
        # 渲染/验证，无人工审批；SIMULATION 限定；证据 SIMULATED）。
        from rosclaw.agentd.tools import (
            TRAJ_PLAN_TOOL,
            TRAJ_RENDER_TOOL,
            TRAJ_SIMULATE_TOOL,
            TRAJ_VERIFY_TOOL,
        )

        descriptors += [
            ToolDescriptorV2(
                tool_id=TRAJ_PLAN_TOOL,
                source=NATIVE_SOURCE,
                execution_class=ExecutionClass.COMPUTE,
                description=(
                    "Generate a parameterized planar Cartesian path "
                    "(shape=star5|circle, center_m, scale_m) — sampled closed "
                    "loop with safe-workspace validation. Returns plan_id handle."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "shape": {"type": "string", "enum": ["star5", "circle"]},
                        "center_m": {"type": "array", "items": {"type": "number"}},
                        "scale_m": {"type": "number"},
                        "plane": {"type": "string", "enum": ["xy"]},
                        "max_segment_m": {"type": "number"},
                    },
                    "required": ["shape"],
                    "additionalProperties": False,
                },
                supported_modes=["SIMULATION"],
                evidence_class=ToolEvidenceClass.SIMULATED,
                reliability=0.99,
                typical_latency_ms=50,
            ),
            ToolDescriptorV2(
                tool_id=TRAJ_SIMULATE_TOOL,
                source=NATIVE_SOURCE,
                execution_class=ExecutionClass.COMPUTE,
                description=(
                    "Run a real MuJoCo dynamics rollout of a planned Cartesian "
                    "trajectory with the SIMULATED UR5e (DLS-IK → joint "
                    "trajectory → physics rollout → actual eef trace + tracking "
                    "metrics). Evidence class: SIM_DYN_ROLLOUT (simulated)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"plan_id": {"type": "string"}},
                    "required": ["plan_id"],
                    "additionalProperties": False,
                },
                supported_modes=["SIMULATION"],
                evidence_class=ToolEvidenceClass.SIMULATED,
                verifier="physics-rollout+tracking-metrics",
                reliability=0.95,
                typical_latency_ms=15000,
            ),
            ToolDescriptorV2(
                tool_id=TRAJ_RENDER_TOOL,
                source=NATIVE_SOURCE,
                execution_class=ExecutionClass.COMPUTE,
                description=(
                    "Render the ACTUAL eef trace of a dynamics rollout into a "
                    "playable GIF artifact (>=30 frames)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "trace_id": {"type": "string"},
                        "format": {"type": "string", "enum": ["gif"]},
                    },
                    "required": ["trace_id"],
                    "additionalProperties": False,
                },
                supported_modes=["SIMULATION"],
                evidence_class=ToolEvidenceClass.SIMULATED,
                reliability=0.99,
                typical_latency_ms=3000,
            ),
            ToolDescriptorV2(
                tool_id=TRAJ_VERIFY_TOOL,
                source=NATIVE_SOURCE,
                execution_class=ExecutionClass.COMPUTE,
                description=(
                    "Verify trajectory tracking against a threshold "
                    "(max_tracking_error_m) — honest PASS/FAIL with metrics."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "trace_id": {"type": "string"},
                        "max_tracking_error_m": {"type": "number"},
                    },
                    "required": ["trace_id", "max_tracking_error_m"],
                    "additionalProperties": False,
                },
                supported_modes=["SIMULATION"],
                evidence_class=ToolEvidenceClass.SIMULATED,
                reliability=0.99,
                typical_latency_ms=100,
            ),
        ]
    for descriptor in descriptors:

        async def _exec(arguments: dict[str, Any], _name: str = descriptor.tool_id) -> str:
            return await registry.execute(_name, arguments)

        catalog.register(descriptor, _exec)
