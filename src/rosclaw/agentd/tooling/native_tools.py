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
    for descriptor in descriptors:

        async def _exec(arguments: dict[str, Any], _name: str = descriptor.tool_id) -> str:
            return await registry.execute(_name, arguments)

        catalog.register(descriptor, _exec)
