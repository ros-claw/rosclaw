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
            output_schema={
                "type": "object",
                "properties": {
                    "evidence_class": {"type": "string", "const": "configured"},
                    "body_id": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["evidence_class", "body_id", "summary"],
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
                output_schema={
                    "type": "object",
                    "properties": {
                        "evidence_class": {"type": "string", "const": "simulated"},
                        "mode": {"type": "string"},
                        "body_id": {"type": "string"},
                        "joints_rad": {"type": "array", "items": {"type": "number"}},
                        "health": {"type": "string"},
                        "fresh": {"type": "boolean"},
                    },
                    "required": ["evidence_class", "mode", "body_id",
                                 "joints_rad", "health", "fresh"],
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

                output_schema={
                    "type": "object",
                    "properties": {
                        "evidence_class": {"type": "string", "const": "simulated"},
                        "run_id": {"type": "string"},
                        "target_m": {"type": "array", "items": {"type": "number"}},
                        "policy": {"type": ["string", "null"]},
                        "physics_steps": {"type": ["integer", "null"]},
                        "collision_check": {"type": "string"},
                        "final_distance_m": {"type": ["number", "null"]},
                        "final_state": {"type": "string"},
                        "task_success": {"type": "boolean"},
                        "evidence_verified": {"type": "boolean"},
                        "verification": {"type": "object"},
                        "receipt_path": {"type": "string"},
                    },
                    "required": ["evidence_class", "run_id", "task_success",
                                 "evidence_verified", "receipt_path"],
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
            SCENE_RENDER_TOOL,
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

                output_schema={
                    "type": "object",
                    "properties": {
                        "ok": {"type": "boolean", "const": True},
                        "plan_id": {"type": "string"},
                        "hash": {"type": "string"},
                        "point_count": {"type": "integer"},
                        "summary": {"type": "string"},
                        "evidence_class": {"type": "string", "const": "simulated"},
                    },
                    "required": ["ok", "plan_id", "hash", "point_count",
                                 "summary", "evidence_class"],
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

                output_schema={
                    "type": "object",
                    "properties": {
                        "ok": {"type": "boolean", "const": True},
                        "trace_id": {"type": "string"},
                        "evidence_level": {"type": "string",
                                           "const": "SIM_DYN_ROLLOUT"},
                        "physics_executed": {"type": "boolean"},
                        "is_safe": {"type": "boolean"},
                        "violations": {"type": "array",
                                       "items": {"type": "string"}},
                        "point_count": {"type": "integer"},
                        "tracking": {"type": "object"},
                        "resource": {"type": "object"},
                        "artifacts": {"type": "object"},
                        "evidence_class": {"type": "string", "const": "simulated"},
                    },
                    "required": ["ok", "trace_id", "evidence_level",
                                 "physics_executed", "is_safe", "violations",
                                 "tracking", "artifacts", "evidence_class"],
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

                output_schema={
                    "type": "object",
                    "properties": {
                        "ok": {"type": "boolean", "const": True},
                        "artifact": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "frames": {"type": "integer"},
                                "format": {"type": "string", "const": "gif"},
                                "bytes": {"type": "integer"},
                                "evidence_level": {"type": "string",
                                                   "const": "SIM_DYN_ROLLOUT"},
                            },
                            "required": ["path", "frames", "format", "bytes",
                                         "evidence_level"],
                            "additionalProperties": False,
                        },
                        "evidence_class": {"type": "string", "const": "simulated"},
                    },
                    "required": ["ok", "artifact", "evidence_class"],
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

                output_schema={
                    "type": "object",
                    "properties": {
                        "ok": {"type": "boolean", "const": True},
                        "verdict": {"type": "string", "enum": ["PASS", "FAIL"]},
                        "threshold_m": {"type": "number"},
                        "metrics": {"type": "object"},
                        "evidence_level": {"type": "string",
                                           "const": "SIM_DYN_ROLLOUT"},
                        "evidence_class": {"type": "string", "const": "simulated"},
                    },
                    "required": ["ok", "verdict", "threshold_m", "metrics",
                                 "evidence_level", "evidence_class"],
                    "additionalProperties": False,
                },
                supported_modes=["SIMULATION"],
                evidence_class=ToolEvidenceClass.SIMULATED,
                reliability=0.99,
                typical_latency_ms=100,
            ),
            # WP-3：原生离线场景渲染（真实 MuJoCo 场景 replay——
            # 不是 2D 折线预览）。
            ToolDescriptorV2(
                tool_id=SCENE_RENDER_TOOL,
                source=NATIVE_SOURCE,
                execution_class=ExecutionClass.COMPUTE,
                description=(
                    "Render a dynamics rollout trace into a real MuJoCo "
                    "scene GIF (canonical MJCF + trajectory state replay + "
                    "camera preset + EGL/OSMesa/Xvfb auto-probe). Offline; "
                    "returns artifact + render receipt (build/input digests)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "trace_id": {"type": "string"},
                        "camera": {"type": "string",
                                   "enum": ["follow", "free", "top"]},
                    },
                    "required": ["trace_id"],
                    "additionalProperties": False,
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "ok": {"type": "boolean", "const": True},
                        "artifact": {"type": "object"},
                        "receipt": {"type": "object"},
                        "evidence_class": {"type": "string", "const": "simulated"},
                    },
                    "required": ["ok", "artifact", "receipt", "evidence_class"],
                    "additionalProperties": False,
                },
                supported_modes=["SIMULATION"],
                evidence_class=ToolEvidenceClass.SIMULATED,
                reliability=0.95,
                typical_latency_ms=30000,
            ),
        ]
    for descriptor in descriptors:

        async def _exec(arguments: dict[str, Any], _name: str = descriptor.tool_id) -> dict[str, Any]:
            return await registry.execute(_name, arguments)

        catalog.register(descriptor, _exec)
    # WP-2/WP-3：引用端口声明（canonical capability——snapshot 连通性
    # 的依据）。simulate 消费 plan 产 trace；scene render 消费 trace
    # 产 render；plan 产 plan。
    if simulation:
        from rosclaw.agentd.tools import TRAJ_PLAN_TOOL as _PLAN
        from rosclaw.agentd.tools import TRAJ_SIMULATE_TOOL as _SIM

        _ref_ports = {
            _PLAN: ([], [{"kind": "plan"}]),
            _SIM: ([{"kind": "plan", "from": _PLAN}], [{"kind": "trace"}]),
            SCENE_RENDER_TOOL: (
                [{"kind": "trace", "from": _SIM}], [{"kind": "render"}],
            ),
        }
        for _tid, (_accepts, _produces) in _ref_ports.items():
            _cap = catalog.capability(_tid)
            if _cap is not None:
                catalog._capabilities[_tid] = _cap.model_copy(update={
                    "accepts_refs": _accepts, "produces_refs": _produces,
                })
