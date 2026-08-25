"""PlanGraphV1（P1-C2，0824 总纲 §7.2）——typed DAG 任务施工图。

能力链不再由模型逐工具即兴编排：节点 op/in/out 是契约，typed refs
（ResourceRef/PlanRef/TraceRef/RenderRef/VerificationRef）按名在节点
间流动。Fast Path（单 capability 直出 refs）与复杂 DAG 共用同一
结果类型与 TaskOutcomeV2——简单任务不被迫复杂化。

冻结不变量：
- op 必须是冻结操作集之一（MTC/BT adapter 经同一 op 语义接入——
  seam 是 handler 注册，不是新 op 方言）；
- node id 唯一；inputs 必须由前序节点的 outputs 提供（无环、无
  悬空引用）——契约层直接拒绝，不靠执行期发现。
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field, field_validator, model_validator

from rosclaw.contracts.common import ContractModel

#: 冻结节点操作集（0824 §7.2）。MTC/BT adapter seam = 为这些 op
#: 注册外部 executor handler——不是发明新 op。
PLAN_NODE_OPS = (
    "resource.resolve",
    "geometry.plan_path",
    "robot.execute_plan",
    "simulation.render",
    "task.verify",
)


class PlanNodeV1(ContractModel):
    """一个 DAG 节点：op + 命名输入/输出（typed refs 按名流动）。"""

    SCHEMA: ClassVar[str] = "rosclaw.plan_node.v1"

    schema_version: Literal["rosclaw.plan_node.v1"] = "rosclaw.plan_node.v1"
    id: str = Field(min_length=1)
    op: str = Field(min_length=1)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)

    @field_validator("op")
    @classmethod
    def _known_op(cls, value: str) -> str:
        if value not in PLAN_NODE_OPS:
            raise ValueError(f"unknown plan node op {value!r} (frozen: {PLAN_NODE_OPS})")
        return value


class PlanGraphV1(ContractModel):
    """typed PlanGraph（一个 task revision 的施工图）。"""

    SCHEMA: ClassVar[str] = "rosclaw.plan_graph.v1"
    HASH_PREFIX: ClassVar[str] = "plan_graph"

    schema_version: Literal["rosclaw.plan_graph.v1"] = "rosclaw.plan_graph.v1"
    graph_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    revision: int = Field(ge=1)
    nodes: list[PlanNodeV1] = Field(min_length=1)
    #: 内容寻址 digest（规范化 JSON sha256）。
    digest: str = Field(min_length=1)

    @model_validator(mode="after")
    def _dag_invariants(self) -> PlanGraphV1:
        seen: set[str] = set()
        for node in self.nodes:
            if node.id in seen:
                raise ValueError(f"duplicate node id {node.id!r}")
            seen.add(node.id)
        available: set[str] = set()
        for node in self.nodes:
            for ref in node.inputs:
                if ref not in available:
                    raise ValueError(
                        f"node {node.id!r} 悬空/前驱未定义输入 {ref!r}（dangling input 或存在环）"
                    )
            available.update(node.outputs)
        return self


__all__ = ["PLAN_NODE_OPS", "PlanGraphV1", "PlanNodeV1"]
