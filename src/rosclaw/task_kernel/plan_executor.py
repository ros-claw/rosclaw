"""PlanExecutor（P1-C2，0824 总纲 §7.2）——typed DAG 确定性执行器。

执行语义：
- 节点按契约顺序执行（DAG 已由 PlanGraphV1 校验——inputs 必来自
  前序 outputs）；
- 每节点：plan.node_started → handler(inputs) → outputs 合并进
  ref 池 → plan.node_completed；失败 → plan.node_failed + 下游
  SKIPPED（不假装继续）；
- handler 契约：inputs: dict[str, Any] → dict[str, Any]（命名
  outputs——与节点声明 outputs 一致）；
- Fast Path = 单节点图——同一执行器、同一结果形状；
- MTC/BT adapter seam = handler 注册（同一 op 语义下接外部
  executor——契约不变）。
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from rosclaw.contracts.agent.plan_graph import PlanGraphV1

PlanNodeHandler = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class PlanExecutionResult:
    """Fast Path 与 DAG 共用结果形状。"""

    ok: bool
    refs: dict[str, Any] = field(default_factory=dict)
    failed_node: str = ""
    failure: str = ""


class PlanExecutor:
    """plan.node_* 事件落 task_events（与 Operation 同一事件流）。"""

    def __init__(self, kernel, conn: sqlite3.Connection) -> None:
        self._kernel = kernel
        self._conn = conn

    def run(
        self,
        graph: PlanGraphV1,
        handlers: dict[str, PlanNodeHandler],
    ) -> PlanExecutionResult:
        refs: dict[str, Any] = {}
        for node in graph.nodes:
            handler = handlers.get(node.op)
            if handler is None:
                return self._fail(
                    graph,
                    node.id,
                    f"NO_HANDLER: op {node.op!r} 无注册 handler",
                    refs,
                )
            self._emit(graph.task_id, "plan.node_started", {"node_id": node.id, "op": node.op})
            inputs = {name: refs[name] for name in node.inputs}
            try:
                outputs = handler(inputs) or {}
            except Exception as exc:  # noqa: BLE001 - 失败是数据
                return self._fail(graph, node.id, str(exc)[:300], refs)
            missing = [name for name in node.outputs if name not in outputs]
            if missing:
                return self._fail(
                    graph,
                    node.id,
                    f"OUTPUT_CONTRACT_VIOLATION: handler 未产出 {missing}",
                    refs,
                )
            refs.update(outputs)
            self._emit(
                graph.task_id,
                "plan.node_completed",
                {"node_id": node.id, "outputs": list(outputs.keys())},
            )
        return PlanExecutionResult(ok=True, refs=refs)

    def _fail(
        self,
        graph: PlanGraphV1,
        node_id: str,
        failure: str,
        refs: dict[str, Any],
    ) -> PlanExecutionResult:
        self._emit(graph.task_id, "plan.node_failed", {"node_id": node_id, "failure": failure})
        return PlanExecutionResult(
            ok=False,
            refs=dict(refs),
            failed_node=node_id,
            failure=failure,
        )

    def _emit(self, task_id: str, event_type: str, payload: dict) -> None:
        self._conn.execute(
            "INSERT INTO task_events (task_id, event_type, payload_json, "
            "created_at) VALUES (?, ?, ?, ?)",
            (
                task_id,
                event_type,
                json.dumps(payload, ensure_ascii=False),
                datetime.now(UTC).isoformat(),
            ),
        )


__all__ = ["PlanExecutionResult", "PlanExecutor", "PlanNodeHandler"]
