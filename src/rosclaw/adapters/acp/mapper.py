"""AgentEventV2 → ACP session update 映射（纯函数，可单测）。

Channel 设计 §22/§26 的冻结映射表实现。只做表示层映射；不携带任何
authority——approval/action 只呈现只读卡片，绝不转成 ACP permission 请求。
"""

from __future__ import annotations

from typing import Any

from rosclaw.contracts.agent.agent_event import Visibility


def _tool_call_id(event, payload: dict) -> str:
    """稳定 tool_call_id（设计 §23）：连续同名调用不得被 UI 合并成一个 call。"""
    return str(payload.get("tool_call_id") or payload.get("call_id") or event.event_id)


def _message_id(event) -> str:
    """稳定 message id（ACP message-id RFD）：同一 turn 的 delta 同属一条消息。"""
    return str(event.turn_id or event.event_id)


def event_to_session_update(event, *, include_debug: bool = False) -> Any | None:
    """Map one AgentEventV2 to an ACP session update (or None to skip).

    Visibility 门（设计 §48）：默认只有 USER 事件离开 ACP；DEBUG/AUDIT
    不外发（``include_debug=True`` 仅供本地调试）。
    """
    from acp import schema

    visibility = getattr(event, "visibility", None)
    if not include_debug and visibility is not None and visibility is not Visibility.USER:
        return None

    etype = event.type.value if hasattr(event.type, "value") else str(event.type)
    payload = event.payload or {}

    if etype == "model.text.delta":
        return schema.AgentMessageChunk(
            session_update="agent_message_chunk",
            content=schema.TextContentBlock(type="text", text=str(payload.get("text", ""))),
            message_id=_message_id(event),
        )

    # -- 思考摘要（§24/§26）：只转发安全 summary，绝不转发 raw CoT -------------
    if etype == "reasoning.summary.delta":
        text = str(payload.get("text", ""))
        if not text:
            return None
        return schema.AgentThoughtChunk(
            session_update="agent_thought_chunk",
            content=schema.TextContentBlock(type="text", text=text),
            message_id=f"{_message_id(event)}:reasoning",
        )
    if etype == "plan.updated":
        entries = payload.get("entries") or []
        if not entries:
            return None
        return schema.AgentPlanUpdate(
            session_update="plan",
            entries=[
                schema.PlanEntry(
                    content=str(item.get("content", ""))[:500],
                    priority=item.get("priority", "medium")
                    if item.get("priority") in ("high", "medium", "low")
                    else "medium",
                    status=item.get("status", "pending")
                    if item.get("status") in ("pending", "in_progress", "completed")
                    else "pending",
                )
                for item in entries[:50]
                if isinstance(item, dict) and item.get("content")
            ],
        )
    if etype in ("reasoning.started", "reasoning.summary.ended"):
        return None  # 生命周期标记，无独立 ACP 呈现

    # -- Tool lifecycle -------------------------------------------------------
    if etype in ("tool.started", "tool.proposed"):
        return schema.ToolCallStart(
            session_update="tool_call",
            tool_call_id=_tool_call_id(event, payload),
            title=f"⚙ {payload.get('name', 'tool')}",
            kind="other",
            status="in_progress",
        )
    if etype == "tool.progress":
        note = payload.get("note") or payload.get("progress") or ""
        return schema.ToolCallUpdate(
            session_update="tool_call_update",
            tool_call_id=_tool_call_id(event, payload),
            status="in_progress",
            title=f"⚙ {payload.get('name', 'tool')}: {note}" if note else None,
        )
    if etype == "tool.completed":
        ok = payload.get("ok", True)
        return schema.ToolCallUpdate(
            session_update="tool_call_update",
            tool_call_id=_tool_call_id(event, payload),
            status="completed" if ok else "failed",
        )

    # -- Worker lifecycle（设计 §29：Worker progress 是第一等 UI）--------------
    if etype.startswith("worker."):
        status = etype.split(".", 1)[1]
        # work_order_id 必须作为稳定 ID（§23）。
        call_id = str(payload.get("work_order_id") or event.event_id)
        worker = payload.get("worker_id", "worker")
        title = f"⛏ {worker}: {status}"
        if status in ("offered", "claimed", "started"):
            return schema.ToolCallStart(
                session_update="tool_call",
                tool_call_id=call_id,
                title=title,
                kind="other",
                status="in_progress",
            )
        terminal_ok = status in ("accepted", "completed")
        terminal_bad = status in ("failed", "expired")
        return schema.ToolCallUpdate(
            session_update="tool_call_update",
            tool_call_id=call_id,
            title=title,
            status=("completed" if terminal_ok else "failed" if terminal_bad else "in_progress"),
        )

    # -- Approval / Action：只读呈现，绝不构成授权（§9.1 边界）------------------
    if etype == "approval.requested":
        text = (
            f"🔐 授权请求 {payload.get('request_id', '')}\n"
            f"{payload.get('title', '')}（风险 {payload.get('risk_tier', 'LOW')}）\n"
            "请经 operator 通道（rosclaw chat / operator.sock）决定；"
            "ACP 客户端的决定不构成物理授权。"
        )
        return schema.AgentMessageChunk(
            session_update="agent_message_chunk",
            content=schema.TextContentBlock(type="text", text=text),
            message_id=event.event_id,
        )
    if etype == "action.proposed":
        text = (
            f"📝 动作提案 {payload.get('proposal_id', '')}\n"
            f"{payload.get('summary', payload.get('title', ''))}\n"
            "仅展示；REAL 执行必须经 trusted Operator 授权链。"
        )
        return schema.AgentMessageChunk(
            session_update="agent_message_chunk",
            content=schema.TextContentBlock(type="text", text=text),
            message_id=event.event_id,
        )
    if etype == "action.progress":
        text = (
            f"🔄 动作进度 {payload.get('proposal_id', payload.get('action_id', ''))}: "
            f"{payload.get('status', payload.get('note', ''))}"
        )
        return schema.AgentMessageChunk(
            session_update="agent_message_chunk",
            content=schema.TextContentBlock(type="text", text=text),
            message_id=event.event_id,
        )
    if etype in ("action.receipt", "receipt.received"):
        lines = "\n".join(f"{k}: {v}" for k, v in list(payload.items())[:6])
        return schema.AgentMessageChunk(
            session_update="agent_message_chunk",
            content=schema.TextContentBlock(type="text", text=f"🧾 执行回执\n{lines}"),
            message_id=event.event_id,
        )

    # -- Usage / 系统 -----------------------------------------------------------
    if etype == "model.request.ended":
        used = int(payload.get("prompt_tokens", 0)) + int(payload.get("completion_tokens", 0))
        size = payload.get("context_size") or payload.get("context_window")
        if not size:
            # 没有上下文窗口大小时不伪造 usage 比例（§22 映射的前提是数据真实）。
            return None
        return schema.UsageUpdate(
            session_update="usage_update",
            used=used,
            size=int(size),
        )
    if etype == "error":
        text = f"⚠ {payload.get('message', payload.get('error', 'unknown error'))}"
        return schema.AgentMessageChunk(
            session_update="agent_message_chunk",
            content=schema.TextContentBlock(type="text", text=text),
            message_id=event.event_id,
        )
    return None
