"""AgentEventV2 → ACP session update 映射（纯函数，可单测）。"""

from __future__ import annotations

from typing import Any


def event_to_session_update(event) -> Any | None:
    """Map one AgentEventV2 to an ACP session update (or None to skip).

    只做表示层映射；不携带任何 authority。
    """
    from acp import schema

    etype = event.type.value if hasattr(event.type, "value") else str(event.type)
    payload = event.payload or {}
    if etype == "model.text.delta":
        return schema.AgentMessageChunk(
            session_update="agent_message_chunk",
            content=schema.TextContentBlock(type="text", text=str(payload.get("text", ""))),
        )
    if etype in ("tool.started", "tool.proposed"):
        return schema.ToolCallStart(
            session_update="tool_call",
            tool_call_id=str(payload.get("name", "tool")),
            title=f"⚙ {payload.get('name', 'tool')}",
            kind="other",
            status="in_progress",
        )
    if etype == "tool.completed":
        ok = payload.get("ok", True)
        return schema.ToolCallUpdate(
            session_update="tool_call_update",
            tool_call_id=str(payload.get("name", "tool")),
            status="completed" if ok else "failed",
        )
    if etype.startswith("worker."):
        status = etype.split(".", 1)[1]
        terminal = status in ("accepted", "failed", "expired")
        return schema.ToolCallStart(
            session_update="tool_call",
            tool_call_id=str(payload.get("work_order_id", "worker")),
            title=f"⛏ {payload.get('worker_id', 'worker')}: {status}",
            kind="other",
            status=(
                "completed"
                if status == "accepted"
                else "failed" if terminal else "in_progress"
            ),
        )
    if etype == "approval.requested":
        # 不伪造 ACP permission=物理授权；只呈现卡片与 id（§9.1 边界）。
        text = (
            f"🔐 授权请求 {payload.get('request_id', '')}\n"
            f"{payload.get('title', '')}（风险 {payload.get('risk_tier', 'LOW')}）\n"
            "请经 operator 通道（rosclaw chat / operator.sock）决定；"
            "ACP 客户端的决定不构成物理授权。"
        )
        return schema.AgentMessageChunk(
            session_update="agent_message_chunk",
            content=schema.TextContentBlock(type="text", text=text),
        )
    if etype in ("action.receipt", "receipt.received"):
        lines = "\n".join(f"{k}: {v}" for k, v in list(payload.items())[:6])
        return schema.AgentMessageChunk(
            session_update="agent_message_chunk",
            content=schema.TextContentBlock(type="text", text=f"🧾 执行回执\n{lines}"),
        )
    if etype == "model.request.ended":
        return schema.AgentMessageChunk(
            session_update="agent_message_chunk",
            content=schema.TextContentBlock(
                type="text",
                text=(
                    f"（tokens: {payload.get('prompt_tokens', 0)}"
                    f"→{payload.get('completion_tokens', 0)}）"
                ),
            ),
        )
    return None
