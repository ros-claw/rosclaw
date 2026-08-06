"""Transcript projection（二次复核 R4/P1-1）：journal → transcript 块。

TUI 不再从底层事件流"猜"聊天记录——服务端把 append-only journal
投影为稳定的 transcript 块（user/assistant/tool/card/decision/receipt/
error），每块带稳定 ID 与 sequence，支持 ``before_seq`` 分页。
journal 仍是权威；projection 是纯函数、可重算、幂等。
"""

from __future__ import annotations

from typing import Any

from rosclaw.contracts.agent.agent_event import AgentEventType, AgentEventV2, Visibility

# 每个 transcript 页的最大 assistant 聚合跨度（防御异常事件流）。
_MAX_DELTA_RUN = 100_000


def project_transcript(events: list[AgentEventV2]) -> list[dict[str, Any]]:
    """把有序 journal 事件投影为 transcript 块。

    块 schema：
      {block_id, kind, sequence, text?, card?, decision?, receipt?, error?}
    block_id 稳定：user=u_<seq>；assistant=a_<turn_id|first_seq>；
    tool=t_<seq>；card/decision/receipt/error 同理按首事件 seq。
    """
    blocks: list[dict[str, Any]] = []
    assistant: dict[str, Any] | None = None
    delta_run = 0

    def close_assistant() -> None:
        nonlocal assistant, delta_run
        if assistant is not None:
            if assistant["text"].strip():
                blocks.append(assistant)
            assistant = None
            delta_run = 0

    for event in events:
        if event.visibility is Visibility.DEBUG:
            continue
        etype = event.type
        payload = event.payload or {}
        seq = event.sequence

        if etype is AgentEventType.TURN_ACCEPTED:
            close_assistant()
            text = str(payload.get("text", ""))
            if text:
                blocks.append(
                    {"block_id": f"u_{seq}", "kind": "user", "sequence": seq, "text": text}
                )
            continue
        if etype is AgentEventType.MODEL_TEXT_DELTA:
            if assistant is None:
                assistant = {
                    "block_id": f"a_{event.turn_id or seq}",
                    "kind": "assistant",
                    "sequence": seq,
                    "text": "",
                }
            if delta_run < _MAX_DELTA_RUN:
                assistant["text"] += str(payload.get("text", ""))
                assistant["sequence"] = seq
                delta_run += 1
            continue
        if etype in (
            AgentEventType.MESSAGE_ENDED,
            AgentEventType.TURN_ENDED,
            AgentEventType.AGENT_SETTLED,
            AgentEventType.MODEL_REQUEST_ENDED,
        ):
            close_assistant()
            continue
        if etype is AgentEventType.APPROVAL_REQUESTED:
            close_assistant()
            blocks.append(
                {
                    "block_id": f"c_{seq}",
                    "kind": "card",
                    "sequence": seq,
                    "card": payload,
                }
            )
            continue
        if etype is AgentEventType.APPROVAL_DECIDED:
            close_assistant()
            blocks.append(
                {
                    "block_id": f"d_{seq}",
                    "kind": "decision",
                    "sequence": seq,
                    "decision": payload,
                }
            )
            continue
        if etype in (AgentEventType.TOOL_PROPOSED, AgentEventType.MODEL_TOOL_CALL_PROPOSED):
            close_assistant()
            blocks.append(
                {
                    "block_id": f"t_{seq}",
                    "kind": "tool_call",
                    "sequence": seq,
                    "card": payload,
                }
            )
            continue
        if etype is AgentEventType.TOOL_COMPLETED:
            close_assistant()
            blocks.append(
                {
                    "block_id": f"tr_{seq}",
                    "kind": "tool_result",
                    "sequence": seq,
                    "card": payload,
                }
            )
            continue
        if etype in (AgentEventType.ACTION_RECEIPT, AgentEventType.RECEIPT_RECEIVED):
            close_assistant()
            blocks.append(
                {
                    "block_id": f"r_{seq}",
                    "kind": "receipt",
                    "sequence": seq,
                    "receipt": payload,
                }
            )
            continue
        if etype in (AgentEventType.ERROR, AgentEventType.AGENT_FAILED):
            close_assistant()
            blocks.append(
                {
                    "block_id": f"e_{seq}",
                    "kind": "error",
                    "sequence": seq,
                    "error": str(payload.get("error", "")),
                }
            )
            continue
        # 其他事件（阶段/状态/用量）不进 transcript——它们属于状态栏。
    close_assistant()
    return blocks
