"""Conversation compaction (openharness 两级压缩的轻量版）.

- **microcompact**：超阈值时先把旧 tool result 折叠为引用占位
  （保留 system、首条 user 锚点与最近 N 条——zeroclaw trim_history 模式）；
- **reactive compact**：模型返回 context-overflow 类错误时压缩并重试一次。
"""

from __future__ import annotations

import json
from typing import Any

from rosclaw.agentd.context.tokens import estimate_tokens

DEFAULT_MAX_MESSAGES = 50
TOOL_RESULT_PLACEHOLDER = "[tool result compacted: see artifact refs in trusted context]"

_OVERFLOW_MARKERS = (
    "context length",
    "context_length",
    "prompt too long",
    "prompt is too long",
    "too long",
    "context window",
    "maximum context",
    "too many tokens",
    "string too long",
)


def is_context_overflow(kind: str, detail: str) -> bool:
    if kind == "context_overflow":
        return True
    if kind not in ("http_error", "invalid_response", "stream_failed", "invoke_failed"):
        return False
    lowered = detail.lower()
    return any(marker in lowered for marker in _OVERFLOW_MARKERS)


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            total += estimate_tokens(json.dumps(content, ensure_ascii=False))
        if message.get("tool_calls"):
            total += estimate_tokens(json.dumps(message["tool_calls"], ensure_ascii=False))
    return total


def microcompact(
    messages: list[dict[str, Any]],
    *,
    keep_recent: int = 8,
    max_messages: int = DEFAULT_MAX_MESSAGES,
) -> tuple[list[dict[str, Any]], int]:
    """折叠旧 tool result + 中段消息。返回 (新消息列表, 折叠数)。"""
    if len(messages) <= max_messages:
        compacted_tools, count = _compact_tool_results(messages, keep_recent)
        return compacted_tools, count
    # 保留首条 user 锚点 + 最近 max_messages-1 条。
    anchor: list[dict[str, Any]] = []
    rest = messages
    if messages and messages[0].get("role") == "user":
        anchor = [messages[0]]
        rest = messages[1:]
    trimmed = anchor + rest[-(max_messages - len(anchor)) :]
    compacted_tools, count = _compact_tool_results(trimmed, keep_recent)
    return compacted_tools, count + max(0, len(messages) - len(trimmed))


def _compact_tool_results(
    messages: list[dict[str, Any]], keep_recent: int
) -> tuple[list[dict[str, Any]], int]:
    boundary = max(0, len(messages) - keep_recent)
    out: list[dict[str, Any]] = []
    count = 0
    for index, message in enumerate(messages):
        if (
            index < boundary
            and message.get("role") == "tool"
            and isinstance(message.get("content"), str)
            and len(message["content"]) > 200
        ):
            message = {**message, "content": TOOL_RESULT_PLACEHOLDER}
            count += 1
        out.append(message)
    return out, count
