"""会话发现（总纲 WP-P0-1）：Pi JSONL 的产品级列表/解析。

不做全量 transcript 扫描——只读 header/session_info/首条用户消息 +
行数 + mtime；损坏文件跳过不拖垮列表。WP-P0-2 的 SessionCatalog
落库后本模块退化为 backfill 来源。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _sessions_dir(home: Path) -> Path:
    return home / "agent" / "sessions"


def list_sessions(home: Path) -> list[dict[str, Any]]:
    """列出全部 Pi 会话（最近活动降序）。损坏 JSONL 跳过。"""
    sessions_dir = _sessions_dir(home)
    if not sessions_dir.is_dir():
        return []
    sessions: list[dict[str, Any]] = []
    for path in sessions_dir.glob("*.jsonl"):
        try:
            info = _parse_session_file(path)
        except Exception:  # noqa: BLE001 — 损坏文件不拖垮列表
            continue
        if info is not None:
            sessions.append(info)
    sessions.sort(key=lambda s: s.get("modified", ""), reverse=True)
    return sessions


def _parse_session_file(path: Path) -> dict[str, Any] | None:
    session_id = ""
    name = ""
    first_message = ""
    message_count = 0
    created = ""
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue  # 跳过坏行而不是整个文件
            etype = entry.get("type")
            if etype == "session":
                session_id = str(entry.get("id", ""))
                created = str(entry.get("timestamp", ""))
            elif etype == "session_info" and entry.get("name"):
                name = str(entry["name"])
            elif etype == "message":
                message_count += 1
                if not first_message:
                    message = entry.get("message") or {}
                    if message.get("role") == "user":
                        content = message.get("content", "")
                        if isinstance(content, list):
                            content = " ".join(
                                str(b.get("text", "")) for b in content if isinstance(b, dict)
                            )
                        first_message = str(content)[:120]
    if not session_id:
        return None
    stat = path.stat()
    return {
        "session_id": session_id,
        "path": str(path),
        "display_name": name,
        "first_message": first_message,
        "message_count": message_count,
        "created": created,
        "modified": str(stat.st_mtime),
    }
