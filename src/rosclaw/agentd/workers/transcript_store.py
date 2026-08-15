"""WorkerTranscriptStore——完整公开 transcript 分页读取（十四审 PR-14.3，
总纲 §4.4）。

与主对话 compaction 完全独立的会话级证据：
- 公开会话 transcript（assistant 全文、工具调用+输出、文件改动、产物
  hash、usage、控制 ACK）按 channel 分频；
- tseq 单调游标，after_seq/before_seq 双向分页；断线用 cursor 补齐，
  不消耗模型 token；
- 十二审 legacy 格式（无 tseq/channel）按行号合成游标并映射频道；
- 不读隐藏推理/system prompt/凭据——transcript 由 worker 侧脱敏后写入。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: 合法频道（legacy 记录映射到 conversation/tools）。
CHANNELS = (
    "conversation",
    "tools",
    "files",
    "artifacts",
    "usage",
    "control",
)


class TranscriptStore:
    def __init__(self, home: Path) -> None:
        self._home = Path(home)

    def _path(self, work_order_id: str) -> Path:
        return self._home / "work" / work_order_id / "transcript.jsonl"

    def _load(self, work_order_id: str) -> list[dict[str, Any]]:
        path = self._path(work_order_id)
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        for idx, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            # legacy（十二审 {ts, role, text}）：行号合成游标 + 频道映射。
            if not isinstance(record.get("tseq"), int):
                record["tseq"] = idx
            if record.get("channel") not in CHANNELS:
                record["channel"] = (
                    "tools" if record.get("role") == "tool" else "conversation"
                )
            records.append(record)
        return records

    def read_page(
        self,
        work_order_id: str,
        *,
        after_seq: int | None = None,
        before_seq: int | None = None,
        limit: int = 50,
        channel: str | None = None,
    ) -> dict[str, Any]:
        """双向分页：after_seq 向后（默认）、before_seq 向前翻页。"""
        records = self._load(work_order_id)
        if channel:
            records = [r for r in records if r["channel"] == channel]
        total = len(records)
        limit = max(1, min(int(limit), 500))
        if before_seq is not None:
            window = [r for r in records if r["tseq"] < before_seq]
            page = window[-limit:]
            has_more = len(window) > len(page)
        else:
            after = after_seq or 0
            window = [r for r in records if r["tseq"] > after]
            page = window[:limit]
            has_more = len(window) > len(page)
        next_cursor = page[-1]["tseq"] if page else (after_seq or 0)
        return {
            "records": page,
            "has_more": has_more,
            "next_cursor": next_cursor,
            "total": total,
        }
