"""WorkerEventStore（十一审 PR-B，总纲 §P0-3）。

每个 attempt 的持久化事件账本（append-only，文件即权威）：

    ~/.rosclaw/work/<work_order_id>/
      order.json        # envelope（W1 起）
      state.json        # 当前状态（status/phase/last_seq/updated_at）
      events.jsonl      # 全部 WorkerEvent（含 liveness/stall_warning）
      stderr.log        # 子进程 stderr（secret 脱敏后落盘）
      artifacts/        # patch/bash-log/媒体（W3 起）

- 事件带 work_order_id/attempt_id/seq/kind/ts；
- stdout/stderr 先过 secret redaction 再持久化；
- 大文本不进事件（工件化 + preview）；
- agentd DB 只索引状态——重启/compact 后 tail 仍可读（文件权威）。
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SECRET_PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9]{12,}"), "sk-***REDACTED***"),
    (
        re.compile(r"(?i)(api[_-]?key|secret|password|token)(\s*[:=]\s*)['\"]?[\w\-]{8,}"),
        r"\1\2***REDACTED***",
    ),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[^-]*-----END [A-Z ]*PRIVATE KEY-----"), "***REDACTED-KEY***"),
]


def redact(text: str) -> str:
    for pattern, repl in _SECRET_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


class WorkerEventStore:
    """文件权威的事件账本（agentd 重启后可继续读/写）。"""

    def __init__(self, home: Path) -> None:
        self._root = Path(home) / "work"
        # 十二审 PR-12.2：单写者内存计数（O(1) append）——每个进程内
        # 唯一写者；首次写时从文件行数初始化（重启续写不重置）。
        self._seq_counters: dict[str, int] = {}

    def _next_seq(self, events_path: Path, work_order_id: str) -> int:
        current = self._seq_counters.get(work_order_id)
        if current is None:
            current = 0
            if events_path.exists():
                with events_path.open("rb") as fh:
                    current = sum(1 for _ in fh)
            self._seq_counters[work_order_id] = current
        current += 1
        self._seq_counters[work_order_id] = current
        return current

    def _dir(self, work_order_id: str) -> Path:
        # 防路径穿越：wo_ 前缀 + hex。
        if not re.fullmatch(r"wo_[A-Za-z0-9]{8,32}", work_order_id):
            raise ValueError(f"invalid work_order_id {work_order_id!r}")
        return self._root / work_order_id

    def dir_of(self, work_order_id: str) -> Path:
        return self._dir(work_order_id)

    # ------------------------------------------------------------------
    def append_event(
        self,
        work_order_id: str,
        attempt_id: str,
        kind: str,
        payload: dict[str, Any],
    ) -> int:
        """追加事件（seq 由文件行数推导——重启后续写不重置）。"""
        d = self._dir(work_order_id)
        d.mkdir(parents=True, exist_ok=True)
        events_path = d / "events.jsonl"
        seq = self._next_seq(events_path, work_order_id)
        record = {
            "v": 2,  # 十二审 PR-12.2：事件 schema 版本
            "work_order_id": work_order_id,
            "attempt_id": attempt_id,
            "seq": seq,
            "kind": kind,
            "ts": _utcnow(),
            **{k: v for k, v in payload.items() if k not in ("work_order_id", "attempt_id", "seq", "ts")},
        }
        # 大文本工件化：超过 2KiB 的字符串字段只留 preview。
        for key, value in list(record.items()):
            if isinstance(value, str) and len(value) > 2048:
                record[key] = value[:2048] + f"…[+{len(value) - 2048} chars truncated]"
            elif isinstance(value, str):
                record[key] = redact(value)
        with events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return int(record["seq"])

    def append_stderr(self, work_order_id: str, chunk: str) -> None:
        d = self._dir(work_order_id)
        d.mkdir(parents=True, exist_ok=True)
        with (d / "stderr.log").open("a", encoding="utf-8") as fh:
            fh.write(redact(chunk))

    def write_state(self, work_order_id: str, state: dict[str, Any]) -> None:
        d = self._dir(work_order_id)
        d.mkdir(parents=True, exist_ok=True)
        state = {**state, "updated_at": _utcnow()}
        tmp = d / "state.json.tmp"
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(d / "state.json")

    def read_state(self, work_order_id: str) -> dict[str, Any] | None:
        path = self._dir(work_order_id) / "state.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    # ------------------------------------------------------------------
    def tail(
        self, work_order_id: str, *, after_seq: int = 0, limit: int = 100
    ) -> list[dict[str, Any]]:
        """cursor 后**最早**的 N 条（分页读取用——与 tail_page 同语义）。"""
        return self.tail_page(work_order_id, after_seq=after_seq, limit=limit)["events"]

    def tail_page(
        self, work_order_id: str, *, after_seq: int = 0, limit: int = 100
    ) -> dict[str, Any]:
        """十二审 PR-12.2：正确分页——返回 cursor 后**最早**的 N 条 +
        next_cursor + has_more。两次轮询间新增超过 N 条时，旧实现
        （取最后 N 条）会永久跳过中间事件；本实现零丢失零重复。
        """
        path = self._dir(work_order_id) / "events.jsonl"
        if not path.exists():
            return {"events": [], "next_cursor": after_seq, "has_more": False}
        events: list[dict[str, Any]] = []
        last_seq = after_seq
        has_more = False
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seq = int(event.get("seq", 0))
                if seq <= after_seq:
                    continue
                if len(events) < limit:
                    events.append(event)
                    last_seq = seq
                else:
                    has_more = True
                    break
        return {"events": events, "next_cursor": last_seq, "has_more": has_more}

    def tail_stderr(self, work_order_id: str, *, max_bytes: int = 4096) -> str:
        path = self._dir(work_order_id) / "stderr.log"
        if not path.exists():
            return ""
        data = path.read_bytes()
        return data[-max_bytes:].decode("utf-8", errors="replace")
