"""Read/query utilities shared by the CLI and Dashboard trace APIs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from rosclaw.firstboot.workspace import resolve_home


class TraceStore:
    """Read completed spans from the local JSONL trace store."""

    def __init__(
        self,
        home: str | Path | None = None,
        path: str | Path | None = None,
    ) -> None:
        self.path = (
            Path(path).expanduser()
            if path is not None
            else resolve_home(str(home) if home else None) / "traces" / "live.jsonl"
        )

    def _paths(self) -> list[Path]:
        rotated = sorted(self.path.parent.glob(f"{self.path.name}.[0-9][0-9][0-9]"))
        return [*rotated, *([self.path] if self.path.exists() else [])]

    def read(
        self,
        *,
        trace_id: str | None = None,
        kinds: set[str] | None = None,
        statuses: set[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for path in self._paths():
            try:
                with path.open(encoding="utf-8") as stream:
                    for line in stream:
                        try:
                            record = json.loads(line)
                        except (json.JSONDecodeError, TypeError):
                            continue
                        if not isinstance(record, dict):
                            continue
                        if trace_id and record.get("trace_id") != trace_id:
                            continue
                        if kinds and str(record.get("span_kind", "")).upper() not in kinds:
                            continue
                        if statuses and str(record.get("status", "")).upper() not in statuses:
                            continue
                        records.append(record)
            except OSError:
                continue
        records.sort(key=lambda item: (item.get("started_at") or 0, item.get("span_id") or ""))
        return records[-limit:] if limit is not None and limit >= 0 else records

    def list_traces(self, limit: int = 50) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in self.read():
            trace_id = record.get("trace_id")
            if trace_id:
                grouped[str(trace_id)].append(record)
        summaries = []
        for trace_id, records in grouped.items():
            first = min((item.get("started_at") or 0 for item in records), default=0)
            last = max(
                (item.get("ended_at") or item.get("started_at") or 0 for item in records), default=0
            )
            statuses = {str(item.get("status", "")) for item in records}
            summaries.append(
                {
                    "trace_id": trace_id,
                    "started_at": first,
                    "ended_at": last,
                    "duration_ms": round(max(0.0, last - first) * 1000, 3),
                    "span_count": len(records),
                    "status": "ERROR"
                    if "ERROR" in statuses
                    else ("BLOCKED" if "BLOCKED" in statuses else "OK"),
                    "kinds": sorted({str(item.get("span_kind", "")) for item in records}),
                }
            )
        summaries.sort(key=lambda item: item["started_at"], reverse=True)
        return summaries[:limit]

    def get_trace(self, trace_id: str) -> dict[str, Any]:
        records = self.read(trace_id=trace_id)
        nodes = {str(item.get("span_id")): {**item, "children": []} for item in records}
        roots: list[dict[str, Any]] = []
        for item in records:
            node = nodes[str(item.get("span_id"))]
            parent = nodes.get(str(item.get("parent_span_id")))
            if parent is None:
                roots.append(node)
            else:
                parent["children"].append(node)
        return {
            "trace_id": trace_id,
            "span_count": len(records),
            "spans": records,
            "tree": roots,
        }

    def find_event(self, event_id: str) -> dict[str, Any] | None:
        return next((item for item in self.read() if item.get("event_id") == event_id), None)
