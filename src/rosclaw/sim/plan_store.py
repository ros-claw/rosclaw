"""持久 PlanStore（总纲 WP-P0-7）：plan 记录落盘。

八审的内存 PlanStore 在 executor 进程重启后丢 plan——crash 恢复
只能重规划或猜。本实现每个 plan 一个 JSON 文件（原子写），状态
（PLANNED/CONSUMED）随文件持久——重启后不重复执行、已消费不复活。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


class PersistentPlanStore:
    """文件后端 PlanStore（与内存版同一接口语义）。"""

    def __init__(self, plans_dir: Path, *, ttl_s: float = 1800.0, capacity: int = 32) -> None:
        self._dir = plans_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._ttl_s = ttl_s
        self._capacity = capacity

    @staticmethod
    def _now() -> float:
        return time.time()

    def _path(self, plan_id: str) -> Path:
        return self._dir / f"{plan_id}.json"

    def _read(self, plan_id: str) -> dict | None:
        path = self._path(plan_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _write(self, record: dict) -> None:
        path = self._path(record["plan_id"])
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    def put(self, trajectory: dict, summary: str) -> dict:
        # 九审 §17.3：随机实例 ID + digest 内容寻址分离。
        import uuid as _uuid

        digest = str(trajectory["hash"])
        plan_id = f"plan_{_uuid.uuid4().hex[:16]}"
        existing = self._read(plan_id)
        if existing is not None:
            return existing
        # 容量：驱逐最旧。
        records = sorted(
            (p for p in self._dir.glob("plan_*.json")),
            key=lambda p: p.stat().st_mtime,
        )
        while len(records) >= self._capacity:
            oldest = records.pop(0)
            oldest.unlink(missing_ok=True)
        record = {
            "plan_id": plan_id,
            "digest": digest,
            "trajectory": trajectory,
            "summary": summary,
            "created_at": self._now(),
            "status": "PLANNED",
        }
        self._write(record)
        return record

    def get_for_execute(self, plan_id: str) -> dict:
        record = self._read(plan_id)
        if record is None:
            raise ValueError(f"unknown plan_id {plan_id!r} (fail closed)")
        if record["status"] != "PLANNED":
            raise ValueError(
                f"plan {plan_id} already consumed — single-use (fail closed)"
            )
        if self._now() - float(record["created_at"]) > self._ttl_s:
            raise ValueError(f"plan {plan_id} expired (fail closed)")
        return record

    def consume(self, plan_id: str) -> None:
        record = self._read(plan_id)
        if record is not None:
            record["status"] = "CONSUMED"
            self._write(record)

    def by_digest(self, digest: str) -> dict | None:
        for path in self._dir.glob("plan_*.json"):
            record = self._read(path.stem)
            if record and record["digest"] == digest:
                return record
        return None

    def clear(self) -> None:
        for path in self._dir.glob("plan_*.json"):
            path.unlink(missing_ok=True)
