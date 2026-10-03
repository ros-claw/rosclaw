"""持久 PlanStore（总纲 WP-P0-7）：plan 记录落盘。

八审的内存 PlanStore 在 executor 进程重启后丢 plan——crash 恢复
只能重规划或猜。本实现每个 plan 一个 JSON 文件（原子写），状态
（PLANNED/CONSUMED）随文件持久——重启后不重复执行、已消费不复活。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from rosclaw.storage.durable import DurableNamespace


class PersistentPlanStore:
    """文件后端 PlanStore（与内存版同一接口语义）。"""

    def __init__(self, plans_dir: Path, *, ttl_s: float = 1800.0, capacity: int = 32) -> None:
        self._dir = plans_dir
        self._durability = DurableNamespace(self._dir, kind="plan")
        self._durability.ensure_directory(self._dir)
        self._ttl_s = ttl_s
        self._capacity = capacity

    @staticmethod
    def _now() -> float:
        return time.time()

    def _path(self, plan_id: str) -> Path:
        path = self._dir / f"{plan_id}.json"
        self._durability.assert_owned(path)
        return path

    def _read(self, plan_id: str) -> dict | None:
        path = self._path(plan_id)
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("PLAN_STORE_CORRUPT: existing record unreadable") from error
        if not isinstance(record, dict):
            raise ValueError("PLAN_STORE_CORRUPT: existing record is not an object")
        return record

    def _write(self, record: dict) -> None:
        path = self._path(record["plan_id"])
        if path.exists():
            self._read(record["plan_id"])  # never replace a corrupt existing record
        self._durability.atomic_replace(
            path, json.dumps(record, ensure_ascii=False).encode("utf-8")
        )

    def put(self, trajectory: dict, summary: str) -> dict:
        with self._durability.transaction(self._dir):
            return self._put_locked(trajectory, summary)

    def _put_locked(self, trajectory: dict, summary: str) -> dict:
        # 九审 §17.3：随机实例 ID + digest 内容寻址分离。
        import uuid as _uuid

        digest = str(trajectory["hash"])
        plan_id = f"plan_{_uuid.uuid4().hex[:16]}"
        existing = self._read(plan_id)
        if existing is not None:
            self._durability.sync_file(self._path(plan_id))
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

    def _read_envelope(self, plan_id: str) -> dict | None:
        """WP-2：envelope/native 原始记录互通——native 原始记录
        （trajectory 本体）包成 envelope 视图。"""
        record = self._read(plan_id)
        if record is None:
            return None
        if "trajectory" not in record:
            if "points" not in record or "hash" not in record:
                return None  # 不可解码——由调用方给 REF_FORMAT_UNKNOWN
            record = {
                "plan_id": plan_id,
                "digest": record["hash"],
                "trajectory": record,
                "summary": record.get("summary", ""),
                "created_at": self._now(),
                "status": record.get("status", "PLANNED"),
            }
        return record

    def get_for_execute(self, plan_id: str) -> dict:
        record = self._read_envelope(plan_id)
        if record is None:
            path = self._path(plan_id)
            if path.exists():
                raise ValueError(
                    f"REF_FORMAT_UNKNOWN: plan {plan_id!r} 记录格式不可解码 (fail closed)"
                )
            raise ValueError(f"REF_NOT_FOUND: plan_id {plan_id!r} 不在共享 PlanStore (fail closed)")
        if record["status"] != "PLANNED":
            raise ValueError(f"plan {plan_id} already consumed — single-use (fail closed)")
        if self._now() - float(record["created_at"]) > self._ttl_s:
            raise ValueError(f"plan {plan_id} expired (fail closed)")
        return record

    def consume(self, plan_id: str) -> None:
        # get_for_execute is a read-only snapshot, not an execution claim.
        # Only this atomic status-check + durable consume may authorize once.
        with self._durability.transaction(self._dir):
            record = self.get_for_execute(plan_id)
            record["status"] = "CONSUMED"
            self._write(record)

    def by_digest(self, digest: str) -> dict | None:
        for path in self._dir.glob("plan_*.json"):
            record = self._read(path.stem)
            if record and record["digest"] == digest:
                return record
        return None

    def clear(self) -> None:
        with self._durability.transaction(self._dir):
            for path in self._dir.glob("plan_*.json"):
                path.unlink(missing_ok=True)
            self._durability.sync_directory(self._dir)
