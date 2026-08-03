"""Export service (批次 E §8.11)：.rcmission bundle + 脱敏。

bundle 结构（zip）：
    manifest.json / conversation.jsonl / mission-events.jsonl /
    compactions.jsonl / public-receipts/ / checksums.txt

默认排除：API key/OAuth token、Provider 请求头、Permit、daemon 私有
ledger/challenge、grant 私签、原始 secret、外部 Worker 环境。
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rosclaw.agentd.service import AgentService

BUNDLE_MAGIC = "rcmission/1"
MAX_BUNDLE_BYTES = 64 * 1024 * 1024

_SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9_-]{8,}|Bearer\s+[A-Za-z0-9._-]{8,}"
    r"|api[_-]?key[\"']?\s*[:=]\s*[\"']?[^\s\"',}]+)",
    re.IGNORECASE,
)


def redact_text(text: str) -> str:
    return _SECRET_RE.sub("<redacted>", text)


class ExportService:
    def __init__(self, service: AgentService) -> None:
        self._service = service

    def export_bundle(self, mission_id: str, out_path: Path) -> dict:
        service = self._service
        mission = service.get_mission(mission_id)
        if mission is None:
            from rosclaw.contracts.common import ValidationError

            raise ValidationError(f"unknown mission {mission_id!r}")
        from rosclaw.agentd.context.compaction import CompactionStore

        meta = service.store.mission_meta(mission_id)
        manifest = {
            "magic": BUNDLE_MAGIC,
            "exported_at": datetime.now(UTC).isoformat(),
            "mission": {
                "mission_id": mission_id,
                "name": meta["display_name"],
                "goal": mission.goal.text,
                "mode": mission.mode.value,
                "state": mission.state.value,
                "created_at": mission.created_at,
                "body_id": mission.body_binding.body_id,
                "context_revision": mission.context_revision,
            },
            "redaction": "secrets/permits/private-signatures excluded by construction",
        }
        # canonical journal（含 entry_id/seq，fork/import 可引用）。
        conversation = service.store.conversation(mission_id)
        conv_lines = [
            json.dumps(redact_text(json.dumps(m, ensure_ascii=False)), ensure_ascii=False)
            for m in conversation
        ]
        events = service.events_replay(mission_id, after_sequence=0, limit=100_000)
        event_lines = [
            redact_text(json.dumps(e.model_dump(mode="json"), ensure_ascii=False))
            for e in events
        ]
        compactions = [
            e.model_dump(mode="json")
            for e in CompactionStore(service.store.connection).list(mission_id)
        ]
        # 只含 public 字段的 grants（私签永不导出）。
        public_grants = [
            {k: v for k, v in g.items() if k in (
                "grant_id", "principal", "mode", "tier", "risk_ceiling",
                "revoked", "consumed", "expires_at", "public_hash",
            )}
            for g in service.list_grants()
        ]

        files: dict[str, str] = {
            "manifest.json": json.dumps(manifest, ensure_ascii=False, indent=2),
            "conversation.jsonl": "\n".join(conv_lines) + "\n",
            "mission-events.jsonl": "\n".join(event_lines) + "\n",
            "compactions.jsonl": "\n".join(
                json.dumps(c, ensure_ascii=False) for c in compactions
            )
            + "\n",
            "public-receipts/grants.json": json.dumps(
                public_grants, ensure_ascii=False, indent=2
            ),
        }
        checksums = "\n".join(
            f"{hashlib.sha256(content.encode()).hexdigest()}  {name}"
            for name, content in sorted(files.items())
        )
        files["checksums.txt"] = checksums + "\n"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        total = sum(len(c.encode()) for c in files.values())
        if total > MAX_BUNDLE_BYTES:
            from rosclaw.contracts.common import ValidationError

            raise ValidationError(f"bundle too large ({total} bytes)")
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, content in files.items():
                zf.writestr(name, content)
        return {
            "path": str(out_path),
            "bytes": out_path.stat().st_size,
            "files": sorted(files),
            "events": len(events),
            "messages": len(conversation),
            "compactions": len(compactions),
        }
