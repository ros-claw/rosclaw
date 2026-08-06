"""Import service (批次 E §8.12)：安全导入 .rcmission。

硬规则：
- magic/schema、大小、文件数、路径穿越、zip bomb 检查；
- checksum 校验；
- secret scan（含 secret 的 bundle 拒绝导入）；
- 导入永远是 IMPORTED_READ_ONLY（archived）新 Mission——imported
  Permit/grant/approval 永远无效，不恢复任何 authority；
- 不执行任何导入的 shell/extension/prompt/code；
- 不覆盖现有 Mission id。
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

from rosclaw.agentd.ui.export_service import BUNDLE_MAGIC, redact_text  # noqa: F401
from rosclaw.contracts.common import ValidationError

if TYPE_CHECKING:
    from rosclaw.agentd.service import AgentService

MAX_FILES = 64
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_MEMBER_BYTES = 16 * 1024 * 1024
REQUIRED = {"manifest.json", "conversation.jsonl", "checksums.txt"}
_SECRET_SCAN = ("sk-", "Bearer ", "api_key\":", "private_key", "permit_secret")


def _read_members(path: Path) -> dict[str, bytes]:
    if not zipfile.is_zipfile(path):
        raise ValidationError("not a zip file")
    with zipfile.ZipFile(path) as zf:
        infos = zf.infolist()
        if len(infos) > MAX_FILES:
            raise ValidationError(f"too many files ({len(infos)} > {MAX_FILES})")
        members: dict[str, bytes] = {}
        total = 0
        for info in infos:
            name = info.filename
            # 路径穿越与绝对路径拒绝。
            if name.startswith(("/", "\\")) or ".." in name.split("/"):
                raise ValidationError(f"unsafe path {name!r}")
            if info.file_size > MAX_MEMBER_BYTES:
                raise ValidationError(f"member {name!r} too large (zip bomb guard)")
            total += info.file_size
            if total > MAX_TOTAL_BYTES:
                raise ValidationError("bundle too large (zip bomb guard)")
            members[name] = zf.read(name)
    return members


def _verify_checksums(members: dict[str, bytes]) -> None:
    checksums = members["checksums.txt"].decode().splitlines()
    for line in checksums:
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        if name not in members:
            raise ValidationError(f"checksum references missing file {name!r}")
        actual = hashlib.sha256(members[name]).hexdigest()
        if actual != digest.strip():
            raise ValidationError(f"checksum mismatch for {name!r}")


class ImportService:
    def __init__(self, service: AgentService) -> None:
        self._service = service

    def preview(self, path: Path) -> dict:
        members = _read_members(path)
        if not set(members) >= REQUIRED:
            raise ValidationError(f"missing required files: {REQUIRED - set(members)}")
        _verify_checksums(members)
        manifest = json.loads(members["manifest.json"].decode())
        if manifest.get("magic") != BUNDLE_MAGIC:
            raise ValidationError(f"bad magic {manifest.get('magic')!r}")
        # secret scan：含 secret 形态的 bundle 拒绝导入。
        for name, content in members.items():
            text = content.decode(errors="replace")
            for pattern in _SECRET_SCAN:
                if pattern in text:
                    raise ValidationError(f"secret-like content in {name} — import refused")
        mission = manifest.get("mission", {})
        return {
            "mission_id": mission.get("mission_id"),
            "goal": mission.get("goal"),
            "mode": mission.get("mode"),
            "exported_at": manifest.get("exported_at"),
            "messages": len(members["conversation.jsonl"].decode().splitlines()),
            "note": "导入后为只读归档 Mission；不恢复任何授权/Permit/审批效力。",
        }

    def import_bundle(self, path: Path) -> dict:
        service = self._service
        preview = self.preview(path)
        members = _read_members(path)
        manifest = json.loads(members["manifest.json"].decode())
        src = manifest.get("mission", {})
        # 新 Mission（永不复用导入的 mission_id）。
        goal = f"[导入] {src.get('goal', 'imported mission')}"
        mission = service.create_mission(goal, mode="SIMULATION")
        service.archive_mission(mission.mission_id)  # IMPORTED_READ_ONLY
        # 对话以导入证据形态进入 journal（untrusted 标记；不恢复 authority）。
        lines = members["conversation.jsonl"].decode().splitlines()
        messages = []
        for line in lines:
            if not line.strip():
                continue
            messages.append(
                {
                    "role": "artifact",
                    "content": f"[imported from {src.get('mission_id')}] {line[:2000]}",
                    "source": "import",
                }
            )
        if messages:
            service.store.append_conversation(
                mission.mission_id, messages, actor_id=service.actor_id
            )
        return {
            "mission_id": mission.mission_id,
            "imported_from": preview["mission_id"],
            "messages_imported": len(messages),
            "read_only": True,
            "authority_restored": False,
        }
