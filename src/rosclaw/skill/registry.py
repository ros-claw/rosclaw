"""Local persistent skill registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from rosclaw.firstboot.workspace import resolve_home
from rosclaw.skill.models import SkillPackage, SkillRef


class SkillLocalRegistry:
    """Persistent registry at ``~/.rosclaw/registry/skills.yaml``."""

    def __init__(self, home: str | Path | None = None) -> None:
        self.home = Path(resolve_home(str(home) if home else None))
        self.path = self.home / "registry" / "skills.yaml"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            try:
                data = yaml.safe_load(self.path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except (OSError, yaml.YAMLError):
                pass
        return {"schema_version": "rosclaw.skill_registry.v1", "skills": {}}

    def _save(self) -> None:
        self.path.write_text(
            yaml.safe_dump(self._data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    def add(self, pkg: SkillPackage) -> None:
        self._data["skills"][pkg.skill_id] = {
            "local_path": str(pkg.root),
            "current_version": pkg.version,
            "current_stage": pkg.skill.metadata.stage if pkg.skill else "draft",
            "last_eval_report": pkg.skill.evidence.latest_eval_report if pkg.skill else None,
        }
        self._save()

    def get(self, ref: SkillRef) -> dict[str, Any] | None:
        key = f"{ref.namespace}/{ref.name}" if ref.namespace else ref.name
        return self._data["skills"].get(key)

    def list_skills(self) -> list[dict[str, Any]]:
        return [{"name": k, **v} for k, v in self._data["skills"].items()]

    def mark_uploaded(self, ref: SkillRef, receipt: dict[str, Any]) -> None:
        key = f"{ref.namespace}/{ref.name}" if ref.namespace else ref.name
        entry = self._data["skills"].setdefault(key, {})
        entry["hub"] = {
            "uploaded": True,
            "visibility": receipt.get("visibility"),
            "last_upload_at": receipt.get("uploaded_at"),
        }
        self._save()

    def mark_rollback(self, ref: SkillRef, from_version: str, to_version: str) -> None:
        key = f"{ref.namespace}/{ref.name}" if ref.namespace else ref.name
        entry = self._data["skills"].setdefault(key, {})
        entry["last_rollback"] = {"from": from_version, "to": to_version}
        self._save()
