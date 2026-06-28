"""Builtin skill registry loader.

Provides ``load_builtins(registry)`` so the CLI can register the small set of
perception-only RealSense skills that ship inside ``rosclaw`` itself.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from rosclaw.body.schema import SkillManifest
from rosclaw.skill_manager.registry import SkillEntry, SkillRegistry

logger = logging.getLogger("rosclaw.skill.builtins")


_BUILTIN_DIR = Path(__file__).parent


def _builtin_skills_dir() -> Path:
    return _BUILTIN_DIR


def _load_registry() -> dict[str, Any]:
    path = _BUILTIN_DIR / "registry.yaml"
    if not path.exists():
        return {"skills": {}}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def _import_handler(skill_path: Path) -> Any:
    """Import ``run`` from a skill's runner.py if present."""
    runner = skill_path / "runner.py"
    if not runner.exists():
        return None
    import importlib.util

    spec = importlib.util.spec_from_file_location("runner", runner)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "run", None)


def _manifest_to_skill_entry(manifest: SkillManifest, skill_path: Path) -> SkillEntry:
    """Convert a body SkillManifest into a SkillRegistry SkillEntry."""
    requires = manifest.requires or {}
    return SkillEntry(
        name=manifest.skill_id,
        description=manifest.display_name or manifest.skill_id,
        skill_type="programmed",
        parameters={},
        preconditions=[],
        success_criteria=[],
        handler=_import_handler(skill_path),
        metadata={
            "manifest_path": str(skill_path / "skill.yaml"),
            "manifest": manifest.to_dict(),
            "builtin": True,
            "requires": requires,
        },
        version=manifest.skill_version,
        requirements=requires,
    )


def load_builtins(registry: SkillRegistry | None = None) -> tuple[SkillRegistry, list[SkillEntry]]:
    """Load all builtin skills into the given (or a new) registry.

    Returns:
        ``(registry, list_of_loaded_entries)``
    """
    if registry is None:
        registry = SkillRegistry()

    data = _load_registry()
    loaded: list[SkillEntry] = []
    for skill_name, meta in data.get("skills", {}).items():
        rel_path = meta.get("path", skill_name)
        skill_path = _BUILTIN_DIR / rel_path
        manifest_path = skill_path / "skill.yaml"
        if not manifest_path.exists():
            logger.warning("Builtin skill manifest missing: %s", manifest_path)
            continue
        try:
            manifest = SkillManifest.from_yaml(manifest_path)
            if not manifest.skill_id:
                manifest.skill_id = skill_name
            entry = _manifest_to_skill_entry(manifest, skill_path)
            registry.register(entry)
            loaded.append(entry)
        except Exception as exc:
            logger.warning("Failed to load builtin skill %s: %s", skill_name, exc)
    return registry, loaded


def list_builtin_skills() -> list[dict[str, Any]]:
    """Return metadata for each builtin skill without registering handlers."""
    data = _load_registry()
    results = []
    for skill_name, meta in data.get("skills", {}).items():
        rel_path = meta.get("path", skill_name)
        manifest_path = _BUILTIN_DIR / rel_path / "skill.yaml"
        info = {"name": skill_name, "builtin": True, "path": str(manifest_path)}
        if manifest_path.exists():
            try:
                manifest = SkillManifest.from_yaml(manifest_path)
                info.update({
                    "display_name": manifest.display_name,
                    "version": manifest.skill_version,
                    "description": manifest.display_name,
                })
            except Exception:
                pass
        results.append(info)
    return results


def get_builtin_skill(name: str) -> SkillEntry | None:
    """Load a single builtin skill entry (with handler)."""
    data = _load_registry()
    meta = data.get("skills", {}).get(name)
    if not meta:
        return None
    skill_path = _BUILTIN_DIR / meta.get("path", name)
    manifest_path = skill_path / "skill.yaml"
    if not manifest_path.exists():
        return None
    manifest = SkillManifest.from_yaml(manifest_path)
    if not manifest.skill_id:
        manifest.skill_id = name
    return _manifest_to_skill_entry(manifest, skill_path)
