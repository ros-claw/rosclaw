"""Local App installation and immutable manifest lock state."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import FileLock

from rosclaw.app.schema import AppManifest
from rosclaw.firstboot.workspace import resolve_home

APP_STORE_SCHEMA_VERSION = "rosclaw.app.store.v1"


class AppStoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class InstalledApp:
    name: str
    version: str
    path: str
    manifest_digest: str
    installed_at: str
    source: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> InstalledApp:
        return cls(
            name=_required(value, "name"),
            version=_required(value, "version"),
            path=_required(value, "path"),
            manifest_digest=_required(value, "manifest_digest"),
            installed_at=_required(value, "installed_at"),
            source=_required(value, "source"),
        )


class AppStore:
    def __init__(self, home: str | Path | None = None) -> None:
        self.home = resolve_home(str(home) if home is not None else None)
        self.root = self.home / "apps" / "installed"
        self.index_path = self.home / "apps" / "apps.lock.json"
        self.lock_path = self.home / "state" / "locks" / "apps.lock"

    def install(self, source: str | Path, *, force: bool = False) -> InstalledApp:
        source_path = self._resolve_source(source)
        manifest = AppManifest.from_path(source_path)
        manifest_path = source_path / "app.yaml" if source_path.is_dir() else source_path
        digest = _digest(manifest_path)
        destination = self.root / manifest.metadata.name / manifest.metadata.version
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.lock_path)):
            records = self._load_unlocked()
            existing = records.get(manifest.metadata.name)
            if existing is not None and not force:
                if (
                    existing.version == manifest.metadata.version
                    and existing.manifest_digest == digest
                ):
                    return existing
                raise AppStoreError(
                    f"App {manifest.metadata.name!r} is installed; use --force to replace it"
                )
            temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir(parents=True)
            shutil.copy2(manifest_path, temporary / "app.yaml")
            copied = AppManifest.from_path(temporary)
            if copied != manifest or _digest(temporary / "app.yaml") != digest:
                shutil.rmtree(temporary)
                raise AppStoreError("App manifest changed during installation")
            if destination.exists():
                shutil.rmtree(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary.replace(destination)
            record = InstalledApp(
                name=manifest.metadata.name,
                version=manifest.metadata.version,
                path=str(destination),
                manifest_digest=digest,
                installed_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                source=str(source),
            )
            records[record.name] = record
            self._save_unlocked(records)
            return record

    def resolve(self, identifier: str) -> tuple[InstalledApp, AppManifest]:
        with FileLock(str(self.lock_path)):
            records = self._load_unlocked()
        record = records.get(identifier)
        if record is None:
            matches = [
                item for item in records.values() if f"{item.name}@{item.version}" == identifier
            ]
            if len(matches) != 1:
                raise AppStoreError(f"App is not installed: {identifier}")
            record = matches[0]
        expected = self.root / record.name / record.version
        path = Path(record.path)
        if path.absolute() != expected.absolute() or path.is_symlink():
            raise AppStoreError("Installed App path does not match managed state")
        manifest_path = path / "app.yaml"
        if not manifest_path.is_file() or _digest(manifest_path) != record.manifest_digest:
            raise AppStoreError("Installed App manifest failed integrity verification")
        manifest = AppManifest.from_path(manifest_path)
        if manifest.metadata.name != record.name or manifest.metadata.version != record.version:
            raise AppStoreError("Installed App identity does not match lock state")
        return record, manifest

    def list_installed(self) -> list[InstalledApp]:
        with FileLock(str(self.lock_path)):
            return sorted(self._load_unlocked().values(), key=lambda item: item.name)

    @staticmethod
    def builtin_root() -> Path:
        return Path(__file__).with_name("builtins")

    def _resolve_source(self, source: str | Path) -> Path:
        candidate = Path(source).expanduser()
        if candidate.exists():
            if candidate.is_symlink():
                raise AppStoreError("App source cannot be a symbolic link")
            return candidate
        builtin_name = str(source)
        if builtin_name.startswith("ros-claw/"):
            builtin_name = builtin_name.removeprefix("ros-claw/")
        if "/" in builtin_name or "\\" in builtin_name:
            raise AppStoreError(f"Unknown bundled App: {source}")
        builtin = self.builtin_root() / builtin_name
        if builtin.is_dir():
            return builtin
        raise AppStoreError(
            "App source must be a local path or bundled App name; remote install is not yet supported"
        )

    def _load_unlocked(self) -> dict[str, InstalledApp]:
        if not self.index_path.exists():
            return {}
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AppStoreError(f"App lock is unreadable: {exc}") from exc
        if raw.get("schema_version") != APP_STORE_SCHEMA_VERSION:
            raise AppStoreError("Unsupported App lock schema")
        apps = raw.get("apps")
        if not isinstance(apps, dict):
            raise AppStoreError("App lock must contain an apps object")
        try:
            return {name: InstalledApp.from_dict(value) for name, value in apps.items()}
        except (TypeError, ValueError) as exc:
            raise AppStoreError(f"App lock contains invalid records: {exc}") from exc

    def _save_unlocked(self, records: dict[str, InstalledApp]) -> None:
        payload = {
            "schema_version": APP_STORE_SCHEMA_VERSION,
            "apps": {name: asdict(record) for name, record in sorted(records.items())},
        }
        temporary = self.index_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.index_path)


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _required(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"{key} must be a non-empty string")
    return item


__all__ = ["APP_STORE_SCHEMA_VERSION", "AppStore", "AppStoreError", "InstalledApp"]
