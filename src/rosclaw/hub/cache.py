"""Local cache and workspace layout for ROSClaw Hub assets."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from rosclaw.firstboot.workspace import resolve_home
from rosclaw.hub.errors import HubError, HubErrorCode
from rosclaw.hub.refs import AssetRef


class HubCache:
    """Manages the ``~/.rosclaw/hub/`` directory tree.

    The cache is content-addressable for blobs and version-addressable for
    manifests. Installed assets are recorded as JSON metadata under
    ``installed/``. All configuration and auth state lives in
    ``~/.rosclaw/config/``.
    """

    def __init__(self, home: str | Path | None = None) -> None:
        home_arg = str(home) if isinstance(home, Path) else home
        self.home = resolve_home(home_arg)
        self.hub_root = self.home / "hub"
        self.blobs_dir = self.hub_root / "blobs" / "sha256"
        self.manifests_dir = self.hub_root / "manifests"
        self.installed_dir = self.hub_root / "installed"
        self.staging_dir = self.hub_root / "staging"
        self.indexes_dir = self.hub_root / "indexes"
        self.backups_dir = self.home / "backups" / "hub"
        self.config_dir = self.home / "config"
        self.locks_dir = self.home / "state" / "locks"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        for path in (
            self.blobs_dir,
            self.manifests_dir,
            self.installed_dir,
            self.staging_dir,
            self.indexes_dir,
            self.backups_dir,
            self.config_dir,
            self.locks_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    def manifest_path(self, ref: AssetRef) -> Path:
        """Return the cached manifest path for ``ref``."""
        if not ref.version:
            raise HubError(
                code=HubErrorCode.MANIFEST_INVALID,
                message="Cannot store a manifest without a version",
                suggested_fix="Use a fully-qualified rosclaw:// reference with @version.",
            )
        return (
            self.manifests_dir
            / ref.type
            / ref.namespace
            / ref.name
            / f"{ref.version}.yaml"
        )

    def blob_path(self, digest: str) -> Path:
        """Return the blob storage path for a content digest."""
        if ":" not in digest:
            raise HubError(
                code=HubErrorCode.CHECKSUM_MISMATCH,
                message=f"Digest must be in the form algorithm:hash, got {digest!r}",
            )
        algorithm, hexdigest = digest.split(":", 1)
        if algorithm != "sha256":
            raise HubError(
                code=HubErrorCode.CHECKSUM_MISMATCH,
                message=f"Unsupported digest algorithm: {algorithm}",
                suggested_fix="ROSClaw Hub currently supports sha256 digests only.",
            )
        return self.blobs_dir / hexdigest

    def installed_path(self, ref: AssetRef) -> Path:
        """Return the installed-state JSON path for ``ref``."""
        if not ref.version:
            raise HubError(
                code=HubErrorCode.MANIFEST_INVALID,
                message="Installed state requires a version",
            )
        return (
            self.installed_dir
            / ref.type
            / ref.namespace
            / ref.name
            / f"{ref.version}.json"
        )

    def staging_path(self, prefix: str | None = None) -> Path:
        """Create and return a unique staging directory."""
        name = f"{prefix or 'stage'}-{uuid.uuid4().hex[:8]}"
        path = self.staging_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def assets_lock_path(self) -> Path:
        """Return the cross-process lock file used for asset mutations."""
        return self.locks_dir / "assets.lock"

    # ------------------------------------------------------------------
    # Manifests
    # ------------------------------------------------------------------
    def put_manifest(self, ref: AssetRef, content: bytes | str) -> Path:
        """Atomically store a manifest in the cache."""
        path = self.manifest_path(ref)
        data = content.encode("utf-8") if isinstance(content, str) else content
        self._atomic_write(path, data)
        return path

    def get_manifest(self, ref: AssetRef) -> Path | None:
        """Return the cached manifest path if it exists."""
        path = self.manifest_path(ref)
        return path if path.exists() else None

    # ------------------------------------------------------------------
    # Blobs
    # ------------------------------------------------------------------
    def put_blob(self, content: bytes, digest: str | None = None) -> Path:
        """Store a blob and optionally verify its digest.

        If ``digest`` is omitted, it is computed from ``content``.
        """
        if digest is None:
            hexdigest = hashlib.sha256(content).hexdigest()
            digest = f"sha256:{hexdigest}"
        path = self.blob_path(digest)
        if path.exists():
            return path

        computed = f"sha256:{hashlib.sha256(content).hexdigest()}"
        if computed != digest:
            raise HubError(
                code=HubErrorCode.CHECKSUM_MISMATCH,
                message="Blob digest does not match content",
                suggested_fix="Re-download the asset or verify the catalog digest.",
            )

        self._atomic_write(path, content)
        return path

    def get_blob(self, digest: str) -> Path | None:
        """Return the blob path if it exists in the cache."""
        path = self.blob_path(digest)
        return path if path.exists() else None

    # ------------------------------------------------------------------
    # Installed-state records
    # ------------------------------------------------------------------
    def set_installed(self, ref: AssetRef, entry: dict[str, Any]) -> Path:
        """Record an asset as installed."""
        path = self.installed_path(ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ref": str(ref),
            "installed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            **entry,
        }
        self._atomic_write(path, json.dumps(record, indent=2, ensure_ascii=False).encode("utf-8"))
        return path

    def get_installed(self, ref: AssetRef) -> dict[str, Any] | None:
        """Return installed-state metadata for ``ref`` if present."""
        path = self.installed_path(ref)
        if not path.exists():
            return None
        try:
            return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            raise HubError(
                code=HubErrorCode.MANIFEST_INVALID,
                message=f"Corrupt installed-state file: {path}",
            ) from exc

    def remove_installed(self, ref: AssetRef) -> bool:
        """Remove the installed-state record for ``ref``."""
        path = self.installed_path(ref)
        if not path.exists():
            return False
        path.unlink()
        # Clean empty parent directories.
        parent = path.parent
        while parent != self.installed_dir and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
        return True

    def list_installed(self) -> list[AssetRef]:
        """Return refs for all recorded installed assets."""
        refs: list[AssetRef] = []
        if not self.installed_dir.exists():
            return refs
        for path in self.installed_dir.rglob("*.json"):
            rel = path.relative_to(self.installed_dir)
            parts = rel.with_suffix("").parts
            if len(parts) != 4:
                continue
            asset_type, namespace, name, version = parts
            refs.append(
                AssetRef(
                    type=asset_type,
                    namespace=namespace,
                    name=name,
                    version=version,
                )
            )
        return refs

    # ------------------------------------------------------------------
    # Backups and atomic writes
    # ------------------------------------------------------------------
    def backup_file(self, path: Path) -> Path:
        """Copy ``path`` into the hub backups directory with a timestamp."""
        if not path.exists():
            raise HubError(
                code=HubErrorCode.ASSET_NOT_FOUND,
                message=f"Cannot backup missing file: {path}",
            )
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        backup_name = f"{path.name}.{timestamp}.bak"
        backup_path = self.backups_dir / backup_name
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_path)
        return backup_path

    def _atomic_write(self, path: Path, data: bytes) -> None:
        """Write ``data`` to ``path`` atomically via a temporary file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except OSError:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def clear_staging(self) -> None:
        """Remove all staging directories."""
        if not self.staging_dir.exists():
            return
        for entry in self.staging_dir.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
