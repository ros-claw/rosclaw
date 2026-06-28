"""Assets lockfile and cross-process lock utilities for the ROSClaw Hub.

The assets lockfile records which Hub assets are installed on the local
machine.  It lives at ``~/.rosclaw/assets.lock`` and is protected during
writes by a ``filelock`` lock file in ``~/.rosclaw/state/locks/assets.lock``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from filelock import FileLock

from rosclaw.firstboot.workspace import get_rosclaw_home, resolve_home
from rosclaw.hub.errors import HubError, HubErrorCode
from rosclaw.hub.refs import AssetRef, parse_ref


def __getattr__(name: str) -> Any:
    """Lazy module-level constants that re-evaluate with the active ROSCLAW_HOME."""
    if name == "DEFAULT_LOCKFILE_PATH":
        return get_rosclaw_home() / "assets.lock"
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass
class LockEntry:
    """Single installed-asset record in the assets lockfile."""

    ref: str
    source: str
    asset_dir: str
    lifecycle_status: str = "installed"
    health_status: str = "pending"
    installed_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z")
    )
    depends_on: list[str] = field(default_factory=list)

    def asset_ref(self) -> AssetRef:
        """Parse the canonical reference for this entry."""
        return parse_ref(self.ref)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LockEntry:
        """Build a :class:`LockEntry` from a serialized dict."""
        return cls(
            ref=data["ref"],
            source=data.get("source", ""),
            asset_dir=data.get("asset_dir", ""),
            lifecycle_status=data.get("lifecycle_status", "installed"),
            health_status=data.get("health_status", "pending"),
            installed_at=data.get("installed_at", ""),
            depends_on=list(data.get("depends_on", [])),
        )


@dataclass
class AssetsLock:
    """In-memory view of ``~/.rosclaw/assets.lock``.

    The file format is a JSON object with a ``version``, an ``updated_at``
    timestamp, and an ``assets`` array of :class:`LockEntry` records keyed by
    canonical ref string for fast lookup.
    """

    path: Path
    version: str = "1.0"
    updated_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z")
    )
    assets: dict[str, LockEntry] = field(default_factory=dict)

    @classmethod
    def default_path(cls, home: str | Path | None = None) -> Path:
        """Return the default lockfile path for a ROSClaw home."""
        home_str = str(home) if home is not None else None
        return cast(Path, resolve_home(home_str)) / "assets.lock"

    @classmethod
    def load(cls, path: str | Path | None = None) -> AssetsLock:
        """Load the lockfile from disk, or return an empty instance."""
        path = Path(path) if path is not None else cls.default_path()
        if not path.exists():
            return cls(path=path)

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise HubError(
                code=HubErrorCode.MANIFEST_INVALID,
                message=f"Corrupt assets lockfile: {path}",
                suggested_fix="Delete the lockfile and reinstall assets.",
            ) from exc
        except OSError as exc:
            raise HubError(
                code=HubErrorCode.MANIFEST_INVALID,
                message=f"Cannot read assets lockfile: {path}",
            ) from exc

        assets = {
            entry["ref"]: LockEntry.from_dict(entry)
            for entry in data.get("assets", [])
            if isinstance(entry, dict) and "ref" in entry
        }
        return cls(
            path=path,
            version=data.get("version", "1.0"),
            updated_at=data.get("updated_at", ""),
            assets=assets,
        )

    def save(self) -> None:
        """Atomically write the lockfile to disk."""
        self.updated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        data = {
            "version": self.version,
            "updated_at": self.updated_at,
            "assets": [entry.to_dict() for entry in self.assets.values()],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def add(self, entry: LockEntry | dict[str, Any]) -> LockEntry:
        """Add or replace an entry by canonical ref."""
        if isinstance(entry, dict):
            entry = LockEntry.from_dict(entry)
        self.assets[entry.ref] = entry
        return entry

    def remove(self, ref: str | AssetRef) -> bool:
        """Remove an entry by ref.  Returns True if it existed."""
        key = str(ref)
        if key in self.assets:
            del self.assets[key]
            return True
        return False

    def get(self, ref: str | AssetRef) -> LockEntry | None:
        """Return the lock entry for ``ref`` if present."""
        return self.assets.get(str(ref))

    def __contains__(self, ref: str | AssetRef) -> bool:
        return str(ref) in self.assets

    def __iter__(self) -> Iterator[LockEntry]:
        return iter(self.assets.values())

    def list_installed(self) -> list[LockEntry]:
        """Return a stable list of installed asset entries."""
        return sorted(self.assets.values(), key=lambda e: e.ref)

    def is_installed(self, ref: str | AssetRef) -> bool:
        """Return whether ``ref`` is currently installed."""
        return str(ref) in self.assets


@contextmanager
def acquire_assets_lock(
    path: str | Path | None = None,
    timeout: float = -1.0,
) -> Iterator[None]:
    """Acquire the cross-process lock protecting asset mutations.

    Args:
        path: Path to the lock file.  Defaults to
            ``~/.rosclaw/state/locks/assets.lock``.
        timeout: Maximum seconds to wait.  ``-1`` waits forever.

    Yields:
        None once the lock is held.
    """
    lock_path = resolve_home() / "state" / "locks" / "assets.lock" if path is None else Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(lock_path))
    try:
        lock.acquire(timeout=timeout)
        yield
    finally:
        lock.release()
