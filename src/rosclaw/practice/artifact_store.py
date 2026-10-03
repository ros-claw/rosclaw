"""Centralized artifact storage for ROSClaw Practice sessions.

The ``ArtifactStore`` is responsible for persisting raw and derived practice
artifacts (JSONL events, Parquet summaries, YAML snapshots) in a consistent
directory layout, computing SHA-256 checksums, and maintaining an
``artifact_manifest.yaml`` per episode. Writes are idempotent: if an artifact
with the same id and checksum already exists, the store returns the existing
record without touching disk.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rosclaw.practice.storage.layout import PracticeLayout
from rosclaw.storage.durable import DurableNamespace

logger = logging.getLogger("rosclaw.practice.artifact_store")


def _utc_now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass
class ArtifactRecord:
    """Metadata record for one stored artifact."""

    artifact_id: str
    artifact_type: str
    path: str
    sha256: str
    size_bytes: int
    schema_name: str
    created_at: str
    session_id: str
    episode_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ArtifactStore:
    """Write and index practice artifacts under a data root."""

    SCHEMA_VERSION = "artifact_manifest.v1"

    def __init__(self, base_dir: Path | str, layout: PracticeLayout | None = None):
        self._base_dir = Path(base_dir)
        self._durability = DurableNamespace(self._base_dir, kind="practice")
        self._durability.ensure_directory(self._base_dir)
        self._layout = layout or PracticeLayout(self._base_dir)
        self._layout.ensure_directories()

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def artifact_dir(
        self,
        session_id: str,
        episode_id: str | None,
        artifact_type: str,
    ) -> Path:
        """Return the directory that will hold an artifact."""
        if episode_id:
            base = (
                self._layout.sessions_dir
                / session_id
                / "episodes"
                / episode_id
                / "artifacts"
                / artifact_type
            )
        else:
            base = self._layout.sessions_dir / session_id / "artifacts" / artifact_type
        self._durability.ensure_directory(base)
        return base

    def manifest_path(self, session_id: str, episode_id: str | None = None) -> Path:
        """Return the artifact manifest path for a session or episode."""
        if episode_id:
            path = (
                self._layout.sessions_dir
                / session_id
                / "episodes"
                / episode_id
                / "artifact_manifest.yaml"
            )
        else:
            path = self._layout.sessions_dir / session_id / "artifact_manifest.yaml"
        self._durability.ensure_directory(path.parent)
        return path

    def _artifact_path(
        self,
        artifact_id: str,
        artifact_type: str,
        ext: str,
        session_id: str,
        episode_id: str | None,
    ) -> Path:
        directory = self.artifact_dir(session_id, episode_id, artifact_type)
        return directory / f"{artifact_id}{ext}"

    # ------------------------------------------------------------------
    # Core write methods
    # ------------------------------------------------------------------

    def write_jsonl(
        self,
        artifact_id: str,
        records: list[dict[str, Any]],
        *,
        session_id: str,
        episode_id: str | None = None,
        artifact_type: str = "events",
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        """Write a list of records as JSONL and register the artifact."""
        path = self._artifact_path(artifact_id, artifact_type, ".jsonl", session_id, episode_id)
        data = "".join(
            json.dumps(record, ensure_ascii=False, default=str) + "\n" for record in records
        ).encode("utf-8")
        return self._write_and_register(
            path, artifact_id, artifact_type, data, session_id, episode_id, metadata
        )

    def write_yaml(
        self,
        artifact_id: str,
        data: dict[str, Any],
        *,
        session_id: str,
        episode_id: str | None = None,
        artifact_type: str = "snapshot",
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        """Write a YAML snapshot and register the artifact."""
        path = self._artifact_path(artifact_id, artifact_type, ".yaml", session_id, episode_id)
        payload = yaml.safe_dump(data, sort_keys=False, allow_unicode=True).encode("utf-8")
        return self._write_and_register(
            path, artifact_id, artifact_type, payload, session_id, episode_id, metadata
        )

    def write_parquet(
        self,
        artifact_id: str,
        table_or_records: Any,
        *,
        session_id: str,
        episode_id: str | None = None,
        artifact_type: str = "summary",
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        """Write a PyArrow Table or list of dicts as Parquet.

        If ``pyarrow`` is not installed, this method raises ``RuntimeError``.
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as e:
            raise RuntimeError(
                "Parquet support requires pyarrow. Install with: pip install pyarrow"
            ) from e

        path = self._artifact_path(artifact_id, artifact_type, ".parquet", session_id, episode_id)

        if isinstance(table_or_records, pa.Table):
            table = table_or_records
        elif isinstance(table_or_records, list):
            table = self._records_to_table(table_or_records)
        else:
            raise TypeError(f"Expected pyarrow.Table or list[dict], got {type(table_or_records)}")

        buf = io.BytesIO()
        pq.write_table(table, buf)
        data = buf.getvalue()
        return self._write_and_register(
            path, artifact_id, artifact_type, data, session_id, episode_id, metadata
        )

    # ------------------------------------------------------------------
    # Registration / manifest
    # ------------------------------------------------------------------

    def _write_and_register(
        self,
        path: Path,
        artifact_id: str,
        artifact_type: str,
        data: bytes,
        session_id: str,
        episode_id: str | None,
        metadata: dict[str, Any] | None,
    ) -> ArtifactRecord:
        """Atomically write *data* to *path* if its checksum is new, then register."""
        self._durability.ensure_directory(path.parent)
        sha256 = hashlib.sha256(data).hexdigest()
        existing = self._lookup_existing(artifact_id, sha256, session_id, episode_id)
        if existing:
            logger.debug(
                "Artifact %s unchanged (sha256=%s...), skipping write",
                artifact_id,
                sha256[:8],
            )
            self._durability.sync_file(Path(existing.path))
            self._durability.sync_file(self.manifest_path(session_id, episode_id))
            return existing
        manifest = self._load_manifest(session_id, episode_id)
        registered = self._find_in_manifest(manifest, artifact_id)
        if not registered and path.exists():
            if path.read_bytes() != data:
                raise ValueError(
                    "ARTIFACT_CORRUPT: unregistered destination differs; preserve existing bytes"
                )
            self._durability.sync_file(path)
        else:
            self._atomic_write(path, data)
        return self._register(
            path,
            artifact_id,
            artifact_type,
            session_id,
            episode_id,
            metadata,
            sha256=sha256,
            size_bytes=len(data),
        )

    def _atomic_write(self, path: Path, data: bytes) -> None:
        self._durability.atomic_replace(path, data)

    def _lookup_existing(
        self,
        artifact_id: str,
        sha256: str,
        session_id: str,
        episode_id: str | None,
    ) -> ArtifactRecord | None:
        """Return the existing artifact record if its checksum matches."""
        manifest = self._load_manifest(session_id, episode_id)
        existing = self._find_in_manifest(manifest, artifact_id)
        if existing:
            path = Path(existing["path"])
            self._durability.assert_owned(path)
            if not path.is_file() or self._compute_sha256(path) != existing.get("sha256"):
                raise ValueError("ARTIFACT_CORRUPT: registered payload missing or checksum differs")
            if existing.get("sha256") == sha256:
                return ArtifactRecord(**existing)
        return None

    def _register(
        self,
        path: Path,
        artifact_id: str,
        artifact_type: str,
        session_id: str,
        episode_id: str | None,
        metadata: dict[str, Any] | None,
        sha256: str | None = None,
        size_bytes: int | None = None,
    ) -> ArtifactRecord:
        """Compute checksum, update manifest, and return the artifact record."""
        sha256 = sha256 or self._compute_sha256(path)
        size_bytes = size_bytes or path.stat().st_size
        manifest = self._load_manifest(session_id, episode_id)

        existing = self._find_in_manifest(manifest, artifact_id)
        if existing and existing.get("sha256") == sha256:
            logger.debug(
                "Artifact %s unchanged (sha256=%s...), skipping manifest update",
                artifact_id,
                sha256[:8],
            )
            return ArtifactRecord(
                artifact_id=artifact_id,
                artifact_type=existing.get("artifact_type", artifact_type),
                path=str(path),
                sha256=sha256,
                size_bytes=existing.get("size_bytes", size_bytes),
                schema_name=existing.get("schema_name", "unknown"),
                created_at=existing.get("created_at", _utc_now_iso()),
                session_id=session_id,
                episode_id=episode_id,
                metadata=existing.get("metadata", {}) or {},
            )

        record = ArtifactRecord(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            path=str(path),
            sha256=sha256,
            size_bytes=size_bytes,
            schema_name=self._guess_schema_name(path),
            created_at=_utc_now_iso(),
            session_id=session_id,
            episode_id=episode_id,
            metadata=metadata or {},
        )

        manifest["artifacts"][artifact_id] = record.to_dict()
        self._save_manifest(session_id, episode_id, manifest)
        return record

    def _load_manifest(self, session_id: str, episode_id: str | None) -> dict[str, Any]:
        path = self.manifest_path(session_id, episode_id)
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
            except (OSError, UnicodeError, yaml.YAMLError) as error:
                raise ValueError(
                    "ARTIFACT_MANIFEST_CORRUPT: existing manifest unreadable"
                ) from error
            if (
                not isinstance(loaded, dict)
                or not isinstance(loaded.get("artifacts"), dict)
                or loaded.get("schema_version") != self.SCHEMA_VERSION
                or loaded.get("session_id") != session_id
                or loaded.get("episode_id") != episode_id
            ):
                raise ValueError("ARTIFACT_MANIFEST_CORRUPT: invalid manifest envelope")
            return loaded
        return {
            "schema_version": self.SCHEMA_VERSION,
            "session_id": session_id,
            "episode_id": episode_id,
            "artifacts": {},
        }

    def _save_manifest(
        self, session_id: str, episode_id: str | None, manifest: dict[str, Any]
    ) -> None:
        path = self.manifest_path(session_id, episode_id)
        manifest["session_id"] = session_id
        manifest["episode_id"] = episode_id
        self._atomic_write(
            path, yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True).encode("utf-8")
        )

    def _find_in_manifest(
        self, manifest: dict[str, Any], artifact_id: str
    ) -> dict[str, Any] | None:
        return manifest.get("artifacts", {}).get(artifact_id)

    def list_artifacts(
        self, session_id: str, episode_id: str | None = None
    ) -> list[ArtifactRecord]:
        """Return all registered artifacts for a session/episode."""
        manifest = self._load_manifest(session_id, episode_id)
        return [ArtifactRecord(**data) for data in manifest.get("artifacts", {}).values()]

    def get_artifact(
        self,
        artifact_id: str,
        session_id: str,
        episode_id: str | None = None,
    ) -> ArtifactRecord | None:
        """Return a single artifact record if it exists in the manifest."""
        manifest = self._load_manifest(session_id, episode_id)
        data = self._find_in_manifest(manifest, artifact_id)
        return ArtifactRecord(**data) if data else None

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _guess_schema_name(path: Path) -> str:
        mapping = {
            ".jsonl": "jsonl.event.stream",
            ".yaml": "yaml.snapshot",
            ".parquet": "parquet.table",
        }
        return mapping.get(path.suffix, "unknown")

    @staticmethod
    def _records_to_table(records: list[dict[str, Any]]) -> Any:
        import pyarrow as pa

        if not records:
            return pa.Table.from_pydict({})

        # Normalize each record to strings/JSON-serializable values for columns
        # that contain nested dicts/lists, so pyarrow can infer a string column.
        normalized: list[dict[str, Any]] = []
        keys: set[str] = set()
        for record in records:
            row: dict[str, Any] = {}
            for k, v in record.items():
                keys.add(k)
                if isinstance(v, (dict, list)):
                    row[k] = json.dumps(v, ensure_ascii=False, default=str)
                else:
                    row[k] = v
            normalized.append(row)

        # Ensure every row has the same keys.
        for row in normalized:
            for k in keys:
                row.setdefault(k, None)

        return pa.Table.from_pylist(normalized)

    def verify_artifact(
        self,
        artifact_id: str,
        session_id: str,
        episode_id: str | None = None,
    ) -> tuple[bool, str]:
        """Check whether the artifact on disk matches its registered checksum."""
        record = self.get_artifact(artifact_id, session_id, episode_id)
        if record is None:
            return False, f"artifact {artifact_id} not found in manifest"
        path = Path(record.path)
        if not path.exists():
            return False, f"artifact file missing: {path}"
        actual_sha256 = self._compute_sha256(path)
        if actual_sha256 != record.sha256:
            return (
                False,
                f"sha256 mismatch: expected {record.sha256[:16]}..., got {actual_sha256[:16]}...",
            )
        return True, "ok"
