"""Transactional local installation and lock state for Robot Packs."""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import FileLock

from rosclaw.firstboot.workspace import resolve_home
from rosclaw.robot_pack.catalog import RobotPackCatalog
from rosclaw.robot_pack.schema import RobotPackManifest, SupportTier
from rosclaw.robot_pack.verifier import verify_robot_pack

_STORE_SCHEMA_VERSION = "rosclaw.robot_pack.store.v1"


class RobotPackStoreError(RuntimeError):
    """Raised when persistent Robot Pack state is invalid or unsafe to mutate."""


@dataclass(frozen=True)
class InstalledRobotPack:
    ref: str
    namespace: str
    name: str
    version: str
    path: str
    manifest_digest: str
    signature_status: str
    trusted: bool
    support_tier: str
    installed_at: str
    source: str
    latest_verification_id: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> InstalledRobotPack:
        required_strings = (
            "ref",
            "namespace",
            "name",
            "version",
            "path",
            "manifest_digest",
            "signature_status",
            "support_tier",
            "installed_at",
            "source",
        )
        for field_name in required_strings:
            if not isinstance(raw.get(field_name), str) or not raw[field_name]:
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(raw.get("trusted"), bool):
            raise ValueError("trusted must be a boolean")
        latest = raw.get("latest_verification_id")
        if latest is not None and (not isinstance(latest, str) or not latest):
            raise ValueError("latest_verification_id must be null or a non-empty string")
        return cls(
            ref=raw["ref"],
            namespace=raw["namespace"],
            name=raw["name"],
            version=raw["version"],
            path=raw["path"],
            manifest_digest=raw["manifest_digest"],
            signature_status=raw["signature_status"],
            trusted=raw["trusted"],
            support_tier=raw["support_tier"],
            installed_at=raw["installed_at"],
            source=raw["source"],
            latest_verification_id=latest,
        )


class RobotPackStore:
    """Own installed pack files and an atomic, cross-process lock document."""

    def __init__(self, home: str | Path | None = None) -> None:
        self.home = resolve_home(str(home) if home is not None else None)
        self.packs_root = self.home / "robots" / "packs"
        self.index_path = self.home / "robots" / "robot-packs.lock.json"
        self.lock_path = self.home / "state" / "locks" / "robot-packs.lock"

    def install(
        self,
        source: str | Path,
        *,
        force: bool = False,
        catalog: RobotPackCatalog | None = None,
    ) -> InstalledRobotPack:
        catalog_entry = (catalog or RobotPackCatalog()).resolve(source)
        verified = catalog_entry.verification
        manifest = verified.require_valid()
        if not verified.trusted or verified.signature_status != "valid":
            raise RobotPackStoreError(
                "Robot Pack installation requires a valid signature from an active trusted key"
            )
        destination = self._destination(manifest)
        self._ensure_dirs()

        with FileLock(str(self.lock_path)):
            existing = self._load_records_unlocked().get(manifest.canonical_ref)
            if existing is not None and not force:
                destination_verification = verify_robot_pack(existing.path)
                if (
                    destination_verification.ok
                    and destination_verification.manifest_digest == verified.manifest_digest
                ):
                    return existing
                raise RobotPackStoreError(
                    f"Robot Pack is already installed but differs from its lock: {manifest.canonical_ref}"
                )

            temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
            backup = destination.parent / f".{destination.name}.old-{uuid.uuid4().hex}"
            temporary.parent.mkdir(parents=True, exist_ok=True)
            activated = False
            try:
                shutil.copytree(catalog_entry.root, temporary, symlinks=False)
                copied = verify_robot_pack(temporary)
                copied_manifest = copied.require_valid()
                if copied.manifest_digest != verified.manifest_digest:
                    raise RobotPackStoreError("Robot Pack changed while it was being installed")
                if not copied.trusted or copied.signature_status != "valid":
                    raise RobotPackStoreError("Copied Robot Pack did not retain trusted integrity")
                if copied_manifest.canonical_ref != manifest.canonical_ref:
                    raise RobotPackStoreError("Copied Robot Pack identity changed during install")

                if destination.exists() or destination.is_symlink():
                    destination.replace(backup)
                temporary.replace(destination)
                activated = True

                now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
                record = InstalledRobotPack(
                    ref=manifest.canonical_ref,
                    namespace=manifest.pack.namespace,
                    name=manifest.pack.name,
                    version=manifest.pack.version,
                    path=str(destination),
                    manifest_digest=str(verified.manifest_digest),
                    signature_status=verified.signature_status,
                    trusted=verified.trusted,
                    support_tier=manifest.support.baseline_tier.value,
                    installed_at=now,
                    source=str(source),
                )
                records = self._load_records_unlocked()
                records[record.ref] = record
                self._save_records_unlocked(records)
            except Exception:
                if temporary.exists():
                    shutil.rmtree(temporary)
                if activated and (destination.exists() or destination.is_symlink()):
                    if destination.is_symlink():
                        destination.unlink()
                    else:
                        shutil.rmtree(destination)
                if backup.exists() or backup.is_symlink():
                    backup.replace(destination)
                raise

            if backup.exists():
                shutil.rmtree(backup, ignore_errors=True)
            return record

    def list_installed(self) -> list[InstalledRobotPack]:
        with FileLock(str(self.lock_path)):
            return sorted(self._load_records_unlocked().values(), key=lambda item: item.ref)

    def resolve_installed(self, identifier: str) -> tuple[InstalledRobotPack, RobotPackManifest]:
        matches: list[tuple[InstalledRobotPack, RobotPackManifest]] = []
        for record in self.list_installed():
            verification = verify_robot_pack(record.path)
            manifest = verification.manifest
            lock_errors: list[str] = []
            if manifest is not None:
                if (
                    record.namespace != manifest.pack.namespace
                    or record.name != manifest.pack.name
                    or record.version != manifest.pack.version
                ):
                    lock_errors.append("lock identity fields do not match signed Pack content")
                if record.signature_status != "valid" or record.trusted is not True:
                    lock_errors.append("lock trust fields do not match signed Pack content")
                if SupportTier(record.support_tier).rank > manifest.support.candidate_tier.rank:
                    lock_errors.append("lock support tier exceeds the signed Pack candidate tier")
            record_matches_content = bool(
                manifest is not None
                and manifest.canonical_ref == record.ref
                and verification.manifest_digest == record.manifest_digest
                and verification.signature_status == "valid"
                and verification.trusted
                and not lock_errors
            )
            if not verification.ok or not record_matches_content or manifest is None:
                record_names = {
                    record.ref,
                    record.name,
                    f"{record.namespace}/{record.name}",
                    *(manifest.pack.aliases if manifest is not None else []),
                }
                if identifier in record_names:
                    detail = "; ".join((*verification.errors, *lock_errors))
                    raise RobotPackStoreError(
                        f"Installed Robot Pack failed integrity verification: {record.ref}: "
                        + (detail or "lock identity, digest, or trust does not match content")
                    )
                continue
            names = {
                record.ref,
                record.name,
                f"{record.namespace}/{record.name}",
                f"{record.namespace}/{record.name}@{record.version}",
                *manifest.pack.aliases,
            }
            if identifier in names:
                matches.append((record, manifest))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RobotPackStoreError(f"Installed Robot Pack reference is ambiguous: {identifier}")
        raise RobotPackStoreError(f"Robot Pack is not installed: {identifier}")

    def record_verification(
        self,
        ref: str,
        *,
        evidence_id: str,
        support_tier: SupportTier,
    ) -> InstalledRobotPack:
        with FileLock(str(self.lock_path)):
            records = self._load_records_unlocked()
            current = records.get(ref)
            if current is None:
                raise RobotPackStoreError(f"Robot Pack is not installed: {ref}")
            current_tier = SupportTier(current.support_tier)
            effective_tier = support_tier if support_tier.rank > current_tier.rank else current_tier
            updated = InstalledRobotPack(
                **{
                    **asdict(current),
                    "support_tier": effective_tier.value,
                    "latest_verification_id": evidence_id,
                }
            )
            records[ref] = updated
            self._save_records_unlocked(records)
            return updated

    def _destination(self, manifest: RobotPackManifest) -> Path:
        return (
            self.packs_root / manifest.pack.namespace / manifest.pack.name / manifest.pack.version
        )

    def _ensure_dirs(self) -> None:
        self.packs_root.mkdir(parents=True, exist_ok=True)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_records_unlocked(self) -> dict[str, InstalledRobotPack]:
        if not self.index_path.exists():
            return {}
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RobotPackStoreError(
                f"Robot Pack lock is unreadable: {self.index_path}: {exc}"
            ) from exc
        if raw.get("schema_version") != _STORE_SCHEMA_VERSION:
            raise RobotPackStoreError("Unsupported Robot Pack lock schema")
        packs = raw.get("packs")
        if not isinstance(packs, dict):
            raise RobotPackStoreError("Robot Pack lock must contain a packs mapping")
        try:
            records: dict[str, InstalledRobotPack] = {}
            for ref, value in packs.items():
                if not isinstance(ref, str) or not isinstance(value, dict):
                    raise ValueError("Pack lock keys must map strings to objects")
                record = InstalledRobotPack.from_dict(value)
                if ref != record.ref:
                    raise ValueError(f"Pack lock key does not match record ref: {ref!r}")
                try:
                    SupportTier(record.support_tier)
                except ValueError as exc:
                    raise ValueError(
                        f"unsupported support tier in Pack lock: {record.support_tier!r}"
                    ) from exc
                expected = (
                    self.packs_root / record.namespace / record.name / record.version
                ).absolute()
                actual = Path(record.path).expanduser().absolute()
                if actual != expected:
                    raise ValueError(
                        f"record path is not its managed Pack destination: {record.path!r}"
                    )
                try:
                    actual.resolve().relative_to(self.packs_root.resolve())
                except ValueError as exc:
                    raise ValueError(
                        f"record path escapes the managed Pack root: {record.path!r}"
                    ) from exc
                records[ref] = record
            return records
        except (KeyError, TypeError, ValueError) as exc:
            raise RobotPackStoreError(f"Robot Pack lock contains an invalid record: {exc}") from exc

    def _save_records_unlocked(self, records: dict[str, InstalledRobotPack]) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _STORE_SCHEMA_VERSION,
            "updated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "packs": {ref: asdict(record) for ref, record in sorted(records.items())},
        }
        temporary = self.index_path.with_suffix(f".json.tmp-{uuid.uuid4().hex}")
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            temporary.replace(self.index_path)
        finally:
            temporary.unlink(missing_ok=True)


__all__ = ["InstalledRobotPack", "RobotPackStore", "RobotPackStoreError"]
