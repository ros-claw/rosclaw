"""Integrity and release-key verification for Robot Pack directories."""

from __future__ import annotations

import base64
import fnmatch
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from rosclaw.robot_pack.schema import RobotPackManifest

_CHECKSUM_RE = re.compile(r"^([0-9a-f]{64})  (.+)$")
_SIGNATURE_DOMAIN = b"ROSCLAW-ROBOT-PACK-SIGNATURE-V1\x00"


class RobotPackVerificationError(ValueError):
    """Raised when a caller requires a valid pack and verification fails."""


@dataclass(frozen=True)
class PackVerificationResult:
    ok: bool
    root: Path
    manifest: RobotPackManifest | None = None
    manifest_digest: str | None = None
    signature_status: str = "not_checked"
    trusted: bool = False
    checked_files: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def require_valid(self) -> RobotPackManifest:
        if not self.ok or self.manifest is None:
            raise RobotPackVerificationError("; ".join(self.errors) or "Robot Pack is invalid")
        return self.manifest

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "root": str(self.root),
            "ref": self.manifest.canonical_ref if self.manifest else None,
            "manifest_digest": self.manifest_digest,
            "signature_status": self.signature_status,
            "trusted": self.trusted,
            "checked_files": list(self.checked_files),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def default_trust_store_path() -> Path:
    return Path(__file__).with_name("trust") / "keys.json"


def signature_payload(manifest_bytes: bytes, checksums_bytes: bytes) -> bytes:
    """Build the domain-separated bytes covered by the detached signature."""

    return _SIGNATURE_DOMAIN + manifest_bytes + b"\x00" + checksums_bytes


def verify_robot_pack(
    root: str | Path,
    *,
    trust_store_path: str | Path | None = None,
) -> PackVerificationResult:
    """Verify schema, path containment, hashes, complete payload coverage, and signature."""

    requested_root = Path(root).expanduser().absolute()
    if requested_root.is_symlink():
        return PackVerificationResult(
            ok=False,
            root=requested_root,
            errors=("Robot Pack root cannot be a symbolic link",),
        )
    pack_root = requested_root.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    checked: list[str] = []
    manifest: RobotPackManifest | None = None
    manifest_digest: str | None = None
    signature_status = "not_checked"
    trusted = False

    if not pack_root.is_dir():
        return PackVerificationResult(
            ok=False,
            root=pack_root,
            errors=(f"Robot Pack directory does not exist: {pack_root}",),
        )

    manifest_path = pack_root / "robot-pack.yaml"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest_digest = f"sha256:{hashlib.sha256(manifest_bytes).hexdigest()}"
        manifest = RobotPackManifest.from_path(manifest_path)
    except Exception as exc:  # noqa: BLE001 - schema failures are reported as data
        errors.append(f"manifest invalid: {exc}")
        return PackVerificationResult(
            ok=False,
            root=pack_root,
            manifest_digest=manifest_digest,
            errors=tuple(errors),
        )

    _reject_symlinks(pack_root, errors)
    checksums_path = _contained_path(pack_root, manifest.integrity.checksums_file, errors)
    if checksums_path is None or not checksums_path.is_file():
        errors.append(f"checksums file missing: {manifest.integrity.checksums_file}")
        checksums_bytes = b""
        expected: dict[str, str] = {}
    else:
        checksums_bytes = checksums_path.read_bytes()
        expected = _parse_checksums(checksums_bytes, errors)

    for relative, expected_digest in expected.items():
        target = _contained_path(pack_root, relative, errors)
        if target is None:
            continue
        if not target.is_file():
            errors.append(f"checksummed file missing: {relative}")
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected_digest:
            errors.append(f"checksum mismatch: {relative}")
            continue
        checked.append(relative)

    tracked_payload = set(expected)
    ignored = {
        "robot-pack.yaml",
        manifest.integrity.checksums_file,
        manifest.integrity.signature.file,
    }
    actual_payload = {
        path.relative_to(pack_root).as_posix()
        for path in pack_root.rglob("*")
        if path.is_file() and path.relative_to(pack_root).as_posix() not in ignored
    }
    for relative in sorted(actual_payload - tracked_payload):
        errors.append(f"untracked payload file: {relative}")
    for relative in sorted(tracked_payload - actual_payload):
        if (pack_root / relative).is_file():
            continue
        errors.append(f"checksums references absent payload: {relative}")

    for component in manifest.components:
        if component.path is None:
            continue
        expected_hex = (component.digest or "").removeprefix("sha256:")
        locked_hex = expected.get(component.path)
        if locked_hex is None:
            errors.append(f"local component is not checksummed: {component.id} ({component.path})")
        elif expected_hex != locked_hex:
            errors.append(f"component digest disagrees with checksums: {component.id}")

    signature = manifest.integrity.signature
    signature_path = _contained_path(pack_root, signature.file, errors)
    if signature_path is None or not signature_path.is_file():
        signature_status = "missing"
        if signature.required:
            errors.append(f"required signature missing: {signature.file}")
    elif checksums_bytes:
        try:
            trust_store = _load_trust_store(
                Path(trust_store_path) if trust_store_path else default_trust_store_path()
            )
            key_entry = trust_store.get(signature.key_id)
            if key_entry is None:
                signature_status = "unknown_key"
                errors.append(f"signature key is not trusted: {signature.key_id}")
            elif key_entry.get("algorithm") != "ed25519":
                signature_status = "unsupported_key"
                errors.append(f"unsupported trust key algorithm: {key_entry.get('algorithm')}")
            elif not _key_scope_allows(key_entry, manifest):
                signature_status = "scope_mismatch"
                errors.append(f"signature key {signature.key_id} is not scoped for this pack")
            else:
                public_bytes = base64.b64decode(key_entry["public_key_base64"], validate=True)
                signature_bytes = base64.b64decode(
                    signature_path.read_text(encoding="ascii").strip(), validate=True
                )
                Ed25519PublicKey.from_public_bytes(public_bytes).verify(
                    signature_bytes,
                    signature_payload(manifest_bytes, checksums_bytes),
                )
                signature_status = "valid"
                trusted = key_entry.get("status") == "trusted"
                if not trusted:
                    errors.append(f"signature key is not active and trusted: {signature.key_id}")
        except InvalidSignature:
            signature_status = "invalid"
            errors.append("detached Ed25519 signature is invalid")
        except Exception as exc:  # noqa: BLE001 - malformed trust/signature data
            signature_status = "invalid"
            errors.append(f"signature verification failed: {exc}")

    return PackVerificationResult(
        ok=not errors,
        root=pack_root,
        manifest=manifest,
        manifest_digest=manifest_digest,
        signature_status=signature_status,
        trusted=trusted,
        checked_files=tuple(sorted(checked)),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _parse_checksums(data: bytes, errors: list[str]) -> dict[str, str]:
    expected: dict[str, str] = {}
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        errors.append("checksums file is not UTF-8")
        return expected
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip():
            continue
        match = _CHECKSUM_RE.fullmatch(raw_line)
        if match is None:
            errors.append(f"invalid checksums line {line_number}")
            continue
        digest, relative = match.groups()
        candidate = PurePosixPath(relative)
        if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
            errors.append(f"unsafe checksums path on line {line_number}: {relative!r}")
            continue
        if relative in expected:
            errors.append(f"duplicate checksums path: {relative}")
            continue
        expected[relative] = digest
    if not expected:
        errors.append("checksums file contains no payload entries")
    return expected


def _contained_path(root: Path, relative: str, errors: list[str]) -> Path | None:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        errors.append(f"path escapes Robot Pack root: {relative!r}")
        return None
    return candidate


def _reject_symlinks(root: Path, errors: list[str]) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            errors.append(f"symlinks are forbidden in Robot Packs: {path.relative_to(root)}")


def _load_trust_store(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "rosclaw.robot_pack.trust.v1":
        raise ValueError("unsupported Robot Pack trust-store schema")
    keys = raw.get("keys")
    if not isinstance(keys, dict):
        raise ValueError("trust-store keys must be a mapping")
    return {str(key): value for key, value in keys.items() if isinstance(value, dict)}


def _key_scope_allows(key: dict[str, Any], manifest: RobotPackManifest) -> bool:
    versioned_ref = (
        f"{manifest.pack.namespace}/{manifest.pack.name}@{manifest.pack.version}"
    )
    scopes = key.get("scopes", [])
    return isinstance(scopes, list) and any(
        isinstance(scope, str) and fnmatch.fnmatchcase(versioned_ref, scope)
        for scope in scopes
    )


__all__ = [
    "PackVerificationResult",
    "RobotPackVerificationError",
    "default_trust_store_path",
    "signature_payload",
    "verify_robot_pack",
]
