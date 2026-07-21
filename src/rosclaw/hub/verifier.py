"""Cryptographic integrity verifier for ROSClaw Hub asset bundles."""

from __future__ import annotations

import base64
import fnmatch
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from rosclaw.hub.errors import HubError
from rosclaw.hub.schema import AssetManifest, load_manifest

_CHECKSUM_RE = re.compile(r"^(?:sha256:)?([0-9a-f]{64})  (.+)$")
_SIGNATURE_DOMAIN = b"ROSCLAW-HUB-ASSET-SIGNATURE-V1\x00"
_TRUST_STORE_SCHEMA = "rosclaw.hub.trust.v1"


@dataclass
class VerificationResult:
    """Outcome of verifying a local asset directory."""

    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_files: list[str] = field(default_factory=list)
    signature_status: str = "not_checked"
    signature_key_id: str | None = None
    trusted: bool = False

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.ok = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


def default_trust_store_path() -> Path:
    """Return the operator override or packaged Hub trust-store path."""

    override = os.environ.get("ROSCLAW_HUB_TRUST_STORE")
    if override:
        return Path(override).expanduser()
    return Path(__file__).with_name("trust") / "keys.json"


def signature_payload(manifest_bytes: bytes, checksums_bytes: bytes) -> bytes:
    """Build the domain-separated payload covered by a Hub signature."""

    return _SIGNATURE_DOMAIN + manifest_bytes + b"\x00" + checksums_bytes


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_relative_path(relative: str) -> bool:
    if (
        not relative
        or "\\" in relative
        or any(ord(character) < 32 or ord(character) == 127 for character in relative)
    ):
        return False
    candidate = PurePosixPath(relative)
    return (
        not candidate.is_absolute()
        and candidate.parts not in ((), (".",))
        and ".." not in candidate.parts
    )


def is_supported_signature_path(relative: object) -> bool:
    """Return whether *relative* is an isolated detached-signature path."""

    if not isinstance(relative, str) or not _safe_relative_path(relative):
        return False
    candidate = PurePosixPath(relative)
    return (
        len(candidate.parts) >= 2
        and candidate.parts[0] == "signatures"
        and candidate.suffix == ".ed25519"
    )


def _contained_path(
    root: Path,
    relative: object,
    result: VerificationResult,
    *,
    label: str,
) -> Path | None:
    if not isinstance(relative, str) or not _safe_relative_path(relative):
        result.add_error(f"Unsafe {label} path: {relative!r}")
        return None
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        result.add_error(f"Unsafe {label} path: {relative!r}")
        return None
    return candidate


def _reject_unsafe_entries(root: Path, result: VerificationResult) -> None:
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            result.add_error(f"Symbolic links are forbidden in Hub assets: {relative}")
        elif not path.is_dir() and not path.is_file():
            result.add_error(f"Non-regular filesystem entries are forbidden: {relative}")
        if not _safe_relative_path(relative):
            result.add_error(f"Unsafe payload path: {relative!r}")


def _parse_checksums(data: bytes, result: VerificationResult) -> dict[str, str]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        result.add_error("Checksums file is not UTF-8")
        return {}

    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        match = _CHECKSUM_RE.fullmatch(raw_line)
        if match is None:
            result.add_error(f"Invalid checksums line {line_number}")
            continue
        digest, relative = match.groups()
        if not _safe_relative_path(relative):
            result.add_error(f"Unsafe checksums path on line {line_number}: {relative!r}")
            continue
        if relative in entries:
            result.add_error(f"Duplicate checksums path: {relative}")
            continue
        entries[relative] = digest
    if not entries:
        result.add_error("Checksums file contains no payload entries")
    return entries


def _load_trust_store(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != _TRUST_STORE_SCHEMA:
        raise ValueError("unsupported Hub trust-store schema")
    keys = raw.get("keys")
    if not isinstance(keys, dict):
        raise ValueError("Hub trust-store keys must be a mapping")
    return {str(key_id): value for key_id, value in keys.items() if isinstance(value, dict)}


def _canonical_ref(manifest: AssetManifest) -> str:
    asset = manifest.asset
    return f"rosclaw://{asset.type.value}/{asset.namespace}/{asset.name}@{asset.version}"


def _key_scope_allows(key: dict[str, Any], manifest: AssetManifest) -> bool:
    scopes = key.get("scopes")
    return isinstance(scopes, list) and any(
        isinstance(scope, str) and fnmatch.fnmatchcase(_canonical_ref(manifest), scope)
        for scope in scopes
    )


def _verify_signature(
    *,
    root: Path,
    manifest: AssetManifest,
    manifest_bytes: bytes,
    checksums_bytes: bytes,
    signing: object,
    trust_store_path: Path,
    result: VerificationResult,
) -> str | None:
    if not isinstance(signing, dict):
        result.signature_status = "missing"
        result.add_error("A trusted signature is required, but security.signing is missing")
        return None

    scheme = signing.get("scheme")
    if scheme is None:
        result.signature_status = "missing"
        result.add_error("A trusted signature is required, but no signing scheme is declared")
        return None
    if scheme != "ed25519":
        result.signature_status = "unsupported_scheme"
        result.add_error(
            f"Unsupported signing scheme: {scheme!r}; only detached Ed25519 is accepted"
        )
        return None

    key_id = signing.get("key_id")
    if not isinstance(key_id, str) or not key_id:
        result.signature_status = "missing_key_id"
        result.add_error("Ed25519 signing requires a non-empty key_id")
        return None
    result.signature_key_id = key_id

    signature_relative = signing.get("file", "signatures/manifest.ed25519")
    if not is_supported_signature_path(signature_relative):
        result.signature_status = "invalid_path"
        result.add_error("Detached signature path must be beneath signatures/ and end in .ed25519")
        return signature_relative if isinstance(signature_relative, str) else None
    signature_path = _contained_path(
        root,
        signature_relative,
        result,
        label="signature",
    )
    if signature_path is None or not signature_path.is_file():
        result.signature_status = "missing"
        result.add_error(f"Required trusted signature is missing: {signature_relative}")
        return signature_relative if isinstance(signature_relative, str) else None

    try:
        trust_store = _load_trust_store(trust_store_path)
    except Exception as exc:  # noqa: BLE001 - malformed operator trust input
        result.signature_status = "trust_store_invalid"
        result.add_error(f"Hub trust store could not be loaded: {exc}")
        return str(signature_relative)

    key = trust_store.get(key_id)
    if key is None:
        result.signature_status = "unknown_key"
        result.add_error(f"Signature key is not trusted: {key_id}")
        return str(signature_relative)
    if key.get("algorithm") != "ed25519":
        result.signature_status = "unsupported_key"
        result.add_error(f"Unsupported trust key algorithm: {key.get('algorithm')!r}")
        return str(signature_relative)
    if key.get("status") != "trusted":
        result.signature_status = "inactive_key"
        result.add_error(f"Signature key is not active and trusted: {key_id}")
        return str(signature_relative)
    if not _key_scope_allows(key, manifest):
        result.signature_status = "scope_mismatch"
        result.add_error(f"Signature key {key_id} is not scoped for {_canonical_ref(manifest)}")
        return str(signature_relative)

    try:
        public_bytes = base64.b64decode(key["public_key_base64"], validate=True)
        signature_bytes = base64.b64decode(
            signature_path.read_text(encoding="ascii").strip(),
            validate=True,
        )
        if len(public_bytes) != 32:
            raise ValueError("Ed25519 public key must contain 32 raw bytes")
        if len(signature_bytes) != 64:
            raise ValueError("Ed25519 signature must contain 64 raw bytes")
        Ed25519PublicKey.from_public_bytes(public_bytes).verify(
            signature_bytes,
            signature_payload(manifest_bytes, checksums_bytes),
        )
    except InvalidSignature:
        result.signature_status = "invalid"
        result.add_error("Detached Ed25519 signature is invalid")
        return str(signature_relative)
    except Exception as exc:  # noqa: BLE001 - malformed key/signature data
        result.signature_status = "invalid"
        result.add_error(f"Ed25519 signature verification failed: {exc}")
        return str(signature_relative)

    result.signature_status = "valid"
    result.trusted = True
    return str(signature_relative)


def verify_asset_dir(
    asset_dir: str | Path,
    *,
    require_signature: bool = True,
    trust_store_path: str | Path | None = None,
) -> VerificationResult:
    """Verify schema, complete payload hashes, path safety, and publisher trust."""

    result = VerificationResult()
    requested_root = Path(asset_dir).expanduser().absolute()
    if requested_root.is_symlink():
        result.add_error("Hub asset root cannot be a symbolic link")
        return result
    root = requested_root.resolve()
    if not root.is_dir():
        result.add_error(f"Manifest validation failed: asset directory not found: {root}")
        return result

    _reject_unsafe_entries(root, result)
    manifest_path = root / "manifest.yaml"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = load_manifest(manifest_path)
    except (OSError, HubError) as exc:
        message = exc.message if isinstance(exc, HubError) else str(exc)
        result.add_error(f"Manifest validation failed: {message}")
        return result

    security = manifest.security
    checksums = security.get("checksums")
    if not isinstance(checksums, dict):
        result.add_error("security.checksums must be a mapping")
        return result
    checksum_algorithm = checksums.get("algorithm", "sha256")
    if checksum_algorithm != "sha256":
        result.add_error(f"Unsupported checksum algorithm: {checksum_algorithm}")
        return result
    checksums_relative = checksums.get("file", "checksums.txt")
    checksums_path = _contained_path(
        root,
        checksums_relative,
        result,
        label="checksums",
    )
    if checksums_path is None or not checksums_path.is_file():
        result.add_error(f"Checksums file missing: {checksums_relative}")
        return result

    try:
        checksums_bytes = checksums_path.read_bytes()
    except OSError as exc:
        result.add_error(f"Checksums file could not be read: {exc}")
        return result
    expected_checksums = _parse_checksums(checksums_bytes, result)

    for relative, expected_digest in expected_checksums.items():
        file_path = _contained_path(root, relative, result, label="checksums")
        if file_path is None:
            continue
        if not file_path.is_file():
            result.add_error(f"Missing file listed in checksums: {relative}")
            continue
        actual_digest = _sha256_hex(file_path.read_bytes())
        if actual_digest != expected_digest:
            result.add_error(
                f"Checksum mismatch for {relative}: expected {expected_digest}, got {actual_digest}"
            )
            continue
        result.checked_files.append(relative)

    signing = security.get("signing")
    signature_relative: str | None = None
    if require_signature:
        signature_relative = _verify_signature(
            root=root,
            manifest=manifest,
            manifest_bytes=manifest_bytes,
            checksums_bytes=checksums_bytes,
            signing=signing,
            trust_store_path=(
                Path(trust_store_path).expanduser()
                if trust_store_path is not None
                else default_trust_store_path()
            ),
            result=result,
        )
    else:
        result.signature_status = "skipped"
        result.add_warning("Signature verification was explicitly disabled")
        if isinstance(signing, dict) and isinstance(signing.get("file"), str):
            candidate = signing["file"]
            if is_supported_signature_path(candidate):
                signature_relative = candidate
            else:
                result.add_error(
                    "Detached signature path must be beneath signatures/ and end in .ed25519"
                )

    excluded = {str(checksums_relative)}
    if signature_relative:
        excluded.add(signature_relative)
    actual_payload = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    for relative in sorted(actual_payload - set(expected_checksums) - excluded):
        result.add_error(f"Untracked payload file: {relative}")

    for index, artifact in enumerate(manifest.artifacts):
        artifact_relative = artifact.get("path")
        file_path = _contained_path(
            root,
            artifact_relative,
            result,
            label=f"artifacts[{index}]",
        )
        if file_path is None:
            continue
        if not file_path.is_file():
            result.add_error(f"Declared artifact missing on disk: {artifact_relative}")
            continue
        declared_digest = artifact.get("digest")
        if declared_digest is None:
            continue
        if not isinstance(declared_digest, str) or ":" not in declared_digest:
            result.add_error(
                f"Invalid artifact digest format for {artifact_relative}: {declared_digest}"
            )
            continue
        algorithm, expected_hex = declared_digest.split(":", 1)
        if algorithm != "sha256":
            result.add_error(
                f"Unsupported artifact digest algorithm for {artifact_relative}: {algorithm}"
            )
            continue
        actual_hex = _sha256_hex(file_path.read_bytes())
        if actual_hex != expected_hex:
            result.add_error(
                f"Artifact digest mismatch for {artifact_relative}: "
                f"expected {expected_hex}, got {actual_hex}"
            )

    for label, declared_relative in (
        ("SBOM", security.get("sbom")),
        ("Provenance", security.get("provenance")),
    ):
        if declared_relative is None:
            continue
        declared_path = _contained_path(
            root,
            declared_relative,
            result,
            label=label.lower(),
        )
        if declared_path is None or not declared_path.is_file():
            result.add_error(f"{label} file missing: {declared_relative}")

    result.checked_files.sort()
    return result


__all__ = [
    "VerificationResult",
    "default_trust_store_path",
    "is_supported_signature_path",
    "signature_payload",
    "verify_asset_dir",
]
