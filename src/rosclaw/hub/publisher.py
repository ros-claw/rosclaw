"""ROSClaw Hub asset publisher.

The :class:`Publisher` prepares a local asset directory for publication:

1. Validates the manifest.
2. Scans the asset directory for secrets and other dangerous content.
3. Computes sha256 digests for all artifacts and updates the manifest.
4. Generates ``checksums.txt`` and optional SBOM / provenance files.
5. Creates a detached Ed25519 signature when publication signing is enabled.
6. Bundles the prepared directory into a ``.rosclaw`` tar.gz package.
7. Uploads the bundle to a registry when requested.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import tarfile
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, cast

import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from rosclaw.hub.cache import HubCache
from rosclaw.hub.errors import HubError, HubErrorCode
from rosclaw.hub.refs import AssetRef, ref_from_dict
from rosclaw.hub.schema import AssetManifest, load_manifest
from rosclaw.hub.verifier import is_supported_signature_path, signature_payload

# ---------------------------------------------------------------------------
# Secret scanner patterns
# ---------------------------------------------------------------------------
_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "private_key",
        re.compile(r"-----BEGIN\s+(?:RSA|OPENSSH|EC|DSA|PGP|ENCRYPTED)?\s*PRIVATE\s+KEY-----"),
    ),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("aws_secret_key", re.compile(r"\b[0-9a-zA-Z/+]{40}\b")),
    (
        "api_key",
        re.compile(r"(?:api[_-]?key|apikey)\s*[:=]\s*['\"]?[\w\-]{16,}['\"]?", re.IGNORECASE),
    ),
    (
        "password",
        re.compile(r"(?:password|passwd|pwd)\s*[:=]\s*['\"][^'\"]{4,}['\"]", re.IGNORECASE),
    ),
    ("bearer_token", re.compile(r"\b[Bb]earer\s+[a-zA-Z0-9_\-\.]{20,}\b")),
    (
        "generic_token",
        re.compile(r"(?:token|secret)\s*[:=]\s*['\"][a-zA-Z0-9_\-\.]{16,}['\"]", re.IGNORECASE),
    ),
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class PublishOptions:
    """Options controlling a publish operation."""

    dry_run: bool = False
    visibility: str | None = None
    sign: bool = False
    registry: str | None = None
    output: Path | None = None
    home: str | Path | None = None
    secret_scan_fail_on_find: bool = True
    signing_key: str | Path | None = None
    signing_key_id: str | None = None


@dataclass
class PublishResult:
    """Outcome of a publish operation."""

    success: bool
    ref: AssetRef
    bundle_path: Path | None
    manifest_digest: str
    size_bytes: int
    dry_run: bool
    warnings: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
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


def _asset_path(asset_dir: Path, relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or not _safe_relative_path(relative):
        raise HubError(
            code=HubErrorCode.PUBLISH_REJECTED,
            message=f"Unsafe {label} path: {relative!r}",
        )
    candidate = (asset_dir / relative).resolve()
    try:
        candidate.relative_to(asset_dir.resolve())
    except ValueError as exc:
        raise HubError(
            code=HubErrorCode.PUBLISH_REJECTED,
            message=f"Unsafe {label} path: {relative!r}",
        ) from exc
    return candidate


def _validate_source_tree(source: Path) -> None:
    issues: list[str] = []
    for path in source.rglob("*"):
        relative = path.relative_to(source).as_posix()
        if path.is_symlink():
            issues.append(f"symbolic link: {relative}")
        elif not path.is_dir() and not path.is_file():
            issues.append(f"non-regular entry: {relative}")
        if not _safe_relative_path(relative):
            issues.append(f"unsafe path: {relative!r}")
    if issues:
        raise HubError(
            code=HubErrorCode.PUBLISH_REJECTED,
            message="Hub asset source tree is unsafe: " + "; ".join(issues),
        )


def _validate_security_layout(
    asset_dir: Path,
    manifest: AssetManifest,
    *,
    signature_file: str | None,
) -> tuple[str, str | None, str | None]:
    """Validate control-file paths before staging can overwrite payload data."""

    checksums = manifest.security.get("checksums")
    if not isinstance(checksums, dict):
        raise HubError(
            code=HubErrorCode.PUBLISH_REJECTED,
            message="security.checksums must be a mapping",
        )
    if checksums.get("algorithm", "sha256") != "sha256":
        raise HubError(
            code=HubErrorCode.PUBLISH_REJECTED,
            message="Hub publication only supports security.checksums.algorithm=sha256",
        )
    checksums_file = checksums.get("file", "checksums.txt")
    if not isinstance(checksums_file, str) or not checksums_file:
        raise HubError(
            code=HubErrorCode.PUBLISH_REJECTED,
            message="security.checksums.file must be a non-empty relative path",
        )

    optional_paths: dict[str, str | None] = {}
    for label, value in (
        ("SBOM", manifest.security.get("sbom")),
        ("provenance", manifest.security.get("provenance")),
    ):
        if value is not None and (not isinstance(value, str) or not value):
            raise HubError(
                code=HubErrorCode.PUBLISH_REJECTED,
                message=f"security.{label.lower()} must be a non-empty relative path",
            )
        optional_paths[label] = value

    controls: dict[Path, str] = {}
    for label, relative in (
        ("manifest", "manifest.yaml"),
        ("checksums", checksums_file),
        ("signature", signature_file),
        ("SBOM", optional_paths["SBOM"]),
        ("provenance", optional_paths["provenance"]),
    ):
        if relative is None:
            continue
        target = _asset_path(asset_dir, relative, label=label)
        prior = controls.get(target)
        if prior is not None:
            raise HubError(
                code=HubErrorCode.PUBLISH_REJECTED,
                message=(
                    f"Security control path collision: {prior} and {label} both use {relative!r}"
                ),
            )
        controls[target] = label

    for index, artifact in enumerate(manifest.artifacts):
        relative = artifact.get("path")
        if not relative:
            continue
        target = _asset_path(asset_dir, relative, label=f"artifacts[{index}]")
        control_label = controls.get(target)
        if control_label is not None:
            raise HubError(
                code=HubErrorCode.PUBLISH_REJECTED,
                message=(
                    f"Artifact path collides with the {control_label} control file: {relative!r}"
                ),
            )

    return checksums_file, optional_paths["SBOM"], optional_paths["provenance"]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _is_text_file(path: Path) -> bool:
    """Return True if *path* looks like a UTF-8 text file."""
    try:
        sample = path.read_bytes()[:1024]
    except OSError:
        return False
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _scan_file(path: Path, rel_path: str) -> list[str]:
    """Scan a single file for secret-like patterns."""
    findings: list[str] = []
    if not _is_text_file(path):
        return findings
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return findings
    lines = text.splitlines()
    for line_number, line in enumerate(lines, start=1):
        for kind, pattern in _SECRET_PATTERNS:
            if pattern.search(line):
                findings.append(f"{kind} pattern found in {rel_path}:{line_number}")
    return findings


def scan_secrets(
    asset_dir: Path,
    *,
    excluded_paths: set[str] | None = None,
) -> list[str]:
    """Recursively scan *asset_dir* for leaked secrets.

    Returns:
        A list of human-readable findings.
    """
    findings: list[str] = []
    excluded_paths = excluded_paths or set()
    for path in sorted(asset_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(asset_dir)
        except ValueError:
            rel = path
        if str(rel) in excluded_paths:
            continue
        findings.extend(_scan_file(path, str(rel)))
    return findings


def _artifact_files(asset_dir: Path, manifest: AssetManifest) -> list[Path]:
    """Return the on-disk paths for declared artifacts."""
    files: list[Path] = []
    for artifact in manifest.artifacts:
        rel = artifact.get("path")
        if not rel:
            continue
        path = _asset_path(asset_dir, rel, label="artifact")
        if path.exists():
            files.append(path)
    return files


def _update_artifact_digests(manifest: AssetManifest, asset_dir: Path) -> list[str]:
    """Compute sha256 digests for declared artifacts and update the manifest.

    Returns:
        Warnings for any missing artifact files.
    """
    warnings: list[str] = []
    for artifact in manifest.artifacts:
        rel_path = artifact.get("path")
        if not rel_path:
            continue
        file_path = _asset_path(asset_dir, rel_path, label="artifact")
        if not file_path.exists():
            warnings.append(f"Declared artifact missing on disk: {rel_path}")
            continue
        data = file_path.read_bytes()
        artifact["digest"] = f"sha256:{_sha256_hex(data)}"
        artifact["size_bytes"] = len(data)
    return warnings


def _manifest_to_yaml_bytes(manifest: AssetManifest) -> bytes:
    """Serialize a manifest to YAML bytes with sorted keys disabled."""
    data: dict[str, Any] = manifest.model_dump(mode="json")
    return cast(
        bytes,
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True).encode("utf-8"),
    )


def _build_checksums_text(
    asset_dir: Path,
    *,
    checksums_file: str,
    signature_file: str | None,
) -> bytes:
    """Hash every payload file except the checksum list and detached signature."""

    lines: list[str] = []
    excluded = {checksums_file}
    if signature_file:
        excluded.add(signature_file)
    for path in sorted(asset_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(asset_dir).as_posix()
        if relative in excluded:
            continue
        lines.append(f"sha256:{_sha256_hex(path.read_bytes())}  {relative}")

    return ("\n".join(lines) + "\n").encode("utf-8")


def _build_sbom(manifest: AssetManifest, asset_dir: Path) -> dict[str, Any]:
    """Build a minimal SPDX 2.3 SBOM for the asset."""
    asset = manifest.asset
    packages: list[dict[str, Any]] = [
        {
            "SPDXID": f"SPDXRef-{asset.type}-{asset.namespace}-{asset.name}",
            "name": asset.name,
            "versionInfo": asset.version,
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": True,
            "supplier": f"Person: {manifest.publisher.display_name}",
        }
    ]
    files: list[dict[str, Any]] = []
    for artifact in manifest.artifacts:
        rel_path = artifact.get("path")
        if not rel_path:
            continue
        file_path = _asset_path(asset_dir, rel_path, label="artifact")
        if not file_path.exists():
            continue
        files.append(
            {
                "SPDXID": f"SPDXRef-File-{artifact.get('name', 'unknown')}",
                "fileName": rel_path,
                "checksums": [
                    {
                        "algorithm": "SHA256",
                        "checksumValue": _sha256_hex(file_path.read_bytes()),
                    }
                ],
            }
        )
    return {
        "spdxVersion": "SPDX-2.3",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"{asset.namespace}-{asset.name}-{asset.version}",
        "documentNamespace": f"https://rosclaw.io/sbom/{asset.type}/{asset.namespace}/{asset.name}@{asset.version}",
        "creationInfo": {
            "created": _now_iso(),
            "creators": [f"Tool: rosclaw-hub-publisher-{_now_iso()}"],
        },
        "packages": packages,
        "files": files,
        "relationships": [
            {
                "spdxElementId": "SPDXRef-DOCUMENT",
                "relatedSpdxElement": packages[0]["SPDXID"],
                "relationshipType": "DESCRIBES",
            }
        ],
    }


def _build_provenance(
    manifest: AssetManifest,
    bundle_digest: str,
) -> dict[str, Any]:
    """Build a minimal SLSA-style provenance statement."""
    asset = manifest.asset
    return {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [
            {
                "name": f"{asset.namespace}/{asset.name}",
                "digest": {"sha256": bundle_digest},
            }
        ],
        "predicateType": "https://slsa.dev/provenance/v1",
        "predicate": {
            "buildDefinition": {
                "buildType": "https://rosclaw.io/build/rosclaw-hub-publish/1",
                "externalParameters": {
                    "asset_type": asset.type.value,
                    "namespace": asset.namespace,
                    "name": asset.name,
                    "version": asset.version,
                },
                "resolvedDependencies": [],
            },
            "runDetails": {
                "builder": {"id": "https://rosclaw.io/builder/rosclaw-hub-publisher"},
                "metadata": {
                    "invocationId": str(uuid.uuid4()),
                    "startedOn": _now_iso(),
                },
            },
        },
    }


def _load_signing_key(path: str | Path) -> Ed25519PrivateKey:
    try:
        key = serialization.load_pem_private_key(
            Path(path).expanduser().read_bytes(),
            password=None,
        )
    except Exception as exc:  # noqa: BLE001 - invalid operator key input
        raise HubError(
            code=HubErrorCode.PUBLISH_REJECTED,
            message=f"Ed25519 private key could not be loaded: {exc}",
        ) from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise HubError(
            code=HubErrorCode.PUBLISH_REJECTED,
            message="Signing private key must be an Ed25519 PKCS8 PEM key",
        )
    return key


def _sign_asset(
    private_key: Ed25519PrivateKey,
    manifest_bytes: bytes,
    checksums_bytes: bytes,
    signature_path: Path,
) -> None:
    signature = private_key.sign(signature_payload(manifest_bytes, checksums_bytes))
    signature_path.parent.mkdir(parents=True, exist_ok=True)
    signature_path.write_text(base64.b64encode(signature).decode("ascii") + "\n", encoding="ascii")


def _catalog_entry_from_manifest(
    manifest: AssetManifest,
    manifest_digest: str,
    manifest_url: str,
    size_bytes: int,
) -> dict[str, Any]:
    """Build a catalog entry dictionary suitable for ``catalog.jsonl``."""
    data: dict[str, Any] = manifest.model_dump(mode="json")
    data["manifest_digest"] = manifest_digest
    data["manifest_url"] = manifest_url
    data["size_bytes"] = size_bytes
    return data


# ---------------------------------------------------------------------------
# Publisher
# ---------------------------------------------------------------------------
class Publisher:
    """Prepare, bundle, and publish ROSClaw Hub assets."""

    def __init__(self, options: PublishOptions | None = None) -> None:
        self.options = options or PublishOptions()
        self.cache = HubCache(self.options.home)

    def prepare(self, asset_dir: str | Path) -> tuple[Path, AssetManifest, list[str]]:
        """Prepare *asset_dir* for publication.

        Returns:
            A tuple of (prepared directory, validated manifest, warnings).
        """
        requested_source = Path(asset_dir).expanduser().absolute()
        if requested_source.is_symlink():
            raise HubError(
                code=HubErrorCode.PUBLISH_REJECTED,
                message="Hub asset root cannot be a symbolic link",
            )
        source = requested_source.resolve()
        if not source.is_dir():
            raise HubError(
                code=HubErrorCode.ASSET_NOT_FOUND,
                message=f"Asset directory not found: {source}",
            )
        _validate_source_tree(source)

        manifest = load_manifest(source / "manifest.yaml")
        warnings: list[str] = []

        declared_signing = manifest.security.get("signing", {})
        declared_signature_file = (
            declared_signing.get("file") if isinstance(declared_signing, dict) else None
        )
        secret_findings = scan_secrets(
            source,
            excluded_paths=(
                {declared_signature_file} if isinstance(declared_signature_file, str) else set()
            ),
        )
        if secret_findings:
            if self.options.secret_scan_fail_on_find and not self.options.dry_run:
                raise HubError(
                    code=HubErrorCode.PUBLISH_REJECTED,
                    message="Secret scan failed:\n  - " + "\n  - ".join(secret_findings),
                    suggested_fix="Remove secrets from the asset before publishing.",
                )
            warnings.extend(secret_findings)

        # Visibility override.
        if self.options.visibility:
            manifest.visibility["scope"] = self.options.visibility

        # Artifact digests.
        artifact_warnings = _update_artifact_digests(manifest, source)
        if artifact_warnings and not self.options.dry_run:
            raise HubError(
                code=HubErrorCode.PUBLISH_REJECTED,
                message="Asset artifacts are incomplete:\n  - " + "\n  - ".join(artifact_warnings),
            )
        warnings.extend(artifact_warnings)

        signing = manifest.security.get("signing", {})
        if not isinstance(signing, dict):
            raise HubError(
                code=HubErrorCode.PUBLISH_REJECTED,
                message="security.signing must be a mapping",
            )
        should_sign = self.options.sign or bool(signing.get("required", False))
        signing_key: Ed25519PrivateKey | None = None
        signature_file: str | None = None
        if should_sign:
            if signing.get("scheme") not in (None, "ed25519"):
                raise HubError(
                    code=HubErrorCode.PUBLISH_REJECTED,
                    message="Signed Hub publication requires security.signing.scheme=ed25519",
                )
            configured_key_id = signing.get("key_id")
            option_key_id = self.options.signing_key_id or os.environ.get(
                "ROSCLAW_HUB_SIGNING_KEY_ID"
            )
            key_id = option_key_id or configured_key_id
            if not isinstance(key_id, str) or not key_id:
                raise HubError(
                    code=HubErrorCode.PUBLISH_REJECTED,
                    message="Signed Hub publication requires a signing key ID",
                )
            if option_key_id and configured_key_id and option_key_id != configured_key_id:
                raise HubError(
                    code=HubErrorCode.PUBLISH_REJECTED,
                    message="Signing key ID does not match security.signing.key_id",
                )
            signing_key_path = self.options.signing_key or os.environ.get("ROSCLAW_HUB_SIGNING_KEY")
            if signing_key_path is None:
                raise HubError(
                    code=HubErrorCode.PUBLISH_REJECTED,
                    message="Signed Hub publication requires an Ed25519 private key",
                    suggested_fix="Pass --signing-key with a PKCS8 PEM key outside the asset directory.",
                )
            signing_key = _load_signing_key(signing_key_path)
            signature_file_value = signing.get("file", "signatures/manifest.ed25519")
            if not isinstance(signature_file_value, str) or not signature_file_value:
                raise HubError(
                    code=HubErrorCode.PUBLISH_REJECTED,
                    message="security.signing.file must be a non-empty relative path",
                )
            signature_file = signature_file_value
            if not is_supported_signature_path(signature_file):
                raise HubError(
                    code=HubErrorCode.PUBLISH_REJECTED,
                    message=(
                        "security.signing.file must be beneath signatures/ and end in .ed25519"
                    ),
                )
            _asset_path(source, signature_file, label="signature")
            signing.update(
                {
                    "required": True,
                    "scheme": "ed25519",
                    "key_id": key_id,
                    "file": signature_file,
                }
            )
        elif isinstance(signing.get("file"), str):
            # An unsigned rebuild must not retain a stale detached signature.
            signature_file = signing["file"]
            if not is_supported_signature_path(signature_file):
                raise HubError(
                    code=HubErrorCode.PUBLISH_REJECTED,
                    message=(
                        "security.signing.file must be beneath signatures/ and end in .ed25519"
                    ),
                )
            _asset_path(source, signature_file, label="signature")

        checksums_file, sbom_file, provenance_file = _validate_security_layout(
            source,
            manifest,
            signature_file=signature_file,
        )

        prepared = self._stage_prepared_asset(
            source=source,
            manifest=manifest,
            checksums_file=checksums_file,
            sbom_file=sbom_file,
            provenance_file=provenance_file,
            signature_file=signature_file,
            signing_key=signing_key,
        )
        return prepared, manifest, warnings

    def _stage_prepared_asset(
        self,
        *,
        source: Path,
        manifest: AssetManifest,
        checksums_file: str,
        sbom_file: str | None,
        provenance_file: str | None,
        signature_file: str | None,
        signing_key: Ed25519PrivateKey | None,
    ) -> Path:
        """Build a prepared staging tree and remove partial output on failure."""

        prepared = self.cache.staging_path(prefix="publish")
        try:
            shutil.copytree(source, prepared, dirs_exist_ok=True)

            manifest_bytes = _manifest_to_yaml_bytes(manifest)
            self.cache._atomic_write(prepared / "manifest.yaml", manifest_bytes)

            if sbom_file:
                sbom = _build_sbom(manifest, prepared)
                self.cache._atomic_write(
                    _asset_path(prepared, sbom_file, label="SBOM"),
                    json.dumps(sbom, indent=2, ensure_ascii=False).encode("utf-8"),
                )

            # Provenance is bound to the immutable manifest. Embedding a
            # bundle's digest inside that same bundle would be self-referential.
            if provenance_file:
                provenance = _build_provenance(manifest, _sha256_hex(manifest_bytes))
                self.cache._atomic_write(
                    _asset_path(prepared, provenance_file, label="provenance"),
                    json.dumps(provenance, indent=2, ensure_ascii=False).encode("utf-8"),
                )

            checksums_path = _asset_path(prepared, checksums_file, label="checksums")
            if signature_file:
                _asset_path(prepared, signature_file, label="signature").unlink(missing_ok=True)
            checksums_bytes = _build_checksums_text(
                prepared,
                checksums_file=checksums_file,
                signature_file=signature_file,
            )
            self.cache._atomic_write(checksums_path, checksums_bytes)

            if signing_key is not None and signature_file is not None:
                _sign_asset(
                    signing_key,
                    manifest_bytes,
                    checksums_bytes,
                    _asset_path(prepared, signature_file, label="signature"),
                )
        except BaseException:
            shutil.rmtree(prepared, ignore_errors=True)
            raise

        return prepared

    def bundle(
        self,
        prepared_dir: Path,
        manifest: AssetManifest,
    ) -> tuple[Path, str, int]:
        """Create a ``.rosclaw`` tar.gz bundle from *prepared_dir*.

        Returns:
            A tuple of (bundle path, bundle sha256 digest, size in bytes).
        """
        ref = ref_from_dict(
            {
                "type": manifest.asset.type.value,
                "namespace": manifest.asset.namespace,
                "name": manifest.asset.name,
                "version": manifest.asset.version,
            }
        )
        bundle_name = (
            self.options.output
            or self.cache.staging_path(prefix="bundle")
            / f"{ref.namespace}-{ref.name}-{ref.version}.rosclaw"
        )
        if bundle_name.is_dir():
            bundle_name = bundle_name / f"{ref.namespace}-{ref.name}-{ref.version}.rosclaw"

        bundle_name.parent.mkdir(parents=True, exist_ok=True)

        with tarfile.open(bundle_name, "w:gz") as tar:
            for path in sorted(prepared_dir.rglob("*")):
                if not path.is_file():
                    continue
                arcname = str(path.relative_to(prepared_dir))
                tar.add(path, arcname=arcname)

        bundle_bytes = bundle_name.read_bytes()
        bundle_digest = _sha256_hex(bundle_bytes)
        size_bytes = len(bundle_bytes)

        return bundle_name, bundle_digest, size_bytes

    def publish(
        self,
        asset_dir: str | Path,
        registry_client: Any | None = None,
    ) -> PublishResult:
        """Prepare, bundle, and optionally upload an asset.

        Args:
            asset_dir: Source asset directory containing ``manifest.yaml``.
            registry_client: Optional registry client with a ``publish_bundle``
                method. If omitted and ``options.registry`` is set, a client is
                built automatically.

        Returns:
            :class:`PublishResult` describing the outcome.
        """
        prepared_dir, manifest, warnings = self.prepare(asset_dir)
        try:
            return self._publish_prepared(
                prepared_dir,
                manifest,
                warnings,
                registry_client=registry_client,
            )
        finally:
            shutil.rmtree(prepared_dir, ignore_errors=True)

    def _publish_prepared(
        self,
        prepared_dir: Path,
        manifest: AssetManifest,
        warnings: list[str],
        *,
        registry_client: Any | None,
    ) -> PublishResult:
        ref = ref_from_dict(
            {
                "type": manifest.asset.type.value,
                "namespace": manifest.asset.namespace,
                "name": manifest.asset.name,
                "version": manifest.asset.version,
            }
        )

        manifest_bytes = _manifest_to_yaml_bytes(manifest)
        manifest_digest = f"sha256:{_sha256_hex(manifest_bytes)}"

        if self.options.registry:
            signing = manifest.security.get("signing", {})
            signature_file = signing.get("file") if isinstance(signing, dict) else None
            if (
                not isinstance(signing, dict)
                or signing.get("scheme") != "ed25519"
                or not isinstance(signature_file, str)
                or not _asset_path(prepared_dir, signature_file, label="signature").is_file()
            ):
                raise HubError(
                    code=HubErrorCode.PUBLISH_REJECTED,
                    message="Registry publication requires a detached Ed25519 signature",
                    suggested_fix="Declare Ed25519 signing and pass --signing-key/--signing-key-id.",
                )

        if self.options.dry_run:
            return PublishResult(
                success=True,
                ref=ref,
                bundle_path=None,
                manifest_digest=manifest_digest,
                size_bytes=0,
                dry_run=True,
                warnings=warnings,
                messages=["Dry-run; no bundle or registry was written."],
            )

        bundle_path, bundle_digest, size_bytes = self.bundle(prepared_dir, manifest)
        messages = [f"Bundle created: {bundle_path}", f"Bundle digest: {bundle_digest}"]

        if self.options.registry:
            client = registry_client or self._registry_client()
            if client is None:
                raise HubError(
                    code=HubErrorCode.AUTH_REQUIRED,
                    message="No registry client available for upload.",
                )
            try:
                upload_info = client.publish_bundle(
                    bundle_path=bundle_path,
                    manifest=manifest,
                    size_bytes=size_bytes,
                )
            except HubError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise HubError(
                    code=HubErrorCode.REGISTRY_UNREACHABLE,
                    message=f"Registry upload failed: {exc}",
                ) from exc
            messages.append(f"Uploaded to registry: {upload_info.get('manifest_url')}")

        return PublishResult(
            success=True,
            ref=ref,
            bundle_path=bundle_path,
            manifest_digest=manifest_digest,
            size_bytes=size_bytes,
            dry_run=False,
            warnings=warnings,
            messages=messages,
        )

    def _registry_client(self) -> Any | None:
        """Build a registry client from ``options.registry`` if possible."""
        from rosclaw.hub.auth import AuthStore
        from rosclaw.hub.client import FakeRegistryClient

        registry = self.options.registry
        if not registry:
            return None
        store = AuthStore(home=self.options.home)
        try:
            token = store.get_token(registry)
        except Exception:  # noqa: BLE001
            token = None
        return FakeRegistryClient(registry, token=token)
