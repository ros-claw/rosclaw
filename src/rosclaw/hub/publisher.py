"""ROSClaw Hub asset publisher.

The :class:`Publisher` prepares a local asset directory for publication:

1. Validates the manifest.
2. Scans the asset directory for secrets and other dangerous content.
3. Computes sha256 digests for all artifacts and updates the manifest.
4. Generates ``checksums.txt`` and optional SBOM / provenance files.
5. Optionally creates a placeholder signing certificate and signature.
6. Bundles the prepared directory into a ``.rosclaw`` tar.gz package.
7. Uploads the bundle to a registry when requested.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import shutil
import tarfile
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml

from rosclaw.hub.cache import HubCache
from rosclaw.hub.errors import HubError, HubErrorCode
from rosclaw.hub.refs import AssetRef, ref_from_dict
from rosclaw.hub.schema import AssetManifest, load_manifest

# ---------------------------------------------------------------------------
# Placeholder signing material
# ---------------------------------------------------------------------------
DUMMY_SIGNING_KEY = b"rosclaw-hub-placeholder-signing-key"

DUMMY_CERT_PEM = """-----BEGIN CERTIFICATE-----
MIIBkTCB+wIJAJHGTVDEAIbdMA0GCSqGSIb3DQEBCwUAMBExDzANBgNVBAMMBnJv
c2NsYXdBMB4XDTI2MDEwMTAwMDAwMFoXDTM2MDEwMTAwMDAwMFowETEPMA0GA1UE
AwwGcm9zY2xhdzCBnzANBgkqhkiG9w0BAQEFAAOBjQAwgYkCgYEAyV0P5fF2Kb3v
2TEqK2h7mM7R1gM0w1j6xK3g8WJG8f7a4Vt6b8lX7M0PBPYe3mP8Qr7lN7hQ9vQ6
yK2h7mM7R1gM0w1j6xK3g8WJG8f7a4Vt6b8lX7M0PBPYe3mP8Qr7lN7hQ9vQ6yK2
h7mM7R1gM0w1j6xK3g8WJG8f7a4Vt6b8lX7M0CAwEAATANBgkqhkiG9w0BAQsFAANB
AF3rXn3jR7Xy2h7mM7R1gM0w1j6xK3g8WJG8f7a4Vt6b8lX7M0PBPYe3mP8Qr7lN
7hQ9vQ6yK2h7mM7R1gM0w1j6xK3g8WJG8f7a4Vt6b8lX7M0=
-----END CERTIFICATE-----
"""


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


def scan_secrets(asset_dir: Path) -> list[str]:
    """Recursively scan *asset_dir* for leaked secrets.

    Returns:
        A list of human-readable findings.
    """
    findings: list[str] = []
    for path in sorted(asset_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(asset_dir)
        except ValueError:
            rel = path
        findings.extend(_scan_file(path, str(rel)))
    return findings


def _artifact_files(asset_dir: Path, manifest: AssetManifest) -> list[Path]:
    """Return the on-disk paths for declared artifacts."""
    files: list[Path] = []
    for artifact in manifest.artifacts:
        rel = artifact.get("path")
        if not rel:
            continue
        path = asset_dir / rel
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
        file_path = asset_dir / rel_path
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


def _build_checksums_text(asset_dir: Path, manifest: AssetManifest) -> bytes:
    """Build the canonical ``checksums.txt`` content.

    Includes ``manifest.yaml`` and every declared artifact. Files are listed
    in a stable order.
    """
    lines: list[str] = []
    manifest_path = asset_dir / "manifest.yaml"
    if manifest_path.exists():
        digest = _sha256_hex(manifest_path.read_bytes())
        lines.append(f"sha256:{digest}  manifest.yaml")

    for artifact in manifest.artifacts:
        rel_path = artifact.get("path")
        if not rel_path:
            continue
        file_path = asset_dir / rel_path
        if not file_path.exists():
            continue
        digest = _sha256_hex(file_path.read_bytes())
        lines.append(f"sha256:{digest}  {rel_path}")

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
        file_path = asset_dir / rel_path
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


def _sign_checksums(checksums_path: Path, signature_path: Path) -> None:
    """Create a placeholder HMAC signature over the checksums file."""
    data = checksums_path.read_bytes()
    signature = hmac.new(DUMMY_SIGNING_KEY, data, hashlib.sha256).digest()
    signature_path.parent.mkdir(parents=True, exist_ok=True)
    signature_path.write_bytes(signature)


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
        source = Path(asset_dir).expanduser().resolve()
        if not source.is_dir():
            raise HubError(
                code=HubErrorCode.ASSET_NOT_FOUND,
                message=f"Asset directory not found: {source}",
            )

        manifest = load_manifest(source / "manifest.yaml")
        warnings: list[str] = []

        # Secret scanning.
        secret_findings = scan_secrets(source)
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
        warnings.extend(_update_artifact_digests(manifest, source))

        # Prepare a staging copy.
        prepared = self.cache.staging_path(prefix="publish")
        shutil.copytree(source, prepared, dirs_exist_ok=True)

        # Write the updated manifest.
        manifest_bytes = _manifest_to_yaml_bytes(manifest)
        self.cache._atomic_write(prepared / "manifest.yaml", manifest_bytes)

        # Checksums.
        checksums_file = manifest.security.get("checksums", {}).get("file", "checksums.txt")
        checksums_path = prepared / checksums_file
        checksums_bytes = _build_checksums_text(prepared, manifest)
        self.cache._atomic_write(checksums_path, checksums_bytes)

        # SBOM.
        sbom_file = manifest.security.get("sbom")
        if sbom_file:
            sbom = _build_sbom(manifest, prepared)
            self.cache._atomic_write(
                prepared / sbom_file,
                json.dumps(sbom, indent=2, ensure_ascii=False).encode("utf-8"),
            )

        # Provenance (needs final bundle digest, so write a placeholder now
        # and replace after bundling).
        provenance_file = manifest.security.get("provenance")
        if provenance_file:
            placeholder_digest = _sha256_hex(b"")
            provenance = _build_provenance(manifest, placeholder_digest)
            self.cache._atomic_write(
                prepared / provenance_file,
                json.dumps(provenance, indent=2, ensure_ascii=False).encode("utf-8"),
            )

        # Signing.
        signing = manifest.security.get("signing", {})
        if self.options.sign or signing.get("required", False):
            cert_rel = signing.get("certificate", "signatures/cert.pem")
            cert_path = prepared / cert_rel
            cert_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache._atomic_write(cert_path, DUMMY_CERT_PEM.encode("utf-8"))
            sig_path = prepared / "signatures" / "signature.bin"
            _sign_checksums(checksums_path, sig_path)

        return prepared, manifest, warnings

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

        # Update provenance with real bundle digest if present.
        provenance_file = manifest.security.get("provenance")
        if provenance_file:
            provenance_path = prepared_dir / provenance_file
            if provenance_path.exists():
                provenance = _build_provenance(manifest, bundle_digest)
                self.cache._atomic_write(
                    provenance_path,
                    json.dumps(provenance, indent=2, ensure_ascii=False).encode("utf-8"),
                )
                # Re-create the bundle so the provenance file is final.
                bundle_name.unlink()
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
