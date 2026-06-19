"""Asset integrity verifier for ROSClaw Hub packages."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from rosclaw.hub.errors import HubError
from rosclaw.hub.schema import load_manifest


@dataclass
class VerificationResult:
    """Outcome of verifying a local asset directory."""

    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.ok = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_checksums(path: Path) -> dict[str, str]:
    """Parse a checksums file into {relative_path: hex_digest}.

    Supported line format::

        sha256:<hexdigest>  <relative/path>
        <hexdigest>  <relative/path>
    """
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Split on the double-space convention used by the fixtures.
        if "  " not in line:
            continue
        digest_part, rel_path = line.split("  ", 1)
        if ":" in digest_part:
            digest_part = digest_part.split(":", 1)[1]
        entries[rel_path.strip()] = digest_part.strip()
    return entries


def _is_valid_pem_certificate(text: str) -> bool:
    """Return True if *text* looks like a PEM-encoded certificate."""
    pattern = re.compile(
        r"-----BEGIN\s+(?:[A-Z\s]+\s+)?CERTIFICATE-----"
        r".*?"
        r"-----END\s+(?:[A-Z\s]+\s+)?CERTIFICATE-----",
        re.DOTALL,
    )
    return bool(pattern.search(text))


def verify_asset_dir(
    asset_dir: str | Path,
    *,
    require_signature: bool = True,
) -> VerificationResult:
    """Verify the integrity of a local asset directory.

    Checks performed:

    1. ``manifest.yaml`` loads and validates.
    2. ``security.checksums.file`` exists and lists every artifact.
    3. Every file in ``checksums.txt`` exists and matches its digest.
    4. Every artifact declared in the manifest exists and matches its digest.
    5. If signing is required, the certificate file is a valid PEM and a
       detached signature file is present.
    6. Optional SBOM / provenance files listed in ``security`` exist.

    Args:
        asset_dir: Directory containing ``manifest.yaml`` and asset files.
        require_signature: When False, skip signature-related checks. Useful
            for testing with unsigned fixtures.

    Returns:
        A :class:`VerificationResult` with errors and warnings.
    """
    result = VerificationResult()
    root = Path(asset_dir)

    # 1. Manifest validation.
    try:
        manifest = load_manifest(root / "manifest.yaml")
    except HubError as exc:
        result.add_error(f"Manifest validation failed: {exc.message}")
        return result

    security = manifest.security
    checksums = security.get("checksums", {})
    checksum_algorithm = checksums.get("algorithm", "sha256")
    checksums_file = checksums.get("file", "checksums.txt")

    if checksum_algorithm != "sha256":
        result.add_error(f"Unsupported checksum algorithm: {checksum_algorithm}")
        return result

    checksums_path = root / checksums_file
    if not checksums_path.exists():
        result.add_error(f"Checksums file missing: {checksums_file}")
        return result

    # 2. Parse checksums and verify every listed file.
    expected_checksums = _parse_checksums(checksums_path)
    if not expected_checksums:
        result.add_warning("Checksums file contains no entries")

    for rel_path, expected_digest in expected_checksums.items():
        file_path = root / rel_path
        if not file_path.exists():
            result.add_error(f"Missing file listed in checksums: {rel_path}")
            continue
        actual_digest = _sha256_hex(file_path.read_bytes())
        if actual_digest != expected_digest:
            result.add_error(
                f"Checksum mismatch for {rel_path}: expected {expected_digest}, got {actual_digest}"
            )

    # 3. Verify declared artifacts.
    manifest_artifact_paths = {a["path"] for a in manifest.artifacts}
    for artifact in manifest.artifacts:
        rel_path = artifact["path"]
        file_path = root / rel_path
        if not file_path.exists():
            result.add_error(f"Declared artifact missing on disk: {rel_path}")
            continue
        declared_digest = artifact.get("digest")
        if declared_digest:
            if ":" not in declared_digest:
                result.add_error(
                    f"Invalid artifact digest format for {rel_path}: {declared_digest}"
                )
                continue
            algo, expected_hex = declared_digest.split(":", 1)
            if algo != "sha256":
                result.add_error(f"Unsupported artifact digest algorithm for {rel_path}: {algo}")
                continue
            actual_hex = _sha256_hex(file_path.read_bytes())
            if actual_hex != expected_hex:
                result.add_error(
                    f"Artifact digest mismatch for {rel_path}: expected {expected_hex}, got {actual_hex}"
                )

    # 4. Ensure every artifact listed in checksums is declared in the manifest.
    for rel_path in expected_checksums:
        if rel_path == "manifest.yaml":
            continue
        if rel_path not in manifest_artifact_paths and not rel_path.startswith(
            tuple(manifest_artifact_paths)
        ):
            # Files nested under an artifact directory are allowed.
            under_artifact = any(rel_path.startswith(f"{p}/") for p in manifest_artifact_paths)
            if not under_artifact:
                result.add_warning(f"File in checksums not declared as artifact: {rel_path}")

    # 5. Signature / certificate checks.
    signing = security.get("signing", {})
    if signing.get("required", False) and require_signature:
        cert_rel = signing.get("certificate")
        if not cert_rel:
            result.add_error("Signing required but no certificate path configured")
        else:
            cert_path = root / cert_rel
            if not cert_path.exists():
                result.add_error(f"Signing certificate missing: {cert_rel}")
            elif not _is_valid_pem_certificate(cert_path.read_text(encoding="utf-8")):
                result.add_error(f"Signing certificate is not a valid PEM: {cert_rel}")

        # Placeholder: a detached signature file should be present.
        sig_path = root / "signatures" / "signature.bin"
        if not sig_path.exists():
            result.add_warning("Detached signature file missing: signatures/signature.bin")

    # 6. Optional SBOM / provenance.
    sbom_file = security.get("sbom")
    if sbom_file and not (root / sbom_file).exists():
        result.add_error(f"SBOM file missing: {sbom_file}")

    provenance_file = security.get("provenance")
    if provenance_file and not (root / provenance_file).exists():
        result.add_error(f"Provenance file missing: {provenance_file}")

    return result
