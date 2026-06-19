"""Generate stub artifacts, checksums, and dummy signatures for Hub fixtures."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import yaml

FIXTURES_DIR = Path(__file__).parent

DUMMY_CERT = """-----BEGIN CERTIFICATE-----
MIIBkTCB+wIJAKHBfpE
-----END CERTIFICATE-----
"""

INVALID_CERT = "this is not a valid certificate\n"


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def generate_for_valid_dir(asset_dir: Path) -> None:
    manifest_path = asset_dir / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    # Create artifact stub files and update their digests.
    for artifact in manifest.get("artifacts", []):
        rel_path = artifact["path"]
        if rel_path == "manifest.yaml":
            # The manifest checksum lives in checksums.txt, not as a self-referential artifact.
            continue
        artifact_path = asset_dir / rel_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        content = f"stub content for {artifact['name']}\n".encode()
        artifact_path.write_bytes(content)
        digest = sha256_of_bytes(content)
        artifact["digest"] = f"sha256:{digest}"

    # Remove the manifest.yaml artifact entry if present.
    manifest["artifacts"] = [
        a for a in manifest.get("artifacts", []) if a.get("path") != "manifest.yaml"
    ]

    # Write manifest back with updated digests.
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    # Build checksums file.
    checksum_lines: list[str] = []
    manifest_bytes = manifest_path.read_bytes()
    checksum_lines.append(f"sha256:{sha256_of_bytes(manifest_bytes)}  manifest.yaml")
    for artifact in manifest.get("artifacts", []):
        artifact_path = asset_dir / artifact["path"]
        digest = sha256_of_bytes(artifact_path.read_bytes())
        checksum_lines.append(f"sha256:{digest}  {artifact['path']}")

    checksums_path = asset_dir / manifest["security"]["checksums"]["file"]
    checksums_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    # Dummy certificate for placeholder signature verification.
    cert_path = asset_dir / manifest["security"]["signing"]["certificate"]
    cert_path.parent.mkdir(parents=True, exist_ok=True)
    cert_path.write_text(DUMMY_CERT, encoding="utf-8")

    # Placeholder detached signature file.
    sig_path = cert_path.parent / "signature.bin"
    sig_path.write_bytes(b"dummy signature placeholder\n")

    # SBOM and provenance stubs if declared.
    sbom_file = manifest["security"].get("sbom")
    if sbom_file:
        sbom_path = asset_dir / sbom_file
        sbom_path.write_text(
            json.dumps({"spdxVersion": "SPDX-2.3", "packages": []}, indent=2) + "\n",
            encoding="utf-8",
        )
    provenance_file = manifest["security"].get("provenance")
    if provenance_file:
        prov_path = asset_dir / provenance_file
        prov_path.write_text(
            json.dumps({"builder": "rosclaw-local", "steps": []}, indent=2) + "\n",
            encoding="utf-8",
        )

    # License file stub.
    license_file = manifest["license"].get("license_file")
    if license_file:
        (asset_dir / license_file).write_text(
            "PLACEHOLDER LICENSE TEXT\n", encoding="utf-8"
        )


def copy_fixture(src_name: str, dst_name: str) -> Path:
    src = FIXTURES_DIR / src_name
    dst = FIXTURES_DIR / dst_name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def main() -> None:
    valid_dirs = sorted(p for p in FIXTURES_DIR.iterdir() if p.is_dir() and p.name.endswith("_valid"))
    for asset_dir in valid_dirs:
        print(f"Generating artifacts for {asset_dir.name}")
        generate_for_valid_dir(asset_dir)

    # Tampered checksum fixture.
    tampered_checksum = copy_fixture("hardware_mcp_valid", "tampered_checksum")
    checksums_path = tampered_checksum / "checksums.txt"
    lines = checksums_path.read_text(encoding="utf-8").splitlines()
    # Corrupt the first artifact checksum line.
    for i, line in enumerate(lines):
        if "  " in line and not line.endswith("manifest.yaml"):
            parts = line.split("  ", 1)
            lines[i] = f"sha256:{'0' * 64}  {parts[1]}"
            break
    checksums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Tampered signature fixture.
    tampered_signature = copy_fixture("hardware_mcp_valid", "tampered_signature")
    manifest = yaml.safe_load((tampered_signature / "manifest.yaml").read_text(encoding="utf-8"))
    cert_path = tampered_signature / manifest["security"]["signing"]["certificate"]
    cert_path.write_text(INVALID_CERT, encoding="utf-8")

    # Incompatible robot fixture.
    incompatible_robot = copy_fixture("hardware_mcp_valid", "incompatible_robot")
    manifest = yaml.safe_load((incompatible_robot / "manifest.yaml").read_text(encoding="utf-8"))
    manifest["compatibility"]["robot"] = {
        "eurdf_profiles": ["nonexistent/robot"],
        "body_kinds": ["quadruped"],
    }
    (incompatible_robot / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    # License requiring acceptance fixture.
    license_acceptance = copy_fixture("hardware_mcp_valid", "license_requires_acceptance")
    manifest = yaml.safe_load((license_acceptance / "manifest.yaml").read_text(encoding="utf-8"))
    manifest["license"] = {
        "spdx": "ROSCLAW-COMMERCIAL-1.0",
        "license_file": "LICENSE",
        "commercial_use": False,
        "redistribution": False,
        "attribution_required": True,
        "export_control": "dual-use",
    }
    manifest["data_rights"]["allowed_usage"] = ["internal-evaluation"]
    (license_acceptance / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    print("Done.")


if __name__ == "__main__":
    main()
