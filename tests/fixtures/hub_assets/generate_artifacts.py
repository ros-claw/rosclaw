"""Generate deterministic payloads and real test-only Hub signatures."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
from pathlib import Path

import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

FIXTURES_DIR = Path(__file__).parent
KEYS_DIR = FIXTURES_DIR.parent / "hub_keys"
KEY_ID = "rosclaw-hub-fixture-v1"
SIGNATURE_DOMAIN = b"ROSCLAW-HUB-ASSET-SIGNATURE-V1\x00"
SIGNATURE_FILE = "signatures/manifest.ed25519"


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_private_key() -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(
        (KEYS_DIR / "fixture-private.pem").read_bytes(),
        password=None,
    )
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("Hub fixture key must be Ed25519")
    return key


def generate_for_valid_dir(asset_dir: Path) -> None:
    manifest_path = asset_dir / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    for artifact in manifest.get("artifacts", []):
        relative = artifact["path"]
        if relative == "manifest.yaml":
            continue
        artifact_path = asset_dir / relative
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        content = f"stub content for {artifact['name']}\n".encode()
        artifact_path.write_bytes(content)
        artifact["digest"] = f"sha256:{sha256_of_bytes(content)}"
    manifest["artifacts"] = [
        artifact
        for artifact in manifest.get("artifacts", [])
        if artifact.get("path") != "manifest.yaml"
    ]
    manifest["security"]["signing"] = {
        "required": True,
        "scheme": "ed25519",
        "key_id": KEY_ID,
        "file": SIGNATURE_FILE,
    }
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    sbom_file = manifest["security"].get("sbom")
    if sbom_file:
        (asset_dir / sbom_file).write_text(
            json.dumps({"spdxVersion": "SPDX-2.3", "packages": []}, indent=2) + "\n",
            encoding="utf-8",
        )
    provenance_file = manifest["security"].get("provenance")
    if provenance_file:
        (asset_dir / provenance_file).write_text(
            json.dumps({"builder": "rosclaw-test-fixture", "steps": []}, indent=2) + "\n",
            encoding="utf-8",
        )
    license_file = manifest["license"].get("license_file")
    if license_file:
        (asset_dir / license_file).write_text("TEST FIXTURE LICENSE TEXT\n", encoding="utf-8")

    signatures_dir = asset_dir / "signatures"
    shutil.rmtree(signatures_dir, ignore_errors=True)
    signature_path = asset_dir / SIGNATURE_FILE
    signature_path.parent.mkdir(parents=True, exist_ok=True)

    checksums_file = manifest["security"]["checksums"]["file"]
    excluded = {checksums_file, SIGNATURE_FILE}
    checksum_lines: list[str] = []
    for path in sorted(asset_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(asset_dir).as_posix()
        if relative in excluded:
            continue
        checksum_lines.append(f"sha256:{sha256_of_bytes(path.read_bytes())}  {relative}")
    checksums_bytes = ("\n".join(checksum_lines) + "\n").encode("utf-8")
    (asset_dir / checksums_file).write_bytes(checksums_bytes)

    signature = load_private_key().sign(
        SIGNATURE_DOMAIN + manifest_path.read_bytes() + b"\x00" + checksums_bytes
    )
    signature_path.write_text(
        base64.b64encode(signature).decode("ascii") + "\n",
        encoding="ascii",
    )


def copy_fixture(src_name: str, dst_name: str) -> Path:
    src = FIXTURES_DIR / src_name
    dst = FIXTURES_DIR / dst_name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def main() -> None:
    valid_dirs = sorted(
        path for path in FIXTURES_DIR.iterdir() if path.is_dir() and path.name.endswith("_valid")
    )
    for asset_dir in valid_dirs:
        print(f"Generating artifacts for {asset_dir.name}")
        generate_for_valid_dir(asset_dir)

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
    generate_for_valid_dir(incompatible_robot)

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
    generate_for_valid_dir(license_acceptance)

    tampered_checksum = copy_fixture("hardware_mcp_valid", "tampered_checksum")
    checksums_path = tampered_checksum / "checksums.txt"
    lines = checksums_path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if "  " in line and not line.endswith("manifest.yaml"):
            _digest, relative = line.split("  ", 1)
            lines[index] = f"sha256:{'0' * 64}  {relative}"
            break
    checksums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    tampered_signature = copy_fixture("hardware_mcp_valid", "tampered_signature")
    signature_path = tampered_signature / SIGNATURE_FILE
    signature = bytearray(base64.b64decode(signature_path.read_text(encoding="ascii")))
    signature[0] ^= 0x01
    signature_path.write_text(
        base64.b64encode(signature).decode("ascii") + "\n",
        encoding="ascii",
    )

    print("Done.")


if __name__ == "__main__":
    main()
