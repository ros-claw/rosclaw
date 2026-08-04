#!/usr/bin/env python3
"""Sign one Robot Pack manifest/checksum pair with an external Ed25519 key."""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from rosclaw.robot_pack.verifier import signature_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pack_root", type=Path)
    parser.add_argument("--private-key", type=Path, required=True)
    args = parser.parse_args()

    pack_root = args.pack_root.resolve()
    key_path = args.private_key.expanduser().resolve()
    if key_path.is_relative_to(pack_root):
        parser.error("--private-key must stay outside the Robot Pack")

    private_key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
    if not isinstance(private_key, Ed25519PrivateKey):
        parser.error("--private-key must contain an Ed25519 private key")

    payload = signature_payload(
        (pack_root / "robot-pack.yaml").read_bytes(),
        (pack_root / "checksums.txt").read_bytes(),
    )
    encoded = base64.b64encode(private_key.sign(payload)) + b"\n"
    signature = pack_root / "signatures" / "manifest.ed25519"
    signature.write_bytes(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
