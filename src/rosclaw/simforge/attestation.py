"""Ed25519 trust boundary for externally produced SimForge evidence."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

_SCALE_CURVE_SIGNATURE_DOMAIN = b"ROSCLAW-SIMFORGE-SCALE-CURVE-V1\x00"
_MAX_KEY_FILE_BYTES = 16_384
_MAX_SCALE_CURVE_BYTES = 32 * 1024 * 1024


def create_simforge_signing_key_pair(
    *,
    private_key_path: Path,
    public_key_path: Path,
    source_checkout: Path,
) -> str:
    """Create a non-overwriting Ed25519 PEM key pair outside the checkout."""

    private_path = _external_path(private_key_path, source_checkout)
    public_path = _external_path(public_key_path, source_checkout)
    if private_path == public_path:
        raise ValueError("private and public key paths must differ")
    key = Ed25519PrivateKey.generate()
    private_bytes = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public_bytes = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    _write_new(private_path, private_bytes, mode=0o600)
    try:
        _write_new(public_path, public_bytes, mode=0o644)
    except Exception:
        private_path.unlink(missing_ok=True)
        raise
    return public_key_fingerprint(key.public_key())


def sign_scale_curve(value: dict[str, Any], *, private_key_path: Path) -> dict[str, str]:
    """Sign a scale-curve document before its ``attestation`` field is added."""

    if "attestation" in value:
        raise ValueError("refusing to sign a scale curve that already has an attestation")
    key_path = private_key_path.expanduser().resolve()
    mode = stat.S_IMODE(key_path.stat().st_mode)
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise PermissionError("SimForge signing key must be mode 0600 or stricter")
    key = _load_private_key(_read_bounded(key_path))
    signature = key.sign(_SCALE_CURVE_SIGNATURE_DOMAIN + canonical_scale_curve(value))
    return {
        "algorithm": "ed25519",
        "key_fingerprint": public_key_fingerprint(key.public_key()),
        "signature_base64": base64.b64encode(signature).decode("ascii"),
    }


def verify_scale_curve_signature(
    value: dict[str, Any],
    *,
    expected_public_key_path: Path,
) -> str:
    """Verify an externally produced scale curve against an independent trust key."""

    attestation = value.get("attestation")
    if not isinstance(attestation, dict) or set(attestation) != {
        "algorithm",
        "key_fingerprint",
        "signature_base64",
    }:
        raise ValueError("scale curve requires an exact Ed25519 attestation")
    if attestation.get("algorithm") != "ed25519":
        raise ValueError("scale curve attestation algorithm must be ed25519")
    public_key = _load_public_key(_read_bounded(expected_public_key_path.expanduser().resolve()))
    fingerprint = public_key_fingerprint(public_key)
    if attestation.get("key_fingerprint") != fingerprint:
        raise ValueError("scale curve signing key is not the configured trust key")
    signature_text = attestation.get("signature_base64")
    if not isinstance(signature_text, str) or len(signature_text) > 128:
        raise ValueError("scale curve signature is malformed")
    try:
        signature = base64.b64decode(signature_text, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("scale curve signature is malformed") from exc
    if len(signature) != 64:
        raise ValueError("scale curve signature must contain 64 bytes")
    unsigned = {key: item for key, item in value.items() if key != "attestation"}
    try:
        public_key.verify(
            signature,
            _SCALE_CURVE_SIGNATURE_DOMAIN + canonical_scale_curve(unsigned),
        )
    except InvalidSignature as exc:
        raise ValueError("scale curve signature verification failed") from exc
    return fingerprint


def scale_curve_commitment(value: dict[str, Any]) -> str:
    unsigned = {key: item for key, item in value.items() if key != "attestation"}
    return "sha256:" + hashlib.sha256(canonical_scale_curve(unsigned)).hexdigest()


def canonical_scale_curve(value: dict[str, Any]) -> bytes:
    if not isinstance(value, dict):
        raise ValueError("scale curve root must be a mapping")
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(payload) > _MAX_SCALE_CURVE_BYTES:
        raise ValueError("scale curve exceeds 32 MiB")
    return payload


def public_key_fingerprint(key: Ed25519PublicKey) -> str:
    raw = key.public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _load_private_key(payload: bytes) -> Ed25519PrivateKey:
    try:
        key = serialization.load_pem_private_key(payload, password=None)
    except (TypeError, ValueError) as exc:
        raise ValueError("SimForge signing key is not valid PKCS8 PEM") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("SimForge signing key must be Ed25519")
    return key


def _load_public_key(payload: bytes) -> Ed25519PublicKey:
    try:
        key = serialization.load_pem_public_key(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("SimForge trust key is not valid PEM") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("SimForge trust key must be Ed25519")
    return key


def _read_bounded(path: Path) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if not 1 <= size <= _MAX_KEY_FILE_BYTES:
        raise ValueError("SimForge key file has an invalid size")
    with path.open("rb") as handle:
        payload = handle.read(_MAX_KEY_FILE_BYTES + 1)
    if not 1 <= len(payload) <= _MAX_KEY_FILE_BYTES:
        raise ValueError("SimForge key file has an invalid size")
    return payload


def _external_path(path: Path, source_checkout: Path) -> Path:
    resolved = path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if resolved == checkout or checkout in resolved.parents:
        raise ValueError("SimForge signing keys must be outside the source checkout")
    if not resolved.parent.is_dir():
        raise FileNotFoundError(resolved.parent)
    return resolved


def _write_new(path: Path, payload: bytes, *, mode: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while creating SimForge signing key")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "create_simforge_signing_key_pair",
    "scale_curve_commitment",
    "sign_scale_curve",
    "verify_scale_curve_signature",
]
