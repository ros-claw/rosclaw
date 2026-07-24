"""Validated four-GPU MJWarp stress attestation for ContactPush."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from rosclaw.simforge.evaluation import (
    StressEvidence,
    _attest_stress_evidence,
)
from rosclaw.simforge.tasks.contact_push_v3 import CONTACT_PUSH_TASK_ID

_CONTACT_PUSH_STRESS_SIGNATURE_DOMAIN = b"ROSCLAW-SIMFORGE-CONTACT-PUSH-STRESS-V1\x00"
_MAX_KEY_FILE_BYTES = 16_384
_MAX_STRESS_SUMMARY_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class ContactPushStressAttestation:
    candidate_hash: str
    physical_gpus: tuple[str, ...]
    worlds: int
    unique_scenarios: int
    world_steps: int
    critical_backend_disagreements: int
    cpu_force_violations: int
    minimum_exact_label_agreement_rate: float
    finite_state: bool
    shards_complete: bool
    source_hash: str
    source_path: Path
    public_key_fingerprint: str
    summary_commitment: str
    gate_evidence: StressEvidence = field(repr=False, compare=False)
    schema_version: str = "rosclaw.contact_push_stress_attestation.v1"

    @classmethod
    def load(
        cls,
        *,
        summary_path: Path,
        expected_candidate_hash: str,
        expected_public_key_path: Path,
        minimum_worlds: int = 1000,
    ) -> ContactPushStressAttestation:
        path = summary_path.expanduser().resolve()
        payload, value = _read_stress_summary(path)
        fingerprint = verify_contact_push_stress_signature(
            value,
            expected_public_key_path=expected_public_key_path,
        )
        if value.get("schema_version") != "rosclaw.contact_push_mjwarp_four_gpu.v1":
            raise ValueError("invalid ContactPush four-GPU summary schema")
        gpus = tuple(map(str, value.get("successful_gpus") or ()))
        requested_gpus = tuple(map(str, value.get("requested_gpus") or ()))
        worlds = int(value.get("worlds", 0))
        unique = int(value.get("unique_scenarios", 0))
        world_steps = int(value.get("world_steps", 0))
        critical = int(value.get("critical_backend_disagreements", -1))
        force = int(value.get("cpu_force_violations", -1))
        agreement = float(value.get("minimum_exact_label_agreement_rate", -1.0))
        finite = value.get("finite_state") is True
        shards = value.get("shards")
        shard_worlds = (
            sum(int(item.get("worlds", 0)) for item in shards)
            if isinstance(shards, list) and all(isinstance(item, dict) for item in shards)
            else -1
        )
        shard_world_steps = (
            sum(int(item.get("world_steps", 0)) for item in shards)
            if isinstance(shards, list) and all(isinstance(item, dict) for item in shards)
            else -1
        )
        complete = (
            value.get("complete") is True
            and value.get("failures") == []
            and requested_gpus == ("0", "1", "2", "3")
            and gpus == ("0", "1", "2", "3")
            and isinstance(shards, list)
            and len(shards) == 4
            and all(
                _valid_shard(
                    shard,
                    gpu=requested_gpus[index],
                    candidate_hash=expected_candidate_hash,
                )
                for index, shard in enumerate(shards)
            )
            and worlds >= minimum_worlds
            and unique == worlds
            and world_steps == worlds * 1250
            and shard_worlds == worlds
            and shard_world_steps == world_steps
            and critical == 0
            and force == 0
            and finite
        )
        if value.get("candidate_hash") != expected_candidate_hash:
            raise ValueError("four-GPU stress candidate hash mismatch")
        if not math.isfinite(agreement) or not 0 <= agreement <= 1:
            raise ValueError("four-GPU label agreement rate is invalid")
        if not complete:
            raise ValueError("four-GPU ContactPush stress evidence is incomplete")
        commitment = contact_push_stress_commitment(value)
        gate_evidence = _attest_stress_evidence(
            StressEvidence(
                task_id=CONTACT_PUSH_TASK_ID,
                candidate_hash=expected_candidate_hash,
                worlds=worlds,
                complete=True,
                critical_backend_disagreements=critical,
                scale_curve_commitment=commitment,
            )
        )
        return cls(
            candidate_hash=expected_candidate_hash,
            physical_gpus=gpus,
            worlds=worlds,
            unique_scenarios=unique,
            world_steps=world_steps,
            critical_backend_disagreements=critical,
            cpu_force_violations=force,
            minimum_exact_label_agreement_rate=agreement,
            finite_state=finite,
            shards_complete=complete,
            source_hash=_hash_bytes(payload),
            source_path=path,
            public_key_fingerprint=fingerprint,
            summary_commitment=commitment,
            gate_evidence=gate_evidence,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_hash": self.candidate_hash,
            "physical_gpus": list(self.physical_gpus),
            "worlds": self.worlds,
            "unique_scenarios": self.unique_scenarios,
            "world_steps": self.world_steps,
            "critical_backend_disagreements": self.critical_backend_disagreements,
            "cpu_force_violations": self.cpu_force_violations,
            "minimum_exact_label_agreement_rate": (self.minimum_exact_label_agreement_rate),
            "finite_state": self.finite_state,
            "shards_complete": self.shards_complete,
            "source_hash": self.source_hash,
            "source_path": str(self.source_path),
            "public_key_fingerprint": self.public_key_fingerprint,
            "summary_commitment": self.summary_commitment,
        }


def sign_contact_push_stress(
    value: dict[str, Any],
    *,
    private_key_path: Path,
) -> dict[str, str]:
    """Sign a verified ContactPush stress summary with a dedicated domain."""

    if "attestation" in value:
        raise ValueError("refusing to sign stress evidence that already has an attestation")
    key_path = private_key_path.expanduser().resolve()
    mode = stat.S_IMODE(key_path.stat().st_mode)
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise PermissionError("ContactPush stress signing key must be mode 0600 or stricter")
    key = _load_private_key(_read_bounded_key(key_path))
    signature = key.sign(
        _CONTACT_PUSH_STRESS_SIGNATURE_DOMAIN + canonical_contact_push_stress(value)
    )
    return {
        "algorithm": "ed25519",
        "key_fingerprint": _public_key_fingerprint(key.public_key()),
        "signature_base64": base64.b64encode(signature).decode("ascii"),
    }


def verify_contact_push_stress_signature(
    value: dict[str, Any],
    *,
    expected_public_key_path: Path,
) -> str:
    """Verify stress evidence against the qualification trust key."""

    attestation = value.get("attestation")
    if not isinstance(attestation, dict) or set(attestation) != {
        "algorithm",
        "key_fingerprint",
        "signature_base64",
    }:
        raise ValueError("ContactPush stress evidence requires an exact Ed25519 attestation")
    if attestation.get("algorithm") != "ed25519":
        raise ValueError("ContactPush stress attestation algorithm must be ed25519")
    public_key = _load_public_key(
        _read_bounded_key(expected_public_key_path.expanduser().resolve())
    )
    fingerprint = _public_key_fingerprint(public_key)
    if attestation.get("key_fingerprint") != fingerprint:
        raise ValueError("ContactPush stress signing key is not the configured trust key")
    signature_text = attestation.get("signature_base64")
    if not isinstance(signature_text, str) or len(signature_text) > 128:
        raise ValueError("ContactPush stress signature is malformed")
    try:
        signature = base64.b64decode(signature_text, validate=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("ContactPush stress signature is malformed") from exc
    if len(signature) != 64:
        raise ValueError("ContactPush stress signature must contain 64 bytes")
    unsigned = {key: item for key, item in value.items() if key != "attestation"}
    try:
        public_key.verify(
            signature,
            _CONTACT_PUSH_STRESS_SIGNATURE_DOMAIN + canonical_contact_push_stress(unsigned),
        )
    except InvalidSignature as exc:
        raise ValueError("ContactPush stress signature verification failed") from exc
    return fingerprint


def contact_push_stress_commitment(value: dict[str, Any]) -> str:
    unsigned = {key: item for key, item in value.items() if key != "attestation"}
    return _hash_bytes(canonical_contact_push_stress(unsigned))


def canonical_contact_push_stress(value: dict[str, Any]) -> bytes:
    if not isinstance(value, dict):
        raise ValueError("ContactPush stress root must be a mapping")
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(payload) > _MAX_STRESS_SUMMARY_BYTES:
        raise ValueError("ContactPush stress summary exceeds 32 MiB")
    return payload


def _valid_shard(value: dict[str, Any], *, gpu: str, candidate_hash: str) -> bool:
    worlds = value.get("worlds")
    agreement = value.get("exact_label_agreement_rate")
    return bool(
        value.get("schema_version") == "rosclaw.contact_push_mjwarp_shard.v1"
        and value.get("backend") == "mujoco_warp"
        and value.get("candidate_hash") == candidate_hash
        and value.get("physical_gpu") == gpu
        and value.get("visible_devices") == gpu
        and not isinstance(worlds, bool)
        and isinstance(worlds, int)
        and 1 <= worlds <= 4096
        and value.get("unique_scenarios") == worlds
        and value.get("steps_per_world") == 1250
        and value.get("world_steps") == worlds * 1250
        and value.get("finite_state") is True
        and value.get("critical_backend_disagreements") == 0
        and value.get("cpu_force_violations") == 0
        and not isinstance(agreement, bool)
        and isinstance(agreement, (int, float))
        and math.isfinite(float(agreement))
        and 0 <= float(agreement) <= 1
        and _is_sha256(value.get("world_set_commitment"))
    )


def _read_stress_summary(path: Path) -> tuple[bytes, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if not 1 <= path.stat().st_size <= _MAX_STRESS_SUMMARY_BYTES:
        raise ValueError("ContactPush stress summary has an invalid size")
    with path.open("rb") as handle:
        payload = handle.read(_MAX_STRESS_SUMMARY_BYTES + 1)
    if not 1 <= len(payload) <= _MAX_STRESS_SUMMARY_BYTES:
        raise ValueError("ContactPush stress summary has an invalid size")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("ContactPush stress summary is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("ContactPush stress summary root must be a mapping")
    return payload, value


def _read_bounded_key(path: Path) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(path)
    if not 1 <= path.stat().st_size <= _MAX_KEY_FILE_BYTES:
        raise ValueError("ContactPush stress key file has an invalid size")
    with path.open("rb") as handle:
        payload = handle.read(_MAX_KEY_FILE_BYTES + 1)
    if not 1 <= len(payload) <= _MAX_KEY_FILE_BYTES:
        raise ValueError("ContactPush stress key file has an invalid size")
    return payload


def _load_private_key(payload: bytes) -> Ed25519PrivateKey:
    try:
        key = serialization.load_pem_private_key(payload, password=None)
    except (TypeError, ValueError) as exc:
        raise ValueError("ContactPush stress signing key is not valid PKCS8 PEM") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("ContactPush stress signing key must be Ed25519")
    return key


def _load_public_key(payload: bytes) -> Ed25519PublicKey:
    try:
        key = serialization.load_pem_public_key(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("ContactPush stress trust key is not valid PEM") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("ContactPush stress trust key must be Ed25519")
    return key


def _public_key_fingerprint(key: Ed25519PublicKey) -> str:
    raw = key.public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    return _hash_bytes(raw)


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _hash_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


__all__ = [
    "ContactPushStressAttestation",
    "contact_push_stress_commitment",
    "sign_contact_push_stress",
    "verify_contact_push_stress_signature",
]
