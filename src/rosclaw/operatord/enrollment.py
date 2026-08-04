"""Operator enrollment（审计 P0-01）：enrollment key 的生成、存储与 proof 签名。

key 只在 operatord 与 rosclawd（ACL 验证侧）持有——agentd 永不读取。
proof = HMAC-SHA256(key, canonical(request_id|approve|nonce|decided_at|
enrollment_id|display_hash))；nonce 一次性（operatord 侧记录已用 nonce，
拒绝重放；daemon 侧 proposal 本身单次决定）。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path

from rosclaw.contracts.common import ValidationError, new_id

DEV_SIM_ONLY_LABEL = "DEV_SIM_ONLY"
KEY_FILE_MODE = 0o600


class EnrollmentError(ValidationError):
    pass


@dataclass(frozen=True)
class OperatorEnrollment:
    enrollment_id: str
    key: bytes
    created_at: str
    uid: int

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.key).hexdigest()[:16]


def _canonical_proof_payload(
    *,
    request_id: str,
    approve: bool,
    nonce: str,
    decided_at: str,
    enrollment_id: str,
    display_hash: str,
) -> bytes:
    return json.dumps(
        {
            "request_id": request_id,
            "approve": approve,
            "nonce": nonce,
            "decided_at": decided_at,
            "enrollment_id": enrollment_id,
            "display_hash": display_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def sign_decision_proof(
    enrollment: OperatorEnrollment,
    *,
    request_id: str,
    approve: bool,
    nonce: str,
    decided_at: str,
    display_hash: str,
) -> str:
    return hmac.new(
        enrollment.key,
        _canonical_proof_payload(
            request_id=request_id,
            approve=approve,
            nonce=nonce,
            decided_at=decided_at,
            enrollment_id=enrollment.enrollment_id,
            display_hash=display_hash,
        ),
        hashlib.sha256,
    ).hexdigest()


def verify_decision_proof(
    key: bytes,
    *,
    request_id: str,
    approve: bool,
    nonce: str,
    decided_at: str,
    enrollment_id: str,
    display_hash: str,
    proof: str,
) -> bool:
    expected = hmac.new(
        key,
        _canonical_proof_payload(
            request_id=request_id,
            approve=approve,
            nonce=nonce,
            decided_at=decided_at,
            enrollment_id=enrollment_id,
            display_hash=display_hash,
        ),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, proof)


def enroll(home: Path, *, uid: int | None = None, created_at: str = "") -> OperatorEnrollment:
    """生成新 enrollment（写 0600；父目录 0700）。"""
    from datetime import UTC, datetime

    home.mkdir(parents=True, exist_ok=True)
    os.chmod(home, 0o700)
    enrollment = OperatorEnrollment(
        enrollment_id=new_id("oen"),
        key=secrets.token_bytes(32),
        created_at=created_at or datetime.now(UTC).isoformat(),
        uid=os.geteuid() if uid is None else uid,
    )
    path = home / "operator-enrollment.json"
    if path.exists():
        raise EnrollmentError(f"enrollment already exists at {path}; refusing to overwrite")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            {
                "enrollment_id": enrollment.enrollment_id,
                "key_hex": enrollment.key.hex(),
                "created_at": enrollment.created_at,
                "uid": enrollment.uid,
            }
        ),
        encoding="utf-8",
    )
    os.chmod(tmp, KEY_FILE_MODE)
    os.rename(tmp, path)
    os.chmod(path, KEY_FILE_MODE)
    return enrollment


def load_enrollment(home: Path) -> OperatorEnrollment:
    path = home / "operator-enrollment.json"
    if not path.exists():
        raise EnrollmentError(f"no operator enrollment at {path} — run `rosclaw operatord enroll`")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise EnrollmentError(f"enrollment file {path} must be 0600 (found {mode:o})")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return OperatorEnrollment(
            enrollment_id=str(data["enrollment_id"]),
            key=bytes.fromhex(str(data["key_hex"])),
            created_at=str(data.get("created_at", "")),
            uid=int(data.get("uid", os.geteuid())),
        )
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise EnrollmentError(f"enrollment file corrupt: {exc} — quarantined") from exc


def load_or_create_enrollment(home: Path) -> OperatorEnrollment:
    try:
        return load_enrollment(home)
    except EnrollmentError:
        return enroll(home)
