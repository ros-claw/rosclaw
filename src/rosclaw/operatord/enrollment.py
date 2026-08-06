"""Operator identity（二次复核 R2/P0-5）：Ed25519 替代共享 HMAC secret。

设计要点：

* 私钥只存 operatord 私有 home（0600、``O_NOFOLLOW|O_CREAT|O_EXCL``、
  原子写 + 文件/目录双 fsync）；**rosclawd 只保存公钥**，双方不再共享
  对称秘密——daemon 侧被读不再泄露可伪造 proof 的材料。
* 损坏的身份文件按时间戳 quarantine（仍 0600），不静默重建——
  重建意味着旧 enrollment 静默失效，必须人来决定。
* 公钥另存 0644 ``operator-pubkey.pem``，供 daemon 管理员
  （`rosclaw operatord register-daemon`）登记进 daemon 的持久化
  registry。
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from rosclaw.contracts.common import ValidationError, new_id
from rosclaw.contracts.operator.decision import (
    generate_ed25519_keypair,
    key_fingerprint,
    private_key_from_pem,
    private_key_to_pem,
    sign_b64,
)

DEV_SIM_ONLY_LABEL = "DEV_SIM_ONLY"
KEY_FILE_MODE = 0o600
IDENTITY_FILE = "operator-identity.json"
PUBKEY_FILE = "operator-pubkey.pem"


class EnrollmentError(ValidationError):
    pass


@dataclass(frozen=True)
class OperatorIdentity:
    enrollment_id: str
    private_key: Ed25519PrivateKey
    public_key_pem: str
    created_at: str
    uid: int

    @property
    def fingerprint(self) -> str:
        return key_fingerprint(self.public_key_pem)

    def sign(self, payload: bytes) -> str:
        return sign_b64(self.private_key, payload)


def _atomic_write_private(path: Path, content: str, *, exclusive: bool) -> None:
    """O_NOFOLLOW +（可选）O_EXCL 原子写，文件+目录双 fsync。"""
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if exclusive:
        flags |= os.O_EXCL
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, flags, KEY_FILE_MODE)
    try:
        os.fchmod(fd, KEY_FILE_MODE)
        os.write(fd, content.encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    if os.path.islink(path):
        raise EnrollmentError(f"refusing to replace symlink at {path}")
    os.rename(tmp, path)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def enroll(home: Path, *, uid: int | None = None, created_at: str = "") -> OperatorIdentity:
    """生成新 operator 身份（私钥 0600；公钥 0644；拒覆盖）。"""
    home.mkdir(parents=True, exist_ok=True)
    os.chmod(home, 0o700)
    path = home / IDENTITY_FILE
    if path.exists() or os.path.islink(path):
        raise EnrollmentError(f"enrollment already exists at {path}; refusing to overwrite")
    private, public_pem = generate_ed25519_keypair()
    identity = OperatorIdentity(
        enrollment_id=new_id("oen"),
        private_key=private,
        public_key_pem=public_pem,
        created_at=created_at or datetime.now(UTC).isoformat(),
        uid=os.geteuid() if uid is None else uid,
    )
    _atomic_write_private(
        path,
        json.dumps(
            {
                "enrollment_id": identity.enrollment_id,
                "private_key_pem": private_key_to_pem(private),
                "public_key_pem": public_pem,
                "created_at": identity.created_at,
                "uid": identity.uid,
            }
        ),
        exclusive=True,
    )
    (home / PUBKEY_FILE).write_text(public_pem, encoding="utf-8")
    os.chmod(home / PUBKEY_FILE, 0o644)
    return identity


def _quarantine(path: Path, reason: str) -> None:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    quarantined = path.with_name(f"{path.name}.corrupt-{stamp}")
    os.rename(path, quarantined)
    os.chmod(quarantined, KEY_FILE_MODE)
    raise EnrollmentError(
        f"enrollment file corrupt ({reason}) — quarantined to {quarantined.name}; "
        "restore from backup or re-enroll explicitly"
    )


def load_identity(home: Path) -> OperatorIdentity:
    path = home / IDENTITY_FILE
    if not path.exists():
        raise EnrollmentError(f"no operator identity at {path} — run `rosclaw operatord enroll`")
    st = path.stat()
    if stat.S_IMODE(st.st_mode) & 0o077:
        raise EnrollmentError(f"identity file {path} must be 0600 (found {stat.S_IMODE(st.st_mode):o})")
    if st.st_nlink != 1:
        raise EnrollmentError(f"identity file {path} must have exactly one link")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        private = private_key_from_pem(str(data["private_key_pem"]))
        return OperatorIdentity(
            enrollment_id=str(data["enrollment_id"]),
            private_key=private,
            public_key_pem=str(data["public_key_pem"]),
            created_at=str(data.get("created_at", "")),
            uid=int(data.get("uid", os.geteuid())),
        )
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        _quarantine(path, str(exc))
        raise  # unreachable — _quarantine raises; keeps type-checkers happy


def load_or_create_identity(home: Path) -> OperatorIdentity:
    try:
        return load_identity(home)
    except EnrollmentError as exc:
        if "corrupt" in str(exc):
            raise
        return enroll(home)


def read_public_key_pem(home: Path) -> str:
    """读 0644 公钥文件（daemon 管理员登记用）。"""
    path = home / PUBKEY_FILE
    if not path.exists():
        raise EnrollmentError(f"no operator pubkey at {path} — run `rosclaw operatord enroll` first")
    return path.read_text(encoding="utf-8")
