"""Persistent operator enrollment registry（二次复核 R2/P0-5）。

替代初版的内存 dict + "空表首调放行"bootstrap：

* registry 持久化在 daemon state 目录（0600、原子写、文件+目录 fsync），
  daemon 重启不丢失、不重新打开抢注窗口；
* 空 registry = **全部拒绝**——只有 daemon 服务 UID（管理员）能
  register/revoke（`rosclaw operatord register-daemon` 以 daemon
  管理员身份执行）；
* 只存公钥 + operator UID + 用途 + 状态，不存任何对称秘密；
* 已焚毁的 challenge nonce 一并持久化（防跨重启重放）。
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from rosclaw.contracts.operator.decision import key_fingerprint, public_key_from_pem

REGISTRY_FILE = "operator-enrollments.json"


class RegistryError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class EnrollmentRecord:
    enrollment_id: str
    public_key_pem: str
    operator_uid: int
    purpose: str
    created_at: str
    status: str = "active"  # active | revoked
    version: int = 1
    revoked_at: str = ""

    @property
    def fingerprint(self) -> str:
        return key_fingerprint(self.public_key_pem)

    def public_dict(self) -> dict:
        data = asdict(self)
        data.pop("public_key_pem")
        data["fingerprint"] = self.fingerprint
        return data


class OperatorRegistry:
    """持久化 enrollment registry。``path=None`` 时纯内存（仅测试）。"""

    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._records: dict[str, EnrollmentRecord] = {}
        self._burned_nonces: set[str] = set()
        if path is not None and path.exists():
            self._load(path)

    def _load(self, path: Path) -> None:
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            raise RegistryError(
                "REGISTRY_PERMISSIONS", f"registry {path} must be 0600 (found {mode:o})"
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for raw in data.get("enrollments", []):
                record = EnrollmentRecord(**raw)
                public_key_from_pem(record.public_key_pem)  # validate
                self._records[record.enrollment_id] = record
            self._burned_nonces = set(data.get("burned_nonces", []))
        except (json.JSONDecodeError, TypeError, KeyError, ValueError) as exc:
            raise RegistryError("REGISTRY_CORRUPT", f"registry corrupt: {exc}") from exc

    def _save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self._path.parent, 0o700)
        payload = json.dumps(
            {
                "enrollments": [asdict(r) for r in self._records.values()],
                "burned_nonces": sorted(self._burned_nonces),
            },
            indent=2,
        )
        tmp = self._path.with_name(self._path.name + ".tmp")
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(tmp, flags, 0o600)
        try:
            os.fchmod(fd, 0o600)
            os.write(fd, payload.encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        if os.path.islink(self._path):
            raise RegistryError("REGISTRY_SYMLINK", f"refusing to replace symlink {self._path}")
        os.rename(tmp, self._path)
        dir_fd = os.open(self._path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    # -- mutations（调用方负责 ACL：仅 daemon 服务 UID） -----------------------

    def register(
        self,
        enrollment_id: str,
        *,
        public_key_pem: str,
        operator_uid: int,
        purpose: str = "operator-decision",
    ) -> EnrollmentRecord:
        public_key_from_pem(public_key_pem)  # 必须是合法 Ed25519 公钥
        existing = self._records.get(enrollment_id)
        if existing is not None:
            if existing.public_key_pem != public_key_pem:
                raise RegistryError(
                    "ENROLLMENT_KEY_CONFLICT",
                    f"enrollment {enrollment_id} already registered with a different key; "
                    "revoke it first (no silent rotation)",
                )
            if existing.status == "revoked":
                raise RegistryError(
                    "ENROLLMENT_REVOKED",
                    f"enrollment {enrollment_id} was revoked; use a new enrollment id",
                )
            return existing
        record = EnrollmentRecord(
            enrollment_id=enrollment_id,
            public_key_pem=public_key_pem,
            operator_uid=operator_uid,
            purpose=purpose,
            created_at=datetime.now(UTC).isoformat(),
        )
        self._records[enrollment_id] = record
        self._save()
        return record

    def revoke(self, enrollment_id: str) -> EnrollmentRecord:
        record = self._records.get(enrollment_id)
        if record is None:
            raise RegistryError("ENROLLMENT_UNKNOWN", f"unknown enrollment {enrollment_id}")
        revoked = EnrollmentRecord(
            **{**asdict(record), "status": "revoked", "revoked_at": datetime.now(UTC).isoformat()}
        )
        self._records[enrollment_id] = revoked
        self._save()
        return revoked

    # -- queries --------------------------------------------------------------

    def get(self, enrollment_id: str) -> EnrollmentRecord | None:
        return self._records.get(enrollment_id)

    def active(self, enrollment_id: str) -> EnrollmentRecord | None:
        record = self._records.get(enrollment_id)
        if record is None or record.status != "active":
            return None
        return record

    def list(self) -> list[EnrollmentRecord]:
        return list(self._records.values())

    def active_operator_uids(self) -> set[int]:
        return {r.operator_uid for r in self._records.values() if r.status == "active"}

    # -- nonce 焚毁（持久化，跨重启防重放） ------------------------------------

    def burn_nonce(self, nonce: str) -> None:
        if nonce in self._burned_nonces:
            raise RegistryError("NONCE_REPLAY", "challenge nonce already used")
        self._burned_nonces.add(nonce)
        self._save()
