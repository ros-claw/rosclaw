"""rosclawd 自身的 Ed25519 签名身份（二次复核 R1/DecisionReceiptV1）。

daemon 用此 key 签 DecisionReceiptV1；agentd 用公钥验证 receipt。
公钥经 socket `daemon.identity` 公开（公钥无需保密，信任锚是本机
socket 的 UID/组隔离）；私钥 0600 存 daemon state 目录。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from rosclaw.contracts.operator.decision import (
    generate_ed25519_keypair,
    key_fingerprint,
    private_key_from_pem,
    private_key_to_pem,
    sign_b64,
)

IDENTITY_FILE = "daemon-identity.json"


@dataclass(frozen=True)
class DaemonIdentity:
    private_key: Ed25519PrivateKey
    public_key_pem: str

    @property
    def key_id(self) -> str:
        return key_fingerprint(self.public_key_pem)

    def sign(self, payload: bytes) -> str:
        return sign_b64(self.private_key, payload)

    @classmethod
    def load_or_create(cls, state_dir: Path | None) -> DaemonIdentity:
        """持久化加载；``state_dir=None`` 时临时生成（仅测试）。"""
        if state_dir is None:
            private, pem = generate_ed25519_keypair()
            return cls(private_key=private, public_key_pem=pem)
        state_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(state_dir, 0o700)
        path = state_dir / IDENTITY_FILE
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                private_key=private_key_from_pem(str(data["private_key_pem"])),
                public_key_pem=str(data["public_key_pem"]),
            )
        private, pem = generate_ed25519_keypair()
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags, 0o600)
        try:
            os.fchmod(fd, 0o600)
            os.write(
                fd,
                json.dumps(
                    {"private_key_pem": private_key_to_pem(private), "public_key_pem": pem}
                ).encode(),
            )
            os.fsync(fd)
        finally:
            os.close(fd)
        return cls(private_key=private, public_key_pem=pem)
