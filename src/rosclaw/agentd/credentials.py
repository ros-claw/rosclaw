"""Owner-only persisted credentials for the AgentD CLI.

Public AgentD configuration keeps credential references only. This store is a
local CLI convenience for resolving those ``env:`` references across process
restarts without putting raw secrets in ``config.yaml`` or the repository.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from contextlib import suppress
from pathlib import Path

_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")


class CredentialStoreError(RuntimeError):
    """Raised when the local credential boundary is unsafe or unreadable."""


class AgentCredentialStore:
    """Atomic owner-only JSON store under ``ROSCLAW_HOME/agentd``."""

    def __init__(self, home: Path) -> None:
        self.path = home / "agentd" / "credentials.json"
        self._prepare_directory()
        self._values = self._load()

    def _error(self, message: str) -> CredentialStoreError:
        return CredentialStoreError(f"unsafe AgentD credential store: {message}")

    @staticmethod
    def _require_owner(info: os.stat_result, path: Path) -> None:
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise CredentialStoreError(
                f"unsafe AgentD credential store: {path} is owned by another user"
            )

    def _secure_mode(self, path: Path, mode: int) -> None:
        if os.name != "posix":
            return
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
            try:
                self._require_owner(os.fstat(descriptor), path)
                os.fchmod(descriptor, mode)
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise self._error(f"cannot secure permissions on {path}: {exc}") from exc

    def _prepare_directory(self) -> None:
        directory = self.path.parent
        try:
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            info = directory.lstat()
        except OSError as exc:
            raise self._error(f"cannot prepare {directory}: {exc}") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise self._error(f"{directory} must be a real directory")
        self._require_owner(info, directory)
        self._secure_mode(directory, 0o700)

    def _validate_file(self) -> bool:
        try:
            info = self.path.lstat()
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise self._error(f"cannot inspect {self.path}: {exc}") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise self._error(f"{self.path} must be a regular file, not a link")
        self._require_owner(info, self.path)
        self._secure_mode(self.path, 0o600)
        return True

    def _load(self) -> dict[str, str]:
        if not self._validate_file():
            return {}
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags)
            with os.fdopen(descriptor, encoding="utf-8") as credential_file:
                info = os.fstat(credential_file.fileno())
                if not stat.S_ISREG(info.st_mode):
                    raise self._error(f"{self.path} is no longer a regular file")
                self._require_owner(info, self.path)
                payload = json.load(credential_file)
        except CredentialStoreError:
            raise
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise self._error(f"{self.path} is not valid UTF-8 JSON") from exc
        except OSError as exc:
            raise self._error(f"cannot read {self.path}: {exc}") from exc
        values = payload.get("environment") if isinstance(payload, dict) else None
        if not isinstance(values, dict):
            raise self._error(f"{self.path} must contain an environment object")
        if not all(
            isinstance(name, str) and _ENV_NAME.fullmatch(name) and isinstance(value, str) and value
            for name, value in values.items()
        ):
            raise self._error(f"{self.path} contains an invalid credential entry")
        return dict(values)

    def _save(self) -> None:
        self._validate_file()
        payload = (json.dumps({"environment": self._values}, indent=2) + "\n").encode()
        descriptor = -1
        temporary_path: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
            )
            temporary_path = Path(temporary_name)
            if os.name == "posix":
                os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as credential_file:
                descriptor = -1
                credential_file.write(payload)
                credential_file.flush()
                os.fsync(credential_file.fileno())
            self._validate_file()
            os.replace(temporary_path, self.path)
            temporary_path = None
            self._secure_mode(self.path, 0o600)
        except CredentialStoreError:
            raise
        except OSError as exc:
            raise self._error(f"cannot write {self.path}: {exc}") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary_path is not None:
                with suppress(OSError):
                    temporary_path.unlink(missing_ok=True)

    def set(self, env_name: str, value: str) -> None:
        if not _ENV_NAME.fullmatch(env_name):
            raise ValueError(f"invalid environment credential name: {env_name!r}")
        if not value:
            raise ValueError("credential value must not be empty")
        self._values[env_name] = value
        self._save()

    def delete(self, env_name: str) -> bool:
        if env_name not in self._values:
            return False
        del self._values[env_name]
        self._save()
        return True

    def inject(self) -> tuple[str, ...]:
        """Load stored values without overriding an explicit process environment."""
        for name, value in self._values.items():
            os.environ.setdefault(name, value)
        return tuple(sorted(self._values))

    def status(self, env_name: str) -> dict[str, object]:
        value = self._values.get(env_name, "")
        return {
            "env_name": env_name,
            "stored": bool(value),
            "fingerprint": hashlib.sha256(value.encode()).hexdigest()[:8] if value else None,
            "path": str(self.path),
        }
