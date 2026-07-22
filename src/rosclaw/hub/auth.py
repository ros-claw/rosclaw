"""Authentication state management for ROSClaw Hub registries."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from rosclaw.firstboot.workspace import resolve_home
from rosclaw.hub.client import FakeRegistryClient
from rosclaw.hub.errors import HubError, HubErrorCode


@dataclass
class AuthProfile:
    """A single stored registry credential profile."""

    registry: str
    token: str
    insecure_local: bool = False


class AuthStore:
    """Persist hub login credentials in ``~/.rosclaw/config/hub_auth.json``.

    The JSON store is owner-only and written atomically. Production deployments
    may still prefer an OS keyring or external secret manager.
    """

    def __init__(self, home: str | Path | None = None) -> None:
        home_arg = str(home) if isinstance(home, Path) else home
        self.home = resolve_home(home_arg)
        self.path = self.home / "config" / "hub_auth.json"
        self._prepare_config_dir()
        self._data: dict[str, Any] = self._load()

    @staticmethod
    def _empty_data() -> dict[str, Any]:
        return {"profiles": {}, "active": None}

    def _storage_error(self, message: str) -> HubError:
        return HubError(
            code=HubErrorCode.AUTH_FAILED,
            message=f"Unsafe Hub credential store: {message}",
            suggested_fix=(
                f"Remove or repair {self.path} and ensure it is owned by the current user."
            ),
        )

    def _prepare_config_dir(self) -> None:
        config_dir = self.path.parent
        try:
            config_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            info = config_dir.lstat()
        except OSError as exc:
            raise self._storage_error(f"cannot prepare {config_dir}: {exc}") from exc

        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise self._storage_error(f"{config_dir} must be a real directory")
        self._require_current_owner(info, config_dir)
        self._chmod_owner_only(config_dir, 0o700)

    def _require_current_owner(self, info: os.stat_result, path: Path) -> None:
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise self._storage_error(f"{path} is owned by another user")

    def _chmod_owner_only(self, path: Path, mode: int) -> None:
        if os.name == "posix":
            try:
                path.chmod(mode, follow_symlinks=False)
            except OSError as exc:
                raise self._storage_error(f"cannot secure permissions on {path}: {exc}") from exc

    def _validate_existing_file(self) -> os.stat_result | None:
        try:
            info = self.path.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise self._storage_error(f"cannot inspect {self.path}: {exc}") from exc

        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise self._storage_error(f"{self.path} must be a regular file, not a link")
        self._require_current_owner(info, self.path)
        self._chmod_owner_only(self.path, 0o600)
        return info

    def _load(self) -> dict[str, Any]:
        if self._validate_existing_file() is None:
            return self._empty_data()

        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags)
            with os.fdopen(descriptor, encoding="utf-8") as auth_file:
                info = os.fstat(auth_file.fileno())
                if not stat.S_ISREG(info.st_mode):
                    raise self._storage_error(f"{self.path} is no longer a regular file")
                self._require_current_owner(info, self.path)
                data = json.load(auth_file)
        except HubError:
            raise
        except FileNotFoundError:
            return self._empty_data()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._empty_data()
        except OSError as exc:
            raise self._storage_error(f"cannot read {self.path}: {exc}") from exc

        if not isinstance(data, dict) or not isinstance(data.get("profiles"), dict):
            return self._empty_data()
        if data.get("active") is not None and not isinstance(data.get("active"), str):
            return self._empty_data()
        return data

    def _save(self) -> None:
        self._validate_existing_file()
        payload = (json.dumps(self._data, indent=2, ensure_ascii=False) + "\n").encode()
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
            with os.fdopen(descriptor, "wb") as auth_file:
                descriptor = -1
                auth_file.write(payload)
                auth_file.flush()
                os.fsync(auth_file.fileno())

            # Reject a link introduced after construction instead of silently
            # replacing it. os.replace itself never writes through the link.
            self._validate_existing_file()
            os.replace(temporary_path, self.path)
            temporary_path = None
            self._chmod_owner_only(self.path, 0o600)
            self._fsync_config_dir()
        except HubError:
            raise
        except OSError as exc:
            raise self._storage_error(f"cannot write {self.path}: {exc}") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary_path is not None:
                with suppress(OSError):
                    temporary_path.unlink(missing_ok=True)

    def _fsync_config_dir(self) -> None:
        if os.name != "posix":
            return
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        try:
            descriptor = os.open(self.path.parent, flags)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise self._storage_error(
                f"credentials were written but {self.path.parent} could not be synced: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Login / logout
    # ------------------------------------------------------------------
    def login(
        self,
        registry: str,
        token: str,
        *,
        insecure_local: bool = False,
        set_active: bool = True,
    ) -> None:
        """Store credentials for a registry."""
        registry = registry.rstrip("/")
        self._data["profiles"][registry] = {
            "token": token,
            "insecure_local": insecure_local,
        }
        if set_active or self._data.get("active") is None:
            self._data["active"] = registry
        self._save()

    def logout(self, registry: str | None = None) -> bool:
        """Remove stored credentials for a registry."""
        registry = registry or self._data.get("active")
        if not registry:
            return False
        registry = registry.rstrip("/")
        if registry not in self._data["profiles"]:
            return False
        del self._data["profiles"][registry]
        if self._data.get("active") == registry:
            remaining = list(self._data["profiles"].keys())
            self._data["active"] = remaining[0] if remaining else None
        self._save()
        return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_active_profile(self) -> dict[str, Any] | None:
        """Return the active registry profile, or ``None``."""
        active = self._data.get("active")
        if not active:
            return None
        profile = self._data["profiles"].get(active)
        if not profile:
            return None
        return {"registry": active, **profile}

    def get_token(self, registry: str | None = None) -> str | None:
        """Return the token for a registry, defaulting to the active one."""
        reg = registry or self._data.get("active")
        if not reg:
            return None
        return cast(
            str | None,
            self._data["profiles"].get(reg.rstrip("/"), {}).get("token"),
        )

    def is_insecure_local(self, registry: str | None = None) -> bool:
        """Return whether the registry was marked as an insecure local server."""
        reg = registry or self._data.get("active")
        if not reg:
            return False
        return cast(
            bool,
            self._data["profiles"].get(reg.rstrip("/"), {}).get("insecure_local", False),
        )

    def list_profiles(self) -> list[dict[str, Any]]:
        """Return all stored profiles."""
        return [{"registry": k, **v} for k, v in self._data["profiles"].items()]

    # ------------------------------------------------------------------
    # Client construction
    # ------------------------------------------------------------------
    def get_client(self, registry: str | None = None) -> FakeRegistryClient:
        """Build a :class:`FakeRegistryClient` for the active or given registry."""
        reg = registry or self._data.get("active")
        if not reg:
            raise HubError(
                code=HubErrorCode.AUTH_REQUIRED,
                message="No active registry. Run `rosclaw hub login` first.",
            )
        token = self.get_token(reg)
        return FakeRegistryClient(reg, token=token)
