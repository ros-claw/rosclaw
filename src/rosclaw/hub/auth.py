"""Authentication state management for ROSClaw Hub registries."""

from __future__ import annotations

import json
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

    In production this should be backed by the OS keyring; the JSON fallback
    exists so the fake-registry testing loop works without extra dependencies.
    """

    def __init__(self, home: str | Path | None = None) -> None:
        home_arg = str(home) if isinstance(home, Path) else home
        self.home = resolve_home(home_arg)
        self.path = self.home / "config" / "hub_auth.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except (OSError, json.JSONDecodeError):
                pass
        return {"profiles": {}, "active": None}

    def _save(self) -> None:
        self.path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

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
            self._data["profiles"]
            .get(reg.rstrip("/"), {})
            .get("insecure_local", False),
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
