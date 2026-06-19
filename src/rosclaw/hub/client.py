"""Hub registry clients for ROSClaw asset discovery and download."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast

from rosclaw.hub.errors import HubError, HubErrorCode
from rosclaw.hub.refs import AssetRef


class HubClient(ABC):
    """Abstract client for a ROSClaw Hub registry."""

    @abstractmethod
    def sync(self) -> list[dict[str, Any]]:
        """Fetch the latest catalog entries from the registry."""
        raise NotImplementedError

    @abstractmethod
    def fetch_manifest(self, ref: AssetRef) -> bytes:
        """Fetch the raw manifest bytes for ``ref``."""
        raise NotImplementedError

    @abstractmethod
    def fetch_blob(self, digest: str) -> bytes:
        """Fetch blob bytes for a content digest."""
        raise NotImplementedError

    @abstractmethod
    def whoami(self) -> dict[str, Any]:
        """Return identity information for the current credentials."""
        raise NotImplementedError


class FakeRegistryClient(HubClient):
    """Client for the offline fake registry used in tests and demos.

    Supports both local directory paths and ``http(s)://`` URLs. The fake
    registry performs no real cryptographic verification; it exists to close
    the E2E testing loop without cloud dependencies.
    """

    # Token hard-coded by the fake fixture server.
    EXPECTED_TOKEN = "fake-valid-token"

    def __init__(
        self,
        registry_url: str,
        token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.registry_url = registry_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._local_root: Path | None = None
        if not self.registry_url.startswith(("http://", "https://")):
            local = self.registry_url
            if local.startswith("file://"):
                local = local[len("file://") :]
            self._local_root = Path(local).expanduser().resolve()

    # ------------------------------------------------------------------
    # HubClient implementation
    # ------------------------------------------------------------------
    def sync(self) -> list[dict[str, Any]]:
        """Download and parse ``catalog.jsonl``."""
        data = self._get("catalog.jsonl")
        entries: list[dict[str, Any]] = []
        for line_number, line in enumerate(data.decode("utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise HubError(
                    code=HubErrorCode.INDEX_VERIFY_FAILED,
                    message=f"Invalid catalog JSONL at line {line_number}: {exc}",
                    suggested_fix="Re-sync the catalog or contact the registry operator.",
                ) from exc
        return entries

    def fetch_manifest(self, ref: AssetRef) -> bytes:
        """Fetch the YAML manifest for ``ref``."""
        if not ref.version:
            raise HubError(
                code=HubErrorCode.MANIFEST_INVALID,
                message="Cannot fetch manifest without a version",
            )
        path = f"manifests/{ref.type}/{ref.namespace}/{ref.name}/{ref.version}.yaml"
        return self._get(path)

    def fetch_blob(self, digest: str) -> bytes:
        """Fetch a content-addressed blob."""
        if ":" not in digest:
            raise HubError(
                code=HubErrorCode.CHECKSUM_MISMATCH,
                message=f"Invalid digest format: {digest!r}",
            )
        algorithm, hexdigest = digest.split(":", 1)
        path = f"blobs/{algorithm}/{hexdigest}"
        return self._get(path)

    def whoami(self) -> dict[str, Any]:
        """Return a fake profile if the token is the expected fixture token."""
        if self.token != self.EXPECTED_TOKEN:
            raise HubError(
                code=HubErrorCode.AUTH_FAILED,
                message="Invalid or expired token",
                suggested_fix="Run `rosclaw hub login` with a valid registry token.",
            )
        return {
            "registry": self.registry_url,
            "user": "rosclaw-tester",
            "role": "admin",
            "email": "tester@rosclaw.local",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get(self, relative_path: str) -> bytes:
        if self._local_root is not None:
            return self._get_local(relative_path)
        return self._get_http(relative_path)

    def _get_local(self, relative_path: str) -> bytes:
        path = self._local_root / relative_path  # type: ignore[operator]
        if not path.exists():
            raise HubError(
                code=HubErrorCode.ASSET_NOT_FOUND,
                message=f"Registry file not found: {relative_path}",
            )
        return path.read_bytes()

    def _get_http(self, relative_path: str) -> bytes:
        url = f"{self.registry_url}/{relative_path}"
        request = urllib.request.Request(url, headers=self._headers())
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return cast(bytes, response.read())
        except urllib.error.HTTPError as exc:
            code = (
                HubErrorCode.ASSET_NOT_FOUND
                if exc.code == 404
                else HubErrorCode.REGISTRY_UNREACHABLE
            )
            raise HubError(
                code=code,
                message=f"Registry returned {exc.code}: {exc.reason}",
            ) from exc
        except urllib.error.URLError as exc:
            raise HubError(
                code=HubErrorCode.REGISTRY_UNREACHABLE,
                message=f"Cannot reach registry: {exc.reason}",
                suggested_fix="Check the registry URL and network connectivity.",
            ) from exc

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers
