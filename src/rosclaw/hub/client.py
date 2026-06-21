"""Hub registry clients for ROSClaw asset discovery and download."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import urllib.error
import urllib.request
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast

import yaml

from rosclaw.hub._compat import extractall_tar
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

    @abstractmethod
    def fetch_bundle(self, ref: AssetRef) -> bytes:
        """Fetch the ``.rosclaw`` bundle bytes for ``ref``.

        Args:
            ref: Concrete asset reference including version.

        Returns:
            Raw ``.rosclaw`` tar.gz bytes.
        """
        raise NotImplementedError

    @abstractmethod
    def publish_bundle(
        self,
        bundle_path: str | Path,
        manifest: Any,
        size_bytes: int,
    ) -> dict[str, Any]:
        """Upload a ``.rosclaw`` bundle to the registry.

        Args:
            bundle_path: Path to the prepared ``.rosclaw`` tar.gz file.
            manifest: The asset manifest associated with the bundle.
            size_bytes: Bundle size in bytes.

        Returns:
            Upload metadata, including ``manifest_url`` when available.
        """
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

    def fetch_bundle(self, ref: AssetRef) -> bytes:
        """Fetch the ``.rosclaw`` bundle bytes for ``ref``."""
        if not ref.version:
            raise HubError(
                code=HubErrorCode.MANIFEST_INVALID,
                message="Cannot fetch bundle without a version",
            )
        path = f"bundles/{ref.type}/{ref.namespace}/{ref.name}/{ref.version}.rosclaw"
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

    def publish_bundle(
        self,
        bundle_path: str | Path,
        manifest: Any,
        size_bytes: int,
    ) -> dict[str, Any]:
        """Upload a ``.rosclaw`` bundle to the fake registry.

        For local registries the bundle is extracted and the manifest/blobs are
        written into the registry tree.  For HTTP registries the bundle is
        posted to the registry's upload endpoint.
        """
        if self._local_root is not None:
            return self._publish_bundle_local(bundle_path, manifest, size_bytes)
        return self._publish_bundle_http(bundle_path, manifest, size_bytes)

    # ------------------------------------------------------------------
    # Publishing helpers
    # ------------------------------------------------------------------
    def _manifest_ref_path(self, manifest: Any) -> str:
        asset = manifest.asset
        return f"manifests/{asset.type.value}/{asset.namespace}/{asset.name}/{asset.version}.yaml"

    def _bundle_ref_path(self, manifest: Any) -> str:
        asset = manifest.asset
        return f"bundles/{asset.type.value}/{asset.namespace}/{asset.name}/{asset.version}.rosclaw"

    @staticmethod
    def _sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _catalog_entry(
        self,
        manifest: Any,
        manifest_digest: str,
        manifest_url: str,
        size_bytes: int,
    ) -> dict[str, Any]:
        """Build a catalog entry matching the publisher's output."""
        data = cast(dict[str, Any], manifest.model_dump(mode="json"))
        data["manifest_digest"] = manifest_digest
        data["manifest_url"] = manifest_url
        data["size_bytes"] = size_bytes
        return data

    def _publish_bundle_local(
        self,
        bundle_path: str | Path,
        manifest: Any,
        size_bytes: int,
    ) -> dict[str, Any]:
        local_root = self._local_root
        if local_root is None:
            raise HubError(
                code=HubErrorCode.REGISTRY_UNREACHABLE,
                message="Local registry root is not configured",
            )

        bundle_path = Path(bundle_path)
        if not bundle_path.exists():
            raise HubError(
                code=HubErrorCode.ASSET_NOT_FOUND,
                message=f"Bundle file not found: {bundle_path}",
            )

        extract_dir = local_root / ".tmp" / uuid.uuid4().hex
        try:
            with tarfile.open(bundle_path, "r:gz") as tar:
                extractall_tar(tar, extract_dir)

            manifest_rel = self._manifest_ref_path(manifest)
            manifest_dest = local_root / manifest_rel
            extracted_manifest = extract_dir / "manifest.yaml"
            if extracted_manifest.exists():
                manifest_bytes = extracted_manifest.read_bytes()
            else:
                manifest_bytes = yaml.safe_dump(
                    manifest.model_dump(mode="json"),
                    sort_keys=False,
                    allow_unicode=True,
                ).encode("utf-8")
            manifest_dest.parent.mkdir(parents=True, exist_ok=True)
            manifest_dest.write_bytes(manifest_bytes)
            manifest_digest = f"sha256:{self._sha256(manifest_bytes)}"

            # Content-address all extracted files as blobs.
            blobs_dir = local_root / "blobs" / "sha256"
            blobs_dir.mkdir(parents=True, exist_ok=True)
            for path in sorted(extract_dir.rglob("*")):
                if not path.is_file():
                    continue
                data = path.read_bytes()
                blob_name = self._sha256(data)
                blob_path = blobs_dir / blob_name
                if not blob_path.exists():
                    blob_path.write_bytes(data)

            catalog_path = local_root / "catalog.jsonl"
            entry = self._catalog_entry(
                manifest,
                manifest_digest,
                manifest_url=manifest_rel,
                size_bytes=size_bytes,
            )
            with catalog_path.open("a", encoding="utf-8") as catalog_file:
                catalog_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

            # Keep the original bundle for install-by-reference.
            bundle_rel = self._bundle_ref_path(manifest)
            bundle_dest = local_root / bundle_rel
            bundle_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(bundle_path, bundle_dest)
        finally:
            if extract_dir.exists():
                shutil.rmtree(extract_dir)

        return {
            "registry": str(local_root),
            "manifest_url": manifest_rel,
            "bundle_url": bundle_rel,
            "manifest_digest": manifest_digest,
            "size_bytes": size_bytes,
        }

    def _publish_bundle_http(
        self,
        bundle_path: str | Path,
        manifest: Any,
        size_bytes: int,
    ) -> dict[str, Any]:
        bundle_path = Path(bundle_path)
        data = bundle_path.read_bytes()
        asset = manifest.asset
        url_path = (
            f"upload/{asset.type.value}/{asset.namespace}/{asset.name}/{asset.version}.rosclaw"
        )
        return self._post_http(url_path, data, content_type="application/gzip")

    def _post_http(
        self,
        relative_path: str,
        data: bytes,
        content_type: str,
    ) -> dict[str, Any]:
        url = f"{self.registry_url}/{relative_path}"
        headers = self._headers()
        headers["Content-Type"] = content_type
        request = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read()
                if not body:
                    return {"manifest_url": url}
                return cast(dict[str, Any], json.loads(body.decode("utf-8")))
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                code = HubErrorCode.AUTH_FAILED
            elif exc.code in (403, 409):
                code = HubErrorCode.PUBLISH_REJECTED
            else:
                code = HubErrorCode.REGISTRY_UNREACHABLE
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
