"""Fake ROSClaw Hub registry server for offline E2E tests.

This is a minimal static-file HTTP server with token authentication. It serves
the same layout used by :class:`rosclaw.hub.client.FakeRegistryClient`:

    /
    ├── catalog.jsonl
    ├── root.json
    ├── timestamp.json
    ├── snapshot.json
    ├── manifests/<type>/<namespace>/<name>/<version>.yaml
    ├── bundles/<type>/<namespace>/<name>/<version>.rosclaw
    └── blobs/<algorithm>/<hexdigest>

Run with:

    python -m tests.fixtures.fake_registry.server --port 8787
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import tarfile
import tempfile
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import yaml

DEFAULT_TOKEN = "fake-valid-token"


def _extractall_tar(tar: tarfile.TarFile, path: Path) -> None:
    """Extract a tar archive safely with Python-version-aware filtering."""
    if sys.version_info >= (3, 12):
        tar.extractall(path=path, filter="data")
    else:
        tar.extractall(path=path)


class _AuthHandler(SimpleHTTPRequestHandler):
    """Static handler that requires ``Authorization: Bearer <token>``.

    Uploads of ``.rosclaw`` bundles to ``/upload/...`` are unpacked, indexed,
    and made available for install-by-reference just like a local registry
    publish.
    """

    def __init__(self, token: str, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.token = token
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        if not self._authorize():
            return
        super().do_GET()

    def do_HEAD(self) -> None:  # noqa: N802
        if not self._authorize():
            return
        super().do_HEAD()

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorize():
            return
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Missing Content-Length")
            return
        try:
            length = int(content_length)
        except ValueError:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Invalid Content-Length")
            return

        body = self.rfile.read(length)
        target = Path(self.translate_path(self.path)).resolve()
        registry_root = Path(self.directory).resolve()
        try:
            target.relative_to(registry_root)
        except ValueError:
            self.send_response(403)
            self.end_headers()
            self.wfile.write(b"Forbidden")
            return

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)

        if self.path.startswith("/upload/") and self.path.endswith(".rosclaw"):
            try:
                response = self._ingest_bundle(body, registry_root, self.path)
            except Exception as exc:  # noqa: BLE001
                self.send_response(400)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(f"Bundle ingestion failed: {exc}".encode())
                return
        else:
            rel = target.relative_to(registry_root).as_posix()
            response = {"manifest_url": rel, "size_bytes": len(body)}

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))

    def log_message(self, fmt: str, *args: object) -> None:
        # Keep test output quiet.
        pass

    def _authorize(self) -> bool:
        header = self.headers.get("Authorization", "")
        expected = f"Bearer {self.token}"
        if header != expected:
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Bearer realm="rosclaw-hub-fake"')
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Unauthorized. Use --token or login with a valid token.\n")
            return False
        return True

    def _ingest_bundle(
        self,
        bundle_bytes: bytes,
        registry_root: Path,
        upload_path: str,
    ) -> dict[str, object]:
        """Extract a bundle, store blobs, and append a catalog entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            extract_dir = Path(tmpdir)
            with tarfile.open(fileobj=io.BytesIO(bundle_bytes), mode="r:gz") as tar:
                _extractall_tar(tar, extract_dir)

            manifest_path = extract_dir / "manifest.yaml"
            if not manifest_path.exists():
                raise ValueError("Bundle is missing manifest.yaml") from None
            manifest_bytes = manifest_path.read_bytes()
            manifest = yaml.safe_load(manifest_bytes)

        asset = manifest.get("asset", {})
        asset_type = asset.get("type")
        namespace = asset.get("namespace")
        name = asset.get("name")
        version = asset.get("version")
        if not all([asset_type, namespace, name, version]):
            raise ValueError("manifest.yaml is missing asset identity fields")

        manifest_rel = f"manifests/{asset_type}/{namespace}/{name}/{version}.yaml"
        manifest_dest = registry_root / manifest_rel
        manifest_dest.parent.mkdir(parents=True, exist_ok=True)
        manifest_dest.write_bytes(manifest_bytes)
        manifest_digest = f"sha256:{hashlib.sha256(manifest_bytes).hexdigest()}"

        # Content-address all extracted files as blobs.
        blobs_dir = registry_root / "blobs" / "sha256"
        blobs_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(extract_dir.rglob("*")):
            if not path.is_file():
                continue
            data = path.read_bytes()
            blob_name = hashlib.sha256(data).hexdigest()
            blob_path = blobs_dir / blob_name
            if not blob_path.exists():
                blob_path.write_bytes(data)

        # Keep the bundle so install-by-reference can fetch it.
        bundle_rel = f"bundles/{asset_type}/{namespace}/{name}/{version}.rosclaw"
        bundle_dest = registry_root / bundle_rel
        bundle_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            registry_root / Path(upload_path.lstrip("/")),
            bundle_dest,
        )

        # Append catalog entry.
        catalog_path = registry_root / "catalog.jsonl"
        entry = dict(manifest)
        entry["schema_version"] = "hub.catalog.v1"
        entry["ref"] = f"rosclaw://{asset_type}/{namespace}/{name}@{version}"
        entry["manifest_digest"] = manifest_digest
        entry["manifest_url"] = manifest_rel
        entry["size_bytes"] = len(bundle_bytes)
        with catalog_path.open("a", encoding="utf-8") as catalog_file:
            catalog_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return {
            "manifest_url": manifest_rel,
            "bundle_url": bundle_rel,
            "manifest_digest": manifest_digest,
            "size_bytes": len(bundle_bytes),
        }


def main(argv: list[str] | None = None) -> int:
    """Start the fake registry server."""
    parser = argparse.ArgumentParser(description="Fake ROSClaw Hub registry")
    parser.add_argument(
        "--port",
        type=int,
        default=8787,
        help="Port to listen on (default: 8787)",
    )
    parser.add_argument(
        "--directory",
        type=str,
        default=str(Path(__file__).parent),
        help="Root directory to serve (default: this fixture directory)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=DEFAULT_TOKEN,
        help=f"Bearer token required for requests (default: {DEFAULT_TOKEN})",
    )
    args = parser.parse_args(argv)

    directory = Path(args.directory).resolve()
    handler = partial(_AuthHandler, args.token, directory=str(directory))
    server = HTTPServer(("", args.port), handler)
    print(f"Fake ROSClaw Hub registry serving {directory} on port {args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
