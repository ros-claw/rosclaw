"""Fake ROSClaw Hub registry server for offline E2E tests.

This is a minimal static-file HTTP server with token authentication. It serves
the same layout used by :class:`rosclaw.hub.client.FakeRegistryClient`:

    /
    ├── catalog.jsonl
    ├── root.json
    ├── timestamp.json
    ├── snapshot.json
    ├── manifests/<type>/<namespace>/<name>/<version>.yaml
    └── blobs/<algorithm>/<hexdigest>

Run with:

    python -m tests.fixtures.fake_registry.server --port 8787
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

DEFAULT_TOKEN = "fake-valid-token"


class _AuthHandler(SimpleHTTPRequestHandler):
    """Static handler that requires ``Authorization: Bearer <token>``."""

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
