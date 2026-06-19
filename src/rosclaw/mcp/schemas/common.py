"""Common MCP envelope and error helpers."""

from __future__ import annotations

import time
import uuid
from typing import Any

SCHEMA_VERSION = "p0.2025-06-19"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def make_response(data: Any, *, trace_id: str | None = None, runtime_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the standard P0 success envelope."""
    return {
        "ok": True,
        "schema_version": SCHEMA_VERSION,
        "trace_id": trace_id or str(uuid.uuid4()),
        "timestamp": _now_iso(),
        "runtime_profile": runtime_profile or {},
        "data": data,
    }


def make_error(
    code: str,
    message: str,
    *,
    trace_id: str | None = None,
    details: dict[str, Any] | None = None,
    runtime_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the standard P0 error envelope."""
    return {
        "ok": False,
        "schema_version": SCHEMA_VERSION,
        "trace_id": trace_id or str(uuid.uuid4()),
        "timestamp": _now_iso(),
        "runtime_profile": runtime_profile or {},
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
        },
    }


class MCPError(Exception):
    """Structured error that can be turned into a P0 error envelope."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def to_envelope(self, trace_id: str | None = None) -> dict[str, Any]:
        return make_error(self.code, self.message, trace_id=trace_id, details=self.details)
