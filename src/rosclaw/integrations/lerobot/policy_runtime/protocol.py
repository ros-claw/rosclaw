"""JSONL request/response protocol for the persistent LeRobot policy runtime.

Protocol: ``rosclaw.lerobot.policy_runtime.v1``

Each message is a single line of JSON terminated by ``\\n``.  Every request
carries a unique ``id``; the response echoes it.  Methods are intentionally
simple JSON-RPC-like verbs so both sides can be implemented without heavy
framework dependencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal


RUNTIME_PROTOCOL_VERSION = "rosclaw.lerobot.policy_runtime.v1"

# Methods supported by the worker service.
RUNTIME_METHODS = (
    "HELLO",
    "PROBE",
    "LOAD_POLICY",
    "WARMUP",
    "CREATE_SESSION",
    "RESET_SESSION",
    "INFER",
    "HEALTH",
    "CLOSE_SESSION",
    "UNLOAD_POLICY",
    "SHUTDOWN",
)

Method = Literal[
    "HELLO",
    "PROBE",
    "LOAD_POLICY",
    "WARMUP",
    "CREATE_SESSION",
    "RESET_SESSION",
    "INFER",
    "HEALTH",
    "CLOSE_SESSION",
    "UNLOAD_POLICY",
    "SHUTDOWN",
]


@dataclass
class RuntimeRequest:
    """A request sent from the ROSClaw client to the LeRobot worker."""

    method: Method
    params: dict[str, Any] = field(default_factory=dict)
    id: str = "0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "method": self.method,
            "params": self.params,
            "id": self.id,
        }

    def to_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False) + "\n"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeRequest":
        return cls(
            method=data.get("method", "PROBE"),  # type: ignore[arg-type]
            params=dict(data.get("params", {})),
            id=str(data.get("id", "0")),
        )


@dataclass
class RuntimeResponse:
    """A response sent from the LeRobot worker back to ROSClaw."""

    id: str
    result: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": self.id,
            "result": self.result,
        }
        if self.error is not None:
            out["error"] = self.error
        return out

    def to_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False) + "\n"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeResponse":
        return cls(
            id=str(data.get("id", "0")),
            result=dict(data.get("result", {})),
            error=data.get("error"),
        )


def encode_request(method: Method, params: dict[str, Any], request_id: str) -> str:
    """Encode a method call into a JSONL line."""
    return RuntimeRequest(method=method, params=params, id=request_id).to_line()


def encode_response(
    request_id: str,
    result: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> str:
    """Encode a result/error into a JSONL line."""
    return RuntimeResponse(
        id=request_id,
        result=result or {},
        error=error,
    ).to_line()


def parse_line(line: str) -> RuntimeRequest | RuntimeResponse | None:
    """Parse a single JSONL line into a request or response object."""
    line = line.strip()
    if not line:
        return None
    data = json.loads(line)
    if "method" in data:
        return RuntimeRequest.from_dict(data)
    return RuntimeResponse.from_dict(data)
