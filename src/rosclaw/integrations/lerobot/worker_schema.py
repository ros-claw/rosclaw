"""Schemas for the LeRobot subprocess worker.

These dataclasses describe the JSON request/response protocol between ROSClaw
and the LeRobot worker process. They are kept free of heavy imports so they can
be used by both the ROSClaw core and the worker itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

WORKER_SCHEMA_VERSION = "rosclaw.lerobot.worker.v1"
WorkerOp = Literal["inspect", "load_test", "infer"]


@dataclass
class WorkerObservation:
    """Observation passed to the worker for inference."""

    task: str = ""
    state: list[float] = field(default_factory=list)
    images: dict[str, str] = field(default_factory=dict)

    def to_worker_dict(self) -> dict[str, Any]:
        """Flatten into LeRobot-style keys."""
        out: dict[str, Any] = {}
        if self.task:
            out["task"] = self.task
        if self.state:
            out["observation.state"] = self.state
        for name, path in self.images.items():
            out[f"observation.images.{name}"] = path
        return out

    @classmethod
    def from_worker_dict(cls, data: dict[str, Any]) -> WorkerObservation:
        """Build from a LeRobot-style flat dict."""
        task = data.get("task", "")
        state = data.get("observation.state", [])
        images: dict[str, str] = {}
        for key, value in data.items():
            if key.startswith("observation.images."):
                images[key.split(".", 2)[2]] = str(value)
        return cls(task=task or "", state=state or [], images=images)


@dataclass
class WorkerRequest:
    """Request sent to the LeRobot worker."""

    op: WorkerOp
    policy_path: str
    revision: str = "main"
    device: str = "cpu"
    dtype: str = "auto"
    allow_network: bool = False
    timeout_sec: int = 120
    observation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": WORKER_SCHEMA_VERSION,
            "op": self.op,
            "policy_path": self.policy_path,
            "revision": self.revision,
            "device": self.device,
            "dtype": self.dtype,
            "allow_network": self.allow_network,
            "timeout_sec": self.timeout_sec,
            "observation": self.observation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkerRequest:
        return cls(
            op=data.get("op", "inspect"),  # type: ignore[arg-type]
            policy_path=data.get("policy_path", ""),
            revision=data.get("revision", "main"),
            device=data.get("device", "cpu"),
            dtype=data.get("dtype", "auto"),
            allow_network=bool(data.get("allow_network", False)),
            timeout_sec=int(data.get("timeout_sec", 120)),
            observation=data.get("observation", {}),
        )


@dataclass
class WorkerAction:
    """Action returned by the worker."""

    type: str = "raw_lerobot_action"
    values: list[float] = field(default_factory=list)
    shape: list[int] = field(default_factory=list)
    dtype: str = "float32"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "values": self.values,
            "shape": self.shape,
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkerAction:
        return cls(
            type=data.get("type", "raw_lerobot_action"),
            values=list(data.get("values", [])),
            shape=list(data.get("shape", [])),
            dtype=data.get("dtype", "float32"),
        )


@dataclass
class WorkerError:
    """Structured error returned by the worker."""

    code: str
    message: str
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkerError:
        return cls(
            code=data.get("code", "unknown"),
            message=data.get("message", ""),
            details=data.get("details", ""),
        )


@dataclass
class WorkerTiming:
    """Timing metadata from the worker."""

    load_time_sec: float | None = None
    infer_time_sec: float | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.load_time_sec is not None:
            out["load_time_sec"] = self.load_time_sec
        if self.infer_time_sec is not None:
            out["infer_time_sec"] = self.infer_time_sec
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkerTiming:
        return cls(
            load_time_sec=data.get("load_time_sec"),
            infer_time_sec=data.get("infer_time_sec"),
        )


@dataclass
class WorkerResponse:
    """Response returned by the LeRobot worker."""

    schema_version: str = WORKER_SCHEMA_VERSION
    status: Literal["ok", "error"] = "ok"
    op: WorkerOp = "inspect"  # type: ignore[assignment]
    policy_path: str = ""
    real_model_loaded: bool = False
    real_inference: bool = False
    policy_metadata: dict[str, Any] = field(default_factory=dict)
    action: WorkerAction | None = None
    timing: WorkerTiming = field(default_factory=WorkerTiming)
    error: WorkerError | None = None
    runtime: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "status": self.status,
            "op": self.op,
            "policy_path": self.policy_path,
            "real_model_loaded": self.real_model_loaded,
            "real_inference": self.real_inference,
            "policy_metadata": self.policy_metadata,
            "timing": self.timing.to_dict(),
            "runtime": self.runtime,
        }
        if self.action is not None:
            out["action"] = self.action.to_dict()
        if self.error is not None:
            out["error"] = self.error.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkerResponse:
        action_data = data.get("action")
        error_data = data.get("error")
        return cls(
            schema_version=data.get("schema_version", WORKER_SCHEMA_VERSION),
            status=data.get("status", "ok"),  # type: ignore[arg-type]
            op=data.get("op", "inspect"),  # type: ignore[arg-type]
            policy_path=data.get("policy_path", ""),
            real_model_loaded=bool(data.get("real_model_loaded", False)),
            real_inference=bool(data.get("real_inference", False)),
            policy_metadata=data.get("policy_metadata", {}),
            action=WorkerAction.from_dict(action_data) if action_data else None,
            timing=WorkerTiming.from_dict(data.get("timing", {})),
            error=WorkerError.from_dict(error_data) if error_data else None,
            runtime=data.get("runtime", {}),
        )

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def error_code(self) -> str:
        if self.error is not None:
            return self.error.code
        return "unknown"

    def error_message(self) -> str:
        if self.error is not None:
            return self.error.message
        return ""
