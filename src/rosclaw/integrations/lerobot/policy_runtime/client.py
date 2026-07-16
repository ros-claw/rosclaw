"""High-level client for the persistent LeRobot policy runtime.

This module wraps the lower-level manager with typed method calls and session
helpers.  It remains free of torch/lerobot imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rosclaw.integrations.lerobot.policy_runtime.manager import PersistentRuntimeManager
from rosclaw.integrations.lerobot.policy_runtime.protocol import RUNTIME_PROTOCOL_VERSION
from rosclaw.integrations.lerobot.policy_runtime.session import PolicySession
from rosclaw.integrations.lerobot.policy_runtime.state import RuntimeState


class RuntimeClient:
    """Convenience client for the persistent policy runtime."""

    def __init__(self, manager: PersistentRuntimeManager):
        self.manager = manager

    @property
    def state(self) -> RuntimeState:
        return self.manager.state

    def start(self) -> RuntimeState:
        return self.manager.start()

    def stop(self) -> None:
        self.manager.stop()

    def hello(self) -> dict[str, Any]:
        return self.manager.call(
            "HELLO",
            {
                "protocol_version": RUNTIME_PROTOCOL_VERSION,
            },
        )

    def probe(self) -> dict[str, Any]:
        return self.manager.call("PROBE", {})

    def load_policy(
        self,
        policy_path: str | Path,
        *,
        revision: str = "main",
        device: str | None = None,
        allow_network: bool | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "policy_path": str(policy_path),
            "revision": revision,
        }
        if device is not None:
            params["device"] = device
        if allow_network is not None:
            params["allow_network"] = allow_network
        return self.manager.call("LOAD_POLICY", params)

    def warmup(
        self,
        observation: dict[str, Any] | None = None,
        *,
        iterations: int = 1,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"iterations": iterations}
        if observation is not None:
            params["observation"] = observation
        return self.manager.call("WARMUP", params)

    def create_session(
        self,
        session_id: str,
        *,
        body_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> PolicySession:
        result = self.manager.call(
            "CREATE_SESSION",
            {
                "session_id": session_id,
                "body_id": body_id,
                "context": context or {},
            },
        )
        if result.get("status") != "ok":
            raise RuntimeError(result.get("error", {}).get("message", "create_session failed"))
        return PolicySession(
            session_id=session_id,
            policy_path=self.manager.policy_path or "",
            body_id=body_id,
            context=context or {},
        )

    def reset_session(self, session_id: str) -> dict[str, Any]:
        return self.manager.call("RESET_SESSION", {"session_id": session_id})

    def infer(
        self,
        session_id: str,
        observation: dict[str, Any],
        *,
        step_index: int | None = None,
        return_chunk: bool = True,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "session_id": session_id,
            "observation": observation,
            "return_chunk": return_chunk,
        }
        if step_index is not None:
            params["step_index"] = step_index
        return self.manager.call("INFER", params)

    def health(self) -> dict[str, Any]:
        return self.manager.call("HEALTH", {})

    def close_session(self, session_id: str) -> dict[str, Any]:
        return self.manager.call("CLOSE_SESSION", {"session_id": session_id})

    def unload_policy(self) -> dict[str, Any]:
        return self.manager.call("UNLOAD_POLICY", {})

    def shutdown(self) -> None:
        self.manager.stop()

    def __enter__(self) -> RuntimeClient:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
