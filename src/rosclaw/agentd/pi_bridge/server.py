"""Pi Bridge UDS（重构规格 §18，PR-PNA-1）：rosclaw-agent ↔ agentd 专用通道。

方法（JSONL，与 operator socket 同传输）：
- pi.session.bind / pi.session.heartbeat / pi.session.release
- pi.status（agentd/mission/body/mode 摘要）
- pi.context（PNA-2 完整 EmbodiedContextEnvelope；PNA-1 先返回
  mission+mode+body+freshness 最小集）

安全：SO_PEERCRED + ephemeral control token（0600 文件；token 不进
命令行/journal/session）；请求体 256KiB 上限。
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rosclaw.agentd.operator_socket import MAX_REQUEST_BYTES, _peer_credentials

if TYPE_CHECKING:
    from rosclaw.agentd.service import AgentService

from rosclaw.agentd.pi_bridge.session_binding import BindingError, SessionBindingStore


def default_pi_bridge_socket(home: Path | None = None) -> Path:
    base = home or Path(os.environ.get("ROSCLAW_HOME", Path.home() / ".rosclaw"))
    return base / "run" / "pi-bridge.sock"


class PiBridgeServer:
    """agentd 内的 Pi bridge：session 绑定 + 状态/上下文投影。"""

    def __init__(self, service: AgentService, socket_path: Path) -> None:
        self._service = service
        self._path = socket_path
        self._server: asyncio.AbstractServer | None = None
        self._bindings = SessionBindingStore(service._store.connection)

    async def start(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self._path.parent, 0o700)
        self._path.unlink(missing_ok=True)
        self._server = await asyncio.start_unix_server(self._handle, path=str(self._path))
        os.chmod(self._path, 0o600)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self._path.unlink(missing_ok=True)

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            principal, peer_pid = _peer_credentials(writer)
        except Exception as exc:  # noqa: BLE001
            writer.write(json.dumps({"ok": False, "error": str(exc)}).encode() + b"\n")
            await writer.drain()
            writer.close()
            return
        try:
            while not reader.at_eof():
                line = await reader.readline()
                if not line:
                    break
                if len(line) > MAX_REQUEST_BYTES:
                    writer.write(b'{"ok": false, "error": "request too large"}\n')
                    await writer.drain()
                    break
                try:
                    request = json.loads(line)
                    response = await self._dispatch(
                        principal, peer_pid, str(request.get("method", "")),
                        request.get("params") or {},
                    )
                except BindingError as exc:
                    response = {"ok": False, "error": exc.message, "code": exc.code}
                except Exception as exc:  # noqa: BLE001
                    response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                writer.write(json.dumps(response, ensure_ascii=False).encode() + b"\n")
                await writer.drain()
        finally:
            writer.close()

    def _authorized(self, params: dict[str, Any]) -> bool:
        """ephemeral control token（与 HTTP 面同一 token；0600 文件分发）。"""
        return str(params.get("token", "")) == self._service.control_token

    async def _dispatch(
        self, principal: str, peer_pid: int, method: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        if not self._authorized(params):
            return {"ok": False, "error": "control token required", "code": "UNAUTHORIZED"}
        service = self._service
        if method == "pi.session.bind":
            mission_id = str(params.get("mission_id", ""))
            mission = service.get_mission(mission_id)
            if mission is None:
                return {"ok": False, "error": "unknown mission", "code": "MISSION_NOT_FOUND"}
            binding = self._bindings.bind(
                pi_session_id=str(params.get("pi_session_id", "")),
                pi_session_path=str(params.get("pi_session_path", "")),
                mission_id=mission_id,
                body_id=mission.body_binding.body_id,
                execution_mode=mission.mode.value,
                created_by=principal,
            )
            lease, token = self._bindings.acquire_lease(
                mission_id=mission_id,
                pi_session_id=binding.pi_session_id,
                owner_pid=peer_pid,
                owner_uid=int(principal.rsplit(":", 1)[-1]),
            )
            return {
                "ok": True,
                "binding": binding.model_dump(mode="json"),
                "lease": lease.model_dump(mode="json"),
                "lease_token": token,
            }
        if method == "pi.session.heartbeat":
            lease = self._bindings.heartbeat_lease(
                str(params.get("mission_id", "")),
                str(params.get("pi_session_id", "")),
                str(params.get("lease_token", "")),
            )
            return {"ok": True, "lease": lease.model_dump(mode="json")}
        if method == "pi.session.release":
            released = self._bindings.release_lease(
                str(params.get("mission_id", "")),
                str(params.get("pi_session_id", "")),
                str(params.get("lease_token", "")),
            )
            return {"ok": True, "released": released}
        if method == "pi.status":
            mission_id = str(params.get("mission_id", ""))
            mission = service.get_mission(mission_id) if mission_id else None
            return {
                "ok": True,
                "agentd": "READY",
                "authorization_profile": service.authorization_profile(),
                "mission": (
                    {
                        "mission_id": mission.mission_id,
                        "state": mission.state.value,
                        "mode": mission.mode.value,
                        "body_id": mission.body_binding.body_id,
                    }
                    if mission
                    else None
                ),
            }
        if method == "pi.context":
            mission_id = str(params.get("mission_id", ""))
            mission = service.get_mission(mission_id)
            if mission is None:
                return {"ok": False, "error": "unknown mission", "code": "MISSION_NOT_FOUND"}
            snapshot = service.snapshot(mission_id)
            return {
                "ok": True,
                "context": {
                    "schema_version": "rosclaw.embodied_context.v0",
                    "mission_id": mission_id,
                    "mode": mission.mode.value,
                    "state": mission.state.value,
                    "body_id": mission.body_binding.body_id,
                    "body_hash": mission.body_binding.effective_body_hash,
                    "last_event_sequence": snapshot.last_event_sequence,
                    "pending_approvals": len(snapshot.pending_approvals),
                },
            }
        return {"ok": False, "error": f"unknown method {method!r}", "code": "METHOD_NOT_FOUND"}
