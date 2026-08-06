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
            if service.get_mission(mission_id) is None:
                return {"ok": False, "error": "unknown mission", "code": "MISSION_NOT_FOUND"}
            # PNA-2：完整 EmbodiedContextEnvelopeV1（TTL + 内容 hash）。
            from rosclaw.agentd.pi_bridge.context import build_embodied_context

            try:
                envelope = build_embodied_context(service, mission_id)
            except ValueError as exc:
                return {"ok": False, "error": str(exc), "code": "CONTEXT_UNAVAILABLE"}
            return {"ok": True, "context": envelope.model_dump(mode="json")}
        if method == "pi.mission.create":
            # PNA-6（规格 §13）：/new /fork 的新 Mission——fork 强制
            # SIMULATION，authority（grant/permit/approval）永不复制。
            goal = str(params.get("goal", "")) or "ROSClaw pi session"
            mode = str(params.get("mode", "SIMULATION")).upper()
            if mode != "SIMULATION":
                return {
                    "ok": False,
                    "error": "pi sessions may only create SIMULATION missions",
                    "code": "MODE_FORBIDDEN",
                }
            mission = service.create_mission(goal, mode="SIMULATION")
            return {"ok": True, "mission_id": mission.mission_id,
                    "mode": mission.mode.value}
        if method == "pi.session.binding.get":
            binding = self._bindings.binding_for_session(str(params.get("pi_session_id", "")))
            if binding is None:
                return {"ok": True, "binding": None}
            mission = service.get_mission(binding.mission_id)
            return {
                "ok": True,
                "binding": binding.model_dump(mode="json"),
                "mission_state": mission.state.value if mission else "MISSING",
                "mission_archived": service.mission_archived(binding.mission_id),
            }
        if method == "pi.action.propose":
            from rosclaw.agentd.pi_bridge.action_coordinator import ActionCoordinator

            coordinator = ActionCoordinator(service)
            try:
                card = await coordinator.propose(
                    mission_id=str(params.get("mission_id", "")),
                    capability_id=str(params.get("capability_id", "")),
                    arguments=dict(params.get("arguments") or {}),
                    expected_effect=str(params.get("expected_effect", "")),
                    risk_tier=str(params.get("risk_tier", "LOW")),
                    title=str(params.get("title", "")),
                )
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                        "code": getattr(exc, "code", "PROPOSE_FAILED")}
            return {"ok": True, "card": card}
        if method == "pi.action.status":
            from rosclaw.agentd.pi_bridge.action_coordinator import ActionCoordinator

            return {"ok": True, **ActionCoordinator(service).decision_status(
                str(params.get("approval_id", ""))
            )}
        if method == "pi.action.execute":
            from rosclaw.agentd.pi_bridge.action_coordinator import ActionCoordinator

            coordinator = ActionCoordinator(service)
            try:
                result = await coordinator.execute(str(params.get("approval_id", "")))
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                        "code": getattr(exc, "code", "EXECUTE_FAILED")}
            return {"ok": result.get("executed") or result.get("status") == "DECLINED",
                    "result": result}
        if method == "pi.events.batch":
            # PNA-8（规格 §24.2）：认知事件镜像——只存 hash/元数据，
            # 拒绝任何像全文的字段（不双写 transcript）。
            events = params.get("events")
            if not isinstance(events, list) or len(events) > 256:
                return {"ok": False, "error": "events must be a list of at most 256",
                        "code": "INVALID_ARGUMENT"}
            stored = 0
            for event in events:
                if not isinstance(event, dict):
                    continue
                summary = str(event.get("summary", ""))
                if len(summary) > 200:
                    return {
                        "ok": False,
                        "error": "mirror summaries must be <= 200 chars (no full-text mirroring)",
                        "code": "FULL_TEXT_FORBIDDEN",
                    }
                content = str(event.get("content", ""))
                if content:
                    return {
                        "ok": False,
                        "error": "mirror events must not carry content text (hash only)",
                        "code": "FULL_TEXT_FORBIDDEN",
                    }
                service._store.connection.execute(
                    "INSERT INTO pi_event_mirrors (mirror_id, pi_session_id, mission_id, "
                    "event_type, pi_entry_id, content_hash, model, usage_json, occurred_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(event.get("mirror_id", "")) or f"mir_{stored}",
                        str(event.get("pi_session_id", "")),
                        str(event.get("mission_id", "")),
                        str(event.get("event_type", "")),
                        str(event.get("pi_entry_id", "")),
                        str(event.get("content_hash", "")),
                        str(event.get("model", "")),
                        json.dumps(event.get("usage", {})),
                        str(event.get("occurred_at", "")),
                    ),
                )
                stored += 1
            service._store.connection.commit()
            return {"ok": True, "stored": stored}
        if method == "pi.worker.status":
            # PNA-4：Worker 状态投影（原位更新 UI 用；只读）。
            mission_id = str(params.get("mission_id", ""))
            orders = service._worker_manager.orders_for_mission(mission_id)
            return {
                "ok": True,
                "orders": [
                    {
                        "work_order_id": o.work_order_id,
                        "assigned_to": o.assigned_to,
                        "status": o.status,
                        "goal": o.goal[:120],
                    }
                    for o in orders
                ],
            }
        if method == "pi.tools.execute":
            # PNA-3：完整验证链（binding/mission/lease/allowlist/idempotency）。
            from rosclaw.agentd.pi_bridge.tool_dispatch import (
                PiToolDispatcher,
                ToolBridgeError,
            )
            from rosclaw.contracts.pi.tool_request import PiToolRequestV1

            try:
                tool_request = PiToolRequestV1(**dict(params.get("request") or {}))
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": f"invalid tool request: {exc}",
                        "code": "INVALID_REQUEST"}
            dispatcher = PiToolDispatcher(service)
            try:
                result = await dispatcher.execute(tool_request)
            except ToolBridgeError as exc:
                return {"ok": False, "error": exc.message, "code": exc.code}
            return {"ok": result.ok, "result": result.model_dump(mode="json"),
                    "code": result.error_code}
        return {"ok": False, "error": f"unknown method {method!r}", "code": "METHOD_NOT_FOUND"}
