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
            session_id = str(params.get("pi_session_id", ""))
            # HOTFIX-1：换绑（或重绑）作废旧 session 的全部 context
            # lease——切换后必须重新拉取 fresh context 才能动作。
            previous = self._bindings.binding_for_session(session_id)
            if previous is not None and previous.mission_id != mission_id:
                from rosclaw.agentd.pi_bridge.context_lease import ContextLeaseStore

                ContextLeaseStore(service._store.connection).revoke_for_session(session_id)
            binding = self._bindings.bind(
                pi_session_id=session_id,
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
        if method == "pi.operator.status":
            # 六审 §7：operator 面真实状态（enrollment + 进程运行）——
            # TUI 的单键初始化依赖它，不再要求用户另开终端。
            from rosclaw.operatord.enrollment import IDENTITY_FILE

            home = service._home
            enrolled = (home / "operatord" / IDENTITY_FILE).exists()
            sock = home / "run" / "operatord.sock"
            running = False
            if sock.exists():
                import socket as _socket

                try:
                    probe = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
                    probe.settimeout(1.0)
                    probe.connect(str(sock))
                    probe.close()
                    running = True
                except OSError:
                    running = False
            return {"ok": True, "enrolled": enrolled, "running": running}
        if method == "pi.operator.bootstrap":
            # 六审 §7：SIM developer 的单键初始化——enroll（如需要）+
            # 启动独立 operatord 进程（生命周期归 agentd service 管理；
            # 决定权/签名仍在 operatord 独立进程）。REAL/SHADOW 一律拒绝。
            mission_id = str(params.get("mission_id", ""))
            if mission_id:
                mission = service.get_mission(mission_id)
                if mission is not None and mission.mode.value != "SIMULATION":
                    return {
                        "ok": False,
                        "error": "operator bootstrap 仅限 SIMULATION developer——"
                        "REAL/SHADOW 要求独立 operator readiness/presence 流程",
                        "code": "MODE_FORBIDDEN",
                    }
            from rosclaw.operatord.enrollment import IDENTITY_FILE, enroll

            home = service._home
            identity_path = home / "operatord" / IDENTITY_FILE
            if not identity_path.exists():
                enroll(home / "operatord")
            sock = home / "run" / "operatord.sock"
            if not sock.exists():
                import subprocess as _sp
                import sys as _sys

                (home / "run").mkdir(parents=True, exist_ok=True)

                proc = _sp.Popen(  # noqa: S603 - 固定入口
                    [
                        _sys.executable, "-m", "rosclaw.entrypoint",
                        "operatord", "start", "--no-human-presence-check",
                    ],
                    env={**os.environ, "ROSCLAW_HOME": str(home)},
                    stdout=(home / "run" / "operatord.out.log").open("ab"),
                    stderr=(home / "run" / "operatord.err.log").open("ab"),
                )
                # 生命周期归 service——close 时终止。
                service._managed_operator = proc
                deadline = asyncio.get_event_loop().time() + 20
                while asyncio.get_event_loop().time() < deadline and not sock.exists():
                    if proc.poll() is not None:
                        return {
                            "ok": False,
                            "error": f"operatord 启动失败（exit {proc.returncode}）——"
                            "见 run/operatord.err.log",
                            "code": "OPERATOR_START_FAILED",
                        }
                    await asyncio.sleep(0.2)
            return {"ok": sock.exists(), "enrolled": True, "running": sock.exists()}
        if method == "pi.capabilities":
            # 六审 §6.2.1/§6.2.6：当前 body 的可信能力面——模型不再靠猜
            # capability ID。动作能力只列 body 兼容项；不兼容/被隔离项进
            # excluded 并附机器原因码。
            mission_id = str(params.get("mission_id", ""))
            mission = service.get_mission(mission_id) if mission_id else None
            if mission is None:
                return {"ok": False, "error": "unknown mission", "code": "MISSION_NOT_FOUND"}
            await service._ensure_mcp_discovered()
            from rosclaw.agentd.tooling.body_compat import check_body_compatibility

            body_id = mission.body_binding.body_id
            observation: list[dict[str, Any]] = []
            actions: list[dict[str, Any]] = []
            excluded: list[dict[str, Any]] = []
            for descriptor in service._tool_catalog.list():
                if descriptor.execution_class.value != "PHYSICAL_ACTION":
                    if descriptor.model_callable:
                        observation.append(
                            {
                                "capability_id": descriptor.tool_id,
                                "version": descriptor.version,
                                "source": descriptor.source,
                                "description": descriptor.description[:120],
                            }
                        )
                    continue
                reason = check_body_compatibility(descriptor, body_id)
                quarantine = service._tool_catalog.quarantine_reason(descriptor.tool_id)
                if quarantine and reason is None:
                    reason = "CAPABILITY_QUARANTINED"
                if mission.mode.value not in list(descriptor.supported_modes):
                    reason = reason or "MODE_FORBIDDEN"
                entry = {
                    "capability_id": descriptor.tool_id,
                    "version": descriptor.version,
                    "source": descriptor.source,
                    "risk_tier": descriptor.risk_tier,
                    "side_effect_class": descriptor.side_effect_class.value,
                    "description": descriptor.description[:120],
                }
                if reason is None:
                    actions.append(entry)
                else:
                    excluded.append({**entry, "reason": reason})
            return {
                "ok": True,
                "body_id": body_id,
                "mode": mission.mode.value,
                "observation_capabilities": observation,
                "action_capabilities": actions,
                "excluded": excluded,
            }
        if method == "pi.context":
            mission_id = str(params.get("mission_id", ""))
            if service.get_mission(mission_id) is None:
                return {"ok": False, "error": "unknown mission", "code": "MISSION_NOT_FOUND"}
            # PNA-2：完整 EmbodiedContextEnvelopeV1（TTL + 内容 hash）。
            # 六审 §6.3：capabilities 在 context_hash 内——lazy discovery
            # 必须先完成，否则发现前后两个 envelope 的 hash 不同
            # （lease 签发后 propose 重建即 CONTEXT_HASH_MISMATCH）。
            await service._ensure_mcp_discovered()
            from rosclaw.agentd.pi_bridge.context import build_embodied_context

            try:
                envelope = build_embodied_context(service, mission_id)
            except ValueError as exc:
                return {"ok": False, "error": str(exc), "code": "CONTEXT_UNAVAILABLE"}
            # HOTFIX-1（P0-4A）：context 校验成功后由 agentd 签发短期
            # ValidatedContextLease——action 准入的权威 freshness 凭证
            # （同一权威源，不信 TUI 自报）。无 session 不签发。
            response: dict[str, Any] = {"ok": True, "context": envelope.model_dump(mode="json")}
            pi_session_id = str(params.get("pi_session_id", ""))
            if pi_session_id:
                # P0-5A：只有合法 writer（binding + writer lease + peer
                # PID/UID 与 lease owner 匹配）才能签 action context
                # lease——观测面（envelope）仍可读，action lease 绝不
                # 发给冒名进程。caller_pid/caller_uid 来自 SO_PEERCRED，
                # JSON 参数不可覆写。
                caller_uid = int(principal.rsplit(":", 1)[-1])
                writer = self._bindings.writer_of(mission_id)
                is_legit_writer = (
                    writer is not None
                    and writer.pi_session_id == pi_session_id
                    and writer.owner_pid == peer_pid
                    and writer.owner_uid == caller_uid
                )
                if not is_legit_writer:
                    # 不签 lease——观测照常返回，动作准入凭证拒发。
                    response["context_lease_denied"] = "not the writer process"
                    return response
                # P0-5B：lease TTL = min(envelope TTL, writer lease 剩余,
                # policy max)——不得长于 prompt 里告诉模型的有效期。
                from datetime import UTC as _UTC
                from datetime import datetime as _dt

                from rosclaw.agentd.pi_bridge.context_lease import (
                    ContextLeaseStore,
                    context_hash_of,
                )

                envelope_ttl = max(
                    0.0,
                    (
                        _dt.fromisoformat(envelope.expires_at) - _dt.now(_UTC)
                    ).total_seconds(),
                )
                writer_ttl = max(
                    0.0,
                    (_dt.fromisoformat(writer.expires_at) - _dt.now(_UTC)).total_seconds(),
                )
                from rosclaw.agentd.pi_bridge.context_lease import LEASE_TTL_SEC

                effective_ttl = min(envelope_ttl, writer_ttl, LEASE_TTL_SEC)
                # 六审 §5.3：binding_id 必须是 session binding ID（此前
                # 错写 writer.lease_id）；writer_lease_id/caller_pid
                # 独立成字段。
                binding = self._bindings.binding_for_session(pi_session_id)
                if binding is None:
                    response["context_lease_denied"] = "no active session binding"
                    return response
                lease = ContextLeaseStore(service._store.connection).issue(
                    pi_session_id=pi_session_id,
                    mission_id=mission_id,
                    context_revision=envelope.context_revision,
                    context_hash=context_hash_of(envelope),
                    body_hash=envelope.body.get("effective_body_hash", ""),
                    mode=service.get_mission(mission_id).mode.value,
                    ttl_sec=effective_ttl,
                    binding_id=binding.binding_id,
                    caller_uid=caller_uid,
                    writer_lease_id=writer.lease_id,
                    caller_pid=peer_pid,
                )
                response["context_lease_id"] = lease.context_lease_id
                response["context_lease_expires_at"] = lease.expires_at
            return response
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
            # P0-NA-10：唯一 admission path——完整请求上下文是建卡前提。
            from rosclaw.agentd.pi_bridge.action_admission import (
                ActionAdmissionService,
                ActionRequestContext,
            )

            admission = ActionAdmissionService(service)
            try:
                card = await admission.propose(
                    request=ActionRequestContext(
                        pi_session_id=str(params.get("pi_session_id", "")),
                        mission_id=str(params.get("mission_id", "")),
                        context_revision=int(params.get("context_revision", -1)),
                        body_hash=str(params.get("body_hash", "")),
                        mode=str(params.get("mode", "")),
                        idempotency_key=str(params.get("idempotency_key", "")),
                        context_lease_id=str(params.get("context_lease_id", "")),
                    ),
                    capability_id=str(params.get("capability_id", "")),
                    arguments=dict(params.get("arguments") or {}),
                    expected_effect=str(params.get("expected_effect", "")),
                    risk_tier=str(params.get("risk_tier", "LOW")),
                    title=str(params.get("title", "")),
                    # P0-5A：SO_PEERCRED 真值注入——JSON 不可覆写。
                    caller_pid=peer_pid,
                    caller_uid=int(principal.rsplit(":", 1)[-1]),
                )
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                        "code": getattr(exc, "code", "PROPOSE_FAILED")}
            return {"ok": True, "card": card}
        if method == "pi.action.status":
            # HOTFIX-1（P0-4B）：status 也必须证明调用方是卡主——只凭
            # approval_id 不得窥探卡状态。
            from rosclaw.agentd.pi_bridge.action_admission import (
                ActionAdmissionService,
            )

            caller_session = str(params.get("pi_session_id", ""))
            if not caller_session:
                return {
                    "ok": False,
                    "error": "pi_session_id required (card ownership check)",
                    "code": "REQUEST_CONTEXT_REQUIRED",
                }
            binding = self._bindings.binding_for_session(caller_session)
            # 六审 §5.2：status 也做 caller 身份校验——同 UID 的另一个
            # 进程知道 session ID 也不得读卡状态。writer owner 必须匹配
            # SO_PEERCRED 的 peer PID/UID。
            caller_uid = int(principal.rsplit(":", 1)[-1])
            writer = (
                self._bindings.writer_of(binding.mission_id) if binding else None
            )
            if (
                binding is None
                or writer is None
                or writer.pi_session_id != caller_session
                or writer.owner_pid != peer_pid
                or writer.owner_uid != caller_uid
            ):
                return {
                    "ok": False,
                    "error": "caller is not the writer process (fail closed)",
                    "code": "CALLER_MISMATCH",
                }
            approval_id = str(params.get("approval_id", ""))
            stored = service._broker.get_request(approval_id)
            if stored is not None and binding.mission_id != stored.mission_id:
                return {
                    "ok": False,
                    "error": "not your card",
                    "code": "FORBIDDEN",
                }
            return {"ok": True, **ActionAdmissionService(service).decision_status(
                approval_id
            )}
        if method == "pi.action.execute":
            # P0-NA-10：execute 也带请求上下文做 TOCTOU 复验。
            from rosclaw.agentd.pi_bridge.action_admission import (
                ActionAdmissionService,
                ActionRequestContext,
            )

            admission = ActionAdmissionService(service)
            # HOTFIX-1（P0-4B）：请求上下文强制必填——没有"只给
            # approval_id 就执行"的绕过路径。
            if not params.get("pi_session_id"):
                return {
                    "ok": False,
                    "error": "full request context required (pi_session_id/mission_id/"
                    "context_revision/body_hash/mode/idempotency_key/context_lease_id)",
                    "code": "REQUEST_CONTEXT_REQUIRED",
                }
            request_ctx = ActionRequestContext(
                pi_session_id=str(params.get("pi_session_id", "")),
                mission_id=str(params.get("mission_id", "")),
                context_revision=int(params.get("context_revision", -1)),
                body_hash=str(params.get("body_hash", "")),
                mode=str(params.get("mode", "")),
                idempotency_key=str(params.get("idempotency_key", "")),
                context_lease_id=str(params.get("context_lease_id", "")),
            )
            try:
                result = await admission.execute(
                    str(params.get("approval_id", "")), request=request_ctx,
                    # P0-5A：SO_PEERCRED 真值注入——JSON 不可覆写。
                    caller_pid=peer_pid,
                    caller_uid=int(principal.rsplit(":", 1)[-1]),
                )
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
                # 六审 §5.5.2：dispatcher 的动作路径也要 caller 身份——
                # SO_PEERCRED 真值注入，JSON 不可覆写。
                result = await dispatcher.execute(
                    tool_request,
                    caller_pid=peer_pid,
                    caller_uid=int(principal.rsplit(":", 1)[-1]),
                )
            except ToolBridgeError as exc:
                return {"ok": False, "error": exc.message, "code": exc.code}
            return {"ok": result.ok, "result": result.model_dump(mode="json"),
                    "code": result.error_code}
        return {"ok": False, "error": f"unknown method {method!r}", "code": "METHOD_NOT_FOUND"}
