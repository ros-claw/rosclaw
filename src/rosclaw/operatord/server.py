"""rosclaw-operatord 服务进程（审计 P0-01）：唯一的人类授权决定点。

职责与红线：
- 唯一持有 operator enrollment key 的进程（0600，load 时校验权限）；
- `approvals.decide`：display hash 匹配 →（daemon 卡：human presence +
  rosclawd ACL 决定）→ 向 agentd 转发带 proof 的 apply_decision；
- `grants.revoke`：带 proof 转发；
- `estop`：直达 rosclawd（不经 agentd、不经模型）；
- 不做 Mission/工具/模型工作；普通 curl/同机进程没有 key，必然失败。

协议：与 agentd 投影 socket 相同的 JSONL（TUI/CLI 只换 socket 路径）。
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rosclaw.agentd.operator_socket import OperatorSocketServer, operator_call
from rosclaw.operatord.enrollment import (
    OperatorEnrollment,
    load_enrollment,
    sign_decision_proof,
)


def default_operatord_socket(home: Path | None = None) -> Path:
    base = home or Path(os.environ.get("ROSCLAW_HOME", Path.home() / ".rosclaw"))
    return base / "run" / "operatord.sock"


def default_agent_projection_socket(home: Path | None = None) -> Path:
    base = home or Path(os.environ.get("ROSCLAW_HOME", Path.home() / ".rosclaw"))
    return base / "run" / "operator.sock"


class OperatorDaemon:
    def __init__(
        self,
        *,
        enrollment: OperatorEnrollment,
        socket_path: Path,
        agent_socket: Path | None = None,
        daemon_client=None,
        require_human_presence: bool = True,
    ) -> None:
        self._enrollment = enrollment
        self._path = socket_path
        self._agent_socket = agent_socket
        self._daemon = daemon_client
        self._require_human_presence = require_human_presence
        self._used_nonces: set[str] = set()
        self._server: OperatorSocketServer | None = None

    async def start(self) -> None:
        self._server = _OperatordSocketServer(self, self._path)
        await self._server.start()

    async def stop(self) -> None:
        if self._server is not None:
            await self._server.stop()
            self._server = None

    # -- dispatch ----------------------------------------------------------------

    async def handle(self, principal: str, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "approvals.list":
            if self._agent_socket is None or not self._agent_socket.exists():
                return {"ok": False, "error": "agentd projection socket unavailable"}
            return await operator_call(self._agent_socket, "approvals.list", params)
        if method == "approvals.decide":
            return await self._decide(principal, params)
        if method == "grants.revoke":
            return await self._revoke(principal, params)
        if method == "estop":
            return await self._estop(principal, params)
        return {"ok": False, "error": f"unknown method {method!r}"}

    # -- decisions ---------------------------------------------------------------

    async def _decide(self, principal: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = str(params.get("request_id", ""))
        display_hash = str(params.get("display_hash", ""))
        approve = bool(params.get("approve"))
        if not request_id or not display_hash:
            return {"ok": False, "error": "request_id and display_hash required"}
        daemon_proposal_id = str(params.get("daemon_proposal_id", ""))
        decided_daemon = False
        if daemon_proposal_id and self._daemon is not None:
            # REAL/daemon 卡：先 human presence，再经 rosclawd ACL 决定。
            if self._require_human_presence and not self._human_present():
                return {
                    "ok": False,
                    "error": (
                        "human presence required for REAL/daemon decisions "
                        "(foreground TTY unavailable)"
                    ),
                }
            decided_daemon = await self._decide_daemon_proposal(
                daemon_proposal_id, request_id, display_hash, approve, principal, params
            )
            if not decided_daemon.get("ok"):
                return decided_daemon
        elif self._daemon is not None and str(params.get("mode", "")).upper() == "REAL":
            return {"ok": False, "error": "REAL decisions require a daemon proposal id"}
        # 组装 proof（nonce 一次性，防重放）。
        nonce = secrets.token_hex(16)
        decided_at = datetime.now(UTC).isoformat()
        proof = sign_decision_proof(
            self._enrollment,
            request_id=request_id,
            approve=approve,
            nonce=nonce,
            decided_at=decided_at,
            display_hash=display_hash,
        )
        if nonce in self._used_nonces:
            return {"ok": False, "error": "nonce replay"}
        self._used_nonces.add(nonce)
        if self._agent_socket is None or not self._agent_socket.exists():
            return {
                "ok": False,
                "error": "agentd projection socket unavailable — decision not applied",
            }
        applied = await operator_call(
            self._agent_socket,
            "approvals.apply_decision",
            {
                "request_id": request_id,
                "display_hash": display_hash,
                "approve": approve,
                "operator_proof": proof,
                "enrollment_id": self._enrollment.enrollment_id,
                "nonce": nonce,
                "decided_at": decided_at,
            },
        )
        if not applied.get("ok"):
            return applied
        return {
            "ok": True,
            "approved": approve,
            "grant_id": applied.get("grant_id"),
            "principal": principal,
            "daemon_decided": decided_daemon or None,
            "profile": applied.get("profile"),
        }

    async def _decide_daemon_proposal(
        self,
        proposal_id: str,
        request_id: str,
        display_hash: str,
        approve: bool,
        principal: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        from rosclaw.daemon.client import DaemonClientError

        nonce = secrets.token_hex(16)
        decided_at = datetime.now(UTC).isoformat()
        proof = sign_decision_proof(
            self._enrollment,
            request_id=request_id,
            approve=approve,
            nonce=nonce,
            decided_at=decided_at,
            display_hash=display_hash,
        )
        try:
            pending = await asyncio.to_thread(self._daemon.list_pending_operator_proposals)
            trusted = next(
                (p for p in pending.get("proposals", []) if p.get("request_id") == proposal_id),
                None,
            )
            if trusted is None:
                return {"ok": False, "error": f"no pending daemon proposal {proposal_id!r}"}
            result = await asyncio.to_thread(
                self._daemon.decide_operator_proposal,
                proposal_id,
                decision="ACCEPT" if approve else "DECLINE",
                principal_id=principal,
                challenge_nonce=trusted["challenge_nonce"],
                action_intent_hash=trusted["action_intent_hash"],
                channel="rosclaw_operatord",
                reason=str(params.get("reason", "operator decision via rosclaw-operatord")),
                operator_proof=proof,
                enrollment_id=self._enrollment.enrollment_id,
                display_hash=display_hash,
                decided_at=decided_at,
            )
            return {"ok": True, "daemon": result}
        except DaemonClientError as exc:
            return {"ok": False, "error": f"daemon decide failed: {exc.code}: {exc}"}

    def _human_present(self) -> bool:
        """human-presence 信号（审计 P0-01.5）：前台 TTY 可用即可交互确认。

        第一版：要求存在前台 TTY（/dev/tty 可读）；后续接按键挑战/桌面
        会话/polkit。无 TTY 一律 fail closed。
        """
        try:
            fd = os.open("/dev/tty", os.O_RDONLY | os.O_NONBLOCK)
        except OSError:
            return False
        else:
            os.close(fd)
            return True

    async def _revoke(self, principal: str, params: dict[str, Any]) -> dict[str, Any]:
        grant_id = str(params.get("grant_id", ""))
        if not grant_id:
            return {"ok": False, "error": "missing grant_id"}
        nonce = secrets.token_hex(16)
        decided_at = datetime.now(UTC).isoformat()
        proof = sign_decision_proof(
            self._enrollment,
            request_id=f"revoke:{grant_id}",
            approve=False,
            nonce=nonce,
            decided_at=decided_at,
            display_hash="",
        )
        if self._agent_socket is None or not self._agent_socket.exists():
            return {"ok": False, "error": "agentd projection socket unavailable"}
        return await operator_call(
            self._agent_socket,
            "approvals.apply_revoke",
            {"grant_id": grant_id, "operator_proof": proof},
        )

    async def _estop(self, principal: str, params: dict[str, Any]) -> dict[str, Any]:
        if self._daemon is None:
            return {
                "ok": False,
                "error": "estop unavailable: rosclawd not connected; nothing was stopped (honest)",
            }
        result = await asyncio.to_thread(
            self._daemon.emergency_stop,
            str(params.get("reason", "operator estop")),
            source=f"operatord:{principal}",
        )
        return {"ok": True, "estop": result, "principal": principal}


class _OperatordSocketServer(OperatorSocketServer):
    """复用 JSONL/peer-identity 机制；dispatch 转给 OperatorDaemon。"""

    def __init__(self, daemon: OperatorDaemon, socket_path: Path) -> None:
        self._daemon = daemon
        self._path = socket_path
        self._server = None

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
        from rosclaw.agentd.operator_socket import MAX_REQUEST_BYTES, _peer_principal

        try:
            principal = _peer_principal(writer)
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
                    response = await self._daemon.handle(
                        principal,
                        str(request.get("method", "")),
                        request.get("params") or {},
                    )
                except Exception as exc:  # noqa: BLE001
                    response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                writer.write(json.dumps(response, ensure_ascii=False).encode() + b"\n")
                await writer.drain()
        finally:
            writer.close()


async def run_operatord(
    *,
    home: Path,
    socket_path: Path | None = None,
    agent_socket: Path | None = None,
    daemon_socket: Path | None = None,
    require_human_presence: bool = True,
) -> OperatorDaemon:
    enrollment = load_enrollment(home / "operatord")
    daemon_client = None
    if daemon_socket is not None and daemon_socket.exists():
        from rosclaw.daemon.client import DaemonClient

        daemon_client = DaemonClient(socket_path=daemon_socket)
    daemon = OperatorDaemon(
        enrollment=enrollment,
        socket_path=socket_path or default_operatord_socket(home),
        agent_socket=agent_socket or default_agent_projection_socket(home),
        daemon_client=daemon_client,
        require_human_presence=require_human_presence,
    )
    await daemon.start()
    return daemon
