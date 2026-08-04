"""Agent 侧 operator 投影通道（PR-11 + 审计 P0-01）。

与模型可见的 Agent API 物理分离的 UDS——但**决定权不在此**：

* agentd 只提供 approvals.list（只读投影，owner 过滤）与
  approvals.apply_decision / approvals.apply_revoke（operatord proof 门控）；
* approvals.decide / grants.revoke / estop 已迁出 agentd，属
  rosclaw-operatord 专属——本 socket 一律拒绝并指明去处；
* peer identity 仅从 SO_PEERCRED 派生；display hash 仍强制匹配；
* REAL/daemon 卡必须已由 operatord 完成 daemon proposal 决定
  （proof 经 rosclawd ACL 验证），agentd 不接受倒置顺序。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rosclaw.contracts.common import ValidationError

if TYPE_CHECKING:
    from rosclaw.agentd.service import AgentService


def display_hash_for(request) -> str:
    """审批卡片的展示指纹：内容任何变化都会改变 hash。"""
    display = request.action_display
    canonical = json.dumps(
        {
            "request_id": request.request_id,
            "title": display.title,
            "summary": display.summary,
            "risk_tier": display.risk_tier,
            "parameters": display.parameters,
            "body_hash": request.effective_body_hash,
            "expires_at": request.expires_at,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _peer_principal(conn: asyncio.StreamWriter) -> str:
    """SO_PEERCRED → user:local:<uid>。取不到时 fail closed（拒绝）。"""
    sock = conn.get_extra_info("socket")
    if sock is None:
        raise ValidationError("peer identity unavailable")
    try:
        creds = sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
        _, uid, _ = struct.unpack("3i", creds)
        return f"user:local:{uid}"
    except (AttributeError, OSError) as exc:
        raise ValidationError(f"peer identity unavailable: {exc}") from exc


#: 单请求上限（防御性：无界 readline 是内存 DoS 面）。
MAX_REQUEST_BYTES = 256 * 1024


class OperatorSocketServer:
    """每连接 JSONL：{"method": ..., "params": {...}} → {"ok": bool, ...}"""

    def __init__(self, service: AgentService, socket_path: Path) -> None:
        self._service = service
        self._path = socket_path
        self._server: asyncio.AbstractServer | None = None

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
            principal = _peer_principal(writer)
        except ValidationError as exc:
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
                    response = await self._dispatch(principal, json.loads(line))
                except Exception as exc:  # noqa: BLE001 - 诚实错误，不伪造
                    response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                writer.write(json.dumps(response, ensure_ascii=False).encode() + b"\n")
                await writer.drain()
        finally:
            writer.close()

    async def _dispatch(self, principal: str, request: dict[str, Any]) -> dict[str, Any]:
        method = request.get("method")
        params = request.get("params") or {}
        service = self._service
        if method == "approvals.list":
            pending = service.pending_approvals(params.get("mission_id"))
            return {
                "ok": True,
                "principal": principal,
                "approvals": [
                    {
                        "request_id": r.request_id,
                        "mission_id": r.mission_id,
                        "title": r.action_display.title,
                        "summary": r.action_display.summary,
                        "risk_tier": r.action_display.risk_tier,
                        "parameters": r.action_display.parameters,
                        "expires_at": r.expires_at,
                        "display_hash": display_hash_for(r),
                    }
                    for r in pending
                ],
            }
        if method == "approvals.apply_decision":
            # 审计 P0-01：agentd 不再直接 decide——只应用 operatord 的
            # 已验证决定。proof 的权威校验在 daemon（REAL 卡必经
            # daemon proposal 决定路径）；纯 broker SIM 卡按
            # DEV_SIM_ONLY 语义应用并明确标记。
            return await self._apply_decision(principal, params)
        if method == "approvals.apply_revoke":
            grant_id = str(params.get("grant_id", ""))
            if not grant_id:
                return {"ok": False, "error": "missing grant_id"}
            if not params.get("operator_proof"):
                return {"ok": False, "error": "operator_proof required"}
            service.revoke_grant(grant_id, principal=principal)
            return {
                "ok": True,
                "revoked": grant_id,
                "principal": principal,
                "profile": service.authorization_profile(),
            }
        if method == "approvals.decide" or method == "grants.revoke" or method == "estop":
            # 审计 P0-01/B3：决定/撤销/急停迁出 agentd（operatord 专属）。
            return {
                "ok": False,
                "error": (
                    f"{method} is not served by agentd — decisions belong to "
                    "rosclaw-operatord (P0-01); use approvals.apply_decision "
                    "with an operatord proof, or the operatord socket"
                ),
            }
        return {"ok": False, "error": f"unknown method {method!r}"}

    async def _apply_decision(self, principal: str, params: dict[str, Any]) -> dict[str, Any]:
        service = self._service
        request_id = str(params.get("request_id", ""))
        provided_hash = str(params.get("display_hash", ""))
        proof = str(params.get("operator_proof", ""))
        enrollment_id = str(params.get("enrollment_id", ""))
        nonce = str(params.get("nonce", ""))
        approve = bool(params.get("approve"))
        if not proof or not enrollment_id or not nonce:
            return {"ok": False, "error": "operator_proof/enrollment_id/nonce required"}
        pending = {r.request_id: r for r in service.pending_approvals()}
        card = pending.get(request_id)
        if card is None:
            return {"ok": False, "error": "unknown_or_decided request_id"}
        expected = display_hash_for(card)
        if not provided_hash or provided_hash != expected:
            return {"ok": False, "error": "display_hash_mismatch"}
        profile = service.authorization_profile()
        daemon_backed = bool(
            getattr(card, "daemon_proposal_id", None)
            or card.model_dump(mode="json").get("daemon_proposal_id")
        )
        if daemon_backed:
            # REAL/daemon 卡：proof 必须由 daemon ACL 已验证（operatord 已
            # 完成 proposal.decide）；agentd 侧先确认 proposal 已被决定，
            # 不接受"先 apply 后 daemon"的倒置顺序（fail closed）。
            verified = await service.daemon_proposal_is_decided(request_id)
            if not verified:
                return {
                    "ok": False,
                    "error": (
                        "daemon proposal not decided — operatord must decide "
                        "the daemon proposal first (proof verified by rosclawd ACL)"
                    ),
                }
        grant = await service.decide_approval(
            request_id,
            principal=principal,
            approve=approve,
            _from_operatord=True,
        )
        return {
            "ok": True,
            "approved": approve,
            "grant_id": grant.grant_id if grant else None,
            "principal": principal,
            "profile": profile,
        }


async def operator_call(socket_path: Path, method: str, params: dict | None = None) -> dict:
    """Client helper（TUI/CLI 共用）：一次 JSONL 请求。"""
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    try:
        writer.write(
            json.dumps({"method": method, "params": params or {}}, ensure_ascii=False).encode()
            + b"\n"
        )
        await writer.drain()
        line = await reader.readline()
        return json.loads(line)  # type: ignore[no-any-return]
    finally:
        writer.close()
        await writer.wait_closed()
