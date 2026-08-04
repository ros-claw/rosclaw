"""Operator 安全通道（PR-11，大纲 §14）。

与模型可见的 Agent API 物理分离的独立 UDS：

* **peer identity**：principal 从 SO_PEERCRED 的 UID 派生
  （``user:local:<uid>``）——请求体里的 principal 字段永远被忽略，
  客户端无法伪造身份（§19.6）。
* **display hash**：approve/deny 必须携带卡片 display_hash；与存储卡片
  重算值不匹配即拒绝（防止 TOCTOU 换卡）。
* **estop**：直达 rosclawd，绕过模型；无 daemon 时诚实报不可用。
* 协议：JSON Lines（每行一个请求 JSON，一行响应 JSON）。
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
        if method == "approvals.decide":
            request_id = str(params.get("request_id", ""))
            provided_hash = str(params.get("display_hash", ""))
            pending = {r.request_id: r for r in service.pending_approvals()}
            card = pending.get(request_id)
            if card is None:
                return {"ok": False, "error": "unknown_or_decided request_id"}
            expected = display_hash_for(card)
            if not provided_hash or provided_hash != expected:
                # display hash 不匹配 → 拒绝（§19.6）。
                return {"ok": False, "error": "display_hash_mismatch"}
            grant = await service.decide_approval(
                request_id,
                principal=principal,  # peer identity 唯一来源；body 里的 principal 被忽略
                approve=bool(params.get("approve")),
            )
            return {
                "ok": True,
                "approved": bool(params.get("approve")),
                "grant_id": grant.grant_id if grant else None,
                "principal": principal,
            }
        if method == "grants.revoke":
            grant_id = str(params.get("grant_id", ""))
            if not grant_id:
                return {"ok": False, "error": "missing grant_id"}
            service.revoke_grant(grant_id, principal=principal)
            return {"ok": True, "revoked": grant_id, "principal": principal}
        if method == "estop":
            # 直达 rosclawd，绕过模型（§14.2）；无 daemon 诚实报不可用。
            result = await service.estop(str(params.get("reason", "operator estop")), principal=principal)
            return {"ok": True, "estop": result, "principal": principal}
        return {"ok": False, "error": f"unknown method {method!r}"}


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
