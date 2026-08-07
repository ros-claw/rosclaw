"""rosclaw-operatord 服务进程（二次复核 R1/P0-1/P0-3）：唯一的人类授权决定点。

职责与红线：
- 唯一持有 operator Ed25519 私钥的进程（0600，load 校验权限/链接数）；
- `approvals.decide`：
  * daemon 卡（REAL/SHADOW）——取 daemon 一次性 challenge（nonce 同源，
    P0-3）→ 校验请求方在前台进程组（P0-1.4）→ /dev/tty 显示不可变
    动作卡并读取显式 Y/N（默认/超时/EOF 一律 deny，P0-1）→ Ed25519
    签 proof → rosclawd 验证并签发 DecisionReceiptV1 → 转发 agentd；
  * SIM 卡（DEV_SIM_ONLY）——Ed25519 签 apply payload，agentd TOFU
    钉住公钥验证；
- `grants.revoke`：签名转发（fail-safe 方向，不要求 tty）；
- `estop`：直达 rosclawd（不经 agentd、不经模型）；
- 不做 Mission/工具/模型工作；没有私钥的进程必然失败。

协议：与 agentd 投影 socket 相同的 JSONL（TUI/CLI 只换 socket 路径）。
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rosclaw.agentd.operator_socket import OperatorSocketServer, operator_call
from rosclaw.contracts.operator.decision import (
    ACCEPT,
    DECLINE,
    DecisionChallengeV1,
    OperatorDecisionProofV1,
    canonical_json,
)
from rosclaw.operatord.enrollment import OperatorIdentity, load_identity
from rosclaw.operatord.human import confirm_on_tty, render_card, requester_is_foreground


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
        identity: OperatorIdentity,
        socket_path: Path,
        agent_socket: Path | None = None,
        daemon_client=None,
        require_human_presence: bool = True,
    ) -> None:
        self._identity = identity
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

    async def handle(
        self,
        principal: str,
        method: str,
        params: dict[str, Any],
        *,
        peer_pid: int = 0,
    ) -> dict[str, Any]:
        if method == "approvals.list":
            if self._agent_socket is None or not self._agent_socket.exists():
                return {"ok": False, "error": "agentd projection socket unavailable"}
            return await operator_call(self._agent_socket, "approvals.list", params)
        if method == "approvals.get":
            # P0-NA-14：精确单卡查询（不扫 list）。
            if self._agent_socket is None or not self._agent_socket.exists():
                return {"ok": False, "error": "agentd projection socket unavailable"}
            return await operator_call(self._agent_socket, "approvals.get", params)
        if method == "approvals.decide":
            return await self._decide(principal, params, peer_pid=peer_pid)
        if method == "grants.revoke":
            return await self._revoke(principal, params)
        if method == "estop":
            return await self._estop(principal, params)
        return {"ok": False, "error": f"unknown method {method!r}"}

    # -- decisions ---------------------------------------------------------------

    async def _decide(
        self, principal: str, params: dict[str, Any], *, peer_pid: int = 0
    ) -> dict[str, Any]:
        request_id = str(params.get("request_id", ""))
        display_hash = str(params.get("display_hash", ""))
        approve = bool(params.get("approve"))
        if not request_id or not display_hash:
            return {"ok": False, "error": "request_id and display_hash required"}
        # 从 agentd 投影找到卡片，确认 display_hash 与 daemon 归属。
        card = await self._find_card(request_id)
        if card is None:
            return {"ok": False, "error": f"no pending approval card {request_id!r}"}
        if str(card.get("display_hash", "")) != display_hash:
            return {"ok": False, "error": "display_hash_mismatch"}
        daemon_proposal_id = str(card.get("daemon_proposal_id", "") or "")
        if daemon_proposal_id:
            return await self._decide_daemon_card(
                card, daemon_proposal_id, principal, params, peer_pid=peer_pid
            )
        if self._daemon is not None and str(card.get("mode", "")).upper() == "REAL":
            return {"ok": False, "error": "REAL decisions require a daemon proposal id"}
        return await self._decide_sim_card(card, approve, principal)

    async def _find_card(self, request_id: str) -> dict[str, Any] | None:
        if self._agent_socket is None or not self._agent_socket.exists():
            return None
        listed = await operator_call(self._agent_socket, "approvals.list", {})
        for entry in listed.get("approvals", []):
            if str(entry.get("request_id", "")) == request_id:
                return entry
        return None

    async def _decide_daemon_card(
        self,
        card: dict[str, Any],
        proposal_id: str,
        principal: str,
        params: dict[str, Any],
        *,
        peer_pid: int,
    ) -> dict[str, Any]:
        """REAL/SHADOW 卡：challenge → 前台校验 → tty Y/N → proof → receipt。"""
        from rosclaw.daemon.client import DaemonClientError

        if self._daemon is None:
            return {"ok": False, "error": "daemon-backed card but rosclawd not connected"}
        # P0-1.4：请求方必须位于其 TTY 的前台进程组——后台进程/重定向
        # 触发的 decide 一律拒绝。
        if self._require_human_presence and not requester_is_foreground(peer_pid):
            return {
                "ok": False,
                "error": (
                    "requester is not in the foreground process group of a "
                    "controlling terminal — REAL/daemon decisions require a "
                    "foreground operator (P0-1)"
                ),
            }
        try:
            challenge_raw = await asyncio.to_thread(
                self._daemon.get_operator_challenge, proposal_id
            )
        except DaemonClientError as exc:
            return {"ok": False, "error": f"challenge unavailable: {exc.code}: {exc}"}
        try:
            challenge = DecisionChallengeV1.from_dict(dict(challenge_raw["challenge"]))
        except (ValueError, KeyError) as exc:
            return {"ok": False, "error": f"invalid daemon challenge: {exc}"}
        # daemon 的 challenge display_hash 必须与 agentd 卡片一致——
        # 两侧任一内容变化都会导致决定失败（P0-1.7/P0-6）。
        if challenge.display_hash != str(card.get("display_hash", "")):
            return {
                "ok": False,
                "error": (
                    "daemon challenge display_hash != agentd card hash — "
                    "card content changed or out of sync; refusing to decide"
                ),
            }
        # P0-1：真实前台 Y/N。tty 的回答就是决定（params.approve 仅是
        # 调用方意图；超时/EOF/N 一律 deny）。
        if self._require_human_presence:
            prompt = render_card(
                title=str(card.get("title", proposal_id)),
                summary=str(card.get("summary", "")),
                risk_tier=str(card.get("risk_tier", "")),
                mode=challenge.execution_mode,
                capability=challenge.capability_id,
                parameters=dict(card.get("parameters", {})),
                display_hash=challenge.display_hash,
                challenge_nonce=challenge.challenge_nonce,
                expires_at=challenge.expires_at,
            )
            answer = await asyncio.to_thread(confirm_on_tty, prompt)
            if answer.decision is None:
                return {"ok": False, "error": f"human confirmation failed: {answer.detail}"}
            decision = ACCEPT if answer.decision else DECLINE
            method = answer.method
        else:
            # 仅 DEV_SIM_ONLY/测试剖面：无 tty 确认，方法名如实记录。
            decision = ACCEPT if bool(params.get("approve")) else DECLINE
            method = "disabled-dev-sim"
        decided_at = datetime.now(UTC).isoformat()
        proof = OperatorDecisionProofV1(
            enrollment_id=self._identity.enrollment_id,
            challenge=challenge,
            decision=decision,
            decided_at=decided_at,
            human_confirmation_method=method,
        )
        proof = replace(
            proof, signature_b64=self._identity.sign(proof.signing_payload())
        )
        try:
            result = await asyncio.to_thread(
                self._daemon.decide_operator_proposal,
                proposal_id,
                decision=decision,
                principal_id=principal,
                channel="rosclaw_operatord",
                reason=str(params.get("reason", "operator decision via rosclaw-operatord")),
                proof=proof.to_dict(),
            )
        except DaemonClientError as exc:
            return {"ok": False, "error": f"daemon decide failed: {exc.code}: {exc}"}
        receipt = result.get("decision_receipt")
        if not isinstance(receipt, dict) or not receipt.get("signature_b64"):
            return {"ok": False, "error": "daemon returned no signed decision receipt"}
        # 转发精确 receipt 给 agentd（R3：agentd 只接受字段全匹配的
        # ACCEPT receipt；DECLINE 只关闭请求）。
        applied = await operator_call(
            self._agent_socket,
            "approvals.apply_decision",
            {
                "request_id": str(card.get("request_id", "")),
                "display_hash": challenge.display_hash,
                "approve": decision == ACCEPT,
                "decision_receipt": receipt,
            },
        )
        if not applied.get("ok"):
            return applied
        return {
            "ok": True,
            "approved": decision == ACCEPT,
            "grant_id": applied.get("grant_id"),
            "principal": principal,
            "daemon_decided": True,
            "receipt_id": receipt.get("proposal_id", ""),
            "profile": applied.get("profile"),
        }

    async def _decide_sim_card(
        self, card: dict[str, Any], approve: bool, principal: str
    ) -> dict[str, Any]:
        """SIM broker 卡（DEV_SIM_ONLY）：Ed25519 签 apply payload。"""
        nonce = secrets.token_hex(16)
        decided_at = datetime.now(UTC).isoformat()
        if nonce in self._used_nonces:
            return {"ok": False, "error": "nonce replay"}
        self._used_nonces.add(nonce)
        if self._agent_socket is None or not self._agent_socket.exists():
            return {
                "ok": False,
                "error": "agentd projection socket unavailable — decision not applied",
            }
        payload = {
            "request_id": str(card.get("request_id", "")),
            "display_hash": str(card.get("display_hash", "")),
            "approve": approve,
            "nonce": nonce,
            "decided_at": decided_at,
            "enrollment_id": self._identity.enrollment_id,
        }
        signature = self._identity.sign(canonical_json(payload))
        applied = await operator_call(
            self._agent_socket,
            "approvals.apply_decision",
            {
                **payload,
                "operator_signature": signature,
                "operator_public_key_pem": self._identity.public_key_pem,
            },
        )
        if not applied.get("ok"):
            return applied
        return {
            "ok": True,
            "approved": approve,
            "grant_id": applied.get("grant_id"),
            "principal": principal,
            "profile": applied.get("profile"),
        }

    async def _revoke(self, principal: str, params: dict[str, Any]) -> dict[str, Any]:
        grant_id = str(params.get("grant_id", ""))
        if not grant_id:
            return {"ok": False, "error": "missing grant_id"}
        if self._agent_socket is None or not self._agent_socket.exists():
            return {"ok": False, "error": "agentd projection socket unavailable"}
        nonce = secrets.token_hex(16)
        payload = {
            "action": "revoke",
            "grant_id": grant_id,
            "nonce": nonce,
            "decided_at": datetime.now(UTC).isoformat(),
            "enrollment_id": self._identity.enrollment_id,
        }
        signature = self._identity.sign(canonical_json(payload))
        return await operator_call(
            self._agent_socket,
            "approvals.apply_revoke",
            {
                "grant_id": grant_id,
                "operator_signature": signature,
                "operator_enrollment_id": self._identity.enrollment_id,
                "operator_public_key_pem": self._identity.public_key_pem,
                "nonce": nonce,
                "decided_at": payload["decided_at"],
            },
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
        from rosclaw.agentd.operator_socket import MAX_REQUEST_BYTES, _peer_credentials

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
                    response = await self._daemon.handle(
                        principal,
                        str(request.get("method", "")),
                        request.get("params") or {},
                        peer_pid=peer_pid,
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
    identity = load_identity(home / "operatord")
    daemon_client = None
    if daemon_socket is not None and daemon_socket.exists():
        from rosclaw.daemon.client import DaemonClient

        daemon_client = DaemonClient(socket_path=daemon_socket)
    daemon = OperatorDaemon(
        identity=identity,
        socket_path=socket_path or default_operatord_socket(home),
        agent_socket=agent_socket or default_agent_projection_socket(home),
        daemon_client=daemon_client,
        require_human_presence=require_human_presence,
    )
    await daemon.start()
    return daemon
