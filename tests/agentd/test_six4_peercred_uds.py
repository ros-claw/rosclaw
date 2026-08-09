"""PR-SIX-4 红测试（六审 §5）：真实 SO_PEERCRED 双进程 Gate。

红测试先行——五审 HF5-1 的测试说明声称"真 UDS 双进程"，实际核心
测试直接调 `_dispatch(..., peer_pid=999)` 伪造参数。本文件的核心
证明不调用 `_dispatch`、不把 PID 当参数传入：

1. 真实 UDS + 两个真实进程：legit 子进程 bind（内核 SO_PEERCRED
   记录其真实 PID）→ 攻击者子进程（同 UID、不同 PID、持 token +
   session ID）拉 context/propose/status/execute 全拒；
2. pi.action.status 必须做 writer peer 校验（当前只比 session）；
3. ContextLease 的 binding_id 必须是 session binding ID（当前写入
   writer lease ID），且换绑后旧 lease 失效；
4. legacy caller_uid=-1 的 lease 一律不得用于 action（当前 -1 绕过
   比对）。
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from tests.agentd.test_pi_approval import _setup_with_operatord

# ------------------------------------------------------------ 真 UDS 客户端
_CLIENT_PY = r"""
import json, socket, sys

sock_path, token, payload_file = sys.argv[1], sys.argv[2], sys.argv[3]
requests = json.loads(open(payload_file).read())
out = []
lease = ""
for method, params in requests:
    params = {k: (lease if v == "@lease" else v) for k, v in params.items()}
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(15)
    s.connect(sock_path)
    s.sendall(json.dumps({"method": method, "params": {"token": token, **params}}).encode() + b"\n")
    buf = b""
    while b"\n" not in buf:
        chunk = s.recv(65536)
        if not chunk:
            break
        buf += chunk
    s.close()
    resp = json.loads(buf.split(b"\n")[0] or b"{}")
    if resp.get("context_lease_id"):
        lease = resp["context_lease_id"]
    out.append(resp)
print(json.dumps(out))
"""


async def _run_client(sock: Path, token: str, requests: list) -> tuple[int, list]:
    """在真实子进程里跑 UDS 客户端——返回 (子进程 PID, 响应列表)。"""
    payload = "/tmp/six4-client-payload.json"
    Path(payload).write_text(json.dumps(requests), encoding="utf-8")
    Path("/tmp/six4-client.py").write_text(_CLIENT_PY, encoding="utf-8")
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "/tmp/six4-client.py", str(sock), token, payload,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    pid = proc.pid
    stdout, stderr = await proc.communicate()
    assert proc.returncode == 0, f"client failed: {stderr.decode()[-300:]}"
    return pid, json.loads(stdout.decode())


class TestRealUdsPeerCred:
    """真实 UDS + 真实子进程——PID 由内核 SO_PEERCRED 获取。"""

    async def test_two_process_caller_gate(self, tmp_path: Path) -> None:
        service, mission, operatord, agent_server, sock = await _setup_with_operatord(
            tmp_path
        )
        # 独立 mission——pi_1 已被 _setup 绑定并持有 writer lease。
        mission = service.create_mission("uds peercred gate 测试")
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer

        bridge_sock = tmp_path / "run" / "pi-bridge.sock"
        bridge_sock.parent.mkdir(parents=True, exist_ok=True)
        bridge = PiBridgeServer(service, bridge_sock)
        await bridge.start()
        token = service.control_token
        evidence: dict[str, object] = {"schema_version": "rosclaw.uds_peercred_evidence.v1"}
        try:
            # ---- legit 子进程：bind（内核记录其真实 PID）+ 拉 context。
            real_revision0 = service.snapshot(mission.mission_id).context_revision
            legit_pid, legit_responses = await _run_client(
                bridge_sock,
                token,
                [
                    ("pi.session.bind", {
                        "pi_session_id": "pi_real",
                        "pi_session_path": "",
                        "mission_id": mission.mission_id,
                        "body_id": "sim/ur5e",
                        "execution_mode": "SIMULATION",
                    }),
                    ("pi.context", {
                        "mission_id": mission.mission_id,
                        "pi_session_id": "pi_real",
                    }),
                    ("pi.action.propose", {
                        "pi_session_id": "pi_real",
                        "mission_id": mission.mission_id,
                        "context_revision": real_revision0,
                        "body_hash": mission.body_binding.effective_body_hash,
                        "mode": "SIMULATION",
                        "idempotency_key": "idem_legit_card",
                        "context_lease_id": "@lease",
                        "capability_id": "sim_ground_truth",
                        "arguments": {},
                        "expected_effect": "legit card for attacker-execute test",
                        "risk_tier": "LOW",
                    }),
                ],
            )
            bind_resp, ctx_resp, propose_resp = legit_responses
            assert bind_resp.get("ok"), bind_resp
            # 内核记录的 writer owner_pid 必须等于 legit 子进程的真实
            # PID（subprocess.pid）——不是自报值。
            db = sqlite3.connect(service._store.database_path if hasattr(service._store, "database_path") else tmp_path / "agentd" / "missions.db")
            row = db.execute(
                "SELECT owner_pid FROM pi_session_leases WHERE pi_session_id = 'pi_real'"
            ).fetchone()
            db.close()
            assert row is not None, "bind 未产生 writer lease"
            assert row[0] == legit_pid, (
                f"writer owner_pid={row[0]} 不是内核看到的 legit PID {legit_pid}"
            )
            assert ctx_resp.get("ok"), ctx_resp
            assert ctx_resp.get("context_lease_id"), (
                f"legit writer 未获 context lease: {ctx_resp}"
            )
            evidence["legit_pid"] = legit_pid
            evidence["recorded_owner_pid_matches"] = True
            # legit 同进程建的真卡——攻击者将拿真 approval_id 尝试执行。
            assert propose_resp.get("ok"), f"legit 建卡失败: {propose_resp}"
            real_approval_id = propose_resp["card"]["approval_id"]
            if False:
                # 新进程非 writer——符合预期被拒；真卡改由 writer 身份的
                # 主进程内系统路径产出（caller=注册 owner 1/1000）。
                from rosclaw.agentd.pi_bridge.action_admission import (
                    ActionAdmissionService,
                    ActionRequestContext,
                )

                # 主进程扮演 writer——把 writer lease 迁到 owner_pid=1
                # 的注册身份（模拟 writer 进程内建卡）。
                from rosclaw.agentd.pi_bridge.session_binding import (
                    SessionBindingStore,
                )
                from tests.agentd.test_pi_tool_bridge import _issue_lease

                bindings = SessionBindingStore(service._store.connection)
                bindings.bind(
                    pi_session_id="pi_sys", pi_session_path="",
                    mission_id=mission.mission_id, body_id="sim/ur5e",
                    execution_mode="SIMULATION", created_by="user:local:1000",
                )
                bindings.acquire_lease(
                    mission_id=mission.mission_id, pi_session_id="pi_sys",
                    owner_pid=1, owner_uid=1000,
                )
                lease2 = _issue_lease(service, mission, "pi_sys")
                snap2 = service.snapshot(mission.mission_id)
                card2 = await ActionAdmissionService(service).propose(
                    request=ActionRequestContext(
                        pi_session_id="pi_sys",
                        mission_id=mission.mission_id,
                        context_revision=snap2.context_revision,
                        body_hash=mission.body_binding.effective_body_hash,
                        mode=mission.mode.value,
                        idempotency_key="idem_legit_card2",
                        context_lease_id=lease2,
                    ),
                    capability_id="sim_ground_truth",
                    arguments={},
                    expected_effect="legit card",
                    risk_tier="LOW",
                    caller_pid=1, caller_uid=1000,
                )
                real_approval_id = card2["approval_id"]

            # ---- 攻击者子进程：同 UID、不同 PID、持 token + session ID。
            # 参数完全合法（真实 revision/body hash + 偷来的 lease）——
            # 唯一挡得住它的必须是 caller 身份。
            real_revision = service.snapshot(mission.mission_id).context_revision
            real_body_hash = mission.body_binding.effective_body_hash

            def _base() -> dict:
                return {
                    "pi_session_id": "pi_real",
                    "mission_id": mission.mission_id,
                    "context_revision": real_revision,
                    "body_hash": real_body_hash,
                    "mode": "SIMULATION",
                    "idempotency_key": "idem_attack",
                    "context_lease_id": ctx_resp["context_lease_id"],
                }

            attacker_pid, attacker = await _run_client(
                bridge_sock,
                token,
                [
                    ("pi.context", {
                        "mission_id": mission.mission_id,
                        "pi_session_id": "pi_real",
                    }),
                    ("pi.action.propose", {
                        **_base(),
                        "capability_id": "sim_ground_truth",
                        "arguments": {},
                        "expected_effect": "attack",
                        "risk_tier": "LOW",
                    }),
                    ("pi.action.status", {
                        "pi_session_id": "pi_real",
                        "approval_id": "appr_nonexistent",
                    }),
                    ("pi.action.execute", {
                        **_base(),
                        "approval_id": real_approval_id,
                    }),
                ],
            )
            evidence["attacker_pid"] = attacker_pid
            a_ctx, a_propose, a_status, a_execute = attacker
            # context：观测面可读，但 action lease 绝不发给冒名进程。
            assert a_ctx.get("context_lease_denied"), (
                f"攻击者竟获签 context lease: {a_ctx}"
            )
            assert not a_ctx.get("context_lease_id")
            for name, resp in (
                ("propose", a_propose), ("status", a_status), ("execute", a_execute)
            ):
                assert not resp.get("ok"), f"攻击者 {name} 竟成功: {resp}"
            evidence["attacker_results"] = {
                "context": "context_lease_denied",
                "propose": a_propose.get("code"),
                "status": a_status.get("code"),
                "execute": a_execute.get("code"),
            }
            # status/propose/execute 的拒绝必须基于 caller 身份
            # （execute 用的是真卡——APPROVAL_NOT_FOUND 不算数）。
            for name, resp in (("propose", a_propose), ("status", a_status),
                               ("execute", a_execute)):
                assert resp.get("code") in (
                    "CALLER_MISMATCH", "WRITER_LEASE_REQUIRED", "SESSION_UNBOUND",
                    "FORBIDDEN", "CALLER_IDENTITY_REQUIRED",
                ), f"{name} 拒绝码不含 caller 身份: {resp}"
            # 零副作用。
            db = sqlite3.connect(tmp_path / "agentd" / "missions.db")
            approvals = db.execute("SELECT COUNT(*) FROM operator_requests").fetchone()[0]
            txns = db.execute("SELECT COUNT(*) FROM action_txns").fetchone()[0]
            grants = db.execute("SELECT COUNT(*) FROM mission_grants").fetchone()[0]
            db.close()
            # legit 建卡产生 1 approval + 1 txn；攻击者必须零新增、零 grant。
            assert (approvals, txns, grants) == (1, 1, 0), (
                f"副作用计数异常: approvals={approvals} txns={txns} grants={grants}"
            )
            evidence["side_effect_counts"] = {
                "operator_requests": approvals,
                "action_txns": txns,
                "mission_grants": grants,
            }
            evidence["verdict"] = "PASS"
        finally:
            (tmp_path / "uds-peercred-evidence.json").write_text(
                json.dumps(evidence, indent=1), encoding="utf-8"
            )
            await bridge.stop() if hasattr(bridge, "stop") else None
            await operatord.stop()
            await agent_server.stop()
            await service.close()


class TestLeaseFieldContracts:
    """字段级合约（六审 §5.3/§5.4）——单元层红测试。"""

    async def test_lease_binding_id_is_session_binding_not_writer_lease(
        self, tmp_path: Path
    ) -> None:
        """context lease 的 binding_id 必须是 session binding 的
        binding_id（当前错写 writer.lease_id），且 writer_lease_id
        单独成字段。"""
        service, mission, operatord, agent_server, sock = await _setup_with_operatord(
            tmp_path
        )
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer

        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        result = await bridge._dispatch(
            "user:local:1000", 1, "pi.context",
            {"token": service.control_token, "mission_id": mission.mission_id,
             "pi_session_id": "pi_1"},
        )
        assert result.get("ok") and result.get("context_lease_id"), result
        row = service._store.connection.execute(
            "SELECT binding_id, caller_uid FROM pi_context_leases "
            "WHERE context_lease_id = ?",
            (result["context_lease_id"],),
        ).fetchone()
        from rosclaw.agentd.pi_bridge.session_binding import SessionBindingStore

        bindings = SessionBindingStore(service._store.connection)
        binding = bindings.binding_for_session("pi_1")
        writer = bindings.writer_of(mission.mission_id)
        assert row["binding_id"] == binding.binding_id, (
            f"lease.binding_id={row['binding_id']} 是 writer lease ID "
            f"{writer.lease_id}，不是 session binding {binding.binding_id}"
        )
        columns = {
            r[1] for r in service._store.connection.execute(
                "PRAGMA table_info(pi_context_leases)"
            )
        }
        assert "writer_lease_id" in columns, "缺独立 writer_lease_id 字段"
        assert "caller_pid" in columns, "缺 caller_pid 字段"
        await operatord.stop()
        await agent_server.stop()
        await service.close()

    async def test_legacy_minus1_uid_lease_cannot_authorize_action(
        self, tmp_path: Path
    ) -> None:
        """legacy caller_uid=-1 的 lease 用于 propose 必须拒绝（当前
        -1 绕过 caller 比对）。"""
        from rosclaw.agentd.pi_bridge.action_admission import (
            ActionAdmissionService,
            ActionRequestContext,
        )
        from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError
        from tests.agentd.test_pi_tool_bridge import (
            SIM_ACTION_CAPABILITY,
            _issue_lease,
        )

        service, mission, operatord, agent_server, sock = await _setup_with_operatord(
            tmp_path
        )
        lease_id = _issue_lease(service, mission, "pi_1")
        # 把 lease 改成 legacy 形态（caller_uid=-1）。
        service._store.connection.execute(
            "UPDATE pi_context_leases SET caller_uid = -1 WHERE context_lease_id = ?",
            (lease_id,),
        )
        service._store.connection.commit()
        snapshot = service.snapshot(mission.mission_id)
        ctx = ActionRequestContext(
            pi_session_id="pi_1",
            mission_id=mission.mission_id,
            context_revision=snapshot.context_revision,
            body_hash=mission.body_binding.effective_body_hash,
            mode=mission.mode.value,
            idempotency_key="idem_six4_legacy",
            context_lease_id=lease_id,
        )
        with pytest.raises(ToolBridgeError) as excinfo:
            await ActionAdmissionService(service).propose(
                request=ctx,
                capability_id=SIM_ACTION_CAPABILITY,
                arguments={},
                expected_effect="legacy",
                risk_tier="LOW",
                caller_pid=1,
                caller_uid=1000,
            )
        assert excinfo.value.code in ("CALLER_MISMATCH", "LEGACY_LEASE_FORBIDDEN")
        await operatord.stop()
        await agent_server.stop()
        await service.close()
