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
import contextlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rosclaw.agentd.operator_socket import MAX_REQUEST_BYTES, _peer_credentials

if TYPE_CHECKING:
    from rosclaw.agentd.service import AgentService

from rosclaw.agentd.pi_bridge.session_binding import BindingError, SessionBindingStore
from rosclaw.task_kernel.service import TASK_ACTIVE


def default_pi_bridge_socket(home: Path | None = None) -> Path:
    base = home or Path(os.environ.get("ROSCLAW_HOME", Path.home() / ".rosclaw"))
    return base / "run" / "pi-bridge.sock"


# ----------------------------------------------------------------------
# 十四审 PR-14.4：Tasks Center 服务端投影与控制面（模块级——单测与
# socket handler 共用，一个用户任务一张卡）。
# ----------------------------------------------------------------------





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
            # 七审 §2.5：SIM 审批策略透出（readiness/UI 不再把
            # OPERATOR_OFFLINE 当成 auto SIM 的 blocker）。
            import json as _json

            sim_policy = "auto"
            safety_file = service._home / "agent" / "safety.json"
            if safety_file.exists():
                try:
                    sim_policy = str(
                        _json.loads(safety_file.read_text(encoding="utf-8")).get(
                            "sim_policy", "auto"
                        )
                    )
                except Exception:  # noqa: BLE001
                    sim_policy = "auto"
            # 七审 PR-SEVEN-5：机器人友好名 + kit 摘要——UI 默认显示
            # display_name，不再只显示内部 body_id。
            kit_status = await service.robot_kit_status()
            return {
                "ok": True,
                "agentd": "READY",
                "authorization_profile": service.authorization_profile(),
                "sim_policy": sim_policy,
                "body_id": service._body_id,
                "body_display": kit_status.get("display_name") or service._body_id,
                "robot_kit": kit_status,
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
        if method == "pi.safety.get":
            import json as _json

            safety_file = service._home / "agent" / "safety.json"
            sim_policy = "auto"
            if safety_file.exists():
                try:
                    sim_policy = str(
                        _json.loads(safety_file.read_text(encoding="utf-8")).get(
                            "sim_policy", "auto"
                        )
                    )
                except Exception:  # noqa: BLE001
                    sim_policy = "auto"
            return {"ok": True, "sim_policy": sim_policy}
        if method == "pi.safety.set":
            import json as _json

            policy = str(params.get("sim_policy", ""))
            if policy not in ("auto", "ask"):
                return {"ok": False, "error": "sim_policy must be auto|ask",
                        "code": "INVALID_ARGUMENT"}
            safety_file = service._home / "agent" / "safety.json"
            safety_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = safety_file.with_suffix(".tmp")
            tmp.write_text(
                _json.dumps({"sim_policy": policy}, indent=1), encoding="utf-8"
            )
            import os as _os

            _os.chmod(tmp, 0o600)
            tmp.replace(safety_file)
            return {"ok": True, "sim_policy": policy}
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
            # 七审 §6 PR-SEVEN-6：强制有效 mission + developer 剖面 +
            # SIMULATION mode——空/未知 mission 一律拒绝（否则可能启动
            # 带 --no-human-presence-check 的 operatord 而无 scope 约束）。
            mission_id = str(params.get("mission_id", ""))
            if not mission_id:
                return {
                    "ok": False,
                    "error": "mission_id required for operator bootstrap (fail closed)",
                    "code": "MISSION_REQUIRED",
                }
            mission = service.get_mission(mission_id)
            if mission is None:
                return {
                    "ok": False,
                    "error": f"unknown mission {mission_id!r}",
                    "code": "MISSION_NOT_FOUND",
                }
            if mission.mode.value != "SIMULATION":
                return {
                    "ok": False,
                    "error": "operator bootstrap 仅限 SIMULATION developer——"
                    "REAL/SHADOW 要求独立 operator readiness/presence 流程",
                    "code": "MODE_FORBIDDEN",
                }
            if service.authorization_profile() != "DEV_SIM_ONLY":
                return {
                    "ok": False,
                    "error": "operator bootstrap 仅限 developer profile",
                    "code": "PROFILE_FORBIDDEN",
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
        if method == "pi.turn.record":
            # 九审 §6.1：用户输入先落账——interactive 以外来源拒绝。
            from rosclaw.agentd.turn_store import TurnStore

            session_id = str(params.get("pi_session_id", ""))
            text = str(params.get("text", ""))
            source = str(params.get("source", ""))
            if not session_id or not text:
                return {"ok": False, "error": "session/text required",
                        "code": "INVALID_ARGUMENT"}
            mission_id = ""
            binding = self._bindings.binding_for_session(session_id)
            if binding is not None:
                mission_id = binding.mission_id
            try:
                turn = TurnStore(service._store.connection).record(
                    pi_session_id=session_id,
                    mission_id=mission_id,
                    text=text,
                    source=source,
                )
            except ValueError as exc:
                return {"ok": False, "error": str(exc), "code": "SOURCE_FORBIDDEN"}
            return {"ok": True, "turn": turn}
        if method == "pi.intent.route":
            # WP-P0-5（总纲 §7.1）：确定性 Intent Router——已知任务
            # 零模型回合；未命中诚实 None（交模型路径）。
            from rosclaw.agentd.intent_router import route_intent

            return {"ok": True, "spec": route_intent(str(params.get("text", "")))}
        if method == "pi.robot.list":
            # 七审 PR-SEVEN-5：第一方 kit 清单（/robots）。
            return await service.robot_list()
        if method == "pi.robot.resolve":
            # 七审 PR-SEVEN-5：自然语言 Robot Resolver（唯一候选自动选）。
            return service.robot_resolve(str(params.get("query", "")))
        if method == "pi.robot.repair":
            # 七审 PR-SEVEN-5：一键修复（幂等；不触碰 REAL 授权）。
            return await service.robot_repair(str(params.get("kit_id", "")))
        if method == "pi.robot.use":
            # 七审 PR-SEVEN-5：切换活跃机器人（无 kit 的 body 一律拒绝）。
            return await service.robot_use(str(params.get("body_id", "")))
        if method == "pi.session.resume_report":
            # WP-P0-3（总纲 §5.4）：恢复对账报告——恢复了什么、重新
            # 验证了什么、哪些权限失效了。已完成任务绝不重放；运行
            # 中任务只 attach；过期卡标 REAUTH。
            import json as _json

            session_id = str(params.get("pi_session_id", ""))
            binding = self._bindings.binding_for_session(session_id)
            lines: list[str] = []
            verdict = "RESUMED"
            mode = ""
            body_id = ""
            if binding is None:
                return {
                    "ok": True,
                    "report": {
                        "verdict": "READ_ONLY",
                        "lines": ["无绑定记录——对话只读打开，不伪装成原 Mission"],
                        "mode": "", "body_id": "",
                    },
                }
            mission = service.get_mission(binding.mission_id)
            if mission is None:
                return {
                    "ok": True,
                    "report": {
                        "verdict": "READ_ONLY",
                        "lines": [
                            "原 Mission 已不存在——对话只读打开；"
                            "可从此会话新建 SIM 任务（不伪装原 Mission）"
                        ],
                        "mode": "", "body_id": "",
                    },
                }
            mode = mission.mode.value
            body_id = mission.body_binding.body_id
            conn = service._store.connection
            # 任务对账（PR-H9：TaskKernel 权威——task_records 已删）。
            kernel_tasks = service._task_kernel.list_tasks(mission.mission_id)
            for task in kernel_tasks[:5]:
                task_id = str(task["task_id"])
                state = str(task["state"])
                if state == "SUCCEEDED":
                    lines.append(f"任务 {task_id} 已验收——不会重新执行")
                elif state == "BLOCKED":
                    lines.append(
                        f"任务 {task_id} 受阻（{str(task.get('block_reason') or '')[:60]}）"
                        "——恢复后需用户决策，不自动重试"
                    )
                elif state in TASK_ACTIVE:
                    lines.append(f"任务 {task_id} 进行中（{state}）——继续对话即恢复")
                else:
                    lines.append(f"任务 {task_id}：{state}")
            # 过期 PENDING 卡（broker 侧）→ REAUTH_NEEDED（恢复不复活
            # 任何授权）。
            from datetime import UTC as _UTC2
            from datetime import datetime as _dt2

            now_iso = _dt2.now(_UTC2).isoformat()
            expired_pending = conn.execute(
                "SELECT request_id FROM operator_requests WHERE status = 'PENDING' "
                "AND json_extract(request_json, '$.expires_at') < ?",
                (now_iso,),
            ).fetchall()
            if expired_pending:
                verdict = "REAUTH_NEEDED"
                lines.append(
                    f"{len(expired_pending)} 张授权卡已过期——需重新确认，"
                    "不会自动恢复执行权"
                )
            # 权限行：恢复后 lease 重新获取；旧动作授权按规则失效。
            lines.append(
                "权限：writer lease 已重新获取；旧动作授权不随恢复复活"
                f"（当前模式 {mode}，仿真策略见 /safety）"
            )
            return {
                "ok": True,
                "report": {
                    "verdict": verdict,
                    "lines": lines,
                    "mode": mode,
                    "body_id": body_id,
                },
            }
        if method == "pi.task.list":
            # PR-H9：TaskKernel 权威（task_records 已删）。
            mission_id = str(params.get("mission_id", ""))
            kernel = service._task_kernel
            return {"ok": True, "tasks": kernel.list_tasks(mission_id)}
        if method == "pi.task.trace":
            # PR-H9：/trace——TaskKernel 审计链（task/revision/事件/
            # 验证/产物——task_records 已删）。
            task_id = str(params.get("task_id", ""))
            kernel = service._task_kernel
            task = kernel.get_task(task_id)
            if task is None:
                return {"ok": False, "error": f"unknown task {task_id!r}", "code": "TASK_NOT_FOUND"}
            conn = service._store.connection
            revisions = [
                dict(r)
                for r in conn.execute(
                    "SELECT revision, goal_delta, created_at FROM "
                    "task_revisions WHERE task_id = ? ORDER BY revision",
                    (task_id,),
                ).fetchall()
            ]
            events = [
                dict(r)
                for r in conn.execute(
                    "SELECT seq, event_type, payload_json, created_at FROM "
                    "task_events WHERE task_id = ? ORDER BY seq LIMIT 200",
                    (task_id,),
                ).fetchall()
            ]
            verifications = [
                dict(r)
                for r in conn.execute(
                    "SELECT verification_id, revision, status, checks_json, "
                    "created_at FROM verifications WHERE task_id = ? "
                    "ORDER BY created_at",
                    (task_id,),
                ).fetchall()
            ]
            artifacts = [
                dict(r)
                for r in conn.execute(
                    "SELECT artifact_id, path, media_type, sha256, size_bytes "
                    "FROM artifacts WHERE task_id = ?",
                    (task_id,),
                ).fetchall()
            ]
            return {
                "ok": True,
                "trace": {
                    "task": task,
                    "revisions": revisions,
                    "events": events,
                    "verifications": verifications,
                    "artifacts": artifacts,
                },
            }
        if method == "pi.context.checkpoint":
            # 八审 §5：EmbodiedCheckpointV1——从权威存储重建（LLM
            # compaction 摘要永远不是安全状态权威）。
            mission_id = str(params.get("mission_id", ""))
            mission = service.get_mission(mission_id)
            if mission is None:
                return {"ok": False, "error": "unknown mission", "code": "MISSION_NOT_FOUND"}
            conn = service._store.connection
            kernel_tasks = service._task_kernel.list_tasks(mission_id)
            nonterminal = [
                (task["task_id"], task["root_goal"], task["state"])
                for task in kernel_tasks
                if task["state"] in TASK_ACTIVE
            ]
            recent = [
                (task["task_id"], task["root_goal"], task["state"],
                 f"rev-{task['active_revision']}")
                for task in kernel_tasks[:5]
            ]
            pending = conn.execute(
                "SELECT request_id FROM operator_requests WHERE status = 'PENDING'"
            ).fetchall()
            receipts = conn.execute(
                "SELECT payload_json FROM agent_events "
                "WHERE type = 'receipt.received' ORDER BY rowid DESC LIMIT 3"
            ).fetchall()
            import json as _json3

            sim_policy = "auto"
            safety_file = service._home / "agent" / "safety.json"
            if safety_file.exists():
                try:
                    sim_policy = str(
                        _json3.loads(safety_file.read_text(encoding="utf-8")).get(
                            "sim_policy", "auto"
                        )
                    )
                except Exception:  # noqa: BLE001
                    sim_policy = "auto"
            return {
                "ok": True,
                "checkpoint": {
                    "schema_version": "rosclaw.embodied_checkpoint.v1",
                    "mission_id": mission_id,
                    "goal": mission.goal.text if hasattr(mission.goal, "text") else str(mission.goal),
                    "mode": mission.mode.value,
                    "body_id": mission.body_binding.body_id,
                    "nonterminal_tasks": [
                        {"task_id": r[0], "goal": r[1], "state": r[2]} for r in nonterminal
                    ],
                    "recent_tasks": [
                        {"task_id": r[0], "goal": r[1], "state": r[2], "plan_id": r[3]}
                        for r in recent
                    ],
                    "pending_approvals": [r[0] for r in pending],
                    "recent_receipt_refs": [
                        _json3.loads(r[0]).get("receipt_id") for r in receipts
                    ],
                    "sim_policy": sim_policy,
                },
            }
        if method == "pi.task.cancel":
            # PR-H9：TaskKernel 权威（终态不可逆由 transition 保证）。
            task_id = str(params.get("task_id", ""))
            task = service._task_kernel.get_task(task_id)
            if task is None:
                return {"ok": False, "error": f"unknown task {task_id!r}",
                        "code": "TASK_NOT_FOUND"}
            service._task_kernel.transition(
                task_id, "CANCELLED", reason="user_cancel"
            )
            return {"ok": True, "task_id": task_id, "state": "CANCELLED"}
        if method == "pi.doctor.task":
            # 七审 PR-SEVEN-5：task readiness（/doctor task <goal>）。
            return await service.doctor_task(str(params.get("goal", "")))
        if method == "pi.usage":
            # 八审 §4 P0-8：/tokens 的用量面（provider 请求/token 分项/
            # 工具计数/延迟分离）。
            mission_id = str(params.get("mission_id", ""))
            if mission_id and service.get_mission(mission_id) is None:
                return {"ok": False, "error": "unknown mission", "code": "MISSION_NOT_FOUND"}
            return {"ok": True, "usage": service.usage_report(mission_id)}
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
            # 七审 §2.2：按 execution_class 精确分桶——不再是"非
            # PHYSICAL_ACTION 就算 observation"（COMPUTE 的 sim_reach
            # 曾被错列为只读观测）。
            observation: list[dict[str, Any]] = []
            compute: list[dict[str, Any]] = []
            actions: list[dict[str, Any]] = []
            excluded: list[dict[str, Any]] = []
            sim_executor_sources = set(service._sim_executors.keys())
            for descriptor in service._tool_catalog.list():
                cls = descriptor.execution_class.value
                if cls == "OBSERVE":
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
                if cls == "COMPUTE":
                    compute.append(
                        {
                            "capability_id": descriptor.tool_id,
                            "version": descriptor.version,
                            "source": descriptor.source,
                            "description": descriptor.description[:120],
                            "effect_domain": "none",
                        }
                    )
                    continue
                if cls != "PHYSICAL_ACTION":
                    continue  # CONTROL/DELEGATE 不混入能力面
                reason = check_body_compatibility(descriptor, body_id)
                quarantine = service._tool_catalog.quarantine_reason(descriptor.tool_id)
                if quarantine and reason is None:
                    reason = "CAPABILITY_QUARANTINED"
                if mission.mode.value not in list(descriptor.supported_modes):
                    reason = reason or "MODE_FORBIDDEN"
                executor_state = (
                    "READY"
                    if descriptor.source in sim_executor_sources
                    else "MISSING"
                )
                entry = {
                    "capability_id": descriptor.tool_id,
                    "version": descriptor.version,
                    "source": descriptor.source,
                    "risk_tier": descriptor.risk_tier,
                    "side_effect_class": descriptor.side_effect_class.value,
                    "description": descriptor.description[:120],
                    # 七审 §6 PR-SEVEN-2.3：每条动作带 effect_domain/
                    # executor_state/body_compatibility。
                    "effect_domain": (
                        "simulation"
                        if mission.mode.value == "SIMULATION"
                        else "real"
                    ),
                    "executor_state": executor_state,
                    "body_compatibility": reason is None,
                }
                if reason is None and executor_state == "READY":
                    actions.append(entry)
                else:
                    excluded.append(
                        {
                            **entry,
                            "reason": reason or "EXECUTOR_FOR_BODY_UNAVAILABLE",
                        }
                    )
            return {
                "ok": True,
                "body_id": body_id,
                "mode": mission.mode.value,
                "observation_capabilities": observation,
                "compute_capabilities": compute,
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
            import os as _os
            if _os.environ.get("ROSCLAW_DEBUG_CONTEXT"):
                with contextlib.suppress(Exception):
                    (service._home / "agentd" / "ctx-at-issue.json").write_text(
                        envelope.model_dump_json(indent=1), encoding="utf-8"
                    )
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
        if method == "pi.task.bind":
            # PR-H2（ADR-0012）：输入事务——用户消息先落账+绑定 root
            # task，再投递 Harness（幽灵执行防线的权威端）。
            kernel = service._task_kernel
            result = kernel.bind_message(
                mission_id=str(params.get("mission_id", "")),
                session_ref=str(params.get("session_ref", "")),
                backend_native_id=str(params.get("backend_native_id", "")),
                message_id=str(params.get("message_id", "")),
                text=str(params.get("text", "")),
                cwd=str(params.get("cwd", "")),
                mode=str(params.get("mode", "SIMULATION")),
                # PR-N1：任务 workspace = ActiveTaskContext 的
                # workspaceRoot（cwd 即工作根——同一事实源）。
                workspace_root=str(params.get("cwd", "")),
                body_id=str(params.get("body_id", "")),
                force_new=bool(params.get("force_new", False)),
            )
            return {"ok": True, **result}
        if method == "pi.kernel.list":
            kernel = service._task_kernel
            return {
                "ok": True,
                "tasks": kernel.list_tasks(str(params.get("mission_id", ""))),
            }
        if method == "pi.kernel.get":
            kernel = service._task_kernel
            task = kernel.get_task(str(params.get("task_id", "")))
            if task is None:
                return {"ok": False, "error": "unknown task"}
            conn = service._store.connection
            revisions = [
                dict(r)
                for r in conn.execute(
                    "SELECT revision, goal_delta, created_at FROM "
                    "task_revisions WHERE task_id = ? ORDER BY revision",
                    (task["task_id"],),
                ).fetchall()
            ]
            events = [
                dict(r)
                for r in conn.execute(
                    "SELECT seq, event_type, payload_json, created_at FROM "
                    "task_events WHERE task_id = ? ORDER BY seq DESC LIMIT 20",
                    (task["task_id"],),
                ).fetchall()
            ]
            return {"ok": True, "task": task, "revisions": revisions,
                    "events": events}
        if method == "pi.kernel.active":
            # PR-H4：TurnGuard 用——当前 session 的活跃 task（无则 null）。
            task = service._task_kernel.active_task_for(
                str(params.get("mission_id", "")),
                str(params.get("session_ref", "")),
            )
            return {"ok": True, "task": task}
        if method == "pi.kernel.accept":
            # PR-N0：/done 用户接受（与系统验收 accepted_at 区分）。
            try:
                service._task_kernel.accept_task(str(params.get("task_id", "")))
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True}
        if method == "pi.kernel.transition":
            kernel = service._task_kernel
            kernel.transition(
                str(params.get("task_id", "")),
                str(params.get("state", "")),
                reason=str(params.get("reason", "")),
            )
            return {"ok": True}
        if method == "pi.inspect":
            # PR-N3：生态索引自检（self/robot/capability/asset）——
            # 程序探测，不靠模型猜。
            from rosclaw.cognition.index.query import robot_chain, search
            from rosclaw.cognition.inspect_cli import ensure_index, inspect_self

            kind = str(params.get("kind", "self"))
            query = str(params.get("query", ""))
            if kind == "self":
                return {"ok": True, "info": inspect_self(service._home)}
            idx = ensure_index(service._home)
            if kind == "robot":
                chain = robot_chain(idx, query)
                if chain is None:
                    return {"ok": False, "error": f"未知机器人 {query!r}"}
                return {"ok": True, "chain": chain}
            hits = search(idx, query or kind, limit=20)
            return {"ok": True, "hits": hits}
        if method == "pi.kernel.artifacts":
            # PR-H8：/artifacts——产物账本（登记才算交付，模型口头提到
            # 不算）。
            conn = service._store.connection
            rows = conn.execute(
                "SELECT artifact_id, task_id, path, media_type, sha256, "
                "size_bytes, created_at FROM artifacts WHERE task_id = ? "
                "ORDER BY created_at",
                (str(params.get("task_id", "")),),
            ).fetchall()
            return {"ok": True, "artifacts": [dict(r) for r in rows]}
        if method == "pi.op.start":
            # PR-H3（§11.4）：长 operation 立即返回 operation_id——
            # 进度/输出经 task_events 流，终态经 followUp 注入一次。
            ops = service._operation_manager
            argv = [str(a) for a in (params.get("argv") or [])]
            if not argv:
                return {"ok": False, "error": "argv required"}
            op = await ops.start(
                task_id=str(params.get("task_id", "")),
                attempt_id=str(params.get("attempt_id", "") or "main"),
                kind=str(params.get("kind", "process")),
                argv=argv,
                cwd=str(params.get("cwd", "") or "") or None,
            )
            return {"ok": True, "operation": op}
        if method == "pi.op.get":
            op = service._operation_manager.get(str(params.get("operation_id", "")))
            return {"ok": bool(op), "operation": op}
        if method == "pi.op.cancel":
            await service._operation_manager.cancel(
                str(params.get("operation_id", ""))
            )
            return {"ok": True}
        if method == "pi.kernel.events":
            # seq 重放（断线从 last_seq+1——不重不漏）。
            events = service._operation_manager.events_since(
                str(params.get("task_id", "")),
                int(params.get("after_seq", 0) or 0),
            )
            return {"ok": True, "events": events}
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
