"""内置 Pi headless Worker adapter（十审 W1，审计 §8.5）。

把 ROSClaw 自带的 `rosclaw-agent worker --headless` 子进程变成受管
Worker：

- 与主 Agent 共用 agentDir/auth.json/models.json——不需要第二份
  API key（WorkOrder 只携带无 secret 的模型快照）；
- stdout JSONL WorkerEvent 流式解析——心跳是子进程真实事件
  （attempt_started/tool_started/usage/...），不是 supervisor 轮询伪造；
- start_new_session 独立进程组；cancel = 整组 SIGTERM→SIGKILL；
- startup/idle timeout：子进程必须在限定时间内产生真实事件，否则
  诚实失败（绝不无限"Working…"）。
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import stat
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from rosclaw.agentd.pi_entry import find_pi_agent_entry
from rosclaw.agentd.workers.adapter import (
    AdapterError,
    RunHandle,
    WorkerProbeResult,
)
from rosclaw.agentd.workers.process import kill_process_tree
from rosclaw.contracts.worker.control import ERROR_CODE_CAUSES
from rosclaw.contracts.worker.order import (
    ResultArtifact,
    ResultClaim,
    WorkOrderV1,
    WorkResultV1,
    WorkUsage,
)

WORKER_ID = "worker:rosclaw:pi"

#: 十审 §7.4 P0 默认：startup 30s（十一审 PR-A：只管启动握手）。
#: 十一审 PR-A 三层分离：
#: - LIVENESS_TIMEOUT：连 liveness 事件都没有才判死（进程真挂死）；
#: - STALL_WARN：语义静默只告警不杀（高 thinking/长命令合法）；
#: - wall/token 预算到期先 wrap-up steer，grace 后再终止。
STARTUP_TIMEOUT_SEC = 30.0
LIVENESS_TIMEOUT_SEC = 15.0
STALL_WARN_SEC = 90.0
WRAPUP_GRACE_SEC = 60.0
#: UNREACHABLE 宽限：全静默 + 零 CPU 进展持续超过该值才按"恢复失败"
#: 终止并进 INTERRUPTED_RESUMABLE（working 永不杀）。
UNREACHABLE_GRACE_SEC = 900.0
#: 十四审 PR-14.1：控制请求 ACK 等待——supervisor 收到 control.ack
#: 前只能显示 PAUSE_REQUESTED，不得乐观落 PAUSED（总纲 §3.2）。
CONTROL_ACK_TIMEOUT_SEC = 30.0
#: 控制取消的优雅退出宽限（ACK/termination.json 落盘）再升级 SIGKILL。
CANCEL_GRACE_SEC = 5.0

#: W3：写能力 profile（Developer Workbench）——workspace 隔离 + diff 工件。
WORKBENCH_PROFILES = ("developer", "sim-builder")


def _probe_process(pid: int) -> str:
    """十三审 HOTFIX-13.2：多信号判活——/proc 存在 + CPU 时间在推进 =
    working；/proc 消失 = dead；存在但 CPU 不动 = alive（疑似挂起）。
    事件管道静默绝不单独判死。"""

    stat_path = f"/proc/{pid}/stat"
    try:
        with open(stat_path) as fh:
            parts1 = fh.read().split()
        cpu1 = int(parts1[13]) + int(parts1[14])
    except (OSError, ValueError, IndexError):
        return "dead"
    import time as _time

    _time.sleep(0.3)
    try:
        with open(stat_path) as fh:
            parts2 = fh.read().split()
        cpu2 = int(parts2[13]) + int(parts2[14])
    except (OSError, ValueError, IndexError):
        return "dead"
    return "working" if cpu2 > cpu1 else "alive"


async def _git(*args: str, cwd: str | None = None) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=cwd,
    )
    out, _ = await proc.communicate()
    return proc.returncode or 0, out.decode(errors="replace")


class PiManagedAdapter:
    """内置 Pi Worker（每个 WorkOrder 一个 headless 子进程）。"""

    def __init__(self, *, rosclaw_home: Path, conn=None) -> None:
        self._home = Path(rosclaw_home)
        self._conn = conn  # PR-D：订单 inputs 回写（可选——无则跳过）
        # PR-E：WAITING_INPUT 状态迁移需要 manager（service 装配后注入）。
        self._manager_ref = None
        self._runs: dict[str, tuple[RunHandle, asyncio.Task]] = {}
        # W4：活动子进程（steer 通道 + pid 文件崩溃对账）。
        self._procs: dict[str, asyncio.subprocess.Process] = {}
        # 十四审 PR-14.1：control.ack 收条（control_id → state）——
        # request_pause/resume 只认 ACK，不认"stdin 已写"。
        self._control_acks: dict[str, dict[str, str]] = {}
        # 十一审 PR-B：持久化事件账本（文件权威，重启/compact 可读）。
        from rosclaw.agentd.workers.event_store import WorkerEventStore

        self._events = WorkerEventStore(rosclaw_home)

    async def probe(self, worker_id: str | None = None) -> WorkerProbeResult:  # noqa: ARG002
        runtime = find_pi_agent_entry()
        if runtime is None:
            return WorkerProbeResult(
                ready=False,
                detail="rosclaw-agent dist 或 Node ≥22.19 不可用——内置 Worker 未就绪",
            )
        return WorkerProbeResult(ready=True, detail="builtin pi worker ready")

    # ------------------------------------------------------------------
    async def start(self, order: WorkOrderV1, credential_refs: dict) -> RunHandle:  # noqa: ARG002
        if order.lease is None:
            raise AdapterError("work order has no lease")
        runtime = find_pi_agent_entry()
        if runtime is None:
            raise AdapterError("rosclaw-agent dist/Node 不可用（内置 Worker 未就绪）")
        node, entry = runtime
        handle = RunHandle(
            work_order_id=order.work_order_id,
            lease_id=order.lease.lease_id,
            worker_id=WORKER_ID,
        )
        task = asyncio.create_task(self._run(order, handle, node, entry))
        self._runs[order.work_order_id] = (handle, task)
        return handle

    async def poll(self, handle: RunHandle) -> RunHandle | WorkResultV1:
        entry = self._runs.get(handle.work_order_id)
        if entry is None:
            raise AdapterError(f"unknown handle {handle.work_order_id}")
        stored, task = entry
        if not task.done():
            return stored
        self._runs.pop(handle.work_order_id, None)
        try:
            return task.result()
        except asyncio.CancelledError:
            return WorkResultV1(
                work_order_id=handle.work_order_id,
                worker_id=WORKER_ID,
                lease_id=handle.lease_id,
                status="CANCELLED",
                summary="worker attempt cancelled",
                warnings=["cancelled"],
            )
        except Exception as exc:  # noqa: BLE001 - worker failure is data
            return WorkResultV1(
                work_order_id=handle.work_order_id,
                worker_id=WORKER_ID,
                lease_id=handle.lease_id,
                status="FAILED",
                summary=f"{type(exc).__name__}: {exc}",
                warnings=["adapter_error"],
            )

    async def cancel(self, handle: RunHandle, reason: str) -> None:
        # 十四审 PR-14.1：先控制取消——worker 优雅 abort 并落
        # termination.json(USER_CANCELLED)；宽限内未退出才取消驱动
        # 任务（_teardown 杀树兜底）。只有用户取消产生 CANCELLED。
        control_id = await self._send_control(handle.work_order_id, "cancel", reason)
        if control_id is not None:
            await self._wait_ack(
                handle.work_order_id, control_id, "CANCELLED",
                timeout=CANCEL_GRACE_SEC,
            )
            proc = self._procs.get(handle.work_order_id)
            if proc is not None and proc.returncode is None:
                # ACK 已回——给 termination.json/exit 一个落盘窗口。
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
            proc = self._procs.get(handle.work_order_id)
            if proc is not None and proc.returncode is not None:
                # 优雅退出——驱动任务自然收尾（post-exit 映射 CANCELLED）。
                return
        entry = self._runs.pop(handle.work_order_id, None)
        if entry is not None:
            entry[1].cancel()

    async def steer(self, work_order_id: str, note: str) -> bool:
        """W4：向运行中的 Worker 发送 steer（stdin JSONL）。进程不在
        （或 stdin 已闭）返回 False——调用方据此诚实降级。"""
        proc = self._procs.get(work_order_id)
        if proc is None or proc.returncode is not None or proc.stdin is None:
            return False
        try:
            proc.stdin.write(
                (json.dumps({"type": "steer", "text": note}, ensure_ascii=False) + "\n").encode()
            )
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            return False
        return True

    def _read_termination(self, work_order_id: str) -> dict | None:
        """termination.json（worker 退出前原子落盘的权威终止原因）。"""
        path = self._home / "work" / work_order_id / "termination.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    async def _send_control(self, work_order_id: str, action: str, reason: str = "") -> str | None:
        """写 control.request（十四审 PR-14.1）——返回 control_id；
        进程不在/stdin 已闭返回 None。"""
        proc = self._procs.get(work_order_id)
        if proc is None or proc.returncode is not None or proc.stdin is None:
            return None
        control_id = f"ctl_{uuid4().hex[:12]}"
        try:
            proc.stdin.write(
                (json.dumps(
                    {
                        "type": "control.request",
                        "control_id": control_id,
                        "action": action,
                        "mode": "safe",
                        "reason": reason,
                    },
                    ensure_ascii=False,
                ) + "\n").encode()
            )
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            return None
        return control_id

    async def _wait_ack(
        self, work_order_id: str, control_id: str, want: str,
        timeout: float | None = None,
    ) -> bool:
        """等 control.ack（state==want）——ACK 是唯一"已生效"证据。
        进程已退出时仍给事件读者 2s 排水窗口：worker 可能在 ACK 后
        立即完成退出（resume→秒回→attempt_finished 的竞态）。"""
        deadline = asyncio.get_running_loop().time() + (
            timeout if timeout is not None else CONTROL_ACK_TIMEOUT_SEC
        )
        while asyncio.get_running_loop().time() < deadline:
            acks = self._control_acks.get(work_order_id, {})
            if acks.get(control_id) == want:
                return True
            proc = self._procs.get(work_order_id)
            if proc is not None and proc.returncode is not None:
                # 进程已死：ack 只能来自尚未排水的 stdout——收窄窗口。
                deadline = min(deadline, asyncio.get_running_loop().time() + 2.0)
            await asyncio.sleep(0.05)
        return False

    async def request_pause(self, work_order_id: str, *, reason: str = "user") -> bool:
        """十四审 PR-14.1：控制暂停——发送 control.request pause 并等待
        control.ack PAUSED（模型循环真实停止、进程存活）才返回 True。"""
        control_id = await self._send_control(work_order_id, "pause", reason)
        if control_id is None:
            return False
        return await self._wait_ack(work_order_id, control_id, "PAUSED")

    async def request_resume(self, work_order_id: str) -> bool:
        """控制恢复——ACK RUNNING 后同一会话继续。"""
        control_id = await self._send_control(work_order_id, "resume")
        if control_id is None:
            return False
        return await self._wait_ack(work_order_id, control_id, "RUNNING")

    async def pause(self, work_order_id: str) -> bool:
        """兼容旧调用点——语义即 request_pause（ACK 后才算暂停）。"""
        return await self.request_pause(work_order_id, reason="budget_hard")

    async def extend(self, work_order_id: str, add_tokens: int) -> bool:  # noqa: ARG002
        """十三审：/job extend——追加预算并唤醒。十四审：唤醒 = 控制
        resume（同一 Pi 会话继续）；预算记账由 dispatcher 落 order_json。"""
        return await self.request_resume(work_order_id)

    def _on_waiting_input(self, work_order_id: str) -> None:
        manager = self._manager_ref
        if manager is not None:
            manager._transition(work_order_id, "BLOCKED", "waiting_input")

    def _on_answered(self, work_order_id: str) -> None:
        manager = self._manager_ref
        if manager is not None:
            manager._transition(work_order_id, "RUNNING", "answer_received")

    async def answer(self, work_order_id: str, text: str) -> bool:
        """十一审 PR-E：WAITING_INPUT 的用户回答（stdin JSONL）。"""
        proc = self._procs.get(work_order_id)
        if proc is None or proc.returncode is not None or proc.stdin is None:
            return False
        try:
            proc.stdin.write(
                (json.dumps({"type": "answer", "text": text}, ensure_ascii=False) + "\n").encode()
            )
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            return False
        return True

    async def reconcile(self, idempotency_key: str) -> str:  # noqa: ARG002
        for _handle, task in self._runs.values():
            if not task.done():
                return "running"
        return "not_found"

    async def _collect_partial_note(
        self, order: WorkOrderV1, workspace: str | None, base_ref: str | None
    ) -> str:
        """十二审 PR-12.5（§7.3）：失败/取消前的部分成果回收——workbench
        单收集已有 diff/媒体工件，返回一句可展示说明（无则空）。"""
        if str(order.inputs.get("worker_profile") or "") not in WORKBENCH_PROFILES:
            return ""
        if not workspace:
            return ""
        try:
            artifacts, _claims, notes = await self._collect_workbench_artifacts(
                order, workspace, base_ref
            )
            if not artifacts:
                return ""
            names = []
            work_dir = self._home / "work" / order.work_order_id / "artifacts"
            for a in artifacts:
                names.append(a.ref.rsplit(":", 1)[-1][:12])
            return (
                f"已回收部分成果 {len(artifacts)} 件（{work_dir}）"
                + (f"；{notes}" if notes else "")
            )
        except Exception:  # noqa: BLE001 - partial 回收失败不阻塞终态
            return ""

    def _write_checkpoint(
        self, order: WorkOrderV1, state: dict, outcome: str, partial_note: str
    ) -> None:
        """十二审 PR-12.5：checkpoint.json + terminal state——任何异常
        路径都有可恢复入口（session_file/partial/note）。"""
        checkpoint = {
            "work_order_id": order.work_order_id,
            "outcome": outcome,
            "phase": state.get("phase", ""),
            "last_semantic_seq": order.inputs.get("_last_seq", ""),
            "session_file": state.get("session_file", ""),
            "partial": partial_note,
            "resumable": bool(state.get("session_file")),
        }
        try:
            path = self._home / "work" / order.work_order_id / "checkpoint.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self._events.write_state(
                order.work_order_id,
                {
                    "status": outcome,
                    "phase": "TERMINAL",
                    "session_file": state.get("session_file", ""),
                    "partial": partial_note,
                },
            )
        except OSError:
            pass
    async def _preflight(self, order: WorkOrderV1) -> None:
        """十二审 PR-12.4（§5.3）+ 十三审 PR-13.6：artifact_build/
        simulation_run 的依赖预检——必须用 Worker 真实执行面（PATH 的
        python3，与 workbench bash 同一 env），不是 agentd 自己的 venv
        （用户实证：agentd venv 有 PIL 但 Worker bash 没有，跑到
        No module named 'PIL' 才失败）。失败即 BLOCKED_PREFLIGHT。"""
        task_type = str(order.inputs.get("task_type") or "")
        if task_type not in ("artifact_build", "simulation_run"):
            return
        import shutil

        problems = []
        has_ffmpeg = shutil.which("ffmpeg") is not None

        has_pil = await self._python_has_module("PIL")
        if not has_ffmpeg and not has_pil:
            problems.append(
                "Worker 环境（PATH python3）无媒体编码器：ffmpeg 与 Pillow"
                " 均不可用——请安装 ffmpeg 或让系统 python3 可 import PIL"
            )
        # 渲染/绘图任务常用 matplotlib——缺失提前报（可选依赖，不阻断，
        # 但写进事件供用户/模型知晓）。
        if not await self._python_has_module("matplotlib"):
            problems_note = "（提示：PATH python3 无 matplotlib——绘图任务可能失败）"
            self._events.append_event(
                order.work_order_id, "", "preflight_note",
                {"note": problems_note},
            )
        ws = str(order.inputs.get("workspace") or "")
        if ws and not os.access(ws, os.W_OK):
            problems.append(f"workspace 不可写: {ws}")
        if problems:
            raise AdapterError("BLOCKED_PREFLIGHT: " + "；".join(problems))

    async def _python_has_module(self, module: str) -> bool:
        """Worker 执行面（PATH python3 + workbench env）探测模块——
        不用 agentd 自己的 venv。"""
        try:
            proc = await asyncio.create_subprocess_exec(
                "python3", "-c", f"import {module}",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                env={
                    k: os.environ[k]
                    for k in ("PATH", "HOME", "LANG", "LC_ALL", "TZ")
                    if k in os.environ
                },
            )
            await asyncio.wait_for(proc.wait(), timeout=10)
            return proc.returncode == 0
        except (OSError, TimeoutError):
            return False

    async def _validate_deliverables(
        self, order: WorkOrderV1, workspace: str
    ) -> list[str]:
        """required deliverable 硬验收（存在/非空/魔数）。返回失败原因
        列表（空=全过）。"""
        from rosclaw.contracts.worker.workspec import validate_media_file

        deliverables = order.inputs.get("deliverables") or []
        task_type = str(order.inputs.get("task_type") or "")
        required_types: list[str] = []
        for d in deliverables:
            if isinstance(d, dict) and d.get("required", True):
                required_types.extend(d.get("media_types") or [])
        if task_type == "artifact_build" and not required_types:
            required_types = ["image/gif", "video/mp4"]
        if task_type == "simulation_run" and not required_types:
            required_types = ["application/json", "image/gif", "video/mp4"]
        if not required_types:
            return []
        changed = await self._changed_files(workspace, str(order.inputs.get("base_sha") or "") or None)
        failures: list[str] = []
        for media_type in dict.fromkeys(required_types):
            # 任一该类型的文件通过即算该 deliverable 满足（or 语义：
            # gif 或 mp4 任一）。
            candidates = []
            if media_type.startswith("image/") or media_type.startswith("video/"):
                ext = media_type.split("/")[-1].replace("jpeg", "jpg")
                candidates = [
                    p for p in Path(workspace).rglob(f"*.{ext}") if ".git" not in p.parts
                ]
                if media_type == "image/gif":
                    candidates += [
                        p for p in Path(workspace).rglob("*.gif") if ".git" not in p.parts
                    ]
            elif media_type == "application/json":
                candidates = [
                    p for p in Path(workspace).rglob("*.json") if ".git" not in p.parts
                ]
            if changed is not None:
                candidates = [p for p in candidates if p.resolve() in changed]
            errors = [validate_media_file(p, media_type) for p in candidates]
            passed = any(e is None for e in errors)
            if not passed:
                failures.append(
                    f"deliverable {media_type} 未通过（候选 {len(candidates)} 个"
                    + (f"，首个错误: {errors[0]}" if errors else "，无文件产出")
                    + "）"
                )
        return failures

    # ------------------------------------------------------------------
    def _annotate(self, work_order_id: str, **fields: str) -> None:
        """订单 inputs 回写（workspace/base_sha 解析快照）——DB 权威。"""
        conn = self._conn
        if conn is None:
            return
        row = conn.execute(
            "SELECT order_json FROM work_orders WHERE work_order_id = ?",
            (work_order_id,),
        ).fetchone()
        if row is None:
            return
        order = WorkOrderV1(**json.loads(row["order_json"]))
        inputs = {**dict(order.inputs), **{k: v for k, v in fields.items() if v}}
        updated = order.model_copy(update={"inputs": inputs})
        conn.execute(
            "UPDATE work_orders SET order_json = ?, updated_at = ? WHERE work_order_id = ?",
            (updated.model_dump_json(), datetime.now(UTC).isoformat(), work_order_id),
        )

    # ------------------------------------------------------------------
    async def _prepare_workspace(self, order: WorkOrderV1) -> tuple[str, Path | None]:
        """W3 Developer Workbench：写能力 profile 在独立 workspace 运行。

        git 仓库 → `git worktree add`（branch rosclaw/<wo>，基于 inputs
        .base_ref 或 HEAD）——diff/promotion 可审查；非 git → 独立 scratch
        目录（诚实降级：patch 以文件清单代替）。只读 profile 原地运行。
        返回 (workspace_cwd, base_ref|None)。
        """
        profile = str(order.inputs.get("worker_profile") or "scout")
        target = str(order.inputs.get("workspace") or Path.cwd())
        if profile not in WORKBENCH_PROFILES:
            if not Path(target).is_dir():
                raise AdapterError(f"workspace {target} 不存在")
            return target, None
        # 十一审 PR-E：auto-retry/resume 复用既有 worktree（不从零开始）。
        # 安全闸：只复用 ROSClaw 自己创建的 work 目录下的 worktree——
        # 绝不把用户主仓库目录当 workbench workspace 直写。
        reuse = str(order.inputs.get("_reuse_workspace") or "")
        if (
            reuse
            and Path(reuse).is_dir()
            and str(Path(reuse).resolve()).startswith(str((self._home / "work").resolve()))
        ):
            base_sha = str(order.inputs.get("base_sha") or "") or None
            return reuse, base_sha
        work_root = self._home / "work" / order.work_order_id
        ws = work_root / "workspace"
        base_ref = str(order.inputs.get("base_ref") or "HEAD")
        if (Path(target) / ".git").exists():
            branch = f"rosclaw/{order.work_order_id}"
            code, out = await _git(
                "worktree", "add", str(ws), "-b", branch, base_ref, cwd=target
            )
            if code != 0:
                raise AdapterError(f"git worktree add 失败: {out.strip()[:300]}")
            _, resolved = await _git("rev-parse", base_ref, cwd=target)
            return str(ws), resolved.strip()
        # 非 git：scratch workspace（诚实——result 里注明无 VCS diff）。
        ws.mkdir(parents=True, exist_ok=True)
        return str(ws), None

    async def _changed_files(
        self, workspace: str, base_ref: str | None
    ) -> set[Path] | None:
        """本 attempt 实际改动的文件集（git status；scratch/无法判定时
        返回 None=不限制）。工件收集与 deliverable 验收共用——防"仓库
        既有文件被算作 Worker 产出"。"""
        if not base_ref:
            return None
        code, out = await _git("status", "--porcelain", cwd=workspace)
        if code != 0:
            return None
        changed: set[Path] = set()
        for line in out.splitlines():
            if not line.strip():
                continue
            rel = line[3:].strip()
            if " -> " in rel:  # rename
                rel = rel.split(" -> ", 1)[1]
            changed.add((Path(workspace) / rel).resolve())
        return changed

    async def _collect_workbench_artifacts(
        self, order: WorkOrderV1, workspace: str, base_ref: str | None
    ) -> tuple[list[ResultArtifact], list[ResultClaim], str]:
        """W3：patch + bash log + 媒体工件；promotion 是独立动作——
        这里只产出可审查证据，绝不自动合并回主仓库。"""
        work_root = self._home / "work" / order.work_order_id
        artifacts_dir = work_root / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[ResultArtifact] = []
        claims: list[ResultClaim] = []
        notes: list[str] = []

        def _file_artifact(path: Path, media_type: str) -> ResultArtifact:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            return ResultArtifact(
                ref=f"artifact://{media_type.split('/')[-1]}/sha256:{digest[:32]}",
                media_type=media_type,
                digest=f"sha256:{digest}",
            )

        if base_ref:
            # 含新文件的完整 diff：先 add -A（只在 worktree 内，不碰主仓库）。
            await _git("add", "-A", cwd=workspace)
            code, diff = await _git("diff", "--cached", base_ref, cwd=workspace)
            patch_path = artifacts_dir / "patch.diff"
            if code == 0 and diff.strip():
                patch_path.write_text(diff, encoding="utf-8")
                artifacts.append(_file_artifact(patch_path, "text/x-diff"))
                claims.append(
                    ResultClaim(
                        claim="produced a reviewable git patch (not merged)",
                        evidence_refs=[artifacts[-1].ref],
                    )
                )
            else:
                patch_path.write_text(
                    "# EMPTY DIFF — worker made no file changes\n", encoding="utf-8"
                )
                artifacts.append(_file_artifact(patch_path, "text/x-diff"))
                notes.append("empty patch: worker 未产生文件改动")
            # promotion 证据：worktree/branch 保留。
            _, branch = await _git("rev-parse", "--abbrev-ref", "HEAD", cwd=workspace)
            notes.append(f"worktree 保留于 {workspace}（branch {branch.strip()}，未合并）")
        else:
            notes.append("workspace 非 git 仓库——无 VCS diff（scratch workspace）")
        bash_log = artifacts_dir / "bash-log.txt"
        if bash_log.exists() and bash_log.stat().st_size > 0:
            artifacts.append(_file_artifact(bash_log, "text/plain"))
            claims.append(
                ResultClaim(
                    claim="bash command log (tests/builds run)",
                    evidence_refs=[artifacts[-1].ref],
                )
            )
        # 媒体产物（sim-builder）：只收本 attempt 实际改动的文件——
        # 自审修复：不得把仓库里既有的图片算成 Worker 产出（工件造假）。
        changed = await self._changed_files(workspace, base_ref)
        for pattern in ("*.png", "*.gif", "*.mp4"):
            for media in sorted(Path(workspace).rglob(pattern))[:50]:
                if ".git" in media.parts or media.stat().st_size == 0:
                    continue
                if changed is not None and media not in changed:
                    continue  # 非本 attempt 改动——不算产出
                media_type = {"png": "image/png", "gif": "image/gif", "mp4": "video/mp4"}[
                    media.suffix.lstrip(".")
                ]
                artifacts.append(_file_artifact(media, media_type))
                claims.append(
                    ResultClaim(
                        claim=f"generated media artifact {media.name}",
                        evidence_refs=[artifacts[-1].ref],
                    )
                )
        return artifacts, claims, "；".join(notes)

    # ------------------------------------------------------------------
    def _write_envelope(
        self, order: WorkOrderV1, *, cwd: str, artifacts_dir: Path
    ) -> tuple[Path, str]:
        """只读 WorkOrder envelope（0600）——无 secret：模型快照只含
        provider/model/thinking，凭据由子进程从同一 agentDir 读取。"""
        work_dir = self._home / "work" / order.work_order_id
        work_dir.mkdir(parents=True, exist_ok=True)
        snapshot = dict(order.inputs.get("model_snapshot") or {})
        # 硬防线：快照不得携带任何凭据字段（worker_cannot_serialize_credentials）。
        forbidden = ("api_key", "apikey", "key", "token", "secret", "authorization")
        if any(k.lower() in forbidden for k in snapshot):
            raise AdapterError("model snapshot must not carry credentials")
        envelope = {
            "work_order_id": order.work_order_id,
            "attempt_id": f"att_{order.lease.lease_id if order.lease else '0'}",
            "profile": str(order.inputs.get("worker_profile") or "scout"),
            "goal": order.goal,
            "instructions": str(order.inputs.get("instructions") or order.goal),
            "cwd": cwd,
            "artifacts_dir": str(artifacts_dir),
            # 十二审 PR-12.3：持久 session（attempt 目录）+ resume。
            "session_dir": str(work_dir / "session"),
            "resume_session_file": str(order.inputs.get("_resume_session") or "") or None,
            # 十一审 PR-A：DoD 注入——期望工件进 envelope（developer 的
            # 最终提示据此要求真实 diff/测试日志）。
            "expected_artifacts": list(order.expected_output.artifacts),
            "budget": {
                "wall_time_sec": order.budgets.wall_time_sec,
                "model_tokens": order.budgets.model_tokens,
            },
            "model": {
                "provider": str(snapshot.get("provider", "")),
                "model": str(snapshot.get("model", "")),
                **({"thinking": str(snapshot["thinking"])} if snapshot.get("thinking") else {}),
            }
            if snapshot
            else None,
        }
        path = work_dir / "order.json"
        path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        return path, cwd

    async def _run(
        self,
        order: WorkOrderV1,
        handle: RunHandle,
        node: str,
        entry: str,
    ) -> WorkResultV1:
        started = datetime.now(UTC)
        loop_start = asyncio.get_running_loop().time()
        # 十二审 PR-12.4：媒体/仿真任务 preflight——缺依赖 5 秒内
        # BLOCKED_PREFLIGHT，不烧 Worker 预算。
        await self._preflight(order)
        # W3：写能力 profile 先准备独立 workspace（git worktree/scratch）。
        workspace, base_ref = await self._prepare_workspace(order)
        # 十一审 PR-D：resolved workspace + base_sha 回写订单（WorkOrder
        # 引用解析后的快照，不靠自然语言路径）。
        self._annotate(order.work_order_id, workspace=workspace, base_sha=base_ref or "")
        artifacts_dir = self._home / "work" / order.work_order_id / "artifacts"
        envelope_path, cwd = self._write_envelope(
            order, cwd=workspace, artifacts_dir=artifacts_dir
        )
        env = {
            k: os.environ[k]
            for k in ("PATH", "HOME", "LANG", "LC_ALL", "TZ", "ROSCLAW_HOME")
            if k in os.environ
        }
        # 十审 W1：与主 Agent 同一模型配置——provider env key 原样透传
        # （与 auth.json 文件凭据二选一，取决于用户配置方式；WorkOrder
        # 本身绝不携带凭据）。
        for key in ("KIMI_API_KEY", "MOONSHOT_API_KEY", "ROSCLAW_KIMI_API_KEY"):
            if os.environ.get(key):
                env[key] = os.environ[key]
        # 建议-0816 P0-1：turn 告警/中止阈值是操作员权威配置——显式
        # 设置才透传（默认无硬中止）。
        for key in ("ROSCLAW_WORKER_TURN_WARN_MS", "ROSCLAW_WORKER_TURN_TIMEOUT_MS"):
            if os.environ.get(key):
                env[key] = os.environ[key]
        # 十五审 PR-RF-1 热修（旅程实证）：models.json 里 "$VAR" 形式的
        # provider apiKey 引用也要透传——Worker 只拿到变量名指向的 env
        # 值（引用透传，不是凭据落盘/入 envelope）。缺失时 worker 401
        # （FAKE_JOURNEY_KEY 案例）。
        with contextlib.suppress(Exception):
            models_json = self._home / "agent" / "models.json"
            if models_json.exists():
                import re as _re

                text = models_json.read_text(encoding="utf-8", errors="replace")
                for var in _re.findall(r'"\$([A-Z][A-Z0-9_]+)"', text):
                    if os.environ.get(var):
                        env[var] = os.environ[var]
        env["ROSCLAW_HOME"] = str(self._home)
        env["ROSCLAW_WORKER_PROTOCOL"] = "pi_headless"
        proc = await asyncio.create_subprocess_exec(
            node,
            entry,
            "worker",
            "--headless",
            "--work-order",
            str(envelope_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
            start_new_session=True,
        )
        # W4：活动进程登记 + pid 文件（agentd 崩溃后对账——orphan 必须
        # 可被下一轮启动清掉）。
        self._procs[order.work_order_id] = proc
        pid_file = self._home / "work" / order.work_order_id / "child.pid"
        pid_file.write_text(f"{proc.pid}\n", encoding="utf-8")
        events: list[dict] = []
        final_report = ""
        failure: dict | None = None
        # 十一审 PR-A：三层分离——liveness（进程活着）≠ activity（模型/
        # 工具在跑）≠ semantic progress（真实产出）。只有全静默
        # （连 liveness 都没有）才算死。
        state = {
            "last_event_at": asyncio.get_running_loop().time(),
            "last_semantic_at": asyncio.get_running_loop().time(),
            "phase": "STARTING",
            "stall_warned": False,
            "wrapup_sent": False,
            "wrapup_at": 0.0,
        }

        async def _read_events() -> None:
            nonlocal final_report, failure
            assert proc.stdout is not None
            while True:
                line = await proc.stdout.readline()
                if not line:
                    return
                try:
                    event = json.loads(line.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    continue
                events.append(event)
                now = asyncio.get_running_loop().time()
                state["last_event_at"] = now
                kind = event.get("kind")
                # PR-B：边读边落账本（含 liveness——UI 需要实时流）。
                # 自审修复：账本写失败（磁盘满等）不得杀死事件读者——
                # 否则健康 Worker 会被 liveness 超时误杀。
                with contextlib.suppress(Exception):
                    self._events.append_event(
                        order.work_order_id,
                        str(event.get("attempt_id", "")),
                        str(kind),
                        {k: v for k, v in event.items() if k not in ("kind", "attempt_id")},
                    )
                if kind == "liveness":
                    # 只证明活着——phase/span 供 UI，不推进语义进度。
                    state["phase"] = str(event.get("phase", state["phase"]))
                    continue
                # 真实语义事件：进度与 stall 恢复。
                handle.progress_seq += 1
                state["last_semantic_at"] = now
                state["stall_warned"] = False
                if kind == "attempt_finished":
                    final_report = str(event.get("report", ""))
                elif kind == "attempt_failed":
                    failure = event
                elif kind == "waiting_input":
                    # PR-E：真实 WAITING_INPUT 状态（RUNNING→BLOCKED）。
                    import contextlib as _cl2

                    with _cl2.suppress(Exception):
                        self._on_waiting_input(order.work_order_id)
                elif kind == "session_persisted":
                    state["session_file"] = str(event.get("session_file", ""))
                elif kind == "control.ack":
                    # 十四审：ACK 收条——request_pause/resume/cancel 的
                    # 唯一"已生效"证据。
                    self._control_acks.setdefault(order.work_order_id, {})[
                        str(event.get("control_id", ""))
                    ] = str(event.get("state", ""))
                elif kind == "answer_received":
                    import contextlib as _cl3

                    with _cl3.suppress(Exception):
                        self._on_answered(order.work_order_id)

        async def _drain_stderr() -> None:
            # PR-B：stderr 脱敏落盘（不再丢弃）。
            assert proc.stderr is not None
            while True:
                line = await proc.stderr.readline()
                if not line:
                    return
                self._events.append_stderr(
                    order.work_order_id, line.decode("utf-8", errors="replace")
                )

        readers = asyncio.gather(_read_events(), _drain_stderr())

        async def _teardown() -> None:
            if proc.returncode is None:
                await kill_process_tree(proc)
            readers.cancel()
            with contextlib.suppress(Exception):
                await readers
            self._procs.pop(order.work_order_id, None)
            pid_file.unlink(missing_ok=True)

        try:
            # startup timeout：attempt_started 必须在限定时间内出现。
            startup_end = asyncio.get_running_loop().time() + STARTUP_TIMEOUT_SEC
            while not events:
                if proc.returncode is not None:
                    # 竞态修复（CI 实证）：进程可能已退出但 stdout 事件尚
                    # 未被 reader 消费——先给 reader 一个排水窗口再判死。
                    await asyncio.sleep(0.2)
                    if not events:
                        await asyncio.wait_for(readers, timeout=5)
                        if not events:
                            raise AdapterError(
                                f"worker attempt failed [WORKER_CRASH]: "
                                f"exited {proc.returncode} before attempt_started"
                            )
                    break
                if asyncio.get_running_loop().time() > startup_end:
                    raise AdapterError(
                        "worker attempt failed [WORKER_CRASH]: "
                        "startup timeout (no attempt_started)"
                    )
                await asyncio.sleep(0.05)
            # 主监控循环（十四审 PR-14.1——Worker 有证据在工作就
            # 让它继续；soft target 只是观察指标，绝不控制进程）：
            # - 默认无硬截止：hard_deadline 仅在显式权威来源下生效；
            # - 全静默 > LIVENESS_TIMEOUT → 多信号探测（pid/CPU）——
            #   活着 = UNREACHABLE（不杀），死了 = INTERRUPTED_RESUMABLE；
            # - 语义静默 > STALL_WARN → 只告警；
            # - wall/token soft target 到期 → 提醒事件（不干预）；
            # - 只有 user/admin_policy 权威的 cost_hard_limit 才控制暂停
            #   （control.ack PAUSED 后才落 BUDGET_PAUSED）。
            policy = order.inputs.get("execution_policy") or {}
            hard_sec = None
            if policy.get("hard_deadline_sec") and policy.get(
                "hard_deadline_source"
            ) in ("user", "benchmark", "admin_policy"):
                hard_sec = float(policy["hard_deadline_sec"])
            soft_target_sec = float(
                policy.get("soft_target_sec") or order.budgets.wall_time_sec or 0
            )
            # 十四审 §3.1：token soft target 只是遥测——模型自报的
            # model_tokens 在多 turn 累计下必然误杀，绝不控制进程。
            token_limit = int(
                policy.get("token_soft_limit") or order.budgets.model_tokens or 0
            )
            cost_hard_limit = 0
            if policy.get("cost_hard_limit_tokens") and policy.get(
                "cost_hard_limit_source"
            ) in ("user", "admin_policy"):
                cost_hard_limit = int(policy["cost_hard_limit_tokens"])
            wall_end = (
                asyncio.get_running_loop().time() + hard_sec if hard_sec else None
            )
            unreachable_since: float | None = None
            while proc.returncode is None:
                try:
                    await asyncio.wait_for(proc.wait(), timeout=1.0)
                    break
                except TimeoutError:
                    pass
                now = asyncio.get_running_loop().time()
                silent = now - state["last_event_at"]
                if silent > LIVENESS_TIMEOUT_SEC:
                    # 多信号判活：事件管道故障 ≠ Worker 没在工作。
                    if unreachable_since is None:
                        unreachable_since = now
                        with contextlib.suppress(Exception):
                            self._manager_ref._transition(
                                order.work_order_id, "UNREACHABLE", "event_pipe_silent"
                            )
                        self._events.append_event(
                            order.work_order_id, "", "unreachable",
                            {"silent_sec": int(silent)},
                        )
                    probe = _probe_process(proc.pid)
                    if probe != "working" and (
                        now - unreachable_since > UNREACHABLE_GRACE_SEC
                    ):
                        # 宽限后仍无进展（挂起）= 恢复失败——终止但保留
                        # 会话/工作区（INTERRUPTED_RESUMABLE，不是 FAILED）。
                        probe = "dead"
                    if probe == "dead":
                        # 进程确认死亡：中断但可恢复（不判 FAILED）。
                        self._write_checkpoint(order, state, "INTERRUPTED", "")
                        with contextlib.suppress(Exception):
                            self._manager_ref._transition(
                                order.work_order_id,
                                "INTERRUPTED_RESUMABLE",
                                "process_dead",
                            )
                        return WorkResultV1(
                            work_order_id=order.work_order_id,
                            worker_id=WORKER_ID,
                            lease_id=handle.lease_id,
                            status="INTERRUPTED",
                            summary="worker 进程死亡——会话/工作区已保留，"
                            "可 /job resume 恢复",
                            warnings=["interrupted_resumable"],
                        )
                    # alive/working：继续等（不杀）。
                else:
                    if unreachable_since is not None:
                        unreachable_since = None
                        with contextlib.suppress(Exception):
                            self._manager_ref._transition(
                                order.work_order_id, "RUNNING", "event_pipe_recovered"
                            )
                semantic_silent = now - state["last_semantic_at"]
                if semantic_silent > STALL_WARN_SEC and not state["stall_warned"]:
                    state["stall_warned"] = True
                    # 告警事件入列 + 落账（UI 标黄）。
                    stall = {
                        "kind": "stall_warning",
                        "phase": state["phase"],
                        "semantic_silent_sec": int(semantic_silent),
                    }
                    events.append(stall)
                    self._events.append_event(
                        order.work_order_id, "", "stall_warning", dict(stall)
                    )
                if wall_end is not None and now > wall_end and not state["wrapup_sent"]:
                    state["wrapup_sent"] = True
                    state["wrapup_at"] = now
                    await self.steer(
                        order.work_order_id,
                        "已接近 wall 时间预算。停止开新工作：保存当前改动、跑最窄的"
                        "验证命令，并产出可恢复的 partial handoff（已改文件/已验证"
                        "内容/未完成事项/阻塞原因）。",
                    )
                if state["wrapup_sent"] and now - state["wrapup_at"] > WRAPUP_GRACE_SEC:
                    raise AdapterError(
                        "worker hard deadline exceeded（wrap-up 宽限后仍未退出）"
                    )
                # soft target：一次性提醒（不干预）。
                if (
                    soft_target_sec
                    and now - loop_start > soft_target_sec
                    and not state.get("soft_notified")
                ):
                    state["soft_notified"] = True
                    self._events.append_event(
                        order.work_order_id, "", "soft_target_exceeded",
                        {"soft_target_sec": soft_target_sec},
                    )
                # token soft target：80%/100% 只告警（绝不暂停——总纲
                # §3.1"预算提醒不是安全审批，不得中断正在做的 Worker"）。
                usage_last = next(
                    (e for e in reversed(events) if e.get("kind") == "usage"), None
                )
                if usage_last and token_limit:
                    spent = int(usage_last.get("input_tokens") or 0) + int(
                        usage_last.get("output_tokens") or 0
                    )
                    if spent >= token_limit and not state.get("budget_notified"):
                        state["budget_notified"] = True
                        self._events.append_event(
                            order.work_order_id, "", "budget_warning",
                            {"spent": spent, "limit": token_limit, "level": "soft"},
                        )
                    elif (
                        spent >= int(token_limit * 0.8)
                        and not state.get("budget_notified_80")
                    ):
                        state["budget_notified_80"] = True
                        self._events.append_event(
                            order.work_order_id, "", "budget_warning",
                            {"spent": spent, "limit": token_limit, "level": "soft"},
                        )
                # hard cost limit（显式 user/admin_policy 权威）：到限 →
                # 控制暂停——先 PAUSE_REQUESTED，ACK PAUSED 后才落
                # BUDGET_PAUSED；ACK 失败诚实回报（不乐观）。
                if usage_last and cost_hard_limit:
                    spent = int(usage_last.get("input_tokens") or 0) + int(
                        usage_last.get("output_tokens") or 0
                    )
                    if spent >= cost_hard_limit and not state.get("budget_paused"):
                        state["budget_paused"] = True
                        with contextlib.suppress(Exception):
                            self._manager_ref._transition(
                                order.work_order_id, "PAUSE_REQUESTED", "cost_hard_limit"
                            )
                        paused = await self.request_pause(
                            order.work_order_id, reason="budget_hard"
                        )
                        with contextlib.suppress(Exception):
                            if paused:
                                self._manager_ref._transition(
                                    order.work_order_id, "BUDGET_PAUSED",
                                    "cost_hard_limit_ack",
                                )
                            else:
                                self._manager_ref._transition(
                                    order.work_order_id, "RUNNING",
                                    "cost_pause_ack_failed",
                                )
                        self._events.append_event(
                            order.work_order_id, "",
                            "budget_paused" if paused else "budget_pause_failed",
                            {"spent": spent, "limit": cost_hard_limit, "acked": paused},
                        )
        except asyncio.CancelledError:
            # 十二审 PR-12.5：先收集部分成果再终止（cancel 也要 partial）。
            partial_note = await self._collect_partial_note(order, workspace, base_ref)
            await _teardown()
            self._write_checkpoint(order, state, "CANCELLED", partial_note)
            raise
        except Exception as exc:
            # 十二审 PR-12.5：超时/崩溃路径——先冻结收集 partial
            # （diff/媒体/测试日志已在 artifacts/），附注后再终止。
            partial_note = await self._collect_partial_note(order, workspace, base_ref)
            await _teardown()
            self._write_checkpoint(order, state, "FAILED", partial_note)
            if partial_note:
                raise AdapterError(f"{exc}；partial: {partial_note}") from exc
            raise
        await readers
        self._procs.pop(order.work_order_id, None)
        # 注意：_control_acks 不在此清理——post-exit 与 _wait_ack 有竞态
        # （worker ACK 后秒退，pop 会抹掉等待者要读的收条）；每单几条
        # 收条，驻留内存可忽略。
        pid_file.unlink(missing_ok=True)
        # 十四审 PR-14.1（总纲 §3.4）：termination.json 是终态原因唯一
        # 权威；exit code 只是 Unix 表象（130 可能是取消/暂停/信号/重启），
        # 不得直接当 FAILED 或自动重试依据。进程来不及写 → SIGNAL_UNKNOWN。
        termination = self._read_termination(order.work_order_id)
        cause = str((termination or {}).get("cause") or "")
        if not cause:
            if failure is not None:
                cause = ERROR_CODE_CAUSES.get(
                    str(failure.get("error_code", "")), "WORKER_CRASH"
                )
            elif final_report or proc.returncode == 0:
                cause = "COMPLETED"
            else:
                cause = "SIGNAL_UNKNOWN"
        detail = str(
            (termination or {}).get("detail")
            or (failure or {}).get("message")
            or ""
        )
        if termination and termination.get("session_file"):
            state["session_file"] = str(termination["session_file"])
        interrupted_causes = {
            "SIGNAL_UNKNOWN", "AGENTD_SHUTDOWN", "USER_PAUSED", "BUDGET_HARD_PAUSED",
        }
        terminal_status = (
            "COMPLETED" if cause == "COMPLETED"
            else "CANCELLED" if cause == "USER_CANCELLED"
            else "INTERRUPTED" if cause in interrupted_causes
            else "FAILED"
        )
        # PR-B：终态落 state.json（重启对账/tail 可读）。
        self._events.write_state(
            order.work_order_id,
            {
                "status": terminal_status,
                "phase": "TERMINAL",
                "last_seq": handle.progress_seq,
                "termination_cause": cause,
                "error": detail or None,
                # PR-12.3：resume 的恢复点（Pi 原生 session 文件）。
                "session_file": state.get("session_file", ""),
            },
        )
        finished = datetime.now(UTC)
        if cause in interrupted_causes:
            # 中断可恢复——不是 FAILED：checkpoint + INTERRUPTED_RESUMABLE。
            self._write_checkpoint(order, state, "INTERRUPTED", "")
            with contextlib.suppress(Exception):
                self._manager_ref._transition(
                    order.work_order_id, "INTERRUPTED_RESUMABLE", cause.lower()
                )
            return WorkResultV1(
                work_order_id=order.work_order_id,
                worker_id=WORKER_ID,
                lease_id=handle.lease_id,
                status="INTERRUPTED",
                summary=f"worker 中断（{cause}）——会话/工作区已保留，"
                "可 /job resume 恢复",
                warnings=["interrupted_resumable"],
            )
        if cause == "USER_CANCELLED":
            # 只有用户取消产生 CANCELLED（控制协议 cancel 动作）。
            self._write_checkpoint(order, state, "CANCELLED", "")
            return WorkResultV1(
                work_order_id=order.work_order_id,
                worker_id=WORKER_ID,
                lease_id=handle.lease_id,
                status="CANCELLED",
                summary="worker 已被用户取消",
                warnings=["cancelled"],
            )
        if cause != "COMPLETED":
            # FAILED——摘要携带权威 cause（Native Agent 不得再猜日志归因）。
            raise AdapterError(f"worker attempt failed [{cause}]: {detail}")
        if not final_report and proc.returncode != 0:
            # 声称完成但无报告且非零退出——矛盾，按崩溃诚实归类。
            raise AdapterError(
                "worker attempt failed [WORKER_CRASH]: 无最终报告"
            )
        usage_last = next((e for e in reversed(events) if e.get("kind") == "usage"), {})
        usage = WorkUsage(
            wall_time_ms=int((finished - started).total_seconds() * 1000),
            prompt_tokens=int(usage_last.get("input_tokens") or 0),
            completion_tokens=int(usage_last.get("output_tokens") or 0),
        )
        digest = hashlib.sha256(final_report.encode()).hexdigest()
        report_artifact = ResultArtifact(
            ref=f"artifact://text/sha256:{digest[:32]}",
            media_type="text/plain",
            digest=f"sha256:{digest}",
        )
        artifacts = [report_artifact]
        claims = [
            ResultClaim(
                claim="pi worker produced a final report",
                evidence_refs=[report_artifact.ref],
            )
        ]
        summary = final_report[:2000]
        # W3：写能力 profile 收集可审查证据（patch/bash log/媒体）——
        # promotion 是独立动作，绝不自动合并。
        if str(order.inputs.get("worker_profile") or "") in WORKBENCH_PROFILES:
            wb_artifacts, wb_claims, notes = await self._collect_workbench_artifacts(
                order, workspace, base_ref
            )
            artifacts.extend(wb_artifacts)
            claims.extend(wb_claims)
            if notes:
                summary = f"{summary}\n\n[workbench] {notes}"
            # 十二审 PR-12.4：required deliverable 未过 = 诚实失败
            # （deliverable 未通过不得宣布完成）。
            failures = await self._validate_deliverables(order, workspace)
            if failures:
                raise AdapterError("DELIVERABLE_FAILED: " + "；".join(failures))
        return WorkResultV1(
            work_order_id=order.work_order_id,
            worker_id=WORKER_ID,
            lease_id=handle.lease_id,
            status="COMPLETED",
            started_at=started.isoformat(),
            finished_at=finished.isoformat(),
            summary=summary,
            artifacts=artifacts,
            claims=claims,
            usage=usage,
        )
