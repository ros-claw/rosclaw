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

from rosclaw.agentd.pi_entry import find_pi_agent_entry
from rosclaw.agentd.workers.adapter import (
    AdapterError,
    RunHandle,
    WorkerProbeResult,
)
from rosclaw.agentd.workers.process import kill_process_tree
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

#: W3：写能力 profile（Developer Workbench）——workspace 隔离 + diff 工件。
WORKBENCH_PROFILES = ("developer", "sim-builder")


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

    async def cancel(self, handle: RunHandle, reason: str) -> None:  # noqa: ARG002
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
        # 媒体产物（sim-builder）：workspace 内新生成的图片/视频。
        for pattern in ("*.png", "*.gif", "*.mp4"):
            for media in sorted(Path(workspace).rglob(pattern))[:5]:
                if ".git" in media.parts or media.stat().st_size == 0:
                    continue
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
                    raise AdapterError(
                        f"worker exited {proc.returncode} before attempt_started"
                    )
                if asyncio.get_running_loop().time() > startup_end:
                    raise AdapterError("worker startup timeout (no attempt_started)")
                await asyncio.sleep(0.05)
            # 主监控循环（十一审 PR-A）：
            # - 全静默（连 liveness 都没有）> LIVENESS_TIMEOUT → 真死，杀；
            # - 语义静默 > STALL_WARN → 只告警（UI 标黄），不杀；
            # - wall 预算到期 → 先 wrap-up steer，grace 后再杀；
            # - token 预算 ≥80% → wrap-up steer 一次。
            wall_end = (
                asyncio.get_running_loop().time() + order.budgets.wall_time_sec
                if order.budgets.wall_time_sec
                else None
            )
            while proc.returncode is None:
                try:
                    await asyncio.wait_for(proc.wait(), timeout=1.0)
                    break
                except TimeoutError:
                    pass
                now = asyncio.get_running_loop().time()
                silent = now - state["last_event_at"]
                if silent > LIVENESS_TIMEOUT_SEC:
                    raise AdapterError(
                        f"worker liveness lost ({LIVENESS_TIMEOUT_SEC:.0f}s 无任何事件"
                        "——进程挂死，已终止)"
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
                        "worker wall budget exceeded（wrap-up 宽限后仍未退出）"
                    )
                # token 预算 80% → wrap-up（一次）。
                usage_last = next(
                    (e for e in reversed(events) if e.get("kind") == "usage"), None
                )
                if (
                    usage_last
                    and order.budgets.model_tokens
                    and not state["wrapup_sent"]
                ):
                    spent = int(usage_last.get("input_tokens") or 0) + int(
                        usage_last.get("output_tokens") or 0
                    )
                    if spent >= int(order.budgets.model_tokens * 0.8):
                        state["wrapup_sent"] = True
                        state["wrapup_at"] = now
                        await self.steer(
                            order.work_order_id,
                            "已接近 token 预算。停止开新工作：保存当前改动并产出"
                            "可恢复的 partial handoff。",
                        )
        except asyncio.CancelledError:
            await _teardown()
            raise
        except Exception:
            await _teardown()
            raise
        await readers
        self._procs.pop(order.work_order_id, None)
        pid_file.unlink(missing_ok=True)
        # PR-B：终态落 state.json（重启对账/tail 可读）。
        self._events.write_state(
            order.work_order_id,
            {
                "status": "FAILED" if failure is not None else "COMPLETED",
                "phase": "TERMINAL",
                "last_seq": handle.progress_seq,
                "error": failure,
            },
        )
        finished = datetime.now(UTC)
        if failure is not None:
            raise AdapterError(
                f"worker attempt failed [{failure.get('error_code', '?')}]: "
                f"{failure.get('message', '')}"
            )
        if not final_report and proc.returncode != 0:
            raise AdapterError(f"worker exited {proc.returncode} without a final report")
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
