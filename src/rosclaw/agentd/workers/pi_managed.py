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

#: 十审 §7.4 P0 默认：startup 10s、idle 60s。
STARTUP_TIMEOUT_SEC = 10.0
IDLE_TIMEOUT_SEC = 60.0


class PiManagedAdapter:
    """内置 Pi Worker（每个 WorkOrder 一个 headless 子进程）。"""

    def __init__(self, *, rosclaw_home: Path) -> None:
        self._home = Path(rosclaw_home)
        self._runs: dict[str, tuple[RunHandle, asyncio.Task]] = {}

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

    async def reconcile(self, idempotency_key: str) -> str:  # noqa: ARG002
        for _handle, task in self._runs.values():
            if not task.done():
                return "running"
        return "not_found"

    # ------------------------------------------------------------------
    def _write_envelope(self, order: WorkOrderV1) -> tuple[Path, str]:
        """只读 WorkOrder envelope（0600）——无 secret：模型快照只含
        provider/model/thinking，凭据由子进程从同一 agentDir 读取。"""
        work_dir = self._home / "work" / order.work_order_id
        work_dir.mkdir(parents=True, exist_ok=True)
        snapshot = dict(order.inputs.get("model_snapshot") or {})
        # 硬防线：快照不得携带任何凭据字段（worker_cannot_serialize_credentials）。
        forbidden = ("api_key", "apikey", "key", "token", "secret", "authorization")
        if any(k.lower() in forbidden for k in snapshot):
            raise AdapterError("model snapshot must not carry credentials")
        cwd = str(order.inputs.get("workspace") or Path.cwd())
        if not Path(cwd).is_dir():
            cwd = str(Path.cwd())
        envelope = {
            "work_order_id": order.work_order_id,
            "attempt_id": f"att_{order.lease.lease_id if order.lease else '0'}",
            "profile": str(order.inputs.get("worker_profile") or "scout"),
            "goal": order.goal,
            "instructions": str(order.inputs.get("instructions") or order.goal),
            "cwd": cwd,
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
        envelope_path, cwd = self._write_envelope(order)
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
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
            start_new_session=True,
        )
        events: list[dict] = []
        final_report = ""
        failure: dict | None = None

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
                handle.progress_seq += 1  # 真实子进程事件才推进心跳
                kind = event.get("kind")
                if kind == "attempt_finished":
                    final_report = str(event.get("report", ""))
                elif kind == "attempt_failed":
                    failure = event

        async def _drain_stderr() -> None:
            assert proc.stderr is not None
            while await proc.stderr.readline():
                pass

        readers = asyncio.gather(_read_events(), _drain_stderr())

        async def _teardown() -> None:
            if proc.returncode is None:
                await kill_process_tree(proc)
            readers.cancel()
            with contextlib.suppress(Exception):
                await readers

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
            # idle timeout：真实事件才刷新；超时不诚实等待。
            while proc.returncode is None:
                last_seq = handle.progress_seq
                try:
                    await asyncio.wait_for(proc.wait(), timeout=IDLE_TIMEOUT_SEC)
                except TimeoutError:
                    if handle.progress_seq == last_seq:
                        raise AdapterError(
                            f"worker idle timeout ({IDLE_TIMEOUT_SEC:.0f}s 无真实事件)"
                        ) from None
        except asyncio.CancelledError:
            await _teardown()
            raise
        except Exception:
            await _teardown()
            raise
        await readers
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
        return WorkResultV1(
            work_order_id=order.work_order_id,
            worker_id=WORKER_ID,
            lease_id=handle.lease_id,
            status="COMPLETED",
            started_at=started.isoformat(),
            finished_at=finished.isoformat(),
            summary=final_report[:2000],
            artifacts=[
                ResultArtifact(
                    ref=f"artifact://text/sha256:{digest[:32]}",
                    media_type="text/plain",
                    digest=f"sha256:{digest}",
                )
            ],
            claims=[
                ResultClaim(
                    claim="pi worker produced a final report",
                    evidence_refs=[f"artifact://text/sha256:{digest[:32]}"],
                )
            ],
            usage=usage,
        )
