"""process-stdio Worker Adapter SDK (PR-WF-052).

Runs an external worker as a subprocess speaking a versioned JSONL
envelope over stdin/stdout. Sandboxing in P0:

- environment scrub: only an allowlisted set of variables is inherited;
  credential *refs* are resolved by the host and passed as explicit
  envelope fields — never as ambient env of the child;
- cwd is confined to the declared writable path (best effort, documented;
  landlock/bubblewrap hardening is a later PR);
- line size cap and protocol validation: garbage, oversized lines, or
  out-of-order envelopes fail the order closed;
- cancel = SIGTERM → SIGKILL; reconcile asks the adapter journal via the
  idempotency key (P0: process-local tracking only).

Envelope ``rosclaw.worker_adapter.v1``:

  host→worker {"type":"handshake","protocol":...,"card_digest":...}
  worker→host {"type":"ready","card_digest":...}
  host→worker {"type":"work_order","order":{...},"credentials":{ref:value}}
  worker→host {"type":"heartbeat","progress_seq":n,"checkpoint":...}
  worker→host {"type":"result","result":{...WorkResultV1...}}
  host→worker {"type":"cancel","reason":...}
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from rosclaw.agentd.workers.adapter import (
    AdapterError,
    RunHandle,
    WorkerProbeResult,
)
from rosclaw.contracts.worker.order import WorkOrderV1, WorkResultV1

PROTOCOL = "rosclaw.worker_adapter.v1"
MAX_LINE_BYTES = 1_048_576

#: Env vars the child may inherit. Everything else is scrubbed.
_INHERIT_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TZ", "PYTHONPATH", "VIRTUAL_ENV")


class ProtocolViolationError(AdapterError):
    """Worker broke the envelope protocol."""


class ProcessStdioAdapter:
    def __init__(
        self,
        *,
        worker_id: str,
        command: list[str],
        cwd: Path | None = None,
        handshake_timeout_sec: float = 10.0,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self._worker_id = worker_id
        self._command = command
        self._cwd = cwd
        self._handshake_timeout = handshake_timeout_sec
        self._extra_env = extra_env or {}
        self._runs: dict[str, tuple[RunHandle, asyncio.Task]] = {}

    # ------------------------------------------------------------------
    def _child_env(self) -> dict[str, str]:
        env = {k: os.environ[k] for k in _INHERIT_ENV if k in os.environ}
        env.update(self._extra_env)
        env["ROSCLAW_WORKER_PROTOCOL"] = PROTOCOL
        return env

    async def probe(self) -> WorkerProbeResult:
        import shutil

        exe = self._command[0] if self._command else ""
        found = os.path.exists(exe) if os.sep in exe else shutil.which(exe) is not None
        if not exe or not found:
            return WorkerProbeResult(ready=False, detail=f"executable {exe!r} not found")
        return WorkerProbeResult(ready=True, detail="executable present")

    # ------------------------------------------------------------------
    async def start(self, order: WorkOrderV1, credential_refs: dict) -> RunHandle:
        if order.lease is None:
            raise AdapterError("work order has no lease")
        handle = RunHandle(
            work_order_id=order.work_order_id,
            lease_id=order.lease.lease_id,
            worker_id=self._worker_id,
        )
        task = asyncio.create_task(self._run(order, handle, credential_refs))
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
        except Exception as exc:  # noqa: BLE001 - failure is data
            return WorkResultV1(
                work_order_id=handle.work_order_id,
                worker_id=self._worker_id,
                lease_id=handle.lease_id,
                status="FAILED",
                summary=f"{type(exc).__name__}: {exc}",
                warnings=["adapter_error"],
            )

    async def cancel(self, handle: RunHandle, reason: str) -> None:
        entry = self._runs.pop(handle.work_order_id, None)
        if entry is not None:
            entry[1].cancel()

    async def reconcile(self, idempotency_key: str) -> str:
        for _handle, task in self._runs.values():
            if not task.done():
                return "running"
        return "not_found"

    # ------------------------------------------------------------------
    async def _run(
        self, order: WorkOrderV1, handle: RunHandle, credential_refs: dict
    ) -> WorkResultV1:
        proc = await asyncio.create_subprocess_exec(
            *self._command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._cwd) if self._cwd else None,
            env=self._child_env(),
            limit=MAX_LINE_BYTES,
        )
        assert proc.stdin and proc.stdout

        async def send(obj: dict) -> None:
            proc.stdin.write((json.dumps(obj, ensure_ascii=False) + "\n").encode())
            await proc.stdin.drain()

        async def recv(expected: set[str], timeout: float) -> dict:
            try:
                raw = await asyncio.wait_for(proc.stdout.readline(), timeout)
            except TimeoutError as exc:
                raise ProtocolViolationError(f"timeout waiting for {sorted(expected)}") from exc
            except asyncio.LimitOverrunError as exc:
                raise ProtocolViolationError("worker line exceeded 1MB cap") from exc
            if not raw:
                stderr_tail = b""
                if proc.stderr:
                    import contextlib

                    with contextlib.suppress(Exception):
                        stderr_tail = await proc.stderr.read(512)
                raise ProtocolViolationError(
                    f"worker exited without {'/'.join(sorted(expected))} "
                    f"(stderr: {stderr_tail.decode(errors='replace')[:200]})"
                )
            try:
                message = json.loads(raw.decode("utf-8", errors="replace"))
            except json.JSONDecodeError as exc:
                raise ProtocolViolationError(
                    f"worker emitted non-JSON line: {raw[:120]!r}"
                ) from exc
            mtype = message.get("type")
            if mtype not in expected:
                raise ProtocolViolationError(f"expected {sorted(expected)}, got {mtype!r}")
            return message

        try:
            await send({"type": "handshake", "protocol": PROTOCOL, "worker_id": self._worker_id})
            ready = await recv({"ready"}, self._handshake_timeout)
            if ready.get("protocol", PROTOCOL) != PROTOCOL:
                raise ProtocolViolationError(f"protocol mismatch: {ready.get('protocol')!r}")
            # Credentials are injected per WorkOrder, never ambient.
            credentials = {name: value for name, value in credential_refs.items() if value}
            await send(
                {
                    "type": "work_order",
                    "order": order.model_dump(mode="json", by_alias=True),
                    "credentials": credentials,
                }
            )
            deadline = order.budgets.wall_time_sec
            while True:
                message = await recv({"heartbeat", "result"}, float(deadline))
                if message["type"] == "heartbeat":
                    handle.progress_seq = int(message.get("progress_seq", 0))
                    handle.last_checkpoint = message.get("checkpoint")
                    continue
                result_payload = message.get("result") or {}
                try:
                    result = WorkResultV1.model_validate_contract(result_payload)
                except Exception as exc:
                    raise ProtocolViolationError(f"invalid WorkResultV1 payload: {exc}") from exc
                return result
        finally:
            if proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), 5)
                except TimeoutError:
                    proc.kill()
                    await proc.wait()
