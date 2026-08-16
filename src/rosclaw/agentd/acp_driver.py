"""ACP Harness 驱动（十五审 PR-RF-3，ADR-0011）。

agentd 侧 ACP client：spawn 完整 Harness（claude-code-acp/pi-acp/…），
stdio ndjson JSON-RPC——initialize → session/new → session/prompt
（流式 session/update 落 WorkerEventStore）→ cancel。

红线：不解析 ANSI/PTY，不从文案猜状态；Harness 侧 permission/fs/
terminal 请求一律结构化拒绝（RF-4 sandbox 后按白名单开放）。
无 ACP 二进制 → 诚实 BLOCKED（preflight 失败不创建执行）。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

#: ACP Harness 注册表（二进制 → runtime id）。
ACP_HARNESSES = {
    "harness:acp:claude-local": "claude-code-acp",
    "harness:acp:pi-acp": "pi-acp",
}


def acp_binary_for(runtime: str) -> str | None:
    binary = ACP_HARNESSES.get(runtime)
    if binary is None:
        return None
    return shutil.which(binary)


#: 十五审 PR-RF-4（总纲 §9）：Harness 进程防护——隔离 workspace/HOME/
#: TMP，凭据只给目标 harness，无 ROS/DDS/设备/rosclawd socket。
#: 本机无 bwrap/unshare（Worker workbench 已实证），隔离在进程环境层
#: 实现——这是 GUARDED_PROCESS（建议-0816 P0-8 诚实命名），不是
#: SANDBOXED：python/node 进程内仍可网络/越界读，红队 Gate P8 如实标注。
_SANDBOX_ENV_KEEP = ("PATH", "LANG", "LC_ALL", "TERM", "TZ")
#: 绝不泄漏给 harness 的变量前缀（rosclaw 私有面/物理面）。
_SANDBOX_ENV_DENY_PREFIX = ("ROSCLAW_", "ROS_", "RMW_", "CYCLONEDDS", "FASTRTPS")


def prepare_sandbox_home(
    work_dir: Path, *, harness_config_dirs: list[str] | None = None
) -> Path:
    """构造隔离 sandbox：work_dir/{home,workspace,tmp}。

    harness_config_dirs：目标 harness 自己的配置目录（如 .claude）——
    symlink 进 sandbox HOME（凭据只给目标 harness）；rosclaw 自身的
    agent/auth.json 永不进 sandbox。
    """
    work_dir = Path(work_dir)
    home = work_dir / "home"
    workspace = work_dir / "workspace"
    tmp = work_dir / "tmp"
    for d in (home, workspace, tmp):
        d.mkdir(parents=True, exist_ok=True)
    for name in harness_config_dirs or []:
        src = Path.home() / name
        if src.is_dir():
            link = home / name
            if not link.exists():
                link.symlink_to(src)
    return work_dir


def sandbox_env(work_dir: Path) -> dict[str, str]:
    """最小 env：白名单基础变量 + 隔离 HOME/TMPDIR；拒绝 rosclaw/
    ROS/DDS 前缀（红队测试锁定）。"""
    env = {k: os.environ[k] for k in _SANDBOX_ENV_KEEP if k in os.environ}
    env["HOME"] = str(Path(work_dir) / "home")
    env["TMPDIR"] = str(Path(work_dir) / "tmp")
    for key in list(env):
        if key.startswith(_SANDBOX_ENV_DENY_PREFIX):
            del env[key]
    return env


class AcpHarnessDriver:
    """一个 ACP session 的驱动（单 execution 单 session——Gate 3）。"""

    def __init__(
        self, runtime: str, *, cwd: str, event_sink,
        sandbox_home: Path | None = None,
    ) -> None:
        self._runtime = runtime
        self._cwd = cwd
        self._sink = event_sink  # async (kind, payload) -> None
        # PR-RF-4：sandbox 根（work/<exec>）；None 时退化为传入 cwd
        # （兼容旧调用——新执行路径必须给）。
        self._sandbox = Path(sandbox_home) if sandbox_home else None
        self._proc: asyncio.subprocess.Process | None = None
        self._next_id = 1
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task: asyncio.Task | None = None

    async def _request(
        self, method: str, params: dict | None = None, *,
        timeout: float | None = 120,
    ) -> dict:
        assert self._proc is not None and self._proc.stdin is not None
        request_id = self._next_id
        self._next_id += 1
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending[request_id] = future
        line = json.dumps(
            {"jsonrpc": "2.0", "id": request_id, "method": method,
             "params": params or {}}
        )
        self._proc.stdin.write(f"{line}\n".encode())
        await self._proc.stdin.drain()
        # timeout=None：session/prompt 是长任务（30min+）——绝不按固定
        # 墙钟杀（ADR-0011：无默认 wall-clock kill）。
        if timeout is None:
            return await future
        return await asyncio.wait_for(future, timeout=timeout)

    async def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    return
                try:
                    msg = json.loads(line.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    continue
                if "id" in msg and ("result" in msg or "error" in msg):
                    future = self._pending.pop(msg["id"], None)
                    if future is not None and not future.done():
                        if "error" in msg:
                            err = msg["error"]
                            future.set_exception(
                                RuntimeError(
                                    f"ACP error {err.get('code')}: {err.get('message')}"
                                )
                            )
                        else:
                            future.set_result(msg["result"])
                elif msg.get("method") == "session/update":
                    params = msg.get("params") or {}
                    update = params.get("update") or {}
                    # sink 异常绝不杀事件读者（否则 prompt 永悬）——
                    # 事件丢失只是观测缺口，不是控制失败。
                    with contextlib.suppress(Exception):
                        await self._on_update(params.get("sessionId", ""), update)
                elif "method" in msg and "id" in msg:
                    # Harness → client 请求：MVP 结构化拒绝（RF-4 后白名单）。
                    await self._respond_error(msg["id"], msg["method"])
        finally:
            # 读者死亡/EOF：所有挂起请求诚实失败（绝不悬挂）。
            for _id, future in self._pending.items():
                if not future.done():
                    future.set_exception(
                        RuntimeError("ACP event reader stopped")
                    )
            self._pending.clear()

    async def _respond_error(self, request_id: Any, method: str) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(
            (json.dumps({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"{method} not granted by ROSClaw policy",
                },
            }) + "\n").encode()
        )
        await self._proc.stdin.drain()

    async def _on_update(self, session_id: str, update: dict) -> None:
        tag = str(update.get("sessionUpdate", ""))
        if tag == "agent_message_chunk":
            content = update.get("content") or {}
            # 2000 字上限（卡片摘要另行截断——事件面不做 160 字式
            # 过度裁剪，十四审 §1.6 教训）。
            await self._sink("message_delta", {
                "text": str(content.get("text", ""))[-2000:],
            })
        elif tag == "tool_call":
            await self._sink("tool_started", {
                "tool": str(update.get("title", "?")),
                "tool_call_id": str(update.get("toolCallId", "")),
            })
        elif tag == "tool_call_update":
            await self._sink("tool_finished", {
                "tool_call_id": str(update.get("toolCallId", "")),
                "status": str(update.get("status", "")),
            })
        elif tag == "plan":
            await self._sink("plan_updated", {
                "entries": len(update.get("entries") or []),
            })

    async def run(self, goal: str) -> dict:
        """跑到终态——返回 {ok, stop_reason, detail}。"""
        binary = acp_binary_for(self._runtime)
        if binary is None:
            return {
                "ok": False,
                "stop_reason": "not_installed",
                "detail": f"ACP Harness {self._runtime} 未安装（preflight 失败）",
            }
        # PR-RF-4：sandbox 化启动——隔离 workspace/HOME/TMP、最小 env、
        # 无 ROSCLAW_/ROS_/DDS 变量、独立进程组。
        if self._sandbox is not None:
            prepare_sandbox_home(self._sandbox)
            cwd = str(self._sandbox / "workspace")
            env = sandbox_env(self._sandbox)
        else:
            cwd = self._cwd
            env = None
        self._proc = await asyncio.create_subprocess_exec(
            binary,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
            start_new_session=True,
        )
        self._reader_task = asyncio.create_task(self._read_loop())
        try:
            await self._request("initialize", {
                "protocolVersion": 1,
                "clientCapabilities": {
                    "fs": {"readTextFile": False, "writeTextFile": False},
                    "terminal": False,
                },
                "clientInfo": {"name": "rosclaw", "version": "1.0"},
            })
            session = await self._request(
                "session/new", {"cwd": cwd, "mcpServers": []}
            )
            session_id = str(session["sessionId"])
            result = await self._request("session/prompt", {
                "sessionId": session_id,
                "prompt": [{"type": "text", "text": goal}],
            }, timeout=None)
            stop = str(result.get("stopReason", "unknown"))
            return {
                "ok": stop in ("end_turn", "max_tokens", "refusal"),
                "stop_reason": stop,
                "detail": f"session {session_id} stop={stop}",
            }
        except (TimeoutError, RuntimeError) as exc:
            return {"ok": False, "stop_reason": "error", "detail": str(exc)[:300]}
        finally:
            if self._proc is not None and self._proc.returncode is None:
                self._proc.terminate()
            if self._reader_task is not None:
                self._reader_task.cancel()
                import contextlib

                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await self._reader_task
