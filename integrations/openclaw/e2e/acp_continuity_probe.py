"""Live session continuity E2E（设计 §49/§52 的 ACP 子进程段）。

真实模型链路：new session → 记住随机码 → **kill ACP 子进程** → 新进程
session/resume 同一 Mission → 询问随机码 → 必须回答正确。

Gateway restart 那一段 continuity 由 Stage 3 的飞书 E2E 覆盖。

用法：

    python acp_continuity_probe.py            # 默认随机码 RC-7F4A9

环境变量：ROSCLAW_BIN、ROSCLAW_HOME（同 acpx_direct_probe.py）。
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import acp
from acp import schema

ROSCLAW_BIN = os.environ.get(
    "ROSCLAW_BIN",
    str(Path(__file__).resolve().parents[3] / ".venv" / "bin" / "rosclaw"),
)
ROSCLAW_HOME = os.environ.get("ROSCLAW_HOME", str(Path.home() / ".rosclaw"))
CODE = sys.argv[1] if len(sys.argv) > 1 else "RC-7F4A9"


class Client(acp.Client):
    def __init__(self) -> None:
        self.chunks: list[str] = []

    async def session_update(self, session_id: str, update, **kwargs) -> None:
        if getattr(update, "session_update", "") == "agent_message_chunk":
            self.chunks.append(update.content.text)


async def start():
    proc = await asyncio.create_subprocess_exec(
        ROSCLAW_BIN,
        "acp",
        "serve",
        "--home",
        ROSCLAW_HOME,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    client = Client()
    conn = acp.connect_to_agent(client, proc.stdin, proc.stdout)
    await conn.initialize(protocol_version=1, client_capabilities=schema.ClientCapabilities())
    return proc, conn, client


async def prompt(conn, sid: str, text: str, timeout: float = 240) -> None:
    resp = await asyncio.wait_for(
        conn.prompt(
            session_id=sid,
            prompt=[schema.TextContentBlock(type="text", text=text)],
        ),
        timeout=timeout,
    )
    print(f"  stop_reason={resp.stop_reason}", flush=True)


async def main() -> int:
    print("[1] 新 session + 记住随机码", flush=True)
    proc1, conn1, _client1 = await start()
    session = await conn1.new_session(cwd=str(Path.home()))
    sid = session.session_id
    print(f"  mission={sid}", flush=True)
    await prompt(conn1, sid, f"请记住测试随机码：{CODE}。只回答：已记住。")

    print("[2] kill ACP 子进程（模拟崩溃）", flush=True)
    proc1.kill()
    await proc1.wait()

    print("[3] 新进程 resume 同一 Mission，询问随机码", flush=True)
    proc2, conn2, client2 = await start()
    try:
        await conn2.resume_session(session_id=sid, cwd=str(Path.home()))
        await prompt(conn2, sid, "我刚才让你记住的随机码是什么？只回答随机码本身。")
        answer = "".join(client2.chunks)
        print(f"  answer={answer.strip()!r}", flush=True)
        if CODE in answer:
            print(f"[PASS] ACP 子进程重启后上下文保留（{CODE}）", flush=True)
            return 0
        print(f"[FAIL] 回答中未找到 {CODE}", flush=True)
        return 1
    finally:
        proc2.kill()
        await proc2.wait()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
