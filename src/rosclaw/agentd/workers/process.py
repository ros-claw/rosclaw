"""Worker 子进程控制（十审 W0/W1 共享）。

所有 worker 子进程一律 start_new_session 独立进程组；cancel/timeout
走 SIGTERM → grace → SIGKILL 的整组清理——任何 Worker 退出路径都不得
留下孤儿进程（含孙进程）。
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal

#: cancel grace——先 SIGTERM，超时 SIGKILL（产品指标：2s 级，硬上限 7s）。
CANCEL_GRACE_SEC = 5.0


async def kill_process_tree(proc, *, grace_sec: float = CANCEL_GRACE_SEC) -> None:
    """杀整个进程组（start_new_session=True 保证子进程是组长）。

    SIGTERM → grace → SIGKILL；最后 reap。进程已退则静默返回。
    """
    if proc.returncode is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError):
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGTERM)
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(proc.wait(), timeout=grace_sec)
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGKILL)
    # 防御：内核都杀不动——不阻塞 cancel 闭环。
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(proc.wait(), timeout=2)
