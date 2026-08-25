"""Pi engine 模型探测（P1-A1，0824 总纲 §10.1）。

probe 与 chat 同一引擎：``node <rosclaw-agent>/dist/src/main.js --probe``
内部走 Pi ModelRuntime（settings.json + models.json + auth.json/env）。
本模块只做进程编排与无 secret 的结果映射——不另起 HTTP chat 栈。

诚实性：engine 缺失/超时/非零退出/坏 JSON 全部落成显式 error，
绝不假 GREEN；stderr 回显前做 key 形态脱敏。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

from rosclaw.agentd import pi_entry
from rosclaw.agentd.models.gateway import ModelProbeResult

#: probe 总超时（四步含网络；比单次 chat 超时宽）。
_PROBE_TIMEOUT_S = 240.0

_SECRET_RE = re.compile(r"sk-[A-Za-z0-9_-]{4,}")


def _sanitize(text: str, *, limit: int = 400) -> str:
    return _SECRET_RE.sub("sk-***", text.strip())[:limit]


async def pi_probe_home(home: Path) -> ModelProbeResult:
    """经 Pi engine 探测 home 的模型配置（settings/models 单源）。"""
    located = pi_entry.find_pi_agent_entry()
    if located is None:
        return ModelProbeResult(
            reachable=False,
            error="PI_ENGINE_MISSING: node>=22.19 或 rosclaw-agent dist "
            "不可用——重新安装或构建发布包",
        )
    node, entry = located
    env = dict(os.environ, ROSCLAW_HOME=str(home))
    try:
        proc = await asyncio.create_subprocess_exec(
            node,
            str(entry),
            "--probe",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
    except OSError as exc:
        return ModelProbeResult(reachable=False, error=f"PI_ENGINE_SPAWN_FAILED: {exc}")
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_PROBE_TIMEOUT_S)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return ModelProbeResult(
            reachable=False,
            error=f"PI_PROBE_TIMEOUT: 超过 {_PROBE_TIMEOUT_S:.0f}s 未收敛",
        )
    out = stdout.decode("utf-8", errors="replace").strip()
    payload: dict | None = None
    for line in reversed(out.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and data.get("engine") == "pi":
                payload = data
                break
    if payload is None:
        return ModelProbeResult(
            reachable=False,
            error=(
                f"PI_PROBE_FAILED: rc={proc.returncode} 无有效报告——"
                f"{_sanitize(stderr.decode('utf-8', errors='replace'))}"
            ),
        )
    return ModelProbeResult(
        reachable=bool(payload.get("reachable")),
        models_visible=tuple(str(m) for m in payload.get("models_visible") or ()),
        expected_model_present=bool(payload.get("expected_model_present")),
        chat_ok=bool(payload.get("chat_ok")),
        tool_call_ok=bool(payload.get("tool_call_ok")),
        error=_sanitize(str(payload.get("error") or "")) or None,
    )
