"""ROSClaw Channel Doctor（Channel 设计 §46 / PR-RC-005）。

**只读检查**：不修改任何 OpenClaw / ROSClaw 配置。唯一的副作用是 ACP
探测会创建并随即 close 一个 probe Mission（journal 保留，输出中明确标注）。

检查分两层：

1. ROSClaw 侧（必须全绿才算 READY）：可执行环境、ROSCLAW_HOME、模型配置、
   ACP initialize / session/new / session/resume / session/close。
2. OpenClaw 侧（缺失记 SKIP 而不是 FAIL，除非 ``require_openclaw=True``）：
   Node、OpenClaw、acpx、harness 注册、Gateway loopback/auth、飞书策略、
   MCP bridge 关闭、permissionMode=deny-all。

OpenClaw 配置 schema 仍在演进（设计 §33 注），config 探针统一走
``openclaw config get <key>`` 并容忍输出格式差异——schema 的权威校验
始终属于 ``openclaw doctor``。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

# 状态：OK / WARN / SKIP / FAIL
_OK, _WARN, _SKIP, _FAIL = "OK", "WARN", "SKIP", "FAIL"

# OpenClaw 的 Node 支持窗口（设计 §7）：22.22.3+ / 24.15+ / 25.9+
_NODE_MIN = {22: (22, 22, 3), 24: (24, 15, 0), 25: (25, 9, 0)}


def _node_supported(version_text: str) -> bool:
    try:
        parts = tuple(int(p) for p in version_text.lstrip("v").split(".")[:3])
    except ValueError:
        return False
    parts = parts + (0,) * (3 - len(parts))
    major = parts[0]
    if major in _NODE_MIN:
        return parts >= _NODE_MIN[major]
    return major > max(_NODE_MIN)


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""


@dataclass
class DoctorReport:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.checks.append(Check(name, status, detail))

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status == _FAIL]

    @property
    def skipped(self) -> list[Check]:
        return [c for c in self.checks if c.status == _SKIP]

    def render(self) -> str:
        lines = ["ROSClaw Channel Doctor", ""]
        for check in self.checks:
            suffix = f" — {check.detail}" if check.detail else ""
            lines.append(f"[{check.status}] {check.name}{suffix}")
        lines.append("")
        if self.failed:
            lines.append("NOT READY")
        else:
            lines.append("READY" if not self.skipped else "READY (with skips)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ROSClaw 侧
# ---------------------------------------------------------------------------


def _check_rosclaw_runtime(report: DoctorReport) -> None:
    try:
        import rosclaw  # noqa: F401

        report.add("rosclaw executable", _OK, sys.executable)
    except Exception as exc:  # noqa: BLE001
        report.add("rosclaw executable", _FAIL, str(exc))


def _check_home(report: DoctorReport, home: Path) -> None:
    if home.is_dir():
        report.add("ROSCLAW_HOME", _OK, str(home))
    else:
        report.add("ROSCLAW_HOME", _FAIL, f"{home} 不存在——先运行 `rosclaw agent init`")


def _check_model_config(report: DoctorReport, home: Path) -> bool:
    from rosclaw.agentd.config import load_agent_config

    config = load_agent_config(home / "config.yaml")
    if not config.profiles:
        report.add(
            "Native Agent model",
            _FAIL,
            "未配置模型——先运行 `rosclaw setup model`",
        )
        return False
    report.add("Native Agent model", _OK, f"{len(config.profiles)} profile(s)")
    return True


def _check_credentials(report: DoctorReport, home: Path) -> None:
    from rosclaw.agentd.credentials import AgentCredentialStore, CredentialStoreError

    try:
        # inject() 只把凭证写进本进程 env（doctor 进程随即退出），并校验
        # credential store 的 owner/权限——正是这里要查的安全属性。
        injected = AgentCredentialStore(home).inject()
    except CredentialStoreError as exc:
        report.add("Agent credentials", _FAIL, str(exc))
        return
    if injected:
        report.add("Agent credentials", _OK, f"{len(injected)} credential(s) injected")
    else:
        report.add("Agent credentials", _WARN, "credential store 为空（或仅凭 env）")


# ---------------------------------------------------------------------------
# ACP 握手探测（真实子进程 + 原始 JSON-RPC 帧）
# ---------------------------------------------------------------------------


async def _rpc(proc, method: str, params: dict, req_id: int, timeout: float) -> dict:
    frame = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
    proc.stdin.write((json.dumps(frame) + "\n").encode())
    await proc.stdin.drain()
    while True:
        line = await asyncio.wait_for(proc.stdout.readline(), timeout=timeout)
        if not line:
            raise RuntimeError(f"{method}: ACP server stdout EOF")
        reply = json.loads(line)  # 非 JSON 输出在此直接失败（stdout 纯净性）
        if reply.get("id") != req_id:
            continue  # 跳过 notification
        if "error" in reply:
            raise RuntimeError(f"{method}: {reply['error'].get('message', reply['error'])}")
        return reply.get("result") or {}


async def _acp_probe(
    report: DoctorReport,
    home: Path,
    timeout: float,
    command: list[str] | None = None,
    env: dict | None = None,
) -> None:
    if command is None:
        command = [
            sys.executable,
            "-m",
            "rosclaw.entrypoint",
            "acp",
            "serve",
            "--home",
            str(home),
        ]
    if env is None:
        env = dict(os.environ)
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    try:
        result = await _rpc(
            proc,
            "initialize",
            {"protocolVersion": 1, "clientCapabilities": {}},
            0,
            timeout,
        )
        name = (result.get("agentInfo") or {}).get("name", "?")
        report.add("ACP initialize", _OK, name)

        session = await _rpc(proc, "session/new", {"cwd": "/", "mcpServers": []}, 1, timeout)
        sid = session.get("sessionId", "")
        report.add(
            "ACP session/new",
            _OK if sid else _FAIL,
            f"{sid}（probe Mission，随后 close）" if sid else "no sessionId",
        )
        if sid:
            await _rpc(
                proc,
                "session/resume",
                {"sessionId": sid, "cwd": "/", "mcpServers": []},
                2,
                timeout,
            )
            report.add("ACP session/resume", _OK, sid)
            await _rpc(proc, "session/close", {"sessionId": sid}, 3, timeout)
            report.add("ACP session/close", _OK, "Mission journal 保留")
    except (TimeoutError, RuntimeError, json.JSONDecodeError, OSError) as exc:
        # 附上 server stderr 末尾——initialize EOF 这类错误的真因（如缺凭证）
        # 只在 stderr 上可见。
        detail = str(exc)
        with contextlib.suppress(Exception):
            err = (await asyncio.wait_for(proc.stderr.read(), timeout=2)).decode(errors="replace")
            tail = [line for line in err.strip().splitlines() if line.strip()]
            if tail:
                detail = f"{detail} | stderr: {tail[-1][:160]}"
        report.add("ACP probe", _FAIL, detail)
    finally:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()  # 进程可能已在 probe 失败前先退出
        await proc.wait()


# ---------------------------------------------------------------------------
# OpenClaw 侧（全部 best-effort；缺失记 SKIP）
# ---------------------------------------------------------------------------


async def _run(cmd: list[str], timeout: float) -> tuple[int, str]:
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode or 0, (out or err).decode(errors="replace").strip()
    except (TimeoutError, OSError) as exc:
        return -1, str(exc)


async def _openclaw_config_get(key: str, timeout: float) -> str | None:
    """`openclaw config get <key>` 的容忍解析：失败/为空返回 None。"""
    code, out = await _run(["openclaw", "config", "get", key], timeout)
    if code != 0 or not out:
        return None
    text = out.splitlines()[-1].strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = text.strip('"').strip("'")
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


async def _check_openclaw(report: DoctorReport, timeout: float) -> None:
    # Node
    node = shutil.which("node")
    if node is None:
        report.add("Node", _SKIP, "node 不在 PATH（设计 §7：需要 22.22.3+/24.15+/25.9+）")
    else:
        code, out = await _run(["node", "-v"], timeout)
        if code == 0 and out and _node_supported(out):
            report.add("Node", _OK, out)
        else:
            report.add(
                "Node",
                _FAIL,
                f"{out or 'unknown'} 不满足 >=22.22.3 <23 / >=24.15.0 <25 / >=25.9.0",
            )

    # OpenClaw
    if shutil.which("openclaw") is None:
        report.add(
            "OpenClaw",
            _SKIP,
            "openclaw 不在 PATH——见 integrations/openclaw/README.md 安装指引",
        )
        for name in (
            "@openclaw/acpx",
            "rosclaw harness",
            "Gateway loopback",
            "Gateway auth",
            "Feishu plugin",
            "Feishu pairing/allowlist policy",
            "OpenClaw MCP bridges disabled",
            "ACPX permissionMode deny-all",
        ):
            report.add(name, _SKIP, "openclaw 不可用")
        return

    code, out = await _run(["openclaw", "--version"], timeout)
    report.add("OpenClaw", _OK if code == 0 else _FAIL, out.splitlines()[0] if out else "")

    # 配置探针（schema 演进中——取不到记 SKIP，不臆测）
    async def _probe(name: str, key: str, expect: str | None = None) -> None:
        value = await _openclaw_config_get(key, timeout)
        if value is None:
            report.add(name, _SKIP, f"`openclaw config get {key}` 无结果")
        elif expect is not None and value.lower() != expect.lower():
            report.add(name, _FAIL, f"{key}={value!r}，期望 {expect!r}")
        else:
            report.add(name, _OK, f"{key}={value}")

    await _probe("@openclaw/acpx", "plugins.entries.acpx.enabled", "true")
    harness = await _openclaw_config_get(
        "plugins.entries.acpx.config.agents.rosclaw.command", timeout
    )
    if harness:
        status = _OK if Path(harness).is_absolute() else _FAIL
        detail = "" if status == _OK else "必须使用绝对路径（设计 §9）"
        report.add("rosclaw harness", status, f"{harness} {detail}".strip())
    else:
        report.add("rosclaw harness", _SKIP, "未注册 agents.rosclaw")
    await _probe("Gateway loopback", "gateway.bind", "loopback")
    auth = await _openclaw_config_get("gateway.auth.mode", timeout)
    if auth is None:
        report.add("Gateway auth", _SKIP, "gateway.auth.mode 无结果")
    elif auth.lower() in ("none", "", "off"):
        report.add("Gateway auth", _FAIL, "gateway 无认证（设计 §13 禁止）")
    else:
        report.add("Gateway auth", _OK, f"mode={auth}")
    await _probe("Feishu plugin", "channels.feishu.enabled", "true")
    dm = await _openclaw_config_get("channels.feishu.dmPolicy", timeout)
    group = await _openclaw_config_get("channels.feishu.groupPolicy", timeout)
    mention = await _openclaw_config_get("channels.feishu.requireMention", timeout)
    if dm is None and group is None:
        report.add("Feishu pairing/allowlist policy", _SKIP, "飞书策略无结果")
    elif dm in ("pairing", "allowlist") and group == "allowlist" and mention == "true":
        # 设计 §16 推荐 DM=pairing；allowlist 是等价可接受的收紧（显式名单）。
        report.add(
            "Feishu pairing/allowlist policy", _OK, f"dm={dm} + group=allowlist + mention"
        )
    else:
        report.add(
            "Feishu pairing/allowlist policy",
            _FAIL,
            f"dmPolicy={dm} groupPolicy={group} requireMention={mention}"
            "（设计 §16：pairing + allowlist + mention）",
        )
    await _probe(
        "OpenClaw MCP bridges disabled",
        "plugins.entries.acpx.config.pluginToolsMcpBridge",
        "false",
    )
    await _probe(
        "ACPX permissionMode deny-all",
        "plugins.entries.acpx.config.permissionMode",
        "deny-all",
    )


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


async def run_doctor_async(
    home: Path,
    *,
    require_openclaw: bool = False,
    timeout: float = 15.0,
    probe_acp: bool = True,
    acp_command: list[str] | None = None,
    acp_env: dict | None = None,
) -> DoctorReport:
    report = DoctorReport()
    _check_rosclaw_runtime(report)
    _check_home(report, home)
    if home.is_dir():
        _check_credentials(report, home)
        if _check_model_config(report, home) and probe_acp:
            await _acp_probe(report, home, timeout, command=acp_command, env=acp_env)
    await _check_openclaw(report, timeout)
    if require_openclaw:
        for check in report.checks:
            if check.status == _SKIP:
                check.status = _FAIL
                check.detail = f"{check.detail}（--require-openclaw）".strip()
    return report


def run_doctor(
    home: Path,
    *,
    require_openclaw: bool = False,
    timeout: float = 15.0,
    probe_acp: bool = True,
    acp_command: list[str] | None = None,
    acp_env: dict | None = None,
) -> DoctorReport:
    return asyncio.run(
        run_doctor_async(
            home,
            require_openclaw=require_openclaw,
            timeout=timeout,
            probe_acp=probe_acp,
            acp_command=acp_command,
            acp_env=acp_env,
        )
    )
