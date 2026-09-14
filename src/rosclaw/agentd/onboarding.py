"""Model onboarding for firstboot / `rosclaw setup model`.

P1-A1（0824 总纲 §10.1）：模型配置与探测**单源**——setup 写
``~/.rosclaw/agent/{settings,models}.json``（Pi ModelRuntime 实际
消费的配置），probe 经 Pi engine（``main.js --probe``，与 chat 同一
ModelRuntime）。不再写 config.yaml 模型段、不再另起 Python HTTP
chat probe。

R0-7（0826 体验审计 §5.R0-7）：readiness 状态格
``UNCONFIGURED / AUTH_READY / CHAT_READY / TOOL_READY /
DEGRADED``——tool probe 失败不覆盖 chat 成功的事实；默认便宜
探测（无严格 tool call），``doctor --deep`` 完整探测。失败诚实
分格——绝不假成功。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rosclaw.agentd.models.gateway import ModelProbeResult
from rosclaw.agentd.models.profiles import (
    KIMI_CN_BASE_URL,
    KIMI_CODE_BASE_URL,
    KIMI_CODE_K3_MODEL,
    KIMI_K3_MODEL,
)
from rosclaw.agentd.pi_config import (
    read_pi_model_config,
    write_pi_model_config,
)
from rosclaw.agentd.pi_probe import pi_probe_home

PROVIDER_CHOICES = ("kimi-code", "kimi-api", "openai-compat", "local", "skip")

_TEMPLATES = {
    "kimi-code": {
        "base_url": KIMI_CODE_BASE_URL,
        "model": KIMI_CODE_K3_MODEL,
        "model_name": "Kimi K3",
        "api_key_ref": "env:ROSCLAW_KIMI_API_KEY",
        "context_window": 262144,
        "max_tokens": 16384,
        "key_hint": "Kimi Coding Plan key (sk-kimi-*) in ROSCLAW_KIMI_API_KEY",
    },
    "kimi-api": {
        "base_url": KIMI_CN_BASE_URL,
        "model": KIMI_K3_MODEL,
        "model_name": "Kimi K3 (open platform)",
        "api_key_ref": "env:MOONSHOT_API_KEY",
        "context_window": 262144,
        "max_tokens": 16384,
        "key_hint": "Moonshot open-platform key in MOONSHOT_API_KEY",
    },
}


def configure_model(
    home: Path,
    choice: str,
    *,
    base_url: str | None = None,
    model: str | None = None,
    api_key_ref: str | None = None,
    reasoning_effort: str = "high",
) -> dict:
    """Write the Pi model config for *choice*. Returns a summary dict."""
    if choice not in PROVIDER_CHOICES:
        raise ValueError(f"unknown provider choice {choice!r}")
    if choice == "skip":
        return {"configured": False, "reason": "user chose to configure later"}
    if choice == "kimi-code" and not base_url and not model:
        # 0914 PR-1（审计 §3.5）：默认映射到 Pi 内置 kimi-coding——
        # 实测等价（2026-08-01）：同一 api.kimi.com/coding/v1、同一
        # OpenAI 兼容协议、同一 Bearer key、模型 ID k3/kimi-for-coding
        # 同服务别名。内置目录让 /login（OAuth/API key）原生可用；
        # 不写 models.json 自定义条目冒充官方服务。自定义网关走
        # openai-compat 显式自定义（保留为 custom provider）。
        settings_path = home / "agent" / "settings.json"
        settings: dict = {}
        if settings_path.exists():
            try:
                settings = json.loads(settings_path.read_text(encoding="utf-8"))
            except ValueError:
                settings = {}
        settings["defaultProvider"] = "kimi-coding"
        settings["defaultModel"] = "kimi-for-coding"
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(
            json.dumps(settings, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if reasoning_effort:
            _write_thinking_level(home, reasoning_effort)
            _write_retry_budget(home)
        return {
            "configured": True,
            "config_path": str(home / "agent"),
            "provider": "kimi-coding",
            "base_url": KIMI_CODE_BASE_URL,
            "model": "kimi-for-coding",
            "api_key_ref": "env:KIMI_API_KEY",
            "key_hint": (
                "chat 内 /login 登录（OAuth 或 API key）；或 export "
                "KIMI_API_KEY（旧别名 ROSCLAW_KIMI_API_KEY 迁移期仍认）"
            ),
        }
    template = _TEMPLATES.get(
        choice,
        {
            "base_url": base_url or "",
            "model": model or "",
            "model_name": model or "",
            "api_key_ref": api_key_ref or "",
            "context_window": 131072,
            "max_tokens": 8192,
            "key_hint": "custom OpenAI-compatible endpoint",
        },
    )
    config = write_pi_model_config(
        home,
        provider=choice,
        base_url=base_url or template["base_url"],
        model=model or template["model"],
        model_name=str(template["model_name"]),
        api_key_ref=api_key_ref or template["api_key_ref"],
        context_window=int(template["context_window"]),
        max_tokens=int(template["max_tokens"]),
    )
    if reasoning_effort:
        # R0-7（0826 体验审计 §2.7/§5.R0-7）：reasoning_effort 真写
        # Pi settings（defaultThinkingLevel——此前被静默忽略）；保留
        # 其他 settings 键。
        _write_thinking_level(home, reasoning_effort)
        # P0-7（0827 审计 §八）：Provider 重试预算进 Pi settings——
        # 配额类确定性错误命中 Pi NON_RETRYABLE 词表时零重试；其余
        # 瞬态错误最多 1 次自动重试（默认 3 次会把确定性 403 烧 4 次）。
        _write_retry_budget(home)
    return {
        "configured": True,
        "config_path": str(home / "agent"),
        "provider": config.provider,
        "base_url": config.base_url,
        "model": config.model,
        "api_key_ref": config.api_key_ref,
        "key_hint": template["key_hint"],
    }


def _write_thinking_level(home: Path, effort: str) -> None:
    """defaultThinkingLevel 写入 agent/settings.json（保留其他键）。"""
    allowed = {"auto", "low", "medium", "high", "off"}
    if effort not in allowed:
        return
    settings_path = home / "agent" / "settings.json"
    settings: dict = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
        except ValueError:
            settings = {}
    if settings.get("defaultThinkingLevel") == effort:
        return
    settings["defaultThinkingLevel"] = effort
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(settings, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_retry_budget(home: Path) -> None:
    """retry.maxRetries=1 写入 agent/settings.json（保留其他键，
    幂等）——P0-7：确定性 provider 错误不得被自动重试烧配额。"""
    settings_path = home / "agent" / "settings.json"
    settings: dict = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
        except ValueError:
            settings = {}
    retry = settings.get("retry")
    if not isinstance(retry, dict):
        retry = {}
    if retry.get("maxRetries") == 1:
        return
    retry["maxRetries"] = 1
    settings["retry"] = retry
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(settings, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


async def probe_home(home: Path, *, deep: bool = False) -> ModelProbeResult:
    """经 Pi engine 探测（与 chat 同一 ModelRuntime、同一配置文件）。

    deep=False：便宜探测（auth + models + chat——默认路径）；
    deep=True：加严格 tool call（doctor --deep）。
    """
    if read_pi_model_config(home) is None:
        return ModelProbeResult(
            reachable=False,
            error="MODEL_NOT_CONFIGURED: 未配置模型——运行 `rosclaw setup model`",
        )
    return await pi_probe_home(home, deep=deep)


def _component_report() -> dict:
    """PR-11 组件检查：Node 与 TUI 资产（P1-A5：modeld 已废除）。"""
    import shutil
    import subprocess

    node_version = None
    node_ok = False
    for candidate in filter(None, [shutil.which("node"), "/usr/bin/node", "/usr/local/bin/node"]):
        try:
            out = subprocess.check_output([candidate, "--version"], text=True, timeout=10).strip()
            parts = [int(p) for p in out.lstrip("v").split(".")]
            if parts >= [22, 19, 0]:
                node_version, node_ok = out, True
                break
            node_version = node_version or out
        except Exception:  # noqa: BLE001
            continue
    from rosclaw.agentd.pi_entry import find_tui_runtime as _find_tui_runtime

    return {
        "node": {"version": node_version, "ok": node_ok, "required": ">=22.19.0"},
        "tui": {"available": _find_tui_runtime() is not None},
    }


def _authorization_report(home: Path) -> dict:
    """审计 P0-01：授权剖面——同 UID 一体运行明确 DEV_SIM_ONLY；
    operatord 缺失时 REAL 硬拒绝（doctor 必须说明）。"""
    from rosclaw.operatord import DEV_SIM_ONLY_LABEL
    from rosclaw.operatord.enrollment import EnrollmentError, read_public_key_pem

    enrolled = False
    fingerprint = None
    try:
        # T0：agentd 只读 0644 公钥——绝不把 operator 私钥加载进本进程。
        from rosclaw.contracts.operator.decision import key_fingerprint

        fingerprint = key_fingerprint(read_public_key_pem(home / "operatord"))
        enrolled = True
    except EnrollmentError:
        pass
    operatord_sock = home / "run" / "operatord.sock"
    running = operatord_sock.exists()
    profile = "OPERATORD_SPLIT" if running else DEV_SIM_ONLY_LABEL
    return {
        "profile": profile,
        "enrolled": enrolled,
        "fingerprint": fingerprint,
        "operatord_socket": str(operatord_sock),
        "operatord_running": running,
        "real_ready": running and enrolled,
        "note": (
            "同 UID 一体运行仅 DEV_SIM_ONLY——REAL 必须 rosclaw-operatord "
            "独立进程 + enrollment + rosclawd ACL"
            if not running
            else "operatord 拆分剖面激活"
        ),
    }


def _pi_engine_report(home: Path) -> dict:
    """重构规格 §27.5 子集：Pi engine 就绪检查（stale dist = FAIL 信号）。"""
    import shutil
    import subprocess as _sp

    entry = (
        Path(__file__).resolve().parents[3]
        / "packages"
        / "rosclaw-agent"
        / "dist"
        / "src"
        / "main.js"
    )
    node_ok = False
    for candidate in filter(None, [shutil.which("node"), "/usr/bin/node"]):
        try:
            out = _sp.check_output([candidate, "--version"], text=True, timeout=10).strip()
            node_ok = [int(p) for p in out.lstrip("v").split(".")] >= [22, 19, 0]
            if node_ok:
                break
        except Exception:  # noqa: BLE001
            continue
    dist_present = entry.exists()
    # stale 检测：dist/main.js 早于任一 src/*.ts 即 stale。
    stale = False
    if dist_present:
        src_dir = entry.parents[2] / "src"
        dist_mtime = entry.stat().st_mtime
        stale = any(p.stat().st_mtime > dist_mtime for p in src_dir.rglob("*.ts"))
    settings = home / "agent" / "settings.json"
    credential_file = home / "agent" / "auth.json"
    return {
        "engine_available": bool(node_ok and dist_present and not stale),
        "node_ok": node_ok,
        "dist_present": dist_present,
        "dist_stale": stale,
        "provider_migrated": settings.exists(),
        "credential_file_present": credential_file.exists(),
        "credential_policy": "developer-file-0600" if credential_file.exists() else "env-only",
        "note": (
            "FAIL: dist 过期（源码新于构建产物）——重新构建发布包，不要手工 npm build"
            if stale
            else "pi engine ready"
            if node_ok and dist_present
            else "pi engine unavailable"
        ),
    }


def _real_tool_success(home: Path) -> bool:
    """chat 真实工具调用成功（账本证据）——探测失败不覆盖真实
    成功事实（0826 旅程：setup 报 NOT_READY 后 chat 真实完成了
    工具调用）。"""
    db_path = home / "agentd" / "missions.db"
    if not db_path.exists():
        return False
    import sqlite3

    try:
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM agent_events "
                "WHERE type = 'tool.completed' AND json_extract("
                "payload_json, '$.ok') = 1 AND json_extract("
                "payload_json, '$.tool_name') IN "
                "('rosclaw_task', 'rosclaw_execute', 'rosclaw_request_action')",
            ).fetchone()
            return bool(row and row[0] > 0)
        finally:
            conn.close()
    except sqlite3.Error:
        return False


def doctor(home: Path, *, deep: bool = False) -> dict:
    """Honest agent readiness report. Never prints raw credentials.

    R0-7（0826 体验审计 §5.R0-7）：
    - 默认便宜探测（models listing + chat，无严格 tool call——
      严格 tool call 是 ``deep=True`` 才跑的完整检查）；
    - 状态格：UNCONFIGURED / AUTH_READY / CHAT_READY /
      TOOL_READY / DEGRADED（chat_ok 但 tool probe 失败——对话
      可用、工具自检退化；tool probe 失败不覆盖 chat 事实）；
    - chat 真实工具调用成功（账本证据）→ TOOL_READY。
    """
    from rosclaw.agentd.config import load_agent_config

    agent_config = load_agent_config(home / "config.yaml")
    model = read_pi_model_config(home)
    report: dict = {
        "agent_enabled": agent_config.enabled,
        "model_backend": "pi",
        "profiles": [f"{model.provider}/{model.model}"] if model else [],
        "default_profile": f"{model.provider}/{model.model}" if model else None,
        "model": (
            {"provider": model.provider, "model": model.model}
            if model
            else {"provider": "", "model": ""}
        ),
    }
    report["components"] = _component_report()
    report["authorization"] = _authorization_report(home)
    # P1-A3：凭据来源只有 env 与 Pi auth.json（NA-FIX-7 可见性保留）。
    from rosclaw.agentd.pi_config import credential_source_report

    report["credential_sources"] = credential_source_report(home)
    if model is None:
        report["status"] = "UNCONFIGURED"
        report["reason"] = "no model profile configured — run `rosclaw setup model`"
        return report
    report["api_key_ref"] = model.api_key_ref
    probe = asyncio.run(probe_home(home, deep=deep))
    report["probe"] = {
        "reachable": probe.reachable,
        "models_visible": list(probe.models_visible),
        "expected_model_present": probe.expected_model_present,
        "chat_ok": probe.chat_ok,
        "tool_call_ok": probe.tool_call_ok,
        "auth_configured": probe.auth_configured,
        "deep": deep,
        "error": probe.error,
    }
    # 0914 PR-1（审计 §3.2/§3.3）：凭据判定单源化——只消费 Pi probe
    # 的解析结果（auth.json/models.json $ENV/env 同一规则），不再
    # 自行读环境变量二次判定。四态分离：凭据存在（credential_present）
    # / 服务端认证与配额（状态格 + quota_state）/ 工具调用（状态格）。
    cred_present = probe.auth_configured
    if cred_present is None:
        # 旧 engine 不上报 auth_configured（升级过渡期）——退回
        # 静态来源枚举（看得到文件/env 存在性）。
        cred_present = any(
            e.get("source") in ("env", "pi-auth-file")
            for e in report["credential_sources"]
        )
    report["credential_present"] = bool(cred_present)
    # 兼容旧字段名（R0-7 报告的 api_key_present）。
    report["api_key_present"] = bool(cred_present)
    err = probe.error or ""
    # R0-7 状态格（不是二元 READY/NOT_READY——tool probe 失败
    # 不覆盖 chat 成功的事实）。
    tool_evidence = _real_tool_success(home)
    report["tool_evidence"] = tool_evidence
    if not cred_present or err.startswith(("AUTH_NOT_CONFIGURED", "MODEL_NOT_CONFIGURED")):
        report["status"] = "UNCONFIGURED"
        report["quota_state"] = "unknown"
        report["reason"] = (
            err
            or "无可用凭据——chat 内 /login 或 `rosclaw setup model` 配置"
        )
    elif err.startswith("QUOTA_EXHAUSTED"):
        # 配额是服务商事实，不是凭据缺失——不得降格 UNCONFIGURED
        # （重新 /login 不会重置额度；指引换已配置模型）。
        report["status"] = "AUTH_READY"
        report["quota_state"] = "exhausted"
        report["reason"] = (
            "凭据已配置；当前配额已用完——用 /model 切换其他已配置模型"
            f"（{err}）"
        )
    elif err.startswith("AUTH_FAILED"):
        report["status"] = "AUTH_READY"
        report["quota_state"] = "unknown"
        report["reason"] = f"凭据已配置但服务端拒绝（{err}）"
    elif err.startswith("RATE_LIMITED"):
        report["status"] = "AUTH_READY"
        report["quota_state"] = "ok"
        report["reason"] = f"限流中，稍后自动恢复（{err}）"
    elif not probe.reachable:
        # 离线/不可达：配置保留，如实说暂时无法连接。
        report["status"] = "AUTH_READY"
        report["quota_state"] = "unknown"
        report["reason"] = (
            f"暂时无法连接——凭据已配置且配置已保留（{err}）"
            if err else "暂时无法连接（凭据已配置，配置已保留）"
        )
    elif probe.chat_ok and (
        (deep and probe.tool_call_ok) or tool_evidence
    ):
        report["quota_state"] = "ok"
        # deep 完整探测通过，或账本有真实工具成功证据。
        report["status"] = "TOOL_READY"
    elif probe.chat_ok and deep and not probe.tool_call_ok:
        report["quota_state"] = "ok"
        report["status"] = "DEGRADED"
        report["reason"] = (
            probe.error
            or "对话可用；工具自检退化（rosclaw doctor --deep 重试）"
        )
    elif probe.chat_ok:
        report["quota_state"] = "ok"
        report["status"] = "CHAT_READY"
    else:
        report["status"] = "AUTH_READY"
        report["quota_state"] = "unknown"
        report["reason"] = err or "chat probe 未通过"
    report["pi_engine"] = _pi_engine_report(home)
    return report
