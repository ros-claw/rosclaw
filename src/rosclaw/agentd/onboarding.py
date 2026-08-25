"""Model onboarding for firstboot / `rosclaw setup model`.

P1-A1（0824 总纲 §10.1）：模型配置与探测**单源**——setup 写
``~/.rosclaw/agent/{settings,models}.json``（Pi ModelRuntime 实际
消费的配置），probe 经 Pi engine（``main.js --probe``，与 chat 同一
ModelRuntime）。不再写 config.yaml 模型段、不再另起 Python HTTP
chat probe。失败诚实记为 ``MODEL_NOT_READY``——绝不假成功。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from rosclaw.agentd.models.gateway import ModelProbeResult, key_fingerprint
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
    if reasoning_effort != "high":
        # thinking level 由 Pi settings 自持（/effort）——setup 不复制。
        pass
    return {
        "configured": True,
        "config_path": str(home / "agent"),
        "provider": config.provider,
        "base_url": config.base_url,
        "model": config.model,
        "api_key_ref": config.api_key_ref,
        "key_hint": template["key_hint"],
    }


async def probe_home(home: Path) -> ModelProbeResult:
    """经 Pi engine 探测（与 chat 同一 ModelRuntime、同一配置文件）。"""
    if read_pi_model_config(home) is None:
        return ModelProbeResult(
            reachable=False,
            error="MODEL_NOT_CONFIGURED: 未配置模型——运行 `rosclaw setup model`",
        )
    return await pi_probe_home(home)


def _component_report() -> dict:
    """批次 D/E 与 PR-11 组件检查：Node、modeld、TUI 资产。"""
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
    from rosclaw.agentd.models.modeld_gateway import _find_modeld_runtime
    from rosclaw.agentd.pi_entry import find_tui_runtime as _find_tui_runtime

    return {
        "node": {"version": node_version, "ok": node_ok, "required": ">=22.19.0"},
        "modeld": {"available": _find_modeld_runtime() is not None},
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


def doctor(home: Path) -> dict:
    """Honest agent readiness report. Never prints raw credentials."""
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
    if model is None:
        report["status"] = "MODEL_NOT_READY"
        report["reason"] = "no model profile configured — run `rosclaw setup model`"
        return report
    import os

    key = ""
    if model.api_key_ref.startswith("env:"):
        key = os.environ.get(model.api_key_ref[4:], "")
    report["api_key_ref"] = model.api_key_ref
    report["api_key_present"] = bool(key)
    if key:
        report["api_key_fingerprint"] = key_fingerprint(key)
    probe = asyncio.run(probe_home(home))
    report["probe"] = {
        "reachable": probe.reachable,
        "models_visible": list(probe.models_visible),
        "expected_model_present": probe.expected_model_present,
        "chat_ok": probe.chat_ok,
        "tool_call_ok": probe.tool_call_ok,
        "error": probe.error,
    }
    ready = bool(
        probe.reachable and probe.chat_ok and probe.tool_call_ok and (key or not model.api_key_ref)
    )
    report["status"] = "READY" if ready else "MODEL_NOT_READY"
    if not ready and not probe.error:
        report["reason"] = "probe incomplete (see probe fields)"
    elif probe.error:
        report["reason"] = probe.error
    report["pi_engine"] = _pi_engine_report(home)
    # NA-FIX-7：凭据来源可见（无 secret 内容，仅来源与指纹）。
    try:
        from rosclaw.agentd.credentials import ModelCredentialBroker

        report["credential_sources"] = ModelCredentialBroker(home).source_report()
    except Exception as exc:  # noqa: BLE001
        report["credential_sources"] = [{"error": str(exc)}]
    return report
