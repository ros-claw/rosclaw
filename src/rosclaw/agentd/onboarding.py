"""Model onboarding for firstboot / `rosclaw agent init` (PR-NA-041).

Writes credential *references* only, then runs the four firstboot probes
(connectivity, model listing, short chat, strict tool call) with the same
endpoint the agent will use. Failure is recorded honestly as
``MODEL_NOT_READY`` — firstboot may complete, but never with fake success.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from rosclaw.agentd.config import load_agent_config, write_agent_config
from rosclaw.agentd.models.gateway import (
    ModelGatewayError,
    ModelProbeResult,
    OpenAICompatGateway,
    key_fingerprint,
)
from rosclaw.agentd.models.policy import ModelProfile
from rosclaw.agentd.models.profiles import (
    KIMI_CN_BASE_URL,
    KIMI_CODE_BASE_URL,
    KIMI_CODE_K3_MODEL,
    KIMI_K3_MODEL,
)

PROVIDER_CHOICES = ("kimi-code", "kimi-api", "openai-compat", "local", "skip")

_TEMPLATES = {
    "kimi-code": {
        "provider_key": "kimi_code",
        "base_url": KIMI_CODE_BASE_URL,
        "model": KIMI_CODE_K3_MODEL,
        "api_key_ref": "env:ROSCLAW_KIMI_API_KEY",
        "key_hint": "Kimi Coding Plan key (sk-kimi-*) in ROSCLAW_KIMI_API_KEY",
    },
    "kimi-api": {
        "provider_key": "kimi_cn",
        "base_url": KIMI_CN_BASE_URL,
        "model": KIMI_K3_MODEL,
        "api_key_ref": "env:MOONSHOT_API_KEY",
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
    """Write the agent/model config for *choice*. Returns a summary dict."""
    if choice not in PROVIDER_CHOICES:
        raise ValueError(f"unknown provider choice {choice!r}")
    if choice == "skip":
        return {"configured": False, "reason": "user chose to configure later"}
    template = _TEMPLATES.get(
        choice,
        {
            "provider_key": choice.replace("-", "_"),
            "base_url": base_url or "",
            "model": model or "",
            "api_key_ref": api_key_ref or "",
            "key_hint": "custom OpenAI-compatible endpoint",
        },
    )
    config_path = home / "config.yaml"
    write_agent_config(
        config_path,
        provider_key=template["provider_key"],
        base_url=base_url or template["base_url"],
        model=model or template["model"],
        api_key_ref=api_key_ref or template["api_key_ref"],
        reasoning_effort=reasoning_effort,
    )
    return {
        "configured": True,
        "config_path": str(config_path),
        "provider": template["provider_key"],
        "base_url": base_url or template["base_url"],
        "model": model or template["model"],
        "api_key_ref": api_key_ref or template["api_key_ref"],
        "key_hint": template["key_hint"],
    }


def _profile_from_config(home: Path) -> ModelProfile:
    config = load_agent_config(home / "config.yaml")
    return config.to_policy().default


async def probe_home(home: Path) -> ModelProbeResult:
    try:
        profile = _profile_from_config(home)
        gateway = OpenAICompatGateway(profile)
    except Exception as exc:  # noqa: BLE001 - doctor must report, not crash
        return ModelProbeResult(reachable=False, error=f"config/credential: {exc}")
    try:
        return await gateway.probe()
    except ModelGatewayError as exc:
        return ModelProbeResult(reachable=False, error=f"{exc.kind}: {exc}")
    finally:
        await gateway.close()


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
    from rosclaw.agentd.cli import _find_tui_runtime
    from rosclaw.agentd.models.modeld_gateway import _find_modeld_runtime

    return {
        "node": {"version": node_version, "ok": node_ok, "required": ">=22.19.0"},
        "modeld": {"available": _find_modeld_runtime() is not None},
        "tui": {"available": _find_tui_runtime() is not None},
    }


def doctor(home: Path) -> dict:
    """Honest agent readiness report. Never prints raw credentials."""
    config = load_agent_config(home / "config.yaml")
    report: dict = {
        "agent_enabled": config.enabled,
        "model_backend": config.model_backend,
        "profiles": [p.name for p in config.profiles],
        "default_profile": config.default_profile if config.profiles else None,
    }
    report["components"] = _component_report()
    if not config.profiles:
        report["status"] = "MODEL_NOT_READY"
        report["reason"] = "no model profile configured — run `rosclaw agent init`"
        return report
    profile = config.to_policy().default
    import os

    key = ""
    if profile.api_key_ref.startswith("env:"):
        key = os.environ.get(profile.api_key_ref[4:], "")
    report["api_key_ref"] = profile.api_key_ref
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
    ready = bool(probe.reachable and probe.chat_ok and probe.tool_call_ok and key)
    report["status"] = "READY" if ready else "MODEL_NOT_READY"
    if not ready and not probe.error:
        report["reason"] = "probe incomplete (see probe fields)"
    elif probe.error:
        report["reason"] = probe.error
    return report
