"""Pi 模型配置单源（P1-A1，0824 总纲 §10.1）。

``~/.rosclaw/agent/settings.json``（defaultProvider/defaultModel）+
``~/.rosclaw/agent/models.json``（provider 目录：baseUrl/api/models）
是 chat（Pi ModelRuntime）实际消费的唯一配置。本模块是 Python 侧
对该配置的**唯一**读写点——setup/doctor/chat gate/status 全从这里
取事实，不再另读 config.yaml 模型段。

安全红线：apiKey 只写 ``$ENV_VAR`` 引用（与 Pi models.json 的 env
间接寻址一致）——原始 key 永远不落盘；``api_key_ref`` 只接受
``env:VAR`` 形式。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

#: apiKey 的 $ENV 引用形式（Pi models.json 约定）。
_ENV_REF_RE = re.compile(r"^\$([A-Z][A-Z0-9_]*)$")


@dataclass(frozen=True)
class PiModelConfig:
    """无 secret 的模型配置视图。"""

    provider: str
    model: str
    base_url: str
    api: str
    api_key_ref: str  # "env:VAR" 或 ""（local 无 key）
    context_window: int
    max_tokens: int


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_pi_model_config(home: Path) -> PiModelConfig | None:
    """读取 chat 实际消费的模型配置；未配置返回 None。"""
    agent_dir = home / "agent"
    settings = _read_json(agent_dir / "settings.json")
    provider = settings.get("defaultProvider")
    model = settings.get("defaultModel")
    if not provider or not model:
        return None
    providers = _read_json(agent_dir / "models.json").get("providers") or {}
    entry = providers.get(provider)
    if not isinstance(entry, dict):
        return None
    api_key_ref = ""
    raw_ref = str(entry.get("apiKey") or "")
    match = _ENV_REF_RE.match(raw_ref)
    if match:
        api_key_ref = f"env:{match.group(1)}"
    elif raw_ref:
        # 非 $ENV 形式 = 疑似原始 key 落盘——绝不回显内容，只报事实。
        api_key_ref = "inline:REDACTED"
    context_window = 0
    max_tokens = 0
    for m in entry.get("models") or []:
        if isinstance(m, dict) and m.get("id") == model:
            context_window = int(m.get("contextWindow") or 0)
            max_tokens = int(m.get("maxTokens") or 0)
            break
    return PiModelConfig(
        provider=str(provider),
        model=str(model),
        base_url=str(entry.get("baseUrl") or ""),
        api=str(entry.get("api") or "openai-completions"),
        api_key_ref=api_key_ref,
        context_window=context_window,
        max_tokens=max_tokens,
    )


def pi_model_configured(home: Path) -> bool:
    return read_pi_model_config(home) is not None


#: provider → 认可的 env 凭据键（P1-A3：credential 单源报告）。
PROVIDER_ENV_KEYS = {
    "kimi-code": ("ROSCLAW_KIMI_API_KEY", "KIMI_API_KEY"),
    "kimi-api": ("MOONSHOT_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
}


def credential_source_report(home: Path) -> list[dict]:
    """凭据来源报告（P1-A3）——只有 env 与 Pi auth.json 两个来源。

    永不打印 secret 内容（只指纹前 8 位）。legacy
    ``agentd/credentials.json`` 只报"已停用"事实（env 名可列——
    值绝不读取/展示/注入）。
    """
    import hashlib
    import os

    report: list[dict] = []
    for provider, keys in PROVIDER_ENV_KEYS.items():
        entry: dict = {"provider": provider, "source": "none", "env_name": keys[0]}
        for key in keys:
            value = os.environ.get(key, "")
            if value:
                entry = {
                    "provider": provider,
                    "source": "env",
                    "env_name": key,
                    "fingerprint": hashlib.sha256(value.encode()).hexdigest()[:8],
                }
                break
        report.append(entry)
    auth_path = home / "agent" / "auth.json"
    auth = _read_json(auth_path)
    for provider in auth:
        report.append({"provider": str(provider), "source": "pi-auth-file"})
    legacy_path = home / "agentd" / "credentials.json"
    if legacy_path.exists():
        names: list[str] = []
        legacy = _read_json(legacy_path)
        env_block = legacy.get("environment")
        if isinstance(env_block, dict):
            names = sorted(str(k) for k in env_block)
        report.append(
            {
                "provider": "(legacy)",
                "source": "legacy-disabled",
                "env_names": names,
                "note": "agentd/credentials.json 已停用（不再读取/注入）——"
                "请用 env 或 chat 内 /login",
            }
        )
    return report


def write_pi_model_config(
    home: Path,
    *,
    provider: str,
    base_url: str,
    model: str,
    api_key_ref: str = "",
    model_name: str = "",
    context_window: int = 131072,
    max_tokens: int = 8192,
    api: str = "openai-completions",
) -> PiModelConfig:
    """合并写入 Pi 配置（幂等、非破坏：保留 settings/models 其他键）。

    ``api_key_ref`` 必须是 ``env:VAR``（或 local 留空）——原始 key
    材料直接拒绝（安全红线：key 只在环境变量里）。
    """
    if api_key_ref:
        if not api_key_ref.startswith("env:") or not api_key_ref[4:]:
            raise ValueError("api_key_ref 必须是 env:VAR 引用——原始 key 绝不落盘")
        api_key_json = f"${api_key_ref[4:]}"
    else:
        api_key_json = ""
    agent_dir = home / "agent"
    settings = _read_json(agent_dir / "settings.json")
    settings["defaultProvider"] = provider
    settings["defaultModel"] = model
    _write_json(agent_dir / "settings.json", settings)

    models_doc = _read_json(agent_dir / "models.json")
    providers = models_doc.setdefault("providers", {})
    entry = providers.get(provider) if isinstance(providers.get(provider), dict) else {}
    model_entry = {
        "id": model,
        "name": model_name or model,
        "contextWindow": context_window,
        "maxTokens": max_tokens,
    }
    existing = [
        m for m in (entry.get("models") or []) if isinstance(m, dict) and m.get("id") != model
    ]
    entry.update(
        {
            "name": entry.get("name") or provider,
            "baseUrl": base_url,
            "api": api,
            "models": [*existing, model_entry],
        }
    )
    if api_key_json:
        entry["apiKey"] = api_key_json
    elif "apiKey" not in entry:
        entry["apiKey"] = ""
    providers[provider] = entry
    _write_json(agent_dir / "models.json", models_doc)
    return PiModelConfig(
        provider=provider,
        model=model,
        base_url=base_url,
        api=api,
        api_key_ref=api_key_ref,
        context_window=context_window,
        max_tokens=max_tokens,
    )
