"""Agentd configuration loading (总纲 §7.2).

Reads the ``agent:`` and ``models:`` sections of ``~/.rosclaw/config.yaml``.
Credentials are references (``env:VAR``) only — never raw keys in YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rosclaw.agentd.models.policy import ModelPolicy, ModelProfile

DEFAULT_CONFIG_PATH = Path.home() / ".rosclaw" / "config.yaml"


@dataclass
class AgentRuntimeConfig:
    """建议-0816 P0-4：Agent Harness 路由冻结——默认 Pi 唯一，
    不因机器上装了别的 Harness 偷换执行者。

    agent_runtime:
      default: pi-sdk
      enabled: [pi-sdk]        # 显式解锁才允许其它 runtime
      auto_discovery: false    # 二进制探测不再影响路由
    """

    default: str = "pi-sdk"
    enabled: list[str] = field(default_factory=lambda: ["pi-sdk"])
    auto_discovery: bool = False


@dataclass
class AgentConfig:
    enabled: bool = True
    # 重构规格 §5/§31 + NA-FIX-9：pi（Pi-backed harness，Native Agent）
    # 是默认；legacy（Python AgentLoop + rosclaw-tui）仅作隐藏回退保留
    # 一个稳定版本（Gate A–E 已全部通过——见 docs 验证矩阵）。
    engine: str = "pi"
    default_profile: str = "embodied_default"
    default_mode: str = "SIMULATION"
    max_tool_rounds: int = 12
    decision_protocol: str = "tool_call"  # tool_call | fenced_json
    legacy_fenced_json_fallback: bool = True
    max_input_tokens: int = 120_000
    dynamic_tool_limit: int = 12
    physical_action_count: int = 0
    language: str = "zh-CN"
    body_id: str | None = None
    sim_body_id: str = "sim/ur5e"
    profiles: list[ModelProfile] = field(default_factory=list)
    model_backend: str = "legacy"  # legacy | modeld（批次 D）
    mcp_servers: list[dict[str, Any]] = field(default_factory=list)
    agent_runtime: AgentRuntimeConfig = field(default_factory=AgentRuntimeConfig)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def active_body_id(self) -> str:
        """Return the configured body, preserving the legacy SIM default."""

        return self.body_id or self.sim_body_id

    def to_policy(self) -> ModelPolicy:
        if not self.profiles:
            raise ValueError("no model profiles configured — run `rosclaw setup model`")
        return ModelPolicy(self.profiles, self.default_profile)


def _profile_from_dict(name: str, data: dict[str, Any]) -> ModelProfile:
    api_key_ref = str(data.get("api_key_ref", ""))
    if "api_key" in data:
        raise ValueError(
            "config.yaml may not contain raw api_key values — use api_key_ref (env:VAR)"
        )
    return ModelProfile(
        name=name,
        provider=str(data.get("provider", "")),
        model=str(data.get("model", "")),
        base_url=str(data.get("base_url", "")),
        api_key_ref=api_key_ref,
        capabilities=tuple(data.get("capabilities", ("llm.chat",))),
        vendor_parameters=dict(data.get("parameters", {})),
        max_output_tokens=int(data.get("budgets", {}).get("max_output_tokens", 16_000)),
        timeout_sec=float(data.get("timeout_sec", 180.0)),
        retry_attempts=int(data.get("retry", {}).get("max_attempts", 3)),
        local=bool(data.get("local", False)),
    )


def load_agent_config(path: Path | None = None) -> AgentConfig:
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return AgentConfig(enabled=False, raw={"missing_config": True})
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    agent = data.get("agent", {}) or {}
    models = data.get("models", {}) or {}
    profiles = [
        _profile_from_dict(name, p or {}) for name, p in (models.get("profiles", {}) or {}).items()
    ]
    context = agent.get("context", {}) or {}
    budgets = agent.get("budgets", {}) or {}
    engine = str(agent.get("engine", "pi")).lower()
    if engine not in ("pi", "legacy"):
        raise ValueError(f"agent.engine 只能是 pi|legacy（got {engine!r}）")
    return AgentConfig(
        enabled=bool(agent.get("enabled", True)),
        engine=engine,
        default_profile=str(agent.get("default_profile", "embodied_default")),
        default_mode=str(agent.get("default_mode", "SIMULATION")),
        max_tool_rounds=int(agent.get("max_tool_rounds", 12)),
        decision_protocol=str(agent.get("decision_protocol", "tool_call")),
        legacy_fenced_json_fallback=bool(agent.get("legacy_fenced_json_fallback", True)),
        max_input_tokens=int(context.get("max_input_tokens", 120_000)),
        dynamic_tool_limit=int(context.get("dynamic_tool_limit", 12)),
        physical_action_count=int(budgets.get("physical_action_count", 0)),
        language=str(agent.get("language", "zh-CN")),
        body_id=(str(agent["body_id"]) if agent.get("body_id") else None),
        sim_body_id=str(agent.get("sim_body_id", "sim/ur5e")),
        profiles=profiles,
        model_backend=str(models.get("backend", "legacy")),
        mcp_servers=[dict(s or {}) for s in (data.get("mcp_servers", []) or [])],
        agent_runtime=AgentRuntimeConfig(
            default=str(data.get("agent_runtime", {}).get("default", "pi-sdk")),
            enabled=list(data.get("agent_runtime", {}).get("enabled", ["pi-sdk"])),
            auto_discovery=bool(
                data.get("agent_runtime", {}).get("auto_discovery", False)
            ),
        ),
        raw={
            "agent": agent,
            "models": models,
            "team": data.get("team", {}) or {},
            # 七审 PR-SEVEN-1：第一方 Robot Kit 开关（disabled 列表）。
            "kits": data.get("kits", {}) or {},
        },
    )


def write_agent_config(
    path: Path,
    *,
    provider_key: str,
    base_url: str,
    model: str,
    api_key_ref: str,
    reasoning_effort: str = "high",
    language: str = "zh-CN",
) -> None:
    """Merge agent/model sections into config.yaml (non-destructive)."""
    existing: dict[str, Any] = {}
    if path.exists():
        existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    existing.setdefault("agent", {})
    existing["agent"].update(
        {
            "enabled": True,
            "default_profile": "embodied_default",
            "max_tool_rounds": 12,
            "default_mode": "SIMULATION",
            "language": language,
            "context": {"max_input_tokens": 120000, "dynamic_tool_limit": 12},
        }
    )
    existing["agent"]["status"] = "CONFIGURED"
    existing["agent"].setdefault("budgets", {"physical_action_count": 0})
    existing.setdefault(
        "team",
        {
            "enabled": False,
            "transport": "local_sim",
            "degraded_policy": "stop_team_actions_keep_local_safety",
        },
    )
    existing.setdefault("models", {}).setdefault("providers", {})
    existing["models"]["providers"][provider_key] = {
        "runtime": "openai_compat",
        "base_url": base_url,
        "api_key_ref": api_key_ref,
        "timeout_sec": 180,
        "retry": {"max_attempts": 3, "retry_on": [408, 429, 500, 502, 503, 504]},
    }
    existing["models"].setdefault("profiles", {})
    existing["models"]["profiles"]["embodied_default"] = {
        "provider": provider_key,
        "model": model,
        "base_url": base_url,
        "api_key_ref": api_key_ref,
        "capabilities": [
            "llm.chat",
            "llm.tool_use",
            "llm.structured_decision",
            "llm.long_context",
        ],
        "parameters": {"reasoning_effort": reasoning_effort},
        "budgets": {"max_output_tokens": 16000},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(existing, allow_unicode=True, sort_keys=False), encoding="utf-8")
