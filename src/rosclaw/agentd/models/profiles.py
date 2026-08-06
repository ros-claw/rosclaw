"""First-class model profile factories (PR-NA-031).

Profiles carry credential *references* only. Kimi K3 uses the standard
OpenAI-compatible runtime — no private SDK. Kimi Code / 会员 keys use a
different product surface than the open-platform API; firstboot must match
endpoint to key type and probe with the same endpoint.
"""

from __future__ import annotations

from rosclaw.agentd.models.policy import (
    CAP_CHAT,
    CAP_LONG_CONTEXT,
    CAP_STRUCTURED,
    CAP_TOOL_USE,
    ModelProfile,
)

KIMI_CN_BASE_URL = "https://api.moonshot.cn/v1"
KIMI_K3_MODEL = "kimi-k3"

# Kimi Code (Coding Plan) — verified 2026-08-01 via direct probe:
# OpenAI-compatible /v1/models + /v1/chat/completions, strict tool calls OK,
# reasoning_effort accepted (low/high/max), model ids: k3 (1M ctx), k3-256k,
# kimi-for-coding*. Keys look like sk-kimi-* and are NOT interchangeable with
# the open-platform keys above.
KIMI_CODE_BASE_URL = "https://api.kimi.com/coding/v1"
KIMI_CODE_K3_MODEL = "k3"


def kimi_k3_profile(
    *,
    api_key_ref: str = "env:MOONSHOT_API_KEY",
    base_url: str = KIMI_CN_BASE_URL,
    reasoning_effort: str = "high",
    name: str = "embodied_default",
) -> ModelProfile:
    return ModelProfile(
        name=name,
        provider="kimi_cn",
        model=KIMI_K3_MODEL,
        base_url=base_url,
        api_key_ref=api_key_ref,
        capabilities=(CAP_CHAT, CAP_TOOL_USE, CAP_STRUCTURED, CAP_LONG_CONTEXT),
        vendor_parameters={"reasoning_effort": reasoning_effort},
        max_output_tokens=16_000,
        timeout_sec=180.0,
        retry_attempts=3,
    )


def kimi_code_k3_profile(
    *,
    api_key_ref: str = "env:ROSCLAW_KIMI_API_KEY",
    base_url: str = KIMI_CODE_BASE_URL,
    reasoning_effort: str = "high",
    name: str = "embodied_default",
) -> ModelProfile:
    """Kimi Coding Plan K3 (sk-kimi-* keys) via its OpenAI-compatible surface."""
    return ModelProfile(
        name=name,
        provider="kimi_code",
        model=KIMI_CODE_K3_MODEL,
        base_url=base_url,
        api_key_ref=api_key_ref,
        capabilities=(CAP_CHAT, CAP_TOOL_USE, CAP_STRUCTURED, CAP_LONG_CONTEXT),
        vendor_parameters={"reasoning_effort": reasoning_effort},
        max_output_tokens=16_000,
        timeout_sec=180.0,
        retry_attempts=3,
    )


def mock_profile(*, name: str = "mock_default", model: str = "mock-model") -> ModelProfile:
    """Profile for deterministic tests / offline development."""
    return ModelProfile(
        name=name,
        provider="mock",
        model=model,
        base_url="",
        api_key_ref="",
        capabilities=(CAP_CHAT, CAP_TOOL_USE, CAP_STRUCTURED, CAP_LONG_CONTEXT),
        local=True,
    )
