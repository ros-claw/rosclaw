"""ModelPolicy — which registered profile serves a task (总纲 §7.1).

The policy only *selects* among already-registered provider/model profiles;
invocation, retries, tracing and protocol quirks stay in the provider layer
and the gateway. Selection inputs: required capabilities, privacy, latency,
cost. The default profile must declare ``llm.structured_decision``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rosclaw.contracts.common import ValidationError

CAP_CHAT = "llm.chat"
CAP_TOOL_USE = "llm.tool_use"
CAP_STRUCTURED = "llm.structured_decision"
CAP_LONG_CONTEXT = "llm.long_context"
CAP_VLM = "vlm.scene_reasoning"


@dataclass(frozen=True)
class ModelProfile:
    name: str
    provider: str  # registered provider key, e.g. "kimi_cn"
    model: str
    base_url: str = ""
    api_key_ref: str = ""  # e.g. env:MOONSHOT_API_KEY — never the key itself
    capabilities: tuple[str, ...] = (CAP_CHAT,)
    vendor_parameters: dict = field(default_factory=dict)
    max_output_tokens: int = 16_000
    timeout_sec: float = 180.0
    retry_attempts: int = 3
    local: bool = False  # privacy: data never leaves the machine
    # Pricing in microunits (1e-6 CNY) per million tokens; 0 = unmetered.
    price_input_per_mtok_microunits: int = 0
    price_output_per_mtok_microunits: int = 0


class ModelPolicy:
    def __init__(self, profiles: list[ModelProfile], default_profile: str) -> None:
        self._profiles = {p.name: p for p in profiles}
        if default_profile not in self._profiles:
            raise ValidationError(f"default profile {default_profile!r} not registered")
        self._default = default_profile

    @property
    def default(self) -> ModelProfile:
        return self._profiles[self._default]

    def fallback_chain(
        self, *, required: tuple[str, ...] = (CAP_STRUCTURED,)
    ) -> list[ModelProfile]:
        """Ordered failover candidates: default first, then other capable
        profiles (stable name order). Cooldown/RPM gating happens in
        FailoverGateway; this is just the static order."""
        capable = [p for p in self._profiles.values() if set(required) <= set(p.capabilities)]
        capable.sort(key=lambda p: (p.name != self._default, p.name))
        return capable

    def select(
        self,
        *,
        required: tuple[str, ...] = (CAP_STRUCTURED,),
        prefer_local: bool = False,
        profile: str | None = None,
    ) -> ModelProfile:
        if profile is not None:
            chosen = self._profiles.get(profile)
            if chosen is None:
                raise ValidationError(f"unknown model profile {profile!r}")
            missing = set(required) - set(chosen.capabilities)
            if missing:
                raise ValidationError(f"profile {profile!r} lacks capabilities {sorted(missing)}")
            return chosen
        candidates = [p for p in self._profiles.values() if set(required) <= set(p.capabilities)]
        if prefer_local:
            locals_only = [p for p in candidates if p.local]
            if locals_only:
                candidates = locals_only
        if not candidates:
            raise ValidationError(
                f"no registered profile satisfies capabilities {sorted(required)}"
            )
        if self._profiles[self._default] in candidates:
            return self._profiles[self._default]
        return sorted(candidates, key=lambda p: p.name)[0]
