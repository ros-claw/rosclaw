"""Cooldown tracking + ordered failover chain (picoclaw cooldown.go/fallback.go).

规则（调研报告 §picoclaw/zeroclaw）：
- 标准错误冷却 `min(1h, 1min * 5^(n-1))`，n 上限 3；billing/auth 类长冷却
  （5h * 2^n 封顶 24h）；24h 无失败自动清零；成功即复位。
- 冷却键用 profile 身份（provider/model/base_url），支持多 key 别名独立熔断。
- format / auth / context_overflow 不重试（立即抛）；rate_limited / timeout /
  unavailable / 5xx 进入下一个候选；最后候选也失败才抛。
- 每候选 token-bucket RPM 限流：非末位拿不到令牌即跳过，末位才等待。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rosclaw.agentd.models.gateway import (
    ModelGateway,
    ModelGatewayError,
    ModelTurnRequest,
)
from rosclaw.agentd.models.policy import ModelProfile
from rosclaw.contracts.agent.model_turn import ModelTurnResultV1

#: 立即可判死刑的错误（重试只会烧钱）。
NON_RETRYABLE_KINDS = frozenset(
    {"format", "auth_error", "invalid_response", "context_overflow", "missing_credential"}
)
_BILLING_KINDS = frozenset({"billing", "payment_required"})


@dataclass
class _CooldownState:
    failures: int = 0
    billing_failures: int = 0
    cooldown_until: float = 0.0
    last_failure_at: float = 0.0


class CooldownTracker:
    def __init__(self, *, now=time.monotonic) -> None:
        self._states: dict[str, _CooldownState] = {}
        self._now = now

    def in_cooldown(self, key: str) -> bool:
        state = self._states.get(key)
        if state is None:
            return False
        now = self._now()
        # 24h 无失败自动清零。
        if state.failures and now - state.last_failure_at > 24 * 3600:
            self._states.pop(key, None)
            return False
        return now < state.cooldown_until

    def record_failure(self, key: str, *, billing: bool = False) -> None:
        state = self._states.setdefault(key, _CooldownState())
        now = self._now()
        state.last_failure_at = now
        if billing:
            state.billing_failures += 1
            state.cooldown_until = now + min(
                24 * 3600.0, 5 * 3600.0 * (2 ** (state.billing_failures - 1))
            )
        else:
            state.failures = min(state.failures + 1, 3)
            state.cooldown_until = now + min(3600.0, 60.0 * (5 ** (state.failures - 1)))

    def record_success(self, key: str) -> None:
        self._states.pop(key, None)


@dataclass
class TokenBucket:
    """简单 RPM 令牌桶（每候选独立）。"""

    rpm: int
    _tokens: float = field(init=False)
    _refilled_at: float = field(init=False, default=-1.0)

    def __post_init__(self) -> None:
        self._tokens = float(self.rpm)

    def take(self, *, now: float) -> bool:
        if self._refilled_at < 0:
            self._refilled_at = now
        elapsed = max(0.0, now - self._refilled_at)
        self._tokens = min(float(self.rpm), self._tokens + elapsed * self.rpm / 60.0)
        self._refilled_at = now
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False


class FailoverGateway:
    """Ordered candidate chain with cooldown + RPM gating."""

    def __init__(
        self,
        candidates: list[tuple[ModelProfile, ModelGateway]],
        *,
        cooldown: CooldownTracker | None = None,
        rpm_limits: dict[str, int] | None = None,
        now=time.monotonic,
    ) -> None:
        if not candidates:
            raise ValueError("failover chain needs at least one candidate")
        self._candidates = candidates
        self._cooldown = cooldown or CooldownTracker(now=now)
        self._now = now
        self._buckets: dict[str, TokenBucket] = {
            p.name: TokenBucket((rpm_limits or {}).get(p.name, 60)) for p, _ in candidates
        }
        self.profile = candidates[0][0]

    def _key(self, profile: ModelProfile) -> str:
        return f"{profile.provider}/{profile.model}/{profile.base_url}"

    async def complete(self, request: ModelTurnRequest) -> ModelTurnResultV1:
        return await self._attempt("complete", request, None)

    async def complete_stream(
        self, request: ModelTurnRequest, on_text_delta=None
    ) -> ModelTurnResultV1:
        return await self._attempt("complete_stream", request, on_text_delta)

    async def _attempt(
        self, method: str, request: ModelTurnRequest, on_text_delta
    ) -> ModelTurnResultV1:
        last_error: ModelGatewayError | None = None
        for index, (profile, gateway) in enumerate(self._candidates):
            key = self._key(profile)
            is_last = index == len(self._candidates) - 1
            if self._cooldown.in_cooldown(key):
                continue
            if not is_last and not self._buckets[profile.name].take(now=self._now()):
                continue  # 非末位饱和即跳过；末位才阻塞等待
            if is_last:
                while not self._buckets[profile.name].take(now=self._now()):
                    import asyncio

                    await asyncio.sleep(0.05)
            try:
                fn = getattr(gateway, method)
                if method == "complete_stream":
                    result = await fn(request, on_text_delta=on_text_delta)
                else:
                    result = await fn(request)
            except ModelGatewayError as exc:
                last_error = exc
                if exc.kind in NON_RETRYABLE_KINDS:
                    raise
                self._cooldown.record_failure(key, billing=exc.kind in _BILLING_KINDS)
                continue
            self._cooldown.record_success(key)
            return result
        raise last_error or ModelGatewayError("all_candidates_exhausted", "no usable profile")

    async def probe(self):
        # 探测首个非冷却候选。
        for profile, gateway in self._candidates:
            if not self._cooldown.in_cooldown(self._key(profile)):
                return await gateway.probe()
        return await self._candidates[-1][1].probe()

    async def close(self) -> None:
        for _, gateway in self._candidates:
            await gateway.close()
