"""ToolResolver — deterministic hard filters before any relevance ranking (大纲 §7.4).

Hard filters (safety; a single failure excludes the tool, reasons are
explainable and auditable):

* body compatibility
* mode allowlist
* required capabilities online
* SelfSnapshot freshness
* permission granted
* policy deny list
* budget not exceeded
* verifier present
* not quarantined
* model_callable

Only then does relevance ranking run (semantic match, latency, reliability,
cost, freshness, evidence level). Safety conditions NEVER enter the score —
a high-scoring tool can never outrank a hard filter.

At most ``MAX_INJECTED_TOOLS`` tools are injected into a model turn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rosclaw.agentd.tooling.catalog import ToolCatalog
from rosclaw.contracts.agent.tool import ToolDescriptorV2

MAX_INJECTED_TOOLS = 12


@dataclass(frozen=True)
class FilterContext:
    """Everything the hard filters need; built by the service per turn."""

    body_type: str = ""
    mode: str = "SIMULATION"
    online_capabilities: frozenset[str] = frozenset()
    self_snapshot_fresh: bool = True
    granted_permissions: frozenset[str] = frozenset()
    policy_denied_tools: frozenset[str] = frozenset()
    budget_exceeded: bool = False
    #: semantic hint for ranking only (task text); never used by hard filters
    task_hint: str = ""


@dataclass(frozen=True)
class FilterDecision:
    tool_id: str
    passed: bool
    reasons: tuple[str, ...] = ()


@dataclass
class ResolutionResult:
    injected: list[ToolDescriptorV2] = field(default_factory=list)
    excluded: list[FilterDecision] = field(default_factory=list)


class ToolResolver:
    def __init__(self, catalog: ToolCatalog, *, max_injected: int = MAX_INJECTED_TOOLS) -> None:
        self._catalog = catalog
        self._max_injected = max_injected

    # -- hard filters (safety) -------------------------------------------------

    def hard_filter(self, descriptor: ToolDescriptorV2, ctx: FilterContext) -> FilterDecision:
        reasons: list[str] = []
        d = descriptor
        if d.required_body_types and ctx.body_type not in d.required_body_types:
            reasons.append(
                f"body_incompatible: requires {sorted(d.required_body_types)}, have {ctx.body_type!r}"
            )
        if ctx.mode not in d.supported_modes:
            reasons.append(f"mode_not_allowed: {ctx.mode} not in {sorted(d.supported_modes)}")
        missing_caps = [c for c in d.required_capabilities if c not in ctx.online_capabilities]
        if missing_caps:
            reasons.append(f"capability_offline: {missing_caps}")
        if d.freshness_ms is not None and not ctx.self_snapshot_fresh:
            reasons.append("self_snapshot_stale: freshness-gated tool with stale SelfSnapshot")
        if d.tool_id in ctx.policy_denied_tools:
            reasons.append("policy_denied")
        if ctx.granted_permissions:
            # permission filter only binds when a permission set is configured
            missing_perms = [
                c for c in d.required_capabilities if c not in ctx.granted_permissions
            ]
            if missing_perms:
                reasons.append(f"permission_not_granted: {missing_perms}")
        if ctx.budget_exceeded and d.cost_hint > 0:
            reasons.append("budget_exceeded")
        if not d.verifier:
            reasons.append("no_verifier")
        quarantine = self._catalog.quarantine_reason(d.tool_id)
        if quarantine is not None:
            reasons.append(f"quarantined: {quarantine}")
        if not d.model_callable:
            reasons.append("not_model_callable")
        return FilterDecision(tool_id=d.tool_id, passed=not reasons, reasons=tuple(reasons))

    # -- relevance ranking (never safety) ---------------------------------------

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {t for t in re.split(r"[^a-zA-Z0-9_一-鿿]+", text.lower()) if len(t) > 1}

    def score(self, descriptor: ToolDescriptorV2, ctx: FilterContext) -> float:
        score = 0.0
        if ctx.task_hint:
            haystack = self._tokenize(f"{descriptor.tool_id} {descriptor.description}")
            need = self._tokenize(ctx.task_hint)
            if need:
                score += 4.0 * len(haystack & need) / len(need)
        score += 2.0 * descriptor.reliability
        score += max(0.0, 1.0 - descriptor.typical_latency_ms / 5000.0)
        score -= min(descriptor.cost_hint, 5.0) * 0.2
        if descriptor.freshness_ms is not None and descriptor.freshness_ms <= 1000:
            score += 0.5
        evidence_rank = {"MEASURED": 1.0, "DERIVED": 0.7, "CONFIGURED": 0.5, "SIMULATED": 0.3}
        score += evidence_rank.get(descriptor.evidence_class.value, 0.0)
        return score

    # -- resolve -----------------------------------------------------------------

    def resolve(self, ctx: FilterContext, *, candidates: list[str] | None = None) -> ResolutionResult:
        """Hard-filter the catalog (or a candidate subset), rank survivors, cap."""
        pool = (
            [d for d in (self._catalog.get(t) for t in candidates) if d is not None]
            if candidates is not None
            else self._catalog.list()
        )
        result = ResolutionResult()
        survivors: list[ToolDescriptorV2] = []
        for descriptor in pool:
            decision = self.hard_filter(descriptor, ctx)
            if decision.passed:
                survivors.append(descriptor)
            else:
                result.excluded.append(decision)
        survivors.sort(key=lambda d: (-self.score(d, ctx), d.tool_id))
        result.injected = survivors[: self._max_injected]
        for overflow in survivors[self._max_injected :]:
            result.excluded.append(
                FilterDecision(
                    tool_id=overflow.tool_id,
                    passed=False,
                    reasons=(f"injection_cap: only {self._max_injected} tools per turn",),
                )
            )
        return result
