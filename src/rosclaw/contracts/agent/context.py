"""EmbodiedContextBundleV1 (ADR-0005, 总纲 §5.3).

Versioned, hashed, layered, truncatable context compiled from trusted,
dynamic, mission, organization and policy sources. The bundle binds body,
self, team and authorization hashes; decisions are only valid against the
revision they were produced from.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from rosclaw.contracts.common import ContractModel


class LayerRef(ContractModel):
    """One context layer: content hash plus either inline summary or refs."""

    SCHEMA = "rosclaw.context_layer.v1"

    schema_version: Literal["rosclaw.context_layer.v1"] = "rosclaw.context_layer.v1"
    hash: str
    content_ref: str | None = None
    inline_summary: str | None = None
    candidate_tools: list[str] | None = None
    evidence_refs: list[str] | None = None
    message_refs: list[str] | None = None
    token_estimate: int = 0


class SelfBinding(ContractModel):
    SCHEMA = "rosclaw.self_binding.v1"

    schema_version: Literal["rosclaw.self_binding.v1"] = "rosclaw.self_binding.v1"
    self_snapshot_hash: str
    sequence: int
    observed_at: str
    max_age_ms: int = 500


class TeamBinding(ContractModel):
    SCHEMA = "rosclaw.team_binding.v1"

    schema_version: Literal["rosclaw.team_binding.v1"] = "rosclaw.team_binding.v1"
    team_id: str | None = None
    epoch: int = 0
    world_revision: int = 0


class AuthorizationContextBinding(ContractModel):
    """Public scope only — signature and private permit stay in the broker."""

    SCHEMA = "rosclaw.authorization_context_binding.v1"

    schema_version: Literal["rosclaw.authorization_context_binding.v1"] = (
        "rosclaw.authorization_context_binding.v1"
    )
    policy_hash: str
    mission_grant_public_hash: str | None = None


class ContextLayers(ContractModel):
    """The nine layers of ADR-0005. L8 is the only untrusted layer."""

    SCHEMA = "rosclaw.context_layers.v1"

    schema_version: Literal["rosclaw.context_layers.v1"] = "rosclaw.context_layers.v1"
    constitution: LayerRef  # L0
    embodiment: LayerRef  # L1
    dynamic_self: LayerRef  # L2
    capabilities: LayerRef  # L3
    mission: LayerRef  # L4
    memory: LayerRef | None = None  # L5
    organization: LayerRef | None = None  # L6
    safety: LayerRef  # L7
    untrusted_inputs: LayerRef | None = None  # L8 (explicitly untrusted)


class TruncationEvent(ContractModel):
    SCHEMA = "rosclaw.truncation_event.v1"

    schema_version: Literal["rosclaw.truncation_event.v1"] = "rosclaw.truncation_event.v1"
    layer: str
    dropped_tokens: int
    reason: str


class ContextBudget(ContractModel):
    SCHEMA = "rosclaw.context_budget.v1"

    schema_version: Literal["rosclaw.context_budget.v1"] = "rosclaw.context_budget.v1"
    maximum_input_tokens: int = 120_000
    used_tokens: int = 0
    truncation_events: list[TruncationEvent] = Field(default_factory=list)


class BodyContextBinding(ContractModel):
    SCHEMA = "rosclaw.body_context_binding.v1"

    schema_version: Literal["rosclaw.body_context_binding.v1"] = "rosclaw.body_context_binding.v1"
    body_id: str
    effective_body_hash: str


class EmbodiedContextBundleV1(ContractModel):
    SCHEMA = "rosclaw.embodied_context.v1"
    HASH_PREFIX = "ctxb"
    HASH_EXCLUDE_FIELD = "bundle_hash"

    schema_version: Literal["rosclaw.embodied_context.v1"] = "rosclaw.embodied_context.v1"
    context_id: str
    context_revision: int
    compiled_at: str
    compiler_version: str = "1.0.0"
    mission_id: str
    body_binding: BodyContextBinding
    self_binding: SelfBinding | None = None
    team_binding: TeamBinding | None = None
    authorization_binding: AuthorizationContextBinding
    layers: ContextLayers
    budget: ContextBudget = Field(default_factory=ContextBudget)
    bundle_hash: str = ""
    extra: dict[str, Any] | None = None

    def finalize_hash(self) -> str:
        """Compute and store the bundle hash (call once after compile)."""
        self.bundle_hash = self.canonical_hash()
        return self.bundle_hash

    def hash_payload(self) -> dict:
        # context_id identifies one compiled instance; the hash is over
        # *content* so identical inputs compile to an identical hash
        # (PR-NA-021 exit criterion: deterministic compile).
        data = super().hash_payload()
        data.pop("context_id", None)
        return data
