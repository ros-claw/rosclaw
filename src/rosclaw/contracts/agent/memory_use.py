"""Memory injection/use evidence; a retrieval alone never proves adoption."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import StrictBool, model_validator

from rosclaw.contracts.common import ContractModel


class MemoryUseEvidenceV1(ContractModel):
    SCHEMA = "rosclaw.memory_use_evidence.v1"

    schema_version: Literal["rosclaw.memory_use_evidence.v1"] = "rosclaw.memory_use_evidence.v1"
    retrieval_id: str
    memory_ref: str
    source_episode_id: str
    source_sha256: str
    evidence_class: Literal["measured", "verified_receipt", "curated"]
    context_bundle_hash: str
    injection_layer: Literal["L5_MEMORY_ADVISORY"] = "L5_MEMORY_ADVISORY"
    referenced_in_decision: StrictBool = False
    decision_request_id: str | None = None
    decision_tool: str | None = None
    actual_tool_event_hash: str | None = None
    subsequent_verification: dict[str, Any] | None = None
    causal_benefit: Literal["NOT_EVALUATED"] = "NOT_EVALUATED"
    authorization: Literal[False] = False

    @model_validator(mode="after")
    def require_observable_correspondence(self):
        if any(
            not isinstance(v, str) or not 0 < len(v) <= 256
            for v in (
                self.retrieval_id,
                self.memory_ref,
                self.source_episode_id,
                self.context_bundle_hash,
            )
        ):
            raise ValueError("bounded retrieval/source/context identities required")
        if len(self.source_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.source_sha256
        ):
            raise ValueError("exact historical source hash required")
        context_digest = self.context_bundle_hash.removeprefix("sha256:")
        if (
            not self.context_bundle_hash.startswith("sha256:")
            or len(context_digest) != 32
            or any(c not in "0123456789abcdef" for c in context_digest)
        ):
            raise ValueError("exact Native envelope hash required")
        if self.actual_tool_event_hash is not None and (
            len(self.actual_tool_event_hash) != 64
            or any(c not in "0123456789abcdef" for c in self.actual_tool_event_hash)
        ):
            raise ValueError("exact actual tool correspondence hash required")
        if self.referenced_in_decision and any(
            not isinstance(v, str) or not 0 < len(v) <= 256
            for v in (self.decision_request_id, self.decision_tool, self.actual_tool_event_hash)
        ):
            raise ValueError("memory adoption needs an actual corresponding tool event")
        if not self.referenced_in_decision and any(
            v is not None
            for v in (
                self.decision_request_id,
                self.decision_tool,
                self.actual_tool_event_hash,
                self.subsequent_verification,
            )
        ):
            raise ValueError("retrieval-only evidence cannot claim decision or verification")
        return self
