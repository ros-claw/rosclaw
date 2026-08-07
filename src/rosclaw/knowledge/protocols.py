"""Narrow Core-facing protocols; knowledge algorithms remain out of Core."""

from __future__ import annotations

from typing import Any, Protocol

from .contracts import (
    HowAdviceBundleV2,
    HowAdviceRequestV2,
    KnowledgeUsageFeedbackV1,
    ReferenceContextV2,
    ReferencePackV2,
    ResearchRequestV2,
)


class KnowProtocol(Protocol):
    def health(self) -> dict[str, Any]: ...

    def doctor(self) -> dict[str, Any]: ...

    def explain(
        self, *, query: str, context: ReferenceContextV2, top_k: int = 10
    ) -> dict[str, Any]: ...

    def project_diff(
        self, *, project_id: str, from_snapshot: str, to_snapshot: str
    ) -> dict[str, Any]: ...

    def refresh_source(self, *, source_id: str, apply: bool = False) -> dict[str, Any]: ...

    def freeze(self, *, label: str) -> dict[str, Any]: ...

    def research(self, request: ResearchRequestV2) -> dict[str, Any]: ...

    def reference_pack(
        self, *, query: str, context: ReferenceContextV2, top_k: int, token_budget: int
    ) -> ReferencePackV2: ...

    def get_reference_pack(self, reference_pack_id: str) -> ReferencePackV2 | None: ...

    def submit_feedback(self, feedback: KnowledgeUsageFeedbackV1) -> bool: ...


class HowProtocol(Protocol):
    def health(self) -> dict[str, Any]: ...

    def doctor(self) -> dict[str, Any]: ...

    def explain(self, advice_id: str) -> dict[str, Any]: ...

    def advise(self, request: HowAdviceRequestV2) -> HowAdviceBundleV2: ...

    def submit_feedback(self, feedback: KnowledgeUsageFeedbackV1) -> bool: ...
