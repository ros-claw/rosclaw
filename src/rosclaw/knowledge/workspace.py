"""Bounded Native Agent reference state containing opaque IDs, never full documents."""

from __future__ import annotations

from pydantic import Field

from .contracts import HowAdviceBundleV2, ReferencePackV2, StrictWireModel


class ActiveReferenceWorkspaceV1(StrictWireModel):
    reference_pack_ids: list[str] = Field(default_factory=list)
    project_ids: list[str] = Field(default_factory=list)
    open_evidence_ids: list[str] = Field(default_factory=list)
    compatibility_warnings: list[str] = Field(default_factory=list)
    stale: bool = False


class ActiveReferenceWorkspace:
    def __init__(self, *, max_items: int = 50) -> None:
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        self.max_items = max_items
        self._packs: list[str] = []
        self._projects: list[str] = []
        self._evidence: list[str] = []
        self._warnings: list[str] = []
        self._stale = False

    def _add(self, target: list[str], values: list[str]) -> None:
        target[:] = list(dict.fromkeys([*target, *values]))[-self.max_items :]

    def observe_pack(self, pack: ReferencePackV2) -> None:
        self._add(self._packs, [pack.reference_pack_id])
        self._add(
            self._projects,
            [item.project_id for item in pack.items if item.project_id is not None],
        )
        self._add(
            self._evidence,
            [evidence.evidence_id for item in pack.items for evidence in item.evidence_refs],
        )
        self._add(
            self._warnings,
            [*pack.warnings, *(warning for item in pack.items for warning in item.incompatibilities)],
        )
        self._stale = self._stale or pack.stale

    def observe_advice(self, advice: HowAdviceBundleV2) -> None:
        if advice.reference_pack_id:
            self._add(self._packs, [advice.reference_pack_id])
        self._add(self._warnings, advice.compatibility_warnings)
        self._stale = self._stale or advice.reference_pack_stale

    def snapshot(self) -> ActiveReferenceWorkspaceV1:
        return ActiveReferenceWorkspaceV1(
            reference_pack_ids=list(self._packs),
            project_ids=list(self._projects),
            open_evidence_ids=list(self._evidence),
            compatibility_warnings=list(self._warnings),
            stale=self._stale,
        )


__all__ = ["ActiveReferenceWorkspace", "ActiveReferenceWorkspaceV1"]
