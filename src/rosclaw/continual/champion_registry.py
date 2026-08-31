"""Canonical, replayable champion lineage for continual learners.

Promotion evidence is not sufficient on its own: a valid paired exam can still
compare against a stale or previously rejected parent.  This registry makes the
active track head part of the promotion authority and fails closed when lineage
does not match it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from rosclaw.feedback.contracts import canonical_hash

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}$")


def _require_hash(label: str, value: str | None) -> None:
    if value is None or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a sha256: content hash")


def _require_identifier(label: str, value: str) -> None:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} is not a valid stable identifier")


class PromotionAuthority(StrEnum):
    """Evidence authority allowed to create each type of lineage record."""

    BASELINE_STRICT_EXAM = "baseline_strict_exam"
    PAIRED_DOMINANCE = "paired_dominance"
    SEALED_SPECIALIST_EXAM = "sealed_specialist_exam"


class ChampionRecordKind(StrEnum):
    """How a record affects its track's active champion."""

    GLOBAL_BASELINE = "global_baseline"
    TRACK_REPLACEMENT = "track_replacement"
    SPECIALIST_BASELINE = "specialist_baseline"
    CANDIDATE_ARCHIVED = "candidate_archived"


@dataclass(frozen=True)
class ChampionRegistryRecord:
    """One immutable result in a champion track.

    ``parent_record_hash`` binds the decision to the exact registry head while
    ``parent_artifact_hash`` binds the external paired evaluation.  A specialist
    baseline may retain ``source_parent_artifact_hash`` as training provenance,
    but that ancestry does not give it authority to replace another track.
    """

    agent_id: str
    track_id: str
    artifact_hash: str
    evidence_hash: str
    scenario_suite_hash: str
    authority: PromotionAuthority
    kind: ChampionRecordKind
    evidence_valid: bool
    promotion_passed: bool
    generation: int
    parent_record_hash: str | None = None
    parent_artifact_hash: str | None = None
    source_parent_artifact_hash: str | None = None
    sim_only: bool = True
    schema_version: str = "rosclaw.continual.champion_registry_record.v1"

    def __post_init__(self) -> None:
        _require_identifier("agent_id", self.agent_id)
        _require_identifier("track_id", self.track_id)
        for label, value in (
            ("artifact_hash", self.artifact_hash),
            ("evidence_hash", self.evidence_hash),
            ("scenario_suite_hash", self.scenario_suite_hash),
        ):
            _require_hash(label, value)
        if self.generation < 0:
            raise ValueError("generation must be non-negative")
        optional_hashes: tuple[tuple[str, str | None], ...] = (
            ("parent_record_hash", self.parent_record_hash),
            ("parent_artifact_hash", self.parent_artifact_hash),
            ("source_parent_artifact_hash", self.source_parent_artifact_hash),
        )
        for optional_label, optional_value in optional_hashes:
            if optional_value is not None:
                _require_hash(optional_label, optional_value)
        if not self.sim_only:
            raise ValueError(
                "continual champion records are SIM_ONLY until a separate real-body gate"
            )
        if self.kind is ChampionRecordKind.GLOBAL_BASELINE:
            self._require_baseline(PromotionAuthority.BASELINE_STRICT_EXAM)
        elif self.kind is ChampionRecordKind.SPECIALIST_BASELINE:
            self._require_baseline(PromotionAuthority.SEALED_SPECIALIST_EXAM)
        elif self.kind is ChampionRecordKind.TRACK_REPLACEMENT:
            self._require_child(promoted=True)
        else:
            self._require_child(promoted=False)

    def _require_baseline(self, authority: PromotionAuthority) -> None:
        if self.authority is not authority:
            raise ValueError("baseline record uses the wrong promotion authority")
        if self.parent_record_hash is not None or self.parent_artifact_hash is not None:
            raise ValueError("baseline record cannot claim a registry parent")
        if self.generation != 0:
            raise ValueError("baseline record generation must be zero")
        if not self.evidence_valid or not self.promotion_passed:
            raise ValueError("baseline record requires valid, passing evidence")

    def _require_child(self, *, promoted: bool) -> None:
        if self.authority is not PromotionAuthority.PAIRED_DOMINANCE:
            raise ValueError("child record requires paired-dominance authority")
        _require_hash("parent_record_hash", self.parent_record_hash)
        _require_hash("parent_artifact_hash", self.parent_artifact_hash)
        if self.generation <= 0:
            raise ValueError("child record generation must be positive")
        if promoted and (not self.evidence_valid or not self.promotion_passed):
            raise ValueError("replacement requires valid, passing evidence")
        if not promoted and self.promotion_passed:
            raise ValueError("archived candidate cannot carry a passing promotion decision")

    @property
    def record_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "agent_id": self.agent_id,
            "track_id": self.track_id,
            "artifact_hash": self.artifact_hash,
            "evidence_hash": self.evidence_hash,
            "scenario_suite_hash": self.scenario_suite_hash,
            "authority": self.authority.value,
            "kind": self.kind.value,
            "evidence_valid": self.evidence_valid,
            "promotion_passed": self.promotion_passed,
            "generation": self.generation,
            "parent_record_hash": self.parent_record_hash,
            "parent_artifact_hash": self.parent_artifact_hash,
            "source_parent_artifact_hash": self.source_parent_artifact_hash,
            "sim_only": self.sim_only,
        }


@dataclass(frozen=True)
class ChampionClaimAudit:
    """Fail-closed comparison of a claimed parent and the active track head."""

    track_id: str
    claimed_parent_artifact_hash: str
    active_record_hash: str | None
    active_artifact_hash: str | None
    valid: bool
    reasons: tuple[str, ...]
    schema_version: str = "rosclaw.continual.champion_claim_audit.v1"

    @property
    def audit_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "track_id": self.track_id,
            "claimed_parent_artifact_hash": self.claimed_parent_artifact_hash,
            "active_record_hash": self.active_record_hash,
            "active_artifact_hash": self.active_artifact_hash,
            "valid": self.valid,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class CanonicalChampionRegistry:
    """Append-only lineage whose active heads are reproduced from its records."""

    records: tuple[ChampionRegistryRecord, ...] = ()
    schema_version: str = "rosclaw.continual.canonical_champion_registry.v1"

    def __post_init__(self) -> None:
        self._replay()

    @property
    def registry_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def append(self, record: ChampionRegistryRecord) -> CanonicalChampionRegistry:
        if any(existing.record_hash == record.record_hash for existing in self.records):
            raise ValueError("champion record is already registered")
        return CanonicalChampionRegistry(records=(*self.records, record))

    def active_head(self, track_id: str) -> ChampionRegistryRecord | None:
        _require_identifier("track_id", track_id)
        return self._replay().get(track_id)

    def audit_parent_claim(
        self,
        *,
        track_id: str,
        claimed_parent_artifact_hash: str,
    ) -> ChampionClaimAudit:
        _require_identifier("track_id", track_id)
        _require_hash("claimed_parent_artifact_hash", claimed_parent_artifact_hash)
        active = self.active_head(track_id)
        reasons: list[str] = []
        if active is None:
            reasons.append("track_has_no_active_head")
        elif active.artifact_hash != claimed_parent_artifact_hash:
            reasons.append("parent_not_active_track_head")
        return ChampionClaimAudit(
            track_id=track_id,
            claimed_parent_artifact_hash=claimed_parent_artifact_hash,
            active_record_hash=None if active is None else active.record_hash,
            active_artifact_hash=None if active is None else active.artifact_hash,
            valid=not reasons,
            reasons=tuple(reasons),
        )

    def _replay(self) -> dict[str, ChampionRegistryRecord]:
        heads: dict[str, ChampionRegistryRecord] = {}
        record_hashes: set[str] = set()
        artifact_keys: set[tuple[str, str]] = set()
        for record in self.records:
            if record.record_hash in record_hashes:
                raise ValueError("duplicate champion record hash")
            artifact_key = (record.track_id, record.artifact_hash)
            if artifact_key in artifact_keys:
                raise ValueError("artifact is already recorded on this track")
            record_hashes.add(record.record_hash)
            artifact_keys.add(artifact_key)
            active = heads.get(record.track_id)
            if record.kind in (
                ChampionRecordKind.GLOBAL_BASELINE,
                ChampionRecordKind.SPECIALIST_BASELINE,
            ):
                if active is not None:
                    raise ValueError("track already has a baseline")
                heads[record.track_id] = record
                continue
            if active is None:
                raise ValueError("child record has no active track head")
            if record.parent_record_hash != active.record_hash:
                raise ValueError("parent record is not the active track head")
            if record.parent_artifact_hash != active.artifact_hash:
                raise ValueError("parent artifact is not the active track head")
            if record.generation != active.generation + 1:
                raise ValueError("child generation does not follow the active track head")
            if record.kind is ChampionRecordKind.TRACK_REPLACEMENT:
                heads[record.track_id] = record
        return heads

    def to_dict(self) -> dict[str, Any]:
        heads = self._replay()
        return {
            "schema_version": self.schema_version,
            "records": [record.to_dict() for record in self.records],
            "active_heads": {
                track_id: {
                    "record_hash": record.record_hash,
                    "artifact_hash": record.artifact_hash,
                    "generation": record.generation,
                }
                for track_id, record in sorted(heads.items())
            },
        }


__all__ = [
    "CanonicalChampionRegistry",
    "ChampionClaimAudit",
    "ChampionRecordKind",
    "ChampionRegistryRecord",
    "PromotionAuthority",
]
