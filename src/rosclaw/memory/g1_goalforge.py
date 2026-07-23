"""Body-scoped episodic memory for GoalForge search warm-starting."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from rosclaw.simforge.tasks.g1_goalforge.concepts import ShotParameters, hash_json

_FEATURES = ("ball_x", "ball_y", "target_y", "target_z")


@dataclass(frozen=True)
class GoalForgeMemoryEntry:
    memory_id: str
    body_hash: str
    scenario_commitment: str
    context: tuple[tuple[str, float], ...]
    safe_patch: ShotParameters
    score: float
    successful: bool
    strict_replay: bool
    evidence_hash: str
    schema_version: str = "rosclaw.g1_goalforge.memory.v1"

    def __post_init__(self) -> None:
        hashes = (self.body_hash, self.scenario_commitment, self.evidence_hash)
        if any(not value.startswith("sha256:") for value in hashes):
            raise ValueError("GoalForge memory evidence must be sha256-addressed")
        values = dict(self.context)
        if set(values) != set(_FEATURES):
            raise ValueError("GoalForge memory context has an invalid feature contract")
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("GoalForge memory context must be finite")
        if not math.isfinite(self.score):
            raise ValueError("GoalForge memory score must be finite")

    @property
    def entry_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["context"] = dict(self.context)
        value["safe_patch"] = self.safe_patch.to_dict()
        return value


@dataclass(frozen=True)
class GoalForgeMemoryRecall:
    query_hash: str
    body_hash: str
    entries: tuple[GoalForgeMemoryEntry, ...]
    distances: tuple[float, ...]
    rejected_wrong_body: int
    schema_version: str = "rosclaw.g1_goalforge.memory_recall.v1"

    @property
    def summary(self) -> tuple[float, ...]:
        if not self.entries:
            return (0.0,) * 9
        patch = self.entries[0].safe_patch
        return (
            patch.stance_offset_x,
            patch.stance_offset_y,
            patch.pelvis_yaw_offset,
            patch.com_shift_y,
            patch.swing_amplitude,
            patch.swing_speed_scale,
            patch.foot_yaw_offset,
            patch.contact_phase_offset,
            patch.recovery_step_length,
        )


class GoalForgeMemory:
    """Append-only successful memory with hard Body-hash isolation."""

    def __init__(self) -> None:
        self._entries: list[GoalForgeMemoryEntry] = []

    def remember(self, entry: GoalForgeMemoryEntry) -> None:
        if not entry.successful or not entry.strict_replay:
            raise ValueError("only successful strict-replay episodes enter GoalForge memory")
        if any(item.memory_id == entry.memory_id for item in self._entries):
            raise ValueError("duplicate GoalForge memory id")
        self._entries.append(entry)

    def recall(
        self,
        *,
        body_hash: str,
        context: dict[str, float],
        limit: int = 3,
    ) -> GoalForgeMemoryRecall:
        if set(context) != set(_FEATURES) or not 1 <= limit <= 16:
            raise ValueError("invalid GoalForge memory query")
        wrong = sum(item.body_hash != body_hash for item in self._entries)
        ranked = sorted(
            (
                (_distance(context, dict(item.context)), item)
                for item in self._entries
                if item.body_hash == body_hash
            ),
            key=lambda value: (value[0], -value[1].score, value[1].memory_id),
        )[:limit]
        query_hash = hash_json({"body_hash": body_hash, "context": context, "limit": limit})
        return GoalForgeMemoryRecall(
            query_hash=query_hash,
            body_hash=body_hash,
            entries=tuple(item for _, item in ranked),
            distances=tuple(distance for distance, _ in ranked),
            rejected_wrong_body=wrong,
        )


def memory_context(observed_context: dict[str, float]) -> dict[str, float]:
    return {name: float(observed_context[name]) for name in _FEATURES}


def _distance(left: dict[str, float], right: dict[str, float]) -> float:
    scales = {"ball_x": 0.25, "ball_y": 0.40, "target_y": 2.40, "target_z": 1.09}
    return math.sqrt(sum(((left[name] - right[name]) / scales[name]) ** 2 for name in _FEATURES))


__all__ = [
    "GoalForgeMemory",
    "GoalForgeMemoryEntry",
    "GoalForgeMemoryRecall",
    "memory_context",
]
