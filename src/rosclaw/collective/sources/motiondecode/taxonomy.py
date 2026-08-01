"""Deterministic, semantics-preserving MotionDecode pilot selection."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class MotionDecodeStratum(StrEnum):
    FOOTBALL = "football"
    BALANCE_PROXY = "balance_proxy"
    GAIT = "gait"
    TRANSITION_RECOVERY = "transition_recovery"
    COORDINATION_SUPPLEMENT = "coordination_supplement"
    OTHER = "other"


@dataclass(frozen=True)
class MotionDecodePilotSelection:
    selected: tuple[tuple[MotionDecodeStratum, Path], ...]
    requested: Mapping[str, int]
    selected_counts: Mapping[str, int]
    shortages: Mapping[str, int]
    substitutions: Mapping[str, int]


_RECOVERY = (
    "ground_recovery",
    "stand_up",
    "get_up",
    "sit_to_stand",
    "stand_to_sit",
    "stand_to_prone",
    "prone_to_stand",
    "pre_fall_prevention",
    "turning_over_to_sitting_up",
    "sitting_up_to_standing",
)
_GAIT = (
    "gait",
    "walking",
    "small_backward_steps",
    "slow_lateral_movement",
    "retreating",
    "approaching",
    "stage_entry",
)
_BALANCE = (
    "balance",
    "single_leg",
    "slight_weight_shift",
    "standing_waiting",
    "spotlight_standing",
    "lower_body_rhythm",
    "standing_high_jump",
)
_COORDINATION = (
    "throw",
    "dance",
    "groove",
    "jump",
    "rhythm",
    "lifting",
    "carrying",
    "pushing",
    "pulling",
)


def classify_motiondecode_path(relative_path: Path) -> MotionDecodeStratum:
    value = relative_path.as_posix().lower()
    if "football" in value or "ball_game" in value:
        return MotionDecodeStratum.FOOTBALL
    if any(token in value for token in _RECOVERY):
        return MotionDecodeStratum.TRANSITION_RECOVERY
    if any(token in value for token in _GAIT):
        return MotionDecodeStratum.GAIT
    if any(token in value for token in _BALANCE):
        return MotionDecodeStratum.BALANCE_PROXY
    if any(token in value for token in _COORDINATION):
        return MotionDecodeStratum.COORDINATION_SUPPLEMENT
    return MotionDecodeStratum.OTHER


def select_motiondecode_pilot(
    relative_paths: Iterable[Path],
    *,
    limit: int = 400,
    seed: int = 20260801,
) -> MotionDecodePilotSelection:
    if not 4 <= limit <= 4000:
        raise ValueError("MotionDecode pilot limit must be in [4, 4000]")
    base = limit // 4
    remainder = limit - base * 4
    requested = {
        MotionDecodeStratum.FOOTBALL.value: base + remainder,
        MotionDecodeStratum.BALANCE_PROXY.value: base,
        MotionDecodeStratum.GAIT.value: base,
        MotionDecodeStratum.TRANSITION_RECOVERY.value: base,
    }
    pools: dict[MotionDecodeStratum, list[Path]] = defaultdict(list)
    for path in relative_paths:
        pools[classify_motiondecode_path(path)].append(path)
    for stratum, paths in pools.items():
        paths.sort(key=lambda item: _selection_key(item, seed=seed, stratum=stratum))

    selected: list[tuple[MotionDecodeStratum, Path]] = []
    shortages: dict[str, int] = {}
    for name, count in requested.items():
        stratum = MotionDecodeStratum(name)
        values = pools[stratum][:count]
        selected.extend((stratum, path) for path in values)
        if len(values) < count:
            shortages[name] = count - len(values)

    # Keep the audit statistically useful without pretending a substitute is
    # football.  Backfilled records retain an explicit supplement stratum.
    missing = limit - len(selected)
    selected_paths = {path for _, path in selected}
    supplements = [
        path
        for stratum in (MotionDecodeStratum.COORDINATION_SUPPLEMENT, MotionDecodeStratum.OTHER)
        for path in pools[stratum]
        if path not in selected_paths
    ]
    supplements.sort(
        key=lambda item: _selection_key(
            item,
            seed=seed,
            stratum=MotionDecodeStratum.COORDINATION_SUPPLEMENT,
        )
    )
    fill = supplements[:missing]
    selected.extend((MotionDecodeStratum.COORDINATION_SUPPLEMENT, path) for path in fill)
    substitutions = (
        {MotionDecodeStratum.COORDINATION_SUPPLEMENT.value: len(fill)} if fill else {}
    )
    selected.sort(key=lambda item: (item[0].value, item[1].as_posix()))
    selected_counts: dict[str, int] = {
        stratum.value: 0 for stratum in MotionDecodeStratum
    }
    for stratum, _ in selected:
        selected_counts[stratum.value] += 1
    return MotionDecodePilotSelection(
        selected=tuple(selected),
        requested=requested,
        selected_counts=dict(selected_counts),
        shortages=shortages,
        substitutions=substitutions,
    )


def _selection_key(path: Path, *, seed: int, stratum: MotionDecodeStratum) -> str:
    value = f"{seed}:{stratum.value}:{path.as_posix()}".encode()
    return hashlib.sha256(value).hexdigest()
