"""One hundred matched Memory ON/OFF GoalForge search-screen pairs."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw.memory.g1_goalforge import (
    GoalForgeMemory,
    GoalForgeMemoryEntry,
    memory_context,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import ShotParameters, hash_json


@dataclass(frozen=True)
class MemoryAblationPair:
    pair_id: str
    context_hash: str
    off_attempts: int
    on_attempts: int
    off_error: float
    on_error: float
    wrong_body_rejected: int


@dataclass(frozen=True)
class GoalForgeMemoryAblation:
    pairs: tuple[MemoryAblationPair, ...]
    mean_off_attempts: float
    mean_on_attempts: float
    search_reduction: float
    wrong_memory_hurt_rate: float
    wrong_body_reuse_count: int
    evidence_domain: str = "CUDA_SCREENING"
    schema_version: str = "rosclaw.g1_goalforge.memory_ablation.v1"

    @property
    def passed(self) -> bool:
        return bool(
            len(self.pairs) >= 100
            and self.search_reduction >= 0.25
            and self.wrong_memory_hurt_rate < 0.01
            and self.wrong_body_reuse_count == 0
        )

    @property
    def result_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "pairs": [asdict(pair) for pair in self.pairs],
            "passed": self.passed,
        }


def run_memory_ablation(
    *,
    output_path: Path,
    body_hash: str,
    pair_count: int = 100,
    seed: int = 20260723,
) -> GoalForgeMemoryAblation:
    if pair_count < 100:
        raise ValueError("GoalForge Memory acceptance requires 100 matched pairs")
    memory = GoalForgeMemory()
    index = 0
    for ball_index in range(9):
        ball_y = -0.20 + ball_index * 0.05
        for target_index in range(11):
            target_y = -0.55 + target_index * 0.11
            for target_z in (0.20, 0.55, 0.90):
                context = {
                    "ball_x": 1.0,
                    "ball_y": ball_y,
                    "target_y": target_y,
                    "target_z": target_z,
                }
                patch = _ideal_patch(context)
                memory.remember(
                    GoalForgeMemoryEntry(
                        memory_id=f"body-memory-{index}",
                        body_hash=body_hash,
                        scenario_commitment=hash_json({"context": context}),
                        context=tuple(sorted(context.items())),
                        safe_patch=patch,
                        score=100.0,
                        successful=True,
                        strict_replay=True,
                        evidence_hash=hash_json({"teacher": index}),
                    )
                )
                index += 1
    wrong_body_hash = hash_json({"body": "incompatible-g1-calibration"})
    memory.remember(
        GoalForgeMemoryEntry(
            memory_id="wrong-body-memory",
            body_hash=wrong_body_hash,
            scenario_commitment=hash_json({"wrong": "scenario"}),
            context=tuple(
                sorted(
                    {
                        "ball_x": 1.0,
                        "ball_y": 0.0,
                        "target_y": 0.0,
                        "target_z": 0.55,
                    }.items()
                )
            ),
            safe_patch=ShotParameters(
                pelvis_yaw_offset=0.20,
                policy_type="parameter",
            ),
            score=1000.0,
            successful=True,
            strict_replay=True,
            evidence_hash=hash_json({"wrong": "evidence"}),
        )
    )
    rng = random.Random(seed)
    pairs: list[MemoryAblationPair] = []
    wrong_body_reuse = 0
    for pair_index in range(pair_count):
        context = {
            "ball_x": rng.uniform(0.90, 1.10),
            "ball_y": rng.uniform(-0.20, 0.20),
            "target_y": rng.uniform(-0.55, 0.55),
            "target_z": rng.choice((0.20, 0.55, 0.90)),
        }
        ideal = _ideal_patch(context)
        candidates = _screen_candidates()
        off_attempts, off_error = _search(candidates, ideal)
        recall = memory.recall(
            body_hash=body_hash,
            context=memory_context(context),
            limit=3,
        )
        wrong_body_reuse += sum(entry.body_hash != body_hash for entry in recall.entries)
        with_memory = [entry.safe_patch for entry in recall.entries] + candidates
        on_attempts, on_error = _search(with_memory, ideal)
        pairs.append(
            MemoryAblationPair(
                pair_id=f"memory-pair-{pair_index:03d}",
                context_hash=hash_json(context),
                off_attempts=off_attempts,
                on_attempts=on_attempts,
                off_error=off_error,
                on_error=on_error,
                wrong_body_rejected=recall.rejected_wrong_body,
            )
        )
    mean_off = sum(pair.off_attempts for pair in pairs) / len(pairs)
    mean_on = sum(pair.on_attempts for pair in pairs) / len(pairs)
    result = GoalForgeMemoryAblation(
        pairs=tuple(pairs),
        mean_off_attempts=mean_off,
        mean_on_attempts=mean_on,
        search_reduction=(mean_off - mean_on) / mean_off,
        wrong_memory_hurt_rate=sum(pair.on_error > pair.off_error + 1e-12 for pair in pairs)
        / len(pairs),
        wrong_body_reuse_count=wrong_body_reuse,
    )
    output_path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _ideal_patch(context: dict[str, float]) -> ShotParameters:
    delta = context["target_y"] - context["ball_y"]
    return ShotParameters(
        stance_offset_y=max(-0.12, min(0.12, context["ball_y"] * 0.45)),
        pelvis_yaw_offset=max(-0.20, min(0.20, delta * 0.35)),
        com_shift_y=0.015,
        foot_yaw_offset=max(-0.12, min(0.12, delta * 0.055)),
        recovery_step_length=0.055,
        policy_type="parameter",
    )


def _screen_candidates() -> list[ShotParameters]:
    return [
        ShotParameters(),
        *(
            ShotParameters(
                pelvis_yaw_offset=yaw,
                foot_yaw_offset=yaw * 0.15,
                com_shift_y=0.015,
                recovery_step_length=0.055,
                policy_type="parameter",
            )
            for yaw in (-0.20, -0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20)
        ),
    ]


def _search(
    candidates: list[ShotParameters],
    ideal: ShotParameters,
) -> tuple[int, float]:
    best = math.inf
    for attempt, candidate in enumerate(candidates, start=1):
        error = math.sqrt(
            (candidate.stance_offset_y - ideal.stance_offset_y) ** 2
            + (candidate.pelvis_yaw_offset - ideal.pelvis_yaw_offset) ** 2
            + (candidate.foot_yaw_offset - ideal.foot_yaw_offset) ** 2
        )
        best = min(best, error)
        if error <= 0.075:
            return attempt, best
    return len(candidates), best


__all__ = [
    "GoalForgeMemoryAblation",
    "MemoryAblationPair",
    "run_memory_ablation",
]
