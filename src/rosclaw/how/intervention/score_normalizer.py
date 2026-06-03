"""rosclaw_how.score_normalizer — direction-aware score arithmetic.

Frontier-Eng / Darwin-style benchmarks mix maximize tasks (success rate,
F1, throughput) with minimize tasks (runtime ns, latency p99, error
count). The v1 hot path treats every score as ``higher is better``,
which silently mislabels every minimize task: ``recent[-1] > recent[0]``
reports "improving" when the agent is actually getting worse.

This module is the single source of truth for direction-aware ops:

* ``normalize_score`` — fold direction into a scalar where higher-is-
  better is always the convention.
* ``delta_normalized`` — signed delta in normalized space; positive means
  "the agent improved" regardless of the original direction.
* ``is_improving_normalized`` — direction-safe replacement for the
  ``recent[-1] > recent[0]`` heuristic in ``state_router``.

No I/O, no logging, no Pydantic. Hot-path safe.
"""
from __future__ import annotations

from typing import Literal

ObjectiveDirection = Literal["maximize", "minimize"]


def normalize_score(
    raw_score: float | None,
    objective_direction: ObjectiveDirection = "maximize",
) -> float | None:
    """Project a raw verifier score into a higher-is-better scalar.

    Returns None when the input is None so callers can preserve "no
    data" semantics through the pipeline."""
    if raw_score is None:
        return None
    if objective_direction == "minimize":
        return -float(raw_score)
    return float(raw_score)


def normalize_scores(
    raw_scores: list[float],
    objective_direction: ObjectiveDirection = "maximize",
) -> list[float]:
    """Vectorized ``normalize_score`` for trace windows."""
    if objective_direction == "minimize":
        return [-float(s) for s in raw_scores]
    return [float(s) for s in raw_scores]


def delta_normalized(
    pre_score: float | None,
    post_score: float | None,
    objective_direction: ObjectiveDirection = "maximize",
) -> float | None:
    """Return ``post - pre`` in normalized space; positive == improved.

    Mirrors ``outcomes.record_feedback`` semantics but does the
    direction fold once at the boundary instead of leaking it through
    every aggregator."""
    if pre_score is None or post_score is None:
        return None
    pre_norm = normalize_score(pre_score, objective_direction)
    post_norm = normalize_score(post_score, objective_direction)
    if pre_norm is None or post_norm is None:
        return None
    return post_norm - pre_norm


def is_improving_normalized(
    previous_scores: list[float],
    objective_direction: ObjectiveDirection = "maximize",
    window: int = 3,
) -> bool:
    """Direction-safe ``recent[-1] > recent[0]``.

    Returns ``True`` when the window is too small to decide (matches
    the v1 ``_is_improving`` fall-through so ``state_router`` doesn't
    over-eagerly trigger CATALYST in the first few iterations)."""
    if len(previous_scores) < window + 1:
        return True
    normalized = normalize_scores(previous_scores[-(window + 1) :], objective_direction)
    return normalized[-1] > normalized[0]


__all__ = [
    "ObjectiveDirection",
    "delta_normalized",
    "is_improving_normalized",
    "normalize_score",
    "normalize_scores",
]
