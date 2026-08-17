"""Typed lineage vocabulary (PR-DF-17 / phase-II §8.1-8.2).

Canonical orientation (§8.5): child/derived -> parent/source.
A Champion points at the Evaluation it was promoted from, never the other
way around.
"""

from __future__ import annotations

from enum import StrEnum


class LineageEntityType(StrEnum):
    ACTION = "action"
    RECEIPT = "receipt"
    PRACTICE = "practice"
    EPISODE = "episode"

    MEMORY = "memory"
    MEMORY_INSIGHT = "memory_insight"

    FAILURE = "failure"
    DIAGNOSIS = "diagnosis"
    PROPOSAL = "proposal"
    PATCH = "patch"
    EXPERIMENT = "experiment"
    DARWIN_BENCHMARK = "darwin_benchmark"
    EVALUATION = "evaluation"

    CHAMPION = "champion"
    DEAD_END = "dead_end"

    SKILL = "skill"


class LineageRelation(StrEnum):
    DERIVED_FROM = "derived_from"
    GENERATED_FROM = "generated_from"
    OBSERVED_IN = "observed_in"
    SUPPORTED_BY = "supported_by"

    DIAGNOSED_FROM = "diagnosed_from"
    PROPOSED_FROM = "proposed_from"
    PATCHED_FROM = "patched_from"
    TESTED_BY = "tested_by"
    EVALUATED_FROM = "evaluated_from"

    PROMOTED_FROM = "promoted_from"
    REJECTED_FROM = "rejected_from"

    RECOVERED_BY = "recovered_by"
    SUPERSEDES = "supersedes"
