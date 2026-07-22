"""TaskDistillationAdapter protocol (数据库优化v3 §4.1)."""

from __future__ import annotations

from typing import Any, Protocol

from rosclaw.memory.v2.models import MemoryItem


class TaskDistillationAdapter(Protocol):
    """Task-specific extraction over one session's events."""

    task_ids: set[str]

    def extract_failures(
        self,
        context: Any,
        events: list[dict[str, Any]],
    ) -> list[MemoryItem]: ...

    def extract_body_patterns(
        self,
        context: Any,
        events: list[dict[str, Any]],
    ) -> list[MemoryItem]: ...

    def extract_skill_evidence(
        self,
        context: Any,
        events: list[dict[str, Any]],
    ) -> list[MemoryItem]: ...

    def build_episode_quality(
        self,
        context: Any,
        events: list[dict[str, Any]],
    ) -> dict[str, Any]: ...
