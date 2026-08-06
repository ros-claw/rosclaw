"""Durable model usage metering (PR-NA-030b).

One row per model turn in ``model_usage`` (migration 003); aggregates are
computed, never stored as mutable counters. Cost is computed from the
profile's per-million-token prices in microunits (1 unit = 1e-6 元, so
budgets can be compared against ``monetary_microunits``).
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Any

from rosclaw.contracts.agent.model_turn import ModelTurnResultV1
from rosclaw.contracts.common import new_id


class UsageRecorder:
    """Writes usage rows on the MissionStore's connection."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def record(self, turn: ModelTurnResultV1) -> str:
        usage_id = new_id("usage")
        usage = turn.usage
        self._conn.execute(
            "INSERT INTO model_usage (usage_id, mission_id, turn_id, provider, model, "
            "profile, prompt_tokens, completion_tokens, reasoning_tokens, total_tokens, "
            "cost_microunits, latency_ms, provider_request_id, context_id, "
            "context_revision, finish_reason, recorded_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                usage_id,
                turn.mission_id or "",
                turn.turn_id,
                turn.provider,
                turn.model,
                turn.profile,
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.reasoning_tokens,
                usage.total_tokens,
                usage.cost_microunits,
                turn.latency_ms,
                turn.provider_request_id,
                turn.context_id,
                turn.context_revision,
                turn.finish_reason,
                datetime.now(UTC).isoformat(),
            ),
        )
        return usage_id

    def mission_totals(self, mission_id: str) -> dict[str, int]:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(prompt_tokens),0) AS pt, "
            "COALESCE(SUM(completion_tokens),0) AS ct, "
            "COALESCE(SUM(total_tokens),0) AS tt, "
            "COALESCE(SUM(cost_microunits),0) AS cost, COUNT(*) AS turns "
            "FROM model_usage WHERE mission_id = ?",
            (mission_id,),
        ).fetchone()
        return {
            "prompt_tokens": int(row["pt"]),
            "completion_tokens": int(row["ct"]),
            "total_tokens": int(row["tt"]),
            "cost_microunits": int(row["cost"]),
            "model_turns": int(row["turns"]),
        }

    def rows(self, mission_id: str) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT * FROM model_usage WHERE mission_id = ? ORDER BY recorded_at",
            (mission_id,),
        )
        return [dict(r) for r in cur.fetchall()]


def estimate_cost_microunits(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    price_input_per_mtok: int,
    price_output_per_mtok: int,
) -> int:
    """Microunits = tokens * per-million price / 1e6 (integer floor)."""
    return (
        prompt_tokens * price_input_per_mtok + completion_tokens * price_output_per_mtok
    ) // 1_000_000
