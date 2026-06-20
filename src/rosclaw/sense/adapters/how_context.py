"""Sense-aware adapter for recovery hint context."""

from __future__ import annotations

from typing import Any

from rosclaw.sense.adapters._base import SenseAdapterBase


class HowContextAdapter(SenseAdapterBase):
    """Enrich recovery context with body readiness and block reasons.

    Input context is expected to contain a ``task`` key.  The adapter adds
    ``body_readiness`` and ``body_block_reasons``.  If body sense is
    unavailable, the input context is returned unchanged.
    """

    def apply(self, context: dict[str, Any]) -> dict[str, Any]:
        task = context.get("task")
        if not task:
            return context

        sense = self._get_sense_dict()
        if sense is None:
            return context

        readiness = sense.get("readiness", {})
        item = readiness.get("capabilities", {}).get(task, {})
        reasons = list(item.get("reasons", []))
        block_reasons = [f"{task}: {r}" for r in reasons] if reasons else [f"{task}: not ready"]

        return {
            **context,
            "body_readiness": readiness,
            "body_block_reasons": block_reasons,
        }
