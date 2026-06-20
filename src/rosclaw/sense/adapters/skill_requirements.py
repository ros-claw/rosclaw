"""Sense-aware adapter for skill requirement checks."""

from __future__ import annotations

from typing import Any

from rosclaw.sense.adapters._base import SenseAdapterBase


class SkillRequirementsAdapter(SenseAdapterBase):
    """Enrich a skill-requirements context with a body-sense readiness check.

    Input context is expected to contain a ``task`` key.  The adapter adds a
    ``body_sense_check`` dictionary with the task's current readiness status
    and blocking reasons.  If body sense is unavailable, the input context is
    returned unchanged.
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
        body_sense_check = {
            "task": task,
            "status": item.get("status", "unknown"),
            "reasons": list(item.get("reasons", [])),
        }
        return {**context, "body_sense_check": body_sense_check}
