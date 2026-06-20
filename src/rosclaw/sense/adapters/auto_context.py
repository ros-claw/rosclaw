"""Sense-aware adapter for auto/experiment failure context."""

from __future__ import annotations

from typing import Any

from rosclaw.sense.adapters._base import SenseAdapterBase


class AutoContextAdapter(SenseAdapterBase):
    """Flag body-condition failures and attach a sense snapshot.

    Adds ``body_condition_failure`` (True when the robot is not ``ready``)
    and ``body_sense_snapshot`` to the caller's context.  If body sense is
    unavailable, the input context is returned unchanged.
    """

    def apply(self, context: dict[str, Any]) -> dict[str, Any]:
        sense = self._get_sense_dict()
        if sense is None:
            return context

        is_failure = sense.get("overall_status") != "ready"
        return {
            **context,
            "body_condition_failure": is_failure,
            "body_sense_snapshot": sense,
        }
