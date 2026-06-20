"""Sense-aware adapter for practice episode metadata."""

from __future__ import annotations

from typing import Any

from rosclaw.sense.adapters._base import SenseAdapterBase


class PracticeWriterAdapter(SenseAdapterBase):
    """Capture a body-sense snapshot at episode start/end.

    Input context is expected to contain a ``phase`` key (``start`` or
    ``end``).  The adapter adds ``body_sense_start`` or ``body_sense_end``
    respectively.  If body sense is unavailable, the input context is
    returned unchanged.
    """

    def apply(self, context: dict[str, Any]) -> dict[str, Any]:
        phase = context.get("phase")
        if phase not in ("start", "end"):
            return context

        sense = self._get_sense_dict()
        if sense is None:
            return context

        return {**context, f"body_sense_{phase}": sense}
