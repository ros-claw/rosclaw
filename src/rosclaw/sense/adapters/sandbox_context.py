"""Sense-aware adapter for sandbox action validation."""

from __future__ import annotations

from typing import Any

from rosclaw.sense.adapters._base import SenseAdapterBase


class SandboxContextAdapter(SenseAdapterBase):
    """Inject a body-sense snapshot into sandbox action context.

    Adds ``body_sense_snapshot`` when a current BodySense is available.
    If sense is unavailable, the input context is returned unchanged.
    """

    def apply(self, context: dict[str, Any]) -> dict[str, Any]:
        sense = self._get_sense_dict()
        if sense is None:
            return context
        return {**context, "body_sense_snapshot": sense}
