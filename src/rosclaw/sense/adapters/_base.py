"""Base helper for sense-aware context adapters."""

from __future__ import annotations

from typing import Any


class SenseAdapterBase:
    """Minimal base for adapters that enrich caller context with BodySense.

    Adapters are stateless transformers: ``apply(context) -> context``.  They
    must never raise; if body sense is unavailable they return the input
    context unchanged.
    """

    def __init__(self, sense_runtime: object | None = None):
        self.sense_runtime = sense_runtime

    def _get_sense_dict(self) -> dict[str, Any] | None:
        """Return the latest BodySense as a dict, or None if unavailable."""
        if self.sense_runtime is None:
            return None
        try:
            sense = self.sense_runtime.get_latest_sense()
            if sense is None:
                sense = self.sense_runtime.tick()
            if sense is None:
                return None
            return sense.to_dict()
        except Exception:
            return None
