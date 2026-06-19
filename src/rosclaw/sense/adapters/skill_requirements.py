"""Adapter stub for skill_requirements integration.

This module is a placeholder for Phase 1.  It defines the integration surface
so that later phases can implement body-sense-aware behavior without changing
import paths.
"""

from __future__ import annotations

from typing import Any


class SkillRequirementsAdapter:
    """Stub adapter: skill_requirements."""

    def __init__(self, sense_runtime: object | None = None):
        self.sense_runtime = sense_runtime

    def apply(self, context: dict[str, Any]) -> dict[str, Any]:
        """No-op pass-through for Phase 1."""
        return context
