"""Interaction policy selection for guarded actions."""

from __future__ import annotations

from enum import StrEnum


class AuthorizationTier(StrEnum):
    EXACT_ACTION = "EXACT_ACTION"
    PLAN = "PLAN"
    MISSION = "MISSION"
    SITE_POLICY = "SITE_POLICY"


def required_authorization_tier(risk_tier: str) -> AuthorizationTier:
    """The first SDK release intentionally confirms each medium+ action exactly."""

    if str(risk_tier).upper() == "LOW":
        return AuthorizationTier.PLAN
    return AuthorizationTier.EXACT_ACTION
