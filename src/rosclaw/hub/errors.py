"""Unified error types for the ROSClaw Hub subsystem."""

from __future__ import annotations

from enum import StrEnum


class HubErrorCode(StrEnum):
    """Canonical error codes returned by Hub operations."""

    AUTH_REQUIRED = "HUB_AUTH_REQUIRED"
    AUTH_FAILED = "HUB_AUTH_FAILED"
    REGISTRY_UNREACHABLE = "HUB_REGISTRY_UNREACHABLE"
    INDEX_VERIFY_FAILED = "HUB_INDEX_VERIFY_FAILED"
    ASSET_NOT_FOUND = "HUB_ASSET_NOT_FOUND"
    AMBIGUOUS_ASSET = "HUB_AMBIGUOUS_ASSET"
    MANIFEST_INVALID = "HUB_MANIFEST_INVALID"
    INCOMPATIBLE_RUNTIME = "HUB_INCOMPATIBLE_RUNTIME"
    INCOMPATIBLE_ROBOT = "HUB_INCOMPATIBLE_ROBOT"
    DEPENDENCY_MISSING = "HUB_DEPENDENCY_MISSING"
    LICENSE_DENIED = "HUB_LICENSE_DENIED"
    PERMISSION_DENIED = "HUB_PERMISSION_DENIED"
    CHECKSUM_MISMATCH = "HUB_CHECKSUM_MISMATCH"
    SIGNATURE_INVALID = "HUB_SIGNATURE_INVALID"
    INSTALL_HEALTH_FAILED = "HUB_INSTALL_HEALTH_FAILED"
    ROLLBACK_FAILED = "HUB_ROLLBACK_FAILED"
    PUBLISH_REJECTED = "HUB_PUBLISH_REJECTED"


class HubError(Exception):
    """Base exception for Hub operations.

    Attributes:
        code: Machine-readable error code.
        message: Human-readable message.
        suggested_fix: Optional hint for the user.
    """

    def __init__(
        self,
        code: HubErrorCode,
        message: str,
        suggested_fix: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.suggested_fix = suggested_fix

    def __str__(self) -> str:
        parts = [f"[{self.code}] {self.message}"]
        if self.suggested_fix:
            parts.append(f"Suggested fix: {self.suggested_fix}")
        return "\n".join(parts)
