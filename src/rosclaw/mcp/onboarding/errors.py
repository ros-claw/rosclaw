"""ROSClaw Hardware MCP onboarding exception hierarchy."""

from __future__ import annotations


class OnboardingError(Exception):
    """Base exception for the Hardware MCP onboarding subsystem."""


class ManifestError(OnboardingError):
    """Manifest parsing or validation failed."""


class ManifestNotFoundError(ManifestError):
    """Requested manifest could not be resolved locally or from a hub."""


class AliasResolutionError(OnboardingError):
    """Failed to resolve a short name or alias to a canonical manifest ID."""


class VersionResolutionError(OnboardingError):
    """Failed to select a compatible manifest version."""


class ArtifactError(OnboardingError):
    """Artifact descriptor is invalid or cannot be fetched."""


class PlatformNotSupportedError(OnboardingError):
    """Current platform is not in the manifest supportedPlatforms list."""


class PreflightError(OnboardingError):
    """A required preflight check failed."""

    def __init__(self, message: str, check_id: str | None = None) -> None:
        super().__init__(message)
        self.check_id = check_id


class InstallationError(OnboardingError):
    """Package or runtime installation failed."""


class PermissionDeniedError(OnboardingError):
    """User denied a required permission or permission is forbidden by default."""


class BindingError(OnboardingError):
    """e-URDF or body.yaml binding failed."""


class BodyNotLinkedError(BindingError):
    """body.yaml is not yet linked."""


class EurdfProfileMissingError(BindingError):
    """Required e-URDF profile is missing and auto-install is disabled or failed."""


class EurdfHashMismatchError(BindingError):
    """e-URDF profile hash does not match the manifest declaration."""


class ClaudeMergeError(OnboardingError):
    """Merging into .mcp.json failed or produced an invalid configuration."""


class HealthCheckError(OnboardingError):
    """A health check could not be executed."""


class RollbackError(OnboardingError):
    """Rollback of a partially applied installation failed."""
