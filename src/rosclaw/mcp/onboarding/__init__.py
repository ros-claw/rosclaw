"""ROSClaw Hardware MCP onboarding subsystem."""

from __future__ import annotations

from rosclaw.mcp.onboarding.errors import (
    ArtifactError,
    BindingError,
    BodyNotLinkedError,
    ClaudeMergeError,
    EurdfHashMismatchError,
    EurdfProfileMissingError,
    HealthCheckError,
    InstallationError,
    ManifestError,
    ManifestNotFoundError,
    OnboardingError,
    PermissionDeniedError,
    PlatformNotSupportedError,
    PreflightError,
    RollbackError,
    VersionResolutionError,
)
from rosclaw.mcp.onboarding.hub_client import HubClient
from rosclaw.mcp.onboarding.installed import InstalledRecord, InstalledRegistry
from rosclaw.mcp.onboarding.lockfile import LockedPackage, Lockfile
from rosclaw.mcp.onboarding.permissions import PermissionState, PermissionStore
from rosclaw.mcp.onboarding.resolver import AliasResolver, SolvedVersion, VersionSolver
from rosclaw.mcp.onboarding.schema import (
    Artifact,
    BodyBindingTemplate,
    ClaudeMcpConfig,
    HealthCheck,
    McpConfig,
    McpManifest,
    PermissionDecl,
    Permissions,
    Publisher,
)

__all__ = [
    "AliasResolver",
    "AliasResolutionError",
    "Artifact",
    "ArtifactError",
    "BindingError",
    "BodyBindingTemplate",
    "BodyNotLinkedError",
    "ClaudeMcpConfig",
    "ClaudeMergeError",
    "EurdfHashMismatchError",
    "EurdfProfileMissingError",
    "HealthCheck",
    "HealthCheckError",
    "HubClient",
    "InstallationError",
    "InstalledRecord",
    "InstalledRegistry",
    "Lockfile",
    "LockedPackage",
    "ManifestError",
    "ManifestNotFoundError",
    "McpConfig",
    "McpManifest",
    "OnboardingError",
    "PermissionDecl",
    "PermissionDeniedError",
    "PermissionState",
    "PermissionStore",
    "Permissions",
    "PlatformNotSupportedError",
    "PreflightError",
    "Publisher",
    "RollbackError",
    "SolvedVersion",
    "VersionResolutionError",
    "VersionSolver",
]
