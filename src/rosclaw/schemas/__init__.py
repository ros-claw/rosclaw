"""rosclaw.schemas — Unified schema namespace for ROSClaw 1.0.

All public dataclasses / Pydantic models used across modules are exported
from here so callers can import a single canonical shape instead of
depending on private module internals.
"""
from __future__ import annotations

# Hardware MCP onboarding
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

# Auto
from .auto_schemas import (
    AutoPatch,
    AutoProposal,
    Champion,
    DeadEnd,
    EvaluationResult,
    ExperimentSpec,
)

# Context
from .context import AgentContext, RuntimeContext, TaskContext

# Events
from .events import EventEnvelope

# Intervention / Evidence
from .intervention import (
    EvidenceTrace,
    InterventionDecision,
    InterventionRequest,
    InterventionTrace,
    ObjectiveDirection,
)

# Provider
from .provider import ProviderRequest, ProviderResponse

# Sandbox
from .sandbox import SandboxDecision, SandboxSession

# Skills
from .skills import (
    ChampionLevel,
    SkillCandidate,
    SkillLineage,
    SkillVersion,
)

__all__ = [
    # context
    "AgentContext",
    "RuntimeContext",
    "TaskContext",
    # events
    "EventEnvelope",
    # intervention
    "ObjectiveDirection",
    "InterventionRequest",
    "InterventionDecision",
    "InterventionTrace",
    "EvidenceTrace",
    # auto
    "AutoProposal",
    "AutoPatch",
    "ExperimentSpec",
    "EvaluationResult",
    "Champion",
    "DeadEnd",
    # skills
    "ChampionLevel",
    "SkillVersion",
    "SkillCandidate",
    "SkillLineage",
    # provider
    "ProviderRequest",
    "ProviderResponse",
    # sandbox
    "SandboxDecision",
    "SandboxSession",
    # hardware MCP onboarding
    "Artifact",
    "BodyBindingTemplate",
    "ClaudeMcpConfig",
    "HealthCheck",
    "McpConfig",
    "McpManifest",
    "PermissionDecl",
    "Permissions",
    "Publisher",
]
