"""Catalog-backed ToolRegistry — the seam between the catalog and AgentLoop.

Implements the loop's ``ToolRegistry`` protocol (``strict_tools`` /
``execute``) and adds two PR-05 capabilities the loop prefers when present:

* ``resolve_tools(...)`` — resolver hard filters + ranking before injection;
* ``evidence_envelope(...)`` — the PR-05 evidence wrapper for observations.

PHYSICAL_ACTION tools are structurally unable to leak through either path.
"""

from __future__ import annotations

from typing import Any

from rosclaw.agentd.models.gateway import StrictTool
from rosclaw.agentd.tooling.artifact_result import ArtifactResultStore
from rosclaw.agentd.tooling.catalog import ToolCatalog
from rosclaw.agentd.tooling.evidence import EvidenceEnvelope, wrap_observation
from rosclaw.agentd.tooling.resolver import FilterContext, ToolResolver
from rosclaw.agentd.tooling.strict_schema import to_strict_tool
from rosclaw.contracts.common import ValidationError


class CatalogToolRegistry:
    def __init__(
        self,
        catalog: ToolCatalog,
        resolver: ToolResolver,
        *,
        artifact_store: ArtifactResultStore | None = None,
        body_type: str = "",
        online_capabilities: frozenset[str] = frozenset(),
        granted_permissions: frozenset[str] = frozenset(),
        policy_denied_tools: frozenset[str] = frozenset(),
        self_snapshot_fresh: bool = True,
    ) -> None:
        self._catalog = catalog
        self._resolver = resolver
        self._artifact_store = artifact_store
        self._body_type = body_type
        self._online_capabilities = online_capabilities
        self._granted_permissions = granted_permissions
        self._policy_denied = policy_denied_tools
        self._self_snapshot_fresh = self_snapshot_fresh

    # -- loop ToolRegistry protocol ----------------------------------------------

    def strict_tools(self, names: list[str]) -> list[StrictTool]:
        """Legacy path: allowlist intersection only (no ranking)."""
        tools: list[StrictTool] = []
        for name in names:
            descriptor = self._catalog.get(self._catalog._canonical(name))
            if descriptor is None or not descriptor.model_callable:
                continue
            if self._catalog.quarantine_reason(name) is not None:
                continue
            tools.append(to_strict_tool(descriptor))
        return tools

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        return await self._catalog.execute(name, arguments)

    # -- PR-05 extensions ---------------------------------------------------------

    def resolve_tools(
        self,
        candidates: list[str],
        *,
        mode: str = "SIMULATION",
        budget_exceeded: bool = False,
        task_hint: str = "",
    ) -> list[StrictTool]:
        ctx = FilterContext(
            body_type=self._body_type,
            mode=mode,
            online_capabilities=self._online_capabilities,
            self_snapshot_fresh=self._self_snapshot_fresh,
            granted_permissions=self._granted_permissions,
            policy_denied_tools=self._policy_denied,
            budget_exceeded=budget_exceeded,
            task_hint=task_hint,
        )
        result = self._resolver.resolve(ctx, candidates=candidates or None)
        return [to_strict_tool(d) for d in result.injected]

    def excluded_reasons(
        self,
        candidates: list[str],
        *,
        mode: str = "SIMULATION",
        budget_exceeded: bool = False,
        task_hint: str = "",
    ) -> dict[str, tuple[str, ...]]:
        """Explainable audit view: why each candidate was not injected."""
        ctx = FilterContext(
            body_type=self._body_type,
            mode=mode,
            online_capabilities=self._online_capabilities,
            self_snapshot_fresh=self._self_snapshot_fresh,
            granted_permissions=self._granted_permissions,
            policy_denied_tools=self._policy_denied,
            budget_exceeded=budget_exceeded,
            task_hint=task_hint,
        )
        result = self._resolver.resolve(ctx, candidates=candidates or None)
        return {d.tool_id: d.reasons for d in result.excluded}

    def evidence_envelope(
        self,
        tool_id: str,
        output: str,
        *,
        body_id: str,
        error: str | None = None,
    ) -> EvidenceEnvelope:
        descriptor = self._catalog.get(tool_id)
        if descriptor is None:
            raise ValidationError(f"tool {tool_id!r} not in catalog")
        return wrap_observation(
            descriptor,
            output,
            body_id=body_id,
            artifact_store=self._artifact_store,
            error=error,
        )
