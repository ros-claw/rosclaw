"""MCP capability adapter (PR-05, 大纲 §7.5).

Discovers tools from an external MCP server (e.g. ``limo-ros-mcp``) over
stdio and classifies every one as OBSERVE or PHYSICAL_ACTION. Classification
is fail-closed:

* ``readOnlyHint: true`` (or explicit ``observation_tools`` config) → OBSERVE.
* Explicit ``action_tools`` config, destructive/open-world annotations, or
  action verb names → PHYSICAL_ACTION.
* Anything ambiguous → PHYSICAL_ACTION (never model-callable).

Any PHYSICAL_ACTION tool is registered with ``model_callable=False`` and
``requires_exact_action_grant=True``: the model can see its descriptor in the
capabilities layer but can never call it; it flows only through
REQUEST_APPROVAL → Operator grant → rosclawd.

Discovery/health failures quarantine the whole source honestly — a dead MCP
server degrades the tool list, it never fabricates observations.

Server auth material is referenced as ``env:VAR`` names only; the adapter
reads values from the process environment at spawn time and never logs them.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from rosclaw.agentd.tooling.catalog import ToolCatalog
from rosclaw.contracts.agent.tool import (
    ExecutionClass,
    ToolDescriptorV2,
    ToolEvidenceClass,
    ToolSideEffectClass,
)
from rosclaw.contracts.common import ValidationError

#: action verb heuristic — names matching any of these are PHYSICAL_ACTION
_ACTION_VERBS = re.compile(
    r"(^|[._-])(play|move|set|write|publish|send|execute|exec|arm|disarm|drive|"
    r"rotate|speak|sound|tone|stop|start|run|goto|navigate|grasp|release|actuate|"
    r"command|control|reset|calibrate|dock|charge)($|[._-])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class McpServerConfig:
    """Static config for one external MCP server (from agentd YAML)."""

    name: str
    command: str
    args: tuple[str, ...] = ()
    #: env var NAMES to pass through (values read from process env at spawn)
    env_refs: tuple[str, ...] = ()
    observation_tools: tuple[str, ...] = ()
    action_tools: tuple[str, ...] = ()
    supported_modes: tuple[str, ...] = ("SIMULATION",)
    required_body_types: tuple[str, ...] = ()
    timeout_ms: int = 5000

    def spawn_env(self) -> dict[str, str]:
        env: dict[str, str] = {}
        for ref in self.env_refs:
            value = os.environ.get(ref)
            if value:  # missing → simply not passed; never logged
                env[ref] = value
        return env


@dataclass
class DiscoveryReport:
    server: str
    ok: bool
    tools: list[ToolDescriptorV2] = field(default_factory=list)
    error: str | None = None


class McpCapabilityAdapter:
    """One adapter per external MCP server."""

    def __init__(self, config: McpServerConfig, catalog: ToolCatalog) -> None:
        self._config = config
        self._catalog = catalog
        self.source = f"mcp:{config.name}"

    # -- classification ---------------------------------------------------------

    def classify(self, tool_name: str, annotations: Any) -> ExecutionClass:
        cfg = self._config
        if tool_name in cfg.action_tools:
            return ExecutionClass.PHYSICAL_ACTION
        if tool_name in cfg.observation_tools:
            return ExecutionClass.OBSERVE
        read_only = bool(getattr(annotations, "readOnlyHint", False)) if annotations else False
        destructive = (
            bool(getattr(annotations, "destructiveHint", False)) if annotations else False
        )
        if destructive:
            return ExecutionClass.PHYSICAL_ACTION
        if read_only and not _ACTION_VERBS.search(tool_name):
            return ExecutionClass.OBSERVE
        if _ACTION_VERBS.search(tool_name):
            return ExecutionClass.PHYSICAL_ACTION
        # ambiguous → fail closed
        return ExecutionClass.PHYSICAL_ACTION

    # -- discovery ---------------------------------------------------------------

    async def discover(self) -> DiscoveryReport:
        """Connect, list tools, classify, and (re)register in the catalog."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:  # pragma: no cover - mcp is a core dep
            raise ValidationError(f"mcp package unavailable: {exc}") from exc

        cfg = self._config
        params = StdioServerParameters(
            command=cfg.command, args=list(cfg.args), env=cfg.spawn_env() or None
        )
        report = DiscoveryReport(server=self.source, ok=False)
        try:
            async with (
                stdio_client(params) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                listed = await session.list_tools()
        except Exception as exc:  # noqa: BLE001 - surfaced as honest degradation
            report.error = f"{type(exc).__name__}: {exc}"
            self._catalog.quarantine_source(self.source, f"discovery_failed: {report.error}")
            return report

        self._catalog.lift_source_quarantine(self.source)
        for tool in listed.tools:
            execution_class = self.classify(tool.name, tool.annotations)
            physical = execution_class is ExecutionClass.PHYSICAL_ACTION
            descriptor = ToolDescriptorV2(
                tool_id=tool.name,
                source=self.source,
                execution_class=execution_class,
                side_effect_class=(
                    ToolSideEffectClass.REVERSIBLE if physical else ToolSideEffectClass.NONE
                ),
                description=tool.description or "",
                input_schema=dict(tool.inputSchema or {}),
                supported_modes=list(cfg.supported_modes),
                required_body_types=list(cfg.required_body_types),
                freshness_ms=500 if not physical else None,
                timeout_ms=cfg.timeout_ms,
                evidence_class=ToolEvidenceClass.MEASURED,
                verifier="schema+timestamp+frame",
                idempotent=not physical,
                model_callable=not physical,
                requires_exact_action_grant=physical,
            )
            self._catalog.replace(descriptor, self._make_executor(tool.name))
            report.tools.append(descriptor)
        report.ok = True
        return report

    async def health_check(self) -> bool:
        """Cheap liveness probe; failure quarantines the source (fail honest)."""
        report = await self.discover()
        return report.ok

    # -- execution ---------------------------------------------------------------

    def _make_executor(self, tool_name: str):
        async def _exec(arguments: dict[str, Any]) -> str:
            import json as _json

            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            cfg = self._config
            params = StdioServerParameters(
                command=cfg.command, args=list(cfg.args), env=cfg.spawn_env() or None
            )
            async with (
                stdio_client(params) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
            if result.isError:
                text = " ".join(
                    getattr(block, "text", "") for block in result.content
                ).strip()
                raise ValidationError(f"mcp tool {tool_name!r} error: {text or 'unknown'}")
            parts = [getattr(block, "text", "") for block in result.content]
            return _json.dumps(
                {"tool": tool_name, "source": self.source, "content": parts},
                ensure_ascii=False,
            )

        return _exec
