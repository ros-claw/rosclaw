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
from base64 import b64decode, b64encode
from binascii import Error as Base64Error
from dataclasses import dataclass, field
from typing import Any

from rosclaw.agentd.tooling.catalog import ToolCatalog
from rosclaw.agentd.tooling.result import ToolExecutionResult, ToolImage
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

_IMAGE_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/webp"})
_MAX_IMAGE_BYTES = 2 * 1024 * 1024
_MAX_IMAGES_PER_RESULT = 4


def _matches_image_magic(mime_type: str, data: bytes) -> bool:
    if mime_type == "image/png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if mime_type == "image/jpeg":
        return data.startswith(b"\xff\xd8\xff")
    if mime_type == "image/webp":
        return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    return False


def _normalize_result(tool_name: str, source: str, result: Any) -> str | ToolExecutionResult:
    """Preserve bounded MCP image blocks while keeping a textual tool receipt."""
    import json as _json

    parts: list[str] = []
    images: list[ToolImage] = []
    dropped: list[str] = []
    for block in result.content:
        block_type = getattr(block, "type", "")
        if block_type == "text":
            parts.append(getattr(block, "text", ""))
            continue
        if block_type != "image":
            dropped.append(f"unsupported:{block_type or 'unknown'}")
            continue
        mime_type = str(getattr(block, "mimeType", ""))
        if mime_type not in _IMAGE_MIME_TYPES:
            dropped.append(f"unsupported_mime:{mime_type or 'missing'}")
            continue
        if len(images) >= _MAX_IMAGES_PER_RESULT:
            dropped.append("image_count_limit")
            continue
        raw_data = getattr(block, "data", "")
        if isinstance(raw_data, bytes):
            decoded = raw_data
            encoded = b64encode(raw_data).decode("ascii")
        else:
            encoded = str(raw_data)
            if len(encoded) > ((_MAX_IMAGE_BYTES + 2) // 3) * 4:
                dropped.append("image_size_limit")
                continue
            try:
                decoded = b64decode(encoded, validate=True)
            except (Base64Error, ValueError):
                dropped.append("invalid_base64")
                continue
        if not decoded or len(decoded) > _MAX_IMAGE_BYTES:
            dropped.append("image_size_limit")
            continue
        if not _matches_image_magic(mime_type, decoded):
            dropped.append("image_signature_mismatch")
            continue
        images.append(ToolImage(mime_type=mime_type, data_base64=encoded))
    # PR-N5B：单文本块且可解析为 JSON 对象 → canonical value 是内层
    # 载荷本身（传输包装 tool/source/content 不是工具输出）。含图片/
    # dropped 块时保留包装（图像证据边界不动）。
    if not images and not dropped and len(parts) == 1:
        try:
            inner = _json.loads(parts[0])
        except ValueError:
            inner = None
        if isinstance(inner, dict):
            return _json.dumps(inner, ensure_ascii=False)
    payload: dict[str, Any] = {
        "tool": tool_name,
        "source": source,
        "content": parts,
    }
    if images:
        payload["image_blocks"] = [
            {"mime_type": image.mime_type, "bytes": len(b64decode(image.data_base64))}
            for image in images
        ]
    if dropped:
        payload["dropped_blocks"] = dropped
    text = _json.dumps(payload, ensure_ascii=False)
    if not images:
        return text
    return ToolExecutionResult(text=text, images=tuple(images))


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
    compute_tools: tuple[str, ...] = ()
    supported_modes: tuple[str, ...] = ("SIMULATION",)
    required_body_types: tuple[str, ...] = ()
    #: 七审 §2.5：该 server 动作工具的效果域（sim 服务器
    #: simulation_state_only；缺省 fail closed 不自动批准）。
    effect_domain: str = ""
    timeout_ms: int = 5000
    #: PR-N5B/N5E 过渡：server 声明的 per-tool output_schema（第一方
    #: kit 显式声明；未声明的工具 execute_v2 诚实 OUTPUT_SCHEMA_MISSING，
    #: 不猜结构）。
    output_schemas: dict = None  # type: ignore[assignment]  # dataclass 默认见 __post_init__
    def __post_init__(self) -> None:
        if self.output_schemas is None:
            object.__setattr__(self, "output_schemas", {})

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

    def __init__(
        self,
        config: McpServerConfig,
        catalog: ToolCatalog,
        client=None,
    ) -> None:
        self._config = config
        self._catalog = catalog
        #: 可选共享持久会话（有状态 SIM 身体必须观测/执行同进程）。
        self._client = client
        self.source = f"mcp:{config.name}"

    # -- classification ---------------------------------------------------------

    def classify(self, tool_name: str, annotations: Any) -> ExecutionClass:
        cfg = self._config
        if tool_name in cfg.action_tools:
            return ExecutionClass.PHYSICAL_ACTION
        if tool_name in cfg.compute_tools:
            return ExecutionClass.COMPUTE
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
        report = DiscoveryReport(server=self.source, ok=False)
        try:
            if self._client is not None:
                tools = await self._client.list_tools()

                class _Listed:
                    pass

                listed = _Listed()
                listed.tools = tools
            else:
                params = StdioServerParameters(
                    command=cfg.command, args=list(cfg.args), env=cfg.spawn_env() or None
                )
                from rosclaw.agentd.tooling.persistent_client import _safe_errlog

                async with (
                    stdio_client(params, errlog=_safe_errlog()) as (read, write),
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
                output_schema=dict(cfg.output_schemas.get(tool.name, {})),
                supported_modes=list(cfg.supported_modes),
                required_body_types=list(cfg.required_body_types),
                effect_domain=cfg.effect_domain if physical else "",
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
        async def _exec(arguments: dict[str, Any]) -> str | ToolExecutionResult:
            import json as _json

            if self._client is not None:
                raw = await self._client.call_tool(tool_name, arguments)
                if isinstance(raw, ToolExecutionResult):
                    return raw
                # PR-N5B：内层 JSON 对象即 canonical value（不套传输包装）。
                if isinstance(raw, str):
                    try:
                        inner = _json.loads(raw)
                    except ValueError:
                        inner = None
                    if isinstance(inner, dict):
                        return inner
                if isinstance(raw, dict):
                    return raw
                return _json.dumps(
                    {"tool": tool_name, "source": self.source, "content": [raw]},
                    ensure_ascii=False,
                )
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
            return _normalize_result(tool_name, self.source, result)

        return _exec
