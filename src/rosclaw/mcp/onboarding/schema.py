"""ROSClaw Hardware MCP manifest schema.

This module defines the dataclass model for the Hardware MCP manifest
(https://schemas.rosclaw.io/mcp/hardware-manifest.schema.json) and related
Claude Code configuration fragments.

All dataclasses provide ``from_dict`` / ``to_dict`` helpers so the layer
remains serialization-agnostic (JSON in production, dict in tests).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


# ── Small helper to keep defaults DRY ──
def _dict(data: dict[str, Any] | None) -> dict[str, Any]:
    return dict(data) if data else {}


def _list(data: list[Any] | None) -> list[Any]:
    return list(data) if data else []


# ── Publisher ──

@dataclass
class Publisher:
    """Publisher metadata for a Hardware MCP manifest."""

    name: str
    namespace: str
    homepage: str | None = None
    support: str | None = None
    verified: bool = False
    verification: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Publisher":
        return cls(
            name=data["name"],
            namespace=data["namespace"],
            homepage=data.get("homepage"),
            support=data.get("support"),
            verified=bool(data.get("verified", False)),
            verification=_dict(data.get("verification")),
        )


# ── Artifact ──

@dataclass
class Artifact:
    """Distribution artifact descriptor.

    Supported types: ``pypi``, ``oci``, ``npm``, ``remote``.
    Only one artifact is declared per manifest in P1.
    """

    type: str
    package: str | None = None
    image: str | None = None
    url: str | None = None
    version: str | None = None
    python: str | None = None
    node: str | None = None
    entrypoint: str | None = None
    install: str | None = None
    platforms: list[str] = field(default_factory=list)
    hashes: dict[str, str] = field(default_factory=dict)
    integrity: str | None = None
    sbom: str | None = None
    signature: dict[str, Any] = field(default_factory=dict)
    auth: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Artifact":
        return cls(
            type=data.get("type", "pypi"),
            package=data.get("package"),
            image=data.get("image"),
            url=data.get("url"),
            version=data.get("version"),
            python=data.get("python"),
            node=data.get("node"),
            entrypoint=data.get("entrypoint"),
            install=data.get("install"),
            platforms=_list(data.get("platforms")),
            hashes=dict(data.get("hashes", {})),
            integrity=data.get("integrity"),
            sbom=data.get("sbom"),
            signature=_dict(data.get("signature")),
            auth=_dict(data.get("auth")),
        )


# ── MCP runtime configuration ──

@dataclass
class McpTransport:
    """MCP transport descriptor for the manifest."""

    type: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "McpTransport":
        return cls(
            type=data.get("type", "stdio"),
            command=data.get("command", ""),
            args=_list(data.get("args")),
            env={str(k): str(v) for k, v in (data.get("env") or {}).items()},
            url=data.get("url"),
        )


@dataclass
class McpCapabilities:
    """Capability flags advertised by the MCP server."""

    tools: bool = True
    resources: bool = False
    prompts: bool = False
    sampling: bool = False
    logging: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "McpCapabilities":
        data = data or {}
        return cls(
            tools=bool(data.get("tools", True)),
            resources=bool(data.get("resources", False)),
            prompts=bool(data.get("prompts", False)),
            sampling=bool(data.get("sampling", False)),
            logging=bool(data.get("logging", False)),
        )


@dataclass
class McpStartup:
    """Startup probes and restart policy."""

    timeout_ms: int = 5000
    ready_pattern: str | None = None
    restart_policy: str = "on-failure"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "McpStartup":
        data = data or {}
        return cls(
            timeout_ms=int(data.get("timeoutMs", 5000)),
            ready_pattern=data.get("readyPattern"),
            restart_policy=data.get("restartPolicy", "on-failure"),
        )


@dataclass
class McpConfig:
    """MCP server runtime configuration embedded in the manifest."""

    server_name: str
    protocol_version: str = "2025-11-25"
    transport: McpTransport = field(default_factory=lambda: McpTransport(type="stdio", command=""))
    capabilities: McpCapabilities = field(default_factory=McpCapabilities)
    expected_tools: list[str] = field(default_factory=list)
    expected_resources: list[str] = field(default_factory=list)
    expected_prompts: list[str] = field(default_factory=list)
    startup: McpStartup = field(default_factory=McpStartup)

    def to_dict(self) -> dict[str, Any]:
        return {
            "serverName": self.server_name,
            "protocolVersion": self.protocol_version,
            "transport": self.transport.to_dict(),
            "capabilities": self.capabilities.to_dict(),
            "expectedTools": list(self.expected_tools),
            "expectedResources": list(self.expected_resources),
            "expectedPrompts": list(self.expected_prompts),
            "startup": self.startup.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "McpConfig":
        return cls(
            server_name=data["serverName"],
            protocol_version=data.get("protocolVersion", "2025-11-25"),
            transport=McpTransport.from_dict(data.get("transport", {})),
            capabilities=McpCapabilities.from_dict(data.get("capabilities")),
            expected_tools=_list(data.get("expectedTools")),
            expected_resources=_list(data.get("expectedResources")),
            expected_prompts=_list(data.get("expectedPrompts")),
            startup=McpStartup.from_dict(data.get("startup")),
        )


# ── Hardware description ──

@dataclass
class HardwareConnection:
    """Connection options for a hardware device."""

    modes: list[str] = field(default_factory=list)
    default_mode: str = "ros2"
    network: dict[str, Any] = field(default_factory=dict)
    ros2: dict[str, Any] = field(default_factory=dict)
    serial: dict[str, Any] = field(default_factory=dict)
    usb: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "HardwareConnection":
        data = data or {}
        return cls(
            modes=_list(data.get("modes")),
            default_mode=data.get("defaultMode", "ros2"),
            network=_dict(data.get("network")),
            ros2=_dict(data.get("ros2")),
            serial=_dict(data.get("serial")),
            usb=_dict(data.get("usb")),
        )


@dataclass
class HardwareIdentity:
    """Hardware identity / serial probing requirements."""

    serial_required: bool = False
    serial_probe: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "HardwareIdentity":
        data = data or {}
        return cls(
            serial_required=bool(data.get("serialRequired", False)),
            serial_probe=_dict(data.get("serialProbe")),
        )


@dataclass
class Hardware:
    """Hardware model and connection metadata."""

    type: str
    vendor: str | None = None
    models: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    connection: HardwareConnection = field(default_factory=HardwareConnection)
    identity: HardwareIdentity = field(default_factory=HardwareIdentity)
    safe_modes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Hardware":
        return cls(
            type=data.get("type", "robot"),
            vendor=data.get("vendor"),
            models=_list(data.get("models")),
            aliases=_list(data.get("aliases")),
            connection=HardwareConnection.from_dict(data.get("connection")),
            identity=HardwareIdentity.from_dict(data.get("identity")),
            safe_modes=_list(data.get("safeModes")),
        )


# ── e-URDF binding ──

@dataclass
class EurdfProfileRef:
    """Reference to a required e-URDF profile."""

    id: str
    version: str
    uri: str | None = None
    sha256: str | None = None
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EurdfProfileRef":
        return cls(
            id=data["id"],
            version=data.get("version", "1.0.0"),
            uri=data.get("uri"),
            sha256=data.get("sha256"),
            required=bool(data.get("required", True)),
        )


@dataclass
class EurdfBinding:
    """e-URDF binding requirements and capability map."""

    profiles: list[EurdfProfileRef] = field(default_factory=list)
    default_profile: str | None = None
    link_policy: dict[str, str] = field(default_factory=dict)
    capability_map: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profiles": [p.to_dict() for p in self.profiles],
            "defaultProfile": self.default_profile,
            "linkPolicy": dict(self.link_policy),
            "capabilityMap": dict(self.capability_map),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EurdfBinding":
        data = data or {}
        return cls(
            profiles=[EurdfProfileRef.from_dict(p) for p in data.get("profiles", [])],
            default_profile=data.get("defaultProfile"),
            link_policy=_dict(data.get("linkPolicy")),
            capability_map=_dict(data.get("capabilityMap")),
        )


# ── body.yaml binding template ──

@dataclass
class BodyBindingTemplate:
    """Template for writing this MCP binding into body.yaml."""

    body_type: str
    binding_key: str
    required_fields: list[str] = field(default_factory=list)
    write_paths: dict[str, str] = field(default_factory=dict)
    template: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bodyType": self.body_type,
            "bindingKey": self.binding_key,
            "requiredFields": list(self.required_fields),
            "writePaths": dict(self.write_paths),
            "template": dict(self.template),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BodyBindingTemplate":
        data = data or {}
        return cls(
            body_type=data.get("bodyType", ""),
            binding_key=data.get("bindingKey", ""),
            required_fields=_list(data.get("requiredFields")),
            write_paths=_dict(data.get("writePaths")),
            template=_dict(data.get("template")),
        )


# ── Permissions ──

@dataclass
class PermissionDecl:
    """A single permission declaration."""

    id: str
    level: str
    description: str = ""
    requires: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PermissionDecl":
        return cls(
            id=data["id"],
            level=data["level"],
            description=data.get("description", ""),
            requires=_list(data.get("requires")),
        )


@dataclass
class DataAccessDecl:
    """Data access permission declaration."""

    id: str
    classification: str
    default: str = "ask"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataAccessDecl":
        return cls(
            id=data["id"],
            classification=data.get("classification", "unknown"),
            default=data.get("default", "ask"),
        )


@dataclass
class Permissions:
    """Permission model for a Hardware MCP."""

    default_mode: str = "least_privilege"
    required: list[PermissionDecl] = field(default_factory=list)
    optional: list[PermissionDecl] = field(default_factory=list)
    data_access: list[DataAccessDecl] = field(default_factory=list)
    network_access: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "defaultMode": self.default_mode,
            "required": [p.to_dict() for p in self.required],
            "optional": [p.to_dict() for p in self.optional],
            "dataAccess": [d.to_dict() for d in self.data_access],
            "networkAccess": dict(self.network_access),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Permissions":
        data = data or {}
        return cls(
            default_mode=data.get("defaultMode", "least_privilege"),
            required=[PermissionDecl.from_dict(p) for p in data.get("required", [])],
            optional=[PermissionDecl.from_dict(p) for p in data.get("optional", [])],
            data_access=[DataAccessDecl.from_dict(d) for d in data.get("dataAccess", [])],
            network_access=_dict(data.get("networkAccess")),
        )


# ── Install declaration ──

@dataclass
class PreflightCheck:
    """A single preflight check to run before installation."""

    id: str
    command: str
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreflightCheck":
        return cls(
            id=data["id"],
            command=data.get("command", ""),
            required=bool(data.get("required", True)),
        )


@dataclass
class Runtime:
    """Runtime installer recipe."""

    type: str
    python: str | None = None
    install_command: str | None = None
    image: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Runtime":
        return cls(
            type=data.get("type", "python"),
            python=data.get("python"),
            install_command=data.get("installCommand"),
            image=data.get("image"),
        )


@dataclass
class PostInstallStep:
    """Post-installation action."""

    id: str
    action: str
    target: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PostInstallStep":
        return cls(
            id=data["id"],
            action=data.get("action", ""),
            target=data.get("target"),
        )


@dataclass
class InstallDecl:
    """Installation metadata: platforms, runtimes, preflight, post-install."""

    supported_platforms: list[str] = field(default_factory=list)
    preferred_runtime: str = "python"
    runtimes: list[Runtime] = field(default_factory=list)
    preflight: list[PreflightCheck] = field(default_factory=list)
    post_install: list[PostInstallStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "supportedPlatforms": list(self.supported_platforms),
            "preferredRuntime": self.preferred_runtime,
            "runtimes": [r.to_dict() for r in self.runtimes],
            "preflight": [p.to_dict() for p in self.preflight],
            "postInstall": [p.to_dict() for p in self.post_install],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "InstallDecl":
        data = data or {}
        return cls(
            supported_platforms=_list(data.get("supportedPlatforms")),
            preferred_runtime=data.get("preferredRuntime", "python"),
            runtimes=[Runtime.from_dict(r) for r in data.get("runtimes", [])],
            preflight=[PreflightCheck.from_dict(p) for p in data.get("preflight", [])],
            post_install=[PostInstallStep.from_dict(p) for p in data.get("postInstall", [])],
        )


# ── Health declaration ──

@dataclass
class HealthCheck:
    """A single health check declaration."""

    id: str
    category: str
    required: bool = True
    expect: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HealthCheck":
        return cls(
            id=data["id"],
            category=data.get("category", "install"),
            required=bool(data.get("required", True)),
            expect=_dict(data.get("expect")),
        )


@dataclass
class HealthDecl:
    """Health check section of the manifest."""

    startup_timeout_ms: int = 5000
    checks: list[HealthCheck] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "startupTimeoutMs": self.startup_timeout_ms,
            "checks": [c.to_dict() for c in self.checks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "HealthDecl":
        data = data or {}
        return cls(
            startup_timeout_ms=int(data.get("startupTimeoutMs", 5000)),
            checks=[HealthCheck.from_dict(c) for c in data.get("checks", [])],
        )


# ── Claude Code merge template ──

@dataclass
class ClaudeMcpConfig:
    """Fragment written into the project ``.mcp.json``."""

    scope_default: str = "project"
    mcp_json: dict[str, Any] = field(default_factory=dict)
    claude_md_snippet: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "scopeDefault": self.scope_default,
            "mcpJson": dict(self.mcp_json),
            "claudeMdSnippet": self.claude_md_snippet,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ClaudeMcpConfig":
        data = data or {}
        return cls(
            scope_default=data.get("scopeDefault", "project"),
            mcp_json=_dict(data.get("mcpJson")),
            claude_md_snippet=data.get("claudeMdSnippet", ""),
        )


# ── Lifecycle / compatibility / security ──

@dataclass
class LifecycleDecl:
    """Lifecycle metadata for a manifest."""

    deprecated: bool = False
    replaced_by: str | None = None
    eol_date: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LifecycleDecl":
        data = data or {}
        return cls(
            deprecated=bool(data.get("deprecated", False)),
            replaced_by=data.get("replacedBy"),
            eol_date=data.get("eolDate"),
        )


@dataclass
class CompatibilityDecl:
    """Compatibility constraints."""

    ros_distros: list[str] = field(default_factory=list)
    python: str | None = None
    rosclaw: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CompatibilityDecl":
        data = data or {}
        return cls(
            ros_distros=_list(data.get("rosDistros")),
            python=data.get("python"),
            rosclaw=data.get("rosclaw"),
        )


@dataclass
class SecurityDecl:
    """Security metadata and sandbox settings."""

    sandbox_required: bool = False
    network_isolation: str = "default"
    secrets: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SecurityDecl":
        data = data or {}
        return cls(
            sandbox_required=bool(data.get("sandboxRequired", False)),
            network_isolation=data.get("networkIsolation", "default"),
            secrets=[dict(s) for s in data.get("secrets", [])],
        )


# ── Top-level manifest ──

@dataclass
class McpManifest:
    """Complete ROSClaw Hardware MCP manifest."""

    id: str
    name: str
    version: str
    display_name: str
    schema_version: str = "1.0.0"
    channel: str = "stable"
    description: str = ""
    tags: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    license: str | None = None
    homepage: str | None = None
    repository: str | None = None
    publisher: Publisher | None = None
    artifact: Artifact | None = None
    mcp: McpConfig | None = None
    hardware: Hardware | None = None
    eurdf: EurdfBinding | None = None
    body_binding: BodyBindingTemplate | None = None
    permissions: Permissions | None = None
    install: InstallDecl | None = None
    health: HealthDecl | None = None
    claude: ClaudeMcpConfig | None = None
    compatibility: CompatibilityDecl | None = None
    security: SecurityDecl | None = None
    lifecycle: LifecycleDecl | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def canonical_id(self) -> str:
        """Return the canonical manifest ID."""
        return self.id

    @property
    def server_name(self) -> str:
        """Return the configured MCP server name, falling back to a safe default."""
        if self.mcp:
            return self.mcp.server_name
        return f"rosclaw-{self.name}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict matching the JSON manifest shape."""
        result: dict[str, Any] = {
            "$schema": "https://schemas.rosclaw.io/mcp/hardware-manifest.schema.json",
            "schemaVersion": self.schema_version,
            "id": self.id,
            "name": self.name,
            "displayName": self.display_name,
            "version": self.version,
            "channel": self.channel,
            "description": self.description,
        }
        if self.tags:
            result["tags"] = list(self.tags)
        if self.categories:
            result["categories"] = list(self.categories)
        if self.license:
            result["license"] = self.license
        if self.homepage:
            result["homepage"] = self.homepage
        if self.repository:
            result["repository"] = self.repository
        if self.publisher:
            result["publisher"] = self.publisher.to_dict()
        if self.artifact:
            result["artifact"] = self.artifact.to_dict()
        if self.mcp:
            result["mcp"] = self.mcp.to_dict()
        if self.hardware:
            result["hardware"] = self.hardware.to_dict()
        if self.eurdf:
            result["eurdf"] = self.eurdf.to_dict()
        if self.body_binding:
            result["bodyBinding"] = self.body_binding.to_dict()
        if self.permissions:
            result["permissions"] = self.permissions.to_dict()
        if self.install:
            result["install"] = self.install.to_dict()
        if self.health:
            result["health"] = self.health.to_dict()
        if self.claude:
            result["claude"] = self.claude.to_dict()
        if self.compatibility:
            result["compatibility"] = self.compatibility.to_dict()
        if self.security:
            result["security"] = self.security.to_dict()
        if self.lifecycle:
            result["lifecycle"] = self.lifecycle.to_dict()
        if self.extra:
            result.update(self.extra)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "McpManifest":
        """Deserialize from a dict, tolerating missing optional sections."""
        extra = {
            k: v
            for k, v in data.items()
            if k
            not in {
                "$schema",
                "schemaVersion",
                "id",
                "name",
                "displayName",
                "version",
                "channel",
                "description",
                "tags",
                "categories",
                "license",
                "homepage",
                "repository",
                "publisher",
                "artifact",
                "mcp",
                "hardware",
                "eurdf",
                "bodyBinding",
                "permissions",
                "install",
                "health",
                "claude",
                "compatibility",
                "security",
                "lifecycle",
            }
        }
        return cls(
            id=data["id"],
            name=data["name"],
            version=data["version"],
            display_name=data.get("displayName", data["name"]),
            schema_version=data.get("schemaVersion", "1.0.0"),
            channel=data.get("channel", "stable"),
            description=data.get("description", ""),
            tags=_list(data.get("tags")),
            categories=_list(data.get("categories")),
            license=data.get("license"),
            homepage=data.get("homepage"),
            repository=data.get("repository"),
            publisher=Publisher.from_dict(data["publisher"]) if "publisher" in data else None,
            artifact=Artifact.from_dict(data["artifact"]) if "artifact" in data else None,
            mcp=McpConfig.from_dict(data["mcp"]) if "mcp" in data else None,
            hardware=Hardware.from_dict(data["hardware"]) if "hardware" in data else None,
            eurdf=EurdfBinding.from_dict(data.get("eurdf")),
            body_binding=BodyBindingTemplate.from_dict(data.get("bodyBinding")),
            permissions=Permissions.from_dict(data.get("permissions")),
            install=InstallDecl.from_dict(data.get("install")),
            health=HealthDecl.from_dict(data.get("health")),
            claude=ClaudeMcpConfig.from_dict(data.get("claude")),
            compatibility=CompatibilityDecl.from_dict(data.get("compatibility")),
            security=SecurityDecl.from_dict(data.get("security")),
            lifecycle=LifecycleDecl.from_dict(data.get("lifecycle")),
            extra=extra,
        )
