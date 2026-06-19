"""Project and runtime detection helpers for agent onboarding."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProjectProfile:
    """Detected project context used to personalize generated files."""

    project_root: Path
    profile_path: Path | None
    robot_id: str | None
    runtime_profile: dict[str, Any]
    has_pyproject: bool
    has_rosclaw_src: bool
    default_transport: str


ROSCLAW_MARKER_FILES = [
    "pyproject.toml",
    "src/rosclaw",
    "rosclaw",
    ".rosclaw",
]


def find_project_root(start: Path | str | None = None, max_depth: int = 10) -> Path:
    """Walk upward looking for a ROSClaw project root.

    A directory is considered a project root if it contains a pyproject.toml
    with a project name of "rosclaw" or any of the ROSCLAW_MARKER_FILES.
    """
    start = Path.cwd() if start is None else Path(start).resolve()

    current = start
    for _ in range(max_depth):
        if current == current.parent:
            break
        if _looks_like_project_root(current):
            return current
        current = current.parent

    # Fallback to the start directory so init can still scaffold a new project.
    return start


def _looks_like_project_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    pyproject = path / "pyproject.toml"
    if pyproject.exists():
        try:
            text = pyproject.read_text(encoding="utf-8")
            if 'name = "rosclaw"' in text or "name='rosclaw'" in text:
                return True
        except OSError:
            pass
    return any((path / marker).exists() for marker in ROSCLAW_MARKER_FILES)


def detect_runtime_profile(project_root: Path, explicit_profile: Path | str | None) -> Path | None:
    """Return the runtime profile YAML path, preferring explicit input."""
    if explicit_profile is not None:
        p = Path(explicit_profile)
        if p.is_absolute():
            return p if p.exists() else None
        return project_root / p if (project_root / p).exists() else None

    candidates = [
        project_root / ".rosclaw" / "profiles" / "default.yaml",
        project_root / ".rosclaw" / "profiles" / "default.yml",
        project_root / "profiles" / "default.yaml",
        project_root / "profiles" / "default.yml",
        project_root / "runtime.yaml",
        project_root / "runtime.yml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_runtime_profile(profile_path: Path | None) -> dict[str, Any]:
    """Load a YAML runtime profile or return an empty dict."""
    if profile_path is None:
        return {}
    try:
        with profile_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, yaml.YAMLError):
        return {}


def detect_robot_id(project_root: Path, runtime_profile: dict[str, Any], explicit_robot: str | None) -> str | None:
    """Choose a robot_id from explicit input, profile, or filesystem heuristics."""
    if explicit_robot:
        return explicit_robot
    profile_robot = runtime_profile.get("robot", {}).get("id")
    if isinstance(profile_robot, str):
        return profile_robot
    runtime_robot = runtime_profile.get("robot_id")
    if isinstance(runtime_robot, str):
        return runtime_robot

    # Try to infer from a robots/ or urdf/ directory.
    for subdir in ("robots", "urdf", "eurdf", "models"):
        robots_dir = project_root / subdir
        if robots_dir.is_dir():
            entries = [p for p in robots_dir.iterdir() if p.is_dir() or p.suffix in (".urdf", ".xacro", ".yaml", ".xml")]
            if entries:
                return entries[0].stem
    return None


def detect_transport(runtime_profile: dict[str, Any], explicit: str | None) -> str:
    """Return the MCP transport to advertise in generated files."""
    if explicit:
        return explicit
    mcp = runtime_profile.get("mcp", {})
    transport = mcp.get("transport") or mcp.get("default_transport")
    if isinstance(transport, str) and transport in ("stdio", "http", "sse"):
        return transport
    return "stdio"


def detect_mcp_port(runtime_profile: dict[str, Any], explicit: int | None) -> int:
    """Return the MCP server port for HTTP/SSE transport."""
    if explicit is not None:
        return explicit
    mcp = runtime_profile.get("mcp", {})
    port = mcp.get("port") or mcp.get("http_port") or mcp.get("sse_port")
    if isinstance(port, int):
        return port
    if isinstance(port, str) and port.isdigit():
        return int(port)
    return 9090


def detect_mcp_host(runtime_profile: dict[str, Any], explicit: str | None) -> str:
    """Return the MCP server host for HTTP/SSE transport."""
    if explicit:
        return explicit
    mcp = runtime_profile.get("mcp", {})
    host = mcp.get("host") or mcp.get("http_host") or mcp.get("sse_host")
    if isinstance(host, str):
        return host
    return "127.0.0.1"


def get_default_runtime_config(project_root: Path, robot_id: str | None) -> dict[str, Any]:
    """Build a minimal runtime config dict from detected values."""
    return {
        "project_root": str(project_root),
        "robot_id": robot_id,
        "workspace": {
            "home": os.environ.get("ROSCLAW_HOME", str(project_root / ".rosclaw")),
        },
    }


def list_available_robots(project_root: Path) -> list[str]:
    """Return a best-effort list of robot ids found in the project."""
    robots: list[str] = []
    for subdir in ("robots", "urdf", "eurdf", "models"):
        robots_dir = project_root / subdir
        if not robots_dir.is_dir():
            continue
        for entry in sorted(robots_dir.iterdir()):
            if entry.is_dir():
                robots.append(entry.name)
            elif entry.suffix in (".urdf", ".xacro", ".yaml", ".xml"):
                robots.append(entry.stem)
    return robots


def build_project_profile(
    project_root: Path | str | None = None,
    profile: Path | str | None = None,
    robot: str | None = None,
    transport: str | None = None,
    host: str | None = None,
    port: int | None = None,
) -> ProjectProfile:
    """Assemble a full project profile from explicit inputs and heuristics."""
    root = find_project_root(project_root)
    profile_path = detect_runtime_profile(root, profile)
    runtime_profile = load_runtime_profile(profile_path)
    robot_id = detect_robot_id(root, runtime_profile, robot)
    default_transport = detect_transport(runtime_profile, transport)
    if default_transport != "stdio":
        runtime_profile.setdefault("mcp", {})
        runtime_profile["mcp"]["host"] = detect_mcp_host(runtime_profile, host)
        runtime_profile["mcp"]["port"] = detect_mcp_port(runtime_profile, port)
        runtime_profile["mcp"]["transport"] = default_transport

    return ProjectProfile(
        project_root=root,
        profile_path=profile_path,
        robot_id=robot_id,
        runtime_profile=runtime_profile,
        has_pyproject=(root / "pyproject.toml").exists(),
        has_rosclaw_src=(root / "src" / "rosclaw").is_dir() or (root / "rosclaw").is_dir(),
        default_transport=default_transport,
    )
