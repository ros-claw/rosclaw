"""Install Hardware MCP servers from public git repositories or local paths.

This module implements ``rosclaw mcp install --from-git <url>`` and
``rosclaw mcp install --local-path <path>`` without requiring a private
registry.  It clones/copies the source, installs dependencies, records the
source commit, registers the server, and generates a runnable transport
configuration.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from rosclaw.firstboot.workspace import resolve_home
from rosclaw.mcp.onboarding.installed import InstalledRecord, InstalledRegistry
from rosclaw.mcp.onboarding.runner import write_runner_script
from rosclaw.mcp.onboarding.schema import (
    McpManifest,
)


@dataclass
class SourceInstallResult:
    """Outcome of installing an MCP from git or a local path."""

    success: bool
    server_name: str
    manifest_id: str
    version: str
    source_type: str
    source_url: str
    local_path: Path
    commit: str | None
    manifest_path: Path
    runtime_config_path: Path
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "server_name": self.server_name,
            "manifest_id": self.manifest_id,
            "version": self.version,
            "source_type": self.source_type,
            "source_url": self.source_url,
            "local_path": str(self.local_path),
            "commit": self.commit,
            "manifest_path": str(self.manifest_path),
            "runtime_config_path": str(self.runtime_config_path),
            "errors": list(self.errors),
        }


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _default_name(source: str) -> str:
    """Derive a server name from a URL or path."""
    base = source.rstrip("/").split("/")[-1]
    if base.endswith(".git"):
        base = base[:-4]
    return base or "unknown"


def _run(cmd: list[str] | str, cwd: Path | None = None, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    """Run a shell command and return the completed process."""
    shell = isinstance(cmd, str)
    return subprocess.run(
        cmd,
        shell=shell,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
        executable="/bin/bash" if shell else None,
        timeout=timeout,
    )


def _git_commit(source_dir: Path) -> str | None:
    """Return the current HEAD commit hash for a git repository."""
    proc = _run(["git", "-C", str(source_dir), "rev-parse", "HEAD"], timeout=30)
    if proc.returncode == 0:
        return proc.stdout.strip()
    return None


def _read_manifest(source_dir: Path) -> tuple[dict[str, Any], Path] | None:
    """Read a manifest.yaml or manifest.json from the source directory."""
    for name in ("manifest.yaml", "manifest.yml", "manifest.json"):
        path = source_dir / name
        if path.exists():
            try:
                if path.suffix == ".json":
                    with open(path, encoding="utf-8") as f:
                        return json.load(f), path
                with open(path, encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}, path
            except (json.JSONDecodeError, yaml.YAMLError, OSError):
                continue
    return None


def _infer_manifest(source_dir: Path, name: str) -> dict[str, Any]:
    """Build a minimal manifest from repository clues when none is present."""
    description = ""
    readme = source_dir / "README.md"
    if readme.exists():
        try:
            lines = readme.read_text(encoding="utf-8").splitlines()
            description = " ".join(line.strip() for line in lines[:3] if line.strip())
        except OSError:
            pass

    entrypoint = "mcp_server.py"
    for candidate in ("mcp_server.py", "server.py", "bridge.py"):
        if (source_dir / candidate).exists():
            entrypoint = candidate
            break

    return {
        "$schema": "https://schemas.rosclaw.io/mcp/hardware-manifest.schema.json",
        "schemaVersion": "1.0.0",
        "id": name,
        "name": name,
        "displayName": name,
        "version": "0.0.0",
        "description": description or f"MCP server installed from {source_dir.name}",
        "publisher": {"name": "local", "namespace": "local"},
        "artifact": {"type": "local", "package": str(source_dir), "entrypoint": entrypoint},
        "mcp": {
            "serverName": name,
            "transport": {"type": "stdio", "command": "python", "args": [entrypoint]},
            "capabilities": {"tools": True},
        },
        "hardware": {"type": "unknown"},
        "permissions": {
            "required": [
                {"id": "mcp:tools:read", "level": "safe", "description": "List and call MCP tools"}
            ]
        },
        "health": {
            "checks": [
                {"id": "install_integrity", "category": "install", "required": True},
                {"id": "protocol_stdio", "category": "protocol", "required": True},
            ]
        },
    }


def _normalize_simple_manifest(data: dict[str, Any], source_dir: Path) -> dict[str, Any]:
    """Convert a simple manifest (e.g. librealsense-mcp) to the full schema."""
    if "schemaVersion" in data and "mcp" in data:
        return data

    name = data.get("robot_id") or data.get("name") or source_dir.name
    display_name = data.get("robot_name") or data.get("displayName") or name
    version = str(data.get("version", "1.0.0"))
    description = data.get("description", "")
    author = data.get("author", "unknown")
    mcp_server = data.get("mcp_server", {})
    command = mcp_server.get("command", "python")
    args = mcp_server.get("args", ["mcp_server.py"])
    hardware = data.get("hardware", {})
    files = data.get("files", [])

    entrypoint = args[0] if args else "mcp_server.py"
    if not (source_dir / entrypoint).exists() and files:
        for f in files:
            if "server" in f and (source_dir / f).exists():
                entrypoint = f
                if "bridge" not in entrypoint.lower():
                    break

    return {
        "$schema": "https://schemas.rosclaw.io/mcp/hardware-manifest.schema.json",
        "schemaVersion": "1.0.0",
        "id": name,
        "name": name,
        "displayName": display_name,
        "version": version,
        "description": description,
        "publisher": {"name": author, "namespace": "local"},
        "artifact": {"type": "local", "package": str(source_dir), "entrypoint": entrypoint},
        "mcp": {
            "serverName": name,
            "transport": {"type": "stdio", "command": command, "args": [entrypoint]},
            "capabilities": {"tools": True, "resources": False},
        },
        "hardware": {
            "type": "sensor",
            "vendor": hardware.get("manufacturer"),
            "models": [hardware.get("model")] if hardware.get("model") else [],
            "connection": {"modes": ["usb"], "defaultMode": "usb"},
        },
        "permissions": {
            "required": [
                {"id": "mcp:tools:read", "level": "safe", "description": "List and call MCP tools"}
            ]
        },
        "health": {
            "checks": [
                {"id": "install_integrity", "category": "install", "required": True},
                {"id": "protocol_stdio", "category": "protocol", "required": True},
            ]
        },
    }


def _install_dependencies(
    source_dir: Path,
    python: str | None,
    no_install_deps: bool,
    errors: list[str],
) -> None:
    """Install Python dependencies if requested and available."""
    if no_install_deps:
        return

    py = python or sys.executable
    requirements = source_dir / "requirements.txt"
    if requirements.exists():
        proc = _run(
            [py, "-m", "pip", "install", "-r", str(requirements)],
            cwd=source_dir,
            timeout=300,
        )
        if proc.returncode != 0:
            errors.append(
                f"pip install requirements failed (exit {proc.returncode}): "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )

    pyproject = source_dir / "pyproject.toml"
    if pyproject.exists() and not requirements.exists():
        # Only install in editable mode if the package declares itself.
        try:
            text = pyproject.read_text(encoding="utf-8")
        except OSError:
            text = ""
        if "[project]" in text or "[tool.setuptools]" in text or "[tool.poetry]" in text:
            proc = _run([py, "-m", "pip", "install", "-e", str(source_dir)], timeout=300)
            if proc.returncode != 0:
                errors.append(
                    f"pip install -e failed (exit {proc.returncode}): "
                    f"{proc.stderr.strip() or proc.stdout.strip()}"
                )


def _write_wrapper(source_dir: Path, installed_dir: Path, python: str, entrypoint: str) -> Path:
    """Write a wrapper script that runs the server with the source dir as cwd."""
    wrapper = installed_dir / "run_server.py"
    script = f'''#!/usr/bin/env python3
# Auto-generated by rosclaw mcp. Do not edit manually.
import os
import subprocess
import sys

SOURCE_DIR = {str(source_dir)!r}
PYTHON = {python!r}
ENTRYPOINT = os.path.join(SOURCE_DIR, {entrypoint!r})

env = os.environ.copy()
env["PYTHONPATH"] = SOURCE_DIR + os.pathsep + env.get("PYTHONPATH", "")
os.chdir(SOURCE_DIR)
sys.exit(subprocess.call([PYTHON, ENTRYPOINT] + sys.argv[1:], env=env))
'''
    wrapper.write_text(script, encoding="utf-8")
    wrapper.chmod(0o755)
    return wrapper


def _write_runtime_config(
    manifest: McpManifest,
    home: Path,
    wrapper: Path,
) -> Path:
    """Write the runtime YAML used by ``rosclaw-mcp-run``."""
    runtime_dir = home / "mcp" / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    path = runtime_dir / f"{manifest.server_name}.yaml"
    config = {
        "id": manifest.id,
        "name": manifest.name,
        "version": manifest.version,
        "server_name": manifest.server_name,
        "transport": {
            "type": "stdio",
            "command": str(sys.executable),
            "args": [str(wrapper)],
            "env": {"ROSCLAW_HOME": str(home)},
        },
        "capabilities": manifest.mcp.capabilities.to_dict() if manifest.mcp else {"tools": True},
        "startup": manifest.mcp.startup.to_dict() if manifest.mcp else {"timeoutMs": 5000},
    }
    tmp = path.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    tmp.replace(path)
    return path


def _build_manifest(source_dir: Path, name: str) -> tuple[McpManifest, Path, list[str]]:
    """Load or infer a manifest and return it with its path and warnings."""
    warnings: list[str] = []
    found = _read_manifest(source_dir)
    if found is not None:
        raw, manifest_path = found
        raw = _normalize_simple_manifest(raw, source_dir)
    else:
        manifest_path = source_dir / "manifest.yaml"
        raw = _infer_manifest(source_dir, name)
        warnings.append(
            "No manifest.yaml/manifest.json found; inferring transport from repository contents."
        )

    return McpManifest.from_dict(raw), manifest_path, warnings


def install_from_git(
    url: str,
    server_name: str | None = None,
    home: Path | str | None = None,
    python: str | None = None,
    no_install_deps: bool = False,
) -> SourceInstallResult:
    """Clone a git repository and register it as an installed MCP server."""
    home_path: Path = resolve_home(str(home) if home else None)
    name = server_name or _default_name(url)
    installed_dir = home_path / "mcp" / "installed" / name
    source_dir = installed_dir / "source"

    errors: list[str] = []

    installed_dir.mkdir(parents=True, exist_ok=True)
    if source_dir.exists():
        shutil.rmtree(source_dir)

    proc = _run(["git", "clone", "--depth=1", url, str(source_dir)], timeout=120)
    if proc.returncode != 0:
        errors.append(f"git clone failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}")
        return SourceInstallResult(
            success=False,
            server_name=name,
            manifest_id=name,
            version="unknown",
            source_type="git",
            source_url=url,
            local_path=source_dir,
            commit=None,
            manifest_path=source_dir / "manifest.yaml",
            runtime_config_path=home_path / "mcp" / "runtime" / f"{name}.yaml",
            errors=errors,
        )

    commit = _git_commit(source_dir)
    manifest, manifest_path, warnings = _build_manifest(source_dir, name)
    errors.extend(warnings)

    _install_dependencies(source_dir, python, no_install_deps, errors)

    py = str(Path(python or sys.executable).expanduser().resolve())
    entrypoint = manifest.artifact.entrypoint if (manifest.artifact and manifest.artifact.entrypoint) else "mcp_server.py"
    wrapper = _write_wrapper(source_dir, installed_dir, py, entrypoint)
    runtime_config_path = _write_runtime_config(manifest, home_path, wrapper)
    write_runner_script(home_path / "mcp" / "bin")

    record = InstalledRecord(
        server_name=manifest.server_name,
        manifest_id=manifest.id,
        name=manifest.name,
        version=manifest.version,
        installed_at=_now(),
        artifact_type="git",
        server_dir=str(source_dir),
        runtime_config_path=str(runtime_config_path),
        extra={
            "source_type": "git",
            "source_url": url,
            "repo_commit": commit,
            "manifest_path": str(manifest_path),
            "transport_command": f"{sys.executable} {wrapper}",
        },
    )
    InstalledRegistry(home=home_path).add(record)

    return SourceInstallResult(
        success=len([e for e in errors if "failed" in e.lower()]) == 0,
        server_name=manifest.server_name,
        manifest_id=manifest.id,
        version=manifest.version,
        source_type="git",
        source_url=url,
        local_path=source_dir,
        commit=commit,
        manifest_path=manifest_path,
        runtime_config_path=runtime_config_path,
        errors=errors,
    )


def install_from_local_path(
    path: Path,
    server_name: str | None = None,
    home: Path | str | None = None,
    python: str | None = None,
    no_install_deps: bool = False,
) -> SourceInstallResult:
    """Register a local directory as an installed MCP server without moving it."""
    home_path: Path = resolve_home(str(home) if home else None)
    source_dir = Path(path).resolve()
    if not source_dir.exists():
        return SourceInstallResult(
            success=False,
            server_name=server_name or source_dir.name,
            manifest_id=server_name or source_dir.name,
            version="unknown",
            source_type="local_path",
            source_url=str(source_dir),
            local_path=source_dir,
            commit=None,
            manifest_path=source_dir / "manifest.yaml",
            runtime_config_path=home_path / "mcp" / "runtime" / f"{server_name or source_dir.name}.yaml",
            errors=[f"Local path does not exist: {source_dir}"],
        )

    name = server_name or source_dir.name
    installed_dir = home_path / "mcp" / "installed" / name
    installed_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    commit = _git_commit(source_dir)
    manifest, manifest_path, warnings = _build_manifest(source_dir, name)
    errors.extend(warnings)

    _install_dependencies(source_dir, python, no_install_deps, errors)

    py = str(Path(python or sys.executable).expanduser().resolve())
    entrypoint = manifest.artifact.entrypoint if (manifest.artifact and manifest.artifact.entrypoint) else "mcp_server.py"
    wrapper = _write_wrapper(source_dir, installed_dir, py, entrypoint)
    runtime_config_path = _write_runtime_config(manifest, home_path, wrapper)
    write_runner_script(home_path / "mcp" / "bin")

    record = InstalledRecord(
        server_name=manifest.server_name,
        manifest_id=manifest.id,
        name=manifest.name,
        version=manifest.version,
        installed_at=_now(),
        artifact_type="local_path",
        server_dir=str(source_dir),
        runtime_config_path=str(runtime_config_path),
        extra={
            "source_type": "local_path",
            "source_url": str(source_dir),
            "repo_commit": commit,
            "manifest_path": str(manifest_path),
            "transport_command": f"{sys.executable} {wrapper}",
        },
    )
    InstalledRegistry(home=home_path).add(record)

    return SourceInstallResult(
        success=len([e for e in errors if "failed" in e.lower()]) == 0,
        server_name=manifest.server_name,
        manifest_id=manifest.id,
        version=manifest.version,
        source_type="local_path",
        source_url=str(source_dir),
        local_path=source_dir,
        commit=commit,
        manifest_path=manifest_path,
        runtime_config_path=runtime_config_path,
        errors=errors,
    )


def get_installed_record(server_name: str, home: Path | str | None = None) -> InstalledRecord | None:
    """Fetch a single installed record including source metadata."""
    home_path: Path = resolve_home(str(home) if home else None)
    return InstalledRegistry(home=home_path).get(server_name)
