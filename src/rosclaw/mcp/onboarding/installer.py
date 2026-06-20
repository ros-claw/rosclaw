"""Hardware MCP staged installer.

Orchestrates alias resolution, version solving, preflight, artifact
installation, runner registration, body binding, Claude ``.mcp.json`` merge,
and local state updates. Every mutating operation is tracked by a
``RollbackContext`` so a failure restores the previous state.
"""

from __future__ import annotations

import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from rosclaw.firstboot.workspace import resolve_home
from rosclaw.mcp.onboarding.binding import BindingResult, BodyBindingManager
from rosclaw.mcp.onboarding.claude_merge import ClaudeMcpMerge, ClaudeMergeResult
from rosclaw.mcp.onboarding.errors import (
    InstallationError,
    OnboardingError,
    PermissionDeniedError,
    PreflightError,
    RollbackError,
)
from rosclaw.mcp.onboarding.hub_client import HubClient
from rosclaw.mcp.onboarding.installed import InstalledRecord, InstalledRegistry
from rosclaw.mcp.onboarding.lockfile import LockedPackage, Lockfile
from rosclaw.mcp.onboarding.permissions import PermissionState, PermissionStore
from rosclaw.mcp.onboarding.preflight import PreflightRunner
from rosclaw.mcp.onboarding.resolver import AliasResolver, SolvedVersion, VersionSolver
from rosclaw.mcp.onboarding.rollback import RollbackContext
from rosclaw.mcp.onboarding.runner import ensure_runner
from rosclaw.mcp.onboarding.schema import Artifact, McpManifest


@dataclass
class InstallPlan:
    """Describes what an installation would do without doing it."""

    manifest: McpManifest
    solved: SolvedVersion
    installer_type: str
    install_command: str | None
    body_patch: dict[str, Any]
    permission_state: PermissionState
    claude_action: str = "skip"

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest.to_dict(),
            "solved": {
                "manifest_id": self.solved.manifest_id,
                "version": self.solved.version,
                "source": self.solved.source,
            },
            "installer_type": self.installer_type,
            "install_command": self.install_command,
            "body_patch": self.body_patch,
            "permission_state": self.permission_state.to_dict(),
            "claude_action": self.claude_action,
        }


@dataclass
class InstallResult:
    """Outcome of an installation attempt."""

    success: bool
    server_name: str
    manifest_id: str
    version: str
    runtime_config_path: Path | None = None
    runner_script_path: Path | None = None
    installed_record: InstalledRecord | None = None
    binding_result: BindingResult | None = None
    claude_result: ClaudeMergeResult | None = None
    permission_state: PermissionState | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "server_name": self.server_name,
            "manifest_id": self.manifest_id,
            "version": self.version,
            "runtime_config_path": str(self.runtime_config_path) if self.runtime_config_path else None,
            "runner_script_path": str(self.runner_script_path) if self.runner_script_path else None,
            "installed_record": self.installed_record.to_dict() if self.installed_record else None,
            "binding_result": self.binding_result.to_dict() if self.binding_result else None,
            "claude_result": self.claude_result.to_dict() if self.claude_result else None,
            "permission_state": self.permission_state.to_dict() if self.permission_state else None,
            "errors": list(self.errors),
        }


class McpInstaller(Protocol):
    """Protocol for an artifact installer."""

    def install(
        self,
        manifest: McpManifest,
        artifact: Artifact,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        ...


class PythonPackageInstaller:
    """Install a Python package artifact with pip."""

    def install(
        self,
        manifest: McpManifest,
        artifact: Artifact,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        package = artifact.package or manifest.name
        version = artifact.version or manifest.version
        cmd = artifact.install or f"{sys.executable} -m pip install {package}=={version}"

        if dry_run:
            return {
                "server_dir": None,
                "command": cmd,
                "dry_run": True,
            }

        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                executable="/bin/bash",
                check=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise InstallationError(f"Failed to run pip install: {exc}") from exc

        if proc.returncode != 0:
            raise InstallationError(
                f"pip install failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
            )

        server_dir = self._find_package_dir(package)
        return {
            "server_dir": str(server_dir) if server_dir else None,
            "command": cmd,
            "dry_run": False,
        }

    def _find_package_dir(self, package: str) -> Path | None:
        """Locate the installed package directory via pip show."""
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pip", "show", package],
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception:  # noqa: BLE001
            return None
        if proc.returncode != 0:
            return None
        location: str | None = None
        for line in proc.stdout.splitlines():
            if line.startswith("Location:"):
                location = line.split(":", 1)[1].strip()
                break
        if not location:
            return None
        return Path(location) / package.replace("-", "_")


class DockerInstaller:
    """Docker/OCI artifact installer (P1 placeholder)."""

    def install(
        self,
        manifest: McpManifest,
        artifact: Artifact,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        if dry_run:
            return {
                "server_dir": None,
                "command": f"docker pull {artifact.image}",
                "dry_run": True,
                "note": "Docker installer not yet implemented",
            }
        raise InstallationError(
            "Docker/OCI artifact installation is not yet implemented"
        )


class RemoteMcpInstaller:
    """Remote URL artifact installer (P1 placeholder)."""

    def install(
        self,
        manifest: McpManifest,
        artifact: Artifact,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        if dry_run:
            return {
                "server_dir": None,
                "command": f"fetch {artifact.url}",
                "dry_run": True,
                "note": "Remote MCP installer not yet implemented",
            }
        raise InstallationError(
            "Remote URL artifact installation is not yet implemented"
        )


def _select_installer(artifact: Artifact) -> McpInstaller:
    """Return the installer appropriate for the artifact type."""
    atype = (artifact.type or "pypi").lower()
    if atype in {"python", "pypi", "pip"}:
        return PythonPackageInstaller()
    if atype in {"docker", "oci", "container"}:
        return DockerInstaller()
    if atype in {"remote", "url", "npm"}:
        return RemoteMcpInstaller()
    raise InstallationError(f"Unsupported artifact type: {artifact.type}")


class InstallEngine:
    """High-level orchestrator for Hardware MCP installation."""

    def __init__(
        self,
        home: Path | str | None = None,
        project_root: Path | str | None = None,
        hub: HubClient | None = None,
        alias_resolver: AliasResolver | None = None,
        version_solver: VersionSolver | None = None,
        preflight_runner: PreflightRunner | None = None,
        permission_store: PermissionStore | None = None,
        registry: InstalledRegistry | None = None,
        lockfile: Lockfile | None = None,
        body_binding: BodyBindingManager | None = None,
        claude_merge: ClaudeMcpMerge | None = None,
    ) -> None:
        self.home = resolve_home(str(home) if home else None)
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.hub = hub or HubClient(home=self.home)
        self.alias_resolver = alias_resolver or AliasResolver(hub=self.hub, home=self.home)
        self.version_solver = version_solver or VersionSolver(hub=self.hub)
        self.preflight_runner = preflight_runner or PreflightRunner()
        self.permission_store = permission_store or PermissionStore(home=self.home)
        self.registry = registry or InstalledRegistry(home=self.home)
        self.lockfile = lockfile or Lockfile(home=self.home)
        self.body_binding = body_binding or BodyBindingManager(workspace=self.home)
        self.claude_merge = claude_merge or ClaudeMcpMerge(project_root=self.project_root)

    def plan(self, alias: str, version: str | None = None) -> InstallPlan:
        """Return the installation plan for ``alias`` without mutating state."""
        manifest_id = self.alias_resolver.resolve_or_canonical(alias)
        solved = self.version_solver.solve(manifest_id, explicit_version=version)
        manifest = solved.manifest or self.hub.fetch_manifest(manifest_id, solved.version)
        artifact = manifest.artifact or Artifact(type="pypi")
        installer = _select_installer(artifact)
        install_result = installer.install(manifest, artifact, dry_run=True)

        body_patch: dict[str, Any] = {}
        claude_action = "skip"
        if manifest.body_binding:
            from rosclaw.mcp.onboarding.binding import _build_body_patch
            body_patch = _build_body_patch(manifest.body_binding)
        if manifest.claude and manifest.claude.mcp_json:
            claude_action = "merge"

        permissions = manifest.permissions or None
        permission_state = (
            self.permission_store.compute_effective(manifest.server_name, permissions, allow_dangerous=False)
            if permissions
            else PermissionState()
        )

        return InstallPlan(
            manifest=manifest,
            solved=solved,
            installer_type=artifact.type or "pypi",
            install_command=install_result.get("command"),
            body_patch=body_patch,
            permission_state=permission_state,
            claude_action=claude_action,
        )

    def install(
        self,
        alias: str,
        version: str | None = None,
        dry_run: bool = False,
        allow_dangerous: bool = False,
        conflict: str = "abort",
        offline: bool = False,
        skip_body: bool = False,
        skip_claude: bool = False,
    ) -> InstallResult:
        """Install a Hardware MCP server.

        Args:
            alias: Short name, alias, or canonical manifest ID.
            version: Optional exact version to install.
            dry_run: If True, do not mutate the filesystem.
            allow_dangerous: Whether to auto-grant dangerous permissions.
            conflict: Strategy for unmanaged ``.mcp.json`` collisions.
            offline: Prefer cache and built-in registry.
            skip_body: Skip body.yaml binding.
            skip_claude: Skip ``.mcp.json`` merge.

        Returns:
            ``InstallResult`` describing the outcome.
        """
        if offline:
            self.hub.offline = True

        manifest_id = self.alias_resolver.resolve_or_canonical(alias)
        solved = self.version_solver.solve(manifest_id, explicit_version=version)
        manifest = solved.manifest or self.hub.fetch_manifest(manifest_id, solved.version)
        if manifest is None or manifest.mcp is None:
            raise InstallationError(f"Manifest {manifest_id} has no MCP configuration")

        server_name = manifest.server_name
        artifact = manifest.artifact or Artifact(type="pypi")
        installer = _select_installer(artifact)
        runtime_dir = self.home / "mcp" / "runtime"
        runtime_config_path = runtime_dir / f"{server_name}.yaml"
        mcp_json_path = self.claude_merge.mcp_json_path
        body_yaml_path = self.body_binding.resolver.body_yaml_path

        result = InstallResult(
            success=False,
            server_name=server_name,
            manifest_id=manifest_id,
            version=solved.version,
        )

        # Permission effective state and forbidden check before preflight.
        permissions = manifest.permissions
        permission_state: PermissionState = PermissionState()
        if permissions:
            permission_state = self.permission_store.compute_effective(
                server_name, permissions, allow_dangerous=allow_dangerous
            )
            forbidden = [p.id for p in permissions.required if p.level == "forbidden_by_default"]
            if forbidden:
                raise PermissionDeniedError(
                    f"Manifest requires forbidden permissions: {', '.join(forbidden)}"
                )
            result.permission_state = permission_state

        # Preflight checks (dry-run reports success without executing).
        try:
            self.preflight_runner.run(manifest, dry_run=dry_run)
        except PreflightError as exc:
            result.errors.append(str(exc))
            return result

        if dry_run:
            # Simulate the remaining steps.
            plan = self.plan(alias, version=version)
            result.success = True
            result.runtime_config_path = runtime_config_path
            result.runner_script_path = self.home / "mcp" / "bin" / "rosclaw-mcp-run"
            result.binding_result = BindingResult(
                binding_key=manifest.body_binding.binding_key if manifest.body_binding else server_name,
                body_yaml_path=body_yaml_path,
                patched_paths=list(plan.body_patch.keys()),
            )
            if manifest.claude and not skip_claude:
                result.claude_result = ClaudeMergeResult(
                    path=mcp_json_path,
                    server_name=manifest.claude_server_name,
                    action="dry-run:merge",
                )
            result.installed_record = InstalledRecord(
                server_name=server_name,
                manifest_id=manifest_id,
                name=manifest.name,
                version=solved.version,
                installed_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                artifact_type=artifact.type or "pypi",
                server_dir="",
                runtime_config_path=str(result.runtime_config_path),
                body_binding_key=manifest.body_binding.binding_key if manifest.body_binding else None,
                status="planned",
            )
            return result

        # Staged rollback context.
        staging_dir = self.home / "mcp" / ".staging" / f"{server_name}-{solved.version}-{uuid.uuid4().hex[:8]}"
        rollback = RollbackContext(staging_dir)

        try:
            # Backup files we are about to mutate.
            for path in (
                self.registry.path,
                self.lockfile.path,
                self.permission_store.path,
                body_yaml_path,
                mcp_json_path,
                runtime_config_path,
                self.home / "mcp" / "bin" / "rosclaw-mcp-run",
            ):
                if path.exists():
                    rollback.backup(path)

            # Persist effective permissions now that backups are in place.
            if permissions:
                self.permission_store.apply_effective(
                    server_name, permissions, allow_dangerous=allow_dangerous
                )

            # Artifact installation.
            install_info = installer.install(manifest, artifact, dry_run=False)
            server_dir = install_info.get("server_dir") or str(self.home / "mcp" / "servers" / server_name)

            # Runtime runner.
            runtime_path, runner_path = ensure_runner(manifest, self.home)
            result.runtime_config_path = runtime_path
            result.runner_script_path = runner_path

            # Body binding.
            binding_result: BindingResult | None = None
            if not skip_body and manifest.body_binding is not None:
                binding_result = self.body_binding.apply_binding(manifest, dry_run=False)
                result.binding_result = binding_result

            # Claude .mcp.json merge.
            claude_result: ClaudeMergeResult | None = None
            if not skip_claude and manifest.claude and manifest.claude.mcp_json:
                claude_server_name = manifest.claude_server_name
                claude_result = self.claude_merge.merge(
                    server_name=claude_server_name,
                    manifest_id=manifest_id,
                    version=solved.version,
                    mcp_json_fragment=manifest.claude.mcp_json,
                    conflict=conflict,
                    dry_run=False,
                )
                result.claude_result = claude_result

            # Update lockfile.
            self.lockfile.set(
                LockedPackage(
                    manifest_id=manifest_id,
                    version=solved.version,
                    name=manifest.name,
                    server_name=server_name,
                    locked_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                )
            )

            # Update installed registry.
            record = InstalledRecord(
                server_name=server_name,
                manifest_id=manifest_id,
                name=manifest.name,
                version=solved.version,
                installed_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                artifact_type=artifact.type or "pypi",
                server_dir=server_dir,
                runtime_config_path=str(runtime_path),
                body_binding_key=binding_result.binding_key if binding_result else None,
                eurdf_profile=binding_result.eurdf_profile if binding_result else None,
                status="installed",
            )
            self.registry.add(record)
            result.installed_record = record

            # Success: discard backups.
            rollback.commit()
            result.success = True
            return result

        except OnboardingError:
            _rollback_safe(rollback)
            raise
        except Exception as exc:  # noqa: BLE001
            _rollback_safe(rollback)
            raise InstallationError(f"Installation failed: {exc}") from exc


def _rollback_safe(rollback: RollbackContext) -> None:
    """Attempt rollback and raise RollbackError on failure."""
    try:
        rollback.rollback()
    except Exception as exc:  # noqa: BLE001
        raise RollbackError(f"Rollback failed: {exc}") from exc
