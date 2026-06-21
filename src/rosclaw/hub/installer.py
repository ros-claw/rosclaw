"""ROSClaw Hub asset installer with transactions and rollback.

The :class:`Installer` coordinates verification, license/permission policy,
dependency resolution, health checks, registry updates, MCP config merge, and
lockfile records.  Every mutating operation acquires the cross-process asset
lock and leaves the system in a consistent state.
"""

from __future__ import annotations

import contextlib
import io
import shutil
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from rosclaw.hub._compat import extractall_tar
from rosclaw.hub.cache import HubCache
from rosclaw.hub.client import FakeRegistryClient
from rosclaw.hub.errors import HubError, HubErrorCode
from rosclaw.hub.health import HealthStatus, run_health_checks
from rosclaw.hub.licenses import check_license
from rosclaw.hub.lifecycle import AssetLifecycleState
from rosclaw.hub.lockfile import AssetsLock, LockEntry, acquire_assets_lock
from rosclaw.hub.mcp_merge import McpMerger
from rosclaw.hub.permissions import check_permissions
from rosclaw.hub.refs import AssetRef, parse_ref
from rosclaw.hub.registry_writer import RegistryWriter
from rosclaw.hub.resolver import Resolver, resolve_dependencies
from rosclaw.hub.schema import AssetManifest, load_manifest, load_manifest_from_bytes
from rosclaw.hub.verifier import verify_asset_dir


@dataclass
class InstallOptions:
    """Options controlling an install transaction."""

    dry_run: bool = False
    accept_license: bool = False
    allow_real_robot: bool | None = None
    allow_safety_config_changes: bool = False
    allow_network_inbound: bool = False
    verify_signature: bool = True
    skip_health: bool = False
    skip_mcp_merge: bool = False
    project_root: Path | None = None


@dataclass
class InstallResult:
    """Outcome of an install or dry-run."""

    success: bool
    ref: AssetRef
    asset_dir: Path
    lifecycle_status: str
    health_status: str
    registry_path: Path | None = None
    mcp_server_name: str | None = None
    dry_run: bool = False
    messages: list[str] = field(default_factory=list)


class Installer:
    """Install, uninstall, and update ROSClaw Hub assets."""

    def __init__(
        self,
        home: str | Path | None = None,
        project_root: str | Path | None = None,
        *,
        cache: HubCache | None = None,
        resolver: Resolver | None = None,
        assets_lock: AssetsLock | None = None,
        registry: RegistryWriter | None = None,
        mcp_merger: McpMerger | None = None,
    ) -> None:
        self.cache = cache or HubCache(home)
        self.resolver = resolver or Resolver(cache=self.cache)
        self.assets_lock = assets_lock or AssetsLock.load(self.cache.home / "assets.lock")
        self.registry = registry or RegistryWriter(cache=self.cache)
        if project_root is not None:
            self.project_root = Path(project_root)
        else:
            self.project_root = Path.cwd()
        self.mcp_merger = mcp_merger
        self._home = home

    def _mcp_merger(self) -> McpMerger:
        if self.mcp_merger is None:
            self.mcp_merger = McpMerger(
                project_root=self.project_root,
                home=self._home,
                cache=self.cache,
            )
        return self.mcp_merger

    def _asset_install_dir(self, ref: AssetRef) -> Path:
        return (
            self.cache.home
            / "hub"
            / "installed"
            / ref.type
            / ref.namespace
            / ref.name
            / (ref.version or "unknown")
        )

    @staticmethod
    def _copy_asset_files(source_dir: Path, target_dir: Path) -> None:
        """Copy asset directory contents to the installation location."""
        if target_dir.exists():
            raise HubError(
                code=HubErrorCode.ASSET_ALREADY_INSTALLED,
                message=f"Asset directory already exists: {target_dir}",
            )
        shutil.copytree(source_dir, target_dir)

    def install_local(
        self,
        asset_dir: str | Path,
        options: InstallOptions | None = None,
    ) -> InstallResult:
        """Install an asset from a local directory.

        The transaction performs verification, policy checks, dependency
        resolution, file copy, registry/MCP updates, health checks, and finally
        records the asset in the lockfile.  If a post-copy step fails, all
        mutations are rolled back.
        """
        options = options or InstallOptions()
        source_dir = Path(asset_dir)
        if not source_dir.is_dir():
            raise HubError(
                code=HubErrorCode.ASSET_NOT_FOUND,
                message=f"Asset directory not found: {source_dir}",
            )

        manifest = load_manifest(source_dir / "manifest.yaml")
        ref = AssetRef(
            type=manifest.asset.type.value,
            namespace=manifest.asset.namespace,
            name=manifest.asset.name,
            version=manifest.asset.version,
        )
        target_dir = self._asset_install_dir(ref)

        messages: list[str] = []

        # Verification.
        verification = verify_asset_dir(
            source_dir,
            require_signature=options.verify_signature,
        )
        if not verification.ok:
            raise HubError(
                code=HubErrorCode.CHECKSUM_MISMATCH,
                message="Verification failed: " + "; ".join(verification.errors),
            )
        if verification.warnings:
            messages.extend(f"Warning: {w}" for w in verification.warnings)

        # License / policy.
        license_result = check_license(
            manifest,
            accept_license=options.accept_license,
            asset_dir=source_dir,
        )
        if not license_result.accepted:
            raise HubError(
                code=HubErrorCode.LICENSE_DENIED,
                message="; ".join(license_result.issues),
            )

        permission_result = check_permissions(
            manifest,
            allow_real_robot=options.allow_real_robot,
            allow_safety_config_changes=options.allow_safety_config_changes,
            allow_network_inbound=options.allow_network_inbound,
        )
        if not permission_result.allowed:
            raise HubError(
                code=HubErrorCode.PERMISSION_DENIED,
                message="; ".join(permission_result.issues),
            )
        if permission_result.requires_human_approval:
            messages.append(
                "Requires human approval for: " + ", ".join(permission_result.dangerous_permissions)
            )

        # Resolve dependencies.
        deps = resolve_dependencies(manifest, self.resolver)
        depends_on = [str(d.ref) for d in deps]

        if options.dry_run:
            return InstallResult(
                success=True,
                ref=ref,
                asset_dir=target_dir,
                lifecycle_status=AssetLifecycleState.INSTALLED.value,
                health_status=HealthStatus.PENDING,
                dry_run=True,
                messages=messages + ["Dry-run; no files were written"],
            )

        with acquire_assets_lock(path=self.cache.assets_lock_path(), timeout=60.0):
            self.assets_lock = AssetsLock.load(self.cache.home / "assets.lock")
            if self.assets_lock.is_installed(ref):
                raise HubError(
                    code=HubErrorCode.ASSET_ALREADY_INSTALLED,
                    message=f"Asset already installed: {ref.canonical()}",
                    suggested_fix="Uninstall first or use `rosclaw hub update`.",
                )

            self._copy_asset_files(source_dir, target_dir)

            registry_path: Path | None = None
            mcp_server_name: str | None = None
            try:
                registry_path = self.registry.add_asset(manifest, target_dir)

                if not options.skip_mcp_merge and self._has_mcp_entrypoint(manifest):
                    mcp_server_name = self._mcp_merger().add_server(manifest, target_dir)

                if options.skip_health:
                    lifecycle_status = AssetLifecycleState.HEALTHY.value
                    health_status = HealthStatus.SKIPPED
                else:
                    health = HealthResultAdapter(
                        run_health_checks(
                            manifest,
                            target_dir,
                            dry_run=False,
                        )
                    )

                    lifecycle_status = (
                        AssetLifecycleState.HEALTHY.value
                        if health.healthy
                        else AssetLifecycleState.UNHEALTHY.value
                    )
                    health_status = (
                        HealthStatus.HEALTHY if health.healthy else HealthStatus.UNHEALTHY
                    )

                entry = LockEntry(
                    ref=str(ref),
                    source=str(source_dir.resolve()),
                    asset_dir=str(target_dir.resolve()),
                    lifecycle_status=lifecycle_status,
                    health_status=health_status,
                    depends_on=depends_on,
                )
                self.assets_lock.add(entry)
                self.assets_lock.save()
                self.cache.set_installed(
                    ref,
                    {
                        "source": entry.source,
                        "asset_dir": entry.asset_dir,
                        "lifecycle_status": entry.lifecycle_status,
                        "health_status": entry.health_status,
                        "depends_on": entry.depends_on,
                    },
                )
            except Exception:
                self._rollback(
                    ref=ref,
                    target_dir=target_dir,
                    registry_path_hint=registry_path,
                    mcp_server_name=mcp_server_name,
                )
                raise

        return InstallResult(
            success=True,
            ref=ref,
            asset_dir=target_dir,
            lifecycle_status=lifecycle_status,
            health_status=health_status,
            registry_path=registry_path,
            mcp_server_name=mcp_server_name,
            dry_run=False,
            messages=messages,
        )

    def install_by_ref(
        self,
        ref: str | AssetRef,
        options: InstallOptions | None = None,
        registry_client: FakeRegistryClient | None = None,
    ) -> InstallResult:
        """Install an asset by ``rosclaw://`` reference from a registry.

        The reference is resolved against locally cached manifests, the manifest
        and bundle are fetched from the registry, the bundle is extracted into a
        staging directory, and then the normal local install path is used.
        """
        options = options or InstallOptions()
        parsed_ref = parse_ref(str(ref)) if isinstance(ref, str) else ref

        resolved = self.resolver.resolve(parsed_ref)
        concrete_ref = resolved.ref

        if registry_client is None:
            from rosclaw.hub.auth import AuthStore

            store = AuthStore(home=self._home)
            profile = store.get_active_profile()
            if not profile:
                raise HubError(
                    code=HubErrorCode.AUTH_REQUIRED,
                    message="No active registry profile. Run `rosclaw hub login` first.",
                )
            registry = cast(str, profile["registry"])
            token = store.get_token(registry)
            registry_client = FakeRegistryClient(registry, token=token)

        # Fetch manifest from registry (sync should have cached it, but be safe).
        manifest_bytes = registry_client.fetch_manifest(concrete_ref)
        load_manifest_from_bytes(manifest_bytes)
        self.cache.put_manifest(concrete_ref, manifest_bytes)

        if options.dry_run:
            return InstallResult(
                success=True,
                ref=concrete_ref,
                asset_dir=self._asset_install_dir(concrete_ref),
                lifecycle_status=AssetLifecycleState.INSTALLED.value,
                health_status=HealthStatus.PENDING,
                dry_run=True,
                messages=[f"Dry-run: would install {concrete_ref.canonical()} from registry"],
            )

        # Fetch bundle and extract to staging.
        bundle_bytes = registry_client.fetch_bundle(concrete_ref)
        staging_dir = self.cache.staging_path(prefix="install-by-ref")
        result: InstallResult | None = None
        try:
            with tarfile.open(fileobj=io.BytesIO(bundle_bytes), mode="r:gz") as tar:
                extractall_tar(tar, staging_dir)

            source_dir = staging_dir
            # Some bundles have a single top-level directory; unwrap it.
            entries = [e for e in staging_dir.iterdir() if e.name not in (".tmp",)]
            if len(entries) == 1 and entries[0].is_dir():
                source_dir = entries[0]

            result = self.install_local(source_dir, options=options)
        except Exception:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            raise
        finally:
            # Leave staging in place on success; installer copied the files.
            if result is not None and staging_dir.exists() and not result.success:
                shutil.rmtree(staging_dir)

        return result

    def uninstall(
        self,
        ref: str | AssetRef,
        options: InstallOptions | None = None,
    ) -> bool:
        """Remove an installed asset.

        Returns:
            True if the asset was installed and removed.
        """
        options = options or InstallOptions()
        parsed_ref = parse_ref(str(ref))

        with acquire_assets_lock(path=self.cache.assets_lock_path(), timeout=60.0):
            self.assets_lock = AssetsLock.load(self.cache.home / "assets.lock")
            entry = self.assets_lock.get(parsed_ref)
            if entry is None:
                return False

            target_dir = Path(entry.asset_dir)

            # Mark removing before destructive operations.
            entry.lifecycle_status = AssetLifecycleState.REMOVING.value
            self.assets_lock.save()

            self.registry.remove_asset(parsed_ref)
            if not options.skip_mcp_merge:
                with contextlib.suppress(HubError):
                    self._mcp_merger().remove_server(parsed_ref)

            if target_dir.exists():
                shutil.rmtree(target_dir)
            self.cache.remove_installed(parsed_ref)

            self.assets_lock.remove(parsed_ref)
            self.assets_lock.save()

        return True

    def update(
        self,
        ref: str | AssetRef,
        asset_dir: str | Path,
        options: InstallOptions | None = None,
    ) -> InstallResult:
        """Replace an installed asset with a new version from *asset_dir*."""
        options = options or InstallOptions()
        parsed_ref = parse_ref(str(ref))
        if not AssetsLock.load(self.cache.home / "assets.lock").is_installed(parsed_ref):
            raise HubError(
                code=HubErrorCode.ASSET_NOT_FOUND,
                message=f"Asset is not installed: {parsed_ref.canonical()}",
            )
        self.uninstall(parsed_ref, options=options)
        return self.install_local(asset_dir, options=options)

    def _has_mcp_entrypoint(self, manifest: AssetManifest) -> bool:
        """Return True if the manifest declares an MCP server entrypoint."""
        entrypoints: dict[str, Any] = manifest.install.get("entrypoints", {})
        mcp_ep = entrypoints.get("mcp", {})
        return bool(mcp_ep.get("command"))

    def _rollback(
        self,
        ref: AssetRef,
        target_dir: Path,
        registry_path_hint: Path | None,
        mcp_server_name: str | None,
    ) -> None:
        """Best-effort rollback of partial install mutations."""
        if target_dir.exists():
            shutil.rmtree(target_dir)
        with contextlib.suppress(Exception):
            self.registry.remove_asset(ref)
        if mcp_server_name is not None:
            with contextlib.suppress(Exception):
                self._mcp_merger().remove_server(ref)
        with contextlib.suppress(Exception):
            self.assets_lock.remove(ref)
            self.assets_lock.save()
        self.cache.remove_installed(ref)


class HealthResultAdapter:
    """Small adapter exposing a ``healthy`` bool over :class:`HealthResult`."""

    def __init__(self, result: Any) -> None:
        self._result = result

    @property
    def healthy(self) -> bool:
        if hasattr(self._result, "healthy"):
            return bool(self._result.healthy)
        if hasattr(self._result, "status"):
            return bool(self._result.status == HealthStatus.HEALTHY)
        return False
