"""rosclaw hub CLI subcommands."""

from __future__ import annotations

import argparse
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, cast

from rosclaw.hub.auth import AuthStore
from rosclaw.hub.cache import HubCache
from rosclaw.hub.client import FakeRegistryClient
from rosclaw.hub.errors import HubError, HubErrorCode
from rosclaw.hub.index import CatalogIndex
from rosclaw.hub.installer import Installer, InstallOptions
from rosclaw.hub.licenses import check_license
from rosclaw.hub.permissions import check_permissions
from rosclaw.hub.publisher import Publisher, PublishOptions
from rosclaw.hub.refs import AssetRef, parse_ref
from rosclaw.hub.schema import dump_manifest_schema, load_manifest
from rosclaw.hub.verifier import verify_asset_dir


def add_hub_subparser(
    subparsers: argparse._SubParsersAction[ArgumentParser],
) -> ArgumentParser:
    """Register the ``rosclaw hub`` subcommand tree."""
    hub_parser = subparsers.add_parser(
        "hub", help="ROSClaw Hub asset discovery, verification, and lifecycle"
    )
    hub_subparsers = hub_parser.add_subparsers(dest="hub_command")

    # validate
    validate_parser = hub_subparsers.add_parser("validate", help="Validate an asset manifest.yaml")
    validate_parser.add_argument("manifest", help="Path to manifest.yaml")
    validate_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # ref
    ref_parser = hub_subparsers.add_parser("ref", help="Asset reference utilities")
    ref_subparsers = ref_parser.add_subparsers(dest="ref_command")
    ref_parse_parser = ref_subparsers.add_parser("parse", help="Parse a rosclaw:// URI")
    ref_parse_parser.add_argument("ref", help="Asset URI")
    ref_parse_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # schema
    schema_parser = hub_subparsers.add_parser("schema", help="Manifest schema utilities")
    schema_subparsers = schema_parser.add_subparsers(dest="schema_command")
    schema_export_parser = schema_subparsers.add_parser(
        "export", help="Export the manifest JSON Schema"
    )
    schema_export_parser.add_argument(
        "--format",
        default="json",
        choices=["json", "yaml"],
        help="Output format (default: json)",
    )
    schema_export_parser.add_argument(
        "--output",
        default=None,
        help="Write schema to file instead of stdout",
    )

    # login
    login_parser = hub_subparsers.add_parser("login", help="Authenticate with a Hub registry")
    login_parser.add_argument("--registry", required=True, help="Registry URL")
    login_parser.add_argument("--token", required=True, help="Access token")
    login_parser.add_argument(
        "--insecure-local",
        action="store_true",
        help="Allow plain HTTP / local-only registry (testing only)",
    )

    # whoami
    hub_subparsers.add_parser("whoami", help="Show the active Hub registry identity")

    # logout
    logout_parser = hub_subparsers.add_parser("logout", help="Forget stored Hub credentials")
    logout_parser.add_argument("--registry", default=None, help="Registry URL")

    # sync
    sync_parser = hub_subparsers.add_parser("sync", help="Sync the local catalog index")
    sync_parser.add_argument("--registry", default=None, help="Registry URL")
    sync_parser.add_argument(
        "--clear", action="store_true", help="Clear the existing index before syncing"
    )

    # search
    search_parser = hub_subparsers.add_parser("search", help="Search the local catalog index")
    search_parser.add_argument("query", nargs="?", default="", help="Search keywords")
    search_parser.add_argument("--type", default=None, help="Filter by asset type")
    search_parser.add_argument("--namespace", default=None, help="Filter by namespace")
    search_parser.add_argument("--official", action="store_true", help="Only official publishers")
    search_parser.add_argument(
        "--license", action="append", default=None, help="Filter by SPDX license (repeatable)"
    )
    search_parser.add_argument("--robot", default=None, help="Filter by robot profile or body kind")
    search_parser.add_argument(
        "--compatible", action="store_true", help="Only assets compatible with this machine"
    )
    search_parser.add_argument("--limit", type=int, default=20, help="Maximum results")
    search_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # verify
    verify_parser = hub_subparsers.add_parser(
        "verify", help="Verify asset integrity (checksums, artifacts, signatures)"
    )
    verify_parser.add_argument("asset_dir", help="Path to asset directory")
    verify_parser.add_argument(
        "--no-signature",
        action="store_true",
        help="Skip signature/certificate checks",
    )
    verify_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # policy
    policy_parser = hub_subparsers.add_parser(
        "policy", help="Check asset against local permission and license policy"
    )
    policy_subparsers = policy_parser.add_subparsers(dest="policy_command")
    policy_check_parser = policy_subparsers.add_parser(
        "check", help="Check permissions and license policy"
    )
    policy_check_parser.add_argument("asset_dir", help="Path to asset directory")
    policy_check_parser.add_argument(
        "--allow-real-robot",
        action="store_true",
        default=None,
        help="Allow assets that require real robot execution (default: allow but flag)",
    )
    policy_check_parser.add_argument(
        "--accept-license",
        action="store_true",
        help="Explicitly accept licenses that require manual acceptance",
    )
    policy_check_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # install
    install_parser = hub_subparsers.add_parser(
        "install",
        help="Install an asset from a local directory or registry reference",
    )
    install_parser.add_argument(
        "asset_dir",
        help="Path to asset directory or rosclaw:// reference",
    )
    install_parser.add_argument("--dry-run", action="store_true", help="Simulate without writing")
    install_parser.add_argument(
        "--yes",
        action="store_true",
        help="Accept license and dangerous permissions automatically",
    )
    install_parser.add_argument(
        "--accept-license", action="store_true", help="Explicitly accept the asset license"
    )
    install_parser.add_argument(
        "--no-mcp-merge",
        action="store_true",
        help="Skip updating .mcp.json",
    )
    install_parser.add_argument(
        "--skip-health",
        action="store_true",
        help="Skip post-install health checks",
    )
    install_parser.add_argument(
        "--no-verify-signature",
        dest="verify_signature",
        action="store_false",
        default=True,
        help="Skip signature/certificate checks",
    )
    install_parser.add_argument(
        "--allow-real-robot",
        action="store_true",
        help="Allow real robot execution",
    )
    install_parser.add_argument(
        "--allow-safety-config-changes",
        action="store_true",
        help="Allow modifications to safety configuration",
    )
    install_parser.add_argument(
        "--allow-network-inbound",
        action="store_true",
        help="Allow non-local inbound network access",
    )
    install_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # uninstall
    uninstall_parser = hub_subparsers.add_parser(
        "uninstall", help="Uninstall an asset by reference"
    )
    uninstall_parser.add_argument("ref", help="Asset URI")
    uninstall_parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    uninstall_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # update
    update_parser = hub_subparsers.add_parser(
        "update", help="Update an installed asset from a local directory"
    )
    update_parser.add_argument("ref", help="Asset URI of the installed asset")
    update_parser.add_argument("asset_dir", help="Path to new asset directory")
    update_parser.add_argument("--dry-run", action="store_true", help="Simulate without writing")
    update_parser.add_argument(
        "--yes",
        action="store_true",
        help="Accept license and dangerous permissions automatically",
    )
    update_parser.add_argument(
        "--accept-license", action="store_true", help="Explicitly accept the asset license"
    )
    update_parser.add_argument(
        "--no-mcp-merge",
        action="store_true",
        help="Skip updating .mcp.json",
    )
    update_parser.add_argument(
        "--skip-health",
        action="store_true",
        help="Skip post-install health checks",
    )
    update_parser.add_argument(
        "--no-verify-signature",
        dest="verify_signature",
        action="store_false",
        default=True,
        help="Skip signature/certificate checks",
    )
    update_parser.add_argument(
        "--allow-real-robot",
        action="store_true",
        help="Allow real robot execution",
    )
    update_parser.add_argument(
        "--allow-safety-config-changes",
        action="store_true",
        help="Allow modifications to safety configuration",
    )
    update_parser.add_argument(
        "--allow-network-inbound",
        action="store_true",
        help="Allow non-local inbound network access",
    )
    update_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # list
    list_parser = hub_subparsers.add_parser("list", help="List installed assets")
    list_parser.add_argument("--installed", action="store_true", help="Only show installed assets")
    list_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # publish
    publish_parser = hub_subparsers.add_parser(
        "publish", help="Prepare and publish a ROSClaw Hub asset"
    )
    publish_parser.add_argument("asset_dir", help="Path to asset directory")
    publish_parser.add_argument(
        "--dry-run", action="store_true", help="Validate and scan without writing"
    )
    publish_parser.add_argument("--private", action="store_true", help="Publish as a private asset")
    publish_parser.add_argument("--public", action="store_true", help="Publish as a public asset")
    publish_parser.add_argument("--sign", action="store_true", help="Create placeholder signature")
    publish_parser.add_argument(
        "--registry", default=None, help="Registry URL (defaults to active profile)"
    )
    publish_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the bundle to this path or directory",
    )
    publish_parser.add_argument("--json", action="store_true", help="Output as JSON")

    return hub_parser


def dispatch_hub_command(args: argparse.Namespace) -> int:
    """Route hub subcommands."""
    command = getattr(args, "hub_command", None)
    if command == "validate":
        return cmd_hub_validate(args)
    if command == "ref":
        ref_command = getattr(args, "ref_command", None)
        if ref_command == "parse":
            return cmd_hub_ref_parse(args)
        print("[ROSClaw] hub ref: no subcommand given. Use: parse")
        return 1
    if command == "schema":
        schema_command = getattr(args, "schema_command", None)
        if schema_command == "export":
            return cmd_hub_schema_export(args)
        print("[ROSClaw] hub schema: no subcommand given. Use: export")
        return 1
    if command == "login":
        return cmd_hub_login(args)
    if command == "whoami":
        return cmd_hub_whoami(args)
    if command == "logout":
        return cmd_hub_logout(args)
    if command == "sync":
        return cmd_hub_sync(args)
    if command == "search":
        return cmd_hub_search(args)
    if command == "verify":
        return cmd_hub_verify(args)
    if command == "policy":
        policy_command = getattr(args, "policy_command", None)
        if policy_command == "check":
            return cmd_hub_policy_check(args)
        print("[ROSClaw] hub policy: no subcommand given. Use: check")
        return 1
    if command == "install":
        return cmd_hub_install(args)
    if command == "uninstall":
        return cmd_hub_uninstall(args)
    if command == "update":
        return cmd_hub_update(args)
    if command == "list":
        return cmd_hub_list(args)
    if command == "publish":
        return cmd_hub_publish(args)
    print(
        "[ROSClaw] hub: no subcommand given. Use: validate, ref, schema, "
        "login, whoami, logout, sync, search, verify, policy, install, "
        "uninstall, update, list, publish"
    )
    return 1


def _ref_to_dict(ref: AssetRef) -> dict[str, Any]:
    """Convert an AssetRef to a JSON-serializable dict."""
    return {
        "type": ref.type,
        "namespace": ref.namespace,
        "name": ref.name,
        "version": ref.version,
        "canonical": str(ref),
    }


def cmd_hub_validate(args: argparse.Namespace) -> int:
    """Validate a manifest file."""
    manifest_path = Path(args.manifest)
    try:
        manifest = load_manifest(manifest_path)
    except HubError as exc:
        if args.json:
            print(
                json.dumps(
                    {"valid": False, "error": exc.message, "code": exc.code.value},
                    indent=2,
                )
            )
        else:
            print(f"[ROSClaw] ❌ Manifest invalid: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 - catch-all for CLI safety
        if args.json:
            print(
                json.dumps(
                    {"valid": False, "error": str(exc), "code": "UNKNOWN"},
                    indent=2,
                )
            )
        else:
            print(f"[ROSClaw] ❌ Manifest validation failed: {exc}")
        return 1

    asset = manifest.asset
    if args.json:
        result = {
            "valid": True,
            "asset": {
                "type": asset.type.value,
                "namespace": asset.namespace,
                "name": asset.name,
                "version": asset.version,
                "title": asset.title,
            },
        }
        print(json.dumps(result, indent=2))
    else:
        print("=" * 60)
        print("Hub Manifest Validation")
        print("=" * 60)
        print("Valid:      ✅ YES")
        print(f"Type:       {asset.type.value}")
        print(f"Namespace:  {asset.namespace}")
        print(f"Name:       {asset.name}")
        print(f"Version:    {asset.version}")
        print(f"Title:      {asset.title}")
        print(f"Publisher:  {manifest.publisher.display_name}")
        print("=" * 60)
    return 0


def cmd_hub_ref_parse(args: argparse.Namespace) -> int:
    """Parse a rosclaw:// URI."""
    try:
        ref = parse_ref(args.ref)
    except HubError as exc:
        if args.json:
            print(
                json.dumps(
                    {"valid": False, "error": exc.message, "code": exc.code.value},
                    indent=2,
                )
            )
        else:
            print(f"[ROSClaw] ❌ Invalid reference: {exc}")
        return 1

    if args.json:
        print(json.dumps(_ref_to_dict(ref), indent=2))
    else:
        print("=" * 60)
        print("Asset Reference")
        print("=" * 60)
        print(f"Type:       {ref.type}")
        print(f"Namespace:  {ref.namespace}")
        print(f"Name:       {ref.name}")
        print(f"Version:    {ref.version or '(unspecified)'}")
        print(f"Canonical:  {ref.canonical()}")
        print("=" * 60)
    return 0


def cmd_hub_schema_export(args: argparse.Namespace) -> int:
    """Export the manifest JSON Schema."""
    fmt = args.format
    try:
        output = dump_manifest_schema(format=fmt)
    except Exception as exc:  # noqa: BLE001
        print(f"[ROSClaw] ❌ Schema export failed: {exc}")
        return 1

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"[ROSClaw] Exported manifest schema to {out_path}")
        return 0

    print(output)
    return 0


def _active_registry_or_fail(store: AuthStore, override: str | None) -> str:
    """Resolve an explicit registry or the active profile."""
    if override:
        return override.rstrip("/")
    profile = store.get_active_profile()
    if not profile:
        print(
            "[ROSClaw] ❌ No active registry. Run `rosclaw hub login --registry <url> --token <token>`"
        )
        raise SystemExit(1)
    return cast(str, profile["registry"])


def cmd_hub_login(args: argparse.Namespace) -> int:
    """Authenticate with a Hub registry."""
    store = AuthStore()
    client = FakeRegistryClient(args.registry, token=args.token)
    try:
        profile = client.whoami()
    except HubError as exc:
        print(f"[ROSClaw] ❌ Login failed: {exc}")
        return 1

    store.login(
        args.registry,
        args.token,
        insecure_local=args.insecure_local,
        set_active=True,
    )
    print("[ROSClaw] ✅ Logged in to Hub")
    print(f"  Registry: {profile['registry']}")
    print(f"  User:     {profile['user']}")
    print(f"  Role:     {profile['role']}")
    return 0


def cmd_hub_whoami(args: argparse.Namespace) -> int:
    """Show the active registry identity."""
    store = AuthStore()
    profile = store.get_active_profile()
    if not profile:
        print("[ROSClaw] ❌ Not logged in to any Hub registry.")
        return 1

    client = FakeRegistryClient(profile["registry"], token=profile["token"])
    try:
        info = client.whoami()
    except HubError as exc:
        print(f"[ROSClaw] ⚠️  Active profile is stored but server rejected it: {exc.message}")
        print(f"  Registry: {profile['registry']}")
        return 1

    print("[ROSClaw] Active Hub identity")
    print(f"  Registry: {info['registry']}")
    print(f"  User:     {info['user']}")
    print(f"  Role:     {info['role']}")
    return 0


def cmd_hub_logout(args: argparse.Namespace) -> int:
    """Forget stored Hub credentials."""
    store = AuthStore()
    registry = args.registry or (store.get_active_profile() or {}).get("registry")
    if not registry:
        print("[ROSClaw] ❌ No active registry to log out from.")
        return 1
    if store.logout(registry):
        print(f"[ROSClaw] ✅ Logged out from {registry}")
        return 0
    print(f"[ROSClaw] ❌ No stored credentials for {registry}")
    return 1


def cmd_hub_sync(args: argparse.Namespace) -> int:
    """Sync the local catalog index."""
    store = AuthStore()
    registry = _active_registry_or_fail(store, args.registry)
    token = store.get_token(registry)
    client = FakeRegistryClient(registry, token=token)

    try:
        entries = client.sync()
    except HubError as exc:
        print(f"[ROSClaw] ❌ Sync failed: {exc}")
        return 1

    cache = HubCache()
    index = CatalogIndex(registry, cache)
    if args.clear:
        index.clear()
    index.index_entries(entries)

    # Cache manifest YAMLs so the resolver can find them for install-by-ref.
    for entry in entries:
        asset = entry.get("asset", {})
        version = asset.get("version")
        if not version:
            continue
        ref = AssetRef(
            type=asset.get("type", ""),
            namespace=asset.get("namespace", ""),
            name=asset.get("name", ""),
            version=version,
        )
        try:
            manifest_bytes = client.fetch_manifest(ref)
            cache.put_manifest(ref, manifest_bytes)
        except HubError:
            # A catalog entry without a reachable manifest is not fatal for sync.
            pass

    print(f"[ROSClaw] ✅ Synced {len(entries)} assets from {registry}")
    print(f"  Indexed: {index.count()}")
    return 0


def _builtin_catalog_entries() -> list[dict[str, Any]]:
    """Return a small offline catalog of official ROSClaw RealSense assets.

    These entries are used when no Hub registry is configured so that
    ``rosclaw hub search realsense`` still offers installable pointers.
    """
    return [
        {
            "asset": {
                "type": "mcp",
                "namespace": "ros-claw",
                "name": "librealsense-mcp",
                "version": "1.0.0",
                "title": "librealsense-mcp",
                "summary": "RealSense RGB-D capture via the pyrealsense2 SDK.",
                "description": (
                    "MCP server that exposes list_devices, start_pipeline, "
                    "capture_aligned_rgbd, and stop_pipeline for Intel RealSense cameras."
                ),
                "tags": ["realsense", "mcp", "rgbd", "d405", "d435i"],
            },
            "publisher": {
                "id": "ros-claw",
                "display_name": "ROSClaw Project",
                "trust_level": "official",
            },
            "visibility": {"scope": "public"},
            "lifecycle": {"status": "stable"},
            "license": {"spdx": "MIT"},
            "compatibility": {
                "hardware": {"required_devices": ["realsense_d405", "realsense_d435i"]},
                "robot": {
                    "eurdf_profiles": ["realsense_d405", "realsense_d435i", "realsense_dual"],
                    "body_kinds": ["perception_only_camera"],
                },
            },
            "manifest_url": "https://github.com/ros-claw/librealsense-mcp",
            "size_bytes": 0,
        },
        {
            "asset": {
                "type": "mcp",
                "namespace": "ros-claw",
                "name": "realsense-ros-mcp",
                "version": "1.0.0",
                "title": "realsense-ros-mcp",
                "summary": "RealSense integration via ROS 2 / realsense2_camera.",
                "description": (
                    "MCP server that exposes RealSense tools through the ROS2 "
                    "realsense2_camera node."
                ),
                "tags": ["realsense", "mcp", "ros2", "rgbd", "d405"],
            },
            "publisher": {
                "id": "ros-claw",
                "display_name": "ROSClaw Project",
                "trust_level": "official",
            },
            "visibility": {"scope": "public"},
            "lifecycle": {"status": "stable"},
            "license": {"spdx": "MIT"},
            "compatibility": {
                "hardware": {"required_devices": ["realsense_d405"]},
                "robot": {
                    "eurdf_profiles": ["realsense_d405", "realsense_dual"],
                    "body_kinds": ["perception_only_camera"],
                },
            },
            "manifest_url": "https://github.com/ros-claw/realsense-ros-mcp",
            "size_bytes": 0,
        },
        {
            "asset": {
                "type": "eurdf",
                "namespace": "ros-claw",
                "name": "realsense_d405",
                "version": "1.0.0",
                "title": "realsense_d405 e-URDF profile",
                "summary": "Perception-only Intel RealSense D405 body profile.",
                "description": (
                    "e-URDF profile for the Intel RealSense D405 depth camera "
                    "with no actuation and RGB-D capabilities."
                ),
                "tags": ["realsense", "d405", "eurdf", "perception-only"],
            },
            "publisher": {
                "id": "ros-claw",
                "display_name": "ROSClaw Project",
                "trust_level": "official",
            },
            "visibility": {"scope": "public"},
            "lifecycle": {"status": "stable"},
            "license": {"spdx": "MIT"},
            "compatibility": {
                "robot": {
                    "eurdf_profiles": ["realsense_d405"],
                    "body_kinds": ["perception_only_camera"],
                },
            },
            "manifest_url": "rosclaw://eurdf/realsense_d405@1.0.0",
            "size_bytes": 0,
        },
    ]


def cmd_hub_search(args: argparse.Namespace) -> int:
    """Search the local catalog index, falling back to a built-in offline catalog."""
    store = AuthStore()
    profile = store.get_active_profile()
    registry: str | None = profile.get("registry") if profile else None

    cache = HubCache()
    if registry:
        index = CatalogIndex(registry, cache)
    else:
        # Offline fallback: seed built-in official pointers once.
        index = CatalogIndex("__builtin__", cache)
        if index.count() == 0:
            index.index_entries(_builtin_catalog_entries())

    if index.count() == 0:
        print("[ROSClaw] ⚠️  Local catalog is empty. Run `rosclaw hub sync` first.")
        return 1

    results = index.search(
        query=args.query or None,
        asset_type=args.type,
        namespace=args.namespace,
        official=args.official,
        licenses=args.license,
        robot=args.robot,
        compatible=args.compatible,
        limit=args.limit,
    )

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return 0

    if not results:
        print(f"[ROSClaw] No results for '{args.query}'")
        return 0

    print(f"[ROSClaw] {len(results)} result(s) for '{args.query}'")
    for entry in results:
        asset = entry.get("asset", {})
        pub = entry.get("publisher", {})
        print(
            f"  {asset.get('type'):15} {pub.get('trust_level', 'unknown'):10} "
            f"rosclaw://{asset.get('type')}/{asset.get('namespace')}/{asset.get('name')}@{asset.get('version')}"
        )
        print(f"    {asset.get('title')}")
        summary = asset.get("summary") or ""
        if summary:
            print(f"    {summary}")
    return 0


def cmd_hub_verify(args: argparse.Namespace) -> int:
    """Verify a local asset directory."""
    asset_dir = Path(args.asset_dir)
    result = verify_asset_dir(asset_dir, require_signature=not args.no_signature)

    if args.json:
        print(
            json.dumps(
                {
                    "ok": result.ok,
                    "errors": result.errors,
                    "warnings": result.warnings,
                },
                indent=2,
            )
        )
        return 0 if result.ok else 1

    if result.ok:
        print("[ROSClaw] ✅ Asset verification passed")
    else:
        print("[ROSClaw] ❌ Asset verification failed")
    for warning in result.warnings:
        print(f"  ⚠️  {warning}")
    for error in result.errors:
        print(f"  ❌ {error}")
    return 0 if result.ok else 1


def cmd_hub_policy_check(args: argparse.Namespace) -> int:
    """Check an asset against permission and license policy."""
    asset_dir = Path(args.asset_dir)
    manifest_path = asset_dir / "manifest.yaml"

    try:
        manifest = load_manifest(manifest_path)
    except HubError as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": exc.message}, indent=2))
        else:
            print(f"[ROSClaw] ❌ Policy check failed: {exc}")
        return 1

    allow_real_robot = args.allow_real_robot if args.allow_real_robot else None
    perm_result = check_permissions(
        manifest,
        allow_real_robot=allow_real_robot,
    )
    license_result = check_license(
        manifest,
        accept_license=args.accept_license,
        asset_dir=asset_dir,
    )

    ok = perm_result.allowed and license_result.accepted

    if args.json:
        print(
            json.dumps(
                {
                    "ok": ok,
                    "permissions": {
                        "allowed": perm_result.allowed,
                        "requires_human_approval": perm_result.requires_human_approval,
                        "dangerous_permissions": perm_result.dangerous_permissions,
                        "issues": perm_result.issues,
                    },
                    "license": {
                        "accepted": license_result.accepted,
                        "requires_acceptance": license_result.requires_acceptance,
                        "issues": license_result.issues,
                    },
                },
                indent=2,
            )
        )
        return 0 if ok else 1

    if ok:
        print("[ROSClaw] ✅ Policy check passed")
    else:
        print("[ROSClaw] ❌ Policy check failed")

    if perm_result.requires_human_approval:
        print("  ⚠️  This asset requires human approval before execution")
    for dangerous in perm_result.dangerous_permissions:
        print(f"  ⚠️  Dangerous permission: {dangerous}")
    for issue in perm_result.issues:
        print(f"  ❌ Permission issue: {issue}")

    if license_result.requires_acceptance:
        print("  ⚠️  License requires explicit acceptance (use --accept-license)")
    for issue in license_result.issues:
        print(f"  ❌ License issue: {issue}")

    return 0 if ok else 1


def _install_options_from_args(args: argparse.Namespace) -> InstallOptions:
    """Build :class:`InstallOptions` from CLI args."""
    return InstallOptions(
        dry_run=args.dry_run,
        accept_license=args.accept_license or args.yes,
        allow_real_robot=True if args.allow_real_robot else None,
        allow_safety_config_changes=args.allow_safety_config_changes,
        allow_network_inbound=args.allow_network_inbound,
        verify_signature=args.verify_signature,
        skip_health=args.skip_health,
        skip_mcp_merge=args.no_mcp_merge,
    )


def cmd_hub_install(args: argparse.Namespace) -> int:
    """Install an asset from a local directory or a registry reference."""
    options = _install_options_from_args(args)
    installer = Installer()
    asset_arg: str = args.asset_dir
    is_ref = asset_arg.startswith("rosclaw://")
    try:
        if is_ref:
            store = AuthStore(home=installer.cache.home)
            profile = store.get_active_profile()
            if not profile:
                raise HubError(
                    code=HubErrorCode.AUTH_REQUIRED,
                    message="No active registry profile. Run `rosclaw hub login` first.",
                )
            registry = cast(str, profile["registry"])
            token = store.get_token(registry)
            client = FakeRegistryClient(registry, token=token)
            result = installer.install_by_ref(asset_arg, options=options, registry_client=client)
        else:
            result = installer.install_local(asset_arg, options=options)
    except HubError as exc:
        if args.json:
            print(
                json.dumps(
                    {"success": False, "error": exc.message, "code": exc.code.value},
                    indent=2,
                )
            )
        else:
            print(f"[ROSClaw] ❌ Install failed: {exc}")
        return 1

    data = {
        "success": result.success,
        "ref": str(result.ref),
        "asset_dir": str(result.asset_dir),
        "lifecycle_status": result.lifecycle_status,
        "health_status": result.health_status,
        "dry_run": result.dry_run,
        "messages": result.messages,
    }
    if result.registry_path:
        data["registry_path"] = str(result.registry_path)
    if result.mcp_server_name:
        data["mcp_server_name"] = result.mcp_server_name

    if args.json:
        print(json.dumps(data, indent=2))
    else:
        label = "Dry-run" if result.dry_run else "Installed"
        print(f"[ROSClaw] ✅ {label}: {result.ref}")
        print(f"  Asset dir: {result.asset_dir}")
        print(f"  Lifecycle: {result.lifecycle_status}")
        print(f"  Health:    {result.health_status}")
        if result.registry_path:
            print(f"  Registry:  {result.registry_path}")
        if result.mcp_server_name:
            print(f"  MCP server: {result.mcp_server_name}")
        for message in result.messages:
            print(f"  ⚠️  {message}")
    return 0


def cmd_hub_uninstall(args: argparse.Namespace) -> int:
    """Uninstall an asset by reference."""
    try:
        ref = parse_ref(args.ref)
    except HubError as exc:
        if args.json:
            print(
                json.dumps(
                    {"success": False, "error": exc.message, "code": exc.code.value},
                    indent=2,
                )
            )
        else:
            print(f"[ROSClaw] ❌ Invalid reference: {exc}")
        return 1

    installer = Installer()
    try:
        removed = installer.uninstall(ref)
    except HubError as exc:
        if args.json:
            print(
                json.dumps(
                    {"success": False, "error": exc.message, "code": exc.code.value},
                    indent=2,
                )
            )
        else:
            print(f"[ROSClaw] ❌ Uninstall failed: {exc}")
        return 1

    if not removed:
        msg = f"Asset is not installed: {ref.canonical()}"
        if args.json:
            print(json.dumps({"success": False, "error": msg}, indent=2))
        else:
            print(f"[ROSClaw] ⚠️  {msg}")
        return 1

    if args.json:
        print(json.dumps({"success": True, "ref": str(ref)}, indent=2))
    else:
        print(f"[ROSClaw] ✅ Uninstalled: {ref}")
    return 0


def cmd_hub_update(args: argparse.Namespace) -> int:
    """Update an installed asset from a local directory."""
    try:
        ref = parse_ref(args.ref)
    except HubError as exc:
        if args.json:
            print(
                json.dumps(
                    {"success": False, "error": exc.message, "code": exc.code.value},
                    indent=2,
                )
            )
        else:
            print(f"[ROSClaw] ❌ Invalid reference: {exc}")
        return 1

    asset_dir = Path(args.asset_dir)
    options = _install_options_from_args(args)
    installer = Installer()
    try:
        result = installer.update(ref, asset_dir, options=options)
    except HubError as exc:
        if args.json:
            print(
                json.dumps(
                    {"success": False, "error": exc.message, "code": exc.code.value},
                    indent=2,
                )
            )
        else:
            print(f"[ROSClaw] ❌ Update failed: {exc}")
        return 1

    data = {
        "success": result.success,
        "ref": str(result.ref),
        "asset_dir": str(result.asset_dir),
        "lifecycle_status": result.lifecycle_status,
        "health_status": result.health_status,
        "dry_run": result.dry_run,
        "messages": result.messages,
    }
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        label = "Dry-run" if result.dry_run else "Updated"
        print(f"[ROSClaw] ✅ {label}: {result.ref}")
        print(f"  Asset dir: {result.asset_dir}")
        print(f"  Lifecycle: {result.lifecycle_status}")
        print(f"  Health:    {result.health_status}")
        for message in result.messages:
            print(f"  ⚠️  {message}")
    return 0


def cmd_hub_list(args: argparse.Namespace) -> int:
    """List installed assets from the lockfile."""
    installer = Installer()
    entries = installer.assets_lock.list_installed()

    data = [
        {
            "ref": entry.ref,
            "asset_dir": entry.asset_dir,
            "lifecycle_status": entry.lifecycle_status,
            "health_status": entry.health_status,
            "installed_at": entry.installed_at,
            "depends_on": entry.depends_on,
        }
        for entry in entries
    ]

    if args.json:
        print(json.dumps(data, indent=2))
        return 0

    if not data:
        print("[ROSClaw] No installed assets.")
        return 0

    print(f"[ROSClaw] {len(data)} installed asset(s)")
    for entry in data:
        print(f"  {entry['ref']}")
        print(f"    lifecycle={entry['lifecycle_status']} health={entry['health_status']}")
        if entry["depends_on"]:
            print(f"    depends_on={', '.join(entry['depends_on'])}")
    return 0


def cmd_hub_publish(args: argparse.Namespace) -> int:
    """Prepare and publish an asset."""
    if args.private and args.public:
        print("[ROSClaw] ❌ Cannot specify both --private and --public")
        return 1

    visibility: str | None = None
    if args.private:
        visibility = "private"
    elif args.public:
        visibility = "public"

    registry: str | None = args.registry
    if not registry and not args.dry_run:
        store = AuthStore()
        profile = store.get_active_profile()
        if profile:
            registry = cast(str, profile["registry"])
        else:
            print(
                "[ROSClaw] ❌ No registry specified and no active profile. "
                "Use --registry or run `rosclaw hub login`."
            )
            return 1

    options = PublishOptions(
        dry_run=args.dry_run,
        visibility=visibility,
        sign=args.sign,
        registry=registry,
        output=args.output,
    )
    publisher = Publisher(options)
    try:
        result = publisher.publish(args.asset_dir)
    except HubError as exc:
        if args.json:
            print(
                json.dumps(
                    {"success": False, "error": exc.message, "code": exc.code.value},
                    indent=2,
                )
            )
        else:
            print(f"[ROSClaw] ❌ Publish failed: {exc}")
        return 1

    data = {
        "success": result.success,
        "ref": str(result.ref),
        "manifest_digest": result.manifest_digest,
        "size_bytes": result.size_bytes,
        "dry_run": result.dry_run,
        "bundle_path": str(result.bundle_path) if result.bundle_path else None,
        "messages": result.messages,
        "warnings": result.warnings,
    }

    if args.json:
        print(json.dumps(data, indent=2))
        return 0

    label = "Dry-run" if result.dry_run else "Published"
    print(f"[ROSClaw] ✅ {label}: {result.ref}")
    print(f"  Manifest digest: {result.manifest_digest}")
    if result.bundle_path:
        print(f"  Bundle path:     {result.bundle_path}")
        print(f"  Size bytes:      {result.size_bytes}")
    for message in result.messages:
        print(f"  ℹ️  {message}")
    for warning in result.warnings:
        print(f"  ⚠️  {warning}")
    return 0
