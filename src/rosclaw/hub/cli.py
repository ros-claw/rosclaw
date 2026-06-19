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
from rosclaw.hub.errors import HubError
from rosclaw.hub.index import CatalogIndex
from rosclaw.hub.licenses import check_license
from rosclaw.hub.permissions import check_permissions
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
    validate_parser = hub_subparsers.add_parser(
        "validate", help="Validate an asset manifest.yaml"
    )
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
    print(
        "[ROSClaw] hub: no subcommand given. Use: validate, ref, schema, "
        "login, whoami, logout, sync, search, verify, policy"
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
    print(f"[ROSClaw] ✅ Synced {len(entries)} assets from {registry}")
    print(f"  Indexed: {index.count()}")
    return 0


def cmd_hub_search(args: argparse.Namespace) -> int:
    """Search the local catalog index."""
    store = AuthStore()
    registry = _active_registry_or_fail(store, None)
    cache = HubCache()
    index = CatalogIndex(registry, cache)

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
        summary = asset.get('summary') or ""
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
