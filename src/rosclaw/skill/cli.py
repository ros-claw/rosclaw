"""CLI handlers for the ROSClaw Skill Hub lifecycle."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from rosclaw.firstboot.workspace import resolve_home
from rosclaw.skill.builtins import get_builtin_skill, list_builtin_skills
from rosclaw.skill.eval import evaluate_skill
from rosclaw.skill.mining import mine_skill_candidate
from rosclaw.skill.models import SkillPackage, SkillRef
from rosclaw.skill.package import (
    package_skill,
    prepare_manifest,
    scan_forbidden_content,
    verify_package,
)
from rosclaw.skill.promote import promote_candidate
from rosclaw.skill.registry import SkillLocalRegistry
from rosclaw.skill.rollback import rollback_skill
from rosclaw.skill.upload import upload_skill
from rosclaw.skill.validators import validate_package

# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


def _resolve_skill_dir(name: str, workspace: str | None = None, cwd_fallback: bool = True) -> Path:
    if workspace:
        return Path(workspace).expanduser().resolve() / name
    # Prefer ~/.rosclaw/skills/NAME, otherwise current dir.
    home_skills = Path(resolve_home(None)) / "skills" / name
    if cwd_fallback and (Path.cwd() / name).exists():
        return Path.cwd() / name
    return home_skills


def _copy_template(template_dir: Path, dest: Path, context: dict[str, str]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for src_path in sorted(template_dir.rglob("*")):
        if not src_path.is_file():
            continue
        rel = src_path.relative_to(template_dir)
        dst_path = dest / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        text = src_path.read_text(encoding="utf-8")
        # Simple brace substitution (str.format would be too strict with JSON braces).
        for key, value in context.items():
            text = text.replace(f"{{{key}}}", value)
        dst_path.write_text(text, encoding="utf-8")
    # Create placeholder files for empty dirs.
    for placeholder in ["policies/checkpoints/.gitkeep", "evidence/practice/.gitkeep", "evidence/eval/.gitkeep", "evidence/videos/.gitkeep", "evidence/reports/.gitkeep", "evidence/signatures/.gitkeep", ".rosclaw/package/.gitkeep"]:
        ph = dest / placeholder
        if not ph.exists():
            ph.parent.mkdir(parents=True, exist_ok=True)
            ph.write_text("", encoding="utf-8")


def _init_context(name: str, robot: str, category: str, namespace: str) -> dict[str, str]:
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    return {
        "name": name,
        "display_name": name.replace("_", " ").replace("-", " ").title(),
        "robot": robot,
        "category": category,
        "namespace": namespace,
        "created_at": now.isoformat().replace("+00:00", "Z"),
        "created_at_date": now.strftime("%Y-%m-%d"),
        "description": f"A reusable ROSClaw skill for {name}.",
    }


def cmd_skill_init(args: argparse.Namespace) -> int:
    name = args.name
    robot = args.robot or "unitree_g1"
    category = args.category or "manipulation"
    namespace = args.namespace or "ros-claw"
    output = Path(args.output).expanduser().resolve() if args.output else _resolve_skill_dir(name)
    template = args.template or "default"

    template_dir = Path(__file__).parent / "templates" / template
    if not template_dir.exists():
        print(f"[ROSClaw] Template not found: {template}")
        return 1

    if output.exists() and not args.force:
        print(f"[ROSClaw] Skill directory already exists: {output}")
        print("[ROSClaw] Use --force to overwrite")
        return 1

    if output.exists():
        shutil.rmtree(output)

    context = _init_context(name, robot, category, namespace)
    _copy_template(template_dir, output, context)

    # Run initial validation.
    pkg = SkillPackage(output).try_load()
    report = validate_package(pkg)

    # Register.
    registry = SkillLocalRegistry()
    registry.add(pkg)

    print(f"[ROSClaw] Created skill package: {output}")
    if report.warnings:
        for w in report.warnings:
            print(f"[ROSClaw] Warning: {w}")
    return 0 if report.ok else 1


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------


def _load_skill_dir_arg(args: argparse.Namespace) -> Path:
    if getattr(args, "skill_dir", None):
        path = Path(args.skill_dir).expanduser()
        if path.exists() or path.is_absolute() or "/" in args.skill_dir or "\\" in args.skill_dir:
            return path.resolve()
        # Treat as skill name.
        return _resolve_skill_dir(args.skill_dir, workspace=getattr(args, "workspace", None))
    if getattr(args, "name", None):
        return _resolve_skill_dir(args.name, workspace=getattr(args, "workspace", None))
    raise ValueError("No skill directory or name provided")


def cmd_skill_validate(args: argparse.Namespace) -> int:
    skill_dir = _load_skill_dir_arg(args)
    if not skill_dir.exists():
        print(f"[ROSClaw] Skill not found: {skill_dir}")
        return 1
    pkg = SkillPackage(skill_dir).try_load()
    report = validate_package(pkg)

    if args.json:
        print(json.dumps({
            "ok": report.ok,
            "errors": report.errors,
            "warnings": report.warnings,
            "checks": report.checks,
        }, indent=2, ensure_ascii=False))
    else:
        print(f"[ROSClaw] Validating {skill_dir.name}")
        for check, ok in report.checks.items():
            print(f"  {'✓' if ok else '✗'} {check}")
        for e in report.errors:
            print(f"  ✗ {e}")
        for w in report.warnings:
            print(f"  ! {w}")
        print(f"[ROSClaw] Result: {'PASS' if report.ok else 'FAIL'}")
    return 0 if report.ok else 1


# ---------------------------------------------------------------------------
# Mine
# ---------------------------------------------------------------------------


def _resolve_output_or_name(value: str | None) -> Path:
    if not value:
        return Path(resolve_home(None)) / "skills"
    path = Path(value).expanduser()
    if path.exists() or path.is_absolute() or "/" in value or "\\" in value:
        return path.resolve()
    return _resolve_skill_dir(value)


def cmd_skill_mine(args: argparse.Namespace) -> int:
    source_dir = Path(args.source).expanduser().resolve()
    output = _resolve_output_or_name(args.output) if args.output else _resolve_skill_dir(args.task)
    if not output.exists():
        print(f"[ROSClaw] Skill output directory does not exist: {output}")
        return 1
    pkg = SkillPackage(output).try_load()
    report = mine_skill_candidate(pkg, source_dir, candidate_id=args.candidate)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"[ROSClaw] Mined candidate {report.candidate_id}")
        print(f"  source episodes: {len(report.source_episodes)}")
        print(f"  score: {report.score}")
        print(f"  generated: {', '.join(report.generated_files)}")
    return 0


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


def cmd_skill_eval(args: argparse.Namespace) -> int:
    skill_dir = _load_skill_dir_arg(args)
    if not skill_dir.exists():
        print(f"[ROSClaw] Skill not found: {skill_dir}")
        return 1
    pkg = SkillPackage(skill_dir).try_load()
    report = evaluate_skill(pkg, candidate_id=args.candidate, mode=args.mode, save_evidence=args.save_evidence)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"[ROSClaw] Eval {report.skill}@{report.candidate_id or report.version}")
        for check, ok in report.checks.items():
            print(f"  {'✓' if ok else '✗'} {check}")
        print(f"  metrics: {json.dumps(report.metrics, ensure_ascii=False)}")
        print(f"[ROSClaw] Decision: {report.decision.upper()}")
        if report.artifacts:
            print(f"  report: {report.artifacts.get('report')}")
    return 0 if report.decision == "pass" else 1


# ---------------------------------------------------------------------------
# Promote
# ---------------------------------------------------------------------------


def cmd_skill_promote(args: argparse.Namespace) -> int:
    ref = SkillRef(args.skill_ref)
    skill_dir = _resolve_skill_dir(ref.name, workspace=getattr(args, "workspace", None))
    if not skill_dir.exists():
        print(f"[ROSClaw] Skill not found: {skill_dir}")
        return 1
    pkg = SkillPackage(skill_dir).try_load()
    candidate_id = ref.candidate_id or (pkg.skill.metadata.candidate_id if pkg.skill else None)
    if not candidate_id:
        print("[ROSClaw] No candidate specified")
        return 1
    try:
        result = promote_candidate(
            pkg,
            candidate_id,
            to_version=args.to_version,
            stage=args.stage,
            require_eval_pass=args.require_eval_pass,
        )
    except ValueError as exc:
        print(f"[ROSClaw] Promotion blocked: {exc}")
        return 1

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"[ROSClaw] Promoted {ref.name}@{candidate_id} to v{result['version']} ({result['stage']})")
        print(f"  package_hash: {result['package_hash']}")
    return 0


# ---------------------------------------------------------------------------
# Package
# ---------------------------------------------------------------------------


def cmd_skill_package(args: argparse.Namespace) -> int:
    skill_dir = _load_skill_dir_arg(args)
    if not skill_dir.exists():
        print(f"[ROSClaw] Skill not found: {skill_dir}")
        return 1
    pkg = SkillPackage(skill_dir).try_load()

    # Pre-package forbidden content scan.
    secrets, paths = scan_forbidden_content(skill_dir)
    if secrets:
        print("[ROSClaw] Secret scan failed:")
        for s in secrets:
            print(f"  {s}")
        return 1
    if paths:
        print("[ROSClaw] Absolute path warnings:")
        for p in paths:
            print(f"  {p}")

    archive = package_skill(
        pkg,
        output_dir=Path(args.output),
        include_evidence=args.include_evidence,
        format=args.format,
    )

    if args.json:
        print(json.dumps({"archive": str(archive), "manifest": prepare_manifest(pkg)}, indent=2, ensure_ascii=False))
    else:
        print(f"[ROSClaw] Packaged: {archive}")
    return 0


def cmd_skill_verify_package(args: argparse.Namespace) -> int:
    result = verify_package(Path(args.archive).expanduser().resolve())
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"[ROSClaw] Verify package: {'PASS' if result['ok'] else 'FAIL'}")
        for e in result["errors"]:
            print(f"  ✗ {e}")
        for w in result["warnings"]:
            print(f"  ! {w}")
    return 0 if result["ok"] else 1


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def cmd_skill_upload(args: argparse.Namespace) -> int:
    skill_dir = _load_skill_dir_arg(args)
    if not skill_dir.exists():
        print(f"[ROSClaw] Skill not found: {skill_dir}")
        return 1
    pkg = SkillPackage(skill_dir).try_load()

    try:
        result = upload_skill(
            pkg,
            visibility=args.visibility,
            hub_base_url=args.hub_base_url,
            api_key_env=args.api_key_env,
            dry_run=args.dry_run,
            force=args.force,
        )
    except RuntimeError as exc:
        print(f"[ROSClaw] Upload failed: {exc}")
        return 1

    if args.json:
        # Mask API key in payload if present.
        payload = result.get("payload", {})
        print(json.dumps({
            "ok": result["ok"],
            "dry_run": result["dry_run"],
            "payload": payload,
        }, indent=2, ensure_ascii=False))
    else:
        print(f"[ROSClaw] Upload: {'DRY-RUN' if result['dry_run'] else 'OK'}")
        print(f"  skill: {result['payload']['name']}")
        print(f"  version: {result['payload']['version']}")
        print(f"  visibility: {args.visibility}")
    return 0


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


def cmd_skill_rollback(args: argparse.Namespace) -> int:
    skill_dir = _load_skill_dir_arg(args)
    if not skill_dir.exists():
        print(f"[ROSClaw] Skill not found: {skill_dir}")
        return 1
    pkg = SkillPackage(skill_dir).try_load()
    try:
        result = rollback_skill(pkg, to_version=args.to, reason=args.reason or "")
    except ValueError as exc:
        print(f"[ROSClaw] Rollback failed: {exc}")
        return 1

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"[ROSClaw] Rolled back to v{result['to_version']}")
        print(f"  evidence: {result['evidence']}")
    return 0


def cmd_skill_search(args: argparse.Namespace) -> int:
    """List builtin skills and local skill-hub packages."""
    builtins = list_builtin_skills()
    registry = SkillLocalRegistry()
    local = registry.list_skills()
    if args.json:
        print(json.dumps({"builtin": builtins, "local": local}, indent=2, ensure_ascii=False))
        return 0
    print("[ROSClaw] Builtin skills")
    for s in builtins:
        print(f"  {s['name']:<30} {s.get('display_name', '')}")
    print("[ROSClaw] Local skill-hub packages")
    for s in local:
        print(f"  {s.get('name', 'unknown')}")
    return 0


def cmd_skill_install(args: argparse.Namespace) -> int:
    """Install a builtin skill reference into the local registry.

    For builtin skills this is effectively a no-op registration; the skill
    remains in-package and is executed from ``rosclaw.skill.builtins``.
    """
    name = args.name
    entry = get_builtin_skill(name)
    if entry is None:
        print(f"[ROSClaw] Builtin skill not found: {name}")
        print("[ROSClaw] Run `rosclaw skill search` to list available skills")
        return 1

    registry = SkillLocalRegistry()
    data = {
        "local_path": str(Path(__file__).parent / "builtins" / name),
        "current_version": entry.version,
        "current_stage": "installable",
        "last_eval_report": None,
        "builtin": True,
    }
    registry._data["skills"][name] = data
    registry._save()
    print(f"[ROSClaw] Installed builtin skill: {name}@{entry.version}")
    return 0


def cmd_skill_inspect(args: argparse.Namespace) -> int:
    """Show details for a builtin or local skill."""
    name = args.name
    entry = get_builtin_skill(name)
    if entry is not None:
        info = {
            "name": entry.name,
            "description": entry.description,
            "version": entry.version,
            "skill_type": entry.skill_type,
            "requirements": entry.requirements,
            "metadata": entry.metadata,
            "builtin": True,
        }
    else:
        registry = SkillLocalRegistry()
        local = {s.get("name"): s for s in registry.list_skills()}
        if name not in local:
            print(f"[ROSClaw] Skill not found: {name}")
            return 1
        info = {"name": name, "local": local[name]}
    if args.json:
        print(json.dumps(info, indent=2, ensure_ascii=False, default=str))
    else:
        print(json.dumps(info, indent=2, ensure_ascii=False, default=str))
    return 0


# ---------------------------------------------------------------------------
# Argument helpers
# ---------------------------------------------------------------------------


def add_skill_hub_parsers(skill_subparsers: Any) -> None:
    search_parser = skill_subparsers.add_parser("search", help="Search builtin and local skills")
    search_parser.add_argument("--json", action="store_true", help="Output as JSON")
    search_parser.set_defaults(func=cmd_skill_search)

    install_parser = skill_subparsers.add_parser("install", help="Install a builtin skill reference")
    install_parser.add_argument("name", help="Skill name")
    install_parser.add_argument("--json", action="store_true", help="Output as JSON")
    install_parser.set_defaults(func=cmd_skill_install)

    inspect_parser = skill_subparsers.add_parser("inspect", help="Inspect a skill")
    inspect_parser.add_argument("name", help="Skill name")
    inspect_parser.add_argument("--json", action="store_true", help="Output as JSON")
    inspect_parser.set_defaults(func=cmd_skill_inspect)

    init_parser = skill_subparsers.add_parser("init", help="Create a local skill package skeleton")
    init_parser.add_argument("name", help="Skill name")
    init_parser.add_argument("--robot", default=None, help="Default robot")
    init_parser.add_argument("--category", default=None, help="Skill category")
    init_parser.add_argument("--namespace", default=None, help="Skill namespace")
    init_parser.add_argument("--template", default=None, help="Template name")
    init_parser.add_argument("--output", default=None, help="Output directory")
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing")
    init_parser.set_defaults(func=cmd_skill_init)

    validate_parser = skill_subparsers.add_parser("validate", help="Validate skill package")
    validate_parser.add_argument("skill_dir", nargs="?", help="Skill directory")
    validate_parser.add_argument("--name", default=None, help="Skill name (used to resolve dir)")
    validate_parser.add_argument("--workspace", default=None, help="Workspace root")
    validate_parser.add_argument("--json", action="store_true", help="Output JSON")
    validate_parser.set_defaults(func=cmd_skill_validate)

    mine_parser = skill_subparsers.add_parser("mine", help="Mine skill candidate from practice episodes")
    mine_parser.add_argument("--from", dest="source", required=True, help="Practice episodes directory")
    mine_parser.add_argument("--task", required=True, help="Task name")
    mine_parser.add_argument("--robot", default=None, help="Robot filter")
    mine_parser.add_argument("--output", default=None, help="Skill output directory")
    mine_parser.add_argument("--candidate", default=None, help="Candidate ID")
    mine_parser.add_argument("--json", action="store_true", help="Output JSON")
    mine_parser.set_defaults(func=cmd_skill_mine)

    eval_parser = skill_subparsers.add_parser("eval", help="Evaluate skill candidate")
    eval_parser.add_argument("skill_dir", nargs="?", help="Skill directory")
    eval_parser.add_argument("--name", default=None, help="Skill name")
    eval_parser.add_argument("--candidate", default=None, help="Candidate ID")
    eval_parser.add_argument("--mode", default="replay", choices=["replay", "sandbox"], help="Eval mode")
    eval_parser.add_argument("--save-evidence", action="store_true", default=True, help="Write eval report")
    eval_parser.add_argument("--json", action="store_true", help="Output JSON")
    eval_parser.set_defaults(func=cmd_skill_eval)

    promote_parser = skill_subparsers.add_parser("promote", help="Promote candidate to version")
    promote_parser.add_argument("skill_ref", help="Skill ref: name@candidate_id")
    promote_parser.add_argument("--to-version", required=True, help="Target version")
    promote_parser.add_argument("--stage", default="validated", help="Target stage")
    promote_parser.add_argument("--require-eval-pass", action="store_true", default=True, help="Require eval pass")
    promote_parser.add_argument("--workspace", default=None, help="Workspace root")
    promote_parser.add_argument("--json", action="store_true", help="Output JSON")
    promote_parser.set_defaults(func=cmd_skill_promote)

    package_parser = skill_subparsers.add_parser("package", help="Package skill into distributable archive")
    package_parser.add_argument("skill_dir", nargs="?", help="Skill directory")
    package_parser.add_argument("--name", default=None, help="Skill name")
    package_parser.add_argument("--output", default="dist", help="Output directory")
    package_parser.add_argument("--format", default="tar.gz", choices=["tar.gz"], help="Archive format")
    package_parser.add_argument("--include-evidence", default="summary", choices=["none", "summary", "full"], help="Evidence inclusion")
    package_parser.add_argument("--workspace", default=None, help="Workspace root")
    package_parser.add_argument("--json", action="store_true", help="Output JSON")
    package_parser.set_defaults(func=cmd_skill_package)

    verify_pkg_parser = skill_subparsers.add_parser("verify-package", help="Verify packaged archive")
    verify_pkg_parser.add_argument("archive", help="Archive path")
    verify_pkg_parser.add_argument("--json", action="store_true", help="Output JSON")
    verify_pkg_parser.set_defaults(func=cmd_skill_verify_package)

    upload_parser = skill_subparsers.add_parser("upload", help="Upload skill metadata to ROSClaw Hub")
    upload_parser.add_argument("skill_dir", nargs="?", help="Skill directory")
    upload_parser.add_argument("--name", default=None, help="Skill name")
    upload_parser.add_argument("--visibility", default="private", choices=["public", "private", "org", "unlisted"], help="Visibility")
    upload_parser.add_argument("--hub-base-url", default="https://www.rosclaw.io", help="Hub base URL")
    upload_parser.add_argument("--api-key-env", default="ROSCLAW_ADMIN_API_KEY", help="API key env var")
    upload_parser.add_argument("--dry-run", action="store_true", help="Dry run")
    upload_parser.add_argument("--force", action="store_true", help="Force update on conflict")
    upload_parser.add_argument("--workspace", default=None, help="Workspace root")
    upload_parser.add_argument("--json", action="store_true", help="Output JSON")
    upload_parser.set_defaults(func=cmd_skill_upload)
