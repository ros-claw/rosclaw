"""Lightweight CLI handlers for product-facing ROSClaw workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_status_capabilities(args: argparse.Namespace) -> int:
    """Show the canonical product capability boundary."""

    from rosclaw.product.readme import CLAIM_LABELS
    from rosclaw.product.status import iter_matrix_entries, load_product_status

    status = load_product_status()
    if getattr(args, "json", False):
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return 0

    release = status["release"]
    print("=" * 72)
    print("ROSClaw Capability Status")
    print("=" * 72)
    print(f"Release:   {release['version']}")
    print(f"Maturity:  {str(release['maturity']).upper()}")
    print("Source:    rosclaw.product/status.yaml")
    print("-" * 72)
    for reference, entry in iter_matrix_entries(status):
        display = entry["display"]["en"]
        claim = str(entry["claim"])
        label = CLAIM_LABELS.get(claim, {}).get("en", claim)
        evidence_ids = [
            str(item.get("id"))
            for item in entry.get("evidence", [])
            if isinstance(item, dict) and item.get("id")
        ]
        print(f"{display}:")
        print(f"  Status:   {label}")
        print(f"  Evidence: {', '.join(evidence_ids) if evidence_ids else 'none'}")
        if reference.startswith("golden_paths."):
            dimensions = entry.get("dimensions", {})
            print(
                "  Surface:  "
                f"sim={dimensions.get('simulation', 'n/a')}, "
                f"read={dimensions.get('hardware_read', 'n/a')}, "
                f"actuation={dimensions.get('hardware_actuation', 'n/a')}, "
                f"agent={dimensions.get('agent_blackbox', 'n/a')}"
            )
    print("=" * 72)
    return 0


def cmd_demo_list(args: argparse.Namespace) -> int:
    """List official evidence-bearing product demos."""

    from rosclaw.product.demo import list_demos

    demos = [demo.to_dict() for demo in list_demos()]
    if args.json:
        print(
            json.dumps(
                {"schema_version": "rosclaw.demo_catalog.v1", "demos": demos},
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    print("Official ROSClaw Demos")
    for demo in demos:
        print(f"  {demo['id']:<18} {demo['mode']:<12} {demo['description']}")
    return 0


def cmd_demo_run(args: argparse.Namespace) -> int:
    """Run an official demo and print its evidence-bearing receipt."""

    from rosclaw.kernel import ActionState
    from rosclaw.product.demo import DemoConfigurationError, DemoNotFoundError, run_demo
    from rosclaw.product.explain import explain_receipt, format_explanation

    target: tuple[float, float, float] | None = None
    if args.target is not None:
        target = (float(args.target[0]), float(args.target[1]), float(args.target[2]))
    home = Path(args.home).expanduser() if args.home else None
    try:
        receipt, receipt_path = run_demo(
            args.demo_id,
            home=home,
            target=target,
            max_steps=args.steps,
            tolerance_m=args.tolerance,
            seed=args.seed,
            trace_id=args.trace_id,
        )
    except (DemoConfigurationError, DemoNotFoundError) as exc:
        return _print_error(args, str(exc), exit_code=2)
    except Exception as exc:  # noqa: BLE001
        return _print_error(args, f"Demo failed: {exc}", exit_code=1)

    if args.json:
        print(json.dumps(receipt.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(format_explanation(explain_receipt(receipt.to_dict(), receipt_path)))
    return (
        0
        if receipt.final_state in {ActionState.COMPLETED, ActionState.DEGRADED} and receipt.verified
        else 1
    )


def cmd_explain_run(args: argparse.Namespace) -> int:
    """Explain a persisted product run."""

    from rosclaw.product.explain import explain_receipt, format_explanation
    from rosclaw.product.runs import ProductRunStore, RunStoreError

    home = Path(args.home).expanduser() if args.home else None
    try:
        receipt, receipt_path = ProductRunStore(home).load(args.run_reference)
        explanation = explain_receipt(receipt, receipt_path)
    except RunStoreError as exc:
        if args.json:
            print(json.dumps({"status": "NOT_FOUND", "error": str(exc)}, indent=2))
        else:
            print(f"[ROSClaw] {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(explanation, indent=2, ensure_ascii=False))
    else:
        print(format_explanation(explanation))
    return 0


def dispatch_product_argv(argv: list[str]) -> int | None:
    """Fast-path product commands; return ``None`` for the legacy CLI."""

    if argv in (["--version"], ["-V"]):
        from rosclaw import __version__

        print(f"rosclaw {__version__}")
        return 0

    is_product_command = (
        len(argv) >= 2
        and (
            (argv[0] == "demo" and argv[1] in {"list", "run"})
            or (argv[0] == "status" and argv[1] == "capabilities")
        )
    ) or (bool(argv) and argv[0] == "explain")
    if not is_product_command:
        return None

    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "product_handler", None)
    if not callable(handler):
        parser.print_help()
        return 1
    return int(handler(args))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rosclaw")
    subparsers = parser.add_subparsers(dest="command")

    status_parser = subparsers.add_parser("status")
    status_subparsers = status_parser.add_subparsers(dest="status_command")
    status_capabilities_parser = status_subparsers.add_parser("capabilities")
    status_capabilities_parser.add_argument("--json", action="store_true")
    status_capabilities_parser.set_defaults(product_handler=cmd_status_capabilities)

    demo_parser = subparsers.add_parser("demo")
    demo_subparsers = demo_parser.add_subparsers(dest="demo_command")
    demo_list_parser = demo_subparsers.add_parser("list")
    demo_list_parser.add_argument("--json", action="store_true")
    demo_list_parser.set_defaults(product_handler=cmd_demo_list)
    demo_run_parser = demo_subparsers.add_parser("run")
    demo_run_parser.add_argument("demo_id")
    demo_run_parser.add_argument(
        "--target",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
    )
    demo_run_parser.add_argument("--steps", type=int, default=1200)
    demo_run_parser.add_argument("--tolerance", type=float, default=0.008)
    demo_run_parser.add_argument("--seed", type=int, default=0)
    demo_run_parser.add_argument("--trace-id", default=None)
    demo_run_parser.add_argument("--home", default=None)
    demo_run_parser.add_argument("--json", action="store_true")
    demo_run_parser.set_defaults(product_handler=cmd_demo_run)

    explain_parser = subparsers.add_parser("explain")
    explain_parser.add_argument("run_reference", nargs="?", default="latest")
    explain_parser.add_argument("--home", default=None)
    explain_parser.add_argument("--json", action="store_true")
    explain_parser.set_defaults(product_handler=cmd_explain_run)
    return parser


def _print_error(args: argparse.Namespace, message: str, *, exit_code: int) -> int:
    if args.json:
        print(json.dumps({"status": "FAILED", "error": message}, indent=2))
    else:
        print(f"[ROSClaw] {message}", file=sys.stderr)
    return exit_code


__all__ = [
    "cmd_demo_list",
    "cmd_demo_run",
    "cmd_explain_run",
    "cmd_status_capabilities",
    "dispatch_product_argv",
]
