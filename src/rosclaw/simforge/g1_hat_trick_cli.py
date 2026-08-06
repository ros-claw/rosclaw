"""Product CLI for GoalForge Hat Trick evidence and visualization."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def dispatch_hat_trick_argv(argv: list[str]) -> int | None:
    if len(argv) >= 3 and argv[:2] == ["goalforge", "coupled-showcase"]:
        return _dispatch_coupled_showcase_argv(argv)
    if len(argv) >= 3 and argv[:2] == ["goalforge", "coupled-relay"]:
        return _dispatch_coupled_relay_argv(argv)
    if len(argv) >= 3 and argv[:2] == ["goalforge", "relay"]:
        return _dispatch_relay_argv(argv)
    if len(argv) < 3 or argv[:2] != ["goalforge", "hat-trick"]:
        return None
    parser = argparse.ArgumentParser(prog="rosclaw goalforge hat-trick")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="execute three strictly replayed MuJoCo shots")
    run.add_argument("--asset-root", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--source-checkout", type=Path, default=Path.cwd())
    export = commands.add_parser("export", help="render a passing Hat Trick report")
    export.add_argument("evidence", type=Path)
    export.add_argument("--asset-root", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--source-checkout", type=Path, default=Path.cwd())
    export.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(argv[2:])
    if args.command == "run":
        from rosclaw.simforge.g1_hat_trick import run_goalforge_hat_trick

        result = run_goalforge_hat_trick(
            asset_root=args.asset_root,
            output_dir=args.output_dir,
            source_checkout=args.source_checkout,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    from rosclaw.simforge.g1_hat_trick_video import render_goalforge_hat_trick_video

    result = render_goalforge_hat_trick_video(
        evidence_path=args.evidence,
        asset_root=args.asset_root,
        output_path=args.output,
        source_checkout=args.source_checkout,
        fps=args.fps,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _dispatch_relay_argv(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rosclaw goalforge relay")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run",
        help="execute a strictly replayed G1 pass-to-high-shot relay",
    )
    run.add_argument("--asset-root", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--source-checkout", type=Path, default=Path.cwd())
    export = commands.add_parser(
        "export",
        help="render a passing G1 relay evidence report",
    )
    export.add_argument("evidence", type=Path)
    export.add_argument("--asset-root", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--source-checkout", type=Path, default=Path.cwd())
    export.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(argv[2:])
    if args.command == "run":
        from rosclaw.simforge.g1_two_player_relay import run_g1_two_player_relay

        result = run_g1_two_player_relay(
            asset_root=args.asset_root,
            output_dir=args.output_dir,
            source_checkout=args.source_checkout,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    from rosclaw.simforge.g1_two_player_relay_video import (
        render_g1_two_player_relay_video,
    )

    result = render_g1_two_player_relay_video(
        evidence_path=args.evidence,
        asset_root=args.asset_root,
        output_path=args.output,
        source_checkout=args.source_checkout,
        fps=args.fps,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _dispatch_coupled_relay_argv(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rosclaw goalforge coupled-relay")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run",
        help="execute a strictly replayed two-G1 relay in one MuJoCo world",
    )
    run.add_argument("--asset-root", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--source-checkout", type=Path, default=Path.cwd())
    export = commands.add_parser(
        "export",
        help="render passing coupled two-G1 evidence",
    )
    export.add_argument("evidence", type=Path)
    export.add_argument("--asset-root", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--source-checkout", type=Path, default=Path.cwd())
    export.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(argv[2:])
    if args.command == "run":
        from rosclaw.simforge.g1_coupled_relay import run_g1_coupled_relay

        result = run_g1_coupled_relay(
            asset_root=args.asset_root,
            output_dir=args.output_dir,
            source_checkout=args.source_checkout,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    from rosclaw.simforge.g1_coupled_relay_video import (
        render_g1_coupled_relay_video,
    )

    result = render_g1_coupled_relay_video(
        evidence_path=args.evidence,
        asset_root=args.asset_root,
        output_path=args.output,
        source_checkout=args.source_checkout,
        fps=args.fps,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _dispatch_coupled_showcase_argv(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rosclaw goalforge coupled-showcase")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run",
        help="execute five strict two-G1 physics challenges",
    )
    run.add_argument("--asset-root", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--source-checkout", type=Path, default=Path.cwd())
    export = commands.add_parser(
        "export",
        help="render a passing five-challenge showcase",
    )
    export.add_argument("evidence", type=Path)
    export.add_argument("--asset-root", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--source-checkout", type=Path, default=Path.cwd())
    export.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(argv[2:])
    if args.command == "run":
        from rosclaw.simforge.g1_coupled_relay_showcase import (
            run_g1_coupled_showcase,
        )

        result = run_g1_coupled_showcase(
            asset_root=args.asset_root,
            output_dir=args.output_dir,
            source_checkout=args.source_checkout,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    # MuJoCo selects its OpenGL backend during import, so headless CLI export
    # must declare EGL before importing the renderer dependency graph.
    os.environ.setdefault("MUJOCO_GL", "egl")
    from rosclaw.simforge.g1_coupled_showcase_video import (
        render_g1_coupled_showcase_video,
    )

    result = render_g1_coupled_showcase_video(
        evidence_path=args.evidence,
        asset_root=args.asset_root,
        output_path=args.output,
        source_checkout=args.source_checkout,
        fps=args.fps,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


__all__ = ["dispatch_hat_trick_argv"]
