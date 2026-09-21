"""`rosclaw sim` CLI（MH11，0916 优化 §五/§十，ADR-0014）。

SimulationRuntime 的 JSON 投影——Native Agent / 脚本 / operator 的
仿真触达面（ToolGateway 原则下与 MCP sim_* 同一权威，不另造实现）：

    rosclaw sim [--root PATH] capabilities
    rosclaw sim [--root PATH] load <model.xml>
    rosclaw sim [--root PATH] inspect <model_ref>
    rosclaw sim [--root PATH] patch <model_ref> --patches '<json>'|@file
    rosclaw sim [--root PATH] snapshot <model_ref> [--state-ref R]
    rosclaw sim [--root PATH] rollout <model_ref> --controller J|@file
                [--duration-s F | --steps N] [--state-ref R] [--seed N]
                [--predicates J|@file]
    rosclaw sim [--root PATH] observe <model_ref> <state_ref> --channels a,b
    rosclaw sim [--root PATH] audit <model_ref> [--checks a,b] [--trace-ref R]
    rosclaw sim [--root PATH] branch-experiment <model_ref> --branches J|@file
                --controller J|@file [--duration-s F | --steps N] [--serial]
    rosclaw sim [--root PATH] compare <receipt_ref> [<receipt_ref>...]
    rosclaw sim [--root PATH] compile-world --spec J|@file [--name N]
    rosclaw sim [--root PATH] interact <model_ref> <state_ref>
                --interaction J|@file [--payload J|@file]
    rosclaw sim [--root PATH] record-dataset <model_ref> --sequences J|@file
    rosclaw sim [--root PATH] sysid --spec J|@file
    rosclaw sim [--root PATH] render <trace_ref> [--camera N]
                [--width W] [--height H] [--max-frames N]

纪律：stdout 只出成功 JSON；失败 = exit 1 + stderr 结构化错误
（不 traceback 糊屏）。--root 默认 cwd——HarnessBench 独立
workspace 下 SimStore 落在会话目录内（不污染 ~/.rosclaw）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

#: 仿真 CLI 的入口集合（dispatch 命中判定用）。
_SUBCOMMANDS = (
    "capabilities",
    "load",
    "inspect",
    "patch",
    "snapshot",
    "rollout",
    "observe",
    "audit",
    "branch-experiment",
    "compare",
    "compile-world",
    "interact",
    "render",
    "record-dataset",
    "sysid",
)


def _json_arg(value: str) -> Any:  # noqa: ANN401
    """'<json>' 或 @file（长 payload 从文件读，不落超长命令行）。"""
    if value.startswith("@"):
        return json.loads(Path(value[1:]).read_text(encoding="utf-8"))
    return json.loads(value)


def _csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rosclaw sim", description=__doc__)
    parser.add_argument("--root", default=None, help="task root（默认 cwd）")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    sub.add_parser("capabilities")

    load = sub.add_parser("load")
    load.add_argument("asset_ref")

    inspect = sub.add_parser("inspect")
    inspect.add_argument("model_ref")

    patch = sub.add_parser("patch")
    patch.add_argument("model_ref")
    patch.add_argument("--patches", required=True)

    snapshot = sub.add_parser("snapshot")
    snapshot.add_argument("model_ref")
    snapshot.add_argument("--state-ref", default=None)

    rollout = sub.add_parser("rollout")
    rollout.add_argument("model_ref")
    rollout.add_argument("--controller", required=True)
    rollout.add_argument("--duration-s", type=float, default=None)
    rollout.add_argument("--steps", type=int, default=None)
    rollout.add_argument("--state-ref", default=None)
    rollout.add_argument("--seed", type=int, default=0)
    rollout.add_argument("--predicates", default=None)

    observe = sub.add_parser("observe")
    observe.add_argument("model_ref")
    observe.add_argument("state_ref")
    observe.add_argument("--channels", required=True)

    audit = sub.add_parser("audit")
    audit.add_argument("model_ref")
    audit.add_argument("--checks", default=None)
    audit.add_argument("--trace-ref", default=None)

    branch = sub.add_parser("branch-experiment")
    branch.add_argument("model_ref")
    branch.add_argument("--branches", required=True)
    branch.add_argument("--controller", required=True)
    branch.add_argument("--duration-s", type=float, default=None)
    branch.add_argument("--steps", type=int, default=None)
    branch.add_argument("--serial", action="store_true")

    compare = sub.add_parser("compare")
    compare.add_argument("receipt_refs", nargs="+")

    world = sub.add_parser("compile-world")
    world.add_argument("--spec", required=True)
    world.add_argument("--name", default="world")

    interact = sub.add_parser("interact")
    interact.add_argument("model_ref")
    interact.add_argument("state_ref")
    interact.add_argument("--interaction", required=True)
    interact.add_argument("--payload", default=None)

    dataset = sub.add_parser("record-dataset")
    dataset.add_argument("model_ref")
    dataset.add_argument("--sequences", required=True)

    sysid = sub.add_parser("sysid")
    sysid.add_argument("--spec", required=True)

    render = sub.add_parser("render")
    render.add_argument("trace_ref")
    render.add_argument("--camera", default=None)
    render.add_argument("--width", type=int, default=640)
    render.add_argument("--height", type=int, default=480)
    render.add_argument("--max-frames", type=int, default=16)

    return parser


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    """子命令 → SimulationRuntime 调用（唯一实现路径）。"""
    from rosclaw.sim.runtime import SimulationRuntime

    root = Path(args.root) if args.root else Path.cwd()
    runtime = SimulationRuntime(root)
    cmd = args.subcommand

    if cmd == "capabilities":
        return runtime.get_capabilities()
    if cmd == "load":
        return runtime.load_model(args.asset_ref)
    if cmd == "inspect":
        return runtime.inspect_model(args.model_ref)
    if cmd == "patch":
        return runtime.patch_model(args.model_ref, _json_arg(args.patches))
    if cmd == "snapshot":
        return runtime.snapshot(args.model_ref, args.state_ref)
    if cmd == "rollout":
        return runtime.rollout(
            args.model_ref,
            controller=_json_arg(args.controller),
            duration_s=args.duration_s,
            steps=args.steps,
            state_ref=args.state_ref,
            seed=args.seed,
            task_predicates=_json_arg(args.predicates) if args.predicates else None,
        )
    if cmd == "observe":
        return runtime.observe(args.model_ref, args.state_ref, _csv(args.channels) or [])
    if cmd == "audit":
        return runtime.audit(args.model_ref, checks=_csv(args.checks), trace_ref=args.trace_ref)
    if cmd == "branch-experiment":
        return runtime.branch_experiment(
            args.model_ref,
            branches=_json_arg(args.branches),
            controller=_json_arg(args.controller),
            duration_s=args.duration_s,
            steps=args.steps,
            parallel=not args.serial,
        )
    if cmd == "compare":
        return runtime.compare(args.receipt_refs)
    if cmd == "compile-world":
        return runtime.compile_world(_json_arg(args.spec), name=args.name)
    if cmd == "interact":
        return runtime.interact(
            args.model_ref,
            args.state_ref,
            _json_arg(args.interaction),
            _json_arg(args.payload) if args.payload else None,
        )
    if cmd == "record-dataset":
        return runtime.record_dataset(args.model_ref, _json_arg(args.sequences))
    if cmd == "sysid":
        return runtime.sysid(_json_arg(args.spec))
    if cmd == "render":
        return runtime.render(
            args.trace_ref,
            camera=args.camera,
            width=args.width,
            height=args.height,
            max_frames=args.max_frames,
        )
    raise ValueError(f"SIM_CLI_UNKNOWN: {cmd}")


def dispatch_sim_argv(argv: list[str]) -> int | None:
    """`rosclaw sim ...` 分派；未命中返回 None（交下一 dispatcher）。"""
    if not argv or argv[0] != "sim":
        return None
    rest = argv[1:]
    # 判定是否属于本域（跳过 --root 及其值后的首个位置参数必须是
    # 已知子命令，否则交下一 dispatcher）。
    cleaned: list[str] = []
    skip_next = False
    for token in rest:
        if skip_next:
            skip_next = False
            continue
        if token == "--root":
            skip_next = True
            continue
        if token.startswith("--root="):
            continue
        cleaned.append(token)
    if not cleaned or cleaned[0] not in _SUBCOMMANDS:
        if rest and rest[0] in ("-h", "--help"):
            _build_parser().print_help()
            return 0
        return None

    args = _build_parser().parse_args(rest)
    import contextlib

    try:
        # stdout 纯度是 CLI 契约：执行期库打印（scipy 迭代报告/
        # mujoco 警告等）一律导去 stderr，结果 JSON 独占 stdout。
        with contextlib.redirect_stdout(sys.stderr):
            result = _execute(args)
    except (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(
            json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
