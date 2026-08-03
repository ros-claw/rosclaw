"""rosclaw-agentd CLI: start/status/doctor + `rosclaw agent` + `rosclaw chat`.

The service runs foreground (uvicorn) or in-process for `rosclaw chat`.
A single-instance lock file prevents two agentd processes on one home.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from rosclaw.agentd.config import load_agent_config
from rosclaw.agentd.onboarding import PROVIDER_CHOICES, configure_model, doctor
from rosclaw.agentd.service import AgentService, create_app

LOCK_NAME = "agentd/agentd.lock"


def _home(args: argparse.Namespace) -> Path:
    return Path(
        getattr(args, "home", None) or os.environ.get("ROSCLAW_HOME", Path.home() / ".rosclaw")
    )


def _acquire_lock(home: Path) -> Path:
    lock = home / LOCK_NAME
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        try:
            pid = int(lock.read_text().strip())
            os.kill(pid, 0)
        except (ValueError, ProcessLookupError, PermissionError):
            lock.unlink(missing_ok=True)
        else:
            raise SystemExit(
                f"another rosclaw-agentd is already running (pid {pid}); stop it or remove {lock}"
            )
    lock.write_text(str(os.getpid()))
    return lock


def _release_lock(lock: Path) -> None:
    try:
        if lock.exists() and lock.read_text().strip() == str(os.getpid()):
            lock.unlink()
    except OSError:
        pass


def cmd_start(args: argparse.Namespace) -> int:
    import uvicorn

    home = _home(args)
    config = load_agent_config(home / "config.yaml")
    if not config.profiles:
        print(
            "未配置模型。先运行 `rosclaw agent init`，或 `rosclaw agent doctor` 查看缺口。",
            file=sys.stderr,
        )
        return 2
    lock = _acquire_lock(home)
    service = AgentService(config, home)
    app = create_app(service)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        asyncio.run(service.close())
        _release_lock(lock)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    home = _home(args)
    config = load_agent_config(home / "config.yaml")
    lock = home / LOCK_NAME
    running = False
    if lock.exists():
        try:
            os.kill(int(lock.read_text().strip()), 0)
            running = True
        except (ValueError, ProcessLookupError, PermissionError):
            running = False
    out = {
        "running": running,
        "agent_enabled": config.enabled,
        "profiles": [p.name for p in config.profiles],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def _worker_registry(home: Path):
    from rosclaw.agentd.mission import MissionStore
    from rosclaw.agentd.workers import WorkerRegistry

    store = MissionStore(home / "agentd" / "missions.db")
    registry = WorkerRegistry(store.connection)
    # Idempotent: ensures built-in WorkerPacks are visible even before the
    # first agentd service run on this home.
    registry.register_builtins(actor_id="user:local:cli")
    from rosclaw.agentd.workers.packs import ALL_PACKS, card_for_pack

    for pack in ALL_PACKS:
        registry.register(card_for_pack(pack), actor_id="user:local:cli")
    return store, registry


def cmd_worker_list(args: argparse.Namespace) -> int:
    store, registry = _worker_registry(_home(args))
    workers = []
    for card in registry.list(status=getattr(args, "status", None)):
        workers.append(
            {
                "worker_id": card.worker_id,
                "kind": card.kind.value,
                "status": registry.status_of(card.worker_id),
                "trust": card.trust.initial_level,
                "capabilities": [c.name for c in card.capabilities],
                "adapter": card.adapter_type,
            }
        )
    print(json.dumps(workers, ensure_ascii=False, indent=2))
    store.close()
    return 0


def cmd_worker_catalog(args: argparse.Namespace) -> int:
    store, registry = _worker_registry(_home(args))
    catalog = [
        {
            "worker_id": c.worker_id,
            "display_name": c.display_name,
            "trust": c.trust.initial_level,
            "installed": registry.status_of(c.worker_id) is not None,
            "capabilities": [cap.name for cap in c.capabilities],
        }
        for c in registry.catalog()
    ]
    print(json.dumps(catalog, ensure_ascii=False, indent=2))
    store.close()
    return 0


def cmd_worker_inspect(args: argparse.Namespace) -> int:
    store, registry = _worker_registry(_home(args))
    card = registry.get(args.worker_id)
    if card is None:
        print(f"worker {args.worker_id} 未注册", file=sys.stderr)
        store.close()
        return 1
    out = card.model_dump(mode="json")
    out["registry_status"] = registry.status_of(card.worker_id)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    store.close()
    return 0


def cmd_worker_set_status(args: argparse.Namespace) -> int:
    store, registry = _worker_registry(_home(args))
    target = "ENABLED" if args.worker_command == "enable" else "DISABLED"
    try:
        registry.set_status(
            args.worker_id, target, actor_id="user:local:cli", reason=args.reason or ""
        )
    except Exception as exc:  # noqa: BLE001
        print(f"操作失败：{exc}", file=sys.stderr)
        store.close()
        return 1
    print(f"{args.worker_id} -> {target}")
    store.close()
    return 0


def cmd_worker_probe(args: argparse.Namespace) -> int:
    """探活一个外部 pack（二进制存在性 + 最小版本）。"""
    from rosclaw.agentd.service import AgentService
    from rosclaw.agentd.workers.packs import ALL_PACKS

    pack = next((p for p in ALL_PACKS if p.worker_id == args.worker_id), None)
    if pack is None:
        card = None
        store, registry = _worker_registry(_home(args))
        card = registry.get(args.worker_id)
        store.close()
        if card is None:
            print(f"未知 worker {args.worker_id}（不在 packs 也不在 registry）", file=sys.stderr)
            return 1
        print("内置 worker，无需外部二进制探活。")
        return 0
    ready, detail = AgentService._probe_pack_sync(
        pack.executable, pack.min_version, pack.install_hint
    )
    print(
        json.dumps(
            {"worker_id": pack.worker_id, "ready": ready, "detail": detail},
            ensure_ascii=False,
            indent=2,
        )
    )
    if not ready:
        store, registry = _worker_registry(_home(args))
        registry.set_status(pack.worker_id, "DISABLED", actor_id="user:local:cli", reason=detail)
        store.close()
        return 1
    store, registry = _worker_registry(_home(args))
    if registry.status_of(pack.worker_id) == "DISABLED":
        registry.set_status(pack.worker_id, "ENABLED", actor_id="user:local:cli", reason="probe ok")
    store.close()
    return 0


def cmd_bench_run(args: argparse.Namespace) -> int:
    from rosclaw.agentd.bench.harness import BenchmarkRunner, aggregate, default_scenarios

    out_dir = Path(args.out)
    seeds = [int(s) for s in args.seeds.split(",")]
    groups = args.groups.split(",")

    def home_factory(seed: int) -> Path:
        home = out_dir / "homes" / f"run_{seed}"
        home.mkdir(parents=True, exist_ok=True)
        return home

    runner = BenchmarkRunner(home_factory, reporter_dir=out_dir / "reports")

    async def go() -> dict:
        results = await runner.run_matrix(default_scenarios(), seeds=seeds, groups=groups)
        return aggregate(results)

    report = asyncio.run(go())
    (out_dir / "aggregate.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    failing = [k for k, v in report["families"].items() if v["unsupported_claim_rate"] > 0]
    return 1 if failing else 0


def cmd_learning_list(args: argparse.Namespace) -> int:
    from rosclaw.agentd.learning.pipeline import LearningPipeline
    from rosclaw.agentd.mission import MissionStore

    store = MissionStore(_home(args) / "agentd" / "missions.db")
    pipe = LearningPipeline(store.connection, actor_id="user:local:cli")
    rows = pipe.list(status=getattr(args, "status", None))
    print(
        json.dumps(
            [
                {
                    "candidate_id": r["candidate_id"],
                    "kind": r["kind"],
                    "title": r["title"],
                    "evidence_class": r["evidence_class"],
                    "status": r["status"],
                }
                for r in rows
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
    store.close()
    return 0


def cmd_learning_extract(args: argparse.Namespace) -> int:
    from rosclaw.agentd.learning.pipeline import LearningPipeline
    from rosclaw.agentd.mission import MissionStore

    store = MissionStore(_home(args) / "agentd" / "missions.db")
    pipe = LearningPipeline(store.connection, actor_id="user:local:cli")
    created = pipe.extract_from_mission(args.mission_id)
    print(f"created {len(created)} candidates: {created}")
    store.close()
    return 0


def cmd_learning_promote(args: argparse.Namespace) -> int:
    from rosclaw.agentd.learning.pipeline import LearningPipeline, PromotionGateError
    from rosclaw.agentd.mission import MissionStore

    store = MissionStore(_home(args) / "agentd" / "missions.db")
    pipe = LearningPipeline(store.connection, actor_id="user:local:cli")
    try:
        pipe.promote(
            args.candidate_id,
            principal=args.principal,
            evaluation_ref=args.evaluation_ref,
        )
    except PromotionGateError as exc:
        print(f"晋升被拒（Darwin 门）：{exc}", file=sys.stderr)
        store.close()
        return 1
    print(f"{args.candidate_id} -> PROMOTED")
    store.close()
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    report = doctor(_home(args))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "READY" else 1


def cmd_init(args: argparse.Namespace) -> int:
    home = _home(args)
    home.mkdir(parents=True, exist_ok=True)
    choice = args.provider
    if choice is None:
        if not sys.stdin.isatty():
            print("非交互环境需要 --provider：" + ", ".join(PROVIDER_CHOICES), file=sys.stderr)
            return 2
        print("选择模型提供方：")
        for i, c in enumerate(PROVIDER_CHOICES, 1):
            print(f"  {i}. {c}")
        raw = input("编号 [1]: ").strip() or "1"
        choice = PROVIDER_CHOICES[max(0, min(len(PROVIDER_CHOICES) - 1, int(raw) - 1))]
    summary = configure_model(
        home,
        choice,
        base_url=args.base_url,
        model=args.model,
        api_key_ref=args.api_key_ref,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary.get("configured"):
        return 0
    print("\n运行 doctor 探测（connectivity / models / chat / tool call）…")
    report = doctor(home)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report.get("status") != "READY":
        print(
            "\n模型尚未就绪（MODEL_NOT_READY）。这是诚实状态，不是假成功；"
            "请检查 key/endpoint 后重试 `rosclaw agent doctor`。",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    home = _home(args)
    config = load_agent_config(home / "config.yaml")
    if not config.profiles:
        print("未配置模型。先运行 `rosclaw agent init`。", file=sys.stderr)
        return 2
    service = AgentService(config, home)
    return asyncio.run(_chat_repl(service, args))


async def _chat_repl(service: AgentService, args: argparse.Namespace) -> int:
    mission = None
    if args.mission:
        mission = service.get_mission(args.mission)
        if mission is None:
            print(f"mission {args.mission} 不存在", file=sys.stderr)
            return 2
    else:
        goal = args.goal or "ROSClaw chat session"
        try:
            mission = service.create_mission(goal, mode=args.mode)
        except Exception as exc:  # noqa: BLE001 - surface honest refusal
            print(f"无法创建 mission：{exc}", file=sys.stderr)
            return 2
    print(f"ROSClaw chat — mission {mission.mission_id} [{mission.mode.value}]")
    print(
        "输入消息开始对话；/state 查看状态；/approvals 待授权；"
        "/approve|/deny <id> 决定授权；/cancel 取消当前回合；/quit 退出。"
    )
    try:
        while True:
            try:
                text = input("\n你> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                continue
            if text == "/quit":
                break
            if text == "/state":
                current = service.get_mission(mission.mission_id)
                print(f"state={current.state.value} mode={current.mode.value}")
                continue
            if text == "/cancel":
                await service.cancel(mission.mission_id)
                print("已请求取消当前回合。")
                continue
            if text == "/approvals":
                pending = service.pending_approvals(mission.mission_id)
                if not pending:
                    print("没有待处理的授权请求。")
                for req in pending:
                    d = req.action_display
                    print(
                        f"  {req.request_id} [{d.risk_tier}] {d.title}: {d.summary} "
                        f"(expires {req.expires_at})"
                    )
                continue
            if text.startswith("/approve ") or text.startswith("/deny "):
                approve = text.startswith("/approve ")
                request_id = text.split(maxsplit=1)[1].strip()
                try:
                    grant = await service.decide_approval(
                        request_id, principal="user:local:1000", approve=approve
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"授权操作失败：{exc}")
                    continue
                if grant is not None:
                    print(
                        f"已批准并签发 grant {grant.grant_id}（public_hash "
                        f"{grant.public_hash[:24]}…，EXACT_ACTION 单次有效）。"
                    )
                else:
                    print("已拒绝该授权请求。")
                continue
            streamed = False

            def on_delta(piece: str) -> None:
                nonlocal streamed
                streamed = True
                print(piece, end="", flush=True)

            print("ROSClaw> ", end="", flush=True)
            result = await service.send_turn(mission.mission_id, text, on_delta)
            if not streamed:
                print(result.reply, end="")
            usage = service.mission_usage(mission.mission_id)
            degraded = f"  [degraded: {result.degraded}]" if result.degraded else ""
            print(
                f"\n[{result.state.value}] 本轮 tokens={result.tokens_used}"
                f" 累计 tokens={usage['total_tokens']} cost={usage['cost_microunits']}µ{degraded}"
            )
    finally:
        await service.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rosclaw-agentd", description="ROSClaw Native Agent")
    parser.add_argument("--home", default=None, help="ROSClaw home (default ~/.rosclaw)")
    sub = parser.add_subparsers(dest="agent_command", required=True)

    p_start = sub.add_parser("start", help="foreground HTTP service")
    p_start.add_argument("--host", default="127.0.0.1")
    p_start.add_argument("--port", type=int, default=8765)
    p_start.set_defaults(func=cmd_start)

    p_status = sub.add_parser("status", help="service status")
    p_status.set_defaults(func=cmd_status)

    p_doctor = sub.add_parser("doctor", help="model/agent readiness probe")
    p_doctor.set_defaults(func=cmd_doctor)

    p_init = sub.add_parser("init", help="configure model provider + probe")
    p_init.add_argument("--provider", choices=PROVIDER_CHOICES, default=None)
    p_init.add_argument("--base-url", default=None)
    p_init.add_argument("--model", default=None)
    p_init.add_argument("--api-key-ref", default=None)
    p_init.set_defaults(func=cmd_init)

    p_chat = sub.add_parser("chat", help="interactive chat (in-process)")
    p_chat.add_argument("--mission", default=None)
    p_chat.add_argument("--mode", default=None, choices=["SIMULATION", "SHADOW", "REAL"])
    p_chat.add_argument("--goal", default=None)
    p_chat.set_defaults(func=cmd_chat)

    p_worker = sub.add_parser("worker", help="Worker Fabric management")
    worker_sub = p_worker.add_subparsers(dest="worker_command", required=True)
    add_worker_subcommands(worker_sub)
    return parser


def add_worker_subcommands(sub) -> None:
    p_wl = sub.add_parser("list", help="list registered workers")
    p_wl.add_argument("--status", default=None, choices=["ENABLED", "DISABLED", "QUARANTINED"])
    p_wl.set_defaults(func=cmd_worker_list)
    p_wc = sub.add_parser("catalog", help="official WorkerPack catalog")
    p_wc.set_defaults(func=cmd_worker_catalog)
    p_wi = sub.add_parser("inspect", help="show a worker card")
    p_wi.add_argument("worker_id")
    p_wi.set_defaults(func=cmd_worker_inspect)
    p_wp = sub.add_parser("probe", help="probe an external pack binary/version")
    p_wp.add_argument("worker_id")
    p_wp.set_defaults(func=cmd_worker_probe)
    for name in ("enable", "disable"):
        p_ws = sub.add_parser(name, help=f"{name} a worker")
        p_ws.add_argument("worker_id")
        p_ws.add_argument("--reason", default="")
        p_ws.set_defaults(func=cmd_worker_set_status)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


# ----------------------------------------------------------------------
# Mount into the main `rosclaw` CLI as `rosclaw agent ...` / `rosclaw chat`.
# ----------------------------------------------------------------------
def add_agent_subparsers(subparsers) -> None:
    # NOTE: `rosclaw agent` is the *external harness* onboarding surface
    # (pre-existing). The Native Agent mounts as `rosclaw agentd`; the
    # standalone console script is `rosclaw-agentd`.
    p_agent = subparsers.add_parser("agentd", help="Native Agent (rosclaw-agentd)")
    agent_sub = p_agent.add_subparsers(dest="agentd_command", required=True)
    for name, fn, helptext in (
        ("start", cmd_start, "foreground HTTP service"),
        ("status", cmd_status, "service status"),
        ("doctor", cmd_doctor, "model/agent readiness probe"),
        ("init", cmd_init, "configure model provider + probe"),
    ):
        p = agent_sub.add_parser(name, help=helptext)
        if name == "start":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8765)
        if name == "init":
            p.add_argument("--provider", choices=PROVIDER_CHOICES, default=None)
            p.add_argument("--base-url", default=None)
            p.add_argument("--model", default=None)
            p.add_argument("--api-key-ref", default=None)
        p.set_defaults(func=fn)

    p_worker = subparsers.add_parser("worker", help="Worker Fabric management")
    worker_sub = p_worker.add_subparsers(dest="worker_command", required=True)
    add_worker_subcommands(worker_sub)

    p_bench = subparsers.add_parser("eval", help="evaluation benchmark harness")
    bench_sub = p_bench.add_subparsers(dest="bench_command", required=True)
    p_brun = bench_sub.add_parser("run", help="run scenario matrix")
    p_brun.add_argument("--seeds", default="1,2,3")
    p_brun.add_argument("--groups", default="A,B")
    p_brun.add_argument("--out", default="bench_out")
    p_brun.set_defaults(func=cmd_bench_run)

    p_learn = subparsers.add_parser("learning", help="learning candidates")
    learn_sub = p_learn.add_subparsers(dest="learning_command", required=True)
    p_ll = learn_sub.add_parser("list", help="list candidates")
    p_ll.add_argument("--status", default=None)
    p_ll.set_defaults(func=cmd_learning_list)
    p_le = learn_sub.add_parser("extract", help="extract candidates from a mission")
    p_le.add_argument("mission_id")
    p_le.set_defaults(func=cmd_learning_extract)
    p_lp = learn_sub.add_parser("promote", help="promote a candidate (Darwin gate)")
    p_lp.add_argument("candidate_id")
    p_lp.add_argument("--principal", default="user:local:1000")
    p_lp.add_argument("--evaluation-ref", required=True)
    p_lp.set_defaults(func=cmd_learning_promote)

    p_chat = subparsers.add_parser("chat", help="chat with the Native Agent")
    p_chat.add_argument("--mission", default=None)
    p_chat.add_argument("--mode", default=None, choices=["SIMULATION", "SHADOW", "REAL"])
    p_chat.add_argument("--goal", default=None)
    p_chat.set_defaults(func=cmd_chat)


def dispatch_agent_command(args: argparse.Namespace) -> int:
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
