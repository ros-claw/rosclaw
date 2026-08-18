"""rosclaw-agentd CLI: start/status/doctor + `rosclaw agent` + `rosclaw chat`.

The service runs foreground (uvicorn) or in-process for `rosclaw chat`.
A single-instance lock file prevents two agentd processes on one home.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import os
import sys
from pathlib import Path

from rosclaw.agentd.config import load_agent_config
from rosclaw.agentd.credentials import (
    AgentCredentialStore,
    CredentialStoreError,
    ModelCredentialBroker,
)
from rosclaw.agentd.onboarding import PROVIDER_CHOICES, configure_model, doctor
from rosclaw.agentd.pi_entry import (
    find_pi_agent_entry as _find_pi_agent_entry,
)
from rosclaw.agentd.pi_entry import (
    find_tui_runtime as _find_tui_runtime,
)
from rosclaw.agentd.service import AgentService, create_app

LOCK_NAME = "agentd/agentd.lock"
CREDENTIAL_ENV_BY_PROVIDER = {
    "kimi-code": "ROSCLAW_KIMI_API_KEY",
    "kimi-api": "MOONSHOT_API_KEY",
}


def _home(args: argparse.Namespace) -> Path:
    return Path(
        getattr(args, "home", None) or os.environ.get("ROSCLAW_HOME", Path.home() / ".rosclaw")
    )


def _load_stored_credentials(home: Path) -> bool:
    """NA-FIX-7：统一经 ModelCredentialBroker——legacy 一次性 read-and-migrate，
    不再双写。"""
    try:
        ModelCredentialBroker(home).migrate_legacy_once()
    except CredentialStoreError as exc:
        print(str(exc), file=sys.stderr)
        return False
    return True


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
    if not _load_stored_credentials(home):
        return 2
    config = load_agent_config(home / "config.yaml")
    if not config.profiles:
        print(
            "未配置模型。先运行 `rosclaw setup model`，或 `rosclaw agent doctor` 查看缺口。",
            file=sys.stderr,
        )
        return 2
    lock = _acquire_lock(home)
    # 十四审 PR-14.7（§1.9）：agentd serve 入口也要先装诊断路由——
    # 十三审只覆盖了 chat 入口，干净安装路径仍漏启动告警。
    _route_internal_diagnostics_to_log(home, debug=bool(getattr(args, "debug", False)))
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
    home = _home(args)
    topic = str(getattr(args, "topic", "") or "")
    if topic:
        # 十六审 P0-C：托管 runtime 探测（simulation 等）——不需要
        # 模型凭据；报告托管解释器的真实 probe 结果。
        from rosclaw.agentd.runtime_manager import doctor_runtime

        report = doctor_runtime(home, topic)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get("ready") else 1
    if not _load_stored_credentials(home):
        return 2
    report = doctor(home)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "READY" else 1


def cmd_init(args: argparse.Namespace) -> int:
    home = _home(args)
    home.mkdir(parents=True, exist_ok=True)
    if not _load_stored_credentials(home):
        return 2
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
    if not _load_stored_credentials(home):
        return 2
    # NA-FIX-9（规格 §5/§31，Gate A–E 已通过）：Native Agent（pi）是默认
    # 引擎；legacy 仅作隐藏回退保留一个稳定版本。
    # 优先级：--legacy > --engine > config.agent.engine > pi。
    if getattr(args, "legacy", False):
        engine = "legacy"
    elif getattr(args, "engine", None):
        engine = args.engine
    else:
        try:
            engine = load_agent_config(home / "config.yaml").engine
        except Exception:  # noqa: BLE001 - 配置问题由后续步骤诚实报出
            engine = "pi"
    # 两种引擎都需要模型配置——缺失时给出同一指引（pi 侧由
    # ModelRuntime/auth 接续处理具体 provider 细节）。
    try:
        if not load_agent_config(home / "config.yaml").profiles:
            print("未配置模型。先运行 `rosclaw setup model`。", file=sys.stderr)
            return 2
    except ValueError:
        print("未配置模型。先运行 `rosclaw setup model`。", file=sys.stderr)
        return 2
    if engine == "pi":
        return _chat_pi(home, args)
    config = load_agent_config(home / "config.yaml")
    # 诊断路由必须先于 AgentService 构造（启动 warning 就在那里出现）。
    _route_internal_diagnostics_to_log(home, debug=bool(getattr(args, "debug", False)))
    service = AgentService(config, home)
    if getattr(args, "basic", False):
        return asyncio.run(_chat_repl(service, args))
    return _chat_tui(service, args)


def _route_internal_diagnostics_to_log(home: Path, *, debug: bool) -> None:
    """十审 W0（P1-PRODUCT-NOISE）：Python warnings（pydantic forward-ref、
    第三方 deprecation 等内部诊断）默认进 logs/python-warnings.log，
    不糊 TUI 第一屏；--debug 或 ROSCLAW_DEBUG 时保持终端可见。
    """
    import logging as _logging
    import warnings as _warnings

    # 十二审 HOTFIX-12.1：pydantic_settings 对带 forward-ref 的第三方
    # settings 模型（如 uvicorn/fastapi 生态的 lifespan 字段）在
    # pydantic 2.13+ 发 IncompleteFieldDefinitionWarning——已知良性
    # 第三方告警。十四审 PR-14.7：按 类别+消息 定向过滤（任何模块
    # 来源——该告警的定义点随第三方版本漂移，按 module 匹配会漏）；
    # 其余 warning 不受影响，仍进日志文件。
    _warnings.filterwarnings(
        "ignore",
        message=".*incomplete definition.*",
        category=UserWarning,
    )
    if debug or os.environ.get("ROSCLAW_DEBUG"):
        return
    try:
        log_dir = home / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = _logging.FileHandler(log_dir / "python-warnings.log")
        handler.setFormatter(
            _logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )
        py_warnings = _logging.getLogger("py.warnings")
        py_warnings.addHandler(handler)
        py_warnings.propagate = False
        _logging.captureWarnings(True)
    except OSError:
        pass


def _chat_pi(home: Path, args: argparse.Namespace) -> int:
    """engine=pi：agentd 内核（sockets/token）+ exec rosclaw-agent。

    规格 §2.1：Pi 是唯一主认知循环——Python AgentLoop 不接收用户 turn；
    agentd 只提供 pi-bridge/operator socket 与控制 token（具身内核服务）。
    """
    import subprocess as _sp
    import threading
    import time

    runtime = _find_pi_agent_entry()
    if runtime is None:
        print(
            "Native Agent 需要 Node ≥22.19 且已构建 packages/rosclaw-agent"
            "（发布包自带；源码环境先 npm ci && npm run build）。"
            "临时回退：rosclaw chat --legacy",
            file=sys.stderr,
        )
        return 2
    node, entry = runtime
    config = load_agent_config(home / "config.yaml")
    # 十审 W0：诊断路由必须先于 AgentService 构造（启动 warning 就在那里）。
    _route_internal_diagnostics_to_log(home, debug=bool(getattr(args, "debug", False)))
    service = AgentService(config, home)
    # Mission：--mission 复用或新建（SIMULATION 默认）。
    # --continue/--resume 不预建 Mission（P0-2：由 session 切换事务
    # 在 Native Agent 侧恢复既有绑定）。
    mission = None
    resume_argv: list[str] = []
    if getattr(args, "continue_last", False):
        resume_argv = ["--continue"]
    elif getattr(args, "resume", None):
        query = str(args.resume)
        if query == "__picker__":
            # WP-P0-1：裸 --resume → 会话选择器。
            resume_argv = ["--browse-sessions"]
        else:
            # WP-P0-1：ID/前缀/标题 → 真实 session 路径（用户不再
            # 需要知道内部 ID；歧义报候选不猜）。
            from rosclaw.agentd.session_list import (
                list_sessions,
                resolve_session_query,
            )

            hit = resolve_session_query(list_sessions(home), query)
            if hit.get("error"):
                print(f"会话未解析：{query}", file=sys.stderr)
                for cand in hit.get("candidates", []):
                    print(
                        "  候选："
                        f"{cand.get('display_name') or cand.get('first_message') or cand['session_id']}"
                        f"（{cand['session_id'][:12]}…）",
                        file=sys.stderr,
                    )
                return 2
            resume_argv = ["--resume-path", hit["path"]]
    if args.mission:
        mission = service.get_mission(args.mission)
        if mission is None:
            print(f"mission {args.mission} 不存在", file=sys.stderr)
            return 2
    elif not resume_argv:
        try:
            mission = service.create_mission(args.goal or "ROSClaw chat session", mode=args.mode)
        except Exception as exc:  # noqa: BLE001
            print(f"无法创建 mission：{exc}", file=sys.stderr)
            return 2
    # 起 HTTP app 仅为驱动 lifespan（operator sock + pi bridge + token 文件）。
    # 六审 SIX-6 旅程暴露：service 的 MCP 持久客户端在 lifespan loop 上
    # 创建——close 必须在同一个 loop 上跑（anyio cancel scope 是 task
    # 绑定的；asyncio.run(service.close()) 换新 loop 会在退出时炸
    # "cancel scope in a different task"，进程非零退出）。
    import uvicorn

    app = create_app(service)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"))
    server_loop = asyncio.new_event_loop()

    serve_done = threading.Event()

    def _run_server() -> None:
        asyncio.set_event_loop(server_loop)
        server_loop.run_until_complete(server.serve())
        serve_done.set()
        # serve 返回后 loop 必须继续受驱动——service.close() 要在这个
        # loop 上跑（MCP 客户端的 anyio cancel scope 绑定它）。
        server_loop.run_forever()

    thread = threading.Thread(target=_run_server, daemon=True)
    thread.start()
    deadline = time.time() + 5.0
    while time.time() < deadline and not (home / "run" / "pi-bridge.sock").exists():
        time.sleep(0.05)
    if not (home / "run" / "pi-bridge.sock").exists():
        print("内核桥未能启动——Native Agent 不可用（agentd 内核未就绪）。", file=sys.stderr)
        server.should_exit = True
        try:
            asyncio.run_coroutine_threadsafe(service.close(), server_loop).result(timeout=15)
        finally:
            server_loop.call_soon_threadsafe(server_loop.stop)
        return 2
    # P0-9：SHADOW/REAL 强制 ROBOT profile——用户不能经 CLI 把
    # REAL 降级为 developer（SIM 桌面用 developer）。
    # P0-NA-12：--resume/--continue 也必须解析出目标 session 的 mission
    # mode——恢复非 SIM Mission 时用 developer profile 是安全降级。
    resume_mode = None
    if resume_argv:
        resume_mode = _resume_target_mode(home, args)
    effective_mode = (
        mission.mode.value if mission is not None else resume_mode
    )
    profile = "robot" if effective_mode is not None and effective_mode != "SIMULATION" else "developer"
    # 六审 §7：产品 supervisor——SIM developer 且已 enrollment 但服务未
    # 运行时，chat 直接代为启动独立 operatord（生命周期归本进程；
    # 决定权/签名仍在 operatord）。未 enrollment 不自动办理——TUI 内
    # 单键初始化（带安全说明）。REAL/robot 永不自动。
    managed_operatord = None
    if profile == "developer":
        from rosclaw.operatord.enrollment import IDENTITY_FILE

        enrolled = (home / "operatord" / IDENTITY_FILE).exists()
        operator_sock = home / "run" / "operatord.sock"
        if enrolled and not operator_sock.exists():
            managed_operatord = _sp.Popen(  # noqa: S603 - 固定入口
                [
                    sys.executable, "-m", "rosclaw.entrypoint",
                    "operatord", "start", "--no-human-presence-check",
                ],
                env=dict(os.environ, ROSCLAW_HOME=str(home)),
                stdout=(home / "run" / "operatord.out.log").open("ab"),
                stderr=(home / "run" / "operatord.err.log").open("ab"),
            )
    argv = [node, entry, "--profile", profile, *resume_argv]
    if mission is not None:
        argv += ["--mission", mission.mission_id]
    # 十一审 PR-D：chat [PATH]/--workspace → 显式传给 Native Agent
    # （git root 归一与自动绑定在 TS 侧）。
    ws_arg = getattr(args, "workspace", None) or getattr(args, "path", None)
    if ws_arg:
        argv += ["--workspace", str(Path(ws_arg).expanduser().resolve())]
    # P0-NA-16：产品版本由 Python launcher 显式传给 Node——TS 侧不得
    # 用内部 npm 子包版本冒充 ROSClaw 产品版本。
    from rosclaw import __version__ as _product_version

    env = dict(
        os.environ,
        ROSCLAW_HOME=str(home),
        ROSCLAW_PRODUCT_VERSION=_product_version,
    )
    try:
        return _sp.call(argv, env=env)  # noqa: S603 - fixed entry
    except KeyboardInterrupt:
        return 0
    finally:
        server.should_exit = True
        # 六审 SIX-6：先等服务真正停下，再在同一 loop 上跑 close——
        # 并发 close 会在 ASGI 关闭途中触发 Event-loop 级竞态。
        serve_done.wait(timeout=10)
        try:
            asyncio.run_coroutine_threadsafe(service.close(), server_loop).result(timeout=15)
        except Exception as exc:  # noqa: BLE001 - 退出路径诚实记录
            print(f"内核关闭异常：{type(exc).__name__}: {exc}", file=sys.stderr)
        finally:
            server_loop.call_soon_threadsafe(server_loop.stop)
            thread.join(timeout=5)
        # supervisor 启动的 operatord 随 chat 退出（已运行他人启动的
        # 不碰——managed_operatord 只在本进程启动时非空）。
        if managed_operatord is not None:
            managed_operatord.terminate()


def _resume_target_mode(home: Path, args: argparse.Namespace) -> str | None:
    """--resume/--continue 目标 session 的 mission mode（P0-NA-12）。

    session 文件头 id → pi_session_bindings → mission.mode。任一环节
    缺失返回 None（按 SIM/developer 处理并如实记录——绑定恢复时
    coordinator 会以权威 mission 为准再次校验）。
    """
    import json as _json

    session_dir = home / "agent" / "sessions"
    session_id = ""
    if getattr(args, "resume", None):
        candidate = str(args.resume)
        # 与 Node 侧同一规则：纯标识符，拒绝路径穿越。
        import re as _re

        if not _re.fullmatch(r"[A-Za-z0-9_-]+", candidate):
            return None
        session_id = candidate
    elif getattr(args, "continue_last", False):
        try:
            newest = max(
                session_dir.glob("*.jsonl"),
                key=lambda p: p.stat().st_mtime,
                default=None,
            )
            if newest is None:
                return None
            header = _json.loads(newest.read_text(encoding="utf-8").split("\n", 1)[0])
            session_id = str(header.get("id", ""))
        except Exception:  # noqa: BLE001 - 恢复失败由 Node 侧诚实报出
            return None
    if not session_id:
        return None
    db_path = home / "agentd" / "missions.db"
    if not db_path.exists():
        return None
    import sqlite3 as _sqlite3

    try:
        conn = _sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT mission_id FROM pi_session_bindings "
                "WHERE pi_session_id = ? AND status = 'ACTIVE'",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            mission_row = conn.execute(
                "SELECT mode FROM missions WHERE mission_id = ?", (row[0],)
            ).fetchone()
            return str(mission_row[0]) if mission_row else None
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        return None


def _chat_tui(service: AgentService, args: argparse.Namespace) -> int:
    """默认 chat：启动本地 AgentService HTTP + exec rosclaw-tui（批次 C）。

    TUI 不可用（Node/资源缺失）时诚实降级到 --basic 并说明原因。
    """
    import subprocess as _sp
    import threading

    runtime = _find_tui_runtime()
    if runtime is None:
        if os.environ.get("ROSCLAW_REQUIRE_TUI") == "1":
            # 审计 P0-05.6：验收环境不允许静默回退。
            print(
                "FAIL: rosclaw-tui 不可用且 ROSCLAW_REQUIRE_TUI=1——"
                "完整验收失败（不允许静默回退 --basic）。",
                file=sys.stderr,
            )
            return 2
        print(
            "rosclaw-tui 不可用（需要 Node >= 22.19 与已构建的 packages/rosclaw-tui；"
            "见 rosclaw doctor）。回退到兼容模式 --basic（显式救援模式）。",
            file=sys.stderr,
        )
        return asyncio.run(_chat_repl(service, args))
    node, entry = runtime

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
        except Exception as exc:  # noqa: BLE001
            print(f"无法创建 mission：{exc}", file=sys.stderr)
            return 2

    import uvicorn

    app = create_app(service)
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # 等待 socket 就绪（最多 5s），拿不到端口就诚实降级。
    import time

    deadline = time.time() + 5.0
    port = None
    while time.time() < deadline:
        if server.started and server.servers:
            for srv in server.servers:
                for sock in srv.sockets:
                    port = sock.getsockname()[1]
                    break
        if port:
            break
        time.sleep(0.05)
    if port is None:
        print("AgentService HTTP 启动失败，回退到 --basic。", file=sys.stderr)
        server.should_exit = True
        return asyncio.run(_chat_repl(service, args))
    try:
        return _sp.call(
            [
                node,
                entry,
                "--url",
                f"http://127.0.0.1:{port}",
                "--mission",
                mission.mission_id,
            ]
        )
    finally:
        server.should_exit = True
        thread.join(timeout=3.0)


def cmd_credential(args: argparse.Namespace) -> int:
    env_name = CREDENTIAL_ENV_BY_PROVIDER[args.provider]
    try:
        store = AgentCredentialStore(_home(args))
        if args.credential_command == "set":
            value = (
                getpass.getpass(f"{env_name}: ").strip()
                if sys.stdin.isatty()
                else sys.stdin.read().strip()
            )
            store.set(env_name, value)
            result = store.status(env_name)
            result["provider"] = args.provider
            result["updated"] = True
        elif args.credential_command == "delete":
            deleted = store.delete(env_name)
            result = store.status(env_name)
            result["provider"] = args.provider
            result["deleted"] = deleted
        else:
            result = store.status(env_name)
            result["provider"] = args.provider
    except (CredentialStoreError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_backend(args: argparse.Namespace) -> int:
    """查看/切换模型 backend（批次 D：Kimi 现有配置无需改动即可迁移）。"""
    import yaml

    home = _home(args)
    config_path = home / "config.yaml"
    data = {}
    if config_path.exists():
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    models = data.setdefault("models", {})
    current = models.get("backend", "legacy")
    if not args.set:
        print(json.dumps({"backend": current}, ensure_ascii=False))
        return 0
    if args.set == current:
        print(f"backend 已是 {current}")
        return 0
    if args.set == "modeld":
        from rosclaw.agentd.models.modeld_gateway import _find_modeld_runtime

        if _find_modeld_runtime() is None:
            print(
                "rosclaw-modeld 不可用（需要 Node >= 22.19 与已构建的 "
                "packages/rosclaw-modeld）。修复后再切换。",
                file=sys.stderr,
            )
            return 2
    models["backend"] = args.set
    config_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    print(
        f"backend: {current} → {args.set}。现有 profile 与 env 凭据引用保持不变；下一 turn 生效。"
    )
    return 0


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
    # The basic REPL is an operator surface.  Expose its approval projection
    # before accepting commands so the independently enrolled operatord can
    # verify and decide the exact cards created by this service instance.
    await service.start_operator_socket()
    print(f"ROSClaw chat — mission {mission.mission_id} [{mission.mode.value}]")
    print(
        "输入消息开始对话；/state 查看状态；/approvals 待授权；"
        "/approve|/deny <id> 经 operatord 决定授权；/compact [focus|--dry-run|--status]；"
        "/cancel 取消当前回合；/quit 退出。"
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
            if text.startswith("/compact"):
                arg = text[len("/compact") :].strip()
                if arg == "--status":
                    status = service.compaction_status(mission.mission_id)
                    print(
                        f"压缩次数={status['compactions']} view_tokens="
                        f"{status['current_view_tokens']} journal_events={status['journal_events']}"
                    )
                    continue
                dry = arg == "--dry-run"
                focus = None if arg in ("", "--dry-run") else arg
                report = await service.compact(mission.mission_id, instructions=focus, dry_run=dry)
                if dry:
                    print(
                        f"[dry-run] tokens={report['tokens_before']} 切点={report['cut_index']}"
                        f"/{report['messages_total']}，预计保留至 tokens≈{report['tokens_after']}"
                    )
                else:
                    print(
                        f"已压缩：{report['tokens_before']}→{report['tokens_after']} tokens，"
                        f"保留 {report.get('kept_messages', '?')} 条（id={report.get('compaction_id', '—')}）"
                    )
                continue
            if text == "/approvals":
                from rosclaw.agentd.operator_socket import display_hash_for

                pending = service.pending_approvals(mission.mission_id)
                if not pending:
                    print("没有待处理的授权请求。")
                for req in pending:
                    d = req.action_display
                    print(
                        f"  {req.request_id} [{d.risk_tier}] {d.title}: {d.summary} "
                        f"(expires {req.expires_at}) hash={display_hash_for(req)}"
                    )
                continue
            if text.startswith("/approve ") or text.startswith("/deny "):
                approve = text.startswith("/approve ")
                request_id = text.split(maxsplit=1)[1].strip()
                # 审计 P0-01：REPL 不直接决定——决定经独立 rosclaw-operatord
                # （enrollment proof + display hash 绑定），与 TUI 同一通道。
                from rosclaw.agentd.operator_socket import display_hash_for, operator_call
                from rosclaw.operatord.server import default_operatord_socket

                target = next(
                    (
                        r
                        for r in service.pending_approvals(mission.mission_id)
                        if r.request_id == request_id
                    ),
                    None,
                )
                if target is None:
                    print("没有找到该待批卡片（可能已过期或已被决定）。")
                    continue
                sock = default_operatord_socket(_home(args))
                if not sock.exists():
                    print(
                        "授权决定已迁至独立 rosclaw-operatord（P0-01），本进程无权决定。\n"
                        "请先运行：rosclaw operatord enroll && rosclaw operatord start"
                    )
                    continue
                reply = await operator_call(
                    sock,
                    "approvals.decide",
                    {
                        "request_id": request_id,
                        "display_hash": display_hash_for(target),
                        "approve": approve,
                    },
                )
                if not reply.get("ok"):
                    print(f"授权操作失败：{reply.get('error', reply)}")
                elif approve:
                    print(
                        f"已批准并签发 grant {reply.get('grant_id', '—')}"
                        "（EXACT_ACTION 单次有效）。"
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
            elif result.decisions and result.decisions[-1].next_intent.value != "ANSWER":
                # Streamed prose is the model's proposal/explanation. The
                # handler reply is the trusted system outcome (approval id,
                # dispatch status, or receipt) and must not be hidden.
                print(f"\nROSClaw system> {result.reply}", end="")
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
    p_doctor.add_argument(
        "topic", nargs="?", default="",
        help="主题探测：simulation = 托管仿真 runtime（无需模型凭据）",
    )
    p_doctor.set_defaults(func=cmd_doctor)

    p_init = sub.add_parser("init", help="configure model provider + probe")
    p_init.add_argument("--provider", choices=PROVIDER_CHOICES, default=None)
    p_init.add_argument("--base-url", default=None)
    p_init.add_argument("--model", default=None)
    p_init.add_argument("--api-key-ref", default=None)
    p_init.set_defaults(func=cmd_init)

    add_credential_subcommands(sub)

    p_backend = sub.add_parser("backend", help="model backend: legacy | modeld")
    p_backend.add_argument("--set", choices=["legacy", "modeld"], default=None)
    p_backend.set_defaults(func=cmd_backend)

    p_chat = sub.add_parser("chat", help="interactive chat (in-process)")
    p_chat.add_argument("--mission", default=None)
    p_chat.add_argument("--mode", default=None, choices=["SIMULATION", "SHADOW", "REAL"])
    p_chat.add_argument("--goal", default=None)
    p_chat.set_defaults(func=cmd_chat)

    p_worker = sub.add_parser("worker", help="Worker Fabric management")
    worker_sub = p_worker.add_subparsers(dest="worker_command", required=True)
    add_worker_subcommands(worker_sub)
    return parser


def add_credential_subcommands(sub) -> None:
    p_credential = sub.add_parser("credential", help="owner-only model credentials")
    credential_sub = p_credential.add_subparsers(dest="credential_command", required=True)
    for name in ("set", "status", "delete"):
        parser = credential_sub.add_parser(name, help=f"{name} a persisted model credential")
        parser.add_argument("--provider", choices=tuple(CREDENTIAL_ENV_BY_PROVIDER), required=True)
        parser.set_defaults(func=cmd_credential)


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
        if name == "doctor":
            p.add_argument(
                "topic", nargs="?", default="",
                help="主题探测：simulation = 托管仿真 runtime（无需模型凭据）",
            )
        p.set_defaults(func=fn)

    add_credential_subcommands(agent_sub)

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
    # 十一审 PR-D：rosclaw chat [PATH]——Project workspace 一等状态。
    p_chat.add_argument("path", nargs="?", default=None,
                        help="项目目录（git 仓库自动绑定为 workspace）")
    p_chat.add_argument("--workspace", default=None, help="显式指定 workspace 路径")
    p_chat.add_argument("--mission", default=None)
    p_chat.add_argument("--mode", default=None, choices=["SIMULATION", "SHADOW", "REAL"])
    p_chat.add_argument("--goal", default=None)
    p_chat.add_argument(
        "--basic",
        action="store_true",
        help="兼容/诊断模式：Python input() 行式 REPL（无 TUI）",
    )
    p_chat.add_argument(
        "--debug",
        action="store_true",
        help="诊断模式：内部 warning/MCP 子进程诊断回显终端（默认写入 logs/）",
    )
    p_chat.add_argument(
        "--engine",
        default=None,
        choices=["pi", "legacy"],
        # NA-FIX-9：公开面不再暴露引擎概念；仅兼容旧脚本/诊断。
        help=argparse.SUPPRESS,
    )
    p_chat.add_argument(
        "--legacy",
        action="store_true",
        # 隐藏回退（保留一个稳定版本后随 legacy 一起退役）。
        help=argparse.SUPPRESS,
    )
    p_chat.add_argument(
        "--continue",
        dest="continue_last",
        action="store_true",
        help="继续最近一次会话（Native Agent session + Mission 绑定）",
    )
    p_chat.add_argument(
        "--resume",
        nargs="?",
        const="__picker__",
        default=None,
        metavar="ID或标题",
        # WP-P0-1：裸 --resume 打开会话选择器；参数支持精确 ID/
        # 唯一前缀/标题（由 session_list.resolve_session_query 解析）。
        help="恢复会话：无参数打开选择器；或给 ID/唯一前缀/标题",
    )
    p_chat.set_defaults(func=cmd_chat)

    # WP-P0-1（总纲 §4.2/§5.1）：会话可发现性——用户不再需要知道
    # 内部 session id。
    p_sessions = subparsers.add_parser(
        "sessions", help="列出/搜索会话（TTY 下引导选择器）"
    )
    p_sessions.add_argument("query", nargs="?", default="", help="搜索标题/内容")
    p_sessions.set_defaults(func=cmd_sessions)
    p_resume = subparsers.add_parser(
        "resume", help="恢复会话：无参数打开选择器；或给 ID/前缀/标题"
    )
    p_resume.add_argument("query", nargs="?", default="", metavar="ID或标题")
    p_resume.set_defaults(func=cmd_resume)
    p_continue = subparsers.add_parser("continue", help="继续最近会话")
    p_continue.set_defaults(func=cmd_continue)


def cmd_sessions(args: argparse.Namespace) -> int:
    # rosclaw sessions——产品级会话列表（WP-P0-2：走 SessionCatalog
    # 产品索引；先 refresh 增量回填，不再每次全量扫 JSONL）。
    from rosclaw.agentd.mission.store import MissionStore
    from rosclaw.agentd.session_catalog import SessionCatalog

    home = _home(args)
    db_path = home / "agentd" / "missions.db"
    if not db_path.exists():
        print("还没有会话。运行 `rosclaw chat` 开始第一个任务。")
        return 0
    store = MissionStore(db_path)
    catalog = SessionCatalog(store.connection)
    catalog.refresh(home)
    query = str(getattr(args, "query", "") or "")
    sessions = catalog.search(query) if query else catalog.list()
    if not sessions:
        print("还没有会话。运行 `rosclaw chat` 开始第一个任务。")
        return 0
    for s in sessions[:30]:
        title = s.get("display_name") or "（未命名）"
        robot = s.get("body_id") or "-"
        state = s.get("lifecycle_state") or ""
        print(f"  {title:<30}  {robot:<12} {state:<10} {s['session_id'][:12]}…")
    print("\n恢复：rosclaw resume <标题或ID> · 继续最近：rosclaw continue")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    # rosclaw resume [ID|标题]——无参数打开选择器。
    query = str(getattr(args, "query", "") or "")
    args.resume = query if query else "__picker__"
    args.continue_last = False
    for attr, default in (("mission", None), ("mode", None), ("goal", None)):
        if not hasattr(args, attr):
            setattr(args, attr, default)
    return cmd_chat(args)


def cmd_continue(args: argparse.Namespace) -> int:
    # rosclaw continue——继续最近会话。
    args.continue_last = True
    args.resume = None
    for attr, default in (("mission", None), ("mode", None), ("goal", None)):
        if not hasattr(args, attr):
            setattr(args, attr, default)
    return cmd_chat(args)


def dispatch_agent_command(args: argparse.Namespace) -> int:
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
