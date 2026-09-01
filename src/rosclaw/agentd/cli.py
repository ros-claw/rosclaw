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
from rosclaw.agentd.pi_entry import (
    find_pi_agent_entry as _find_pi_agent_entry,
)
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
    from rosclaw.agentd.pi_config import pi_model_configured

    if not pi_model_configured(home):
        print(
            "未配置模型。先运行 `rosclaw setup model`，或 `rosclaw agent doctor` 查看缺口。",
            file=sys.stderr,
        )
        return 2
    config = load_agent_config(home / "config.yaml")
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


# ----------------------------------------------------------------------
# R0-4（0826 体验审计 §5.R0-4）：artifact 用户可达交付面。
# ----------------------------------------------------------------------


def _artifact_rows(args: argparse.Namespace, artifact_id: str = ""):
    from rosclaw.agentd.mission.store import MissionStore

    home = _home(args)
    db_path = home / "agentd" / "missions.db"
    if not db_path.exists():
        return None, []
    store = MissionStore(db_path)
    conn = store.connection
    if artifact_id:
        rows = conn.execute(
            "SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,),
        ).fetchall()
    else:
        task_id = str(getattr(args, "task", "") or "")
        if task_id:
            rows = conn.execute(
                "SELECT * FROM artifacts WHERE task_id = ? "
                "ORDER BY created_at DESC LIMIT 100",
                (task_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM artifacts ORDER BY created_at DESC LIMIT 100",
            ).fetchall()
    return conn, [dict(r) for r in rows]


def _artifact_view(row: dict) -> dict:
    from rosclaw.task_kernel.deliverables import artifact_delivery_kind

    raw_digest = str(row["sha256"])
    return {
        "artifact_id": str(row["artifact_id"]),
        "task_id": str(row["task_id"]),
        "kind": artifact_delivery_kind(row),
        "media_type": str(row["media_type"]),
        "path": str(row["path"]),
        "size_bytes": int(row["size_bytes"]),
        "digest": (
            raw_digest
            if raw_digest.startswith("sha256:")
            else f"sha256:{raw_digest}"
        ),
        "open_command": f"rosclaw artifact open {row['artifact_id']}",
    }


def cmd_artifact_list(args: argparse.Namespace) -> int:
    conn, rows = _artifact_rows(args)
    if conn is None:
        print("还没有产物账本（agentd 未初始化）。")
        return 0
    views = [_artifact_view(r) for r in rows]
    if getattr(args, "json", False):
        print(json.dumps(views, ensure_ascii=False, indent=2))
        return 0
    if not views:
        print("还没有登记的交付物。")
        return 0
    for v in views:
        print(
            f"  {v['artifact_id']:<28} {v['kind']:<12} {v['media_type']:<16}"
            f" {v['size_bytes']:>9}B  {v['path']}"
        )
    print("\n打开：rosclaw artifact open <id> · 导出：rosclaw artifact export <id> <path>")
    return 0


def cmd_artifact_open(args: argparse.Namespace) -> int:
    conn, rows = _artifact_rows(args, str(args.artifact_id))
    if conn is None or not rows:
        print(f"未知交付物 {args.artifact_id!r}（rosclaw artifact list 可查）")
        return 2
    view = _artifact_view(rows[0])
    path = Path(view["path"])
    if not path.exists():
        print(f"交付物文件缺失：{path}（账本登记于 {view['task_id']}）")
        return 3
    import shutil

    has_display = bool(
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )
    opener = shutil.which("xdg-open")
    if has_display and opener:
        import subprocess

        subprocess.Popen(  # noqa: S603 - 系统 opener，参数无拼接
            [opener, str(path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print(f"已用系统默认程序打开：{path}")
        return 0
    # SSH/纯终端：给可复制路径 + 导出提示（OSC 8 链接是有显示时
    # 的增强，不是依赖）。
    print(f"{path}")
    print(f"（无显示环境——导出查看：rosclaw artifact export {view['artifact_id']} <path>）")
    return 0


def cmd_artifact_show(args: argparse.Namespace) -> int:
    """0901 P0-1：类型/大小/摘要/任务/血缘详情。"""
    conn, rows = _artifact_rows(args, str(args.artifact_id))
    if conn is None or not rows:
        print(f"未知交付物 {args.artifact_id!r}（rosclaw artifact list 可查）")
        return 2
    view = _artifact_view(rows[0])
    meta = rows[0].get("metadata_json")
    lineage = {}
    if isinstance(meta, str):
        import contextlib

        with contextlib.suppress(ValueError):
            lineage = (json.loads(meta) or {}).get("lineage") or {}
    if getattr(args, "json", False):
        print(json.dumps({**view, "lineage": lineage}, ensure_ascii=False, indent=2))
        return 0
    print(f"artifact:  {view['artifact_id']}")
    print(f"kind:      {view['kind']}")
    print(f"media:     {view['media_type']}")
    print(f"size:      {view['size_bytes']}B")
    print(f"digest:    {view['digest']}")
    print(f"task:      {view['task_id']}")
    print(f"path:      {view['path']}")
    if lineage:
        print(f"lineage:   {json.dumps(lineage, ensure_ascii=False)[:200]}")
    return 0


def cmd_artifact_path(args: argparse.Namespace) -> int:
    """0901 P0-1：只输出绝对路径（SSH/脚本一等）。"""
    conn, rows = _artifact_rows(args, str(args.artifact_id))
    if conn is None or not rows:
        print(f"未知交付物 {args.artifact_id!r}", file=sys.stderr)
        return 2
    print(_artifact_view(rows[0])["path"])
    return 0


def cmd_artifact_export(args: argparse.Namespace) -> int:
    import shutil

    conn, rows = _artifact_rows(args, str(args.artifact_id))
    if conn is None or not rows:
        print(f"未知交付物 {args.artifact_id!r}（rosclaw artifact list 可查）")
        return 2
    view = _artifact_view(rows[0])
    src = Path(view["path"])
    if not src.exists():
        print(f"交付物文件缺失：{src}（账本登记于 {view['task_id']}）")
        return 3
    dest = Path(str(args.dest))
    if dest.is_dir():
        dest = dest / src.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"已导出：{dest}（{view['size_bytes']}B，{view['digest'][:19]}…）")
    return 0


def dispatch_artifact_argv(argv: list[str]) -> int | None:
    """0901 P0-1：`rosclaw artifact ...` 入口快路径（entrypoint
    dispatch 链调用）。

    实证事故：artifact list/open/export 只注册在 rosclaw-agentd
    子命令树里，用户面 `rosclaw artifact open <id>` 落到 legacy
    parser 打顶层帮助——TerminalPresenter 给用户的 open_command
    不可达。这里把 artifact 一族直接挂进 `rosclaw` 入口（与
    rosclaw-agentd 同一组 handler——不复制实现）。"""
    if not argv or argv[0] != "artifact":
        return None
    parser = argparse.ArgumentParser(
        prog="rosclaw artifact",
        description="交付物查看/打开/导出",
    )
    sub = parser.add_subparsers(dest="artifact_command", required=True)
    p_list = sub.add_parser("list", help="列出登记的交付物")
    p_list.add_argument("--task", default="", help="按 task_id 过滤")
    p_list.add_argument("--json", action="store_true", help="机器可读输出")
    p_list.set_defaults(func=cmd_artifact_list)
    p_show = sub.add_parser("show", help="交付物详情（类型/大小/任务/血缘）")
    p_show.add_argument("artifact_id")
    p_show.add_argument("--json", action="store_true", help="机器可读输出")
    p_show.set_defaults(func=cmd_artifact_show)
    p_path = sub.add_parser("path", help="只输出绝对路径（SSH/脚本可用）")
    p_path.add_argument("artifact_id")
    p_path.set_defaults(func=cmd_artifact_path)
    p_open = sub.add_parser("open", help="打开交付物（无显示环境给路径）")
    p_open.add_argument("artifact_id")
    p_open.set_defaults(func=cmd_artifact_open)
    p_export = sub.add_parser("export", help="导出交付物到指定路径")
    p_export.add_argument("artifact_id")
    p_export.add_argument("dest")
    p_export.set_defaults(func=cmd_artifact_export)
    args = parser.parse_args(argv[1:])
    # --home 由 _home(args) 统一解析（env ROSCLAW_HOME 回落）。
    return args.func(args)


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
    report = doctor(home, deep=bool(getattr(args, "deep", False)))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    ok_states = {"TOOL_READY", "CHAT_READY", "DEGRADED"}
    return 0 if report.get("status") in ok_states else 1


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
    if not summary.get("configured"):
        print(json.dumps(summary, ensure_ascii=False))
        return 0
    # R0-7（0826 体验审计 §2.7）：默认便宜探测（auth+models+
    # chat——严格 tool call 归 `rosclaw agentd doctor --deep`）。
    report = doctor(home)
    if getattr(args, "json", False):
        print(json.dumps(
            {"configure": summary, "doctor": report},
            ensure_ascii=False, indent=2,
        ))
    else:
        # 默认 ≤6 行人类摘要（不是一百行 JSON）。
        status = str(report.get("status", ""))
        effort = ""
        try:
            settings = json.loads(
                (home / "agent" / "settings.json").read_text(encoding="utf-8")
            )
            effort = str(settings.get("defaultThinkingLevel", ""))
        except (OSError, ValueError):
            effort = ""
        print(f"模型已配置：{summary.get('provider')} / {summary.get('model')}")
        print(f"凭据引用：{summary.get('api_key_ref')}（值只在环境变量）")
        if effort:
            print(f"推理强度：{effort}（/effort 可改）")
        print(f"探测（便宜档）：{status}")
        if status == "TOOL_READY":
            print("对话与工具调用均可用。")
        elif status == "CHAT_READY":
            print("对话可用；工具调用完整探测：`rosclaw agentd doctor --deep`")
        elif status == "DEGRADED":
            print(f"对话可用，工具自检退化——{report.get('reason', '')}")
        else:
            print(f"未就绪：{report.get('reason', status)}")
    ok_states = {"TOOL_READY", "CHAT_READY", "DEGRADED"}
    if report.get("status") not in ok_states:
        print(
            "\n模型尚未通过便宜探测。这是诚实状态，不是假成功；"
            "请检查 key/endpoint 后重试 `rosclaw agent doctor`。",
            file=sys.stderr,
        )
        return 1
    return 0


def _ensure_home_env(home: Path) -> None:
    """agentd 进程导出 ROSCLAW_HOME（0827 复核实证）：此前只传给
    子进程——agentd 自身没有时，ur5e_mcp 的 PlanStore 回落内存
    （PlanRef 生产/消费分裂）或 conformance 把工具对误杀出模型面。
    setdefault：显式 export 的值优先。"""
    os.environ.setdefault("ROSCLAW_HOME", str(home))


def _ensure_home_env(home: Path) -> str | None:
    """agentd 进程导出 ROSCLAW_HOME（0827 复核实证）：此前只传给
    子进程——agentd 自身没有时，ur5e_mcp 的 PlanStore 回落内存
    （PlanRef 生产/消费分裂）或 conformance 把工具对误杀出模型面。

    返回之前的值（None = 此前未设置）——调用方在退出时恢复
    （cmd_chat 是库函数也是进程入口：进程 env 不得永久改写，否则
    同进程后续调用方/测试的 get_rosclaw_home 被劫持——p1a1 测试
    实证：泄漏的 ROSCLAW_HOME 让 35 个 body 测试 split-brain）。"""
    previous = os.environ.get("ROSCLAW_HOME")
    if previous is None:
        os.environ["ROSCLAW_HOME"] = str(home)
    return previous


def _restore_home_env(previous: str | None) -> None:
    """_ensure_home_env 的配对恢复（进程 env 不永久改写）。"""
    if previous is None:
        os.environ.pop("ROSCLAW_HOME", None)
    else:
        os.environ["ROSCLAW_HOME"] = previous


def cmd_chat(args: argparse.Namespace) -> int:
    home = _home(args)
    previous_home_env = _ensure_home_env(home)
    try:
        return _cmd_chat_impl(args, home)
    finally:
        _restore_home_env(previous_home_env)


def _cmd_chat_impl(args: argparse.Namespace, home: Path) -> int:
    # PR-H9：legacy 引擎（Python AgentLoop console）已删除——Native
    # Agent（Harness Backend 主会话）是唯一引擎（ADR-0012）。
    if getattr(args, "legacy", False) or (getattr(args, "engine", None) not in (None, "pi")):
        print(
            "legacy 引擎已随 H9 删除（旧 Python AgentLoop console）——"
            "rosclaw chat 即 Native Agent，无需 --engine/--legacy。",
            file=sys.stderr,
        )
        return 2
    # P1-A1：chat 准入读 Pi 配置单源（agent/settings.json+models.json——
    # chat 引擎实际消费的那份），不再读 config.yaml 模型段。
    from rosclaw.agentd.pi_config import pi_model_configured

    if not pi_model_configured(home):
        print("未配置模型。先运行 `rosclaw setup model`。", file=sys.stderr)
        return 2
    return _chat_pi(home, args)


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
        handler.setFormatter(_logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
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
            "（发布包自带；源码环境先 npm ci && npm run build）。",
            file=sys.stderr,
        )
        return 2
    # P0-5：安装一致性门禁——wheel 与 TS dist 混合构建（0823 报告/
    # 实例漂移事故）直接阻断，不带着不可信的能力声明进会话。
    from rosclaw.version_diag import (
        assert_installation_coherent,
        collect_diagnostics,
        is_stamped_install,
    )

    try:
        # 只约束带构建戳的安装产物；源码 checkout（无戳）不阻断
        # ——开发树 python/TS 各自演进是常态。
        if is_stamped_install():
            assert_installation_coherent(collect_diagnostics(home=home))
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 2
    node, entry = runtime
    config = load_agent_config(home / "config.yaml")
    # P0-7（0827 审计·真实 K3 复验实证）：retry 预算只在 setup 写
    # 不够——手工/legacy 配置的家目录绕过 setup（pi 默认 3 次重试
    # 把确定性 403 烧 4 次）。chat 启动幂等补齐（保留其他键）。
    from rosclaw.agentd.onboarding import _write_retry_budget

    _write_retry_budget(home)
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
        # WP-P0-1：裸 --resume → 会话选择器；P1-A5：查询原样传 Pi
        # 入口（TS 单份解析），歧义/未命中由 TS 侧诚实报错。
        resume_argv = ["--browse-sessions"] if query == "__picker__" else ["--resume", query]
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
    effective_mode = mission.mode.value if mission is not None else resume_mode
    profile = (
        "robot" if effective_mode is not None and effective_mode != "SIMULATION" else "developer"
    )
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
                    sys.executable,
                    "-m",
                    "rosclaw.entrypoint",
                    "operatord",
                    "start",
                    "--no-human-presence-check",
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
        "topic",
        nargs="?",
        default="",
        help="主题探测：simulation = 托管仿真 runtime（无需模型凭据）",
    )
    p_doctor.add_argument(
        "--deep",
        action="store_true",
        help="完整探测（严格 tool call——可能产生一次模型请求）",
    )
    p_doctor.set_defaults(func=cmd_doctor)

    p_init = sub.add_parser("init", help="configure model provider + probe")
    p_init.add_argument("--provider", choices=PROVIDER_CHOICES, default=None)
    p_init.add_argument("--base-url", default=None)
    p_init.add_argument("--model", default=None)
    p_init.add_argument("--api-key-ref", default=None)
    p_init.add_argument("--json", action="store_true", help="完整结构化报告")
    p_init.set_defaults(func=cmd_init)

    p_chat = sub.add_parser("chat", help="interactive chat (in-process)")
    p_chat.add_argument("--mission", default=None)
    p_chat.add_argument("--mode", default=None, choices=["SIMULATION", "SHADOW", "REAL"])
    p_chat.add_argument("--goal", default=None)
    p_chat.set_defaults(func=cmd_chat)

    p_art = sub.add_parser("artifact", help="交付物查看/打开/导出（R0-4）")
    art_sub = p_art.add_subparsers(dest="artifact_command", required=True)
    p_al = art_sub.add_parser("list", help="列出登记的交付物")
    p_al.add_argument("--task", default="", help="按 task_id 过滤")
    p_al.add_argument("--json", action="store_true", help="机器可读输出")
    p_al.set_defaults(func=cmd_artifact_list)
    p_ao = art_sub.add_parser("open", help="打开交付物（无显示环境给路径）")
    p_ao.add_argument("artifact_id")
    p_ao.set_defaults(func=cmd_artifact_open)
    p_ae = art_sub.add_parser("export", help="导出交付物到指定路径")
    p_ae.add_argument("artifact_id")
    p_ae.add_argument("dest")
    p_ae.set_defaults(func=cmd_artifact_export)

    return parser


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
                "topic",
                nargs="?",
                default="",
                help="主题探测：simulation = 托管仿真 runtime（无需模型凭据）",
            )
        p.set_defaults(func=fn)

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

    # R0-4：交付物用户可达面（SSH/纯终端一等）。
    p_art = subparsers.add_parser("artifact", help="交付物查看/打开/导出")
    art_sub = p_art.add_subparsers(dest="artifact_command", required=True)
    p_al = art_sub.add_parser("list", help="列出登记的交付物")
    p_al.add_argument("--task", default="", help="按 task_id 过滤")
    p_al.add_argument("--json", action="store_true", help="机器可读输出")
    p_al.set_defaults(func=cmd_artifact_list)
    p_ao = art_sub.add_parser("open", help="打开交付物（无显示环境给路径）")
    p_ao.add_argument("artifact_id")
    p_ao.set_defaults(func=cmd_artifact_open)
    p_ae = art_sub.add_parser("export", help="导出交付物到指定路径")
    p_ae.add_argument("artifact_id")
    p_ae.add_argument("dest")
    p_ae.set_defaults(func=cmd_artifact_export)

    p_chat = subparsers.add_parser("chat", help="chat with the Native Agent")
    # 十一审 PR-D：rosclaw chat [PATH]——Project workspace 一等状态。
    p_chat.add_argument(
        "path", nargs="?", default=None, help="项目目录（git 仓库自动绑定为 workspace）"
    )
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
        # 唯一前缀/标题（由 Pi 入口 TS 单份解析——P1-A5）。
        help="恢复会话：无参数打开选择器；或给 ID/唯一前缀/标题",
    )
    p_chat.set_defaults(func=cmd_chat)

    # WP-P0-1（总纲 §4.2/§5.1）：会话可发现性——用户不再需要知道
    # 内部 session id。
    p_sessions = subparsers.add_parser("sessions", help="列出/搜索会话（TTY 下引导选择器）")
    p_sessions.add_argument("query", nargs="?", default="", help="搜索标题/内容")
    p_sessions.set_defaults(func=cmd_sessions)
    p_resume = subparsers.add_parser("resume", help="恢复会话：无参数打开选择器；或给 ID/前缀/标题")
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
