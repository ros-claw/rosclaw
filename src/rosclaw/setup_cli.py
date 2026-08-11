"""`rosclaw setup` 统一向导出（三审 P0-CLI-01）。

此前 root help 把 setup 列为首要入口并承诺 "Configure model, body and
integrations"，但 parser 里只有 `setup lerobot`——公开命令契约失真。

本模块是 setup 的唯一事实源（typed command spec → help/dispatch 同源）：

```text
rosclaw setup                          # 状态总览 + 引导（幂等、可重入）
rosclaw setup status [--json]
rosclaw setup model [--provider ...]   # 复用 agentd onboarding
rosclaw setup body                     # 复用 body resolver/registry
rosclaw setup operator                 # 复用 operatord enroll/status
rosclaw setup worker [codex|claude-code|hermes]  # worker pack 探测
rosclaw setup integration lerobot      # 复用 legacy setup lerobot
```

实现原则（复核 §P0-CLI-01）：只封装已有实现——不重造 provider/body/
worker 层；幂等（重复运行不覆盖已完成配置）；TTY 与 --json 均有
稳定 schema 与退出码。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

WORKER_PACKS = ("codex", "claude-code", "hermes")
INTEGRATIONS = ("lerobot",)


def _home() -> Path:
    return Path(os.environ.get("ROSCLAW_HOME", Path.home() / ".rosclaw"))


# ---------------------------------------------------------------- 状态探测


def _model_status(home: Path) -> dict:
    """模型配置状态（复用 agentd onboarding 的探测，不重新实现）。"""
    try:
        from rosclaw.agentd.onboarding import doctor

        report = doctor(home)
        model = report.get("model", {})
        ready = report.get("status") == "READY"
        return {
            "state": "READY" if ready else "NEEDS_SETUP",
            "provider": model.get("provider", ""),
            "model": model.get("model", ""),
            "detail": "ready" if ready else str(report.get("status", "unknown")),
        }
    except Exception as exc:  # noqa: BLE001 - 状态探测不崩向导
        return {"state": "NEEDS_SETUP", "detail": f"probe failed: {exc}"}


def _body_status(home: Path) -> dict:
    try:
        from rosclaw.body.resolver import BodyResolver

        body = BodyResolver().get_effective_body()
        if body is None:
            return {"state": "NEEDS_SETUP", "detail": "no body linked (sim default available)"}
        return {"state": "READY", "body_id": getattr(body, "body_id", ""), "detail": "linked"}
    except Exception as exc:  # noqa: BLE001
        return {"state": "NEEDS_SETUP", "detail": str(exc)[:120]}


def _operator_status(home: Path) -> dict:
    try:
        from rosclaw.operatord.enrollment import load_identity

        identity = load_identity(home / "operatord")
        sock = home / "run" / "operatord.sock"
        running = sock.exists()
        return {
            "state": "READY" if running else "NEEDS_SETUP",
            "enrollment_id": identity.enrollment_id,
            "detail": "enrolled + running" if running else "enrolled, not running",
        }
    except Exception:  # noqa: BLE001 - 未 enroll 即 NEEDS_SETUP
        return {"state": "NEEDS_SETUP", "detail": "not enrolled"}


def _worker_status() -> dict:
    try:
        from rosclaw.agentd.workers.packs import ALL_PACKS

        packs = []
        for pack in ALL_PACKS:
            import shutil

            found = shutil.which(pack.executable) is not None
            packs.append(
                {
                    "worker_id": pack.worker_id,
                    "state": "READY" if found else "NEEDS_SETUP",
                    "detail": f"{pack.executable} {'found' if found else 'not found'}",
                }
            )
        ready = sum(1 for p in packs if p["state"] == "READY")
        return {
            "state": "READY" if ready else "NEEDS_SETUP",
            "packs": packs,
            "detail": f"{ready}/{len(packs)} external packs available (native worker always available)",
        }
    except Exception as exc:  # noqa: BLE001
        return {"state": "NEEDS_SETUP", "detail": str(exc)[:120]}


def _integration_status(home: Path) -> dict:
    config = home / "integrations" / "lerobot.yaml"
    return {
        "state": "READY" if config.exists() else "NEEDS_SETUP",
        "detail": "lerobot configured" if config.exists() else "no integrations configured",
    }


def _robot_kit_status(home: Path) -> dict:
    """七审 PR-SEVEN-1.4：robot kit 完整性——setup status/body 必须
    展示它（不再只有 identity linked）。"""
    try:
        from rosclaw.agentd.config import load_agent_config
        from rosclaw.sim.robot_kit import kit_for_body

        config = load_agent_config(home / "config.yaml")
        kit = kit_for_body(config.active_body_id)
        if kit is None:
            return {
                "state": "NEEDS_SETUP",
                "detail": f"no first-party kit for body {config.active_body_id}",
            }
        # manifest 完整 + 模块可导入 = 可激活（服务期会原子装配）。
        import importlib.util

        module_ok = importlib.util.find_spec(kit.executor_module) is not None
        return {
            "state": "READY" if module_ok else "BROKEN",
            "kit_id": kit.kit_id,
            "robot": kit.display_name,
            "detail": (
                f"{kit.display_name} · 动作能力 {len(kit.action_tools)} · "
                f"观测能力 {len(kit.observation_tools)} · executor module "
                f"{'ok' if module_ok else 'MISSING'}"
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {"state": "BROKEN", "detail": str(exc)[:120]}


def _safety_status(home: Path) -> dict:
    safety = home / "agent" / "safety.json"
    if not safety.exists():
        return {"state": "READY", "detail": "sim_policy=auto（默认：安全仿真自动执行）"}
    try:
        policy = json.loads(safety.read_text(encoding="utf-8")).get("sim_policy", "auto")
        return {"state": "READY", "detail": f"sim_policy={policy}"}
    except Exception as exc:  # noqa: BLE001
        return {"state": "NEEDS_SETUP", "detail": str(exc)[:120]}


def _language_status(home: Path) -> dict:
    locale_file = home / "agent" / "locale.json"
    if not locale_file.exists():
        return {"state": "READY", "detail": "auto（跟随系统/用户语言）"}
    try:
        data = json.loads(locale_file.read_text(encoding="utf-8"))
        return {"state": "READY", "detail": f"ui={data.get('ui_locale', 'auto')}"}
    except Exception as exc:  # noqa: BLE001
        return {"state": "NEEDS_SETUP", "detail": str(exc)[:120]}


def _collect_status(home: Path) -> dict:
    return {
        "schema_version": "rosclaw.setup.status.v1",
        "model": _model_status(home),
        "body": _body_status(home),
        "robot_kit": _robot_kit_status(home),
        "operator": _operator_status(home),
        "safety": _safety_status(home),
        "language": _language_status(home),
        "worker": _worker_status(),
        "integration": _integration_status(home),
    }


# ---------------------------------------------------------------- 子命令


def _cmd_status(args: argparse.Namespace) -> int:
    home = _home()
    status = _collect_status(home)
    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0
    print("ROSClaw 设置状态：")
    for area, info in status.items():
        if area == "schema_version":
            continue
        state = info.get("state", "?")
        detail = info.get("detail", "")
        marker = "✓" if state == "READY" else "○"
        print(f"  {marker} {area:12} {state:12} {detail}")
    needs = [
        k for k, v in status.items()
        if k != "schema_version" and v.get("state") != "READY"
    ]
    if needs:
        print(f"\n未完成：{', '.join(needs)}——运行 `rosclaw setup <area>` 配置。")
    else:
        print("\n全部就绪。")
    return 0


def _cmd_model(args: argparse.Namespace) -> int:
    """复用 agentd onboarding.configure_model + doctor（不重造 provider 层）。"""
    from rosclaw.agentd.cli import cmd_init

    init_args = argparse.Namespace(
        home=str(_home()),
        provider=args.provider,
        base_url=args.base_url,
        model=args.model,
        api_key_ref=args.api_key_ref,
    )
    return cmd_init(init_args)


def _cmd_body(args: argparse.Namespace) -> int:
    home = _home()
    status = _body_status(home)
    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0
    if status["state"] == "READY":
        print(f"Body 已链接：{status.get('body_id', '')}")
        print("重新配置：`rosclaw body link <body-id>`（body 子命令族）。")
        return 0
    print("未链接实体 Body——SIM 默认（sim/ur5e）可用于仿真。")
    print("链接真实 Body：`rosclaw body link <body-id>`；发现可用 Body：`rosclaw body list`。")
    return 1


def _cmd_operator(args: argparse.Namespace) -> int:
    home = _home()
    status = _operator_status(home)
    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0 if status["state"] == "READY" else 1
    if status["state"] == "READY":
        print(f"Operator 已就绪：{status.get('enrollment_id', '')}（运行中）")
        return 0
    if "not enrolled" in status.get("detail", ""):
        print("Operator 未登记。登记（生成 Ed25519 身份，私钥 0600 仅存本机）：")
        from rosclaw.operatord.enrollment import enroll

        identity = enroll(home / "operatord")
        print(f"  已登记：{identity.enrollment_id}（fingerprint {identity.fingerprint}）")
        print("启动：`rosclaw operatord start`（SIM 测试可加 --no-human-presence-check）")
        return 0
    print("Operator 已登记未运行。启动：`rosclaw operatord start`")
    return 1


def _cmd_worker(args: argparse.Namespace) -> int:
    status = _worker_status()
    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0
    print("Worker 状态（native worker 始终可用；外部 pack 按需）：")
    for pack in status.get("packs", []):
        marker = "✓" if pack["state"] == "READY" else "○"
        print(f"  {marker} {pack['worker_id']:28} {pack['detail']}")
    target = args.worker_name
    if target:
        matched = [p for p in status.get("packs", []) if target in p["worker_id"]]
        if not matched:
            print(f"未知 worker pack：{target}（可选 {', '.join(WORKER_PACKS)}）", file=sys.stderr)
            return 2
        pack = matched[0]
        if pack["state"] != "READY":
            print(f"\n{pack['worker_id']} 未就绪：{pack['detail']}")
            print("安装对应 CLI 后重跑 `rosclaw setup worker` 探测。")
            return 1
        print(f"\n{pack['worker_id']} 就绪。")
    return 0


def _cmd_integration(args: argparse.Namespace) -> int:
    if args.integration_name == "lerobot":
        # 复用 legacy setup lerobot（保留一个版本的兼容路径）。
        from rosclaw.cli import main as legacy_main

        sys.argv = ["rosclaw", "setup", "lerobot", *sys.argv[sys.argv.index("lerobot") + 1:]]
        return legacy_main()
    print(f"未知 integration：{args.integration_name}（可选 {', '.join(INTEGRATIONS)}）",
          file=sys.stderr)
    return 2


def _cmd_wizard(args: argparse.Namespace) -> int:
    """裸 `rosclaw setup`：状态总览 + 下一步指引（幂等、可重入）。"""
    args.json = getattr(args, "json", False)
    return _cmd_status(args)


# ---------------------------------------------------------------- dispatch

_HELP = """rosclaw setup — 统一设置向导

用法：
  rosclaw setup                    状态总览 + 下一步指引
  rosclaw setup status [--json]    机器可读状态
  rosclaw setup model              配置模型提供方（Kimi/OpenAI 兼容/本地）
  rosclaw setup body               查看/引导 Body 链接
  rosclaw setup operator           登记/启动独立授权进程
  rosclaw setup safety [POLICY]    SIM 审批策略（auto|ask-every-time）
  rosclaw setup language [LOCALE]  UI 语言（zh-CN|en-US|auto）
  rosclaw setup worker [NAME]      探测外部 Worker（codex/claude-code/hermes）
  rosclaw setup integration lerobot  配置 LeRobot 集成
  rosclaw setup demo               运行第一个可验证仿真任务（无需模型）
"""


def _cmd_safety(args: argparse.Namespace) -> int:
    """setup safety——SIM 审批策略（auto=安全仿真自动执行 /
    ask-every-time=每次人工确认；REAL 永远人工）。"""
    home = _home()
    safety = home / "agent" / "safety.json"
    value = getattr(args, "policy", None)
    if value is None:
        current = "auto"
        if safety.exists():
            try:
                current = json.loads(safety.read_text(encoding="utf-8")).get(
                    "sim_policy", "auto"
                )
            except Exception:  # noqa: BLE001
                current = "auto"
        print(f"SIM 审批策略：{current}（auto=安全仿真自动执行 / "
              "ask=每次人工确认；REAL 永远人工确认）")
        return 0
    mapping = {"auto": "auto", "ask-every-time": "ask", "ask": "ask"}
    if value not in mapping:
        print(f"未知策略 {value!r}（auto|ask-every-time）", file=sys.stderr)
        return 2
    safety.parent.mkdir(parents=True, exist_ok=True)
    tmp = safety.with_suffix(".tmp")
    tmp.write_text(json.dumps({"sim_policy": mapping[value]}, indent=1), encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(safety)
    print(f"SIM 审批策略已更新：{mapping[value]}")
    return 0


def _cmd_language(args: argparse.Namespace) -> int:
    """setup language——UI 语言（zh-CN|en-US|auto）。"""
    home = _home()
    locale_file = home / "agent" / "locale.json"
    value = getattr(args, "locale", None)
    if value is None:
        current = "auto"
        if locale_file.exists():
            try:
                current = json.loads(locale_file.read_text(encoding="utf-8")).get(
                    "ui_locale", "auto"
                )
            except Exception:  # noqa: BLE001
                current = "auto"
        print(f"UI 语言：{current}（zh-CN|en-US|auto）")
        return 0
    aliases = {"zh-CN": "zh-CN", "中文": "zh-CN", "en-US": "en-US",
               "English": "en-US", "auto": "auto"}
    if value not in aliases:
        print(f"未知语言 {value!r}（zh-CN|en-US|auto）", file=sys.stderr)
        return 2
    locale_file.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if locale_file.exists():
        try:
            existing = json.loads(locale_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            existing = {}
    existing["ui_locale"] = aliases[value]
    tmp = locale_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(existing, indent=1), encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(locale_file)
    print(f"UI 语言已更新：{aliases[value]}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    """setup demo——第一个可验证仿真任务（内核直跑 draw_shape，
    无需模型）：验证安装能完成'规划→策略→执行→验证'全链。"""
    import asyncio

    home = _home()
    # 注意：import 必须在 asyncio.run 之前完成——在运行中的事件循环
    # 里首次 import agentd/mcp 模块会触发 heisenbug（实测：discovery
    # 静默失败 → catalog 空 → planning 失败；提前 import 则稳定）。
    from rosclaw.agentd.config import load_agent_config
    from rosclaw.agentd.models.gateway import MockModelGateway
    from rosclaw.agentd.models.profiles import mock_profile
    from rosclaw.agentd.pi_bridge.action_admission import ActionRequestContext
    from rosclaw.agentd.pi_bridge.context import build_embodied_context
    from rosclaw.agentd.pi_bridge.context_lease import (
        ContextLeaseStore,
        context_hash_of,
    )
    from rosclaw.agentd.pi_bridge.session_binding import SessionBindingStore
    from rosclaw.agentd.service import AgentService
    from rosclaw.agentd.task_runner import TaskRunner

    async def _run() -> dict:
        config = load_agent_config(home / "config.yaml")
        # demo 验证的是内核管道（规划→策略→执行→验证），不是模型——
        # 用 mock gateway 使无模型配置的全新安装也能跑通（诚实标注）。
        service = AgentService(
            config, home, gateway=MockModelGateway(mock_profile(), [])
        )
        try:
            mission = service.create_mission("setup demo: draw star")
            await service._ensure_mcp_discovered()

            bindings = SessionBindingStore(service._store.connection)
            bindings.bind(
                pi_session_id="setup_demo", pi_session_path="",
                mission_id=mission.mission_id,
                body_id=mission.body_binding.body_id,
                execution_mode=mission.mode.value,
                created_by="user:local:1000",
            )
            import os as _os

            lease, _token = bindings.acquire_lease(
                mission_id=mission.mission_id, pi_session_id="setup_demo",
                owner_pid=_os.getpid(), owner_uid=_os.getuid(),
            )
            envelope = build_embodied_context(service, mission.mission_id)
            lease_id = ContextLeaseStore(service._store.connection).issue(
                pi_session_id="setup_demo", mission_id=mission.mission_id,
                context_revision=envelope.context_revision,
                context_hash=context_hash_of(envelope),
                body_hash=mission.body_binding.effective_body_hash,
                mode=mission.mode.value,
                binding_id=bindings.binding_for_session("setup_demo").binding_id,
                writer_lease_id=lease.lease_id,
                caller_uid=_os.getuid(), caller_pid=_os.getpid(),
            ).context_lease_id
            snapshot = service.snapshot(mission.mission_id)
            ctx = ActionRequestContext(
                pi_session_id="setup_demo", mission_id=mission.mission_id,
                context_revision=snapshot.context_revision,
                body_hash=mission.body_binding.effective_body_hash,
                mode=mission.mode.value,
                idempotency_key="setup_demo_draw_shape",
                context_lease_id=lease_id,
            )
            return await TaskRunner(service).run(
                request_ctx=ctx, goal="draw_shape",
                parameters={"shape": "star5", "center_m": [0.35, 0.25, 0.30],
                            "radius_m": 0.10},
                caller_pid=_os.getpid(), caller_uid=_os.getuid(),
            )
        finally:
            await service.close()

    result = asyncio.run(_run())
    print(f"demo task {result['task_id']}: {result['state']}")
    print(f"  证据等级：{result.get('evidence_level', '?')}——{result.get('user_view', '')}")
    verification = result.get("verification") or {}
    if verification:
        print(f"  几何验证：{verification.get('verdict')} "
              f"(rmse={verification.get('rmse_m')}, closure={verification.get('closure_error_m')})")
    return 0 if result["state"] == "VERIFIED" else 1


def dispatch_setup_argv(argv: list[str]) -> int | None:
    """`setup` 子命令族分发。未命中返回 None（交给 legacy parser）。"""
    if not argv or argv[0] != "setup":
        return None
    # 兼容：`setup lerobot` 直达 integration（一个版本的隐藏 alias）。
    if len(argv) >= 2 and argv[1] == "lerobot":
        return _cmd_integration(argparse.Namespace(integration_name="lerobot"))
    if len(argv) == 1:
        return _cmd_wizard(argparse.Namespace(json=False))
    sub = argv[1]
    rest = argv[2:]
    if sub in ("-h", "--help", "help"):
        print(_HELP)
        return 0
    if sub == "status":
        parser = argparse.ArgumentParser(prog="rosclaw setup status")
        parser.add_argument("--json", action="store_true")
        return _cmd_status(parser.parse_args(rest))
    if sub == "model":
        parser = argparse.ArgumentParser(prog="rosclaw setup model")
        parser.add_argument("--provider", default=None)
        parser.add_argument("--base-url", default=None)
        parser.add_argument("--model", default=None)
        parser.add_argument("--api-key-ref", default=None)
        return _cmd_model(parser.parse_args(rest))
    if sub == "body":
        parser = argparse.ArgumentParser(prog="rosclaw setup body")
        parser.add_argument("--json", action="store_true")
        return _cmd_body(parser.parse_args(rest))
    if sub == "operator":
        parser = argparse.ArgumentParser(prog="rosclaw setup operator")
        parser.add_argument("--json", action="store_true")
        return _cmd_operator(parser.parse_args(rest))
    if sub == "worker":
        parser = argparse.ArgumentParser(prog="rosclaw setup worker")
        parser.add_argument("worker_name", nargs="?", default=None)
        parser.add_argument("--json", action="store_true")
        return _cmd_worker(parser.parse_args(rest))
    if sub == "safety":
        parser = argparse.ArgumentParser(prog="rosclaw setup safety")
        parser.add_argument("policy", nargs="?", default=None)
        return _cmd_safety(parser.parse_args(rest))
    if sub == "language":
        parser = argparse.ArgumentParser(prog="rosclaw setup language")
        parser.add_argument("locale", nargs="?", default=None)
        return _cmd_language(parser.parse_args(rest))
    if sub == "demo":
        return _cmd_demo(argparse.Namespace())
    if sub == "integration":
        parser = argparse.ArgumentParser(prog="rosclaw setup integration")
        parser.add_argument("integration_name", choices=list(INTEGRATIONS))
        return _cmd_integration(parser.parse_args(rest))
    print(f"未知 setup 子命令：{sub}\n\n{_HELP}", file=sys.stderr)
    return 2
