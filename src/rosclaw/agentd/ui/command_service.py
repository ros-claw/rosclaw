"""CommandService (批次 B，补充实施文档 §5.1/§5.2)。

命令的语义源与执行点。规则：

* 命令永远不进入模型上下文——TUI/客户端先解析，命中注册表就走控制 API；
* 未注册的命令返回 unknown_command（客户端可选择是否作为普通文本发送）；
* availability 按 mission state 判定，不可用时给出 disabled_reason；
* 执行幂等：同一 idempotency_key 重复提交返回首次结果；
* SAFETY_CONTROL 的命令不在这里——/approve、/estop 走专用端点。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from rosclaw.contracts.ui.commands import (
    CommandCategory,
    CommandOwner,
    CommandRequestV1,
    CommandResultV1,
    CommandSpecV1,
)

if TYPE_CHECKING:
    from rosclaw.agentd.service import AgentService

HandlerFn = Callable[[CommandRequestV1], Awaitable[CommandResultV1]]

#: 全部内建命令的机器可读参数 schema（审计 P0-03）。type: string|enum|rest
#: （rest 吃掉剩余全部文本）；interaction 指示 TUI 需要的交互形态。
_ARGS_SCHEMAS: dict[str, dict] = {
    "compact": {
        "positional": [{"name": "focus", "type": "rest", "required": False}],
        "flags": {"dry-run": {"type": "boolean"}},
        "interaction": "none",
    },
    "cancel": {"interaction": "none"},
    "rename": {
        "positional": [{"name": "name", "type": "rest", "required": True}],
        "interaction": "none",
    },
    "archive": {"interaction": "confirm"},
    "status": {"interaction": "none"},
    "tools": {"interaction": "none"},
    "providers": {"interaction": "none"},
    "model": {
        "positional": [{"name": "target", "type": "string", "required": False}],
        "interaction": "select",
        "interaction_source": "models",
    },
    "login": {
        "positional": [{"name": "provider", "type": "string", "required": True}],
        "interaction": "secret",
    },
    "logout": {
        "positional": [{"name": "provider", "type": "string", "required": True}],
        "interaction": "confirm",
    },
    "workers": {"interaction": "none"},
    "worker": {
        "positional": [
            {
                "name": "subcommand",
                "type": "enum",
                "enum": ["inspect", "enable", "disable", "probe"],
                "required": True,
            },
            {"name": "worker_id", "type": "string", "required": True},
        ],
        "interaction": "select",
        "interaction_source": "workers",
    },
    "grants": {"interaction": "none"},
    "revoke": {
        "positional": [{"name": "grant_id", "type": "string", "required": True}],
        "interaction": "confirm",
    },
    "body": {"interaction": "none"},
    "doctor": {"interaction": "none"},
    "mode": {
        "positional": [
            {"name": "mode", "type": "enum", "enum": ["SIMULATION", "SHADOW", "REAL"], "required": False}
        ],
        "interaction": "none",
    },
    "context": {
        "positional": [
            {
                "name": "subcommand",
                "type": "enum",
                "enum": ["layers", "usage", "compactions", "refresh"],
                "required": False,
            }
        ],
        "interaction": "none",
    },
    "session": {"interaction": "none"},
    "new": {
        "positional": [{"name": "goal", "type": "rest", "required": True}],
        "interaction": "none",
    },
    "retry": {"interaction": "none"},
    "failover": {"interaction": "none"},
    "thinking": {
        "positional": [
            {"name": "effort", "type": "enum", "enum": ["low", "high", "max"], "required": False}
        ],
        "interaction": "none",
    },
    "scoped-models": {
        "positional": [
            {"name": "subcommand", "type": "enum", "enum": ["add", "remove", "list"], "required": False},
            {"name": "target", "type": "string", "required": False},
        ],
        "interaction": "none",
    },
    "export": {
        "positional": [{"name": "path", "type": "string", "required": False}],
        "interaction": "path",
    },
    "import": {
        "positional": [{"name": "path", "type": "string", "required": True}],
        "interaction": "path",
    },
    "share": {"interaction": "none"},
    "reload": {
        "positional": [{"name": "domains", "type": "rest", "required": False}],
        "interaction": "none",
    },
    "settings": {
        "positional": [
            {"name": "key", "type": "string", "required": False},
            {"name": "value", "type": "rest", "required": False},
        ],
        "interaction": "none",
    },
    "tree": {"interaction": "none"},
    "fork": {
        "positional": [
            {"name": "from_entry_id", "type": "string", "required": False},
            {"name": "label", "type": "rest", "required": False},
        ],
        "interaction": "none",
    },
    "clone": {"interaction": "none"},
}


class CommandService:
    def __init__(self, service: AgentService) -> None:
        self._service = service
        self._handlers: dict[str, HandlerFn] = {}
        self._specs: dict[str, CommandSpecV1] = {}
        self._idempotency: dict[str, CommandResultV1] = {}
        self._register_builtins()
        _register_batch_e(service, self._register)
        _register_batch_e2(service, self._register)
        _register_batch_f(service, self._register)

    # -- registry ----------------------------------------------------------------

    def _register(self, spec: CommandSpecV1, handler: HandlerFn) -> None:
        if not spec.args_schema:
            spec = spec.model_copy(
                update={"args_schema": _ARGS_SCHEMAS.get(spec.name, {"interaction": "none"})}
            )
        self._specs[spec.name] = spec
        self._handlers[spec.handler] = handler

    def specs(self, *, mission_state: str | None = None, turn_in_flight: bool = False) -> list[CommandSpecV1]:
        """All server commands, annotated with availability for this context."""
        out: list[CommandSpecV1] = []
        for spec in sorted(self._specs.values(), key=lambda s: s.name):
            reason = ""
            if spec.availability and mission_state and mission_state not in spec.availability:
                reason = f"当前状态 {mission_state} 不可用（需要 {'/'.join(spec.availability)}）"
            if turn_in_flight and not spec.during_turn:
                reason = reason or "turn 运行中不可用"
            out.append(spec.model_copy(update={"disabled_reason": reason}))
        return out

    def get(self, name: str) -> CommandSpecV1 | None:
        return self._specs.get(name)

    # -- execution ---------------------------------------------------------------

    async def execute(self, request: CommandRequestV1) -> CommandResultV1:
        cached = self._idempotency.get(request.idempotency_key)
        if cached is not None:
            return cached
        spec = self._specs.get(request.command_name)
        if spec is None:
            return CommandResultV1(
                request_id=request.request_id,
                command_name=request.command_name,
                ok=False,
                error_code="unknown_command",
                message=f"未知命令 /{request.command_name}（可作为普通文本发送）",
            )
        mission_state = None
        turn_in_flight = False
        if request.mission_id:
            mission = self._service.get_mission(request.mission_id)
            if mission is None:
                return CommandResultV1(
                    request_id=request.request_id,
                    command_name=request.command_name,
                    ok=False,
                    error_code="unknown_mission",
                    message=f"未知 mission {request.mission_id!r}",
                )
            mission_state = mission.state.value
            turn_in_flight = self._service.turn_in_flight(request.mission_id)
        annotated = self.specs(mission_state=mission_state, turn_in_flight=turn_in_flight)
        current = next(s for s in annotated if s.name == spec.name)
        if current.disabled_reason:
            return CommandResultV1(
                request_id=request.request_id,
                command_name=request.command_name,
                ok=False,
                error_code="command_unavailable",
                message=current.disabled_reason,
            )
        handler = self._handlers[spec.handler]
        result = await handler(request)
        if result.ok:
            self._idempotency[request.idempotency_key] = result
            # 有界缓存（防长期运行膨胀；淘汰最老一半）。
            if len(self._idempotency) > 2000:
                for key in list(self._idempotency)[:1000]:
                    self._idempotency.pop(key, None)
        return result

    # -- builtin commands -----------------------------------------------------------

    def _register_builtins(self) -> None:
        service = self._service

        async def _compact(req: CommandRequestV1) -> CommandResultV1:
            report = await service.compact(
                req.mission_id or "",
                instructions=req.arguments.get("focus"),
                dry_run=bool(req.arguments.get("dry_run", False)),
            )
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=True,
                message=(
                    f"压缩完成：{report.get('tokens_before', 0)} → "
                    f"{report.get('tokens_after', 0)} tokens"
                    + ("（dry-run，未写入）" if report.get("dry_run") else "")
                ),
                data=report,
            )

        async def _cancel(req: CommandRequestV1) -> CommandResultV1:
            await service.cancel(req.mission_id or "")
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=True,
                message="已请求取消当前 turn（已派发动作不受影响）",
            )

        async def _rename(req: CommandRequestV1) -> CommandResultV1:
            name = str(req.arguments.get("name", "")).strip()
            if not name:
                return CommandResultV1(
                    request_id=req.request_id,
                    command_name=req.command_name,
                    ok=False,
                    error_code="invalid_arguments",
                    message="/rename 需要 name 参数",
                )
            service.rename_mission(req.mission_id or "", name)
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=True,
                message=f"已重命名为「{name}」",
            )

        async def _archive(req: CommandRequestV1) -> CommandResultV1:
            service.archive_mission(req.mission_id or "")
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=True,
                message="已归档（只读，不再接受新 turn）",
            )

        async def _status(req: CommandRequestV1) -> CommandResultV1:
            data = service.status_snapshot(req.mission_id)
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=True,
                message="status ok",
                data=data,
            )

        async def _tools(req: CommandRequestV1) -> CommandResultV1:
            await service._ensure_mcp_discovered()
            items = [
                {
                    "tool_id": d.tool_id,
                    "source": d.source,
                    "execution_class": d.execution_class.value,
                    "model_callable": d.model_callable,
                    "quarantined": service.tool_catalog.quarantine_reason(d.tool_id) is not None,
                    "modes": list(d.supported_modes),
                }
                for d in service.tool_catalog.list()
            ]
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=True,
                message=f"{len(items)} 个已注册工具",
                data={"tools": items},
            )

        # -- MODEL_CONTROL（批次 D） -------------------------------------------

        async def _providers(req: CommandRequestV1) -> CommandResultV1:
            data = await service.modeld_providers()
            providers = data.get("providers", [])
            lines = [
                f"{p['id']:14} {p.get('auth', '?'):14} {p.get('name', '')}"
                for p in providers
            ]
            current = service.current_model_label()
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=bool(data.get("available", True)),
                message=(
                    f"当前模型：{current}\n" + "\n".join(lines)
                    if lines
                    else data.get("error", "modeld 不可用")
                ),
                data={"providers": providers, "current": current},
            )

        async def _model(req: CommandRequestV1) -> CommandResultV1:
            target = str(req.arguments.get("target", "")).strip()
            if not target:
                data = await service.modeld_providers()
                models: dict[str, list[str]] = {}
                for p in data.get("providers", []):
                    listing = await service.modeld_models(p["id"])
                    models[p["id"]] = [m["id"] for m in listing.get("models", [])]
                return CommandResultV1(
                    request_id=req.request_id,
                    command_name=req.command_name,
                    ok=True,
                    message=f"当前模型：{service.current_model_label()}",
                    data={"models": models, "current": service.current_model_label()},
                )
            if "/" not in target:
                return CommandResultV1(
                    request_id=req.request_id,
                    command_name=req.command_name,
                    ok=False,
                    error_code="invalid_arguments",
                    message="/model 需要 <provider>/<model>，如 /model kimi-code/k3",
                )
            provider, _, model = target.partition("/")
            result = service.switch_model(provider, model)
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=result.get("ok", False),
                message=result.get("message", ""),
                error_code=result.get("error_code", ""),
            )

        async def _login(req: CommandRequestV1) -> CommandResultV1:
            provider = str(req.arguments.get("provider", "")).strip()
            api_key = str(req.arguments.get("api_key", "")).strip()
            if not provider or not api_key:
                return CommandResultV1(
                    request_id=req.request_id,
                    command_name=req.command_name,
                    ok=False,
                    error_code="invalid_arguments",
                    message="/login 需要 provider 与 api_key（TUI 会用 masked input 收集）",
                )
            result = await service.modeld_login(provider, api_key)
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=bool(result.get("ok")),
                message=(
                    f"已登录 {provider}（凭据存于 modeld credential store；不会进入对话）"
                    if result.get("ok")
                    else str(result.get("error", "login failed"))
                ),
            )

        async def _logout(req: CommandRequestV1) -> CommandResultV1:
            provider = str(req.arguments.get("provider", "")).strip()
            if not provider:
                return CommandResultV1(
                    request_id=req.request_id,
                    command_name=req.command_name,
                    ok=False,
                    error_code="invalid_arguments",
                    message="/logout 需要 provider",
                )
            result = await service.modeld_logout(provider)
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=bool(result.get("ok", True)),
                message=f"已登出 {provider}（活动 turn 不受影响，下一 turn 生效）",
            )

        self._register(
            CommandSpecV1(
                name="providers",
                description="列出 Provider 与认证状态（不含 secret）",
                category=CommandCategory.MODEL,
                owner=CommandOwner.MODEL_CONTROL,
                during_turn=True,
                handler="model.providers",
            ),
            _providers,
        )
        self._register(
            CommandSpecV1(
                name="model",
                description="查看或切换模型（切换不改变工具权限/Mission mode/grant）",
                argument_hint="[<provider>/<model>]",
                category=CommandCategory.MODEL,
                owner=CommandOwner.MODEL_CONTROL,
                mutability="CONTROL_STATE",
                handler="model.select",
            ),
            _model,
        )
        self._register(
            CommandSpecV1(
                name="login",
                description="Provider API key 登录（secret 不进对话）",
                argument_hint="<provider>",
                category=CommandCategory.MODEL,
                owner=CommandOwner.MODEL_CONTROL,
                mutability="PERSISTED",
                handler="model.login",
            ),
            _login,
        )
        self._register(
            CommandSpecV1(
                name="logout",
                description="Provider 登出",
                argument_hint="<provider>",
                category=CommandCategory.MODEL,
                owner=CommandOwner.MODEL_CONTROL,
                mutability="PERSISTED",
                confirmation="CONFIRM",
                handler="model.logout",
            ),
            _logout,
        )

        self._register(
            CommandSpecV1(
                name="compact",
                description="压缩会话历史（canonical journal 保留）",
                argument_hint="[focus <文字> | dry-run]",
                category=CommandCategory.MISSION,
                owner=CommandOwner.MISSION_CONTROL,
                mutability="PERSISTED",
                handler="mission.compact",
            ),
            _compact,
        )
        self._register(
            CommandSpecV1(
                name="cancel",
                description="请求停止当前 turn（不同于物理急停）",
                category=CommandCategory.EXECUTION,
                owner=CommandOwner.AGENT_CONTROL,
                during_turn=True,
                handler="turn.cancel",
            ),
            _cancel,
        )
        self._register(
            CommandSpecV1(
                name="rename",
                description="修改 Mission 显示名（不改变 goal）",
                argument_hint="<name>",
                category=CommandCategory.MISSION,
                owner=CommandOwner.MISSION_CONTROL,
                mutability="PERSISTED",
                handler="mission.rename",
            ),
            _rename,
        )
        self._register(
            CommandSpecV1(
                name="archive",
                description="归档 Mission（只读）",
                category=CommandCategory.MISSION,
                owner=CommandOwner.MISSION_CONTROL,
                mutability="PERSISTED",
                confirmation="CONFIRM",
                handler="mission.archive",
            ),
            _archive,
        )
        self._register(
            CommandSpecV1(
                name="status",
                description="AgentService/model/Worker/授权状态总览",
                category=CommandCategory.HELP_UI,
                owner=CommandOwner.AGENT_CONTROL,
                during_turn=True,
                handler="agent.status",
            ),
            _status,
        )
        self._register(
            CommandSpecV1(
                name="tools",
                description="工具目录：分类、来源、可用性、健康状态",
                category=CommandCategory.EXECUTION,
                owner=CommandOwner.AGENT_CONTROL,
                during_turn=True,
                handler="agent.tools",
            ),
            _tools,
        )


# ---------------------------------------------------------------------------
# 批次 E：执行与安全可见性命令（workers/grants/body/doctor/mode/context/
# session/new/retry/failover/thinking/scoped-models）
# ---------------------------------------------------------------------------
def _register_batch_f(service, register) -> None:
    """批次 F 第一阶段：/tree 只读、/fork 开新 SIMULATION mission。"""

    async def _tree(req: CommandRequestV1) -> CommandResultV1:
        try:
            tree = service.branches.tree(req.mission_id or "")
        except Exception as exc:  # noqa: BLE001
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=False,
                error_code="unknown_mission",
                message=str(exc),
            )
        return CommandResultV1(
            request_id=req.request_id,
            command_name=req.command_name,
            ok=True,
            message=(
                f"推理分支 {len(tree['reasoning_branches'])} 条；"
                f"物理事实线 {len(tree['physical_lane'])} 个事件（不可回滚）"
            ),
            data=tree,
        )

    async def _fork(req: CommandRequestV1) -> CommandResultV1:
        try:
            branch = service.branches.fork(
                req.mission_id or "",
                from_entry_id=req.arguments.get("from_entry_id") or None,
                label=str(req.arguments.get("label", "")),
            )
        except Exception as exc:  # noqa: BLE001
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=False,
                error_code="fork_refused",
                message=str(exc),
            )
        return CommandResultV1(
            request_id=req.request_id,
            command_name=req.command_name,
            ok=True,
            message=(
                f"已 fork 为新 SIMULATION mission {branch.forked_mission_id}；"
                "authority 未复制（无 grant/approval/Permit/lease），"
                "首轮编译注入最新 Body/Self。"
            ),
            data=branch.model_dump(mode="json"),
        )

    async def _clone(req: CommandRequestV1) -> CommandResultV1:
        return CommandResultV1(
            request_id=req.request_id,
            command_name=req.command_name,
            ok=False,
            error_code="not_implemented",
            message="/clone 在批次 F 第二阶段提供；/fork 已可创建推理分支（不复制 authority）。",
        )

    register(
        CommandSpecV1(
            name="tree",
            description="推理分支树 + 不可变物理事实线（只读）",
            category=CommandCategory.MISSION,
            owner=CommandOwner.MISSION_CONTROL,
            during_turn=True,
            handler="branch.tree",
        ),
        _tree,
    )
    register(
        CommandSpecV1(
            name="fork",
            description="从当前/指定 entry 分叉为新 SIMULATION mission（不复制 authority）",
            argument_hint="[from_entry_id] [label]",
            category=CommandCategory.MISSION,
            owner=CommandOwner.MISSION_CONTROL,
            mutability="PERSISTED",
            handler="branch.fork",
        ),
        _fork,
    )
    register(
        CommandSpecV1(
            name="clone",
            description="克隆 Mission（第二阶段）",
            category=CommandCategory.MISSION,
            owner=CommandOwner.MISSION_CONTROL,
            handler="branch.clone",
        ),
        _clone,
    )


def _register_batch_e(service, register) -> None:  # noqa: C901 - 命令集合体
    """注册批次 E 命令。register(spec, handler) 与 CommandService._register 同签名。"""

    async def _workers(req: CommandRequestV1) -> CommandResultV1:
        rows = []
        for card in service._registry.list():
            status = service._registry.status_of(card.worker_id) or "UNKNOWN"
            rows.append(
                {
                    "worker_id": card.worker_id,
                    "kind": card.kind.value,
                    "status": status,
                    "trust": card.trust.initial_level,
                    "capabilities": [c.name for c in card.capabilities],
                    "active_orders": len(
                        service._worker_manager.active_orders_for_worker(card.worker_id)
                    ),
                }
            )
        return CommandResultV1(
            request_id=req.request_id,
            command_name=req.command_name,
            ok=True,
            message=f"{len(rows)} 个 worker",
            data={"workers": rows},
        )

    async def _worker(req: CommandRequestV1) -> CommandResultV1:
        sub = str(req.arguments.get("subcommand", "")).strip()
        worker_id = str(req.arguments.get("worker_id", "")).strip()
        if not worker_id:
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=False,
                error_code="invalid_arguments",
                message="/worker 需要 worker_id（inspect|enable|disable|probe）",
            )
        card = service._registry.get(worker_id)
        if card is None:
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=False,
                error_code="unknown_worker",
                message=f"未知 worker {worker_id!r}",
            )
        if sub == "inspect":
            data = card.model_dump(mode="json")
            data["registry_status"] = service._registry.status_of(worker_id)
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=True,
                message=f"{worker_id}: {data['registry_status']}",
                data=data,
            )
        if sub == "enable":
            service._registry.set_status(
                worker_id, "ENABLED", actor_id=service.actor_id, reason="operator /worker enable"
            )
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=True,
                message=f"{worker_id} 已启用（写审计事件）",
            )
        if sub == "disable":
            active = service._worker_manager.active_orders_for_worker(worker_id)
            if active:
                return CommandResultV1(
                    request_id=req.request_id,
                    command_name=req.command_name,
                    ok=False,
                    error_code="active_orders",
                    message=(
                        f"{worker_id} 有 {len(active)} 个未终态 WorkOrder；"
                        "请先 /cancel 或等待 drain，不会被静默杀死。"
                    ),
                )
            service._registry.set_status(
                worker_id, "DISABLED", actor_id=service.actor_id, reason="operator /worker disable"
            )
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=True,
                message=f"{worker_id} 已停用（写审计事件）",
            )
        if sub == "probe":
            from rosclaw.agentd.workers.packs import ALL_PACKS

            pack = next((p for p in ALL_PACKS if p.worker_id == worker_id), None)
            if pack is None:
                return CommandResultV1(
                    request_id=req.request_id,
                    command_name=req.command_name,
                    ok=True,
                    message=f"{worker_id} 是内置 worker，无需外部二进制探活。",
                )
            ready, detail = service._probe_pack_sync(
                pack.executable, pack.min_version, pack.install_hint
            )
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=ready,
                message=detail,
                data={"ready": ready},
            )
        return CommandResultV1(
            request_id=req.request_id,
            command_name=req.command_name,
            ok=False,
            error_code="invalid_arguments",
            message=f"未知子命令 {sub!r}（inspect|enable|disable|probe）",
        )

    async def _grants(req: CommandRequestV1) -> CommandResultV1:
        grants = [
            {k: v for k, v in g.items() if k in (
                "grant_id", "principal", "mode", "tier", "risk_ceiling",
                "revoked", "consumed", "expires_at", "public_hash",
            )}
            for g in service.list_grants()
        ]
        return CommandResultV1(
            request_id=req.request_id,
            command_name=req.command_name,
            ok=True,
            message=f"{len(grants)} 个 grant（仅 public scope）",
            data={"grants": grants},
        )

    async def _revoke(req: CommandRequestV1) -> CommandResultV1:
        grant_id = str(req.arguments.get("grant_id", "")).strip()
        if not grant_id:
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=False,
                error_code="invalid_arguments",
                message="/revoke 需要 grant_id（需要 Operator 身份）",
            )
        try:
            service.revoke_grant(grant_id, principal=req.arguments.get("principal", "user:local:1000"))
        except Exception as exc:  # noqa: BLE001
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=False,
                error_code="revoke_failed",
                message=str(exc),
            )
        return CommandResultV1(
            request_id=req.request_id,
            command_name=req.command_name,
            ok=True,
            message=f"grant {grant_id} 已撤销",
        )

    async def _body(req: CommandRequestV1) -> CommandResultV1:
        body = service._body_source.get_body(service._body_id)
        if body is None:
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=False,
                error_code="body_unavailable",
                message="配置的 body 不可用（fail closed 显示，不伪造）",
            )
        return CommandResultV1(
            request_id=req.request_id,
            command_name=req.command_name,
            ok=True,
            message=body.summary,
            data={
                "body_id": body.body_id,
                "effective_body_hash": body.effective_body_hash,
                "calibrated": body.calibrated,
                "issues": list(body.issues),
            },
        )

    async def _doctor(req: CommandRequestV1) -> CommandResultV1:
        from rosclaw.agentd.onboarding import doctor

        report = doctor(service._home)
        status = report.get("status", "UNKNOWN")
        return CommandResultV1(
            request_id=req.request_id,
            command_name=req.command_name,
            ok=True,
            message=status if status == "READY" else f"{status}: {report.get('reason', '')}",
            data=report,
        )

    async def _mode(req: CommandRequestV1) -> CommandResultV1:
        mission = service.get_mission(req.mission_id or "")
        if mission is None:
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=False,
                error_code="unknown_mission",
                message="未知 mission",
            )
        target = str(req.arguments.get("mode", "")).strip()
        if target and target != mission.mode.value:
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=False,
                error_code="mode_change_forbidden",
                message=(
                    f"当前 mode {mission.mode.value} 不能原地升级为 {target}；"
                    "请创建新的 REAL Mission 或走 rebind/authorization 工作流（§8.18）。"
                ),
            )
        return CommandResultV1(
            request_id=req.request_id,
            command_name=req.command_name,
            ok=True,
            message=f"当前 mode: {mission.mode.value}",
            data={"mode": mission.mode.value},
        )

    async def _context(req: CommandRequestV1) -> CommandResultV1:
        sub = str(req.arguments.get("subcommand", "layers")).strip()
        from rosclaw.agentd.context.compaction import CompactionStore

        store = CompactionStore(service.store.connection)
        if sub == "compactions":
            entries = store.list(req.mission_id or "")
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=True,
                message=f"{len(entries)} 条 compaction 记录",
                data={
                    "compactions": [
                        {
                            "compaction_id": e.compaction_id,
                            "reason": e.reason,
                            "tokens_before": e.tokens_before,
                            "tokens_after": e.tokens_after,
                            "covered_span_hash": e.covered_span_hash,
                            "supersedes": e.supersedes,
                            "created_at": e.created_at,
                        }
                        for e in entries
                    ]
                },
            )
        if sub == "refresh":
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=True,
                message="下一轮 turn 将重新编译观测与 ContextCompiler 输入（不代表允许动作）。",
            )
        loop = service._loops.get(req.mission_id or "")
        bundle = getattr(loop, "_current_bundle", None) if loop else None
        if bundle is None:
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=True,
                message="尚未编译上下文（发送第一条消息后可用 /context）。",
                data={"compiled": False},
            )
        layers = {
            name: {
                "hash": getattr(bundle.layers, name).hash[:16],
                "tokens": getattr(bundle.layers, name).token_estimate,
            }
            for name in (
                "constitution", "embodiment", "dynamic_self", "capabilities",
                "mission", "memory", "organization", "safety",
            )
        }
        return CommandResultV1(
            request_id=req.request_id,
            command_name=req.command_name,
            ok=True,
            message=(
                f"context {bundle.context_id} rev{bundle.context_revision}；"
                f"compactions={store.count(req.mission_id or '')}"
            ),
            data={
                "compiled": True,
                "context_id": bundle.context_id,
                "context_revision": bundle.context_revision,
                "layers": layers,
                "compaction_count": store.count(req.mission_id or ""),
            },
        )

    async def _session(req: CommandRequestV1) -> CommandResultV1:
        mission = service.get_mission(req.mission_id or "")
        if mission is None:
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=False,
                error_code="unknown_mission",
                message="未知 mission",
            )
        meta = service.store.mission_meta(mission.mission_id)
        usage = service.mission_usage(mission.mission_id)
        return CommandResultV1(
            request_id=req.request_id,
            command_name=req.command_name,
            ok=True,
            message=f"{mission.mission_id} [{mission.state.value}/{mission.mode.value}]",
            data={
                "mission_id": mission.mission_id,
                "name": meta["display_name"],
                "goal": mission.goal.text,
                "created_at": mission.created_at,
                "updated_at": mission.updated_at,
                "usage": usage,
                "last_event_sequence": service._events.latest_sequence(mission.mission_id),
                "archived": meta["archived"],
            },
        )

    async def _new(req: CommandRequestV1) -> CommandResultV1:
        goal = str(req.arguments.get("goal", "")).strip()
        if not goal:
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=False,
                error_code="invalid_arguments",
                message="/new 需要 goal（默认 SIMULATION）",
            )
        mode = str(req.arguments.get("mode", "SIMULATION"))
        try:
            mission = service.create_mission(goal, mode=mode)
        except Exception as exc:  # noqa: BLE001
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=False,
                error_code="create_failed",
                message=str(exc),
            )
        return CommandResultV1(
            request_id=req.request_id,
            command_name=req.command_name,
            ok=True,
            message=f"已创建 Mission {mission.mission_id} [{mission.mode.value}]",
            data={"mission_id": mission.mission_id, "mode": mission.mode.value},
        )

    async def _retry(req: CommandRequestV1) -> CommandResultV1:
        mission_id = req.mission_id or ""
        open_orders = [
            o for o in service._worker_manager.orders_for_mission(mission_id)
            if o.status not in ("ACCEPTED", "REJECTED", "EXPIRED", "CANCELLED", "FAILED")
        ]
        if open_orders:
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=False,
                error_code="side_effects_pending",
                message=(
                    f"{len(open_orders)} 个 WorkOrder 未终态——/retry 不重放"
                    "（§8.7：先 reconcile，不得简单重放）。"
                ),
            )
        history = service.store.conversation(mission_id)
        last_user = next(
            (m for m in reversed(history)
             if m.get("role") == "user" and not str(m.get("content", "")).startswith("[")),
            None,
        )
        if last_user is None:
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=False,
                error_code="nothing_to_retry",
                message="没有可重试的用户消息。",
            )
        turn_id = await service.submit_turn_v2(mission_id, str(last_user["content"]))
        return CommandResultV1(
            request_id=req.request_id,
            command_name=req.command_name,
            ok=True,
            message=f"已重新提交最后一条用户消息（turn {turn_id}）",
            data={"turn_id": turn_id},
        )

    async def _failover(req: CommandRequestV1) -> CommandResultV1:
        status_fn = getattr(service._gateway, "failover_status", None)
        if status_fn is None:
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=True,
                message="当前 gateway 无 failover 链（单网关直连）。",
            )
        data = status_fn()
        lines = [f"active: {data['active']}"] + [
            f"  {c}" + (
                f"  [cooldown {cd['remaining_sec']}s failures={cd['failures']}]"
                if (cd := data["cooldowns"].get(c)) and cd["in_cooldown"] else ""
            )
            for c in data["candidates"]
        ]
        return CommandResultV1(
            request_id=req.request_id,
            command_name=req.command_name,
            ok=True,
            message="\n".join(lines),
            data=data,
        )

    async def _thinking(req: CommandRequestV1) -> CommandResultV1:
        effort = str(req.arguments.get("effort", "")).strip()
        if effort not in ("low", "high", "max"):
            current = service._gateway.profile.vendor_parameters.get("reasoning_effort", "?")
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=True,
                message=f"当前 reasoning effort: {current}（设置：/thinking low|high|max）",
            )
        profile = service._gateway.profile
        params = dict(profile.vendor_parameters)
        params["reasoning_effort"] = effort
        object.__setattr__(profile, "vendor_parameters", params)
        return CommandResultV1(
            request_id=req.request_id,
            command_name=req.command_name,
            ok=True,
            message=f"reasoning effort → {effort}（下一 turn 生效；Provider 不支持时无效）",
        )

    async def _scoped_models(req: CommandRequestV1) -> CommandResultV1:
        sub = str(req.arguments.get("subcommand", "list")).strip()
        target = str(req.arguments.get("target", "")).strip()
        scoped = service.scoped_models
        if sub == "add" and target:
            scoped.add(target)
        elif sub == "remove" and target:
            scoped.discard(target)
        return CommandResultV1(
            request_id=req.request_id,
            command_name=req.command_name,
            ok=True,
            message=f"scoped models: {sorted(scoped) or '(空)'}",
            data={"scoped_models": sorted(scoped)},
        )

    register(
        CommandSpecV1(
            name="workers",
            description="Worker registry：状态/能力/trust/在途订单",
            category=CommandCategory.EXECUTION,
            owner=CommandOwner.AGENT_CONTROL,
            during_turn=True,
            handler="workers.list",
        ),
        _workers,
    )
    register(
        CommandSpecV1(
            name="worker",
            description="Worker inspect/enable/disable/probe（写审计；disable 先 drain）",
            argument_hint="<inspect|enable|disable|probe> <worker_id>",
            category=CommandCategory.EXECUTION,
            owner=CommandOwner.AGENT_CONTROL,
            mutability="CONTROL_STATE",
            handler="workers.manage",
        ),
        _worker,
    )
    register(
        CommandSpecV1(
            name="grants",
            description="当前 grants（仅 public scope/status）",
            category=CommandCategory.SAFETY,
            owner=CommandOwner.AGENT_CONTROL,
            during_turn=True,
            handler="grants.list",
        ),
        _grants,
    )
    register(
        CommandSpecV1(
            name="revoke",
            description="撤销 grant（需要 Operator 身份）",
            argument_hint="<grant_id>",
            category=CommandCategory.SAFETY,
            owner=CommandOwner.AGENT_CONTROL,
            mutability="PERSISTED",
            confirmation="CONFIRM",
            handler="grants.revoke",
        ),
        _revoke,
    )
    register(
        CommandSpecV1(
            name="body",
            description="EffectiveBody hash/校准/问题（refresh 不授权）",
            category=CommandCategory.EXECUTION,
            owner=CommandOwner.AGENT_CONTROL,
            during_turn=True,
            handler="body.show",
        ),
        _body,
    )
    register(
        CommandSpecV1(
            name="doctor",
            description="agentd/modeld/Provider/MCP/Worker/rosclawd 就绪检查",
            category=CommandCategory.HELP_UI,
            owner=CommandOwner.AGENT_CONTROL,
            during_turn=True,
            handler="agent.doctor",
        ),
        _doctor,
    )
    register(
        CommandSpecV1(
            name="mode",
            description="显示当前 Mission mode（不能原地升级）",
            category=CommandCategory.SAFETY,
            owner=CommandOwner.AGENT_CONTROL,
            during_turn=True,
            handler="mode.show",
        ),
        _mode,
    )
    register(
        CommandSpecV1(
            name="context",
            description="上下文组成：layers/usage/compactions/refresh",
            argument_hint="[layers|usage|compactions|refresh]",
            category=CommandCategory.MISSION,
            owner=CommandOwner.AGENT_CONTROL,
            during_turn=True,
            handler="context.show",
        ),
        _context,
    )
    register(
        CommandSpecV1(
            name="session",
            description="Mission id/name/创建时间/用量/event seq",
            category=CommandCategory.MISSION,
            owner=CommandOwner.AGENT_CONTROL,
            during_turn=True,
            handler="session.show",
        ),
        _session,
    )
    register(
        CommandSpecV1(
            name="new",
            description="创建新 Mission（默认 SIMULATION）",
            argument_hint="<goal>",
            category=CommandCategory.MISSION,
            owner=CommandOwner.MISSION_CONTROL,
            mutability="PERSISTED",
            handler="mission.new",
        ),
        _new,
    )
    register(
        CommandSpecV1(
            name="retry",
            description="重试最后一条无副作用的用户消息（不盲重放）",
            category=CommandCategory.MODEL,
            owner=CommandOwner.AGENT_CONTROL,
            handler="turn.retry",
        ),
        _retry,
    )
    register(
        CommandSpecV1(
            name="failover",
            description="模型候选/冷却/上次错误总览",
            category=CommandCategory.MODEL,
            owner=CommandOwner.MODEL_CONTROL,
            during_turn=True,
            handler="model.failover",
        ),
        _failover,
    )
    register(
        CommandSpecV1(
            name="thinking",
            description="设置公开 reasoning effort（low|high|max）",
            argument_hint="[low|high|max]",
            category=CommandCategory.MODEL,
            owner=CommandOwner.MODEL_CONTROL,
            mutability="CONTROL_STATE",
            handler="model.thinking",
        ),
        _thinking,
    )
    register(
        CommandSpecV1(
            name="scoped-models",
            description="快捷切换模型集合（add/remove/list）",
            argument_hint="[add|remove <provider/model>]",
            category=CommandCategory.MODEL,
            owner=CommandOwner.MODEL_CONTROL,
            mutability="CONTROL_STATE",
            handler="model.scoped",
        ),
        _scoped_models,
    )


# ---------------------------------------------------------------------------
# 批次 E（第二部分）：export/import/share/reload/settings
# ---------------------------------------------------------------------------
def _register_batch_e2(service, register) -> None:
    async def _export(req: CommandRequestV1) -> CommandResultV1:
        from pathlib import Path

        mission_id = req.mission_id or ""
        out = req.arguments.get("path")
        path = Path(out) if out else service._home / "exports" / f"{mission_id}.rcmission"
        try:
            report = service.exporter.export_bundle(mission_id, path)
        except Exception as exc:  # noqa: BLE001
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=False,
                error_code="export_failed",
                message=str(exc),
            )
        return CommandResultV1(
            request_id=req.request_id,
            command_name=req.command_name,
            ok=True,
            message=f"已导出 {report['path']}（{report['bytes']} bytes，已脱敏）",
            data=report,
        )

    async def _import(req: CommandRequestV1) -> CommandResultV1:
        from pathlib import Path

        raw = str(req.arguments.get("path", "")).strip()
        if not raw:
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=False,
                error_code="invalid_arguments",
                message="/import 需要 bundle 路径",
            )
        try:
            result = service.importer.import_bundle(Path(raw))
        except Exception as exc:  # noqa: BLE001
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=False,
                error_code="import_refused",
                message=str(exc),
            )
        return CommandResultV1(
            request_id=req.request_id,
            command_name=req.command_name,
            ok=True,
            message=(
                f"已导入为只读 Mission {result['mission_id']}；"
                "不恢复任何授权/Permit/审批效力。"
            ),
            data=result,
        )

    async def _share(req: CommandRequestV1) -> CommandResultV1:
        return CommandResultV1(
            request_id=req.request_id,
            command_name=req.command_name,
            ok=False,
            error_code="not_implemented",
            message=(
                "/share 第一版只支持本地脱敏导出：请用 /export 生成 .rcmission 后自行分发；"
                "绝不默认上传 gist/云（§8.13）。"
            ),
        )

    async def _reload(req: CommandRequestV1) -> CommandResultV1:
        raw = str(req.arguments.get("domains", "")).strip()
        domains = raw.split() if raw else ["prompts", "workers"]
        results = service.reload_domains(domains)
        lines = [f"{d}: {'ok' if r['ok'] else '拒绝'} — {r['detail']}" for d, r in results.items()]
        from rosclaw.contracts.agent.agent_event import AgentEventType

        if req.mission_id:
            await service._events.append(
                req.mission_id,
                AgentEventType.CONFIG_RELOADED,
                {"domains": domains, "ok": all(r["ok"] for r in results.values())},
            )
        return CommandResultV1(
            request_id=req.request_id,
            command_name=req.command_name,
            ok=all(r["ok"] for r in results.values()),
            message="\n".join(lines),
            data=results,
        )

    async def _settings(req: CommandRequestV1) -> CommandResultV1:
        key = str(req.arguments.get("key", "")).strip()
        value = req.arguments.get("value")
        if not key:
            current = service.settings.get()
            safe = {
                k: v
                for k, v in current.items()
                if k in ("agent", "models") and "key" not in str(v).lower()
            }
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=True,
                message="settings（仅非安全键可写）",
                data={"settings": safe},
            )
        if value is None:
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=True,
                message=f"{key} = {service.settings.get_key(key)}",
            )
        try:
            change = service.settings.set_key(key, value)
        except Exception as exc:  # noqa: BLE001
            return CommandResultV1(
                request_id=req.request_id,
                command_name=req.command_name,
                ok=False,
                error_code="settings_rejected",
                message=str(exc),
            )
        if req.mission_id:
            from rosclaw.contracts.agent.agent_event import AgentEventType

            await service._events.append(
                req.mission_id,
                AgentEventType.CONFIG_RELOADED,
                {"settings_key": key},
            )
        return CommandResultV1(
            request_id=req.request_id,
            command_name=req.command_name,
            ok=True,
            message=f"{key}: {change['old']} → {change['new']}（已原子写入并审计）",
            data=change,
        )

    register(
        CommandSpecV1(
            name="export",
            description="导出 .rcmission 脱敏 bundle（manifest/conversation/events/compactions/checksums）",
            argument_hint="[path]",
            category=CommandCategory.MISSION,
            owner=CommandOwner.MISSION_CONTROL,
            mutability="NONE",
            handler="mission.export",
        ),
        _export,
    )
    register(
        CommandSpecV1(
            name="import",
            description="导入 .rcmission（只读归档；不恢复任何授权效力）",
            argument_hint="<path>",
            category=CommandCategory.MISSION,
            owner=CommandOwner.MISSION_CONTROL,
            mutability="PERSISTED",
            confirmation="CONFIRM",
            handler="mission.import",
        ),
        _import,
    )
    register(
        CommandSpecV1(
            name="share",
            description="分享（第一版仅本地脱敏导出，不默认上传）",
            category=CommandCategory.MISSION,
            owner=CommandOwner.MISSION_CONTROL,
            handler="mission.share",
        ),
        _share,
    )
    register(
        CommandSpecV1(
            name="reload",
            description="分域原子重载（prompts/workers/models；安全域永远拒绝）",
            argument_hint="[domain ...]",
            category=CommandCategory.EXECUTION,
            owner=CommandOwner.AGENT_CONTROL,
            mutability="CONTROL_STATE",
            handler="agent.reload",
        ),
        _reload,
    )
    register(
        CommandSpecV1(
            name="settings",
            description="查看/修改非安全设置（白名单键、原子写、审计）",
            argument_hint="[<key> [<value>]]",
            category=CommandCategory.HELP_UI,
            owner=CommandOwner.AGENT_CONTROL,
            mutability="PERSISTED",
            handler="agent.settings",
        ),
        _settings,
    )
