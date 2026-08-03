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


class CommandService:
    def __init__(self, service: AgentService) -> None:
        self._service = service
        self._handlers: dict[str, HandlerFn] = {}
        self._specs: dict[str, CommandSpecV1] = {}
        self._idempotency: dict[str, CommandResultV1] = {}
        self._register_builtins()

    # -- registry ----------------------------------------------------------------

    def _register(self, spec: CommandSpecV1, handler: HandlerFn) -> None:
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
