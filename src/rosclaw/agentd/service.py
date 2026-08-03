"""AgentService — the rosclaw-agentd application object (PR-NA-040).

Assembles MissionStore + ContextCompiler + ModelGateway + AgentLoop from
``AgentConfig``. Runs unprivileged; holds no hardware authority. Exposes a
small local HTTP/WebSocket-free JSON API for the CLI and the console.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path

from fastapi import Request as _Request
from pydantic import BaseModel as _BaseModel

from rosclaw.agentd.config import AgentConfig
from rosclaw.agentd.context.compiler import ContextCompiler
from rosclaw.agentd.context.prompt_registry import load_prompt
from rosclaw.agentd.context.sources import SourceBundle
from rosclaw.agentd.loop import AgentLoop, LoopTurnResult
from rosclaw.agentd.mission import MissionStore
from rosclaw.agentd.models.gateway import (
    ModelGateway,
    ModelGatewayError,
    ModelProbeResult,
    OpenAICompatGateway,
)
from rosclaw.agentd.runtime_sources import (
    CatalogCapabilitySource,
    ConfigConsentSource,
    DaemonSelfSource,
    EmptyMemorySource,
    ResolverBodySource,
    SimBodySource,
    SimSelfSource,
)
from rosclaw.agentd.tools import BuiltinToolRegistry
from rosclaw.agentd.usage import UsageRecorder
from rosclaw.contracts.agent.mission import (
    BodyBinding,
    Budgets,
    ExecutionMode,
    Goal,
    MissionSessionV1,
)
from rosclaw.contracts.common import ValidationError

AGENTD_DIR = "agentd"


class RegistryOrgSource:
    """L6 organization layer backed by the WorkerRegistry."""

    def __init__(self, registry) -> None:
        self._registry = registry

    def get_org(self):
        from rosclaw.agentd.context.sources import OrgFacts

        lines = []
        for card in self._registry.list():
            status = self._registry.status_of(card.worker_id) or "UNKNOWN"
            caps = ", ".join(c.name for c in card.capabilities)
            lines.append(f"- {card.worker_id} [{card.kind.value}/{status}] capabilities: {caps}")
        return OrgFacts(workers_summary="Registered workers:\n" + "\n".join(lines) if lines else "")


class BrokerConsentSource:
    """L7 consent layer bound to real grants (§5.5: grant 生效/撤销/过期
    触发重编译——public hash 变化即触发）。"""

    def __init__(self, base, conn) -> None:
        self._base = base
        self._conn = conn

    @property
    def policy_hash(self) -> str:
        return self._base.policy_hash

    def get_consent(self, mission_id: str):
        import json as _json
        from datetime import UTC, datetime

        from rosclaw.agentd.context.sources import ConsentFacts

        facts = self._base.get_consent(mission_id)
        row = self._conn.execute(
            "SELECT g.public_json FROM mission_grants AS g "
            "JOIN operator_requests AS r ON r.request_id = g.request_id "
            "WHERE g.revoked = 0 AND g.consumed = 0 AND g.expires_at > ? "
            "AND r.mission_id = ? ORDER BY g.created_at DESC LIMIT 1",
            (datetime.now(UTC).isoformat(), mission_id),
        ).fetchone()
        grant_hash = None
        scope_summary = facts.public_scope_summary
        if row is not None:
            public = _json.loads(row["public_json"])
            grant_hash = public.get("public_hash")
            scope = public.get("scope") if isinstance(public.get("scope"), dict) else {}
            scope_summary = (
                f"{scope_summary}\nActive mission grant: grant_id={public.get('grant_id')}; "
                f"tier={scope.get('tier')}; risk_ceiling={public.get('risk_ceiling')}; "
                f"public_hash={grant_hash}. Reference this exact public grant_id in "
                "REQUEST_ACTION. No private signature or permit is exposed."
            ).strip()
        return ConsentFacts(
            policy_hash=facts.policy_hash,
            mission_grant_public_hash=grant_hash,
            public_scope_summary=scope_summary,
            allowed_risk_tiers=facts.allowed_risk_tiers,
        )


class AgentService:
    def __init__(
        self,
        config: AgentConfig,
        rosclaw_home: Path,
        *,
        gateway: ModelGateway | None = None,
    ) -> None:
        self._config = config
        self._home = rosclaw_home
        db_dir = rosclaw_home / AGENTD_DIR
        db_dir.mkdir(parents=True, exist_ok=True)
        self._store = MissionStore(db_dir / "missions.db")
        # Worker registry first: the context compiler's org layer reads it.
        from rosclaw.agentd.handlers import ServiceIntentHandlers
        from rosclaw.agentd.workers import NativeWorkerAdapter, WorkerManager, WorkerRegistry

        self._body_id = config.active_body_id
        self._daemon_client = None
        daemon_socket = os.environ.get("ROSCLAW_DAEMON_SOCKET") or str(
            rosclaw_home / "run" / "rosclawd.sock"
        )
        if Path(daemon_socket).exists():
            from rosclaw.daemon.client import DaemonClient

            self._daemon_client = DaemonClient(socket_path=daemon_socket)
        self._registry = WorkerRegistry(self._store.connection)
        self._registry.register_builtins(actor_id=self.actor_id)
        self._simulation_body = self._body_id.startswith("sim/")
        if self._simulation_body:
            self._body_source = SimBodySource(self._body_id)
            self_source = SimSelfSource()
        else:
            self._body_source = ResolverBodySource(
                workspace=rosclaw_home,
                body_id=self._body_id,
            )
            self_source = DaemonSelfSource(self._daemon_client)
        body = self._body_source.get_body(self._body_id)
        self._tools = BuiltinToolRegistry(
            body_id=self._body_id,
            body_summary=body.summary if body else "configured body is unavailable",
        )
        # PR-05: Tool/Capability Catalog — descriptors, resolver, evidence.
        from rosclaw.agentd.tooling.artifact_result import ArtifactResultStore
        from rosclaw.agentd.tooling.catalog import ToolCatalog
        from rosclaw.agentd.tooling.catalog_registry import CatalogToolRegistry
        from rosclaw.agentd.tooling.mcp_adapter import McpCapabilityAdapter, McpServerConfig
        from rosclaw.agentd.tooling.native_tools import register_native_tools
        from rosclaw.agentd.tooling.resolver import ToolResolver

        self._tool_catalog = ToolCatalog()
        register_native_tools(self._tool_catalog, self._tools, simulation=self._simulation_body)
        self._artifact_store = ArtifactResultStore(db_dir)
        self._tool_resolver = ToolResolver(self._tool_catalog)
        self._tool_registry = CatalogToolRegistry(
            self._tool_catalog,
            self._tool_resolver,
            artifact_store=self._artifact_store,
            body_type=self._body_id,
        )
        self._mcp_adapters = [
            McpCapabilityAdapter(
                McpServerConfig(
                    name=str(s.get("name", "")),
                    command=str(s.get("command", "")),
                    args=tuple(str(a) for a in s.get("args", []) or []),
                    env_refs=tuple(str(r) for r in s.get("env_refs", []) or []),
                    observation_tools=tuple(s.get("observation_tools", []) or ()),
                    action_tools=tuple(s.get("action_tools", []) or ()),
                    supported_modes=tuple(s.get("supported_modes", ("SIMULATION",)) or ()),
                    required_body_types=tuple(s.get("required_body_types", []) or ()),
                    timeout_ms=int(s.get("timeout_ms", 5000)),
                ),
                self._tool_catalog,
            )
            for s in self._config.mcp_servers
            if s.get("name") and s.get("command")
        ]
        self._mcp_discovered = False
        # 批次 B：命令注册表（命令永不进入模型上下文）。
        from rosclaw.agentd.ui.command_service import CommandService
        from rosclaw.agentd.ui.interaction_service import InteractionService

        self._commands = CommandService(self)
        self._interactions = InteractionService()
        #: /scoped-models 快捷切换集合（内存态）。
        self._scoped_models: set[str] = set()
        # 批次 E：导出/导入/设置。
        from rosclaw.agentd.ui.export_service import ExportService
        from rosclaw.agentd.ui.import_service import ImportService
        from rosclaw.agentd.ui.settings_service import SettingsService

        self._exporter = ExportService(self)
        self._importer = ImportService(self)
        self._settings = SettingsService(self._home / "config.yaml")
        consent_source = ConfigConsentSource()
        from rosclaw.operator import OperatorBroker

        self._broker = OperatorBroker(
            self._store.connection, policy_hash=consent_source.policy_hash
        )
        self._compiler = ContextCompiler(
            SourceBundle(
                constitution_text=load_prompt("native_agent_v1.md").text,
                body=self._body_source,
                self_source=self_source,
                capabilities=CatalogCapabilitySource(self._tool_catalog),
                memory=EmptyMemorySource(),
                organization=RegistryOrgSource(self._registry),
                consent=BrokerConsentSource(consent_source, self._store.connection),
                runtime_status_summary=(
                    "agentd local; simulated body; rosclawd optional"
                    if self._simulation_body
                    else "agentd bound to a real body through rosclawd; physical evidence is daemon-owned"
                ),
            ),
            max_input_tokens=config.max_input_tokens,
            dynamic_tool_limit=config.dynamic_tool_limit,
        )
        if gateway is not None:
            self._gateway: ModelGateway = gateway
        else:
            from rosclaw.agentd.models.failover import FailoverGateway

            policy = config.to_policy()
            chain = policy.fallback_chain()
            if config.model_backend == "modeld":
                # 批次 D：AgentLoop 不再直接接触 OpenAI-compatible 协议细节。
                from rosclaw.agentd.models.modeld_gateway import ModeldGateway

                candidates = [(p, ModeldGateway(p, home=self._home)) for p in chain]
            else:
                candidates = [(p, OpenAICompatGateway(p)) for p in chain]
            # 单 profile 也走 FailoverGateway：统一的 cooldown/RPM 语义。
            self._gateway = FailoverGateway(candidates)
        self._prompt = load_prompt("native_agent_v1.md")
        self._loops: dict[str, AgentLoop] = {}
        self._lock = asyncio.Lock()
        self._usage = UsageRecorder(self._store.connection)
        # AgentEventV2 journal + live bus (PR-02).
        from rosclaw.agentd.events import AgentEventStore

        self._events = AgentEventStore(self._store.connection)
        self._turn_tasks: dict[str, asyncio.Task] = {}
        from rosclaw.agentd.runner import MissionRunner

        self._runner = MissionRunner(self)
        # External harness packs (PR-WF-054): register cards by probe result
        # (missing binary → DISABLED with T0 note, never fake readiness).
        # 同步探活（init 可能在 async 上下文中被构造，不能 run_until_complete）。
        from rosclaw.agentd.workers.external import ExternalHarnessAdapter
        from rosclaw.agentd.workers.packs import ALL_PACKS, card_for_pack

        external_adapter = ExternalHarnessAdapter(cwd=rosclaw_home)
        for pack in ALL_PACKS:
            card = card_for_pack(pack)
            self._registry.register(card, actor_id=self.actor_id)
            ready, detail = self._probe_pack_sync(
                pack.executable, pack.min_version, pack.install_hint
            )
            if not ready:
                self._registry.set_status(
                    pack.worker_id, "DISABLED", actor_id=self.actor_id, reason=detail
                )
        self._worker_manager = WorkerManager(
            self._store.connection,
            adapters={
                "native_inproc": NativeWorkerAdapter(self._gateway),
                "external_cli": external_adapter,
            },
            actor_id=self.actor_id,
            event_recorder=self._record_worker_event,
        )
        self._handlers = ServiceIntentHandlers(
            registry=self._registry,
            manager=self._worker_manager,
            actor_id=self.actor_id,
            broker=self._broker,
            body_id=self._body_id,
            body_hash=body.effective_body_hash if body else "",
            mode=config.default_mode,
        )
        # Daemon action channel (K3) + consent channel (ADR-0007): only when
        # a rosclawd client is actually available — otherwise both degrade
        # honestly.
        self._action_channel = None
        self._consent_channel = None
        if self._daemon_client is not None:
            from rosclaw.agentd.action_channel import DaemonActionChannel
            from rosclaw.agentd.consent_channel import DaemonConsentChannel

            self._action_channel = DaemonActionChannel(
                self._daemon_client,
                actor_id=self.actor_id,
                body_id=self._body_id,
                body_hash=body.effective_body_hash if body else "",
            )
            self._handlers._action_channel = self._action_channel
            self._consent_channel = DaemonConsentChannel(
                self._daemon_client,
                actor_id=self.actor_id,
                body_id=self._body_id,
                body_hash=body.effective_body_hash if body else "",
            )
            self._handlers._consent_channel = self._consent_channel
        # Team Fabric: enabled via config `team.enabled`. Local coordinator
        # in P0 (local_sim); ROS 2/Zenoh transports are later PRs.
        team_cfg = (config.raw.get("team") or {}) if config.raw else {}
        if team_cfg.get("enabled"):
            from rosclaw.team import TeamCoordinator

            self._team_coordinator = TeamCoordinator(
                self._store.connection,
                team_id=str(team_cfg.get("team_id", "default_team")),
                actor_id=self.actor_id,
                policy_hash=consent_source.policy_hash,
            )
            self._handlers._team_coordinator = self._team_coordinator
        else:
            self._team_coordinator = None

    # ------------------------------------------------------------------
    @staticmethod
    def _probe_pack_sync(executable: str, min_version: str, install_hint: str) -> tuple[bool, str]:
        """Synchronous pack probe (used at init; adapter.probe is the async path)."""
        import shutil
        import subprocess

        from rosclaw.agentd.workers.packs import version_ok

        exe = shutil.which(executable)
        if exe is None:
            return False, f"二进制 {executable!r} 未找到（T0 Discovered）。{install_hint}"
        try:
            out = subprocess.run(
                [executable, "--version"],
                capture_output=True,
                timeout=15,
                text=True,
            )
            version_text = (out.stdout or out.stderr).strip().split()[0]
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, f"version probe failed: {exc}"
        if not version_ok(version_text, min_version):
            return False, f"{version_text} < 最小兼容版本 {min_version}，请升级。"
        return True, version_text

    # ------------------------------------------------------------------
    @property
    def store(self) -> MissionStore:
        return self._store

    @property
    def actor_id(self) -> str:
        safe = self._body_id.replace("/", "_")
        return f"agent:rosclaw-native:{safe}"

    def _loop_for(self, mission_id: str) -> AgentLoop:
        loop = self._loops.get(mission_id)
        if loop is None:
            loop = AgentLoop(
                store=self._store,
                compiler=self._compiler,
                gateway=self._gateway,
                prompt=self._prompt,
                tools=self._tool_registry,
                handlers=self._handlers,
                actor_id=self.actor_id,
                max_tool_rounds=self._config.max_tool_rounds,
                usage_recorder=self._usage,
                event_sink=self._event_sink_for(mission_id),
                decision_protocol=self._config.decision_protocol,
                legacy_fenced_json_fallback=self._config.legacy_fenced_json_fallback,
            )
            self._loops[mission_id] = loop
        return loop

    def _event_sink_for(self, mission_id: str):
        """Per-mission event sink closure (PR-02)."""
        from rosclaw.contracts.agent.agent_event import Visibility

        async def sink(type, payload, *, visibility=None, task_id=None) -> None:
            await self._events.append(
                mission_id,
                type,
                payload,
                visibility=visibility or Visibility.USER,
                task_id=task_id,
            )

        return sink

    # ------------------------------------------------------------------
    # /v2 event-streaming surface (PR-02): turn submit decoupled from SSE.
    # ------------------------------------------------------------------
    async def submit_turn_v2(self, mission_id: str, text: str) -> str:
        """202-style submit via MissionRunner (wake-tracked, PR-03)."""
        return await self._runner.submit_turn(mission_id, text)

    def events_replay(self, mission_id: str, *, after_sequence: int = 0, limit: int = 1000):
        return self._events.replay(mission_id, after_sequence=after_sequence, limit=limit)

    def events_subscribe(self, mission_id: str) -> asyncio.Queue:
        return self._events.bus.subscribe(mission_id)

    def events_unsubscribe(self, mission_id: str, queue: asyncio.Queue) -> None:
        self._events.bus.unsubscribe(mission_id, queue)

    async def cancel_turn_v2(self, mission_id: str) -> None:
        await self.cancel(mission_id)
        task = self._turn_tasks.get(mission_id)
        if task is not None and not task.done():
            task.cancel()

    # ------------------------------------------------------------------
    def create_mission(
        self,
        goal_text: str,
        *,
        mode: str | None = None,
        owner_principal: str = "user:local:1000",
    ) -> MissionSessionV1:
        requested_mode = ExecutionMode(mode or self._config.default_mode)
        body = self._body_source.get_body(self._body_id)
        if body is None:
            raise ValidationError(
                f"body {self._body_id!r} is not linked, hash-valid, and available"
            )
        if requested_mode is not ExecutionMode.SIMULATION:
            gaps: list[str] = []
            if requested_mode is ExecutionMode.REAL and self._config.physical_action_count <= 0:
                gaps.append("agent.budgets.physical_action_count must be greater than zero")
            if self._simulation_body:
                gaps.append("configured body is simulated, not a live BodyResolver body")
            if self._daemon_client is None:
                gaps.append("rosclawd socket is unavailable")
            else:
                try:
                    status = self._daemon_client.get_runtime_status()
                except Exception as exc:  # noqa: BLE001 - report an honest prerequisite gap
                    gaps.append(f"rosclawd status unavailable: {exc}")
                else:
                    if not status.get("running"):
                        gaps.append("rosclawd is not running")
                    if status.get("robot_id") != self._body_id:
                        gaps.append(
                            f"rosclawd robot_id {status.get('robot_id')!r} != {self._body_id!r}"
                        )
                    pack = status.get("robot_pack") or {}
                    if not pack.get("loaded") or pack.get("signature_status") != "valid":
                        gaps.append("verified Robot Pack is not loaded")
                    suffix = f":{requested_mode.value}"
                    if not any(
                        str(item).endswith(suffix)
                        for item in status.get("registered_executors") or []
                    ):
                        gaps.append(f"no {requested_mode.value} executor is registered")
            if gaps:
                raise ValidationError(
                    f"mode {requested_mode.value} requested but prerequisites are missing: "
                    + "; ".join(gaps)
                )
        return self._store.create_mission(
            owner_principal=owner_principal,
            goal=Goal(text=goal_text, language=self._config.language),
            body_binding=BodyBinding(
                body_id=body.body_id, effective_body_hash=body.effective_body_hash
            ),
            mode=requested_mode,
            budgets=Budgets(physical_action_count=self._config.physical_action_count),
            actor_id=self.actor_id,
        )

    def list_missions(self) -> list[MissionSessionV1]:
        return self._store.list_missions()

    def get_mission(self, mission_id: str) -> MissionSessionV1 | None:
        return self._store.get_mission(mission_id)

    async def send_turn(self, mission_id: str, text: str, on_text_delta=None) -> LoopTurnResult:
        mission = self._store.get_mission(mission_id)
        if mission is None:
            raise ValidationError(f"unknown mission {mission_id!r}")
        if self.mission_archived(mission_id):
            raise ValidationError(
                f"mission {mission_id!r} is archived (read-only); create a new mission"
            )
        await self._ensure_mcp_discovered()
        async with self._runner.lock_for(mission_id):
            if self._handlers is not None:
                self._handlers._mode = mission.mode.value
                self._handlers._principal = mission.owner_principal
            loop = self._loop_for(mission_id)
            return await loop.run_user_turn(
                mission, text, now=datetime.now(UTC), on_text_delta=on_text_delta
            )

    def mission_usage(self, mission_id: str) -> dict:
        return self._usage.mission_totals(mission_id)

    async def _ensure_mcp_discovered(self) -> None:
        """Discover configured MCP servers once (PR-05); failures quarantine
        the source honestly and never block the turn."""
        if self._mcp_discovered:
            return
        self._mcp_discovered = True
        for adapter in self._mcp_adapters:
            try:
                await adapter.discover()
            except Exception:  # noqa: BLE001 - discovery must never break chat
                self._tool_catalog.quarantine_source(adapter.source, "discovery_crashed")

    @property
    def tool_catalog(self):
        return self._tool_catalog

    @property
    def tool_resolver(self):
        return self._tool_resolver

    # -- modeld 管理面（批次 D：/providers /model /login /logout） ---------------
    def _modeld_mgmt(self):
        """共享的 modeld 管理通道（懒启动；runtime 缺失 → None）。"""
        if getattr(self, "_modeld_mgmt_instance", None) is not None:
            return self._modeld_mgmt_instance
        from rosclaw.agentd.models.modeld_gateway import ModeldGateway, _find_modeld_runtime

        if _find_modeld_runtime() is None:
            return None
        profile = (
            self._config.to_policy().default
            if self._config.profiles
            else getattr(self._gateway, "profile", None)
        )
        if profile is None:
            return None
        self._modeld_mgmt_instance = ModeldGateway(profile, home=self._home)
        return self._modeld_mgmt_instance

    async def modeld_providers(self) -> dict:
        mgmt = self._modeld_mgmt()
        if mgmt is None:
            return {"available": False, "providers": [], "error": "modeld runtime unavailable"}
        try:
            data = await mgmt.manage("GET", "/v1/providers")
            data["available"] = True
            return data
        except Exception as exc:  # noqa: BLE001
            return {"available": False, "providers": [], "error": str(exc)}

    async def modeld_models(self, provider: str) -> dict:
        mgmt = self._modeld_mgmt()
        if mgmt is None:
            return {"models": [], "error": "modeld runtime unavailable"}
        return await mgmt.manage("GET", f"/v1/models?provider={provider}")

    async def modeld_login(self, provider: str, api_key: str) -> dict:
        """API key 登录：secret 只经内存进 modeld，不落 mission journal。"""
        mgmt = self._modeld_mgmt()
        if mgmt is None:
            return {"ok": False, "error": "modeld runtime unavailable"}
        return await mgmt.manage(
            "POST", f"/v1/auth/{provider}/login", {"mode": "api_key", "api_key": api_key}
        )

    async def modeld_logout(self, provider: str) -> dict:
        mgmt = self._modeld_mgmt()
        if mgmt is None:
            return {"ok": False, "error": "modeld runtime unavailable"}
        return await mgmt.manage("POST", f"/v1/auth/{provider}/logout", {})

    def current_model_label(self) -> str:
        profile = self._gateway.profile
        return f"{profile.provider}/{profile.model}（profile: {profile.name}）"

    def switch_model(self, provider: str, model: str) -> dict:
        """切换默认 profile 的 provider/model（内存态；持久化走 /settings）。

        切换永不改变工具权限、Mission mode 或 grant（§8.5）。
        """
        if not self._config.profiles:
            return {"ok": False, "error_code": "no_profiles", "message": "未配置模型"}
        profile = self._config.to_policy().default
        old = f"{profile.provider}/{profile.model}"
        if self._config.model_backend != "modeld":
            return {
                "ok": False,
                "error_code": "legacy_backend",
                "message": (
                    "legacy backend 的运行时在启动时绑定 endpoint，/model 暂不支持热切换；"
                    "请设置 models.backend: modeld 后重试（Kimi 现有配置无需改动）"
                ),
            }
        # modeld provider 名映射与 ModeldGateway 一致。
        from rosclaw.agentd.models.modeld_gateway import _PROVIDER_MAP

        mapped = _PROVIDER_MAP.get(provider, provider)
        # profile 对象被 config / FailoverGateway candidates / 既有 AgentLoop
        # 共享；frozen dataclass 的就地字段替换让"下一 turn 生效"在所有
        # 引用点同时成立（当前 turn 不中断）。
        object.__setattr__(profile, "provider", mapped)
        object.__setattr__(profile, "model", model)
        return {
            "ok": True,
            "message": (
                f"模型已从 {old} 切换为 {mapped}/{model}（下一 turn 生效；"
                "当前 turn 不中断；持久化配置修改将在 /settings 提供）"
            ),
        }

    def _record_worker_event(self, mission_id: str, to_status: str, payload: dict) -> None:
        """Sync bridge: WorkerManager transitions → AgentEventV2 (批次 B)."""
        import asyncio as _asyncio

        from rosclaw.contracts.agent.agent_event import AgentEventType

        event_type = {
            "CLAIMED": AgentEventType.WORKER_CLAIMED,
            "RUNNING": AgentEventType.WORKER_STARTED,
            "SUBMITTED": AgentEventType.WORKER_SUBMITTED,
            "VERIFYING": AgentEventType.WORKER_VERIFYING,
            "ACCEPTED": AgentEventType.WORKER_ACCEPTED,
            "FAILED": AgentEventType.WORKER_FAILED,
            "EXPIRED": AgentEventType.WORKER_EXPIRED,
        }.get(to_status)
        if event_type is None:
            return
        try:
            loop = _asyncio.get_running_loop()
        except RuntimeError:
            return  # no loop (e.g. sync CLI path) — worker_events table is the record
        loop.create_task(self._events.append(mission_id, event_type, payload))

    @property
    def commands(self):
        return self._commands

    @property
    def interactions(self):
        return self._interactions

    @property
    def scoped_models(self) -> set[str]:
        return self._scoped_models

    @property
    def exporter(self):
        return self._exporter

    @property
    def importer(self):
        return self._importer

    @property
    def settings(self):
        return self._settings

    def reload_domains(self, domains: list[str]) -> dict:
        """/reload（§8.15）：分域原子重载；安全域永远拒绝。

        可 reload：prompts（prompt registry）、workers（pack 重新探活注册）、
        models（modeld provider catalog refresh 提示）。
        不可 reload：rosclawd Policy、Robot Pack 签名、Body 安全边界、
        Permit、设备权限、REAL 风险上限。
        """
        results: dict[str, dict] = {}
        for domain in domains:
            if domain == "prompts":
                from rosclaw.agentd.context.prompt_registry import load_prompt

                try:
                    self._prompt = load_prompt("native_agent_v1.md")
                    results[domain] = {
                        "ok": True,
                        "detail": f"prompt v{self._prompt.version} hash={self._prompt.content_hash[:16]}（活跃 turn 不换 prompt，下一 turn 生效）",
                    }
                except Exception as exc:  # noqa: BLE001
                    results[domain] = {"ok": False, "detail": f"{exc}（保持旧配置）"}
            elif domain == "workers":
                from rosclaw.agentd.workers.packs import ALL_PACKS, card_for_pack

                refreshed = []
                for pack in ALL_PACKS:
                    try:
                        self._registry.register(card_for_pack(pack), actor_id=self.actor_id)
                        refreshed.append(pack.worker_id)
                    except Exception as exc:  # noqa: BLE001
                        results.setdefault(domain, {"ok": False, "detail": str(exc)})
                else:
                    results[domain] = {"ok": True, "detail": f"re-registered {len(refreshed)} packs"}
            elif domain == "models":
                results[domain] = {
                    "ok": True,
                    "detail": "modeld provider catalog 为启动时构建；/model 可切换，重启后重建。",
                }
            elif domain in ("policy", "robot_pack", "body", "permits", "permissions", "safety"):
                results[domain] = {
                    "ok": False,
                    "detail": f"{domain} 属安全域，/reload 永不修改（走专用管理面）",
                }
            else:
                results[domain] = {"ok": False, "detail": f"未知 reload 域 {domain!r}"}
        return results

    def conversation(self, mission_id: str) -> list[dict]:
        return self._store.conversation(mission_id)

    # ------------------------------------------------------------------
    async def compact(
        self,
        mission_id: str,
        *,
        instructions: str | None = None,
        dry_run: bool = False,
    ) -> dict:
        """`/compact` 的服务端实现（PR-07）。"""
        mission = self._store.get_mission(mission_id)
        if mission is None:
            raise ValidationError(f"unknown mission {mission_id!r}")
        async with self._runner.lock_for(mission_id):
            loop = self._loop_for(mission_id)
            return await loop.compact_conversation(
                mission, reason="manual", focus=instructions, dry_run=dry_run
            )

    def compaction_status(self, mission_id: str) -> dict:
        from rosclaw.agentd.context.compaction import (
            CompactionStore,
            estimate_messages_tokens,
        )

        store = CompactionStore(self._store.connection)
        entries = store.list(mission_id)
        return {
            "compactions": len(entries),
            "last": entries[-1].model_dump(mode="json") if entries else None,
            "current_view_tokens": estimate_messages_tokens(self._store.conversation(mission_id)),
            "journal_events": len(self._store.events(mission_id)),
        }

    async def cancel(self, mission_id: str) -> None:
        from rosclaw.contracts.agent.agent_event import AgentEventType

        await self._events.append(mission_id, AgentEventType.TURN_CANCEL_REQUESTED, {})
        loop = self._loops.get(mission_id)
        if loop is not None:
            loop.request_cancel()

    # ------------------------------------------------------------------
    # 批次 B：UI 控制面（命令/快照/归档/重命名）
    # ------------------------------------------------------------------
    def turn_in_flight(self, mission_id: str) -> bool:
        task = self._turn_tasks.get(mission_id)
        return task is not None and not task.done()

    def rename_mission(self, mission_id: str, name: str) -> None:
        if self._store.get_mission(mission_id) is None:
            raise ValidationError(f"unknown mission {mission_id!r}")
        self._store.set_mission_meta(mission_id, display_name=name)
        import asyncio

        from rosclaw.contracts.agent.agent_event import AgentEventType

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self._events.append(
                    mission_id, AgentEventType.MISSION_RENAMED, {"name": name[:120]}
                )
            )
        except RuntimeError:
            pass

    def archive_mission(self, mission_id: str) -> None:
        if self._store.get_mission(mission_id) is None:
            raise ValidationError(f"unknown mission {mission_id!r}")
        self._store.set_mission_meta(mission_id, archived=True)
        import asyncio

        from rosclaw.contracts.agent.agent_event import AgentEventType

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self._events.append(mission_id, AgentEventType.MISSION_ARCHIVED, {})
            )
        except RuntimeError:
            pass

    def mission_archived(self, mission_id: str) -> bool:
        return bool(self._store.mission_meta(mission_id)["archived"])

    def status_snapshot(self, mission_id: str | None = None) -> dict:
        data: dict = {
            "agent": "rosclaw-agentd",
            "body_id": self._body_id,
            "daemon_connected": self._daemon_client is not None,
            "mcp_servers": [a.source for a in self._mcp_adapters],
            "model_profile": self._gateway.profile.name,
            "model": self._gateway.profile.model,
            "tools_registered": len(self._tool_catalog.list()),
        }
        if mission_id:
            mission = self._store.get_mission(mission_id)
            if mission is not None:
                data["mission"] = {
                    "state": mission.state.value,
                    "mode": mission.mode.value,
                    "pending_approvals": len(self.pending_approvals(mission_id)),
                    "turn_in_flight": self.turn_in_flight(mission_id),
                    "archived": self.mission_archived(mission_id),
                }
        return data

    def snapshot(self, mission_id: str):
        """MissionSnapshotV1 — 重连校准用权威快照（批次 B §5.3）。"""
        from rosclaw.contracts.ui.snapshots import MissionSnapshotV1

        mission = self._store.get_mission(mission_id)
        if mission is None:
            raise ValidationError(f"unknown mission {mission_id!r}")
        from rosclaw.agentd.context.compaction import CompactionStore
        from rosclaw.agentd.mission.store import _utcnow

        meta = self._store.mission_meta(mission_id)
        grants = [
            {
                "grant_id": g.get("grant_id"),
                "tier": g.get("tier"),
                "risk_ceiling": g.get("risk_ceiling"),
                "expires_at": g.get("expires_at"),
            }
            for g in self.list_grants()
            if not g.get("revoked") and not g.get("consumed")
        ]
        orders = [
            {
                "work_order_id": o.work_order_id,
                "status": o.status,
                "assigned_to": o.assigned_to,
            }
            for o in self._worker_manager.orders_for_mission(mission_id)
            if o.status not in ("ACCEPTED", "REJECTED", "EXPIRED", "CANCELLED")
        ]
        return MissionSnapshotV1(
            mission_id=mission_id,
            name=meta["display_name"] or mission.goal.text[:60],
            goal_text=mission.goal.text,
            state=mission.state.value,
            mode=mission.mode.value,
            body_id=mission.body_binding.body_id,
            context_id=f"ctx_{mission_id}",
            context_revision=mission.context_revision,
            task_graph_revision=mission.task_graph_revision,
            last_event_sequence=self._events.latest_sequence(mission_id),
            turn_in_flight=self.turn_in_flight(mission_id),
            pending_approvals=[
                {
                    "request_id": r.request_id,
                    "title": r.action_display.title,
                    "risk_tier": r.action_display.risk_tier,
                    "expires_at": r.expires_at,
                }
                for r in self.pending_approvals(mission_id)
            ],
            active_grants=grants,
            open_work_orders=orders,
            usage=self._usage.mission_totals(mission_id),
            budgets=mission.budgets.model_dump(mode="json"),
            compaction_count=CompactionStore(self._store.connection).count(mission_id),
            tool_count=len(self._tool_catalog.list()),
            captured_at=_utcnow(),
        )

    # ------------------------------------------------------------------
    # approvals (Operator Broker surface for CLI/console)
    # ------------------------------------------------------------------
    def pending_approvals(self, mission_id: str | None = None):
        return self._broker.pending_requests(mission_id)

    async def decide_approval(self, request_id: str, *, principal: str, approve: bool):
        """认知层裁决 +（有 consent channel 时）daemon proposal 物理层裁决。

        ACCEPT 时 daemon 独立签发 permit、提交动作并监督到终态 receipt——
        agentd 只是发起方与见证方，不持有 permit。
        """
        grant = self._broker.decide(request_id, principal=principal, approve=approve)
        approval_req = self._broker.get_request(request_id)
        if approval_req is not None:
            from rosclaw.contracts.agent.agent_event import AgentEventType

            await self._events.append(
                approval_req.mission_id,
                AgentEventType.APPROVAL_DECIDED,
                {
                    "request_id": request_id,
                    "approved": approve,
                    "grant_id": grant.grant_id if grant else None,
                },
            )
            self._runner.notify_approval_decided(
                approval_req.mission_id,
                request_id,
                approved=approve,
                grant_id=grant.grant_id if grant else None,
            )
        if self._consent_channel is not None:
            request = self._broker.get_request(request_id)
            proposal_id = getattr(request, "daemon_proposal_id", None) or (
                request.model_dump(mode="json").get("daemon_proposal_id") if request else None
            )
            if proposal_id:
                from rosclaw.agentd.consent_channel import ConsentChannelError

                try:
                    await self._consent_channel.decide(
                        proposal_id,
                        principal_id=principal,
                        accept=approve,
                        channel="rosclaw_console",
                        reason="operator approved via rosclaw console",
                    )
                except ConsentChannelError as exc:
                    # 认知层已裁决但物理层失败：如实报告，不伪造派发。
                    raise ValidationError(
                        f"agentd 授权已记录，但 daemon proposal 裁决失败（未派发）：{exc}"
                    ) from exc
        return grant

    def list_grants(self):
        rows = self._store.connection.execute(
            "SELECT public_json, revoked, consumed, expires_at FROM mission_grants "
            "ORDER BY created_at DESC"
        ).fetchall()
        import json as _json

        from rosclaw.contracts.operator.grant import MissionGrantV1

        grants = []
        for row in rows:
            grant = MissionGrantV1(**_json.loads(row["public_json"]))
            grants.append(
                {
                    "grant_id": grant.grant_id,
                    "principal": grant.principal,
                    "mode": grant.mode,
                    "tier": grant.scope.tier,
                    "risk_ceiling": grant.risk_ceiling,
                    "revoked": bool(row["revoked"]),
                    "consumed": bool(row["consumed"]),
                    "expires_at": row["expires_at"],
                    "public_hash": grant.public_hash,
                }
            )
        return grants

    def revoke_grant(self, grant_id: str, *, principal: str) -> None:
        self._broker.revoke(grant_id, principal=principal)
        import asyncio as _asyncio

        from rosclaw.contracts.agent.agent_event import AgentEventType

        try:
            loop = _asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(
            self._append_global_event(AgentEventType.GRANT_REVOKED, {"grant_id": grant_id})
        )

    async def _append_global_event(self, event_type, payload: dict) -> None:
        """Grant events span missions; journal against the owning mission when
        resolvable, otherwise against the first mission that referenced it."""
        row = self._store.connection.execute(
            "SELECT r.mission_id FROM mission_grants g "
            "JOIN operator_requests r ON r.request_id = g.request_id "
            "WHERE g.grant_id = ?",
            (payload.get("grant_id"),),
        ).fetchone()
        mission_id = row["mission_id"] if row else ""
        if mission_id:
            await self._events.append(mission_id, event_type, payload)

    # ------------------------------------------------------------------
    async def probe(self) -> ModelProbeResult:
        try:
            return await self._gateway.probe()
        except ModelGatewayError as exc:
            return ModelProbeResult(reachable=False, error=f"{exc.kind}: {exc}")

    def status(self) -> dict:
        profile = self._gateway.profile
        return {
            "agent_enabled": self._config.enabled,
            "default_mode": self._config.default_mode,
            "body_id": self._body_id,
            "daemon_connected": self._daemon_client is not None,
            "profile": profile.name,
            "provider": profile.provider,
            "model": profile.model,
            "base_url": profile.base_url,
            "api_key_ref": profile.api_key_ref,
            "missions": len(self._store.list_missions()),
            "maturity": "experimental",
        }

    async def estop(self, reason: str, *, principal: str) -> dict:
        """PR-11：直达 rosclawd 的急停路径，绕过模型。

        无 daemon 时诚实报不可用——绝不假装已停（§12.5/§14.2）。
        """
        if self._daemon_client is None:
            raise ValidationError(
                "estop unavailable: rosclawd not connected; nothing was stopped (honest)"
            )
        return await asyncio.to_thread(
            self._daemon_client.emergency_stop,
            reason,
            source=f"operator:{principal}",
        )

    async def start_operator_socket(self, socket_path: Path | None = None) -> Path:
        """PR-11：operator.sock（peer identity + display hash + estop）。幂等。"""
        from rosclaw.agentd.operator_socket import OperatorSocketServer

        existing = getattr(self, "_operator_socket", None)
        if existing is not None:
            return existing._path
        path = socket_path or (self._home / "run" / "operator.sock")
        self._operator_socket = OperatorSocketServer(self, path)
        await self._operator_socket.start()
        return path

    async def close(self) -> None:
        if getattr(self, "_operator_socket", None) is not None:
            await self._operator_socket.stop()
            self._operator_socket = None
        await self._gateway.close()
        self._store.close()


# ----------------------------------------------------------------------
# Local HTTP API (console + CLI clients). No domain state lives in the
# HTTP layer; everything delegates to AgentService.
# ----------------------------------------------------------------------
class MissionCreate(_BaseModel):
    goal: str
    mode: str | None = None


class TurnCreate(_BaseModel):
    text: str


class DecisionCreate(_BaseModel):
    approve: bool
    principal: str = "user:local:1000"


class CommandRequestCreate(_BaseModel):
    request_id: str
    idempotency_key: str
    command_name: str
    arguments: dict = {}


class InteractionRespond(_BaseModel):
    value: object = None
    idempotency_key: str = ""


def _turn_payload(result) -> dict:
    return {
        "mission_id": result.mission_id,
        "reply": result.reply,
        "state": result.state.value,
        "tool_rounds": result.tool_rounds,
        "model_turns": result.model_turns,
        "tokens_used": result.tokens_used,
        "degraded": result.degraded,
    }


def create_app(service: AgentService):
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse

    @asynccontextmanager
    async def lifespan(_app):
        # PR-11：operator.sock 随 HTTP 服务启动（同一事件循环）。
        await service.start_operator_socket()
        yield

    app = FastAPI(title="rosclaw-agentd", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def csrf_origin_guard(request: _Request, call_next):
        """PR-11 §14.1：浏览器 Console 的 CSRF/Origin 防线。

        - 携带 Origin 的变更请求只允许本机来源（localhost/127.0.0.1/
          [::1]）——任意网页不得向 localhost approval API 发请求（§19.6）。
        - 设置 ROSCLAW_CONSOLE_TOKEN 时，变更请求必须带 X-Rosclaw-Token
          （一次性 pairing token 语义）。
        """
        if request.method in ("POST", "PUT", "DELETE", "PATCH"):
            origin = request.headers.get("origin")
            if origin:
                from urllib.parse import urlparse

                host = urlparse(origin).hostname or ""
                if host not in ("localhost", "127.0.0.1", "::1"):
                    return JSONResponse(
                        status_code=403,
                        content={"detail": f"origin {host!r} rejected (CSRF guard)"},
                    )
            pairing = os.environ.get("ROSCLAW_CONSOLE_TOKEN")
            if pairing and request.headers.get("x-rosclaw-token") != pairing:
                return JSONResponse(
                    status_code=403, content={"detail": "pairing token required"}
                )
        return await call_next(request)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "service": "rosclaw-agentd"}

    @app.get("/status")
    async def status() -> dict:
        return service.status()

    @app.get("/probe")
    async def probe() -> dict:
        result = await service.probe()
        return {
            "reachable": result.reachable,
            "models_visible": list(result.models_visible),
            "expected_model_present": result.expected_model_present,
            "chat_ok": result.chat_ok,
            "tool_call_ok": result.tool_call_ok,
            "error": result.error,
        }

    @app.post("/missions", status_code=201)
    async def create_mission(payload: MissionCreate) -> dict:
        try:
            mission = service.create_mission(payload.goal, mode=payload.mode)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return mission.model_dump(mode="json")

    @app.get("/missions")
    async def list_missions() -> list[dict]:
        return [m.model_dump(mode="json") for m in service.list_missions()]

    @app.get("/missions/{mission_id}")
    async def get_mission(mission_id: str) -> dict:
        mission = service.get_mission(mission_id)
        if mission is None:
            raise HTTPException(status_code=404, detail="mission not found")
        return mission.model_dump(mode="json")

    @app.post("/missions/{mission_id}/turns")
    async def send_turn(mission_id: str, payload: TurnCreate) -> dict:
        try:
            result = await service.send_turn(mission_id, payload.text)
        except ValidationError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _turn_payload(result)

    # ------------------------------------------------------------------
    # /v2 (PR-02): submit decoupled from event stream; TUI may disconnect
    # freely — the mission keeps running server-side.
    # ------------------------------------------------------------------
    @app.post("/v2/missions/{mission_id}/turns", status_code=202)
    async def v2_submit_turn(mission_id: str, payload: TurnCreate) -> dict:
        try:
            turn_id = await service.submit_turn_v2(mission_id, payload.text)
        except ValidationError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"turn_id": turn_id, "mission_id": mission_id, "accepted": True}

    @app.get("/v2/missions/{mission_id}/events")
    async def v2_events(
        mission_id: str,
        request: _Request,
        after_sequence: int = 0,
        visibility: str | None = None,
        follow: bool = True,
    ):
        """Long-lived SSE: replay journal from after_sequence, then live.
        ``follow=false`` returns after replay (bounded read).
        批次 B：支持标准 Last-Event-ID 头（断线恢复优先于 query 参数）。"""
        import json as _json

        from fastapi.responses import StreamingResponse

        if service.get_mission(mission_id) is None:
            raise HTTPException(status_code=404, detail="mission not found")

        last_event_id = request.headers.get("last-event-id")
        if last_event_id and last_event_id.isdigit():
            after_sequence = max(after_sequence, int(last_event_id))

        def frame(event) -> str:
            return f"id: {event.sequence}\ndata: {_json.dumps(event.model_dump(mode='json'), ensure_ascii=False)}\n\n"

        async def stream():
            for event in service.events_replay(mission_id, after_sequence=after_sequence):
                if visibility and event.visibility.value != visibility:
                    continue
                yield frame(event)
            if not follow:
                return
            queue = service.events_subscribe(mission_id)
            try:
                while True:
                    event = await queue.get()
                    if visibility and event.visibility.value != visibility:
                        continue
                    yield frame(event)
            finally:
                service.events_unsubscribe(mission_id, queue)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/v2/missions/{mission_id}/cancel")
    async def v2_cancel(mission_id: str) -> dict:
        await service.cancel_turn_v2(mission_id)
        return {"cancelled": True}

    # ------------------------------------------------------------------
    # 批次 B：capabilities / commands / snapshot / interactions
    # ------------------------------------------------------------------
    @app.get("/v1/capabilities")
    async def v1_capabilities(mission_id: str | None = None) -> dict:
        """服务端命令注册表（含 disabled_reason）。"""
        state = None
        in_flight = False
        if mission_id:
            mission = service.get_mission(mission_id)
            if mission is None:
                raise HTTPException(status_code=404, detail="mission not found")
            state = mission.state.value
            in_flight = service.turn_in_flight(mission_id)
        specs = service.commands.specs(mission_state=state, turn_in_flight=in_flight)
        return {
            "commands": [s.model_dump(mode="json") for s in specs],
            "event_stream": {"url": "/v2/missions/{mission_id}/events", "resume": "Last-Event-ID"},
            "snapshot_url": "/v1/missions/{mission_id}/snapshot",
        }

    @app.post("/v1/missions/{mission_id}/commands")
    async def v1_command(mission_id: str, payload: CommandRequestCreate) -> dict:
        from rosclaw.contracts.ui.commands import CommandRequestV1

        if service.get_mission(mission_id) is None:
            raise HTTPException(status_code=404, detail="mission not found")
        request = CommandRequestV1(
            request_id=payload.request_id,
            idempotency_key=payload.idempotency_key,
            command_name=payload.command_name.lstrip("/"),
            arguments=payload.arguments,
            mission_id=mission_id,
        )
        result = await service.commands.execute(request)
        return result.model_dump(mode="json")

    @app.get("/v1/missions/{mission_id}/snapshot")
    async def v1_snapshot(mission_id: str) -> dict:
        try:
            return service.snapshot(mission_id).model_dump(mode="json")
        except ValidationError:
            raise HTTPException(status_code=404, detail="mission not found") from None

    @app.post("/v1/interactions/{interaction_id}/respond")
    async def v1_interaction_respond(interaction_id: str, payload: InteractionRespond) -> dict:
        """通用 select/confirm/input/editor 响应。物理/授权决定永远走
        /approvals/{request_id}/decide——本端点绝不能改变授权状态。"""
        try:
            return service.interactions.respond(
                interaction_id, value=payload.value, idempotency_key=payload.idempotency_key
            )
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    @app.post("/missions/{mission_id}/turns/stream")
    async def send_turn_stream(mission_id: str, payload: TurnCreate):
        """SSE: text deltas as they arrive, then one final result event."""
        import json as _json

        from fastapi.responses import StreamingResponse

        queue: asyncio.Queue[dict] = asyncio.Queue()

        def on_delta(piece: str) -> None:
            queue.put_nowait({"type": "delta", "text": piece})

        async def run() -> None:
            try:
                result = await service.send_turn(mission_id, payload.text, on_delta)
                queue.put_nowait({"type": "final", **_turn_payload(result)})
            except Exception as exc:  # noqa: BLE001 - surfaced as SSE data
                queue.put_nowait({"type": "error", "detail": str(exc)})
            finally:
                queue.put_nowait({"type": "eof"})

        async def events():
            task = asyncio.create_task(run())
            try:
                while True:
                    event = await queue.get()
                    if event["type"] == "eof":
                        break
                    yield f"data: {_json.dumps(event, ensure_ascii=False)}\n\n"
            finally:
                if not task.done():
                    task.cancel()

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/missions/{mission_id}/usage")
    async def mission_usage(mission_id: str) -> dict:
        if service.get_mission(mission_id) is None:
            raise HTTPException(status_code=404, detail="mission not found")
        return service.mission_usage(mission_id)

    @app.get("/missions/{mission_id}/conversation")
    async def mission_conversation(mission_id: str) -> list[dict]:
        if service.get_mission(mission_id) is None:
            raise HTTPException(status_code=404, detail="mission not found")
        return service.conversation(mission_id)

    @app.post("/missions/{mission_id}/cancel")
    async def cancel(mission_id: str) -> dict:
        await service.cancel(mission_id)
        return {"cancelled": True}

    @app.get("/approvals/pending")
    async def approvals_pending(mission_id: str | None = None) -> list[dict]:
        return [r.model_dump(mode="json") for r in service.pending_approvals(mission_id)]

    @app.post("/approvals/{request_id}/decide")
    async def approvals_decide(request_id: str, payload: DecisionCreate) -> dict:
        try:
            grant = await service.decide_approval(
                request_id, principal=payload.principal, approve=payload.approve
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "approved": payload.approve,
            "grant_id": grant.grant_id if grant else None,
            "public_hash": grant.public_hash if grant else None,
        }

    @app.get("/grants")
    async def grants_list() -> list[dict]:
        return service.list_grants()

    @app.post("/grants/{grant_id}/revoke")
    async def grants_revoke(grant_id: str) -> dict:
        try:
            service.revoke_grant(grant_id, principal="user:local:1000")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"revoked": True}

    @app.get("/console", response_class=HTMLResponse)
    async def console() -> str:
        return _CONSOLE_HTML

    return app


_CONSOLE_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>ROSClaw Console</title>
<style>
body{font-family:system-ui,sans-serif;max-width:840px;margin:2rem auto;padding:0 1rem}
#log{border:1px solid #ccc;border-radius:8px;padding:1rem;height:55vh;overflow:auto;white-space:pre-wrap}
.row{display:flex;gap:.5rem;margin-top:1rem}
input{flex:1;padding:.5rem}button{padding:.5rem 1rem}
.meta{color:#666;font-size:.85em}
.turn{margin:.4rem 0}.who{font-weight:600}
.tool{color:#795e26;font-size:.85em}
</style></head><body>
<h2>ROSClaw Console <span class="meta" id="status"></span></h2>
<div class="row"><input id="goal" placeholder="新 Mission 目标（SIMULATION）">
<button onclick="createMission()">创建 Mission</button></div>
<div id="log"></div>
<div class="row"><input id="text" placeholder="对当前 Mission 说话…"
 onkeydown="if(event.key==='Enter')send()"><button onclick="send()">发送</button></div>
<script>
let missionId = null;
const logEl = () => document.getElementById('log');
function add(cls, who, text){ const d=document.createElement('div');
  d.className='turn'; d.innerHTML=`<span class="who ${cls}">${who}</span> `;
  const s=document.createElement('span'); s.textContent=text; d.appendChild(s);
  logEl().appendChild(d); logEl().scrollTop=logEl().scrollHeight; return s; }
async function status(){ const r = await fetch('/status'); const s = await r.json();
  document.getElementById('status').textContent =
    `profile=${s.profile} model=${s.model} mode=${s.default_mode}`; }
async function createMission(){
  const goal = document.getElementById('goal').value; if(!goal) return;
  const r = await fetch('/missions',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({goal})});
  const m = await r.json();
  if(r.status!==201){ add('meta','系统','创建失败: '+(m.detail||'')); return; }
  missionId = m.mission_id; add('meta','系统',`mission ${missionId} 已创建（${m.mode}）`); }
async function send(){
  if(!missionId){ add('meta','系统','请先创建 Mission'); return; }
  const t = document.getElementById('text'); const text = t.value; if(!text) return;
  t.value=''; add('','你',text);
  const span = add('','ROSClaw','');
  const r = await fetch(`/missions/${missionId}/turns/stream`,{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
  const reader = r.body.getReader(); const dec = new TextDecoder(); let buf='';
  while(true){ const {done, value} = await reader.read(); if(done) break;
    buf += dec.decode(value, {stream:true});
    const parts = buf.split('\\n\\n'); buf = parts.pop();
    for(const p of parts){ const line = p.split('\\n').find(l=>l.startsWith('data: '));
      if(!line) continue; const ev = JSON.parse(line.slice(6));
      if(ev.type==='delta'){ span.textContent += ev.text;
        logEl().scrollTop=logEl().scrollHeight; }
      else if(ev.type==='final'){
        if(!span.textContent && ev.reply) span.textContent = ev.reply;
        add('meta','状态',`state=${ev.state} 工具轮次=${ev.tool_rounds} tokens=${ev.tokens_used}`
          +(ev.degraded?` degraded=${ev.degraded}`:''));
        const u = await (await fetch(`/missions/${missionId}/usage`)).json();
        add('meta','用量',`累计 tokens=${u.total_tokens} 轮次=${u.model_turns} 成本(微单位)=${u.cost_microunits}`);}
      else if(ev.type==='error'){ span.textContent = '错误: '+ev.detail; } } } }
status();
</script></body></html>"""
