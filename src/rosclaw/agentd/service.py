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
    ConfigConsentSource,
    DaemonSelfSource,
    EmptyMemorySource,
    ResolverBodySource,
    SimBodySource,
    SimSelfSource,
    StaticCapabilitySource,
)
from rosclaw.agentd.tools import SIM_BODY_TOOL, SIM_STATE_TOOL, BuiltinToolRegistry
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
        tool_names = [SIM_BODY_TOOL]
        if self._simulation_body:
            tool_names.insert(0, SIM_STATE_TOOL)
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
                capabilities=StaticCapabilitySource(tool_names),
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
            candidates = [(p, OpenAICompatGateway(p)) for p in chain]
            # 单 profile 也走 FailoverGateway：统一的 cooldown/RPM 语义。
            self._gateway = FailoverGateway(candidates)
        self._prompt = load_prompt("native_agent_v1.md")
        self._loops: dict[str, AgentLoop] = {}
        self._lock = asyncio.Lock()
        self._usage = UsageRecorder(self._store.connection)
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
                tools=self._tools,
                handlers=self._handlers,
                actor_id=self.actor_id,
                max_tool_rounds=self._config.max_tool_rounds,
                usage_recorder=self._usage,
            )
            self._loops[mission_id] = loop
        return loop

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
        async with self._lock:
            if self._handlers is not None:
                self._handlers._mode = mission.mode.value
                self._handlers._principal = mission.owner_principal
            loop = self._loop_for(mission_id)
            return await loop.run_user_turn(
                mission, text, now=datetime.now(UTC), on_text_delta=on_text_delta
            )

    def mission_usage(self, mission_id: str) -> dict:
        return self._usage.mission_totals(mission_id)

    def conversation(self, mission_id: str) -> list[dict]:
        return self._store.conversation(mission_id)

    async def cancel(self, mission_id: str) -> None:
        loop = self._loops.get(mission_id)
        if loop is not None:
            loop.request_cancel()

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

    async def close(self) -> None:
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
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="rosclaw-agentd", version="0.1.0")

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
