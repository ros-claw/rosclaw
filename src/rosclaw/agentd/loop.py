"""AgentLoop — the deterministic mission state machine driver (PR-NA-032).

The loop owns *control flow*; the model only proposes decisions and tool
calls inside it. Properties:

- explicit transitions via MissionStore (journal + idempotency), never
  model-invented states;
- hard round/token budgets per turn (durable counters);
- tools are allowlisted and schema-strict; tool results are appended with
  the *complete* assistant message (K3 tool-loop continuity);
- decisions are extracted from a fenced DecisionV1 JSON block, validated
  against the current context revision; invalid decisions get one bounded
  repair round, then the loop pauses (fail closed);
- pause/cancel between model calls; crash recovery comes from the store —
  the loop never reconstructs state from chat text;
- intent-specific execution (workers, physical dispatch, approvals) goes to
  injectable handlers; an unavailable handler produces an honest reply and
  a PAUSE, never a fabricated success.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from rosclaw.agentd.context.compiler import (
    CompilationError,
    ContextCompiler,
    StaleSourceError,
)
from rosclaw.agentd.context.prompt_registry import PromptInfo
from rosclaw.agentd.context.sources import ConversationMessage
from rosclaw.agentd.decisions.validator import DecisionRejectedError, DecisionValidator
from rosclaw.agentd.mission import BudgetExceededError, MissionStore
from rosclaw.agentd.models.gateway import (
    ModelGateway,
    ModelGatewayError,
    ModelTurnRequest,
    StrictTool,
)
from rosclaw.contracts.agent.decision import DecisionV1, NextIntent
from rosclaw.contracts.agent.mission import MissionSessionV1, MissionState
from rosclaw.contracts.agent.model_turn import ModelTurnResultV1
from rosclaw.contracts.common import new_id

_DECISION_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

ToolExecutor = Callable[[dict[str, Any]], Awaitable[str]]


class ToolRegistry(Protocol):
    def strict_tools(self, names: list[str]) -> list[StrictTool]: ...

    async def execute(self, name: str, arguments: dict[str, Any]) -> str: ...


class IntentHandlers(Protocol):
    """Optional executors for intents the loop itself cannot perform."""

    async def hire_worker(self, decision: DecisionV1) -> str: ...

    async def request_approval(self, decision: DecisionV1) -> str: ...

    async def request_action(self, decision: DecisionV1) -> str: ...

    async def team_coordinate(self, decision: DecisionV1) -> str: ...


@dataclass
class LoopTurnResult:
    mission_id: str
    reply: str
    state: MissionState
    decisions: list[DecisionV1] = field(default_factory=list)
    tool_rounds: int = 0
    model_turns: int = 0
    tokens_used: int = 0
    degraded: str | None = None  # honest degradation note, if any


class AgentLoop:
    def __init__(
        self,
        *,
        store: MissionStore,
        compiler: ContextCompiler,
        gateway: ModelGateway,
        prompt: PromptInfo,
        tools: ToolRegistry | None = None,
        handlers: IntentHandlers | None = None,
        actor_id: str,
        max_tool_rounds: int = 12,
        max_decision_repairs: int = 2,
        usage_recorder=None,
    ) -> None:
        self._store = store
        self._compiler = compiler
        self._gateway = gateway
        self._prompt = prompt
        self._tools = tools
        self._handlers = handlers
        self._actor_id = actor_id
        self._max_tool_rounds = max_tool_rounds
        self._max_decision_repairs = max_decision_repairs
        self._cancel_requested = False
        self._conversation: list[dict[str, Any]] = []
        self._conversation_loaded_for: str | None = None
        self._persisted_count = 0
        self._usage_recorder = usage_recorder

    # ------------------------------------------------------------------
    def request_cancel(self) -> None:
        self._cancel_requested = True

    def _restore_conversation(self, mission_id: str) -> None:
        """Reload journaled conversation on first use (resume continuity)."""
        if self._conversation_loaded_for == mission_id:
            return
        self._conversation_loaded_for = mission_id
        if not self._conversation:
            self._conversation = self._store.conversation(mission_id)
            self._persisted_count = len(self._conversation)

    def _persist_conversation(self, mission_id: str) -> None:
        new_messages = self._conversation[self._persisted_count :]
        if new_messages:
            self._store.append_conversation(mission_id, new_messages, actor_id=self._actor_id)
            self._persisted_count = len(self._conversation)

    # ------------------------------------------------------------------
    async def run_user_turn(
        self,
        mission: MissionSessionV1,
        user_text: str,
        *,
        now: datetime,
        on_text_delta=None,
    ) -> LoopTurnResult:
        self._restore_conversation(mission.mission_id)
        self._trace_id = new_id("tr")
        display_filter = None
        if on_text_delta is not None:
            from rosclaw.agentd.stream_filter import DecisionBlockFilter

            display_filter = DecisionBlockFilter(on_text_delta)
        try:
            return await self._run_user_turn_inner(
                mission,
                user_text,
                now=now,
                on_text_delta=display_filter.feed if display_filter else None,
            )
        finally:
            if display_filter is not None:
                display_filter.flush()
            self._persist_conversation(mission.mission_id)

    async def _run_user_turn_inner(
        self,
        mission: MissionSessionV1,
        user_text: str,
        *,
        now: datetime,
        on_text_delta=None,
    ) -> LoopTurnResult:
        self._cancel_requested = False
        result = LoopTurnResult(mission_id=mission.mission_id, reply="", state=mission.state)

        if mission.state is MissionState.IDLE:
            mission = self._store.transition(
                mission.mission_id,
                MissionState.UNDERSTAND,
                reason_code="new_goal",
                actor_id=self._actor_id,
                trace_id=self._trace_id,
            )
        self._conversation.append({"role": "user", "content": user_text})

        # Compile embodied context (fail closed → honest degradation). The
        # revision is bumped durably so a decision bound to an older revision
        # is rejected next turn (stale-decision guard, 总纲 §5.4.13).
        context_revision = self._store.bump_context_revision(mission.mission_id)
        try:
            bundle = self._compiler.compile(
                mission,
                self._store.get_task_graph(mission.mission_id),
                [
                    ConversationMessage(role=m["role"], content=str(m["content"]))
                    for m in self._conversation
                ],
                context_revision=context_revision,
                now=now,
                # Stable per mission: the model can reliably echo it from the
                # trusted header; freshness is carried by context_revision and
                # bundle_hash, not by a rotating id.
                context_id=f"ctx_{mission.mission_id}",
            )
        except StaleSourceError as exc:
            result.degraded = f"stale_source: {exc}"
            result.reply = (
                f"当前身体/自我状态已过期，我无法诚实地继续推理。需要先刷新观测（{exc}）。"
            )
            result.state = self._safe_transition(
                mission.mission_id, mission.state, MissionState.WAIT_INPUT, "stale_source"
            )
            return result
        except CompilationError as exc:
            result.degraded = f"compilation_failed: {exc}"
            result.reply = f"上下文编译失败（fail closed）：{exc}"
            result.state = mission.state
            return result

        # Attribution: the compile manifest (with prompt hash, §6.3) is
        # persisted before any model call.
        self._store.record_context_manifest(bundle, prompt_hash=self._prompt.content_hash)

        if mission.state is MissionState.UNDERSTAND:
            mission = self._store.transition(
                mission.mission_id,
                MissionState.GROUND,
                reason_code="intent_understood",
                actor_id=self._actor_id,
                trace_id=self._trace_id,
            )
        if mission.state is MissionState.GROUND:
            mission = self._store.transition(
                mission.mission_id,
                MissionState.PLAN,
                reason_code="context_compiled",
                actor_id=self._actor_id,
                trace_id=self._trace_id,
            )

        system_prompt = self._render_system_prompt(bundle)
        candidate_tools = bundle.layers.capabilities.candidate_tools or []
        tools: list[StrictTool] = []
        if self._tools is not None and candidate_tools:
            tools = self._tools.strict_tools(candidate_tools)

        repairs_left = self._max_decision_repairs
        for _round in range(self._max_tool_rounds + 1):
            if self._cancel_requested:
                result.reply = "已按你的要求取消。"
                result.state = self._safe_transition(
                    mission.mission_id,
                    self._current_state(mission.mission_id),
                    MissionState.FAILED,
                    "cancelled_by_user",
                )
                return result
            # 主动 microcompact：估算超阈值先折叠旧 tool result / 中段历史。
            self._maybe_microcompact(result)
            request = ModelTurnRequest(
                system_prompt=system_prompt,
                messages=list(self._conversation),
                tools=tools,
                max_output_tokens=self._gateway.profile.max_output_tokens,
                mission_id=mission.mission_id,
                context_id=bundle.context_id,
                context_revision=bundle.context_revision,
            )
            try:
                turn = await self._gateway.complete_stream(request, on_text_delta=on_text_delta)
            except ModelGatewayError as exc:
                from rosclaw.agentd.context.compact import is_context_overflow, microcompact

                if is_context_overflow(exc.kind, str(exc)):
                    # Reactive compact（openharness 模式）：压缩后重试一次。
                    self._conversation, _ = microcompact(self._conversation, keep_recent=4)
                    request.messages = list(self._conversation)
                    try:
                        turn = await self._gateway.complete_stream(
                            request, on_text_delta=on_text_delta
                        )
                        result.degraded = "reactive_compacted"
                    except ModelGatewayError as exc2:
                        result.degraded = f"model_error: {exc2.kind}"
                        result.reply = f"模型调用失败（{exc2.kind}）：{exc2}"
                        result.state = self._current_state(mission.mission_id)
                        return result
                else:
                    result.degraded = f"model_error: {exc.kind}"
                    result.reply = f"模型调用失败（{exc.kind}）：{exc}"
                    result.state = self._current_state(mission.mission_id)
                    return result
            result.model_turns += 1
            if not self._record_usage(mission.mission_id, turn, result):
                # 预算超限（§4.2）：必须进入 WAIT_INPUT/SUSPENDED，不得继续
                # 执行本轮决策。
                result.degraded = "budget_exceeded"
                result.reply = (
                    "本轮模型/费用预算已超限，mission 已暂停等待你的指示"
                    "（提高预算或缩小目标）。未执行后续决策。"
                )
                current = self._current_state(mission.mission_id)
                result.state = self._safe_transition(
                    mission.mission_id, current, MissionState.WAIT_INPUT, "budget_exceeded"
                )
                if result.state == current and current is MissionState.MONITOR:
                    result.state = self._safe_transition(
                        mission.mission_id, current, MissionState.SUSPENDED, "budget_exceeded"
                    )
                return result

            if turn.tool_calls:
                if self._tools is None:
                    result.reply = "模型请求了工具，但当前没有可用工具执行器。"
                    result.degraded = "tools_unavailable"
                    result.state = self._current_state(mission.mission_id)
                    return result
                # K3 continuity: append the *complete* assistant message.
                self._conversation.append(dict(turn.assistant_message))
                for call in turn.tool_calls:
                    try:
                        arguments = json.loads(call.arguments_json)
                    except json.JSONDecodeError:
                        arguments = {}
                    try:
                        output = await self._tools.execute(call.name, arguments)
                    except Exception as exc:  # noqa: BLE001 - surfaced as data
                        output = json.dumps(
                            {"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False
                        )
                    self._conversation.append(
                        {"role": "tool", "tool_call_id": call.call_id, "content": output}
                    )
                    result.tool_rounds += 1
                continue

            # Final answer turn: extract and validate a decision if present.
            decision, saw_malformed = self._extract_decision(turn)
            if decision is None and saw_malformed:
                # A mangled/truncated protocol block must not pass as prose.
                if repairs_left > 0:
                    repairs_left -= 1
                    self._conversation.append(dict(turn.assistant_message))
                    self._conversation.append(
                        {
                            "role": "user",
                            "content": (
                                "Your DecisionV1 JSON block was malformed or truncated. "
                                "Re-emit exactly one COMPLETE, valid DecisionV1 JSON "
                                "object in a fenced code block, using the minimal field "
                                "set from the system instructions."
                            ),
                        }
                    )
                    continue
                result.reply = "模型多次给出残缺的决策块，已暂停（fail closed）。"
                result.degraded = "decision_rejected: malformed_block"
                result.state = self._safe_transition(
                    mission.mission_id,
                    self._current_state(mission.mission_id),
                    MissionState.FAILED,
                    "decision_rejected",
                )
                return result
            if decision is not None:
                validator = DecisionValidator(
                    current_context_id=bundle.context_id,
                    current_context_revision=bundle.context_revision,
                )
                try:
                    validator.validate(decision, mission_id=mission.mission_id)
                except DecisionRejectedError as exc:
                    # Attribution: rejected decisions are evidence too (§12.3).
                    self._store.record_decision(
                        decision,
                        validated=False,
                        reason_code=exc.reason_code,
                        actor_id=self._actor_id,
                    )
                    if repairs_left > 0:
                        repairs_left -= 1
                        self._conversation.append(dict(turn.assistant_message))
                        self._conversation.append(
                            {
                                "role": "user",
                                "content": (
                                    f"Your DecisionV1 was rejected ({exc.reason_code}: {exc}). "
                                    "Return a corrected DecisionV1 JSON block. "
                                    f"Copy EXACTLY: context_id={bundle.context_id}, "
                                    f"context_revision={bundle.context_revision}. "
                                    "Do not fabricate execution results, receipts, or "
                                    "reference ids — nothing has been dispatched."
                                ),
                            }
                        )
                        continue
                    result.reply = "模型多次给出无法校验的决策，已暂停（fail closed）。"
                    result.degraded = f"decision_rejected: {exc.reason_code}"
                    result.state = self._safe_transition(
                        mission.mission_id,
                        self._current_state(mission.mission_id),
                        MissionState.FAILED,
                        "decision_rejected",
                    )
                    return result
                self._store.record_decision(
                    decision, validated=True, reason_code=None, actor_id=self._actor_id
                )
                result.decisions.append(decision)

            reply, new_state = await self._apply_decision(mission, decision, turn)
            result.reply = reply
            result.state = new_state
            self._conversation.append({"role": "assistant", "content": reply})
            return result

        result.reply = "已达到本轮工具调用上限，暂停等待进一步指示。"
        result.degraded = "tool_rounds_exhausted"
        result.state = self._current_state(mission.mission_id)
        return result

    # ------------------------------------------------------------------
    def _render_system_prompt(self, bundle) -> str:
        parts = [self._prompt.text]
        layers = bundle.layers
        parts.append(
            "\nTRUSTED CONTEXT (compiled)\n"
            f"context_id: {bundle.context_id}\n"
            f"context_revision: {bundle.context_revision}\n"
            f"bundle_hash: {bundle.bundle_hash}"
        )
        parts.append(f"\n[EMBODIMENT — trusted]\n{layers.embodiment.inline_summary}")
        parts.append(f"\n[DYNAMIC SELF — trusted]\n{layers.dynamic_self.inline_summary}")
        if layers.capabilities.inline_summary:
            parts.append(f"\n[CAPABILITIES — trusted]\n{layers.capabilities.inline_summary}")
        parts.append(f"\n[MISSION — trusted]\n{layers.mission.inline_summary}")
        if layers.memory and layers.memory.inline_summary:
            parts.append(f"\n[MEMORY — evidence-gated]\n{layers.memory.inline_summary}")
        if layers.organization and layers.organization.inline_summary:
            parts.append(f"\n[WORKERS & TEAM — trusted]\n{layers.organization.inline_summary}")
        parts.append(f"\n[SAFETY & CONSENT — trusted]\n{layers.safety.inline_summary}")
        parts.append(
            "\nWhen you are ready to conclude this cognitive step, append exactly one "
            "DecisionV1 JSON object in a fenced code block. Copy context_id and "
            "context_revision EXACTLY from the TRUSTED CONTEXT header above — never "
            "invent them. Minimal valid example:\n"
            '{"schema_version": "rosclaw.decision.v1", "decision_id": "dec_<unique>", '
            f'"mission_id": "{bundle.mission_id}", "context_id": "{bundle.context_id}", '
            f'"context_revision": {bundle.context_revision}, "next_intent": "ANSWER", '
            '"summary": "<one sentence>", "evidence_refs": []}\n'
            "Optional fields must use these exact shapes or be omitted entirely: "
            "assumptions: list of objects with keys claim/evidence_ref/confidence; "
            "uncertainty: object with keys level (LOW|MODERATE|HIGH) and reasons (list); "
            "verification: object with keys schema_ref and verifiers (list); "
            "proposed_operation: object with keys type and payload_ref/payload; "
            "on_failure: object with keys intent and reason. Do not add other keys. "
            "The operation payload must NEVER contain the keys mode, permit, "
            "signature, or credential — authorization is referenced by grant_id only.\n"
            "Allowed proposed_operation.type per next_intent (any other type is "
            "rejected): OBSERVE→observe|refresh_state; PLAN_PATCH→task_graph_patch; "
            "HIRE_WORKER→create_work_order; TEAM_COORDINATE→team_message|team_bid|"
            "team_task_claim; REQUEST_APPROVAL→approval_request; "
            "REQUEST_ACTION→request_action; VERIFY→verify_receipt|verify_observation; "
            "FAIL_SAFE→estop_request; ANSWER/WAIT/PAUSE→no operation."
        )
        return "\n".join(parts)

    def _extract_decision(self, turn: ModelTurnResultV1) -> tuple[DecisionV1 | None, bool]:
        """Return (decision, saw_malformed_attempt)."""
        content = turn.content or ""
        for match in _DECISION_BLOCK_RE.finditer(content):
            try:
                payload = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            if payload.get("schema_version") != "rosclaw.decision.v1":
                continue
            try:
                return DecisionV1.model_validate_contract(payload), False
            except Exception:  # noqa: BLE001 - try next block / treat as text
                continue
        # No valid block: was the model *trying* to emit one?
        saw_attempt = '"schema_version": "rosclaw.decision.v1"' in content or (
            "```" in content and "rosclaw.decision.v1" in content
        )
        return None, saw_attempt

    async def _apply_decision(
        self,
        mission: MissionSessionV1,
        decision: DecisionV1 | None,
        turn: ModelTurnResultV1,
    ) -> tuple[str, MissionState]:
        current = self._current_state(mission.mission_id)
        intent = decision.next_intent if decision else NextIntent.ANSWER

        if intent in (NextIntent.PAUSE, NextIntent.FAIL_SAFE):
            note = decision.summary if decision else ""
            return (
                f"已暂停。{note}",
                self._safe_transition(
                    mission.mission_id, current, MissionState.FAILED, intent.value.lower()
                ),
            )

        if intent is NextIntent.HIRE_WORKER:
            if self._handlers is None:
                return (
                    "需要雇佣 Worker，但 Worker Fabric 尚未启用；"
                    "我将不伪造委派结果，先给出本地可行方案或等待 Worker 上线。",
                    current,
                )
            staffed = self._safe_transition(
                mission.mission_id, current, MissionState.STAFF, "hire_worker"
            )
            text = await self._handlers.hire_worker(decision)  # type: ignore[arg-type]
            next_state = self._safe_transition(
                mission.mission_id, staffed, MissionState.VALIDATE, "work_result_verified"
            )
            if next_state is MissionState.VALIDATE:
                next_state = self._safe_transition(
                    mission.mission_id,
                    next_state,
                    MissionState.LEARN,
                    "delegation_complete",
                )
                if next_state is MissionState.LEARN:
                    next_state = self._safe_transition(
                        mission.mission_id, next_state, MissionState.IDLE, "mission_complete"
                    )
            return text, next_state

        if intent is NextIntent.REQUEST_APPROVAL:
            if self._handlers is None:
                return (
                    "该步骤需要人类授权，但授权通道不可用；已停止继续推进（fail closed）。",
                    current,
                )
            text = await self._handlers.request_approval(decision)  # type: ignore[arg-type]
            next_state = current
            if current is MissionState.PLAN:
                next_state = self._safe_transition(
                    mission.mission_id, current, MissionState.VALIDATE, "plan_ready"
                )
            if next_state is MissionState.VALIDATE:
                next_state = self._safe_transition(
                    mission.mission_id,
                    next_state,
                    MissionState.WAIT_APPROVAL,
                    "approval_requested",
                )
            return text, next_state

        if intent is NextIntent.REQUEST_ACTION:
            if self._handlers is None:
                return (
                    "需要物理动作，但当前没有动作通道；未提交任何动作请求（fail closed）。",
                    current,
                )
            text = await self._handlers.request_action(decision)  # type: ignore[arg-type]
            # Verified grant in SIMULATION: no physical dispatch exists yet;
            # complete honestly rather than pretending a receipt.
            next_state = current
            if current is MissionState.WAIT_APPROVAL:
                next_state = self._safe_transition(
                    mission.mission_id, current, MissionState.DISPATCH, "grant_effective"
                )
            elif current is MissionState.PLAN:
                next_state = self._safe_transition(
                    mission.mission_id, current, MissionState.VALIDATE, "plan_ready"
                )
            if next_state is MissionState.DISPATCH:
                next_state = self._safe_transition(
                    mission.mission_id, next_state, MissionState.MONITOR, "sim_no_dispatch"
                )
            if next_state is MissionState.MONITOR:
                next_state = self._safe_transition(
                    mission.mission_id, next_state, MissionState.VERIFY, "sim_verified_grant"
                )
            if next_state in (MissionState.VERIFY, MissionState.VALIDATE):
                next_state = self._safe_transition(
                    mission.mission_id, next_state, MissionState.LEARN, "action_path_complete"
                )
            if next_state is MissionState.LEARN:
                next_state = self._safe_transition(
                    mission.mission_id, next_state, MissionState.IDLE, "mission_complete"
                )
            return text, next_state

        if intent is NextIntent.TEAM_COORDINATE:
            if self._handlers is None:
                return (
                    "Team Fabric 尚未启用；未进行团队协调（fail closed）。",
                    current,
                )
            text = await self._handlers.team_coordinate(decision)  # type: ignore[arg-type]
            next_state = current
            if current is MissionState.PLAN:
                next_state = self._safe_transition(
                    mission.mission_id, current, MissionState.VALIDATE, "plan_ready"
                )
                if next_state is MissionState.VALIDATE:
                    next_state = self._safe_transition(
                        mission.mission_id,
                        next_state,
                        MissionState.LEARN,
                        "team_coordination_complete",
                    )
                    if next_state is MissionState.LEARN:
                        next_state = self._safe_transition(
                            mission.mission_id,
                            next_state,
                            MissionState.IDLE,
                            "mission_complete",
                        )
            return text, next_state

        # ANSWER / OBSERVE / VERIFY / WAIT / PLAN_PATCH(无处理器时按文本回复)
        raw_reply = turn.content.strip() if turn.content else ""
        if decision is not None:
            # The DecisionV1 block is machine output, not user-facing prose.
            raw_reply = _DECISION_BLOCK_RE.sub("", raw_reply).strip()
        reply = raw_reply or (decision.summary if decision else "")
        if current is MissionState.PLAN:
            next_state = self._safe_transition(
                mission.mission_id, current, MissionState.VALIDATE, "plan_ready"
            )
            if next_state is MissionState.VALIDATE:
                next_state = self._safe_transition(
                    mission.mission_id,
                    next_state,
                    MissionState.LEARN,
                    "answer_requires_no_dispatch",
                )
                if next_state is MissionState.LEARN:
                    next_state = self._safe_transition(
                        mission.mission_id, next_state, MissionState.IDLE, "mission_complete"
                    )
            return reply, next_state
        return reply, current

    def _record_usage(
        self, mission_id: str, turn: ModelTurnResultV1, result: LoopTurnResult
    ) -> bool:
        """Persist usage + charge budgets. False when a budget is exceeded."""
        tokens = turn.usage.total_tokens or (
            turn.usage.prompt_tokens + turn.usage.completion_tokens
        )
        result.tokens_used += tokens
        if self._usage_recorder is not None:
            self._usage_recorder.record(turn)
        try:
            self._store.add_budget_usage(
                mission_id,
                {
                    "model_tokens": tokens,
                    "monetary_microunits": turn.usage.cost_microunits,
                },
            )
        except BudgetExceededError:
            # The durable counter rolled back; the caller must park the
            # mission instead of executing the decision (§4.2).
            return False
        return True

    def _current_state(self, mission_id: str) -> MissionState:
        mission = self._store.get_mission(mission_id)
        if mission is None:
            return MissionState.FAILED
        return mission.state

    def _maybe_microcompact(self, result: LoopTurnResult) -> None:
        """Proactive compaction when the conversation estimate exceeds 80%
        of the configured input budget."""
        from rosclaw.agentd.context.compact import (
            estimate_messages_tokens,
            microcompact,
        )

        budget = getattr(self._compiler, "_max_input_tokens", 120_000)
        if estimate_messages_tokens(self._conversation) <= budget * 0.8:
            return
        self._conversation, folded = microcompact(self._conversation)
        result.degraded = f"microcompacted:{folded}"

    def _safe_transition(
        self,
        mission_id: str,
        from_state: MissionState,
        to_state: MissionState,
        reason: str,
    ) -> MissionState:
        """Transition if legal; otherwise stay (never force an illegal edge)."""
        try:
            mission = self._store.transition(
                mission_id,
                to_state,
                reason_code=reason,
                actor_id=self._actor_id,
                trace_id=getattr(self, "_trace_id", None),
            )
            return mission.state
        except Exception:  # noqa: BLE001 - illegal edge: keep current state
            return from_state
