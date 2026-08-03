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
from rosclaw.agentd.decisions.submit_tool import (
    SUBMIT_DECISION_TOOL,
    build_decision_payload,
)
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
        event_sink=None,
        decision_protocol: str = "tool_call",
        legacy_fenced_json_fallback: bool = True,
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
        self._event_sink = event_sink
        self._decision_protocol = decision_protocol
        self._legacy_fenced_json_fallback = legacy_fenced_json_fallback

    # ------------------------------------------------------------------
    async def _emit(self, type, payload: dict, *, visibility=None, task_id=None) -> None:
        """Journaled UI event (PR-02). Never blocks the domain operation."""
        if self._event_sink is None:
            return
        import contextlib

        with contextlib.suppress(Exception):  # event fan-out must not break the loop
            await self._event_sink(type, payload, visibility=visibility, task_id=task_id)

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
        self._current_bundle = bundle
        from rosclaw.contracts.agent.agent_event import AgentEventType, Visibility

        await self._emit(
            AgentEventType.CONTEXT_USAGE,
            {
                "context_id": bundle.context_id,
                "context_revision": bundle.context_revision,
                "layer_tokens": {
                    "constitution": bundle.layers.constitution.token_estimate,
                    "embodiment": bundle.layers.embodiment.token_estimate,
                    "capabilities": bundle.layers.capabilities.token_estimate,
                    "mission": bundle.layers.mission.token_estimate,
                    "safety": bundle.layers.safety.token_estimate,
                },
            },
            visibility=Visibility.DEBUG,
        )
        await self._emit(
            AgentEventType.MODEL_SELECTED,
            {"profile": self._gateway.profile.name, "model": self._gateway.profile.model},
            visibility=Visibility.DEBUG,
        )

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
            # PR-05: when the registry is catalog-backed, injection goes
            # through resolver hard filters + relevance ranking (≤12).
            resolve = getattr(self._tools, "resolve_tools", None)
            if resolve is not None:
                tools = resolve(
                    candidate_tools,
                    mode=mission.mode.value,
                    task_hint=mission.goal.text if mission.goal else "",
                )
            else:
                tools = self._tools.strict_tools(candidate_tools)
        if self._decision_protocol == "tool_call":
            from rosclaw.agentd.decisions.submit_tool import submit_decision_tool

            tools.append(submit_decision_tool())

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
            # 主动压缩：按 history_budget 持久化 compact（PR-07）。
            await self._maybe_compact(mission, result)
            from rosclaw.contracts.agent.agent_event import AgentEventType, Visibility

            await self._emit(
                AgentEventType.MODEL_REQUEST_STARTED,
                {"profile": self._gateway.profile.name, "model": self._gateway.profile.model},
                visibility=Visibility.DEBUG,
            )
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
                if on_text_delta is not None:
                    await self._emit(AgentEventType.MESSAGE_STARTED, {}, visibility=Visibility.DEBUG)
                turn = await self._gateway.complete_stream(request, on_text_delta=on_text_delta)
            except ModelGatewayError as exc:
                from rosclaw.agentd.context.compact import is_context_overflow

                if is_context_overflow(exc.kind, str(exc)):
                    # Reactive compact：走持久化压缩（PR-07 引擎），绝不原地
                    # 改写 self._conversation —— 那会让 _persisted_count 指向
                    # 错误边界、新消息落不了盘（补充实施文档 §3.3）。
                    await self._emit(AgentEventType.COMPACTION_STARTED, {"reason": "overflow"})
                    await self.compact_conversation(
                        mission, reason="overflow", keep_recent_tokens=8_000
                    )
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
            await self._emit(
                AgentEventType.MODEL_REQUEST_ENDED,
                {
                    "prompt_tokens": turn.usage.prompt_tokens,
                    "completion_tokens": turn.usage.completion_tokens,
                    "finish_reason": turn.finish_reason,
                },
                visibility=Visibility.DEBUG,
            )
            if on_text_delta is not None:
                await self._emit(AgentEventType.MESSAGE_ENDED, {}, visibility=Visibility.DEBUG)
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

            # PR-06：先处理内部协议工具提交；其余 tool_calls 正常执行。
            decision: DecisionV1 | None = None
            saw_malformed = False
            if turn.tool_calls:
                # K3 continuity: append the *complete* assistant message.
                self._conversation.append(dict(turn.assistant_message))
                for call in turn.tool_calls:
                    await self._emit(
                        AgentEventType.MODEL_TOOL_CALL_PROPOSED,
                        {"name": call.name, "call_id": call.call_id},
                        visibility=Visibility.DEBUG,
                    )
                    try:
                        arguments = json.loads(call.arguments_json)
                    except json.JSONDecodeError:
                        arguments = {}
                    if call.name == SUBMIT_DECISION_TOOL:
                        # 内部协议工具：不执行，只提交 DecisionV1（服务端补齐绑定）。
                        try:
                            submitted = build_decision_payload(
                                arguments,
                                mission_id=mission.mission_id,
                                context_id=bundle.context_id,
                                context_revision=bundle.context_revision,
                            )
                            decision = DecisionV1.model_validate_contract(submitted)
                            # Kimi/OpenAI 要求每个 tool_call_id 都有响应消息：
                            # 内部协议工具也必须回执（标记为协议确认，不执行）。
                            self._conversation.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": call.call_id,
                                    "content": json.dumps(
                                        {
                                            "accepted": True,
                                            "decision_id": decision.decision_id,
                                            "note": "protocol tool: decision registered, not executed",
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            )
                        except Exception as exc:  # noqa: BLE001 - 回执错误继续修复
                            self._conversation.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": call.call_id,
                                    "content": json.dumps(
                                        {"error": f"invalid DecisionV1: {exc}"},
                                        ensure_ascii=False,
                                    ),
                                }
                            )
                        break
                    if self._tools is None:
                        result.reply = "模型请求了工具，但当前没有可用工具执行器。"
                        result.degraded = "tools_unavailable"
                        result.state = self._current_state(mission.mission_id)
                        return result
                    await self._emit(
                        AgentEventType.TOOL_STARTED,
                        {"name": call.name, "arguments": arguments},
                    )
                    tool_ok = True
                    try:
                        output = await self._tools.execute(call.name, arguments)
                    except Exception as exc:  # noqa: BLE001 - surfaced as data
                        tool_ok = False
                        output = json.dumps(
                            {"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False
                        )
                    await self._emit(
                        AgentEventType.TOOL_COMPLETED,
                        {"name": call.name, "ok": tool_ok},
                    )
                    self._conversation.append(
                        {"role": "tool", "tool_call_id": call.call_id, "content": output}
                    )
                    result.tool_rounds += 1
                if decision is None:
                    continue

            # Final answer turn: 决策来自协议工具或（legacy fallback）fenced JSON。
            if decision is None:
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
                    await self._emit(
                        AgentEventType.VERIFICATION_COMPLETED,
                        {
                            "decision_id": decision.decision_id,
                            "validated": False,
                            "reason_code": exc.reason_code,
                        },
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
                await self._emit(
                    AgentEventType.VERIFICATION_COMPLETED,
                    {
                        "decision_id": decision.decision_id,
                        "validated": True,
                        "intent": decision.next_intent.value,
                    },
                )
                result.decisions.append(decision)

            # §5.2 OBSERVE：执行只读观测、记录证据、context_revision+1，
            # 然后让模型带着新证据继续推理（不结束回合）。
            if decision is not None and decision.next_intent is NextIntent.OBSERVE:
                handled = await self._handle_observe(mission, decision, bundle, result)
                if handled:
                    continue

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
    async def _handle_observe(
        self,
        mission: MissionSessionV1,
        decision: DecisionV1,
        bundle,
        result: LoopTurnResult,
    ) -> bool:
        """§5.2 ObserveIntentHandler. True = evidence appended, keep reasoning."""
        payload = (decision.proposed_operation.payload if decision.proposed_operation else {}) or {}
        tool_name = payload.get("tool") or payload.get("capability")
        # 只允许显式只读/观测类工具；MCP action tool 永远不走这条路。
        if not tool_name or self._tools is None:
            self._conversation.append(
                {
                    "role": "user",
                    "content": (
                        "[observation] OBSERVE requested but no read-only tool was "
                        "named/available. Re-decide: name a specific read-only tool "
                        "from the capabilities layer, or choose another intent."
                    ),
                }
            )
            return True
        from rosclaw.agentd.tooling.strict_schema import canonical_name

        allowed = {canonical_name(t.name) for t in self._tools.strict_tools([tool_name])}
        if canonical_name(tool_name) not in allowed:
            self._conversation.append(
                {
                    "role": "user",
                    "content": (
                        f"[observation] tool {tool_name!r} is not an available read-only "
                        "capability here. Choose an available one or another intent."
                    ),
                }
            )
            return True
        arguments = payload.get("arguments") or {}
        error: str | None = None
        try:
            output = await self._tools.execute(tool_name, arguments)
            ok = True
        except Exception as exc:  # noqa: BLE001 - surfaced as data
            output = json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)
            error = f"{type(exc).__name__}: {exc}"
            ok = False
        # PR-05 证据封装：EvidenceEnvelope（时间戳/body/来源/证据类/freshness/
        # artifact ref/UNTRUSTED 包裹）；legacy registry 回退到旧格式。
        import hashlib

        digest = hashlib.sha256(output.encode()).hexdigest()[:24]
        artifact_ref = f"artifact://observation/sha256:{digest}"
        make_envelope = getattr(self._tools, "evidence_envelope", None)
        if make_envelope is not None:
            envelope = make_envelope(
                tool_name, output, body_id=bundle.body_binding.body_id, error=error
            )
            evidence_note = envelope.render_for_model()
            artifact_ref = envelope.artifact_ref or artifact_ref
        else:
            from datetime import UTC, datetime

            evidence_note = (
                f"[observation — evidence]\n"
                f"tool: {tool_name}\n"
                f"timestamp: {datetime.now(UTC).isoformat()}\n"
                f"body_id: {bundle.body_binding.body_id}\n"
                f"source: native_tool\n"
                f"evidence_class: measured\n"
                f"artifact_ref: {artifact_ref}\n"
                f"result: {output[:2000]}"
            )
        self._conversation.append(
            {
                "role": "tool",
                "tool_call_id": f"obs_{digest}",
                "content": evidence_note,
                "atomic_group": f"obs_{digest}",
            }
        )
        # context_revision += 1（权威存储，非内存计数）。
        self._store.bump_context_revision(mission.mission_id)
        from rosclaw.contracts.agent.agent_event import AgentEventType

        await self._emit(
            AgentEventType.TOOL_COMPLETED,
            {"name": tool_name, "ok": ok, "observation": True, "artifact_ref": artifact_ref},
        )
        # 告诉模型观测已到达，继续推理。
        self._conversation.append(
            {
                "role": "user",
                "content": (
                    "[observation] Fresh evidence attached above. Continue reasoning "
                    "with it and emit your next DecisionV1 when ready."
                ),
                "atomic_group": f"obs_{digest}",
            }
        )
        result.tool_rounds += 1
        return True

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
            " For REAL REQUEST_APPROVAL, proposed_operation.payload MUST include a non-empty "
            "capability_id and an arguments object copied exactly from a trusted capability "
            "contract, in addition to the human-readable title/summary/risk fields. Never "
            "request approval for guessed or incomplete action arguments. After approval, read "
            "the Active mission grant in SAFETY & CONSENT and reference its exact public grant_id "
            "in REQUEST_ACTION."
        )
        return "\n".join(parts)

    def _extract_decision(self, turn: ModelTurnResultV1) -> tuple[DecisionV1 | None, bool]:
        """Return (decision, saw_malformed_attempt)."""
        if not self._legacy_fenced_json_fallback:
            # PR-06：禁用 fenced JSON 时，块标记算"模型走了旧协议"的修复信号。
            content = turn.content or ""
            attempt = "rosclaw.decision.v1" in content
            return None, attempt
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

    def _completion_clear(self, mission_id: str) -> bool:
        """ANSWER/委派后才能 LEARN 的硬条件（大纲 §5.1）。"""
        from rosclaw.contracts.agent.task_graph import TaskStatus

        graph = self._store.get_task_graph(mission_id)
        open_nodes = [
            n
            for n in graph.nodes
            if n.status not in (TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED)
        ]
        if open_nodes:
            return False
        running = self._store.connection.execute(
            "SELECT COUNT(*) AS n FROM work_orders WHERE mission_id = ? AND status IN "
            "('DRAFT','OFFERED','CLAIMED','RUNNING','SUBMITTED','VERIFYING')",
            (mission_id,),
        ).fetchone()
        if running["n"]:
            return False
        return not (self._handlers is not None and self._handlers.has_pending_approval(mission_id))

    def _maybe_learn(self, mission_id: str, from_state: MissionState, reason: str) -> MissionState:
        """LEARN→IDLE only when completion is genuinely clear (§5.1)."""
        if from_state is MissionState.VALIDATE and self._completion_clear(mission_id):
            learned = self._safe_transition(mission_id, from_state, MissionState.LEARN, reason)
            if learned is MissionState.LEARN:
                return self._safe_transition(
                    mission_id, learned, MissionState.IDLE, "mission_complete"
                )
            return learned
        return from_state

    async def _apply_decision(
        self,
        mission: MissionSessionV1,
        decision: DecisionV1 | None,
        turn: ModelTurnResultV1,
    ) -> tuple[str, MissionState]:
        current = self._current_state(mission.mission_id)
        intent = decision.next_intent if decision else NextIntent.ANSWER

        if intent is NextIntent.PAUSE:
            note = decision.summary if decision else ""
            # §5.9：PAUSE 是主动暂停（SUSPENDED 优先；MONITOR/WAIT 外不可达时
            # 退到 WAIT_INPUT 等待用户），不是失败。
            paused = self._safe_transition(
                mission.mission_id, current, MissionState.SUSPENDED, "pause"
            )
            if paused is current:
                paused = self._safe_transition(
                    mission.mission_id, current, MissionState.WAIT_INPUT, "pause"
                )
            return (f"已暂停。{note}", paused)

        if intent is NextIntent.FAIL_SAFE:
            note = decision.summary if decision else ""
            from rosclaw.contracts.agent.agent_event import AgentEventType

            await self._emit(
                AgentEventType.ERROR,
                {"safety": "FAIL_SAFE", "note": note, "incident": True},
            )
            return (
                f"已执行 FAIL_SAFE：停止规划，取消未派发任务并请求安全停止。{note}",
                self._safe_transition(
                    mission.mission_id, current, MissionState.FAILED, "fail_safe"
                ),
            )

        if intent is NextIntent.WAIT:
            # §5.8：建立 WakeCondition（runner 在事件/截止时间自动唤醒）。
            payload = (
                decision.proposed_operation.payload
                if decision and decision.proposed_operation
                else {}
            ) or {}
            runner = getattr(self, "_wake_registrar", None)
            if runner is not None:
                runner(
                    mission.mission_id,
                    reference_id=payload.get("reference_id"),
                    notice=decision.summary if decision else "wait condition",
                )
            reply = _DECISION_BLOCK_RE.sub("", turn.content or "").strip() or (
                decision.summary if decision else "进入等待。"
            )
            return reply, current

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
            outcome = await self._handlers.hire_worker(decision)  # type: ignore[arg-type]
            next_state = self._safe_transition(
                mission.mission_id, staffed, MissionState.VALIDATE, "work_result_verified"
            )
            # §5.4：artifact 绑到 TaskNode 后回 PLAN 或 VALIDATE；
            # 只有完成条件全清才 LEARN→IDLE。
            next_state = self._maybe_learn(mission.mission_id, next_state, "delegation_verified")
            return outcome.text, next_state

        if intent is NextIntent.REQUEST_APPROVAL:
            if self._handlers is None:
                return (
                    "该步骤需要人类授权，但授权通道不可用；已停止继续推进（fail closed）。",
                    current,
                )
            outcome = await self._handlers.request_approval(decision)  # type: ignore[arg-type]
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
            return outcome.text, next_state

        if intent is NextIntent.REQUEST_ACTION:
            if self._handlers is None:
                return (
                    "需要物理动作，但当前没有动作通道；未提交任何动作请求（fail closed）。",
                    current,
                )
            from rosclaw.contracts.agent.agent_event import AgentEventType

            await self._emit(
                AgentEventType.ACTION_PROPOSED,
                {
                    "decision_id": decision.decision_id if decision else None,
                },
            )
            outcome = await self._handlers.request_action(decision)  # type: ignore[arg-type]
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
                    mission.mission_id, next_state, MissionState.MONITOR, "awaiting_terminal"
                )
            # §5.6：只有 verified terminal receipt 才能 MONITOR→VERIFY→LEARN；
            # 否则停留 MONITOR/DISPATCH（提交≠完成）。
            if next_state is MissionState.MONITOR and outcome.terminal_receipt:
                next_state = self._safe_transition(
                    mission.mission_id, next_state, MissionState.VERIFY, "receipt_terminal"
                )
            if next_state is MissionState.VERIFY:
                next_state = self._maybe_learn(mission.mission_id, next_state, "action_verified")
            elif next_state is MissionState.VALIDATE:
                next_state = self._maybe_learn(
                    mission.mission_id, next_state, "action_path_complete"
                )
            return outcome.text, next_state

        if intent is NextIntent.TEAM_COORDINATE:
            if self._handlers is None:
                return (
                    "Team Fabric 尚未启用；未进行团队协调（fail closed）。",
                    current,
                )
            outcome = await self._handlers.team_coordinate(decision)  # type: ignore[arg-type]
            next_state = current
            if current is MissionState.PLAN:
                next_state = self._safe_transition(
                    mission.mission_id, current, MissionState.VALIDATE, "plan_ready"
                )
                next_state = self._maybe_learn(
                    mission.mission_id, next_state, "team_coordination_complete"
                )
            return outcome.text, next_state

        if intent is NextIntent.VERIFY:
            # §5.7：VerifierRegistry 确定性验证；失败回 PLAN（可恢复）或 FAILED。
            if decision is None or decision.verification is None:
                return "VERIFY 缺少 verification 载荷，无法验证（fail closed）。", current
            payload = (
                decision.proposed_operation.payload if decision.proposed_operation else {}
            ) or {}
            from rosclaw.agentd.verifiers import VerifierRegistry

            registry = getattr(self, "_verifier_registry", None) or VerifierRegistry()
            context = dict(payload.get("context") or {})
            context.setdefault("evidence_refs", list(decision.evidence_refs))
            try:
                verdict = registry.run_many(list(decision.verification.verifiers), context)
            except Exception as exc:  # noqa: BLE001 - unknown verifier is fail-closed
                return f"验证器不可用（{exc}），不报告为成功。", current
            from rosclaw.contracts.agent.agent_event import AgentEventType

            await self._emit(
                AgentEventType.VERIFICATION_COMPLETED,
                {
                    "verifier_id": verdict.verifier_id,
                    "success": verdict.success,
                    "failure_reason": verdict.failure_reason,
                    "human_attested": verdict.human_attested,
                },
            )
            if verdict.success:
                # 合法路径：MONITOR→VERIFY 或直接 VALIDATE→LEARN；PLAN 先过 VALIDATE。
                next_state = current
                if current is MissionState.PLAN:
                    next_state = self._safe_transition(
                        mission.mission_id, current, MissionState.VALIDATE, "plan_ready"
                    )
                elif current is MissionState.MONITOR:
                    next_state = self._safe_transition(
                        mission.mission_id, current, MissionState.VERIFY, "verification_passed"
                    )
                next_state = self._maybe_learn(
                    mission.mission_id, next_state, "verification_passed"
                )
                reply = _DECISION_BLOCK_RE.sub("", turn.content or "").strip() or (
                    f"验证通过（{verdict.verifier_id}）。"
                )
                return reply, next_state
            # 失败：可恢复回 PLAN 重新规划。
            reply = (
                f"验证未通过（{verdict.verifier_id}: {verdict.failure_reason}）。"
                "回到规划阶段重新评估。"
            )
            return reply, self._safe_transition(
                mission.mission_id, current, MissionState.PLAN, "verification_failed_replan"
            )

        if intent is NextIntent.PLAN_PATCH:
            # §5.3：校验并真正提交 TaskGraphPatchV1（CAS + DAG + 事件）。
            if decision is None or decision.proposed_operation is None:
                return "PLAN_PATCH 缺少 proposed_operation（fail closed）。", current
            payload = decision.proposed_operation.payload
            if not payload:
                return "PLAN_PATCH payload 为空（fail closed）。", current
            from rosclaw.agentd.mission import RevisionConflictError
            from rosclaw.contracts.agent.task_graph import TaskGraphPatchV1
            from rosclaw.contracts.common import ValidationError

            try:
                patch = TaskGraphPatchV1.model_validate_contract(payload)
                new_revision = self._store.apply_patch(patch, actor_id=self._actor_id)
            except (ValidationError, RevisionConflictError) as exc:
                return f"TaskGraphPatch 被拒绝（{exc}），图未变更。", current
            from rosclaw.contracts.agent.agent_event import AgentEventType

            await self._emit(
                AgentEventType.TASK_GRAPH_COMMITTED,
                {"patch_id": patch.patch_id, "new_revision": new_revision},
            )
            reply = _DECISION_BLOCK_RE.sub("", turn.content or "").strip() or (
                f"任务图已提交（revision {new_revision}）。"
            )
            return reply, current

        # ANSWER（§5.1）：完成条件全清才 LEARN；否则只是解释，Mission 保持。
        raw_reply = turn.content.strip() if turn.content else ""
        if decision is not None:
            # The DecisionV1 block is machine output, not user-facing prose.
            raw_reply = _DECISION_BLOCK_RE.sub("", raw_reply).strip()
        reply = raw_reply or (decision.summary if decision else "")
        if current is MissionState.PLAN:
            next_state = self._safe_transition(
                mission.mission_id, current, MissionState.VALIDATE, "plan_ready"
            )
            next_state = self._maybe_learn(
                mission.mission_id, next_state, "answer_requires_no_dispatch"
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

    async def compact_conversation(
        self,
        mission: MissionSessionV1,
        *,
        reason: str,
        focus: str | None = None,
        dry_run: bool = False,
        keep_recent_tokens: int = 20_000,
    ) -> dict:
        """PR-07：持久化压缩（Pi 算法 + canonical journal 保留）。

        返回报告（dry_run 不写入）。物理事实永远不进 summary——
        下次编译仍从权威存储重建。
        """
        from rosclaw.agentd.context.compaction import (
            CompactionStore,
            build_compaction_view,
            deterministic_summary,
            estimate_messages_tokens,
            find_cut_point,
        )
        from rosclaw.agentd.mission.store import _utcnow
        from rosclaw.contracts.agent.agent_event import AgentEventType
        from rosclaw.contracts.agent.compaction import CompactionEntryV1

        self._restore_conversation(mission.mission_id)
        tokens_before = estimate_messages_tokens(self._conversation)
        cut = find_cut_point(self._conversation, keep_recent_tokens=keep_recent_tokens)
        if cut == 0 and reason == "manual" and len(self._conversation) > 4:
            # 手动 /compact 即使低于阈值也压缩（保留最近完整回合）。
            cut = max(1, len(self._conversation) - 2)
            while cut > 1 and self._conversation[cut].get("role") == "tool":
                cut -= 1
        report: dict = {
            "tokens_before": tokens_before,
            "cut_index": cut,
            "messages_total": len(self._conversation),
            "dry_run": dry_run,
        }
        if dry_run or cut <= 0:
            report["tokens_after"] = estimate_messages_tokens(self._conversation[cut:])
            return report

        span = self._conversation[:cut]
        kept = self._conversation[cut:]
        summary = deterministic_summary(span, goal=mission.goal.text, focus=focus)
        store = CompactionStore(self._store.connection)
        previous = store.latest(mission.mission_id)
        from rosclaw.contracts.common import content_hash

        span_hash = content_hash("cmp_span", span)
        covered_ids = [m["entry_id"] for m in span if m.get("entry_id")]
        protected = sorted({m["atomic_group"] for m in span if m.get("atomic_group")})
        entry = CompactionEntryV1(
            compaction_id=new_id("cmp"),
            mission_id=mission.mission_id,
            created_at=_utcnow(),
            reason=reason,  # type: ignore[arg-type]
            summary=summary,
            first_kept_event_id=f"msg_{cut}",
            tokens_before=tokens_before,
            tokens_after=estimate_messages_tokens(kept),
            evidence_refs=[],
            task_graph_revision=mission.task_graph_revision,
            context_revision=mission.context_revision,
            summary_model="deterministic-fallback",
            usage={},
            covered_entry_ids=covered_ids,
            covered_span_hash=span_hash,
            supersedes=previous.compaction_id if previous else None,
            prompt_version=getattr(self._prompt, "version", "") or "",
            provider=self._gateway.profile.provider,
            model=self._gateway.profile.model,
            protected_groups=protected,
        )
        store.save(entry)
        # view = summary（untrusted）+ kept；canonical journal 追加 summary 消息。
        view = build_compaction_view(entry, kept)
        self._store.append_conversation(mission.mission_id, [view[0]], actor_id=self._actor_id)
        # §8 风险点：view 缩短后 persisted 计数必须对齐新 view。
        self._conversation = view
        self._persisted_count = len(view)
        await self._emit(
            AgentEventType.COMPACTION_COMPLETED,
            {
                "compaction_id": entry.compaction_id,
                "reason": reason,
                "tokens_before": tokens_before,
                "tokens_after": entry.tokens_after,
            },
        )
        report.update(
            {
                "compaction_id": entry.compaction_id,
                "tokens_after": entry.tokens_after,
                "kept_messages": len(kept),
            }
        )
        return report

    async def _maybe_compact(self, mission: MissionSessionV1, result: LoopTurnResult) -> None:
        """§8.5：按 history_budget 主动持久化压缩。"""
        from rosclaw.agentd.context.compaction import (
            compute_history_budget,
            estimate_messages_tokens,
        )

        bundle = getattr(self, "_current_bundle", None)
        protected = 0
        if bundle is not None:
            protected = sum(
                layer.token_estimate
                for layer in (
                    bundle.layers.constitution,
                    bundle.layers.embodiment,
                    bundle.layers.dynamic_self,
                    bundle.layers.safety,
                )
            )
        window = getattr(self._compiler, "_max_input_tokens", 120_000)
        budget = compute_history_budget(
            context_window=window,
            protected_tokens=protected,
            tool_schema_tokens=2_000,
            max_output_tokens=16_384,
            safety_margin=4_096,
        )
        if estimate_messages_tokens(self._conversation) <= budget:
            return
        from rosclaw.contracts.agent.agent_event import AgentEventType

        await self._emit(AgentEventType.COMPACTION_STARTED, {"reason": "threshold"})
        await self.compact_conversation(mission, reason="threshold")
        result.degraded = "auto_compacted"

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
