"""P1-B2 产品旅程（0824 总纲 §12.3）：operation progress 流式进 TUI。

PTY `rosclaw chat` + 假模型：process_start 启动一个分阶段输出的长
命令（progress-step-1..3，间隔 1.2s）→ 断言 operation **运行期间**
TUI 已按 operation_id 原位渲染最新输出行（widget——用户不需要
/logs 或模型轮询）→ 终态 followUp 闭环照旧（WP-1 语义不回归）。

红证据：B2 前 TUI 只有 Working… 占位——运行中看不到任何输出行。
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

import pytest

from rosclaw.agentd.pi_entry import find_pi_agent_entry
from tests.agentd.test_h1_native_work import (
    _FakeServer,
    _Handler,
    _prepare_home,
)
from tests.agentd.test_product_journey import (
    PtySession,
    _chunk,
    _sse,
    _tool_call_frames,
)

pytestmark = pytest.mark.skipif(
    not find_pi_agent_entry(),
    reason="无 Node/dist（CI 全回归 job 未构建）——诚实 skip",
)


class _ProgressFake:
    """process_start（分阶段长命令）→ 首回合结束；终态 followUp → 汇报。"""

    def __init__(self) -> None:
        self.requests: list[dict] = []

    def answer(self, body: dict) -> bytes:
        self.requests.append(body)
        messages = body.get("messages", [])

        def _text(m: dict) -> str:
            content = m.get("content", "")
            if isinstance(content, list):
                return " ".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
            return str(content)

        if not body.get("stream"):
            return json.dumps(
                {
                    "id": "c",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "fake-k3",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "pong"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 5},
                }
            ).encode()
        # Official compaction has no tools and serializes the conversation in
        # one user message. Do not route every stream or a tool-enabled request
        # mentioning summaries here. Include the split-turn checkpoint format.
        summary_messages = [m for m in messages if m.get("role") not in ("system", "developer")]
        summary_text = _text(summary_messages[0]) if len(summary_messages) == 1 else ""
        summary_envelope = (
            summary_text.startswith("<conversation>\n") and "\n</conversation>\n\n" in summary_text
        ) or (
            summary_text.startswith("# Conversation\n") and "\n\n# Instructions\n" in summary_text
        )
        if (
            not body.get("tools")
            and len(summary_messages) == 1
            and summary_messages[0].get("role") == "user"
            and summary_envelope
        ):
            # Preserve evidence verbatim (apart from body context), including
            # staged output, identity and state. Do not substitute completion
            # prose for an operation record. Updates retain previous evidence
            # before newer evidence, and split turns retain only their prefix.
            header = (
                "Progress checkpoint: serialized evidence follows, not a live notification.\n"
                "Keep operation identity/state and staged output. Progress belongs to "
                "the live widget; running output is not terminal success. Preserve "
                "the terminal notification/report obligation; never infer successful "
                "output or restart existing work from a checkpoint. Unknown fields "
                "remain unknown.\n"
            )
            if summary_text.startswith("<conversation>\n"):
                conversation, rest = summary_text[len("<conversation>\n") :].split(
                    "\n</conversation>\n\n", 1
                )
                previous = ""
                if rest.startswith("<previous-summary>\n"):
                    previous = rest[len("<previous-summary>\n") :].split(
                        "\n</previous-summary>", 1
                    )[0]
            else:
                conversation = summary_text[len("# Conversation\n") :].split(
                    "\n\n# Instructions\n", 1
                )[0]
                previous = ""
            evidence = "\n".join(
                part
                for part in (
                    previous.removeprefix(header),
                    conversation,
                )
                if part
            )
            evidence = re.sub(
                r"<ROSCLAW_TRUSTED_CONTEXT>.*?</ROSCLAW_TRUSTED_CONTEXT>",
                "[body context omitted; not operation evidence]",
                evidence,
                flags=re.S,
            )
            frames = [
                _sse(_chunk(header + evidence)),
                _sse(_chunk("", "stop")),
                b"data: [DONE]\n\n",
            ]
            return b"".join(frames)

        def _context(m: dict) -> bool:
            text = _text(m).strip()
            return m.get("role") in ("user", "system", "developer") and (
                text.startswith("<ROSCLAW_TRUSTED_CONTEXT>")
                and text.endswith("</ROSCLAW_TRUSTED_CONTEXT>")
            )

        # Only recognize a context suffix. Earlier terminal history must not
        # override a new request or an ordinary tool result.
        tail = list(messages)
        while tail and (_context(tail[-1]) or tail[-1].get("role") in ("system", "developer")):
            tail.pop()
        if (
            len(tail) >= 2
            and tail[-1].get("role") == "assistant"
            and _text(tail[-1]) == "I have processed the tool results."
            and tail[-2].get("role") == "tool"
        ):
            tail.pop()
        last = tail[-1] if tail else {}
        has_tool_result = last.get("role") == "tool"
        last_text = _text(last).strip()
        if has_tool_result:
            frames = [
                _sse(_chunk("已在后台启动 operation，完成会通知。")),
                _sse(_chunk("", "stop")),
                b"data: [DONE]\n\n",
            ]
            return b"".join(frames)
        # Match the whole production formatter, including its instruction.
        # Custom metadata is absent after official custom-to-user conversion.
        production = (
            re.fullmatch(
                r"后台 Operation (op_[a-f0-9]+) 已终止：([A-Z][A-Z0-9_]*)"
                r"(?:（([^\r\n（）]+)）)?"
                r"。用 process_output 查看输出，然后在同一任务里继续（验证/修复/交付）。",
                last_text,
            )
            if last.get("role") == "user"
            else None
        )
        legacy = (
            re.match(
                r"^(?:operation\s+)?(op_[a-f0-9]+)\b[^\n]*已终止",
                last_text,
                re.I,
            )
            if last.get("role") == "user"
            else None
        )
        terminal = production or legacy
        if terminal:
            if production:
                operation_id, state, failure_code = production.groups()
                # Report only the recorded terminal state, not invented output.
                # Even LOST or an unfamiliar future state must not imply success.
                status = (
                    "后台 Operation 已完成并汇报。"
                    if state == "SUCCEEDED"
                    else "Operation 已终止；终态非成功或未知。"
                )
                failure = f"（{failure_code}）" if failure_code else ""
                reply = (
                    f"{status}Operation {operation_id} 终态：{state}{failure}。"
                    "尚未读取输出；用 process_output 查看输出，然后在同一任务里继续（验证/修复/交付）。"
                )
            else:
                # Preserve the old direct-fixture protocol, including stateless
                # legacy terminals, independently of the production protocol.
                failed = re.search(r"\b(?:FAILED|CANCELLED|CANCELED|TIMED_OUT|LOST)\b", last_text)
                reply = (
                    f"Operation {terminal.group(1)} 已终止；终态非成功，请检查输出。"
                    if failed
                    else "后台 Operation 已完成并汇报。"
                )
            frames = [_sse(_chunk(reply)), _sse(_chunk("", "stop")), b"data: [DONE]\n\n"]
            return b"".join(frames)
        frames = _tool_call_frames(
            "call_opstart",
            "process_start",
            json.dumps(
                {
                    "command": "for i in 1 2 3; do echo progress-step-$i; sleep 2.5; done",
                }
            ),
        )
        frames.append(b"data: [DONE]\n\n")
        return b"".join(frames)


class _ProgressFakeServer(_FakeServer):
    def __init__(self) -> None:
        self.fake = _ProgressFake()
        handler = type("H", (_Handler,), {"fake": self.fake})
        import threading as _threading
        from http.server import ThreadingHTTPServer

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.server.server_address[1]
        _threading.Thread(target=self.server.serve_forever, daemon=True).start()


class TestOperationProgressWidget:
    def test_progress_visible_during_run(self, tmp_path: Path) -> None:
        fake = _ProgressFakeServer()
        home, env = _prepare_home(tmp_path, fake.base_url)
        workspace = tmp_path / "ws"
        workspace.mkdir()
        session = PtySession(
            [sys.executable, "-m", "rosclaw.entrypoint", "chat"],
            env,
            log_path=tmp_path / "pty-b2.log",
            cwd=workspace,
        )
        try:
            session.expect(b"ROSClaw Native Agent", timeout=120)
            session.send("后台跑分阶段命令\r")
            session.expect("已在后台启动 operation".encode(), timeout=180)
            # operation 运行期间（~3.6s 总时长内）widget 原位渲染输出
            # 行——progress-step-1 或 2 必须先于完成通知可见。
            session.expect(b"progress-step-1", timeout=60)
            seen = bytes(session.output)
            match = re.search(rb"progress-step-(\d)", seen)
            assert match, "运行期间未见任何输出行（widget 未渲染）"
            assert int(match.group(1)) < 3, "只在完成后才看到输出——不是流式（看到了最后一步才出现）"
            # 终态 followUp 闭环不回归。
            session.expect("后台 Operation 已完成并汇报".encode(), timeout=180)
            db = sqlite3.connect(home / "agentd" / "missions.db")
            state = db.execute(
                "SELECT state FROM operations ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            db.close()
            assert state and state[0] == "SUCCEEDED", state
            session.expect_with_resend(b"rosclaw continue", "/quit\r", timeout=60)
            session.proc.wait(timeout=30)
        finally:
            session.stop()
            fake.close()


class TestProgressFakeSummarization:
    """Decode actual fake responses without launching a server or a process."""

    @staticmethod
    def _events(payload: bytes) -> list[dict]:
        frames = payload.decode().split("\n\n")
        assert frames[-2:] == ["data: [DONE]", ""]
        return [json.loads(frame.removeprefix("data: ")) for frame in frames[:-2]]

    @classmethod
    def _start_call(cls, payload: bytes) -> dict:
        choices = [event["choices"][0] for event in cls._events(payload)]
        assert choices[-1]["finish_reason"] == "tool_calls"
        parts = [part for choice in choices for part in choice["delta"].get("tool_calls", [])]
        assert parts[0]["id"] == "call_opstart"
        assert parts[0]["function"]["name"] == "process_start"
        return json.loads("".join(part["function"].get("arguments", "") for part in parts))

    @staticmethod
    def _summary_body(kind: str, blocks: bool, tools: bool) -> dict:
        # Match the official history/update and split-turn serialization;
        # a quoted terminal marker must not become a real followUp turn.
        conversation = "[user]: 后台跑分阶段命令\n[assistant]: op_abc123 已终止"
        if kind == "prefix":
            text = f"# Conversation\n{conversation}\n\n# Instructions\nCreate a concise checkpoint."
        else:
            text = f"<conversation>\n{conversation}\n</conversation>\n\n"
            if kind == "update":
                text += "<previous-summary>\nStaged work requested.\n</previous-summary>\n\n"
            text += "Summarize the conversation."
        body = {
            "stream": True,
            "messages": [
                {"role": "system", "content": "Create a structured checkpoint."},
                {"role": "user", "content": [{"type": "text", "text": text}] if blocks else text},
            ],
        }
        if tools:
            body["tools"] = []
        return body

    @pytest.mark.parametrize("kind", ["initial", "update", "prefix"])
    @pytest.mark.parametrize("blocks", [False, True])
    @pytest.mark.parametrize("empty_tools", [False, True])
    def test_no_tools_summary_is_text(self, kind: str, blocks: bool, empty_tools: bool) -> None:
        fake = _ProgressFake()
        body = self._summary_body(kind, blocks, empty_tools)
        choices = [event["choices"][0] for event in self._events(fake.answer(body))]
        assert choices[-1]["finish_reason"] == "stop"
        assert all(not choice["delta"].get("tool_calls") for choice in choices)
        text = "".join(choice["delta"].get("content", "") for choice in choices)
        assert text.strip() and text != "pong"
        assert "Progress checkpoint:" in text
        assert "后台 Operation 已完成并汇报" not in text
        assert fake.requests == [body]

    def test_summary_preserves_staged_start_and_terminal_reply(self) -> None:
        fake = _ProgressFake()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "process_start",
                    "parameters": {"type": "object"},
                },
            }
        ]
        start = {
            "stream": True,
            "tools": tools,
            "messages": [
                {"role": "user", "content": "后台跑分阶段命令；summary is ordinary user text"},
            ],
        }
        assert self._start_call(fake.answer(start)) == {
            "command": "for i in 1 2 3; do echo progress-step-$i; sleep 2.5; done",
        }
        fake.answer(self._summary_body("initial", False, False))
        started = {
            "stream": True,
            "tools": tools,
            "messages": [
                {"role": "tool", "tool_call_id": "call_opstart", "content": "op_abc123"},
            ],
        }
        choices = [event["choices"][0] for event in self._events(fake.answer(started))]
        assert choices[-1]["finish_reason"] == "stop"
        assert all(not choice["delta"].get("tool_calls") for choice in choices)
        assert "已在后台启动 operation" in "".join(
            choice["delta"].get("content", "") for choice in choices
        )
        fake.answer(self._summary_body("update", True, True))
        terminal = {
            "stream": True,
            "tools": tools,
            "messages": [
                {"role": "user", "content": "operation op_abc123 已终止"},
            ],
        }
        choices = [event["choices"][0] for event in self._events(fake.answer(terminal))]
        assert choices[-1]["finish_reason"] == "stop"
        assert all(not choice["delta"].get("tool_calls") for choice in choices)
        assert "后台 Operation 已完成并汇报" in "".join(
            choice["delta"].get("content", "") for choice in choices
        )
        assert fake.requests[0] is start
        assert started in fake.requests and terminal in fake.requests

    @pytest.mark.parametrize("variant", ["enabled_tools", "plain_no_tools", "assistant", "history"])
    def test_ambiguous_stream_keeps_original_tool_route(self, variant: str) -> None:
        fake = _ProgressFake()
        body = self._summary_body("initial", False, False)
        body["messages"][-1]["content"] = "<conversation>\nsummary\n</conversation>\n\nSummarize."
        if variant == "enabled_tools":
            body["tools"] = [{"type": "function", "function": {"name": "process_start"}}]
        elif variant == "plain_no_tools":
            body["messages"][-1]["content"] = "Please summarize progress before starting work."
        elif variant == "assistant":
            body["messages"][-1]["role"] = "assistant"
        else:
            body["messages"].insert(1, {"role": "user", "content": "Earlier ordinary turn"})
        assert "command" in self._start_call(fake.answer(body))

    @pytest.mark.parametrize("stream", [False, None])
    def test_nonstream_remains_json_pong(self, stream: bool | None) -> None:
        fake = _ProgressFake()
        body = self._summary_body("initial", True, True)
        body.pop("stream")
        if stream is not None:
            body["stream"] = stream
        response = json.loads(fake.answer(body))
        assert response["object"] == "chat.completion"
        choice = response["choices"][0]
        assert choice["message"] == {"role": "assistant", "content": "pong"}
        assert choice["finish_reason"] == "stop"
        assert fake.requests == [body]


class TestProgressFakeNotificationOrdering:
    """Exercise answer(), not source inspection or a live provider."""

    @staticmethod
    def _context(blocks: bool = False) -> dict:
        text = "<ROSCLAW_TRUSTED_CONTEXT>\nfresh body; no user request\n</ROSCLAW_TRUSTED_CONTEXT>"
        return {"role": "user", "content": [{"type": "text", "text": text}] if blocks else text}

    @staticmethod
    def _body(messages: list[dict]) -> dict:
        return {
            "stream": True,
            "tools": [{"type": "function", "function": {"name": "process_start"}}],
            "messages": messages,
        }

    @staticmethod
    def _text(payload: bytes) -> str:
        choices = [event["choices"][0] for event in TestProgressFakeSummarization._events(payload)]
        assert choices[-1]["finish_reason"] == "stop"
        assert all(not choice["delta"].get("tool_calls") for choice in choices)
        return "".join(choice["delta"].get("content", "") for choice in choices)

    @pytest.mark.parametrize("blocks", [False, True])
    def test_terminal_then_context_reports_without_restarting(self, blocks: bool) -> None:
        body = self._body(
            [
                {"role": "user", "content": "后台跑分阶段命令"},
                {"role": "tool", "tool_call_id": "call_opstart", "content": "op_fed123"},
                {"role": "assistant", "content": "已在后台启动 operation，完成会通知。"},
                {"role": "user", "content": "Operation op_fed123 已终止（SUCCEEDED）。"},
                self._context(blocks),
            ]
        )
        fake = _ProgressFake()
        assert "后台 Operation 已完成并汇报" in self._text(fake.answer(body))
        assert fake.requests == [body]

    @pytest.mark.parametrize("state", ["FAILED", "CANCELLED", "TIMED_OUT"])
    def test_non_success_terminal_is_not_successful_output(self, state: str) -> None:
        body = self._body(
            [
                {"role": "user", "content": f"Operation op_fed123 已终止（{state}）。"},
                self._context(),
            ]
        )
        text = self._text(_ProgressFake().answer(body))
        assert "op_fed123" in text and "非成功" in text
        assert "后台 Operation 已完成并汇报" not in text
        assert "progress-step-3" not in text

    @pytest.mark.parametrize("bridge", [False, True])
    def test_tool_result_followed_by_context_still_acknowledges_start(self, bridge: bool) -> None:
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_opstart",
                "content": "operation op_fed123 已终止 is quoted tool text",
            }
        ]
        if bridge:
            messages.append({"role": "assistant", "content": "I have processed the tool results."})
        messages.append(self._context())
        text = self._text(_ProgressFake().answer(self._body(messages)))
        assert "已在后台启动 operation" in text
        assert "后台 Operation 已完成并汇报" not in text

    @pytest.mark.parametrize("variant", ["new_request", "quoted", "assistant", "missing_id"])
    def test_old_terminal_or_quoted_text_does_not_end_new_turn(self, variant: str) -> None:
        last = {"role": "user", "content": "后台跑一个新分阶段命令"}
        if variant == "quoted":
            last["content"] = "Explain this quote: operation op_fed123 已终止"
        elif variant == "assistant":
            last = {"role": "assistant", "content": "operation op_fed123 已终止"}
        elif variant == "missing_id":
            last["content"] = "Operation 已终止 is ordinary text"
        body = self._body(
            [
                {"role": "user", "content": "operation op_aaa111 已终止"},
                {"role": "assistant", "content": "Prior report"},
                last,
                self._context(),
            ]
        )
        assert TestProgressFakeSummarization._start_call(_ProgressFake().answer(body)) == {
            "command": "for i in 1 2 3; do echo progress-step-$i; sleep 2.5; done",
        }

    @pytest.mark.parametrize("kind", ["initial", "update", "prefix"])
    def test_checkpoint_retains_staged_state_and_true_output(self, kind: str) -> None:
        fake = _ProgressFake()
        original = (
            "[user]: Run staged work; report when terminal.\n"
            '[assistant]: process_start {"command":"staged-command"}\n'
            '[toolResult]: {"operation_id":"op_fed123","state":"RUNNING"}\n'
            "[user]: operation op_fed123 progress output: progress-step-1"
        )
        body = TestProgressFakeSummarization._summary_body("initial", False, False)
        body["messages"][-1]["content"] = (
            f"<conversation>\n{original}\n</conversation>\n\nSummarize."
        )
        first = self._text(fake.answer(body))
        assert original in first
        assert "SUCCEEDED" not in first and "progress-step-3" not in first
        terminal = (
            "[user]: Operation op_fed123 已终止（FAILED）。\noutput: progress-step-2\nexit_code=7"
        )
        if kind == "prefix":
            prompt = f"# Conversation\n{original}\n{terminal}\n\n# Instructions\nSummarize only this prefix."
        elif kind == "update":
            prompt = f"<conversation>\n{terminal}\n</conversation>\n\n<previous-summary>\n{first}\n</previous-summary>\n\nUpdate."
        else:
            prompt = f"<conversation>\n{original}\n{terminal}\n</conversation>\n\nSummarize."
        body = {
            "stream": True,
            "tools": [],
            "messages": [
                {"role": "system", "content": "Summarize."},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ],
        }
        checkpoint = self._text(fake.answer(body))
        assert original in checkpoint and terminal in checkpoint
        assert checkpoint.index('"state":"RUNNING"') < checkpoint.index("FAILED")
        assert "terminal notification/report obligation" in checkpoint
        assert "Unknown fields remain unknown" in checkpoint
        assert "SUCCEEDED" not in checkpoint and "progress-step-3" not in checkpoint
        assert "op_abc123" not in checkpoint and "后台 Operation 已完成并汇报" not in checkpoint
        # A historical checkpoint is not a completion for a new user request.
        resumed = self._body(
            [
                {"role": "user", "content": checkpoint},
                {"role": "user", "content": "后台跑另一个分阶段命令"},
                self._context(),
            ]
        )
        assert "command" in TestProgressFakeSummarization._start_call(fake.answer(resumed))


class TestProgressFakeProductionProtocol:
    """Full notification/context fixtures; projection is static, HTTP unknown."""

    @staticmethod
    def _window() -> tuple[dict, dict]:
        # Complete selected raw entries from the supplied P1b2 session window.
        notification = {
            "type": "custom_message",
            "customType": "rosclaw.operation.result",
            "content": "后台 Operation op_8e46def7e2f2407ea26e7264 已终止：SUCCEEDED。用 process_output 查看输出，然后在同一任务里继续（验证/修复/交付）。",
            "display": False,
            "details": {"operation_id": "op_8e46def7e2f2407ea26e7264", "state": "SUCCEEDED"},
            "id": "0cf7fbb2",
            "parentId": "a31265b4",
            "timestamp": "2026-10-16T01:55:39.690Z",
        }
        context = {
            "type": "custom_message",
            "customType": "rosclaw.embodied_context",
            "content": (
                "<ROSCLAW_TRUSTED_CONTEXT>\n"
                "mission: mis_9ef1d4d852e64874a5cf35f0  mode: SIMULATION  revision: 0\n"
                "body: sim/ur5e (hash body_6fd7f48bf79)\n"
                "body_summary: SIMULATED body sim/ur5e (evidence_class=simulated; not usable as REAL physical proof)\n"
                "workspace_window: radius [0.1, 0.8]m, z [0.02, 1.2]m\n"
                "pending_approvals: 0\n"
                "task_run: /tmp/pih3s2pytest16v3/test_progress_visible_during_r0/rh/runs/task_00b40bb317844035b03f01c9/r1（交付物→outputs/ 证据→evidence/ 草稿→scratch/ 日志→logs/；scratch 不得登记为交付物）\n"
                "generated_at: 2026-10-16T01:55:39.688532+00:00  expires_at: 2026-10-16T01:56:09.688532+00:00\n"
                "规则：以上为本轮唯一权威具身事实；历史消息中的旧状态以此为准。\n"
                "</ROSCLAW_TRUSTED_CONTEXT>"
            ),
            "display": False,
            "details": {"stale": False, "note": "fresh"},
            "id": "6b635b21",
            "parentId": "0cf7fbb2",
            "timestamp": "2026-10-16T01:55:39.690Z",
        }
        return notification, context

    @staticmethod
    def _notification(operation_id: str, state: str, failure_code: str | None) -> str:
        failure = f"（{failure_code}）" if failure_code else ""
        return (
            f"后台 Operation {operation_id} 已终止：{state}{failure}"
            "。用 process_output 查看输出，然后在同一任务里继续（验证/修复/交付）。"
        )

    @staticmethod
    def _project(notification: dict, context: dict, blocks: bool, bridge: bool) -> list[dict]:
        # Source-derived projection, NOT actual HTTP capture: custom->user,
        # then provider text/string conversion. The conditional bridge variant
        # replaces the recorded assistant ack, rather than claiming it occurred.
        # The small tool antecedent is synthetic, not a full captured tool row.
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_opstart",
                "content": "Operation 已启动：op_8e46def7e2f2407ea26e7264（后台运行）。",
            }
        ]
        messages.append(
            {
                "role": "assistant",
                "content": "I have processed the tool results."
                if bridge
                else "已在后台启动 operation，完成会通知。",
            }
        )
        for entry in (notification, context):
            content = entry["content"]
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": content}] if blocks else content,
                }
            )
        return messages

    @pytest.mark.parametrize("blocks", [False, True])
    @pytest.mark.parametrize("bridge", [False, True])
    def test_complete_observed_notification_after_official_projection(
        self, blocks: bool, bridge: bool
    ) -> None:
        notification, context = self._window()
        assert notification["parentId"] == "a31265b4"
        assert context["parentId"] == notification["id"]
        assert context["timestamp"] == notification["timestamp"]
        assert notification["content"] == self._notification(
            notification["details"]["operation_id"], "SUCCEEDED", None
        )
        messages = self._project(notification, context, blocks, bridge)
        assert [m["role"] for m in messages] == ["tool", "assistant", "user", "user"]
        assert all("customType" not in m and "details" not in m for m in messages)
        body = TestProgressFakeNotificationOrdering._body(messages)
        fake = _ProgressFake()
        reply = TestProgressFakeNotificationOrdering._text(fake.answer(body))
        assert "后台 Operation 已完成并汇报" in reply
        assert "op_8e46def7e2f2407ea26e7264" in reply and "SUCCEEDED" in reply
        assert "尚未读取输出" in reply and "process_output" in reply
        assert "progress-step-3" not in reply
        assert fake.requests == [body]

    @pytest.mark.parametrize("state", ["SUCCEEDED", "FAILED", "CANCELLED", "LOST", "FUTURE_STATE"])
    @pytest.mark.parametrize("failure_code", [None, "PROCESS_EXIT_NONZERO"])
    def test_full_formatter_reports_only_recorded_state(
        self, state: str, failure_code: str | None
    ) -> None:
        notification, context = self._window()
        notification["content"] = self._notification("op_fed123", state, failure_code)
        notification["details"] = {"operation_id": "op_fed123", "state": state}
        messages = [{"role": "user", "content": self._notification("op_aaa111", "SUCCEEDED", None)}]
        messages += self._project(notification, context, True, True)
        reply = TestProgressFakeNotificationOrdering._text(
            _ProgressFake().answer(TestProgressFakeNotificationOrdering._body(messages)),
        )
        assert "op_fed123" in reply and state in reply and "op_aaa111" not in reply
        assert "尚未读取输出" in reply and "process_output" in reply
        assert "progress-step-" not in reply
        if failure_code:
            assert f"（{failure_code}）" in reply
        if state == "SUCCEEDED":
            assert "后台 Operation 已完成并汇报" in reply
        else:
            assert "非成功" in reply
            assert "SUCCEEDED" not in reply and "后台 Operation 已完成并汇报" not in reply

    @pytest.mark.parametrize("kind", ["initial", "update", "prefix"])
    def test_full_production_checkpoint_keeps_state_and_pending_output(self, kind: str) -> None:
        notification, context = self._window()
        operation_id = notification["details"]["operation_id"]
        original = (
            f'[Tool result]: {{"operation_id":"{operation_id}","state":"RUNNING"}}\n'
            "[User]: progress output: progress-step-1"
        )
        terminal = self._notification(operation_id, "LOST", "WORKER_LOST")
        newer = f"[User]: {terminal}\n[User]: {context['content']}"
        fake = _ProgressFake()
        initial = {
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": f"<conversation>\n{original}\n</conversation>\n\nSummarize.",
                }
            ],
        }
        first = TestProgressFakeNotificationOrdering._text(fake.answer(initial))
        if kind == "update":
            prompt = f"<conversation>\n{newer}\n</conversation>\n\n<previous-summary>\n{first}\n</previous-summary>\n\nUpdate."
        elif kind == "prefix":
            prompt = (
                f"# Conversation\n{original}\n{newer}\n\n# Instructions\nSummarize this prefix."
            )
        else:
            prompt = f"<conversation>\n{original}\n{newer}\n</conversation>\n\nSummarize."
        body = {
            "stream": True,
            "tools": [],
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        }
        checkpoint = TestProgressFakeNotificationOrdering._text(fake.answer(body))
        assert original in checkpoint and terminal in checkpoint
        assert checkpoint.index(original) < checkpoint.index(terminal)
        assert "LOST" in checkpoint and "WORKER_LOST" in checkpoint
        assert (
            "process_output" in checkpoint
            and "terminal notification/report obligation" in checkpoint
        )
        assert "SUCCEEDED" not in checkpoint and "progress-step-3" not in checkpoint
        assert "后台 Operation 已完成并汇报" not in checkpoint
        assert "[body context omitted; not operation evidence]" in checkpoint
        resumed = TestProgressFakeNotificationOrdering._body(
            [
                {
                    "role": "user",
                    "content": "The conversation history before this point was compacted into the following summary:\n\n<summary>\n"
                    + checkpoint
                    + "\n</summary>",
                },
                {"role": "user", "content": "后台跑另一个分阶段命令"},
                {"role": "user", "content": context["content"]},
            ]
        )
        assert "command" in TestProgressFakeSummarization._start_call(fake.answer(resumed))

    @pytest.mark.parametrize("variant", ["quoted", "assistant", "later_user", "missing_id"])
    def test_production_text_does_not_override_role_or_latest_turn(self, variant: str) -> None:
        notification, context = self._window()
        text = notification["content"]
        last = {"role": "user", "content": "Explain this quoted notification: " + text}
        if variant == "assistant":
            last = {"role": "assistant", "content": text}
        elif variant == "later_user":
            last = {"role": "user", "content": "后台跑一个新分阶段命令"}
        elif variant == "missing_id":
            last = {"role": "user", "content": text.replace("op_8e46def7e2f2407ea26e7264", "")}
        body = TestProgressFakeNotificationOrdering._body(
            [
                {"role": "user", "content": text},
                {"role": "assistant", "content": "Prior report"},
                last,
                {"role": "user", "content": context["content"]},
            ]
        )
        assert "command" in TestProgressFakeSummarization._start_call(_ProgressFake().answer(body))

    @pytest.mark.parametrize("bridge", [False, True])
    def test_full_notification_quoted_in_tool_still_acknowledges_start(self, bridge: bool) -> None:
        notification, context = self._window()
        messages = [
            {"role": "tool", "tool_call_id": "call_opstart", "content": notification["content"]}
        ]
        if bridge:
            messages.append({"role": "assistant", "content": "I have processed the tool results."})
        messages.append({"role": "user", "content": [{"type": "text", "text": context["content"]}]})
        reply = TestProgressFakeNotificationOrdering._text(
            _ProgressFake().answer(TestProgressFakeNotificationOrdering._body(messages)),
        )
        assert "已在后台启动 operation" in reply
        assert "后台 Operation 已完成并汇报" not in reply
