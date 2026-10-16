"""PR-H3 产品 Gate（总纲 v2 §21 Gate E 子集）：长 Operation 闭环。

PTY `rosclaw chat` + 假模型编排 + 真实 OperationManager：
1. 模型 process_start → 立即返回 operation_id（回合不死等）；
2. 后台进程真实跑完 → 终态一次性 followUp 注入同一 session；
3. 模型收到通知回合 → 用 process_output 读输出 → 汇报；
4. 账本：operations SUCCEEDED + task_events 完整链（started/output/
   completed）；零 WorkOrder；progress 不经 LLM（通知只一次）。
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


class _OperationFake:
    """编排：process_start → 首回合结束；终态 followUp → 读输出汇报。"""

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

        def _context(m: dict) -> bool:
            text = _text(m).strip()
            return m.get("role") in ("user", "system", "developer") and (
                text.startswith("<ROSCLAW_TRUSTED_CONTEXT>")
                and text.endswith("</ROSCLAW_TRUSTED_CONTEXT>")
            )

        # Custom messages become separate user messages on the inspected SDK
        # path. Only skip a recognized context suffix, never search all history
        # for a terminal keyword (which may belong to an earlier user turn).
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
        # Official compaction uses a standalone serialized user message and no
        # tools, not the ordinary tool-enabled agent turn. Streaming alone (or
        # the word "summary" in trusted context) is not a discriminator. Match
        # both history/update and split-turn envelopes from compaction.js.
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
            # Keep serialized evidence, including IDs, states, tool arguments
            # and real output, rather than replacing it with a generic claim.
            # Previous evidence precedes the new conversation on updates. This
            # is deliberately lossless except for the separate body-context
            # wrapper: a fake must not invent state through abstractive prose.
            header = (
                "Operation checkpoint: serialized evidence follows, not a live notification.\n"
                "Keep operation identity/state and pending output obligations. A terminal "
                "state alone does not prove output was read. Use process_output for the "
                "recorded ID before reporting; never infer successful output or restart "
                "existing work from a checkpoint. Unknown fields remain unknown.\n"
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
        if has_tool_result:
            call_id = str(last.get("tool_call_id", ""))
            if call_id == "call_opstart":
                # 首回合结束——不死等（operation 后台跑）。
                frames = [
                    _sse(_chunk("已在后台启动 operation，完成会通知。")),
                    _sse(_chunk("", "stop")),
                    b"data: [DONE]\n\n",
                ]
                return b"".join(frames)
            if call_id == "call_opoutput":
                frames = [
                    _sse(_chunk("后台 Operation 输出已读到并完成汇报。")),
                    _sse(_chunk("", "stop")),
                    b"data: [DONE]\n\n",
                ]
                return b"".join(frames)
        # operation-watcher.ts emits this complete literal, not the older
        # shortened test shape. Custom entries reach the provider as users;
        # details/customType are no longer present there (convertToLlm).
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
            # Terminal notification can precede a fresh context message. Read
            # output for its actual ID, including failed/cancelled operations.
            frames = _tool_call_frames(
                "call_opoutput",
                "process_output",
                json.dumps({"operation_id": terminal.group(1)}),
            )
            frames.append(b"data: [DONE]\n\n")
            return b"".join(frames)
        # 用户回合 → process_start。
        frames = _tool_call_frames(
            "call_opstart",
            "process_start",
            json.dumps({"command": "echo op-output-line && sleep 0.5"}),
        )
        frames.append(b"data: [DONE]\n\n")
        return b"".join(frames)


class _OpFakeServer(_FakeServer):
    def __init__(self) -> None:
        self.fake = _OperationFake()
        handler = type("H", (_Handler,), {"fake": self.fake})
        import threading as _threading
        from http.server import ThreadingHTTPServer

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.server.server_address[1]
        _threading.Thread(target=self.server.serve_forever, daemon=True).start()


class TestOperationClosedLoop:
    def test_process_operation_full_loop(self, tmp_path: Path) -> None:
        fake = _OpFakeServer()
        home, env = _prepare_home(tmp_path, fake.base_url)
        workspace = tmp_path / "ws"
        workspace.mkdir()
        session = PtySession(
            [sys.executable, "-m", "rosclaw.entrypoint", "chat"],
            env,
            log_path=tmp_path / "pty-h3.log",
            cwd=workspace,
        )
        try:
            session.expect(b"ROSClaw Native Agent", timeout=120)
            session.send("后台跑一个长命令并告诉我结果\r")
            session.expect("已在后台启动 operation".encode(), timeout=180)
            # 终态 followUp → 读输出 → 汇报（同一 session，用户零追问）。
            session.expect("后台 Operation 输出已读到并完成汇报".encode(), timeout=180)
            db = sqlite3.connect(home / "agentd" / "missions.db")
            ops = db.execute("SELECT operation_id, state, ended_at FROM operations").fetchall()
            events = db.execute("SELECT event_type FROM task_events ORDER BY seq").fetchall()
            orders = db.execute("SELECT COUNT(*) FROM work_orders").fetchone()[0]
            db.close()
            assert len(ops) == 1, f"一个 operation: {ops}"
            assert ops[0][1] == "SUCCEEDED", ops
            assert ops[0][2], "终态必须冻结 ended_at"
            kinds = [e[0] for e in events]
            assert "operation.admitted" in kinds
            assert "operation.output" in kinds
            assert "operation.completed" in kinds
            assert orders == 0
            session.expect_with_resend(b"rosclaw continue", "/quit\r", timeout=60)
            session.proc.wait(timeout=30)
        finally:
            session.stop()
            fake.close()


class TestOperationFakeSummarization:
    """Direct fake calls only: inspect the real SSE, never execute a command."""

    @staticmethod
    def _events(payload: bytes) -> list[dict]:
        frames = payload.decode().split("\n\n")
        assert frames[-2:] == ["data: [DONE]", ""]
        return [json.loads(frame.removeprefix("data: ")) for frame in frames[:-2]]

    @classmethod
    def _call(cls, payload: bytes, name: str, call_id: str) -> dict:
        choices = [event["choices"][0] for event in cls._events(payload)]
        assert choices[-1]["finish_reason"] == "tool_calls"
        parts = [part for choice in choices for part in choice["delta"].get("tool_calls", [])]
        assert parts[0]["id"] == call_id
        assert parts[0]["function"]["name"] == name
        return json.loads("".join(part["function"].get("arguments", "") for part in parts))

    @staticmethod
    def _summary_body(kind: str, blocks: bool, tools: bool) -> dict:
        # These are the serialized request shapes in official compaction.js,
        # not a keyword-only synthetic summary flag. Include a terminal marker
        # inside the serialized conversation to catch wrong followUp routing.
        conversation = "[user]: 后台跑命令\n[assistant]: op_abc123 已终止"
        if kind == "prefix":
            text = f"# Conversation\n{conversation}\n\n# Instructions\nCreate a concise checkpoint."
        else:
            text = f"<conversation>\n{conversation}\n</conversation>\n\n"
            if kind == "update":
                text += "<previous-summary>\nBackground work requested.\n</previous-summary>\n\n"
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
        fake = _OperationFake()
        body = self._summary_body(kind, blocks, empty_tools)
        events = self._events(fake.answer(body))
        choices = [event["choices"][0] for event in events]
        assert choices[-1]["finish_reason"] == "stop"
        assert all(not choice["delta"].get("tool_calls") for choice in choices)
        text = "".join(choice["delta"].get("content", "") for choice in choices)
        assert text.strip() and text != "pong"
        assert "Operation checkpoint:" in text
        assert "后台 Operation 输出已读到并完成汇报" not in text
        assert fake.requests == [body]

    def test_summary_does_not_hide_operation_tool_roundtrip(self) -> None:
        fake = _OperationFake()
        tools = [
            {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
            for name in ("process_start", "process_output")
        ]
        start = {
            "stream": True,
            "tools": tools,
            "messages": [
                {"role": "user", "content": "后台跑命令；summary is ordinary user text"},
            ],
        }
        args = self._call(fake.answer(start), "process_start", "call_opstart")
        assert args == {"command": "echo op-output-line && sleep 0.5"}
        fake.answer(self._summary_body("initial", False, False))
        started = {
            "stream": True,
            "tools": tools,
            "messages": [
                {"role": "tool", "tool_call_id": "call_opstart", "content": "op_abc123"},
            ],
        }
        started_choices = [event["choices"][0] for event in self._events(fake.answer(started))]
        assert started_choices[-1]["finish_reason"] == "stop"
        assert all(not choice["delta"].get("tool_calls") for choice in started_choices)
        assert "已在后台启动 operation" in "".join(
            choice["delta"].get("content", "") for choice in started_choices
        )
        terminal = {
            "stream": True,
            "tools": tools,
            "messages": [
                {"role": "user", "content": "operation op_abc123 已终止"},
            ],
        }
        assert self._call(fake.answer(terminal), "process_output", "call_opoutput") == {
            "operation_id": "op_abc123",
        }
        fake.answer(self._summary_body("update", True, True))
        output = {
            "stream": True,
            "tools": tools,
            "messages": [
                {"role": "tool", "tool_call_id": "call_opoutput", "content": "op-output-line"},
            ],
        }
        # Decode escaped JSON content instead of searching encoded SSE bytes.
        choices = [event["choices"][0] for event in self._events(fake.answer(output))]
        assert choices[-1]["finish_reason"] == "stop"
        assert all(not choice["delta"].get("tool_calls") for choice in choices)
        assert "后台 Operation 输出已读到并完成汇报" in "".join(
            choice["delta"].get("content", "") for choice in choices
        )
        assert fake.requests[0] is start
        assert terminal in fake.requests and output in fake.requests

    @pytest.mark.parametrize("variant", ["enabled_tools", "plain_no_tools", "assistant", "history"])
    def test_ambiguous_stream_keeps_original_tool_route(self, variant: str) -> None:
        fake = _OperationFake()
        body = self._summary_body("initial", False, False)
        # No terminal marker in these negative cases: original route is start.
        body["messages"][-1]["content"] = "<conversation>\nsummary\n</conversation>\n\nSummarize."
        if variant == "enabled_tools":
            body["tools"] = [{"type": "function", "function": {"name": "process_start"}}]
        elif variant == "plain_no_tools":
            body["messages"][-1]["content"] = "Please summarize progress before starting work."
        elif variant == "assistant":
            body["messages"][-1]["role"] = "assistant"
        else:
            body["messages"].insert(1, {"role": "user", "content": "Earlier ordinary turn"})
        assert "command" in self._call(fake.answer(body), "process_start", "call_opstart")

    @pytest.mark.parametrize("stream", [False, None])
    def test_nonstream_remains_json_pong(self, stream: bool | None) -> None:
        fake = _OperationFake()
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


class TestOperationFakeNotificationOrdering:
    """Request-shape regressions, not a claim that the missing HTTP tail is known."""

    @staticmethod
    def _context(blocks: bool = False) -> dict:
        text = "<ROSCLAW_TRUSTED_CONTEXT>\nfresh body; summary is not a request\n</ROSCLAW_TRUSTED_CONTEXT>"
        return {"role": "user", "content": [{"type": "text", "text": text}] if blocks else text}

    @staticmethod
    def _body(messages: list[dict]) -> dict:
        return {
            "stream": True,
            "tools": [
                {"type": "function", "function": {"name": name}}
                for name in ("process_start", "process_output")
            ],
            "messages": messages,
        }

    @staticmethod
    def _text(payload: bytes) -> str:
        choices = [event["choices"][0] for event in TestOperationFakeSummarization._events(payload)]
        assert choices[-1]["finish_reason"] == "stop"
        assert all(not choice["delta"].get("tool_calls") for choice in choices)
        return "".join(choice["delta"].get("content", "") for choice in choices)

    @pytest.mark.parametrize("blocks", [False, True])
    @pytest.mark.parametrize("state", ["SUCCEEDED", "FAILED", "CANCELLED"])
    def test_terminal_then_context_reads_actual_id(self, blocks: bool, state: str) -> None:
        fake = _OperationFake()
        terminal = f"Operation op_fed123 已终止（{state}）。"
        body = self._body(
            [
                {"role": "user", "content": "operation op_aaa111 已终止（SUCCEEDED）。"},
                {"role": "assistant", "content": "Earlier report"},
                {"role": "user", "content": terminal},
                self._context(blocks),
            ]
        )
        assert TestOperationFakeSummarization._call(
            fake.answer(body),
            "process_output",
            "call_opoutput",
        ) == {"operation_id": "op_fed123"}
        assert fake.requests == [body]

    @pytest.mark.parametrize("call_id", ["call_opstart", "call_opoutput"])
    @pytest.mark.parametrize("bridge", [False, True])
    def test_tool_then_context_is_not_a_notification(self, call_id: str, bridge: bool) -> None:
        messages = [
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": "operation op_fed123 已终止; actual tool output",
            }
        ]
        if bridge:
            messages.append({"role": "assistant", "content": "I have processed the tool results."})
        messages.append(self._context())
        text = self._text(_OperationFake().answer(self._body(messages)))
        expected = (
            "已在后台启动 operation"
            if call_id == "call_opstart"
            else "后台 Operation 输出已读到并完成汇报"
        )
        assert expected in text

    @pytest.mark.parametrize("variant", ["new_request", "quoted", "assistant", "missing_id"])
    def test_old_or_quoted_terminal_does_not_complete_new_turn(self, variant: str) -> None:
        last = {"role": "user", "content": "后台跑一个新命令"}
        if variant == "quoted":
            last["content"] = "Explain this history: operation op_fed123 已终止"
        elif variant == "assistant":
            last = {"role": "assistant", "content": "operation op_fed123 已终止"}
        elif variant == "missing_id":
            last["content"] = "Operation 已终止 is a phrase, not a notification"
        body = self._body(
            [
                {"role": "user", "content": "operation op_aaa111 已终止"},
                {"role": "assistant", "content": "Prior report"},
                last,
                self._context(),
            ]
        )
        assert TestOperationFakeSummarization._call(
            _OperationFake().answer(body),
            "process_start",
            "call_opstart",
        ) == {"command": "echo op-output-line && sleep 0.5"}

    @pytest.mark.parametrize("kind", ["initial", "update", "prefix"])
    def test_checkpoint_retains_identity_state_output_and_update_order(self, kind: str) -> None:
        fake = _OperationFake()
        original = (
            "[user]: Run background work and report actual output.\n"
            '[assistant]: process_start {"command":"echo actual-7"}\n'
            '[toolResult]: {"operation_id":"op_fed123","state":"RUNNING"}'
        )
        body = TestOperationFakeSummarization._summary_body("initial", False, False)
        body["messages"][-1]["content"] = (
            f"<conversation>\n{original}\n</conversation>\n\nSummarize."
        )
        first = self._text(fake.answer(body))
        assert original in first
        assert "SUCCEEDED" not in first and "op-output-line" not in first
        terminal = (
            "[user]: Operation op_fed123 已终止（FAILED）。\n[toolResult]: actual-7\nexit_code=7"
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
        assert "process_output" in checkpoint and "Unknown fields remain unknown" in checkpoint
        assert "SUCCEEDED" not in checkpoint and "op_abc123" not in checkpoint
        assert "后台 Operation 输出已读到并完成汇报" not in checkpoint
        # A new user request after a quoted checkpoint still starts new work.
        resumed = self._body(
            [
                {"role": "user", "content": checkpoint},
                {"role": "user", "content": "后台跑另一个命令"},
                self._context(),
            ]
        )
        assert "command" in TestOperationFakeSummarization._call(
            fake.answer(resumed),
            "process_start",
            "call_opstart",
        )


class TestOperationFakeProductionProtocol:
    """Static source projections, not captured HTTP or an SDK execution test."""

    @staticmethod
    def _window() -> tuple[dict, dict]:
        # Complete selected raw entries from the supplied H3 window. Keep the
        # real parent chain, timestamps and FULL strings; never presence flags.
        notification = {
            "type": "custom_message",
            "customType": "rosclaw.operation.result",
            "content": "后台 Operation op_2db9e0c53ec148c69e060539 已终止：SUCCEEDED。用 process_output 查看输出，然后在同一任务里继续（验证/修复/交付）。",
            "display": False,
            "details": {"operation_id": "op_2db9e0c53ec148c69e060539", "state": "SUCCEEDED"},
            "id": "45ece9cc",
            "parentId": "06e3c289",
            "timestamp": "2026-10-16T01:52:30.179Z",
        }
        context = {
            "type": "custom_message",
            "customType": "rosclaw.embodied_context",
            "content": (
                "<ROSCLAW_TRUSTED_CONTEXT>\n"
                "mission: mis_36d62c95f76f47b388844085  mode: SIMULATION  revision: 0\n"
                "body: sim/ur5e (hash body_6fd7f48bf79)\n"
                "body_summary: SIMULATED body sim/ur5e (evidence_class=simulated; not usable as REAL physical proof)\n"
                "workspace_window: radius [0.1, 0.8]m, z [0.02, 1.2]m\n"
                "pending_approvals: 0\n"
                "task_run: /tmp/pih3s2pytest16v3/test_process_operation_full_lo0/rh/runs/task_449284e9a62742aca5a32ea4/r1（交付物→outputs/ 证据→evidence/ 草稿→scratch/ 日志→logs/；scratch 不得登记为交付物）\n"
                "generated_at: 2026-10-16T01:52:30.176752+00:00  expires_at: 2026-10-16T01:53:00.176752+00:00\n"
                "规则：以上为本轮唯一权威具身事实；历史消息中的旧状态以此为准。\n"
                "</ROSCLAW_TRUSTED_CONTEXT>"
            ),
            "display": False,
            "details": {"stale": False, "note": "fresh"},
            "id": "ec5fbea5",
            "parentId": "45ece9cc",
            "timestamp": "2026-10-16T01:52:30.179Z",
        }
        return notification, context

    @staticmethod
    def _notification(operation_id: str, state: str, failure_code: str | None) -> str:
        # Verbatim formatter 446..449, including optional failure punctuation.
        failure = f"（{failure_code}）" if failure_code else ""
        return (
            f"后台 Operation {operation_id} 已终止：{state}{failure}"
            "。用 process_output 查看输出，然后在同一任务里继续（验证/修复/交付）。"
        )

    @staticmethod
    def _project(notification: dict, context: dict, blocks: bool, bridge: bool) -> list[dict]:
        # convertToLlm maps custom -> user and keeps content in order, without
        # details/type. Provider conversion retains the text blocks. The bridge
        # variant models ONLY the documented conditional tool->user branch;
        # the actual captured row9 was an assistant acknowledgement instead.
        # This small tool antecedent is synthetic, not a full captured tool row.
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_opstart",
                "content": "Operation 已启动：op_2db9e0c53ec148c69e060539（后台运行）。",
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
        assert notification["parentId"] == "06e3c289"
        assert context["parentId"] == notification["id"]
        assert context["timestamp"] == notification["timestamp"]
        assert notification["content"] == self._notification(
            notification["details"]["operation_id"], "SUCCEEDED", None
        )
        messages = self._project(notification, context, blocks, bridge)
        assert [m["role"] for m in messages] == ["tool", "assistant", "user", "user"]
        assert all("customType" not in m and "details" not in m for m in messages)
        body = TestOperationFakeNotificationOrdering._body(messages)
        fake = _OperationFake()
        assert TestOperationFakeSummarization._call(
            fake.answer(body), "process_output", "call_opoutput"
        ) == {
            "operation_id": "op_2db9e0c53ec148c69e060539",
        }
        assert fake.requests == [body]

    @pytest.mark.parametrize("state", ["SUCCEEDED", "FAILED", "CANCELLED", "LOST"])
    @pytest.mark.parametrize("failure_code", [None, "PROCESS_EXIT_NONZERO"])
    def test_full_formatter_state_and_optional_failure_read_exact_id(
        self, state: str, failure_code: str | None
    ) -> None:
        notification, context = self._window()
        notification["content"] = self._notification("op_fed123", state, failure_code)
        notification["details"] = {"operation_id": "op_fed123", "state": state}
        messages = [{"role": "user", "content": self._notification("op_aaa111", "SUCCEEDED", None)}]
        messages += self._project(notification, context, True, True)
        body = TestOperationFakeNotificationOrdering._body(messages)
        assert TestOperationFakeSummarization._call(
            _OperationFake().answer(body), "process_output", "call_opoutput"
        ) == {
            "operation_id": "op_fed123",
        }

    @pytest.mark.parametrize("kind", ["initial", "update", "prefix"])
    def test_full_production_checkpoint_preserves_real_identity_and_unread_output(
        self, kind: str
    ) -> None:
        notification, context = self._window()
        operation_id = notification["details"]["operation_id"]
        original = f'[Tool result]: {{"operation_id":"{operation_id}","state":"RUNNING"}}'
        terminal = self._notification(operation_id, "FAILED", "PROCESS_EXIT_NONZERO")
        newer = f"[User]: {terminal}\n[User]: {context['content']}"
        fake = _OperationFake()
        initial = {
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": f"<conversation>\n{original}\n</conversation>\n\nSummarize.",
                }
            ],
        }
        first = TestOperationFakeNotificationOrdering._text(fake.answer(initial))
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
        checkpoint = TestOperationFakeNotificationOrdering._text(fake.answer(body))
        assert original in checkpoint and terminal in checkpoint
        assert checkpoint.index(original) < checkpoint.index(terminal)
        assert "PROCESS_EXIT_NONZERO" in checkpoint and "SUCCEEDED" not in checkpoint
        assert "process_output" in checkpoint and "before reporting" in checkpoint
        assert (
            "op-output-line" not in checkpoint
            and "后台 Operation 输出已读到并完成汇报" not in checkpoint
        )
        assert "[body context omitted; not operation evidence]" in checkpoint
        # Serialized production text is evidence, NOT a live followUp; a new
        # latest user request must not read the historical terminal ID.
        resumed = TestOperationFakeNotificationOrdering._body(
            [
                {
                    "role": "user",
                    "content": "The conversation history before this point was compacted into the following summary:\n\n<summary>\n"
                    + checkpoint
                    + "\n</summary>",
                },
                {"role": "user", "content": "后台跑另一个命令"},
                {"role": "user", "content": context["content"]},
            ]
        )
        assert "command" in TestOperationFakeSummarization._call(
            fake.answer(resumed), "process_start", "call_opstart"
        )

    @pytest.mark.parametrize("variant", ["quoted", "assistant", "later_user", "missing_id"])
    def test_production_text_does_not_override_role_or_latest_turn(self, variant: str) -> None:
        notification, context = self._window()
        text = notification["content"]
        last = {"role": "user", "content": "Explain this quoted notification: " + text}
        if variant == "assistant":
            last = {"role": "assistant", "content": text}
        elif variant == "later_user":
            last = {"role": "user", "content": "后台跑一个新命令"}
        elif variant == "missing_id":
            last = {"role": "user", "content": text.replace("op_2db9e0c53ec148c69e060539", "")}
        body = TestOperationFakeNotificationOrdering._body(
            [
                {"role": "user", "content": text},
                {"role": "assistant", "content": "Prior report"},
                last,
                {"role": "user", "content": context["content"]},
            ]
        )
        assert "command" in TestOperationFakeSummarization._call(
            _OperationFake().answer(body), "process_start", "call_opstart"
        )

    @pytest.mark.parametrize("call_id", ["call_opstart", "call_opoutput"])
    @pytest.mark.parametrize("bridge", [False, True])
    def test_full_notification_quoted_in_tool_is_not_a_custom_result(
        self, call_id: str, bridge: bool
    ) -> None:
        notification, context = self._window()
        messages = [{"role": "tool", "tool_call_id": call_id, "content": notification["content"]}]
        if bridge:
            messages.append({"role": "assistant", "content": "I have processed the tool results."})
        messages.append({"role": "user", "content": [{"type": "text", "text": context["content"]}]})
        reply = TestOperationFakeNotificationOrdering._text(
            _OperationFake().answer(TestOperationFakeNotificationOrdering._body(messages)),
        )
        expected = (
            "已在后台启动 operation"
            if call_id == "call_opstart"
            else "后台 Operation 输出已读到并完成汇报"
        )
        assert expected in reply
