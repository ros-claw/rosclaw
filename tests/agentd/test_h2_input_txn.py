"""PR-H2 产品 Gate（总纲 v2 §21 Gate B）：输入与 Root Task 一致性。

PTY `rosclaw chat` 真实输入：
1. 首条任务消息 → root task 创建（revision 1）+ 消息可见（不消失）；
2. 追问 → revision 2（同一 task，不裂变）；
3. /new 后的新目标 → 新 task；
4. work_orders/task_executions 恒为 0（Worker 不参与默认链）。
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

from tests.agentd.test_h1_native_work import (
    _FakeServer,
    _prepare_home,
)
from tests.agentd.test_product_journey import PtySession

REPO = Path(__file__).resolve().parents[2]

import pytest  # noqa: E402

from rosclaw.agentd.pi_entry import find_pi_agent_entry  # noqa: E402

pytestmark = pytest.mark.skipif(
    not find_pi_agent_entry(),
    reason="无 Node/dist（CI 全回归 job 未构建）——诚实 skip",
)


H2_INPUTS = ("画一个五角星", "改成画圆形", "完全不同的新任务")


def completion_marker(text: str) -> bytes:
    return f"H2-TURN-{H2_INPUTS.index(text) + 1}-COMPLETE".encode()


class InputIdentifiedFake:
    """Keep H1's real write/bash flow; label only its final presentation.

    HTTP response IDs are already unique upstream. They are not visible PTY
    turn barriers; constant call_write/call_bash IDs are routing IDs, not turns.
    """

    def __init__(self, delegate) -> None:
        self.delegate = delegate.answer

    def answer(self, body: dict) -> bytes:
        payload = self.delegate(body)
        phrase = "已在同一会话直接完成".encode()
        if phrase not in payload:
            return payload
        import json

        # Match the latest user input, not an earlier assistant echo. No task
        # identity or admission decision is made here: only fixture output.
        for message in reversed(body.get("messages", [])):
            if message.get("role") != "user":
                continue
            content = json.dumps(message.get("content", ""), ensure_ascii=False)
            matches = [text for text in H2_INPUTS if text in content]
            if matches:
                assert len(matches) == 1, content
                return payload.replace(phrase, completion_marker(matches[0]) + b" " + phrase)
        raise AssertionError("H2 final response has no identifiable current user input")


def assert_input_binding(home: Path, text: str, *, force_new: bool) -> None:
    """Read-only evidence from the real persist/admission chain, no row seeding."""
    import contextlib

    with contextlib.closing(sqlite3.connect(home / "agentd" / "missions.db")) as db:
        rows = db.execute(
            "SELECT u.delivery_state, u.task_id, r.task_id, r.revision, "
            "t.active_revision, b.active FROM user_inputs u "
            "JOIN task_revisions r ON r.user_message_id = u.message_id "
            "JOIN tasks t ON t.task_id = r.task_id "
            "JOIN task_session_bindings b ON b.task_id = t.task_id "
            "AND b.session_ref = u.session_ref AND b.role = 'primary' "
            "WHERE u.text = ?",
            (text,),
        ).fetchall()
    assert len(rows) == 1, (text, rows)
    state, input_task, revision_task, revision, active_revision, active = rows[0]
    assert state == ("FORCE_NEW" if force_new else "PERSISTED"), rows
    assert input_task == revision_task and revision == active_revision and active == 1, rows


class TestInputRootTaskGate:
    def test_input_binds_single_root_task(self, tmp_path: Path) -> None:
        """大道至简 R0-1/R0-2b：输入 → Pi 模型回合 → 首个 effectful
        工具调用（write hello.txt）由 admission 建 root task；追问
         bump revision（不裂变）；/newtask 后新目标开新 task。
        与本测试创立时同一意图，但驱动面从「确定性链抢答」换成
        「Pi 工作」——h1 假模型对所有输入走 write→bash→回答流。
        降级环境（无 bwrap）：bash 的授权卡拒绝（fail closed）。
        """
        fake = _FakeServer()
        fake.fake.answer = InputIdentifiedFake(fake.fake).answer
        home, env = _prepare_home(tmp_path, fake.base_url)
        workspace = tmp_path / "ws"
        workspace.mkdir()
        session = PtySession(
            [sys.executable, "-m", "rosclaw.entrypoint", "chat"],
            env,
            log_path=tmp_path / "pty-h2.log",
            cwd=workspace,
        )

        def _drive_one_input(text: str) -> None:
            """Wait for this input's labelled final, then inspect real binding.

            An after offset alone cannot exclude a redraw of the old final.
            SIM write/bash admission is unchanged; no approval card is faked.
            """
            marker = len(session.clean)
            session.send(text + "\r")
            session.expect(completion_marker(text), timeout=180, after=marker)
            assert_input_binding(home, text, force_new=text == H2_INPUTS[2])

        try:
            session.expect(b"ROSClaw Native Agent", timeout=120)
            _drive_one_input("画一个五角星")
            db = sqlite3.connect(home / "agentd" / "missions.db")
            tasks = db.execute("SELECT task_id, state, active_revision FROM tasks").fetchall()
            db.close()
            assert len(tasks) == 1, f"首个 effectful 调用必须只建一个 root task: {tasks}"
            task_id = tasks[0][0]
            assert tasks[0][2] == 1
            assert "画一个五角星" in session.clean.decode("utf-8", errors="replace"), (
                "屏幕无指令回声"
            )
            db = sqlite3.connect(home / "agentd" / "missions.db")
            input_row = db.execute(
                "SELECT text FROM user_inputs WHERE text LIKE '%画一个五角星%'"
            ).fetchone()
            db.close()
            assert input_row, "user_inputs 权威账本缺输入（幽灵执行防线）"

            # 追问 → revision 2（同一 task，不裂变）。
            _drive_one_input("改成画圆形")
            db = sqlite3.connect(home / "agentd" / "missions.db")
            tasks = db.execute("SELECT task_id, active_revision FROM tasks").fetchall()
            revisions = db.execute(
                "SELECT COUNT(*) FROM task_revisions WHERE task_id = ?",
                (task_id,),
            ).fetchone()[0]
            db.close()
            assert len(tasks) == 1, f"追问不得裂变: {tasks}"
            assert tasks[0][1] == 2 and revisions == 2

            # /newtask 后新目标 → 新 task。
            command_marker = len(session.clean)
            session.send("/newtask\r")
            session.expect(
                "下一条消息将开始新任务".encode(),
                timeout=60,
                after=command_marker,
            )
            _drive_one_input("完全不同的新任务")
            db = sqlite3.connect(home / "agentd" / "missions.db")
            count = db.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
            orders = db.execute("SELECT COUNT(*) FROM work_orders").fetchone()[0]
            execs = db.execute("SELECT COUNT(*) FROM task_executions").fetchone()[0]
            db.close()
            assert count == 2, f"/newtask 必须开新 task: {count}"
            assert orders == 0 and execs == 0
            session.expect_with_resend(b"rosclaw continue", "/quit\r", timeout=60)
            session.proc.wait(timeout=30)
        finally:
            try:
                session.stop()
                session.proc.wait(timeout=30)
            finally:
                fake.close()


def test_input_identified_fake_preserves_flow_and_excludes_old_redraw() -> None:
    """Exercise actual H1 producer and response-ID rewriter, not a mirror formula.

    The PTY integration above supplies the real persistence/admission evidence;
    this finite negative control isolates its presentation prerequisite.
    """
    import json

    from tests.agentd.test_h1_native_work import _DirectWorkFake
    from tests.agentd.test_product_journey import _FakeResponseIdentity

    provider = InputIdentifiedFake(_DirectWorkFake())
    history = []
    finals = []
    response_ids = []

    def frames(body: dict) -> list[dict]:
        raw = _FakeResponseIdentity.apply(provider.answer(body), stream=True)
        result = [
            json.loads(line[6:])
            for line in raw.splitlines()
            if line.startswith(b"data: ") and line != b"data: [DONE]"
        ]
        ids = {frame["id"] for frame in result}
        assert len(ids) == 1
        response_ids.append(ids.pop())
        return result

    for text in H2_INPUTS:
        history.append({"role": "user", "content": text})
        write = frames({"stream": True, "messages": list(history)})
        call = write[1]["choices"][0]["delta"]["tool_calls"][0]
        assert (call["id"], call["function"]["name"]) == ("call_write", "write")
        history.append({"role": "tool", "tool_call_id": "call_write", "content": "ok"})
        bash = frames({"stream": True, "messages": list(history)})
        call = bash[1]["choices"][0]["delta"]["tool_calls"][0]
        assert (call["id"], call["function"]["name"]) == ("call_bash", "bash")
        history.append({"role": "tool", "tool_call_id": "call_bash", "content": "ok"})
        final = frames({"stream": True, "messages": list(history)})
        assert final[-1]["choices"][0]["finish_reason"] == "stop"
        content = final[0]["choices"][0]["delta"]["content"].encode()
        assert completion_marker(text) in content
        finals.append(content)
        history.append({"role": "assistant", "content": content.decode()})

    assert len(set(response_ids)) == 9
    # Exact failure of the tempting offset-only premise: old content can be
    # newly appended by a redraw and satisfy the old predicate immediately.
    before = finals[0]
    marker = len(before)
    redrawn = before + finals[0]
    assert "已在同一会话直接完成".encode() in redrawn[marker:]
    assert completion_marker(H2_INPUTS[1]) not in redrawn[marker:]
    assert completion_marker(H2_INPUTS[1]) in (redrawn + finals[1])[marker:]
