"""Runner infra 失败诚实性（live 标定实证 2026-09-23）：

stall/未收束的跑真实烧了 settle_timeout 全程，旧实现落
wall_time_s=0.0——聚合 median/P95 被假零污染，且丢 model 字段
（09-23 上午 pilot 6 场 ERROR 全是 0.0s/"?"）。修复：
RunInfraError 携带部分记录，error_record 保留真实指标。
"""

from __future__ import annotations

from datetime import UTC

import pytest

from benchmarks.harnessbench import runner
from benchmarks.harnessbench.runner import RunInfraError, error_record, run_leg


def test_error_record_preserves_partial_from_infra_error() -> None:
    """stall 异常携带的部分记录（真实 wall_time/tool_calls）必须落账。"""
    partial = {
        "leg": "B",
        "task_id": "R01",
        "category": "repair",
        "model": "deepseekv4",
        "run": 1,
        "wall_time_s": 1210.4,
        "tool_calls": 3,
        "glue_bytes": 128,
        "verdict": "ERROR",
    }
    exc = RunInfraError("INFRA_STALL: 两次重发后仍无模型活动", partial)
    record = error_record(exc, "B", "R01", 1, model="deepseekv4")
    assert record["verdict"] == "ERROR"
    assert record["wall_time_s"] == 1210.4  # 真实耗时，不是 0.0
    assert record["tool_calls"] == 3
    assert record["glue_bytes"] == 128
    assert record["model"] == "deepseekv4"
    assert record["oracle"]["reason"] == "runner_error"
    assert record["oracle"]["verified_success"] is False
    assert record["error"].startswith("INFRA_STALL")


def test_error_record_generic_exception_zeroed_but_complete() -> None:
    """非 infra 异常（无部分记录）：指标归零但字段齐全（聚合不 KeyError）。"""
    record = error_record(ValueError("boom"), "A", "U01", 2, model="kimi-k3")
    assert record["verdict"] == "ERROR"
    assert record["wall_time_s"] == 0.0
    assert record["tool_calls"] == 0
    assert record["model"] == "kimi-k3"
    for key in ("bash_python_loc", "python_loc", "xml_loc", "infra_retries", "glue_bytes"):
        assert key in record
    assert record["oracle"]["reason"] == "runner_error"


def test_run_infra_error_is_assertion_error_compat() -> None:
    """向后兼容：既有 catch AssertionError 的调用方不受影响。"""
    exc = RunInfraError("stall", {})
    assert isinstance(exc, AssertionError)


def test_run_leg_stall_raises_with_real_wall_time(tmp_path, monkeypatch) -> None:
    """run_leg 在 settle 超时时抛 RunInfraError 且 partial 带真实耗时。"""

    class _FakeSession:
        def __init__(self, argv, env, cwd=None, log_path=None) -> None:  # noqa: ANN001
            self.output = bytearray()

        def expect(self, pattern: bytes, timeout: float = 60) -> None:
            pass

        def send(self, data: str) -> None:
            pass

        def stop(self) -> None:
            pass

    def _stall(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("回合 1200.0s 未收束（见 PTY 日志）")

    monkeypatch.setattr("tests.agentd.test_product_journey.PtySession", _FakeSession, raising=True)
    monkeypatch.setattr(runner, "_wait_settled", _stall)

    with pytest.raises(RunInfraError) as excinfo:
        run_leg("B", "U01", tmp_path, 1, settle_timeout=1.0, model="deepseekv4")
    partial = excinfo.value.partial
    assert partial["leg"] == "B"
    assert partial["task_id"] == "U01"
    assert partial["model"] == "deepseekv4"
    assert isinstance(partial["wall_time_s"], float)
    assert partial["wall_time_s"] >= 0.0
    # 交给 error_record 后真实耗时不被抹零。
    record = error_record(excinfo.value, "B", "U01", 1, model="deepseekv4")
    assert record["wall_time_s"] == partial["wall_time_s"]


def test_recoverable_failure_markers() -> None:
    """provider 失败标记覆盖（live 实证 2026-09-23 D02）：kimi 超时
    波次必须触发重发，正常文本不误触。"""
    from benchmarks.harnessbench.runner import _is_recoverable_failure

    assert _is_recoverable_failure(b"Error: Request timed out. ")
    assert _is_recoverable_failure(b"Error: Retry failed after 1 attempts: Request timed out.")
    assert _is_recoverable_failure(b"Operation aborted")
    assert _is_recoverable_failure("已取消本次请求".encode())
    # 正常输出/模型散文里的英文单词不误触
    assert not _is_recoverable_failure(b"working... analysing trace")
    assert not _is_recoverable_failure("超时重试是常见策略".encode())
    assert not _is_recoverable_failure(b"")


def test_terminal_stop_near_deadline_does_not_require_quiet(tmp_path, monkeypatch):
    """A real final stop at 597s must not become a 600s infra timeout."""
    import json
    import threading
    from datetime import datetime

    clock = [0.0]
    sessions = tmp_path / "rh" / "agent" / "sessions"
    sessions.mkdir(parents=True)

    class Session:
        _lock = threading.Lock()
        output = bytearray(b"prompt")

    def sleep(seconds):
        clock[0] += seconds
        if clock[0] == 597:
            entry = {
                "type": "message",
                "timestamp": datetime.fromtimestamp(1000 + clock[0], UTC).isoformat(),
                "message": {"role": "assistant", "stopReason": "stop", "content": []},
            }
            (sessions / "test.jsonl").write_text(json.dumps(entry) + "\n")

    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(runner.time, "time", lambda: 1000 + clock[0])
    monkeypatch.setattr(runner.time, "sleep", sleep)
    assert runner._wait_settled(Session(), tmp_path, 600, session_dir=sessions) == 0
    assert clock[0] == 597


@pytest.mark.parametrize("reason,role", [("toolUse", "assistant"), (None, "toolResult")])
def test_terminal_detector_does_not_accept_tools(tmp_path, reason, role):
    import json

    p = tmp_path / "sessions"
    p.mkdir()
    (p / "s.jsonl").write_text(
        json.dumps(
            {
                "type": "message",
                "timestamp": "2026-10-03T06:16:42Z",
                "message": {"role": role, "stopReason": reason},
            }
        )
        + "\n"
    )
    assert not runner._session_turn_stopped(p, 0, float("inf"))


def test_terminal_detector_rejects_old_history_and_late_stop(tmp_path):
    import json

    p = tmp_path / "sessions"
    p.mkdir()
    (p / "s.jsonl").write_text(
        json.dumps(
            {
                "type": "message",
                "timestamp": "1970-01-01T00:01:40Z",
                "message": {"role": "assistant", "stopReason": "stop"},
            }
        )
        + "\n"
    )
    assert not runner._session_turn_stopped(p, 101, 200)
    assert not runner._session_turn_stopped(p, 0, 99)
    assert runner._session_turn_stopped(p, 0, 100)


def test_terminal_stop_waits_for_native_background_operation(tmp_path):
    import json
    import sqlite3

    home = tmp_path / "rh"
    p = home / "agent" / "sessions"
    p.mkdir(parents=True)
    (p / "s.jsonl").write_text(
        json.dumps(
            {
                "type": "message",
                "timestamp": "1970-01-01T00:01:40Z",
                "message": {"role": "assistant", "stopReason": "stop"},
            }
        )
        + "\n"
    )
    (home / "agentd").mkdir()
    with sqlite3.connect(home / "agentd" / "missions.db") as conn:
        conn.execute("CREATE TABLE operations (state TEXT, ended_at TEXT)")
        conn.execute("INSERT INTO operations VALUES ('DEGRADED', NULL)")
    assert not runner._session_turn_stopped(p, 0, 200)
    with sqlite3.connect(home / "agentd" / "missions.db") as conn:
        conn.execute("UPDATE operations SET state='SUCCEEDED'")
    assert runner._session_turn_stopped(p, 0, 200)


def test_terminal_detector_does_not_accept_partial_followup(tmp_path):
    import json

    p = tmp_path / "sessions"
    p.mkdir()
    (p / "s.jsonl").write_text(
        json.dumps(
            {
                "type": "message",
                "timestamp": "1970-01-01T00:01:40Z",
                "message": {"role": "assistant", "stopReason": "stop"},
            }
        )
        + '\n{"type":"message"'
    )
    assert not runner._session_turn_stopped(p, 0, 200)


def test_background_finished_after_budget_is_not_accepted(tmp_path):
    import json
    import sqlite3

    home = tmp_path / "rh"
    p = home / "agent" / "sessions"
    p.mkdir(parents=True)
    (p / "s.jsonl").write_text(
        json.dumps(
            {
                "type": "message",
                "timestamp": "1970-01-01T00:01:40Z",
                "message": {"role": "assistant", "stopReason": "stop"},
            }
        )
        + "\n"
    )
    (home / "agentd").mkdir()
    with sqlite3.connect(home / "agentd" / "missions.db") as conn:
        conn.execute("CREATE TABLE operations (state TEXT, ended_at TEXT)")
        conn.execute("INSERT INTO operations VALUES ('SUCCEEDED', '1970-01-01T00:03:21Z')")
    assert not runner._session_turn_stopped(p, 0, 200)


def test_followup_user_message_invalidates_previous_stop(tmp_path):
    import json

    p = tmp_path / "sessions"
    p.mkdir()
    entries = [
        {
            "type": "message",
            "timestamp": "1970-01-01T00:01:40Z",
            "message": {"role": "assistant", "stopReason": "stop"},
        },
        {
            "type": "message",
            "timestamp": "1970-01-01T00:01:41Z",
            "message": {"role": "user", "content": []},
        },
    ]
    (p / "s.jsonl").write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    assert not runner._session_turn_stopped(p, 0, 200)
