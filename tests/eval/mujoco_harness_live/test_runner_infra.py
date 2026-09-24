"""Runner infra 失败诚实性（live 标定实证 2026-09-23）：

stall/未收束的跑真实烧了 settle_timeout 全程，旧实现落
wall_time_s=0.0——聚合 median/P95 被假零污染，且丢 model 字段
（09-23 上午 pilot 6 场 ERROR 全是 0.0s/"?"）。修复：
RunInfraError 携带部分记录，error_record 保留真实指标。
"""

from __future__ import annotations

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

    monkeypatch.setattr(
        "tests.agentd.test_product_journey.PtySession", _FakeSession, raising=True
    )
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
