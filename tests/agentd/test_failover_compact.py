"""Failover chain + compaction tests (model-layer leftovers).

- cooldown: 标准错误 1m→5m→25m 递增、成功复位、24h 清零、billing 长冷却
- failover: 非重试错误立即抛、rate_limited 跳下一个候选、冷却中跳过、
  末位等待、RPM 限流
- microcompact: 旧 tool result 折叠、中段裁剪保留锚点
- reactive compact: context-overflow 错误 → 压缩重试成功
"""

from __future__ import annotations

import pytest

from rosclaw.agentd.context.compact import (
    is_context_overflow,
    microcompact,
)
from rosclaw.agentd.models.failover import (
    CooldownTracker,
    FailoverGateway,
    TokenBucket,
)
from rosclaw.agentd.models.gateway import MockModelGateway, ModelGatewayError
from rosclaw.agentd.models.profiles import mock_profile
from rosclaw.contracts.agent.model_turn import ModelTurnResultV1


def _turn() -> ModelTurnResultV1:
    return ModelTurnResultV1(
        turn_id="t",
        provider="mock",
        model="m",
        content="ok",
        assistant_message={"role": "assistant", "content": "ok"},
    )


class TestCooldown:
    def test_escalating_cooldown(self) -> None:
        now = [0.0]
        tracker = CooldownTracker(now=lambda: now[0])
        tracker.record_failure("k")
        assert tracker.in_cooldown("k")
        now[0] = 59.0
        assert tracker.in_cooldown("k")  # 1min first cooldown
        now[0] = 61.0
        assert not tracker.in_cooldown("k")
        tracker.record_failure("k")
        now[0] = 61.0 + 4 * 60
        assert tracker.in_cooldown("k")  # 5min second cooldown
        now[0] = 61.0 + 6 * 60
        assert not tracker.in_cooldown("k")

    def test_success_resets(self) -> None:
        tracker = CooldownTracker()
        tracker.record_failure("k")
        tracker.record_success("k")
        assert not tracker.in_cooldown("k")

    def test_billing_long_cooldown(self) -> None:
        now = [0.0]
        tracker = CooldownTracker(now=lambda: now[0])
        tracker.record_failure("k", billing=True)
        now[0] = 3600.0
        assert tracker.in_cooldown("k")  # billing: 5h first

    def test_24h_auto_clear(self) -> None:
        now = [0.0]
        tracker = CooldownTracker(now=lambda: now[0])
        tracker.record_failure("k")
        now[0] = 25 * 3600.0
        assert not tracker.in_cooldown("k")


class TestFailover:
    async def test_rate_limit_falls_to_next(self) -> None:
        def fail(request):
            raise ModelGatewayError("rate_limited", "429")

        bad = MockModelGateway(mock_profile(name="bad", model="bad-model"), [fail])
        good = MockModelGateway(mock_profile(name="good", model="good-model"), [_turn()])
        gateway = FailoverGateway(
            [(bad.profile, bad), (good.profile, good)],
            now=__import__("time").monotonic,
        )
        from rosclaw.agentd.models.gateway import ModelTurnRequest

        result = await gateway.complete(ModelTurnRequest(system_prompt="s", messages=[]))
        assert result.content == "ok"
        assert len(good.requests) == 1

    async def test_non_retryable_raises_immediately(self) -> None:
        def fail(request):
            raise ModelGatewayError("auth_error", "401")

        bad = MockModelGateway(mock_profile(name="bad", model="bad-model"), [fail])
        good = MockModelGateway(mock_profile(name="good", model="good-model"), [_turn()])
        gateway = FailoverGateway([(bad.profile, bad), (good.profile, good)])
        from rosclaw.agentd.models.gateway import ModelTurnRequest

        with pytest.raises(ModelGatewayError, match="auth_error"):
            await gateway.complete(ModelTurnRequest(system_prompt="s", messages=[]))
        assert len(good.requests) == 0  # 不重试也不轮换

    async def test_cooldown_skips_candidate(self) -> None:
        calls = []

        def fail(request):
            calls.append("fail")
            raise ModelGatewayError("timeout", "slow")

        bad = MockModelGateway(mock_profile(name="bad", model="bad-model"), [fail, fail, _turn()])
        good = MockModelGateway(mock_profile(name="good", model="good-model"), [_turn(), _turn()])
        gateway = FailoverGateway([(bad.profile, bad), (good.profile, good)])
        from rosclaw.agentd.models.gateway import ModelTurnRequest

        await gateway.complete(ModelTurnRequest(system_prompt="s", messages=[]))
        # bad 已进入冷却，第二次直接跳过 bad 走 good。
        await gateway.complete(ModelTurnRequest(system_prompt="s", messages=[]))
        assert calls == ["fail"]
        assert len(good.requests) == 2

    def test_token_bucket(self) -> None:
        bucket = TokenBucket(2)
        assert bucket.take(now=0.0)
        assert bucket.take(now=0.0)
        assert not bucket.take(now=0.0)
        assert bucket.take(now=31.0)  # 半分钟后补 1 个


class TestCompaction:
    def test_microcompact_folds_old_tool_results(self) -> None:
        messages = [{"role": "user", "content": "开始"}]
        for i in range(10):
            messages.append({"role": "assistant", "content": None})
            messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "X" * 500})
        compacted, folded = microcompact(messages, keep_recent=4)
        assert folded == 8
        tool_contents = [m["content"] for m in compacted if m.get("role") == "tool"]
        assert tool_contents[-1] == "X" * 500  # 最近的不动
        assert "compacted" in tool_contents[0]

    def test_trim_keeps_first_user_anchor(self) -> None:
        messages = [{"role": "user", "content": "锚点"}] + [
            {"role": "user", "content": f"m{i}"} for i in range(80)
        ]
        compacted, _ = microcompact(messages, max_messages=50)
        assert compacted[0]["content"] == "锚点"
        assert len(compacted) == 50

    def test_overflow_detection(self) -> None:
        assert is_context_overflow("http_error", "HTTP 400: prompt is too long")
        assert is_context_overflow("invalid_response", "context_length_exceeded")
        assert not is_context_overflow("rate_limited", "429")
        assert not is_context_overflow("http_error", "HTTP 500 internal")


class TestReactiveCompaction:
    async def test_overflow_compact_and_retry(self) -> None:
        from rosclaw.agentd.context.compact import microcompact as mc
        from rosclaw.agentd.models.gateway import ModelTurnRequest

        calls = []

        def overflow_once(request):
            calls.append(len(request.messages))
            if len(calls) == 1:
                raise ModelGatewayError("http_error", "HTTP 400: prompt too long")
            return _turn()

        gateway = MockModelGateway(mock_profile(), [overflow_once, overflow_once])
        messages = [{"role": "user", "content": "锚点"}] + [
            {"role": "tool", "tool_call_id": f"c{i}", "content": "Y" * 300} for i in range(20)
        ]
        request = ModelTurnRequest(system_prompt="s", messages=messages)
        try:
            await gateway.complete(request)
            raise AssertionError("should have raised")
        except ModelGatewayError as exc:
            assert is_context_overflow(exc.kind, str(exc))
            compacted, _ = mc(messages, keep_recent=4)
            result = await gateway.complete(ModelTurnRequest(system_prompt="s", messages=compacted))
            assert result.content == "ok"
