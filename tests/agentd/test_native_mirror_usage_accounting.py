"""NATIVE-TOKENS：native message_end 镜像用量入账回归测试。

覆盖修复语义：
- pi.usage 的 native_usage 只数 assistant message_end（turn_end 永不
  重复计数），known/cache/unknown 计数真实而非恒零。
- 同 mirror_id 相同载荷 = 幂等 ok stored0；同 id 不同载荷 = typed
  conflict 且原始字节不变；批次内任一冲突/非法 usage 原子拒绝——
  不提交更早前缀。
- bool/负数/非有限 token 计数与非有限成本在落库前 typed 拒绝。
- 空 pi_entry_id 且内容/usage 相同的两条历史镜像仍是两条独立付费
  消息（不按内容 hash 去重）。
- usage/成本缺失计 UNKNOWN——绝不显示为已确认零付费或免费。
- legacy（人民币 microunits）与 native（USD 估计）分离，不求虚假
  总计；legacy 顶层合计保持向后兼容。
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from rosclaw.agentd.config import load_agent_config
from rosclaw.agentd.models.gateway import MockModelGateway
from rosclaw.agentd.models.profiles import mock_profile
from rosclaw.agentd.pi_bridge.server import PiBridgeServer
from rosclaw.agentd.service import AgentService
from rosclaw.agentd.usage import UsageRecorder, validate_mirror_usage
from rosclaw.contracts.agent.model_turn import ModelTurnResultV1, ModelUsage
from tests.agentd.conftest import LOCAL_PRINCIPAL


def _event(
    mission_id: str,
    mirror_id: str = "mir_t1",
    *,
    event_type: str = "message_end",
    entry_id: str = "resp_1",
    usage: dict | None = None,
) -> dict:
    return {
        "mirror_id": mirror_id,
        "pi_session_id": "pi_test",
        "mission_id": mission_id,
        "event_type": event_type,
        "pi_entry_id": entry_id,
        "content_hash": "sha256:inert",
        "model": "k3",
        "usage": {
            "input": 10,
            "cacheRead": 20,
            "cacheWrite": 3,
            "output": 7,
            "reasoning": 2,
            "totalTokens": 40,
            "cost": {
                "input": 0.1,
                "output": 0.2,
                "cacheRead": 0.01,
                "cacheWrite": 0.02,
                "total": 0.33,
            },
        }
        if usage is None
        else usage,
        "occurred_at": "2026-10-06T00:00:00+00:00",
    }


@pytest.fixture
async def service(tmp_path: Path):
    turn = ModelTurnResultV1(
        turn_id="unused",
        provider="mock",
        model="unused",
        content="unused",
        assistant_message={"role": "assistant", "content": "unused"},
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )
    svc = AgentService(
        load_agent_config(tmp_path / "config.yaml"),
        tmp_path,
        gateway=MockModelGateway(mock_profile(), [turn]),
    )
    yield svc
    svc._store.connection.rollback()
    await svc.close()


class _Harness:
    def __init__(self, service: AgentService, mission_id: str, tmp_path: Path) -> None:
        self.service = service
        self.mission_id = mission_id
        self.bridge = PiBridgeServer(service, tmp_path / "run" / "pi.sock")

    async def send(self, events: list[dict]) -> dict:
        return await self.bridge._dispatch(
            LOCAL_PRINCIPAL,
            1,
            "pi.events.batch",
            {"token": self.service.control_token, "events": events},
        )

    def snapshot(self) -> list[tuple]:
        return [
            tuple(r)
            for r in self.service._store.connection.execute(
                "SELECT * FROM pi_event_mirrors ORDER BY mirror_id"
            )
        ]

    def native(self) -> dict:
        return self.service.usage_report(self.mission_id)["native_usage"]


@pytest.fixture
async def harness(service: AgentService, tmp_path: Path):
    mission = service.create_mission("native usage accounting probe")
    return _Harness(service, mission.mission_id, tmp_path)


class TestNativeUsageReport:
    async def test_known_cache_math_and_usd(self, harness: _Harness) -> None:
        result = await harness.send([_event(harness.mission_id)])
        assert result["ok"], result
        n = harness.native()
        assert n["known_message_count"] == 1
        assert n["input_uncached"] == 10
        assert n["cache_read"] == 20
        assert n["cache_write"] == 3
        # PI 归一化 input 不含缓存：inclusive = input+cacheRead+cacheWrite。
        assert n["input_including_cache"] == 33
        assert n["output"] == 7
        # reasoning 是 output 子集——单列但不二次累加。
        assert n["reasoning_subset_output"] == 2
        assert n["total_tokens"] == 40
        assert n["unknown_message_count"] == 0
        assert math.isclose(n["cost_usd_estimate"], 0.33)
        assert n["cost_unknown_message_count"] == 0
        assert n["identity_scope"]

    async def test_turn_end_never_counted(self, harness: _Harness) -> None:
        result = await harness.send(
            [
                _event(harness.mission_id, "mir_msg"),
                _event(harness.mission_id, "mir_turn", event_type="turn_end"),
            ]
        )
        assert result["ok"], result
        n = harness.native()
        assert n["known_message_count"] == 1
        assert n["total_tokens"] == 40

    async def test_missing_usage_is_unknown_not_free(self, harness: _Harness) -> None:
        result = await harness.send([_event(harness.mission_id, usage={})])
        assert result["ok"], result
        n = harness.native()
        assert n["known_message_count"] == 0
        assert n["unknown_message_count"] == 1
        # 未知不是免费的 0。
        assert n["cost_usd_estimate"] is None
        assert n["cost_unknown_message_count"] == 1

    async def test_missing_cost_is_unknown_not_free(self, harness: _Harness) -> None:
        event = _event(harness.mission_id)
        event["usage"].pop("cost")
        result = await harness.send([event])
        assert result["ok"], result
        n = harness.native()
        assert n["known_message_count"] == 1
        assert n["cost_usd_estimate"] is None
        assert n["cost_unknown_message_count"] == 1

    async def test_historical_identical_records_stay_distinct(self, harness: _Harness) -> None:
        # 两条空 pi_entry_id、内容 hash/usage 完全相同的历史镜像是两笔
        # 独立付费消息——不按内容 hash 去重。
        result = await harness.send(
            [
                _event(harness.mission_id, "mir_h1", entry_id=""),
                _event(harness.mission_id, "mir_h2", entry_id=""),
            ]
        )
        assert result["ok"], result
        n = harness.native()
        assert n["known_message_count"] == 2
        assert n["total_tokens"] == 80

    async def test_mixed_sources_separate_no_false_grand_total(self, harness: _Harness) -> None:
        assert (await harness.send([_event(harness.mission_id)]))["ok"]
        UsageRecorder(harness.service._store.connection).record(
            ModelTurnResultV1(
                turn_id="legacy",
                mission_id=harness.mission_id,
                provider="legacy",
                model="legacy",
                profile="p",
                usage=ModelUsage(
                    prompt_tokens=5,
                    completion_tokens=4,
                    reasoning_tokens=0,
                    total_tokens=9,
                    cost_microunits=7,
                ),
                latency_ms=1,
                finish_reason="stop",
            )
        )
        harness.service._store.connection.commit()
        report = harness.service.usage_report(harness.mission_id)
        # legacy 顶层合计向后兼容、保持旧口径（人民币 microunits）。
        assert report["total_tokens"] == 9
        assert report["cost_microunits"] == 7
        assert report["model_turns"] == 1
        # native 单列——绝不与 legacy 求和成虚假总计。
        assert report["native_usage"]["total_tokens"] == 40
        assert math.isclose(report["native_usage"]["cost_usd_estimate"], 0.33)
        assert report["legacy_scope"]


class TestBatchIdempotenceAndAtomicity:
    async def test_identical_mirror_id_replay_ok_stored0(self, harness: _Harness) -> None:
        event = _event(harness.mission_id)
        assert (await harness.send([event]))["ok"]
        before = harness.snapshot()
        result = await harness.send([event])
        assert result["ok"] and result["stored"] == 0
        assert harness.snapshot() == before

    async def test_same_id_different_payload_typed_conflict(self, harness: _Harness) -> None:
        event = _event(harness.mission_id)
        assert (await harness.send([event]))["ok"]
        before = harness.snapshot()
        event["usage"]["output"] = 8
        result = await harness.send([event])
        assert not result["ok"]
        assert result.get("code") == "MIRROR_CONFLICT"
        # 原始字节不变。
        assert harness.snapshot() == before

    async def test_batch_prefix_conflict_atomic(self, harness: _Harness) -> None:
        event = _event(harness.mission_id)
        assert (await harness.send([event]))["ok"]
        before = harness.snapshot()
        conflicted = _event(harness.mission_id)
        conflicted["usage"]["output"] = 8
        result = await harness.send(
            [
                _event(harness.mission_id, "mir_new_prefix"),
                conflicted,
            ]
        )
        assert not result["ok"]
        assert result.get("code")
        # 原子性：合法前缀 mir_new_prefix 也未提交。
        assert harness.snapshot() == before

    async def test_fulltext_prohibition_preserved(self, harness: _Harness) -> None:
        event = _event(harness.mission_id)
        event["content"] = "private assistant text"
        result = await harness.send([event])
        assert not result["ok"]
        assert result.get("code") == "FULL_TEXT_FORBIDDEN"
        assert harness.snapshot() == []


class TestUsageValidation:
    @pytest.mark.parametrize(
        "field,value",
        [
            ("input", True),
            ("output", -1),
            ("cacheRead", float("nan")),
            ("cacheWrite", float("inf")),
            ("totalTokens", "40"),
        ],
    )
    async def test_invalid_token_counts_typed_before_insert(
        self, harness: _Harness, field: str, value: object
    ) -> None:
        event = _event(harness.mission_id)
        event["usage"][field] = value
        before = harness.snapshot()
        result = await harness.send([_event(harness.mission_id, "mir_ok"), event])
        assert not result["ok"]
        assert result.get("code") == "INVALID_USAGE"
        # 原子性：前面的合法事件也未提交。
        assert harness.snapshot() == before

    async def test_nonfinite_cost_typed(self, harness: _Harness) -> None:
        event = _event(harness.mission_id)
        event["usage"]["cost"]["total"] = float("nan")
        result = await harness.send([event])
        assert not result["ok"]
        assert result.get("code") == "INVALID_USAGE"
        assert harness.snapshot() == []

    async def test_negative_cost_typed(self, harness: _Harness) -> None:
        event = _event(harness.mission_id)
        event["usage"]["cost"]["total"] = -0.01
        result = await harness.send([event])
        assert not result["ok"]
        assert result.get("code") == "INVALID_USAGE"

    def test_validate_mirror_usage_accepts_empty_and_valid(self) -> None:
        assert validate_mirror_usage({}) is None
        assert validate_mirror_usage(None) is None
        assert validate_mirror_usage({"input": 0, "cost": {"total": 0.0}}) is None
        assert validate_mirror_usage({"input": True}) is not None
        assert validate_mirror_usage({"output": -1}) is not None
        assert validate_mirror_usage({"cacheRead": float("nan")}) is not None
        assert validate_mirror_usage({"cost": {"total": float("inf")}}) is not None

    def test_validate_mirror_usage_integer_boundaries(self) -> None:
        # 分数 token 是损坏数据——typed 拒绝，绝不截断。
        assert validate_mirror_usage({"input": 1.5}) is not None
        # 超出 JS 精确整数范围 → typed 拒绝。
        assert validate_mirror_usage({"input": 9007199254740992}) is not None
        # Python 超大 int（2**1024）：typed 拒绝，绝不 OverflowError。
        assert validate_mirror_usage({"input": 2**1024}) is not None
        assert validate_mirror_usage({"cost": {"total": 2**1024}}) is not None
        # 边界值本身合法。
        assert validate_mirror_usage({"input": 9007199254740991}) is None
        # 有限非负小数成本合法。
        assert validate_mirror_usage({"cost": {"total": 0.33}}) is None
        assert validate_mirror_usage({"cost": {"total": -0.01}}) is not None


class TestPartialUsage:
    async def test_cost_only_unknown_tokens_cost_preserved(self, harness: _Harness) -> None:
        # 仅成本记录：USD 估计已知，token 用量未知——不是精确的零。
        result = await harness.send([_event(harness.mission_id, usage={"cost": {"total": 1.0}})])
        assert result["ok"], result
        n = harness.native()
        assert n["known_message_count"] == 0
        assert n["unknown_message_count"] >= 1
        assert math.isclose(n["cost_usd_estimate"], 1.0)

    async def test_output_only_preserves_output_input_total_unknown(
        self, harness: _Harness
    ) -> None:
        result = await harness.send([_event(harness.mission_id, usage={"output": 7})])
        assert result["ok"], result
        n = harness.native()
        assert n["known_message_count"] == 0
        assert n["unknown_message_count"] >= 1
        # 已知字段事实保留。
        assert n["output"] == 7
        # 成本缺失 → 整体未知。
        assert n["cost_usd_estimate"] is None


class TestIntegerAndAggregateBoundaries:
    @pytest.mark.parametrize("value", [1.5, 9007199254740992])
    async def test_invalid_integer_token_typed_atomic(
        self, harness: _Harness, value: object
    ) -> None:
        event = _event(harness.mission_id)
        event["usage"]["input"] = value
        before = harness.snapshot()
        result = await harness.send([_event(harness.mission_id, "mir_ok"), event])
        assert not result["ok"]
        assert result.get("code") == "INVALID_USAGE"
        assert harness.snapshot() == before

    async def test_js_safe_boundary_accepted(self, harness: _Harness) -> None:
        usage = {
            "input": 9007199254740991,
            "cacheRead": 0,
            "cacheWrite": 0,
            "output": 0,
            "reasoning": 0,
            "totalTokens": 9007199254740991,
        }
        result = await harness.send([_event(harness.mission_id, usage=usage)])
        assert result["ok"], result
        n = harness.native()
        assert n["input_including_cache"] == 9007199254740991
        assert n["output"] == 0

    async def test_python_2pow1024_token_and_cost_typed_atomic(self, harness: _Harness) -> None:
        for mutate in ("token", "cost"):
            event = _event(harness.mission_id, f"mir_huge_{mutate}")
            if mutate == "token":
                event["usage"]["input"] = 2**1024
            else:
                event["usage"]["cost"]["total"] = 2**1024
            before = harness.snapshot()
            result = await harness.send(
                [
                    _event(harness.mission_id, f"mir_prefix_{mutate}"),
                    event,
                ]
            )
            assert not result["ok"]
            assert result.get("code") == "INVALID_USAGE"
            assert harness.snapshot() == before

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True, 1.5])
    async def test_historical_invalid_rows_no_crash_valid_cost_kept(
        self, harness: _Harness, value: object
    ) -> None:
        # 历史库里的非法行（绕过写入期校验）绝不能让 /tokens 崩溃；
        # 非法字段计 unknown，独立合法的成本仍然保留。
        import json as _json

        assert (await harness.send([_event(harness.mission_id)]))["ok"]
        bad = {"input": value, "output": 7, "cost": {"total": 0.5}}
        harness.service._store.connection.execute(
            "UPDATE pi_event_mirrors SET usage_json = ?", (_json.dumps(bad),)
        )
        harness.service._store.connection.commit()
        n = harness.native()
        assert n["unknown_message_count"] >= 1
        assert math.isclose(n["cost_usd_estimate"], 0.5)

    async def test_aggregate_beyond_js_safe_int_is_decimal_string(self, harness: _Harness) -> None:
        big = {
            "input": 9007199254740991,
            "cacheRead": 0,
            "cacheWrite": 0,
            "output": 0,
            "reasoning": 0,
            "totalTokens": 9007199254740991,
        }
        small = {
            "input": 2,
            "cacheRead": 0,
            "cacheWrite": 0,
            "output": 0,
            "reasoning": 0,
            "totalTokens": 2,
        }
        result = await harness.send(
            [
                _event(harness.mission_id, "mir_big", usage=big),
                _event(harness.mission_id, "mir_small", usage=small),
            ]
        )
        assert result["ok"], result
        n = harness.native()
        # 9007199254740993 超出 JS 精确整数范围——精确十进制字符串，
        # 绝不输出不安全的 JSON 数值。
        assert n["input_including_cache"] == "9007199254740993"
        import json as _json

        _json.dumps(n, allow_nan=False)

    async def test_finite_cost_sum_overflow_unknown_usd(self, harness: _Harness) -> None:
        a = _event(harness.mission_id, "mir_c1")
        a["usage"]["cost"]["total"] = 1e308
        b = _event(harness.mission_id, "mir_c2")
        b["usage"]["cost"]["total"] = 1e308
        result = await harness.send([a, b])
        assert result["ok"], result
        n = harness.native()
        # 各自有限的成本之和溢出 → USD 估计未知，绝不 Infinity/NaN。
        assert n["cost_usd_estimate"] is None
        import json as _json

        _json.dumps(n, allow_nan=False)


def _terminal(
    usage: dict, stop_reason: str | None = "stop", response_id: str | None = "response_1"
) -> dict:
    envelope: dict = {"stopReason": stop_reason, "responseId": response_id}
    return {**usage, "_rosclaw_terminal": envelope}


_ZERO_USAGE = {
    "input": 0,
    "cacheRead": 0,
    "cacheWrite": 0,
    "output": 0,
    "reasoning": 0,
    "totalTokens": 0,
    "cost": {"total": 0},
}


class TestTerminalAuthority:
    """NATIVE-TOKENS-TERMINAL：终端出处语义——completed 与
    error/aborted/pending/deferred 严格区分；responseId 不是完成权威。"""

    async def test_sdk_optional_reasoning_absent_keeps_mandatory_known(
        self, harness: _Harness
    ) -> None:
        usage = _terminal(
            {
                "input": 10,
                "cacheRead": 20,
                "cacheWrite": 3,
                "output": 7,
                "totalTokens": 40,
                "cost": {"total": 0.33},
            }
        )
        result = await harness.send([_event(harness.mission_id, usage=usage)])
        assert result["ok"], result
        n = harness.native()
        # 缺失的可选 reasoning 不让必备合计变成未知。
        assert n["known_message_count"] == 1
        assert n["total_tokens"] == 40
        # reasoning 细分显式未知——绝不伪造 0。
        assert n["reasoning_subset_output"] is None
        assert n["reasoning_unknown_message_count"] == 1

    async def test_reasoning_zero_and_positive_stay_output_subset(self, harness: _Harness) -> None:
        base = _event(harness.mission_id)["usage"]
        zero = _terminal({**base, "reasoning": 0})
        positive = _event(harness.mission_id, "mir_rp")["usage"]
        result = await harness.send(
            [
                _event(harness.mission_id, "mir_rz", usage=zero),
                _event(harness.mission_id, "mir_rp", usage=_terminal(positive)),
            ]
        )
        assert result["ok"], result
        n = harness.native()
        assert n["known_message_count"] == 2
        # reasoning 是 output 子集：单列、不二次累加；0 与正值都合法。
        assert n["reasoning_subset_output"] == 2
        assert n["output"] == 14

    @pytest.mark.parametrize("bad_reasoning", [True, -1, 1.5, float("nan"), "2"])
    async def test_malformed_present_reasoning_rejected_atomically(
        self, harness: _Harness, bad_reasoning: object
    ) -> None:
        event = _event(harness.mission_id)
        event["usage"]["reasoning"] = bad_reasoning
        before = harness.snapshot()
        result = await harness.send([_event(harness.mission_id, "mir_ok"), event])
        assert not result["ok"]
        assert result.get("code") == "INVALID_USAGE"
        assert harness.snapshot() == before

    @pytest.mark.parametrize("status", ["stop", "toolUse", "length"])
    async def test_explicit_completed_zero_remains_valid_free(
        self, harness: _Harness, status: str
    ) -> None:
        usage = _terminal(dict(_ZERO_USAGE), status)
        result = await harness.send([_event(harness.mission_id, usage=usage)])
        assert result["ok"], result
        n = harness.native()
        assert n["known_message_count"] == 1
        assert n["cost_usd_estimate"] == 0

    @pytest.mark.parametrize("status", ["error", "aborted", "pending", "deferred"])
    @pytest.mark.parametrize("response_id", [None, "response_1"])
    async def test_incomplete_terminal_zero_never_confirmed_free(
        self, harness: _Harness, status: str, response_id: str | None
    ) -> None:
        usage = _terminal(dict(_ZERO_USAGE), status, response_id)
        result = await harness.send(
            [_event(harness.mission_id, usage=usage, entry_id=response_id or "")]
        )
        assert result["ok"], result
        n = harness.native()
        # responseId 存在也不是完成/计费权威。
        assert n["known_message_count"] == 0
        assert n["unknown_message_count"] == 1
        assert n["cost_usd_estimate"] is None
        assert n["cost_unknown_message_count"] >= 1
        assert n["billing_authority_unknown_message_count"] >= 1

    async def test_incomplete_terminal_positive_partial_facts_preserved(
        self, harness: _Harness
    ) -> None:
        usage = _terminal(
            {
                "input": 1891,
                "cacheRead": 249600,
                "cacheWrite": 0,
                "output": 0,
                "totalTokens": 251491,
                "cost": {"total": 0.080553},
            },
            "aborted",
            "msg_eKu7L5edtB4Q12HBl8C3777U",
        )
        result = await harness.send([_event(harness.mission_id, usage=usage)])
        assert result["ok"], result
        n = harness.native()
        # 已上报的部分成本/token 事实保留为下限——绝不是 0，也绝不
        # 被说成完整账单。
        assert n["unknown_message_count"] == 1
        assert math.isclose(n["cost_usd_known_subtotal"], 0.080553)
        assert n["cost_usd_estimate"] is None
        assert n["total_tokens"] == 251491

    async def test_historical_zero_placeholder_without_status_unknown(
        self, harness: _Harness
    ) -> None:
        result = await harness.send(
            [
                _event(harness.mission_id, "mir_z1", entry_id="", usage=dict(_ZERO_USAGE)),
                _event(harness.mission_id, "mir_z2", entry_id="", usage=dict(_ZERO_USAGE)),
            ]
        )
        assert result["ok"], result
        n = harness.native()
        assert n["known_message_count"] == 0
        assert n["unknown_message_count"] == 2
        assert n["cost_usd_estimate"] is None
        assert n["cost_unknown_message_count"] == 2
        assert n["billing_authority_unknown_message_count"] >= 2

    async def test_historical_nonzero_facts_and_billing_authority_independent(
        self, harness: _Harness
    ) -> None:
        result = await harness.send(
            [
                _event(harness.mission_id, "mir_h1", entry_id=""),
                _event(harness.mission_id, "mir_h2", entry_id=""),
            ]
        )
        assert result["ok"], result
        n = harness.native()
        # 历史非零用量保留已知事实/旧估计，但计费权威独立标注未知——
        # 绝不反向推断历史 abort/error。
        assert n["known_message_count"] == 2
        assert n["total_tokens"] == 80
        assert math.isclose(n["cost_usd_estimate"], 0.66)
        assert math.isclose(n["cost_usd_known_subtotal"], 0.66)
        assert n["billing_authority_unknown_message_count"] >= 2

    @pytest.mark.parametrize(
        "bad",
        [
            True,
            "not_a_status",
            {"stopReason": True, "responseId": None},
            {"stopReason": "not_a_status", "responseId": None},
            {"stopReason": "stop", "responseId": 3},
        ],
    )
    async def test_bad_terminal_metadata_typed_atomic(self, harness: _Harness, bad: object) -> None:
        event = _event(harness.mission_id)
        event["usage"]["_rosclaw_terminal"] = bad
        before = harness.snapshot()
        result = await harness.send([_event(harness.mission_id, "mir_prefix"), event])
        assert not result["ok"]
        assert result.get("code") == "INVALID_USAGE"
        # 原子性：合法前缀也未提交。
        assert harness.snapshot() == before


class TestStableEntryProtocol:
    """BOUNDED-IDENTITY：prospective durable stable-entry/v1 协议。

    - 显式 identity_protocol="stable-entry/v1" 且 pi_entry_id 非空：
      durable 稳定身份 (session, mission, event_type, entry_id) 准入——
      同键同语义载荷幂等（新 transport mirror_id/occurred_at 不重复
      计数），同键不同语义载荷 typed conflict 原子回滚。
    - 标记存在但非法/未知：任何变更前 typed 拒绝整批；缺省=legacy，
      mirror_id 行为完全不变（无回填/无历史合并）。
    """

    @staticmethod
    def _stable(mission_id: str, mirror_id: str, entry_id: str, **kw) -> dict:
        event = _event(mission_id, mirror_id, entry_id=entry_id, **kw)
        event["identity_protocol"] = "stable-entry/v1"
        return event

    async def test_stable_replay_new_transport_id_no_double_count(self, harness: _Harness) -> None:
        event = self._stable(harness.mission_id, "mir_orig", "resp_stable")
        result = await harness.send([event])
        assert result["ok"] and result["stored"] == 1
        before = harness.snapshot()
        native_before = harness.native()
        # 同稳定身份、同语义载荷，新 transport mirror_id + 新到达时间：
        # durable 幂等——stored0，绝不重复计费。
        replay = dict(event)
        replay["mirror_id"] = "mir_new_transport"
        replay["occurred_at"] = "2030-01-01T00:00:00+00:00"
        result = await harness.send([replay])
        assert result["ok"] and result["stored"] == 0
        assert harness.snapshot() == before
        assert harness.native() == native_before

    async def test_stable_conflict_typed_atomic(self, harness: _Harness) -> None:
        event = self._stable(harness.mission_id, "mir_orig", "resp_stable")
        assert (await harness.send([event]))["ok"]
        before = harness.snapshot()
        conflict = self._stable(harness.mission_id, "mir_other", "resp_stable")
        conflict["usage"] = dict(conflict["usage"], output=8)
        result = await harness.send(
            [self._stable(harness.mission_id, "mir_new_prefix", "resp_new"), conflict]
        )
        assert not result["ok"]
        assert result.get("code") == "MIRROR_CONFLICT"
        # 原子性：合法前缀未提交，原始行不变。
        assert harness.snapshot() == before
        # 同稳定身份同载荷重放仍幂等。
        same = await harness.send([event])
        assert same["ok"] and same["stored"] == 0

    async def test_stable_namespace_separation(self, harness: _Harness) -> None:
        base = self._stable(harness.mission_id, "mir_ns_a", "resp_ns")
        other_session = dict(base, mirror_id="mir_ns_b", pi_session_id="pi_other")
        other_type = self._stable(harness.mission_id, "mir_ns_c", "resp_ns", event_type="turn_end")
        result = await harness.send([base, other_session, other_type])
        assert result["ok"] and result["stored"] == 3

    async def test_stable_identityless_not_deduplicated(self, harness: _Harness) -> None:
        # 空 pi_entry_id 即使声明 stable-entry/v1 也不按稳定身份合并——
        # 无身份消息保持 transport 身份独立。
        result = await harness.send(
            [
                self._stable(harness.mission_id, "mir_e1", ""),
                self._stable(harness.mission_id, "mir_e2", ""),
            ]
        )
        assert result["ok"] and result["stored"] == 2

    @pytest.mark.parametrize("marker", ["stable-entry/v2", "", None, True, {"version": 1}])
    async def test_malformed_protocol_rejected_before_mutation(
        self, harness: _Harness, marker: object
    ) -> None:
        good = self._stable(harness.mission_id, "mir_good", "resp_good")
        bad = self._stable(harness.mission_id, "mir_bad", "resp_bad")
        bad["identity_protocol"] = marker
        result = await harness.send([good, bad])
        assert not result["ok"]
        assert result.get("code") == "UNSUPPORTED_IDENTITY_PROTOCOL"
        # 任何变更前整批拒绝——合法前缀也未提交。
        assert harness.snapshot() == []

    async def test_absent_marker_legacy_behavior_preserved(self, harness: _Harness) -> None:
        # legacy（无标记）：同 entry_id 不同 mirror_id 不同载荷仍然允许——
        # 无历史回填/合并，mirror_id 准入契约不变。
        first = _event(harness.mission_id, "mir_l1", entry_id="resp_legacy")
        second = _event(harness.mission_id, "mir_l2", entry_id="resp_legacy")
        second["usage"] = dict(second["usage"], output=8)
        assert (await harness.send([first]))["ok"]
        result = await harness.send([second])
        assert result["ok"] and result["stored"] == 1
        assert len(harness.snapshot()) == 2
