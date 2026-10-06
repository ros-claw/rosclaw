"""Durable model usage metering (PR-NA-030b).

One row per model turn in ``model_usage`` (migration 003); aggregates are
computed, never stored as mutable counters. Cost is computed from the
profile's per-million-token prices in microunits (1 unit = 1e-6 元, so
budgets can be compared against ``monetary_microunits``).
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import UTC, datetime
from typing import Any

from rosclaw.contracts.agent.model_turn import ModelTurnResultV1
from rosclaw.contracts.common import new_id

# PI 归一化用量字段（native message_end 镜像）：input 不含缓存；
# inclusive = input + cacheRead + cacheWrite；reasoning 是 output 子集。
_NATIVE_TOKEN_FIELDS = ("input", "cacheRead", "cacheWrite", "output", "reasoning", "totalTokens")

# PI/JavaScript 精确整数上限（2^53 - 1）。token 计数必须是不超过该
# 上限的有限非负整数；超出即 typed 拒绝——绝不做 float 截断/四舍五入。
_JS_SAFE_INTEGER_MAX = 9007199254740991

_TOKEN_RULE = "a finite non-negative integer <= 2^53-1 (JS-safe)"
_COST_RULE = "a finite non-negative number (USD)"

# NATIVE-TOKENS-TERMINAL：assistant message_end 的有界终端出处信封
# （hash-only 元数据，绝不携带原始错误/思考/文本）。stopReason 是
# SDK StopReason 实值；responseId 是当前 provider 响应身份（string
# 或 null）——responseId 存在本身绝不是完成/计费权威。
_TERMINAL_METADATA_KEY = "_rosclaw_terminal"
_TERMINAL_COMPLETED = ("stop", "toolUse", "length")
_TERMINAL_INCOMPLETE = ("error", "aborted", "pending", "deferred")
_TERMINAL_STATUSES = _TERMINAL_COMPLETED + _TERMINAL_INCOMPLETE
# 必备 token 合计字段。reasoning 在已安装 SDK 中合法可选
# （Usage.reasoning?: number，provider 不上报细分时缺失）——缺失
# 是"细分未知"，绝不是伪造 0，也不影响必备合计的 known 判定。
_MANDATORY_TOKEN_FIELDS = ("input", "cacheRead", "cacheWrite", "output", "totalTokens")


def _validate_terminal_metadata(value: Any) -> str | None:
    """校验有界终端出处信封：只允许 stopReason/responseId 两个字段，
    类型/取值非法一律 typed 拒绝（在任何落库之前）。"""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, dict):
        return f"usage.{_TERMINAL_METADATA_KEY} must be an object"
    for key in value:
        if key not in ("stopReason", "responseId"):
            return (
                f"usage.{_TERMINAL_METADATA_KEY} carries only stopReason/responseId "
                "(bounded hash-only metadata)"
            )
    stop = value.get("stopReason")
    if stop is not None and (
        isinstance(stop, bool) or not isinstance(stop, str) or stop not in _TERMINAL_STATUSES
    ):
        return f"usage.{_TERMINAL_METADATA_KEY}.stopReason must be one of {_TERMINAL_STATUSES}"
    response_id = value.get("responseId")
    if response_id is not None and (
        isinstance(response_id, bool) or not isinstance(response_id, str) or len(response_id) > 256
    ):
        return f"usage.{_TERMINAL_METADATA_KEY}.responseId must be a string <= 256 chars or null"
    return None


def _terminal_status(usage: dict[str, Any]) -> str | None:
    """聚合期保守读取终端状态：信封缺失/畸形 → None（无终端证据，
    绝不崩溃、绝不伪造 completed）。"""
    meta = usage.get(_TERMINAL_METADATA_KEY)
    if not isinstance(meta, dict):
        return None
    stop = meta.get("stopReason")
    if isinstance(stop, str) and stop in _TERMINAL_STATUSES:
        return stop
    return None


def _validate_token_value(field: str, value: Any) -> str | None:
    """单个 token 计数字段校验。Python 超大 int（如 2**1024）在
    调用 math.isfinite/float() 之前就被范围比较 typed 拒绝——
    绝不抛 OverflowError。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return f"usage.{field} must be {_TOKEN_RULE}"
    if isinstance(value, int):
        if value < 0 or value > _JS_SAFE_INTEGER_MAX:
            return f"usage.{field} must be {_TOKEN_RULE}"
        return None
    # float：必须有限、非负、且是整数值（分数 token 是损坏数据，
    # 静默 int() 截断会把坏数据变成假精确值）。
    if (
        not math.isfinite(value)
        or value < 0
        or not value.is_integer()
        or value > _JS_SAFE_INTEGER_MAX
    ):
        return f"usage.{field} must be {_TOKEN_RULE}"
    return None


def _validate_cost_value(field: str, value: Any) -> str | None:
    """单个 USD 成本字段校验：有限非负小数合法；Python 超大 int
    成本（2**1024）不能安全转 float → typed 拒绝（捕获
    OverflowError，绝不让它冒出调用方）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return f"usage.{field} must be {_COST_RULE}"
    if isinstance(value, int):
        if value < 0:
            return f"usage.{field} must be {_COST_RULE}"
        try:
            converted = float(value)
        except OverflowError:
            return f"usage.{field} must be {_COST_RULE}"
        if not math.isfinite(converted):
            return f"usage.{field} must be {_COST_RULE}"
        return None
    if not math.isfinite(value) or value < 0:
        return f"usage.{field} must be {_COST_RULE}"
    return None


def validate_mirror_usage(usage: Any) -> str | None:
    """校验 native 镜像携带的 PI 归一化 usage 载荷。

    返回 None 表示合法；否则返回拒绝原因（由调用方包装成 typed
    error）。bool/分数/负数/非有限/超出 JS 精确整数范围的 token 计数
    与非有限/负数成本一律在任何落库之前拒绝。空/缺失 usage 是合法
    的（UNKNOWN 语义——进行中/中止/缺失用量绝不能被当作已确认的
    零付费）。
    """
    if usage is None:
        return None
    if not isinstance(usage, dict):
        return "usage must be an object"
    for key in _NATIVE_TOKEN_FIELDS:
        value = usage.get(key)
        if value is None:
            continue
        error = _validate_token_value(key, value)
        if error is not None:
            return error
    cost = usage.get("cost")
    if cost is not None:
        if not isinstance(cost, dict):
            return "usage.cost must be an object"
        for key, value in cost.items():
            if value is None:
                continue
            error = _validate_cost_value(f"cost.{key}", value)
            if error is not None:
                return error
    return _validate_terminal_metadata(usage.get(_TERMINAL_METADATA_KEY))


def _known_token(value: Any) -> int | None:
    """历史镜像字段的保守读取：合法→int；缺失/非法→None（未知，
    绝不截断/伪造 0）。对 2**1024 等超大 int 安全（不做 float 转换）。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, int):
        return value if 0 <= value <= _JS_SAFE_INTEGER_MAX else None
    if math.isfinite(value) and value >= 0 and value.is_integer() and value <= _JS_SAFE_INTEGER_MAX:
        return int(value)
    return None


def _json_safe_count(value: int) -> int | str:
    """聚合计数超出 JS 精确整数范围时，输出精确十进制字符串
    （UNKNOWN 数值精度显式标注）——绝不产出不安全的 JSON 数值。
    """
    if value > _JS_SAFE_INTEGER_MAX:
        return str(value)
    return value


def native_mirror_usage_report(conn: sqlite3.Connection, mission_id: str) -> dict[str, Any]:
    """聚合 pi_event_mirrors 中 native assistant message_end 的付费用量。

    只数 message_end（turn_end 永不重复计数）；聚合实时计算，不存
    可变计数器。usage 缺失/为空/部分/含非法字段的消息计入
    unknown——绝不显示为已确认零付费；部分记录的已知字段事实
    （如 output）与独立合法的成本仍然保留。历史镜像 pi_entry_id
    可能为空，不构成稳定身份；两条空 entry、内容/用量相同的历史
    记录仍是两条独立付费消息（不按内容 hash 去重）。
    """
    rows = conn.execute(
        "SELECT usage_json FROM pi_event_mirrors "
        "WHERE mission_id = ? AND event_type = 'message_end'",
        (mission_id,),
    ).fetchall()
    known = 0
    unknown = 0
    billing_authority_unknown = 0
    reasoning_unknown = 0
    input_uncached = 0
    cache_read = 0
    cache_write = 0
    output = 0
    reasoning = 0
    total_tokens = 0
    cost_sum = 0.0
    cost_known = 0
    cost_unknown = 0
    cost_subtotal = 0.0
    cost_reported = 0
    for row in rows:
        try:
            usage = json.loads(row["usage_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            usage = {}
        if not isinstance(usage, dict) or not usage:
            # 进行中/中止/缺失：未知，不是免费的 0。
            unknown += 1
            cost_unknown += 1
            billing_authority_unknown += 1
            reasoning_unknown += 1
            continue
        # 终端出处：completed（stop/toolUse/length）/ incomplete
        # （error/aborted/pending/deferred）/ None（历史无终端证据）。
        status = _terminal_status(usage)
        completed = status in _TERMINAL_COMPLETED
        incomplete = status in _TERMINAL_INCOMPLETE
        if not completed:
            # 无终端完成凭证的记录：计费权威未知——独立于已上报的
            # token/成本事实计数，绝不反向推断历史 abort/error。
            billing_authority_unknown += 1
        # 逐字段保守读取：非法/缺失字段是 UNKNOWN，不截断不伪造。
        fields = {key: _known_token(usage.get(key)) for key in _NATIVE_TOKEN_FIELDS}
        # reasoning 是 SDK 合法可选细分：缺失只让 reasoning 细分未知，
        # 不影响必备合计字段的 known 判定。
        if fields["reasoning"] is None:
            reasoning_unknown += 1
        mandatory_complete = all(fields[key] is not None for key in _MANDATORY_TOKEN_FIELDS)
        cost = usage.get("cost")
        total_cost = cost.get("total") if isinstance(cost, dict) else None
        cost_valid = (
            total_cost is not None and _validate_cost_value("cost.total", total_cost) is None
        )
        # 零初始化占位：必备 token 全 0 且成本缺失/为 0。没有 completed
        # 终端状态时它是"未知是否发生计费"，绝不能被确认为精确的免费 0。
        zero_placeholder = (
            mandatory_complete
            and all(fields[key] == 0 for key in _MANDATORY_TOKEN_FIELDS)
            and (not cost_valid or float(total_cost) == 0.0)
        )
        if mandatory_complete and not incomplete and (completed or not zero_placeholder):
            known += 1
        else:
            # 部分/非法用量、无 completed 终端状态的零占位、以及
            # error/aborted/pending/deferred（即便带 responseId——身份
            # 不是完成权威）：保留已知字段事实，但完整合计是未知。
            unknown += 1
        # 已知字段事实仍然计入分项聚合（每列只加已确认的值）。
        input_uncached += fields["input"] or 0
        cache_read += fields["cacheRead"] or 0
        cache_write += fields["cacheWrite"] or 0
        output += fields["output"] or 0
        # reasoning 是 output 的子集——永不二次累加。
        reasoning += fields["reasoning"] or 0
        total_tokens += fields["totalTokens"] or 0
        # 成本独立判定：token 字段非法不连坐独立合法的 USD 成本。
        if cost_valid:
            # 已上报成本是独立事实——进入 known subtotal（部分记录的
            # 成本下限，绝不因未知记录被清零）。
            cost_subtotal += float(total_cost)
            cost_reported += 1
            if incomplete:
                # error/aborted/pending/deferred：已上报金额保留为
                # 下限事实，但完成/计费权威未建立——不计入完整估计。
                cost_unknown += 1
            elif float(total_cost) == 0.0 and not completed:
                # 零初始化成本占位（无 completed 终端状态）：未知，
                # 不是已确认免费。
                cost_unknown += 1
            else:
                # USD 估计（provider 原生口径），与 legacy 元账本分开。
                cost_sum += float(total_cost)
                cost_known += 1
        else:
            cost_unknown += 1
    # 个别有限成本之和溢出（如 1e308+1e308）→ 整体 USD 未知，
    # 绝不输出 Infinity/NaN。
    if not math.isfinite(cost_sum):
        cost_known = 0
        cost_unknown = max(cost_unknown, 1)
        cost_sum = 0.0
    return {
        "known_message_count": known,
        "input_uncached": _json_safe_count(input_uncached),
        "cache_read": _json_safe_count(cache_read),
        "cache_write": _json_safe_count(cache_write),
        # PI 归一化 input 已剔除缓存；含缓存合计 = input+cacheRead+cacheWrite。
        "input_including_cache": _json_safe_count(input_uncached + cache_read + cache_write),
        "output": _json_safe_count(output),
        # reasoning 细分在任何记录缺失/非法时为 None（未知），绝不伪造 0。
        "reasoning_subset_output": None if reasoning_unknown else _json_safe_count(reasoning),
        "reasoning_unknown_message_count": reasoning_unknown,
        "total_tokens": _json_safe_count(total_tokens),
        "unknown_message_count": unknown,
        # 缺少终端完成凭证（历史无状态 + error/aborted/pending/deferred）
        # 的记录数——独立于 known/cost 事实，绝不意味着零消费。
        "billing_authority_unknown_message_count": billing_authority_unknown,
        # 已上报 USD 成本下限（含未完成终端记录的部分金额事实）；无已
        # 上报成本或合计溢出时为 None。
        "cost_usd_known_subtotal": (
            round(cost_subtotal, 9)
            if cost_reported and math.isfinite(cost_subtotal)
            else (None if cost_reported else 0.0)
        ),
        # 任一消息成本缺失 → 整体未知（null），绝不显示成 0/免费。
        "cost_usd_estimate": round(cost_sum, 9) if cost_known and not cost_unknown else None,
        "cost_unknown_message_count": cost_unknown,
        "identity_scope": (
            "assistant message_end mirrors only; provider responseId preserved when "
            "present but never treated as completion/billing authority; terminal "
            "stopReason (stop/toolUse/length vs error/aborted/pending/deferred) is "
            "carried via the bounded _rosclaw_terminal envelope; historical mirrors "
            "with empty pi_entry_id have unknown identity and are never deduplicated "
            "by content hash; zero-initialized usage without a completed terminal "
            "status stays unknown (never a confirmed free zero); explicit completed "
            "zero usage remains valid; absent optional SDK reasoning marks the "
            "reasoning breakdown unknown without invalidating mandatory totals; "
            "aggregates beyond the JS exact-integer range are reported as exact "
            "decimal strings (never unsafe JSON numbers); partial/invalid usage "
            "preserves known per-field facts while complete totals stay unknown"
        ),
    }


class UsageRecorder:
    """Writes usage rows on the MissionStore's connection."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def record(self, turn: ModelTurnResultV1) -> str:
        usage_id = new_id("usage")
        usage = turn.usage
        self._conn.execute(
            "INSERT INTO model_usage (usage_id, mission_id, turn_id, provider, model, "
            "profile, prompt_tokens, completion_tokens, reasoning_tokens, total_tokens, "
            "cost_microunits, latency_ms, provider_request_id, context_id, "
            "context_revision, finish_reason, recorded_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                usage_id,
                turn.mission_id or "",
                turn.turn_id,
                turn.provider,
                turn.model,
                turn.profile,
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.reasoning_tokens,
                usage.total_tokens,
                usage.cost_microunits,
                turn.latency_ms,
                turn.provider_request_id,
                turn.context_id,
                turn.context_revision,
                turn.finish_reason,
                datetime.now(UTC).isoformat(),
            ),
        )
        return usage_id

    def mission_totals(self, mission_id: str) -> dict[str, int]:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(prompt_tokens),0) AS pt, "
            "COALESCE(SUM(completion_tokens),0) AS ct, "
            "COALESCE(SUM(total_tokens),0) AS tt, "
            "COALESCE(SUM(cost_microunits),0) AS cost, COUNT(*) AS turns "
            "FROM model_usage WHERE mission_id = ?",
            (mission_id,),
        ).fetchone()
        return {
            "prompt_tokens": int(row["pt"]),
            "completion_tokens": int(row["ct"]),
            "total_tokens": int(row["tt"]),
            "cost_microunits": int(row["cost"]),
            "model_turns": int(row["turns"]),
        }

    def rows(self, mission_id: str) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT * FROM model_usage WHERE mission_id = ? ORDER BY recorded_at",
            (mission_id,),
        )
        return [dict(r) for r in cur.fetchall()]


def estimate_cost_microunits(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    price_input_per_mtok: int,
    price_output_per_mtok: int,
) -> int:
    """Microunits = tokens * per-million price / 1e6 (integer floor)."""
    return (
        prompt_tokens * price_input_per_mtok + completion_tokens * price_output_per_mtok
    ) // 1_000_000
