"""ExactActionV1（五审 P0-5C）：不可变精确动作合约。

Operator 批准的对象必须就是 executor 接收的对象——canonical bytes/
hash 逐字节一致。不再是"title 里碰巧有 capability 名"。

不变量：
- capability_id/version/source 一等字段（SIM/REAL 同一合约）；
- normalized_arguments 在建卡前展开 MCP/contract 默认值——人批准
  660Hz/0.25s/18%，不是 `{}`；
- arguments_hash 与 action_intent_hash 由 normalized 参数与全部
  关键上下文计算；
- display_hash 绑定 capability/mission/mode/context/revision/
  normalized arguments/risk/expected_effect/TTL——任何一项变化
  都使 hash 不同，卡片与执行精确互锁。
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import Field

from rosclaw.contracts.common import ContractModel, ValidationError
from rosclaw.contracts.pi.canonical import canonical_dumps


class ExactActionV1(ContractModel):
    SCHEMA = "rosclaw.exact_action.v1"

    schema_version: Literal["rosclaw.exact_action.v1"] = "rosclaw.exact_action.v1"
    capability_id: str = Field(min_length=1)
    capability_version: str = "1.0.0"
    capability_source: str = ""
    normalized_arguments: dict[str, Any]
    arguments_hash: str
    authoritative_risk_tier: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    side_effect_class: str = "IRREVERSIBLE"
    mission_id: str
    mode: Literal["SIMULATION", "SHADOW", "REAL"]
    body_id: str
    body_hash: str
    context_revision: int
    context_hash: str
    expected_effect: str = ""
    failure_handling: str = ""
    verification_plan: list[str] = Field(default_factory=list)
    # 六审 §6.2.5：执行通道身份（按 body+capability source 解析）——
    # execute 按它路由 SIM executor，不用全局通道。
    executor_identity: str = ""
    created_at: str
    expires_at: str
    action_intent_hash: str

    @property
    def title(self) -> str:
        """展示标题由合约派生——capability 永远是标题的一部分，
        不允许"危险 capability + 无害 title"分离（五审场景 D）。"""
        return f"{self.capability_id}({self.arguments_hash[:8]})"


def compute_arguments_hash(normalized_arguments: dict[str, Any]) -> str:
    """normalized arguments 的内容 hash（canonical，与 TS 同源）。"""
    return hashlib.sha256(canonical_dumps(normalized_arguments).encode()).hexdigest()


def compute_action_intent_hash(action: ExactActionV1) -> str:
    """完整动作意图 hash——capability/mission/mode/body/context/args/
    risk/effect 全部绑定。任何一项变化 → hash 不同。"""
    payload = action.model_dump(mode="json")
    payload.pop("action_intent_hash", None)
    payload.pop("schema_version", None)
    return hashlib.sha256(canonical_dumps(payload).encode()).hexdigest()


def build_exact_action(
    *,
    capability_id: str,
    capability_version: str = "1.0.0",
    capability_source: str = "",
    normalized_arguments: dict[str, Any],
    authoritative_risk_tier: str,
    side_effect_class: str = "IRREVERSIBLE",
    mission_id: str,
    mode: str,
    body_id: str,
    body_hash: str,
    context_revision: int,
    context_hash: str,
    expected_effect: str = "",
    failure_handling: str = "",
    verification_plan: list[str] | None = None,
    created_at: str,
    expires_at: str,
    executor_identity: str = "",
) -> ExactActionV1:
    """构造 + 计算两个 hash（缺 capability/args/mission 即 fail closed）。"""
    if not capability_id.strip():
        raise ValidationError("ExactActionV1 requires non-empty capability_id")
    action = ExactActionV1(
        capability_id=capability_id,
        capability_version=capability_version,
        capability_source=capability_source,
        normalized_arguments=normalized_arguments,
        arguments_hash=compute_arguments_hash(normalized_arguments),
        authoritative_risk_tier=authoritative_risk_tier,  # type: ignore[arg-type]
        side_effect_class=side_effect_class,
        mission_id=mission_id,
        mode=mode,  # type: ignore[arg-type]
        body_id=body_id,
        body_hash=body_hash,
        context_revision=context_revision,
        context_hash=context_hash,
        expected_effect=expected_effect,
        failure_handling=failure_handling,
        verification_plan=verification_plan or [],
        executor_identity=executor_identity,
        created_at=created_at,
        expires_at=expires_at,
        action_intent_hash="",
    )
    return action.model_copy(
        update={"action_intent_hash": compute_action_intent_hash(action)}
    )
