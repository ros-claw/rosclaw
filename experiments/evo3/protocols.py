"""EVO-3 experiment protocol definitions (数据库优化v4 §11).

Four experiments.  Arms:

    A — No Memory (baseline)
    B — Fixed Cooldown (static recovery, no regime awareness)
    C — Regime-aware Memory (selective pipeline: ABSTAIN unless gates pass)

Safety invariants (v4 §17.12): never exceed verified temperature/current/
protection thresholds; experiments trigger on NATURAL degradation, never
force the hardware outside its envelope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Arm(StrEnum):
    A_NO_MEMORY = "A_no_memory"
    B_FIXED_COOLDOWN = "B_fixed_cooldown"
    C_REGIME_AWARE = "C_regime_aware"


@dataclass(frozen=True)
class ExperimentProtocol:
    experiment_id: str
    title: str
    arms: tuple[Arm, ...]
    sessions_per_arm: int
    rounds_per_session: int
    across_days: bool
    hand_balance: bool
    randomize_order: bool
    trigger: dict[str, Any] = field(default_factory=dict)
    safety_limits: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# v4 §11.1 实验一：健康工况拒绝干预
# ---------------------------------------------------------------------------

EXP1_HEALTHY_ABSTAIN = ExperimentProtocol(
    experiment_id="evo3_exp1_healthy_abstain",
    title="健康工况下 Regime-aware Memory 应选择 ABSTAIN",
    arms=(Arm.A_NO_MEMORY, Arm.B_FIXED_COOLDOWN, Arm.C_REGIME_AWARE),
    sessions_per_arm=12,
    rounds_per_session=100,
    across_days=True,
    hand_balance=True,
    randomize_order=True,
    trigger={"kind": "scheduled", "note": "short cool sessions, 45–50 °C"},
    safety_limits={
        "max_temperature_c": 56.0,
        "max_current_ma": 1200,
        "abort_on_protection_event": True,
    },
    expected={
        "C": {
            "intervention_coverage": "very_low",
            "abstain_correctness": "high",
            "invalid_rate_vs_A": "not_higher",
            "memory_hurt": "≈0",
        }
    },
)

# ---------------------------------------------------------------------------
# v4 §11.2 实验二：匹配热退化工况
# ---------------------------------------------------------------------------

EXP2_MATCHED_DEGRADATION = ExperimentProtocol(
    experiment_id="evo3_exp2_matched_degradation",
    title="只在进入历史记忆对应工况后允许干预",
    arms=(Arm.A_NO_MEMORY, Arm.B_FIXED_COOLDOWN, Arm.C_REGIME_AWARE),
    sessions_per_arm=12,
    rounds_per_session=100,
    across_days=True,
    hand_balance=True,
    randomize_order=True,
    trigger={
        "kind": "regime_crossover",
        "conditions": [
            "temperature_slope_c_per_min >= 0.15",
            "position_error_p95 >= 15.0",
            "recent_invalid_rate >= 0.06",
        ],
        "conditions_required": 2,
        "note": "自然长时运行，达到阈值后随机进入策略；严禁主动超过安全界限",
    },
    safety_limits={
        "max_temperature_c": 58.0,
        "max_current_ma": 1200,
        "abort_on_protection_event": True,
        "cooldown_on_trigger": True,
    },
    expected={"C": {"recovery_sr_vs_A": "higher", "memory_hurt": "≈0"}},
)

# ---------------------------------------------------------------------------
# v4 §11.3 实验三：反工况测试（本轮最重要的负测试，离线可证）
# ---------------------------------------------------------------------------

EXP3_COUNTER_REGIME = ExperimentProtocol(
    experiment_id="evo3_exp3_counter_regime",
    title="热退化记忆放入健康工况 → Retrieve=true, Applicable=false, ABSTAIN",
    arms=(Arm.C_REGIME_AWARE,),
    sessions_per_arm=0,  # offline replay, no hardware
    rounds_per_session=0,
    across_days=False,
    hand_balance=False,
    randomize_order=False,
    trigger={"kind": "offline_replay"},
    safety_limits={},
    expected={"retrieve": True, "applicable": False, "decision": "ABSTAIN"},
)

# ---------------------------------------------------------------------------
# v4 §11.4 实验四：控制编舞保护（离线可证）
# ---------------------------------------------------------------------------

EXP4_CHOREOGRAPHY_PROTECTION = ExperimentProtocol(
    experiment_id="evo3_exp4_choreography_protection",
    title="run1 有害 Patch 重送 ChoreographyValidator → 100% BLOCK, 0 真机执行",
    arms=(Arm.C_REGIME_AWARE,),
    sessions_per_arm=0,
    rounds_per_session=0,
    across_days=False,
    hand_balance=False,
    randomize_order=False,
    trigger={"kind": "offline_replay"},
    safety_limits={},
    expected={"block_rate": 1.0, "real_executions": 0},
)

PROTOCOLS = {
    p.experiment_id: p
    for p in (
        EXP1_HEALTHY_ABSTAIN,
        EXP2_MATCHED_DEGRADATION,
        EXP3_COUNTER_REGIME,
        EXP4_CHOREOGRAPHY_PROTECTION,
    )
}


def crossover_triggered(
    *,
    temperature_slope: float | None,
    position_error_p95: float | None,
    recent_invalid_rate: float | None,
    protocol: ExperimentProtocol = EXP2_MATCHED_DEGRADATION,
) -> bool:
    """Regime-crossover trigger (v4 §11.2): at least N of the conditions
    must hold — and missing evidence never counts as holding."""
    conditions = protocol.trigger.get("conditions", [])
    required = int(protocol.trigger.get("conditions_required", len(conditions)))
    held = 0
    if temperature_slope is not None and temperature_slope >= 0.15:
        held += 1
    if position_error_p95 is not None and position_error_p95 >= 15.0:
        held += 1
    if recent_invalid_rate is not None and recent_invalid_rate >= 0.06:
        held += 1
    return held >= required
