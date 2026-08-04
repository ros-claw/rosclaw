"""证据等级命名（审计 §1.2）：每个功能只允许宣称其最高证据等级。"""

from __future__ import annotations

from enum import StrEnum


class EvidenceLevel(StrEnum):
    E0_SPECIFIED = "E0_SPECIFIED"
    E1_COMPONENT_VERIFIED = "E1_COMPONENT_VERIFIED"
    E2_PROCESS_VERIFIED = "E2_PROCESS_VERIFIED"
    E3_SIM_VERIFIED = "E3_SIM_VERIFIED"
    E4_SHADOW_VERIFIED = "E4_SHADOW_VERIFIED"
    E5_REAL_DEVELOPER_OBSERVED = "E5_REAL_DEVELOPER_OBSERVED"
    E6_REAL_INDEPENDENTLY_VERIFIED = "E6_REAL_INDEPENDENTLY_VERIFIED"
    E7_OPERATIONALLY_QUALIFIED = "E7_OPERATIONALLY_QUALIFIED"


RANK: dict[EvidenceLevel, int] = {level: i for i, level in enumerate(EvidenceLevel)}


def at_most(level: EvidenceLevel, ceiling: EvidenceLevel) -> bool:
    return RANK[level] <= RANK[ceiling]
