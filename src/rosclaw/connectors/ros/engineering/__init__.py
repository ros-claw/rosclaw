"""Pure read-only task dossiers; no execution or admission authority."""

from .case_compiler import compile_engineering_case, summarize_case
from .contracts import ConstraintDeltaV1, EvidenceRefV1, RepairIntentV1, RosEngineeringCaseV1

__all__ = [
    "compile_engineering_case",
    "summarize_case",
    "RosEngineeringCaseV1",
    "ConstraintDeltaV1",
    "EvidenceRefV1",
    "RepairIntentV1",
]
