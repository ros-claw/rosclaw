"""Opt-in engineering facts in the existing protected dynamic-self layer.

Historical episode summaries belong in benchmark replay, never live Self.
This adapter does not create capabilities, Memory facts or authority.
"""

import json
from dataclasses import replace
from datetime import datetime

from rosclaw.agentd.context.compiler import CompilationError, StaleSourceError
from rosclaw.connectors.ros.engineering import RosEngineeringCaseV1, summarize_case
from rosclaw.contracts.common import content_hash


class EngineeringSelfSource:
    def __init__(self, base, case):
        self.base = base
        self.case = case

    def get_self(self, body_id):
        facts = self.base.get_self(body_id)
        if facts is None:
            return None
        case = RosEngineeringCaseV1.from_dict(self.case.model_dump(mode="json"))
        summary = json.dumps(summarize_case(case), ensure_ascii=False, sort_keys=True)
        return replace(
            facts,
            self_snapshot_hash=content_hash("selfsnap", [facts.self_snapshot_hash, case.case_hash]),
            summary=facts.summary + "\nROS engineering decision card (NOT admission):\n" + summary,
            health="DEGRADED"
            if any(c.status != "SATISFIED" for c in case.constraints)
            else facts.health,
        )


def augment_engineering_sources(sources, model, case, mission, *, now: datetime):
    """Reuse ContextCompiler Body/Self/Mission checks; bind a freshly compiled case.

    Require the *same* review clock as this compilation. Merely checking case age
    is insufficient: a scan at age 400 ms could become stale before the next
    prompt. The host must recompile constraint deltas for each new observation.
    """
    if not isinstance(case, RosEngineeringCaseV1):
        raise CompilationError("engineering source must be a validated RosEngineeringCaseV1")
    try:
        case = RosEngineeringCaseV1.from_dict(case.model_dump(mode="json"))
    except ValueError as error:
        raise CompilationError("engineering case changed after compilation") from error
    if case.evidence_domain != "OBSERVATION":
        raise CompilationError("historical engineering replay cannot enter live Self")
    if case.snapshot_hash != model.snapshot_hash:
        raise CompilationError("engineering case/ROS snapshot mismatch")
    if not case.body_hash or case.body_hash != mission.body_binding.effective_body_hash:
        raise CompilationError("engineering case/mission Body binding mismatch")
    if case.source_status == "MISMATCH":
        raise CompilationError("engineering case source mismatch")
    if datetime.fromisoformat(case.reviewed_at) != now:
        raise StaleSourceError("recompile engineering constraints at the current context clock")
    return replace(sources, self_source=EngineeringSelfSource(sources.self_source, case))
