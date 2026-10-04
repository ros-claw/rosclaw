"""ROS context projection into the existing KNOW/HOW v2 facade.

No graph database or recovery engine is owned here. Reference packs and advice
retain their native provenance, stale warnings and advisory-only boundary.
"""

from datetime import UTC, datetime

from rosclaw.connectors.ros.diagnosis import diagnose
from rosclaw.connectors.ros.intelligence import RosSystemModel
from rosclaw.contracts.common import content_hash
from rosclaw.knowledge.contracts import HowAdviceRequestV2, ReferenceContextV2


class RosKnowledgeAdvisor:
    def __init__(self, facade):
        self.facade = facade

    def _context(self, model, task, now):
        if not -100 <= model.age_ms(now) <= 5000:
            raise ValueError("fresh ROS observations are required for knowledge consultation")
        diagnosis = diagnose(model, now=now)
        issues = sorted({issue["issue_code"] for issue in diagnosis["issues"]})
        return ReferenceContextV2(
            task=task,
            robot=model.robot_id,
            ros_distro=model.environment.get("distro"),
            simulator=model.environment.get("simulator"),
            current_stage="ROS readiness",
            current_failure=",".join(issues) or None,
        )

    def reference(self, model: RosSystemModel, task: str, *, now=None):
        now = now or datetime.now(UTC)
        context = self._context(model, task, now)
        return self.facade.reference_pack(query=task, context=context, top_k=8, token_budget=4000)

    def advise(self, model: RosSystemModel, task: str, *, now=None):
        now = now or datetime.now(UTC)
        context = self._context(model, task, now)
        request = HowAdviceRequestV2.model_validate(
            {
                "request_id": content_hash("rosadvice", [model.snapshot_id, task]),
                "mode": "diagnose" if context.current_failure else "consult",
                "query": task,
                "context": {
                    "body": {
                        "robot_model": model.body.get("robot_model"),
                        "robot_type": model.body.get("robot_type"),
                    },
                    "software": {"ros_distro": context.ros_distro, "simulator": context.simulator},
                    "runtime": {
                        "task": task,
                        "current_stage": context.current_stage,
                        "current_failure": context.current_failure,
                        "verifier_signals": ["snapshot:" + model.snapshot_hash],
                    },
                },
                "top_k": 8,
                "token_budget": 4000,
            }
        )
        return self.facade.advise(request)
