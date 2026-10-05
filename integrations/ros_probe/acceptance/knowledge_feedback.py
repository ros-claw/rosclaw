"""Durable KNOW/HOW governance acceptance linked to an actual SIM episode.

The accepted Native episode ran with knowledge disabled. This explicitly records
unknown contribution and never claims the advice was used or beneficial.
Run write and read phases in separate processes against an owned SeekDB copy.
"""

import argparse
import json
from pathlib import Path

from rosclaw.core.event_bus import EventBus
from rosclaw.knowledge.contracts import KnowledgeUsageFeedbackV1
from rosclaw.knowledge.facade import KnowledgeFacade
from rosclaw.knowledge.feedback_adapter import build_usage_feedback
from rosclaw.knowledge.service_manager import KnowledgeServiceConfig, KnowledgeServiceManager


def run(args):
    manager = KnowledgeServiceManager(
        KnowledgeServiceConfig(
            mode="inprocess", know_store_mode="embedded", know_store_path=str(args.store)
        )
    )
    bus = EventBus()
    events = []
    bus.subscribe("how.feedback.recorded", lambda event: events.append(event.payload))
    facade = KnowledgeFacade(manager, event_bus=bus)
    try:
        if manager.startup_error:
            raise RuntimeError(manager.startup_error)
        advice = json.loads(args.advice.read_text())
        pack = facade.get_reference_pack(advice["reference_pack_id"])
        if pack is None or not pack.items:
            raise RuntimeError("persisted source-backed reference pack required")
        units = {unit for item in pack.items for unit in item.knowledge_unit_ids}
        unit = advice["recommendations"][0]["knowledge_unit_ids"][0]
        if unit not in units:
            raise RuntimeError("advice must cite the persisted reference pack")
        store = manager.know.store
        if args.phase == "write":
            verification = json.loads(args.verification.read_text())
            if verification["verification_status"] != "PASS" or not verification["success"]:
                raise ValueError("actual accepted verifier required")
            if not args.practice.is_file():
                raise ValueError("actual Practice episode required")
            feedback = build_usage_feedback(
                reference_pack_id=pack.reference_pack_id,
                advice_id=advice["advice_id"],
                knowledge_unit_id=unit,
                context_hash=advice["context_hash"],
                verdict="unknown",
                used_by_agent=False,
                origin="verifier",
                reason="Verified SIM episode; knowledge was disabled during execution. Advice contribution is unmeasured.",
                receipt_ref=str(args.verification),
                practice_ref=str(args.practice),
            )
            if facade.feedback(feedback, via_how=True) is not True:
                raise RuntimeError("first feedback was not created")
            if facade.feedback(feedback, via_how=True) is not False:
                raise RuntimeError("identical feedback was not idempotent")
            conflicting = feedback.model_copy(update={"reason": "conflicting immutable payload"})
            try:
                facade.feedback(conflicting, via_how=True)
            except ValueError as exc:
                if "ID conflict" not in str(exc):
                    raise
            else:
                raise RuntimeError("conflicting feedback ID was accepted")
            args.feedback.write_text(feedback.model_dump_json(indent=2) + "\n")
        else:
            feedback = KnowledgeUsageFeedbackV1.model_validate_json(args.feedback.read_text())
            if facade.feedback(feedback, via_how=True) is not False:
                raise RuntimeError("feedback did not survive process restart")
        records = [
            r for r in store.list_feedback_governance() if r.feedback_id == feedback.feedback_id
        ]
        if len(records) != 1:
            raise RuntimeError("exactly one persisted governance record required")
        record = records[0]
        if (
            record.queue != "manual_review"
            or record.status != "pending_review"
            or not record.requires_human_review
            or record.automatic_mutation_allowed
        ):
            raise RuntimeError("unknown contribution must not mutate knowledge automatically")
        result = {
            "status": "PASS",
            "phase": args.phase,
            "evidence_domain": "DURABLE_FEEDBACK_LINKAGE",
            "advice_contribution_verified": False,
            "reference_pack_id": pack.reference_pack_id,
            "feedback": feedback.model_dump(mode="json"),
            "governance": record.model_dump(mode="json"),
            "events": events,
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"status": "PASS", "phase": args.phase}))
    finally:
        manager.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["write", "read"])
    for key in ("store", "advice", "verification", "practice", "feedback", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    run(parser.parse_args())
