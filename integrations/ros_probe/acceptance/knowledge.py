"""Live KNOW/HOW integration using the existing durable embedded service."""

import argparse
import asyncio
import json
from pathlib import Path

from rosclaw_know.contracts import ResearchRequestV2
from rosclaw_know.sources import GitHubAdapter, ResearchOrchestrator

from rosclaw.connectors.ros.expert import inspect_system
from rosclaw.connectors.ros.know.advisor import RosKnowledgeAdvisor
from rosclaw.core.event_bus import EventBus
from rosclaw.knowledge.facade import KnowledgeFacade
from rosclaw.knowledge.service_manager import KnowledgeServiceConfig, KnowledgeServiceManager

PIN = "65a6598c3587cb947978227c01af421e18576f0a"


class PinnedCoverageSource(GitHubAdapter):
    def __init__(self, store):
        super().__init__(max_documents=6, max_issue_documents=0)
        self.store = store

    async def discover(self, request):
        candidates = await super().discover(request)
        return [
            candidate.model_copy(update={"snapshot_ref": PIN})
            for candidate in candidates
            if candidate.source.repository == "open-navigation/opennav_coverage"
        ]

    async def snapshot(self, candidate):
        snapshot = await super().snapshot(candidate)
        existing = self.store.get_snapshot(snapshot.snapshot_id)
        if existing is not None:
            if (
                existing.content_hash != snapshot.content_hash
                or existing.version_value != snapshot.version_value
            ):
                raise ValueError("pinned source snapshot content/version mismatch")
            # Re-fetching the same immutable commit retains its original
            # capture record rather than replacing fetched_at under one ID.
            return existing
        return snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--endpoint", default="ws://127.0.0.1:19090")
    args = parser.parse_args()
    root = args.directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manager = KnowledgeServiceManager(
        KnowledgeServiceConfig(
            mode="inprocess", know_store_mode="embedded", know_store_path=str(root / "seekdb")
        )
    )
    try:
        if manager.startup_error:
            raise RuntimeError(manager.startup_error)
        bus = EventBus()
        events = []
        for topic in ["know.reference_pack.created", "how.advice.created", "how.advice.abstained"]:
            bus.subscribe(
                topic, lambda event: events.append({"topic": event.topic, "payload": event.payload})
            )
        facade = KnowledgeFacade(manager, event_bus=bus)
        adapter = PinnedCoverageSource(manager.know.store)
        research = asyncio.run(
            ResearchOrchestrator(manager.know.store, {"github": adapter}, run_timeout=60).run(
                ResearchRequestV2(
                    request_id="ros-coverage-official-source",
                    topic="repo:open-navigation/opennav_coverage",
                    goal="Reuse versioned official coverage navigation components.",
                    max_sources=1,
                    token_budget=50000,
                    source_types=["repository"],
                )
            )
        )
        (root / "research.json").write_text(research.model_dump_json(indent=2) + "\n")
        if research.snapshots != 1 or research.knowledge_units < 1:
            raise RuntimeError("official source was not indexed with provenance")
        model = inspect_system(endpoint=args.endpoint, robot_id="ros_expert_base", deep=True)
        advisor = RosKnowledgeAdvisor(facade)
        query = "Nav2 opennav coverage complete area cleaning"
        pack = advisor.reference(model, query)
        advice = advisor.advise(model, query)
        if not pack.items:
            raise RuntimeError("source-backed reference pack has no evidence items")
        for name, value in [("research", research), ("reference_pack", pack), ("advice", advice)]:
            (root / (name + ".json")).write_text(value.model_dump_json(indent=2) + "\n")
        (root / "events.json").write_text(json.dumps(events, indent=2) + "\n")
        (root / "health.json").write_text(json.dumps(manager.health(), indent=2) + "\n")
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "store": "seekdb_embedded",
                    "reference_items": len(pack.items),
                    "advice_abstained": advice.abstained,
                    "action_authority": False,
                    "automatic_rule_promotion": False,
                }
            )
        )
    finally:
        manager.close()


if __name__ == "__main__":
    main()
