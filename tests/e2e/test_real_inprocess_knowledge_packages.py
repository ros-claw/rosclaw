from __future__ import annotations

import json

import pytest

pytest.importorskip("rosclaw_know.legacy")
pytest.importorskip("rosclaw_how.v2")

from rosclaw_how.v2 import AdviceEngine  # noqa: E402
from rosclaw_know.legacy import import_legacy_assets  # noqa: E402
from rosclaw_know.store import InMemoryKnowStore  # noqa: E402

from rosclaw.knowledge.context_adapter import build_how_context  # noqa: E402
from rosclaw.knowledge.contracts import HowAdviceRequestV2  # noqa: E402
from rosclaw.knowledge.facade import KnowledgeFacade  # noqa: E402
from rosclaw.knowledge.feedback_adapter import build_usage_feedback  # noqa: E402
from rosclaw.knowledge.service_manager import (  # noqa: E402
    KnowledgeServiceConfig,
    KnowledgeServiceManager,
)


def test_real_inprocess_know_how_reference_loop(tmp_path):
    bridge = tmp_path / "bridge_index.json"
    bridge.write_text(
        json.dumps(
            {
                "schema_version": "v2",
                "symptom_clusters": {
                    "uvc_error_5": {
                        "standard_name": "uvcvideo Region of Interest returned -5",
                        "domain": "Perception",
                        "associated_patterns": [],
                        "cross_domain_analogies": [
                            {
                                "insight": "Kernel and camera controls can be incompatible.",
                                "action_suggestion": "Inspect the pinned compatibility evidence.",
                            }
                        ],
                    }
                },
            },
            sort_keys=True,
        )
    )
    store = InMemoryKnowStore()
    import_legacy_assets(store, bridge_path=bridge)
    manager = KnowledgeServiceManager(
        KnowledgeServiceConfig(mode="inprocess"), inprocess_store=store
    )
    assert manager.startup_error is None
    # Assert the actual optional package engine is active, not a fixture facade.
    assert isinstance(manager.how.engine, AdviceEngine)
    facade = KnowledgeFacade(manager)
    pack = facade.reference_pack(
        query="uvcvideo Region of Interest returned -5",
        context={
            "task": "diagnose RealSense",
            "robot": "limo",
            "current_failure": "uvcvideo Region of Interest returned -5",
        },
    )
    assert pack.items[0].evidence_refs[0].snapshot_id.startswith("snapshot_legacy_")
    context = build_how_context(
        task="diagnose RealSense",
        robot_model="limo",
        current_failure="uvcvideo Region of Interest returned -5",
    )
    advice = facade.advise(
        HowAdviceRequestV2(
            request_id="request_real_inprocess",
            mode="diagnose",
            query="uvcvideo Region of Interest returned -5",
            context=context,
        )
    )
    assert not advice.abstained
    assert advice.reference_pack_id == pack.reference_pack_id
    assert all(item.safety_class == "advisory" for item in advice.recommendations)
    feedback = build_usage_feedback(
        reference_pack_id=pack.reference_pack_id,
        advice_id=advice.advice_id,
        knowledge_unit_id=pack.items[0].knowledge_unit_ids[0],
        context_hash=context.context_hash(),
        verdict="useful",
        receipt_ref="fixture_receipt",
    )
    assert facade.feedback(feedback)
    assert len(store.feedback) == 1
    assert not hasattr(store, "trajectories")
    manager.close()
