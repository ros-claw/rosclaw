from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rosclaw.knowledge.contracts import (
    EvidenceRefV2,
    HowAdviceBundleV2,
    ReferenceContextV2,
    ReferencePackItemV2,
    ReferencePackV2,
)


def make_reference_pack() -> ReferencePackV2:
    evidence = EvidenceRefV2(
        evidence_id="ev_1",
        source_id="src_1",
        snapshot_id="snap_commit_abc",
        document_id="doc_1",
        path="docs/guide.md",
        start_line=10,
        end_line=20,
        url="https://example.test/repo/blob/abc/docs/guide.md#L10-L20",
        content_hash="a" * 64,
        excerpt="Pinned evidence about a documented control mechanism.",
    )
    return ReferencePackV2(
        reference_pack_id="pack_1",
        query="camera error",
        context=ReferenceContextV2(task="diagnose camera", robot="limo"),
        generated_at=datetime.now(UTC),
        index_version="idx_1",
        items=[
            ReferencePackItemV2(
                rank=1,
                project_id="project_1",
                knowledge_unit_ids=["unit_1"],
                title="Pinned upstream issue",
                why_relevant="Exact error token matched.",
                relevance_dimensions=["exact"],
                mechanism="Driver capability mismatch.",
                what_to_borrow=["Check the documented version gate."],
                exact_files=["docs/guide.md"],
                source_version="commit:abc",
                evidence_refs=[evidence],
                score=1.0,
                score_breakdown={"exact": 1.0},
            )
        ],
        token_budget=8000,
    )


@pytest.fixture
def reference_pack() -> ReferencePackV2:
    return make_reference_pack()


class FakeKnow:
    def __init__(self, pack: ReferencePackV2):
        self.pack = pack
        self.feedback = []

    def health(self):
        return {"status": "ok", "transport": "fixture", "store": "know_only"}

    def research(self, request):
        return {
            "status": "completed",
            "request_id": request.request_id,
            "source_count": 1,
            "snapshot_count": 1,
        }

    def reference_pack(self, **kwargs):
        return self.pack

    def get_reference_pack(self, reference_pack_id):
        return self.pack if reference_pack_id == self.pack.reference_pack_id else None

    def submit_feedback(self, feedback):
        self.feedback.append(feedback)
        return True


class FakeHow:
    def __init__(self, pack: ReferencePackV2, know: FakeKnow):
        self.pack = pack
        self.know = know

    def health(self):
        return {"status": "ok", "transport": "fixture", "advisory_only": True}

    def advise(self, request):
        return HowAdviceBundleV2(
            advice_id="advice_1",
            mode=request.mode,
            context_hash=request.context.context_hash(),
            reference_pack_id=self.pack.reference_pack_id,
            summary="Fixture evidence-backed advice.",
            recommendations=[],
            abstained=False,
            created_at=datetime.now(UTC),
        )

    def submit_feedback(self, feedback):
        return self.know.submit_feedback(feedback)
