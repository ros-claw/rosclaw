"""Shared L5 Memory renderer; historical advice never becomes authority."""

from rosclaw.agentd.context.sources import EVIDENCE_RANK, EvidenceClass

ADMITTED = {EvidenceClass.MEASURED, EvidenceClass.VERIFIED_RECEIPT, EvidenceClass.CURATED}


def compile_memory_layer(items, *, body_id):
    admitted = [
        item
        for item in items
        if item.evidence_class in ADMITTED and item.body_scope in (None, body_id)
    ]
    admitted.sort(key=lambda item: EVIDENCE_RANK[item.evidence_class])
    summary = "\n".join(
        f"- [{item.evidence_class.value}] {item.summary} ({item.ref})" for item in admitted
    )
    return admitted, summary, len(items) - len(admitted)
