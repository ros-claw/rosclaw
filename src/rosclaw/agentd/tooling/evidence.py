"""Evidence wrapper for tool observations (PR-05, 大纲 §7.1).

Every tool result that reaches the model is wrapped in an EvidenceEnvelope:
timestamp, body, source, evidence class, freshness, and an artifact ref for
oversized payloads. Tool output is *untrusted content* — the envelope text
carries the same ``<untrusted_input>`` markers the ContextCompiler uses, so
an observation can never masquerade as a system fact or an authorization.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from pydantic import Field

from rosclaw.contracts.agent.tool import ToolDescriptorV2
from rosclaw.contracts.common import ContractModel

#: Outputs larger than this are spilled to the artifact store; the model sees
#: the ref + a head excerpt, never a silent truncation.
ARTIFACT_SPILL_BYTES = 8 * 1024

UNTRUSTED_OPEN = '<untrusted_input source="{source}" trust="untrusted">\n'
UNTRUSTED_CLOSE = "\n</untrusted_input>"


class EvidenceEnvelope(ContractModel):
    SCHEMA = "rosclaw.evidence_envelope.v1"

    schema_version: str = "rosclaw.evidence_envelope.v1"
    tool_id: str
    ok: bool
    timestamp: str
    body_id: str
    source: str
    evidence_class: str
    fresh: bool
    artifact_ref: str | None = None
    error: str | None = None
    #: head excerpt when spilled to artifact store
    content: str = Field(default="")
    content_digest: str = ""

    def render_for_model(self) -> str:
        """Text injected into the conversation; untrusted-marked, honest."""
        lines = [
            "[observation — evidence]",
            f"tool: {self.tool_id}",
            f"timestamp: {self.timestamp}",
            f"body_id: {self.body_id}",
            f"source: {self.source}",
            f"evidence_class: {self.evidence_class}",
            f"fresh: {str(self.fresh).lower()}",
        ]
        if self.artifact_ref:
            lines.append(f"artifact_ref: {self.artifact_ref}")
        if not self.ok:
            lines.append(f"error: {self.error or 'unknown'}")
            lines.append("result: (no observation obtained — do not fabricate one)")
        else:
            lines.append("result:")
        inner = "\n".join(lines)
        if self.ok:
            inner += "\n" + UNTRUSTED_OPEN.format(source=self.source) + self.content + UNTRUSTED_CLOSE
        return inner


def wrap_observation(
    descriptor: ToolDescriptorV2,
    output: str,
    *,
    body_id: str,
    artifact_store=None,
    error: str | None = None,
) -> EvidenceEnvelope:
    """Wrap a raw tool output in an evidence envelope (fail-honest on error)."""
    ok = error is None
    digest = hashlib.sha256(output.encode()).hexdigest()[:24]
    artifact_ref: str | None = None
    content = output
    if ok and artifact_store is not None and len(output.encode()) > ARTIFACT_SPILL_BYTES:
        artifact_ref = artifact_store.put(output, prefix="observation")
        content = output[:2000] + f"\n… [完整输出 {len(output)} 字符已落盘 {artifact_ref}]"
    elif ok:
        artifact_ref = f"artifact://observation/sha256:{digest}"
    return EvidenceEnvelope(
        tool_id=descriptor.tool_id,
        ok=ok,
        timestamp=datetime.now(UTC).isoformat(),
        body_id=body_id,
        source=descriptor.source,
        evidence_class=descriptor.evidence_class.value,
        fresh=True,
        artifact_ref=artifact_ref,
        error=error,
        content=content,
        content_digest=digest,
    )
