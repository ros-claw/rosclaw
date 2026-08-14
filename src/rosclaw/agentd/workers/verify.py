"""Deterministic WorkResult verification (总纲 §9.5).

``COMPLETED`` only means the worker *submitted*. Acceptance requires:
schema shape, secret scan on all text artifacts, claim-evidence binding,
and budget sanity. Independent (different-family) review is layered on by
the manager for critical verifications.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rosclaw.contracts.worker.order import WorkOrderV1, WorkResultV1

_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*['\"]?[\w\-]{12,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]


@dataclass(frozen=True)
class VerificationReport:
    accepted: bool
    verifier_results: dict[str, bool] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()


def _scan_secrets(result: WorkResultV1) -> list[str]:
    findings: list[str] = []
    texts: list[str] = [result.summary]
    for artifact in result.artifacts:
        texts.append(artifact.ref)
    for claim in result.claims:
        texts.append(claim.claim)
    for text in texts:
        for pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                findings.append(f"secret-like content matched {pattern.pattern[:32]}…")
    return findings


def verify_result(order: WorkOrderV1, result: WorkResultV1) -> VerificationReport:
    checks: dict[str, bool] = {}
    reasons: list[str] = []

    # 1. Identity binding: result must answer this order, worker, lease.
    bound = (
        result.work_order_id == order.work_order_id
        and result.worker_id == (order.assigned_to or result.worker_id)
        and order.lease is not None
        and result.lease_id == order.lease.lease_id
    )
    checks["identity_binding"] = bound
    if not bound:
        reasons.append("result does not match order/worker/lease")

    # 2. Expected output schema/artifacts.
    expected = set(order.expected_output.artifacts)
    produced = {a.media_type for a in result.artifacts}
    if expected:
        # artifacts are named expectations like "git_patch"/"test_report";
        # match on media_type or ref suffix to stay schema-light in P0.
        names = {a.ref.rsplit("/", 1)[-1].split(":")[0] for a in result.artifacts} | produced
        missing = expected - names
        checks["expected_artifacts"] = not missing
        if missing:
            reasons.append(f"missing expected artifacts: {sorted(missing)}")

    # 3. Secret scan.
    secrets = _scan_secrets(result)
    checks["secret_scan"] = not secrets
    reasons.extend(secrets)

    # 4. Claims must cite evidence.
    unsupported = [c.claim for c in result.claims if not c.evidence_refs]
    checks["claims_have_evidence"] = not unsupported
    if unsupported:
        reasons.append(f"unsupported claims: {unsupported}")

    # 5. Budget sanity: reported usage must not exceed the order envelope.
    over = []
    if (
        order.budgets.model_tokens
        and result.usage.prompt_tokens + result.usage.completion_tokens
        > order.budgets.model_tokens * 2
    ):
        over.append("tokens >> envelope (possible fabricated or runaway usage)")
    # 十三审 HOTFIX-13.2：wall_time_sec 是 soft target（提醒），不是预算
    # 边界——只有显式权威 hard deadline 才把超时当证据异常。
    policy = order.inputs.get("execution_policy") or {}
    hard = policy.get("hard_deadline_sec") if policy.get(
        "hard_deadline_source"
    ) in ("user", "benchmark", "admin_policy") else None
    if hard and result.usage.wall_time_ms > float(hard) * 1000 * 2:
        over.append("wall time >> hard deadline envelope")
    checks["usage_sane"] = not over
    reasons.extend(over)

    # 6. Status consistency: COMPLETED with no artifacts and expected output
    #    is a fabricated success.
    if result.status == "COMPLETED" and order.expected_output.artifacts and not result.artifacts:
        checks["not_fabricated"] = False
        reasons.append("COMPLETED with no artifacts though artifacts were expected")
    else:
        checks.setdefault("not_fabricated", True)

    return VerificationReport(
        accepted=all(checks.values()),
        verifier_results=checks,
        reasons=tuple(reasons),
    )
