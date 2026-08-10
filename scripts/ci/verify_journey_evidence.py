#!/usr/bin/env python3
"""Journey Evidence 独立验证器（六审 §9.3/PR-SIX-6）。

离线验证 sanitized_assertions.json 的全部 ID/hash/状态关系——第三方
只下载 publishable artifact 即可复核，无需相信生成证据的测试自身：

- schema_version 必须是 rosclaw.journey_evidence.v2（且过 JSON Schema）；
- 每条 ActionTxn：approval↔approval 记录、grant↔approval、grant 精确
  消费且未撤销、receipt 事件 receipt_id+action_id 双绑定、receipt
  trust/domain/final_state/usable_for_real 诚实；
- receipt_id != action_id（独立身份）；
- context lease 的 context_hash 非空；
- 事件链包含 approval.requested→decided→grant.consumed→receipt.received
  有序子序列；
- reasoning 禁带字段计数全零；
- verdicts 全 true。

用法: verify_journey_evidence.py <sanitized_assertions.json>
退出码：0=PASS，1=FAIL（逐条打印失败原因）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent / "journey_evidence_v2.schema.json"


def verify(evidence: dict) -> list[str]:
    failures: list[str] = []
    # 0. JSON Schema 结构校验。
    try:
        import jsonschema

        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        jsonschema.validate(evidence, schema)
    except Exception as exc:
        failures.append(f"schema: {exc}")
        return failures  # 结构不符时关系校验无意义
    if evidence.get("schema_version") != "rosclaw.journey_evidence.v2":
        failures.append(f"schema_version != v2: {evidence.get('schema_version')}")

    txns = evidence.get("action_txns", [])
    approvals = {a["request_id"]: a for a in evidence.get("approvals", [])}
    grants = {g["grant_id"]: g for g in evidence.get("grants", [])}
    receipts = evidence.get("receipts", [])
    leases = evidence.get("context_leases", [])
    event_chain = evidence.get("event_chain", [])

    if not txns:
        failures.append("action_txns 为空——旅程必须至少完成一次动作")
    for txn in txns:
        tid = txn.get("txn_id", "?")
        approval = approvals.get(txn.get("approval_id"))
        if approval is None:
            failures.append(f"{tid}: approval {txn.get('approval_id')} 不在 approvals")
        else:
            if approval.get("status") != "APPROVED":
                failures.append(f"{tid}: approval status={approval.get('status')}")
            if approval.get("display_hash") and txn.get("display_hash") and (
                approval["display_hash"] != txn["display_hash"]
            ):
                failures.append(f"{tid}: approval.display_hash != txn.display_hash")
        grant = grants.get(txn.get("grant_id"))
        if grant is None:
            failures.append(f"{tid}: grant {txn.get('grant_id')} 不在 grants")
        else:
            if grant.get("request_id") != txn.get("approval_id"):
                failures.append(f"{tid}: grant 绑定的 request_id 与 approval 不符")
            if grant.get("consumed") != 1 or grant.get("revoked") != 0:
                failures.append(
                    f"{tid}: grant consumed={grant.get('consumed')} revoked={grant.get('revoked')}"
                )
        if txn.get("receipt_id") == txn.get("action_id"):
            failures.append(f"{tid}: receipt_id 与 action_id 同值——非独立身份")
        matched = [
            r for r in receipts
            if r.get("receipt_id") == txn.get("receipt_id")
        ]
        if not matched:
            failures.append(f"{tid}: 无 receipt_id={txn.get('receipt_id')} 的 receipt 事件")
        else:
            receipt = matched[0]
            if receipt.get("action_id") != txn.get("action_id"):
                failures.append(f"{tid}: receipt.action_id != txn.action_id")
            if receipt.get("final_state") != "COMPLETED":
                failures.append(f"{tid}: receipt final_state={receipt.get('final_state')}")
            if receipt.get("trust_level") != "SIMULATED":
                failures.append(f"{tid}: receipt trust_level={receipt.get('trust_level')}")
            if receipt.get("evidence_domain") != "simulation":
                failures.append(f"{tid}: receipt evidence_domain={receipt.get('evidence_domain')}")
            if receipt.get("usable_for_real_execution") is not False:
                failures.append(f"{tid}: usable_for_real_execution 不是 false")
        if txn.get("state") != "COMPLETED":
            failures.append(f"{tid}: txn state={txn.get('state')}")

    for lease in leases:
        if not lease.get("context_hash"):
            failures.append(
                f"lease {lease.get('context_lease_id', '?')}: context_hash 为空"
            )

    # 事件链有序子序列。
    expected = [
        "approval.requested", "approval.decided",
        "grant.consumed", "receipt.received",
    ]
    cursor = 0
    for event_type in event_chain:
        if cursor < len(expected) and event_type == expected[cursor]:
            cursor += 1
    if cursor < len(expected):
        failures.append(
            f"事件链缺环/乱序: 匹配到 {cursor}/{len(expected)}（{event_chain}）"
        )

    for marker, count in evidence.get("reasoning_forbidden_field_counts", {}).items():
        if count != 0:
            failures.append(f"reasoning 禁带字段 {marker} 计数={count}")
    for name, verdict in evidence.get("verdicts", {}).items():
        if verdict is not True:
            failures.append(f"verdict {name}={verdict}")
    return failures


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: verify_journey_evidence.py <sanitized_assertions.json>", file=sys.stderr)
        return 2
    evidence = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    failures = verify(evidence)
    if failures:
        print("FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("PASS: journey evidence 全链独立验证通过")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
