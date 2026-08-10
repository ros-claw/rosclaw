"""PR-SIX-6 红测试（六审 §9）：Evidence V3 + 独立验证器。

红测试先行——当前证据缺口（五审 V2 已确认优点，但）：

1. sanitized_assertions.json 的 receipts 没有 receipt_id、approvals
   没有 display_hash/action_intent_hash——第三方无法独立比较
   txn↔receipt、approval↔txn 的 hash 关系，只能相信 pytest verdict；
2. 没有 rosclaw.journey_evidence.v2 schema，也没有离线验证器——
   `scripts/ci/verify_journey_evidence.py` 必须能独立判定一份证据
   的 ID/hash/状态全链（篡改即 FAIL）；
3. workflow 缺 evidence-verify job（由生成证据的同一个测试自我声明
   不算独立验证）；
4. artifact 打包重复（pytest-0/pytest-current 同文件多份），且完整
   PTY/失败 DB 与脱敏证据混在一起——必须 publishable 与 restricted
   分离（PTY/failure-dump 仅失败时上传、短保留）；
5. manifest 缺 test_conclusion/bundle_digest/evidence_schema_version。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "native-agent-gate.yml"
VERIFIER = REPO / "scripts" / "ci" / "verify_journey_evidence.py"
SCHEMA = REPO / "scripts" / "ci" / "journey_evidence_v2.schema.json"
JOURNEY = REPO / "tests" / "agentd" / "test_product_journey.py"


def _minimal_v2_evidence() -> dict:
    """结构完整、关系自洽的最小 v2 证据（验证器必须接受）。"""
    return {
        "schema_version": "rosclaw.journey_evidence.v2",
        "session_id": "sess_1",
        "mission_id": "mis_1",
        "action_txns": [
            {
                "txn_id": "atxn_1",
                "approval_id": "appr_1",
                "grant_id": "grant_1",
                "action_id": "act_1",
                "receipt_id": "rcpt_1",
                "arguments_hash": "a" * 64,
                "display_hash": "b" * 64,
                "state": "COMPLETED",
                "capability_id": "ur5e.move_to_pose",
            }
        ],
        "approvals": [
            {
                "request_id": "appr_1",
                "status": "APPROVED",
                "decided_by": "operator",
                "display_hash": "b" * 64,
                "action_intent_hash": "c" * 64,
            }
        ],
        "grants": [
            {"grant_id": "grant_1", "request_id": "appr_1",
             "consumed": 1, "revoked": 0}
        ],
        "receipts": [
            {
                "receipt_id": "rcpt_1",
                "action_id": "act_1",
                "final_state": "COMPLETED",
                "trust_level": "SIMULATED",
                "evidence_domain": "simulation",
                "usable_for_real_execution": False,
            }
        ],
        "context_leases": [
            {"context_lease_id": "ctxl_1", "context_hash": "d" * 32,
             "context_revision": 0, "revoked": 0}
        ],
        "event_chain": [
            "approval.requested", "approval.decided",
            "grant.consumed", "receipt.received",
        ],
        "reasoning_forbidden_field_counts": {"reasoning_content": 0},
        "compaction_entry_id": "cmp_1",
        "verdicts": {"chain_ok": True},
        # 七审 PR-SEVEN-7：journey scope 是独立验证的强制面。
        "journey_scope": {
            "journey": "A",
            "install_origin": "release_tarball",
            "config_origin": "generated_no_server_fixtures",
            "robot_kit_digest": "sha256:" + "0" * 64,
            "source_checkout_accessible": False,
        },
    }


def test_evidence_writer_includes_receipt_id_and_hashes() -> None:
    """journey 的 sanitized writer 必须包含 receipt_id 与 approval hash
    独立副本（当前只写 action_id/status——红）。"""
    source = JOURNEY.read_text(encoding="utf-8")
    import re

    writer = re.search(
        r"def _write_sanitized_evidence.*?(?=\n    def )", source, re.DOTALL
    )
    assert writer, "找不到 _write_sanitized_evidence"
    body = writer.group(0)
    receipts_block = re.search(r'evidence\["receipts"\](.*?)(?:evidence\[|\Z)', body, re.DOTALL)
    assert receipts_block and '"receipt_id"' in receipts_block.group(1), (
        "receipts 脱敏记录缺 receipt_id——txn.receipt_id 无法独立比对"
    )
    # 七审 PR-SEVEN-7：display/action intent hash 在 append 的 dict 里
    # （import 已提升模块级）——直接对 writer 全体断言，不再依赖
    # 巧合匹配的局部块。
    assert 'evidence["approvals"]' in body, "writer 缺 approvals 段"
    assert "display_hash" in body
    assert "action_intent_hash" in body
    # schema version 升到 v2
    assert "rosclaw.journey_evidence.v2" in source, "证据 schema 未升 v2"


def test_verifier_exists_and_schema_exists() -> None:
    assert VERIFIER.exists(), "缺 scripts/ci/verify_journey_evidence.py"
    assert SCHEMA.exists(), "缺 journey_evidence_v2.schema.json"


def test_verifier_accepts_valid_evidence(tmp_path: Path) -> None:
    evidence = tmp_path / "sanitized_assertions.json"
    evidence.write_text(json.dumps(_minimal_v2_evidence()), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(VERIFIER), str(evidence)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"合法证据被误判: {result.stderr[-300:]}"


def test_verifier_rejects_tampered_evidence(tmp_path: Path) -> None:
    """篡改任一链环（receipt_id 与 txn 不符）→ 验证器必须 FAIL。"""
    bad = _minimal_v2_evidence()
    bad["receipts"][0]["receipt_id"] = "rcpt_FORGED"
    evidence = tmp_path / "sanitized_assertions.json"
    evidence.write_text(json.dumps(bad), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(VERIFIER), str(evidence)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0, "篡改 receipt_id 竟通过验证"
    # 状态篡改也必须 FAIL。
    bad2 = _minimal_v2_evidence()
    bad2["receipts"][0]["usable_for_real_execution"] = True
    evidence.write_text(json.dumps(bad2), encoding="utf-8")
    result2 = subprocess.run(
        [sys.executable, str(VERIFIER), str(evidence)],
        capture_output=True, text=True, timeout=60,
    )
    assert result2.returncode != 0, "usable_for_real=True 竟通过验证"


def test_workflow_has_independent_evidence_verify_job() -> None:
    jobs = _workflow_jobs()
    assert "evidence-verify" in jobs, (
        "缺 evidence-verify job——证据由生成者自我声明，无独立验证"
    )
    verify = jobs["evidence-verify"]
    assert "journey" in (verify.get("needs") or []), (
        "evidence-verify 必须 needs: journey"
    )
    run_blob = "\n".join(str(s.get("run", "")) for s in verify.get("steps", []))
    assert "verify_journey_evidence.py" in run_blob
    assert "download-artifact" in json.dumps(verify), "verify job 未下载 artifact"


def test_publishable_and_restricted_artifacts_separated() -> None:
    """PTY 全量/失败 DB 属 restricted——仅失败时上传、短保留；
    publishable 证据常驻。"""
    jobs = _workflow_jobs()
    journey = jobs["journey"]
    for step in journey.get("steps", []):
        if not isinstance(step, dict) or "upload-artifact" not in str(step.get("uses", "")):
            continue
        name = str(step.get("with", {}).get("name", ""))
        path = str(step.get("with", {}).get("path", ""))
        restricted = "pty-" in path or "failure-dump" in path
        if restricted:
            assert step.get("if") == "failure()", (
                f"restricted 证据 {name} 未限失败时上传: if={step.get('if')}"
            )
            retention = step.get("with", {}).get("retention-days", 999)
            assert int(retention) <= 14, (
                f"restricted 证据 {name} 保留期过长: {retention}"
            )
    # 去重：publishable 路径不得再用会命中多份副本的通配（pytest-0 +
    # pytest-current 双份）——先 curate 到单一目录再上传。
    journey_run = json.dumps(journey)
    assert "curated" in journey_run or "evidence-pack" in journey_run, (
        "journey 未做 artifact 去重（curated 单份）"
    )


def test_manifest_records_conclusion_and_digests() -> None:
    script = REPO / "scripts" / "ci" / "write_evidence_manifest.sh"
    text = script.read_text(encoding="utf-8")
    for field in ("test_conclusion", "bundle_digest", "evidence_schema_version"):
        assert field in text, f"manifest 缺 {field}"


def _workflow_jobs() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]
