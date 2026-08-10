"""PR-HF5-5 红测试（五审 HOTFIX-5 余项）：Gate Evidence V2 结构断言。

红测试先行——对 native-agent-gate.yml 与 journey artifact 契约的结构
断言，修复前必须全红：

1. main push exact-commit run：workflow 必须有 push: branches: [main]
   （merge commit 不得无 exact-commit Actions 证据）；
2. 每个 job（含 soak）都写+传 evidence manifest，且 manifest 含
   checked_out_sha/head_sha/base_sha/event_name/workflow_hash——
   merge_ref_sha 存 ref 名不是 SHA 的旧缺陷必须闭合；
3. path filters 覆盖安全依赖：contracts/migrations/daemon/limo——
   这些文件单独变化不得绕过 Native Agent Gate；
4. 证据 Gate 的 Actions 必须 pin full commit SHA（不是浮动 major tag）；
5. journey artifact 必须包含脱敏机器可读证据：sanitized_assertions.json
   （全 ID 链 + hash + verdicts）、session structure report、junit.xml、
   installed bundle digest——第三方仅下载 artifact 即可复核
   context→approval→grant→txn→receipt 全链。
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "native-agent-gate.yml"
JOURNEY_TEST = REPO / "tests" / "agentd" / "test_product_journey.py"


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_main_push_exact_commit_run() -> None:
    """merge commit 必须有 exact-commit run：push: branches: [main]。"""
    triggers = _workflow()[True]  # YAML 把 on 解析为 True
    assert "push" in triggers, "workflow 缺 push 触发器——main merge commit 无 exact run"
    push = triggers["push"]
    branches = push.get("branches", []) if isinstance(push, dict) else []
    assert "main" in branches, f"push 触发器不含 main: {push}"


def test_every_job_writes_and_uploads_manifest_v2() -> None:
    """每个 job（含 soak）都经共享脚本写 manifest 并上传；脚本字段含
    checked_out_sha/head_sha/base_sha/event_name/workflow_hash。"""
    script = REPO / "scripts" / "ci" / "write_evidence_manifest.sh"
    assert script.exists(), "缺共享 manifest 脚本 scripts/ci/write_evidence_manifest.sh"
    script_text = script.read_text(encoding="utf-8")
    required = (
        "checked_out_sha",
        "head_sha",
        "base_sha",
        "event_name",
        "workflow_hash",
    )
    for field in required:
        assert field in script_text, (
            f"manifest 脚本缺 {field}（五审 §11.3：merge_ref_sha 存 ref 名不是 "
            "SHA——必须新增 checked_out_sha 等真实字段）"
        )
    # checked_out_sha 必须来自 git rev-parse HEAD（真实 checkout commit）。
    assert "git rev-parse HEAD" in script_text
    jobs = _workflow()["jobs"]
    assert jobs, "workflow 无 jobs"
    for job_id, job in jobs.items():
        if job_id == "evidence-verify":
            continue  # 验证 job 消费证据，不产出 manifest
        steps = job.get("steps", [])
        run_blobs = "\n".join(str(s.get("run", "")) for s in steps)
        uploads = [
            s for s in steps
            if isinstance(s, dict) and "upload-artifact" in str(s.get("uses", ""))
        ]
        assert "write_evidence_manifest.sh" in run_blobs, (
            f"job {job_id} 缺 Write evidence manifest 步骤"
        )
        assert any(
            "evidence-manifest" in str(u.get("with", {}).get("path", ""))
            or "evidence-manifest" in str(u.get("with", {}).get("name", ""))
            for u in uploads
        ), f"job {job_id} 缺 Upload evidence manifest 步骤"


def test_path_filters_cover_security_dependencies() -> None:
    """contracts/migrations/daemon/limo 单独变化不得绕过 Gate。"""
    triggers = _workflow()[True]
    paths = triggers["pull_request"].get("paths", [])
    required = (
        "src/rosclaw/contracts/**",
        "src/rosclaw/storage/migrations/**",
        "src/rosclaw/daemon/**",
        "src/rosclaw/limo/**",
    )
    for pattern in required:
        assert pattern in paths, f"path filters 缺 {pattern}（五审 §11.6）"


def test_actions_pinned_to_full_sha() -> None:
    """证据 Gate 的 actions/* 必须 pin full commit SHA（40 hex），
    不是浮动 major tag（五审 §11.7）。"""
    text = WORKFLOW.read_text(encoding="utf-8")
    uses = re.findall(r"uses:\s*(\S+)", text)
    assert uses, "workflow 无 uses"
    floating = []
    for use in uses:
        if not use.startswith("actions/"):
            continue
        ref = use.rsplit("@", 1)[-1]
        if not re.fullmatch(r"[0-9a-f]{40}", ref):
            floating.append(use)
    assert not floating, f"Actions 未 pin full SHA: {floating}"


def test_journey_artifact_contains_machine_readable_evidence() -> None:
    """journey 测试必须产出脱敏机器可读证据，且 workflow 上传它们。"""
    source = JOURNEY_TEST.read_text(encoding="utf-8")
    # 测试侧：写 sanitized_assertions.json（全链 ID/hash/verdicts）。
    assert "sanitized_assertions.json" in source, (
        "journey 未产出 sanitized_assertions.json（五审 §11.4/§11.8）"
    )
    for key in (
        "approval_id", "grant_id", "txn_id", "action_id", "receipt_id",
        "arguments_hash", "display_hash",
    ):
        assert key in source, f"sanitized assertions 缺字段 {key}"
    # session 结构报告（类型/计数，无正文）。
    assert "session_structure" in source, "缺脱敏 session structure report"
    # bundle digest。
    assert "installed_bundle_digest" in source, "缺 installed bundle digest"
    # workflow 侧：上传这些证据 + junit。
    workflow_text = WORKFLOW.read_text(encoding="utf-8")
    for artifact in ("sanitized_assertions.json", "junit.xml", "installed_bundle_digest"):
        assert artifact in workflow_text, f"workflow 未上传 {artifact}"
