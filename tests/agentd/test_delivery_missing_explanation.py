"""Delivery UX metadata tests; dummy MP4 bytes are never decoded video proof."""

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from rosclaw.contracts.pi.tool_request import PiToolRequestV1, PiToolResultV1
from rosclaw.storage.migrations import MigrationRunner
from rosclaw.task_kernel.coordinator import TaskCoordinator
from rosclaw.task_kernel.service import TaskKernel


@pytest.fixture
def env(tmp_path):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    MigrationRunner().apply(conn, "sqlite")
    kernel = TaskKernel(conn, tmp_path)
    kernel.persist_input(
        mission_id="delivery",
        session_ref="session",
        message_id="input",
        text="生成交付物",
    )
    bound = kernel.ensure_task_for_effect(
        mission_id="delivery",
        session_ref="session",
        backend_native_id="session",
        cwd=str(tmp_path),
    )
    dispatcher = PiToolDispatcher(
        SimpleNamespace(
            _task_kernel=kernel,
            _store=SimpleNamespace(connection=conn),
        )
    )
    try:
        yield kernel, conn, dispatcher, str(bound["task_id"]), tmp_path
    finally:
        conn.close()


def request(**arguments):
    return PiToolRequestV1(
        request_id="delivery_request",
        pi_session_id="session",
        mission_id="delivery",
        tool_name="rosclaw_deliver",
        requested_at="2026-10-18T00:00:00Z",
        idempotency_key="delivery_idem",
        arguments=arguments,
    )


def require_scene(env):
    kernel, conn, _, task_id, _ = env
    revision = int(kernel.get_task(task_id)["active_revision"])
    spec = json.loads(
        conn.execute(
            "SELECT task_spec_json FROM task_revisions WHERE task_id=? AND revision=?",
            (task_id, revision),
        ).fetchone()[0]
    )
    spec["deliverables"] = [
        {"kind": "scene_video", "media_type": "video/mp4", "required": True},
    ]
    conn.execute(
        "UPDATE task_revisions SET task_spec_json=? WHERE task_id=? AND revision=?",
        (json.dumps(spec), task_id, revision),
    )


def observe(env):
    result = PiToolResultV1(
        request_id="delivery_request",
        ok=True,
        status="REGISTERED",
        summary="文件已登记",
        artifact_refs=["art_original"],
    )
    env[2]._coordinator_consider(request(), result)
    assert result.ok and result.status == "REGISTERED"
    assert result.artifact_refs == ["art_original"]
    return result


def ledger(env):
    return [
        dict(row)
        for row in env[1].execute(
            "SELECT * FROM artifacts WHERE task_id=? ORDER BY artifact_id", (env[3],)
        )
    ]


def test_real_generic_mp4_is_accessible_but_required_scene_missing(env):
    kernel, conn, dispatcher, task_id, root = env
    require_scene(env)
    payload = b"DUMMY METADATA FIXTURE; NOT DECODED VIDEO"
    path = root / "clip.mp4"
    path.write_bytes(payload)
    # Real delivery routing, registration, coordinator and original verifier.
    result = asyncio.run(
        dispatcher._dispatch(
            request(
                path=str(path),
                cwd=str(root),
                media_type="video/mp4",
                role="video",
            )
        )
    )
    assert result.ok and result.status == "REGISTERED"
    original_refs = list(result.artifact_refs)
    rows = ledger(env)
    assert len(rows) == 1
    row = rows[0]
    assert row["producer"] == "model:rosclaw_artifact_register"
    assert json.loads(row["metadata_json"])["role"] == "video"
    assert not json.loads(row["metadata_json"]).get("lineage")
    assert row["sha256"].removeprefix("sha256:") == hashlib.sha256(payload).hexdigest()
    assert Path(row["path"]).read_bytes() == payload
    outcome = TaskCoordinator(kernel).consider(task_id)
    assert (outcome["lifecycle"], outcome["verification"], outcome["delivery"]) == (
        "ACTIVE",
        "PARTIAL",
        "MISSING",
    )
    assert outcome["verification_scope"] == "declared_deliverables"
    assert outcome["task_semantic_verification"] == "CONFIGURED_CHECKS_ONLY"
    ref = outcome["artifact_refs"][0]
    assert ref["kind"] == "data"
    assert ref["artifact_id"] == row["artifact_id"]
    assert Path(ref["path"]).read_bytes() == payload
    for text in (
        "lifecycle=ACTIVE",
        "verification=PARTIAL",
        "delivery=MISSING",
        "DELIVERABLE_MISSING",
        "required kind=scene_video",
        "已登记 Ref",
        ref["open_command"],
        "文件登记不等于",
        "验收未过；未确认目标达成",
    ):
        assert text in result.summary
    assert str(root) not in result.summary
    before_state = kernel.get_task(task_id)["state"]
    again = observe(env)
    assert "required kind=scene_video" in again.summary
    assert observe(env).summary == again.summary
    assert kernel.get_task(task_id)["state"] == before_state == "RUNNING"
    assert ledger(env) == rows
    assert result.artifact_refs == original_refs
    assert conn.execute("SELECT COUNT(*) FROM task_outcomes").fetchone()[0] == 0


def test_real_mixed_failure_keeps_configured_failure(env):
    kernel, _, _, task_id, root = env
    require_scene(env)
    kernel.set_acceptance(task_id, {"required_files": ["absent.txt"]})
    path = root / "clip.mp4"
    path.write_bytes(b"dummy")
    kernel.register_artifact(
        task_id=task_id,
        path=str(path),
        media_type="video/mp4",
        producer="model:rosclaw_artifact_register",
        metadata={"role": "video"},
    )
    before = ledger(env)
    outcome = TaskCoordinator(kernel).consider(task_id)
    failures = outcome["repair_directive"]["failures"]
    assert any(f.startswith("DELIVERABLE_MISSING:") for f in failures)
    others = [f for f in failures if not f.startswith("DELIVERABLE_MISSING:")]
    assert others
    result = observe(env)
    assert "verification=FAIL" in result.summary
    assert "DELIVERABLE_MISSING" in result.summary
    assert others == ["缺交付文件 absent.txt"]
    assert "未分类失败诊断=1 项（原因未确认；原文不展示）" in result.summary
    for failure in others:
        assert failure not in result.summary
    assert "absent.txt" not in result.summary
    assert "required kind=scene_video" in result.summary
    assert "verification_scope=configured_acceptance_and_deliverables" in result.summary
    assert "验收未过；未确认目标达成" in result.summary
    assert outcome["verification"] == "FAIL"
    assert outcome["lifecycle"] == "ACTIVE"
    assert kernel.get_task(task_id)["state"] != "SUCCEEDED"
    for ref in outcome["artifact_refs"]:
        assert ref["open_command"] in result.summary
        assert Path(ref["path"]).read_bytes() == b"dummy"
    assert observe(env).summary == result.summary
    assert "task_semantic_verification=CONFIGURED_CHECKS_ONLY" in result.summary
    assert ledger(env) == before
    assert kernel.get_acceptance_spec(task_id) is not None


@pytest.mark.parametrize("role", ["diagnostic_report", "progress_report"])
def test_real_intermediate_is_not_finish_signal(env, role):
    kernel, conn, dispatcher, task_id, root = env
    require_scene(env)
    path = root / "evidence.mp4"
    path.write_bytes(b"intermediate metadata fixture")
    result = asyncio.run(
        dispatcher._dispatch(
            request(
                path=str(path),
                media_type="video/mp4",
                role=role,
            )
        )
    )
    assert result.ok
    assert "任务验收" not in result.summary
    assert "DELIVERABLE_MISSING" not in result.summary
    assert kernel.get_task(task_id)["state"] == "RUNNING"
    assert conn.execute("SELECT COUNT(*) FROM verifications").fetchone()[0] == 0
    assert len(ledger(env)) == 1


@pytest.mark.parametrize("legacy", [False, True])
def test_real_no_failure_and_legacy_unknown_do_not_invent_missing(env, legacy):
    kernel, conn, _, task_id, root = env
    path = root / "report.txt"
    path.write_text("software fixture", encoding="utf-8")
    kernel.register_artifact(task_id=task_id, path=str(path), media_type="text/plain")
    outcome = TaskCoordinator(kernel).consider(task_id)
    if legacy:
        outcome.pop("verification_scope", None)
        outcome.pop("task_semantic_verification", None)
        conn.execute(
            "UPDATE task_outcomes SET outcome_json=? WHERE task_id=?",
            (json.dumps(outcome), task_id),
        )
    before = ledger(env)
    result = observe(env)
    assert "DELIVERABLE_MISSING" not in result.summary
    assert "当前验收缺项/失败" not in result.summary
    assert "verification=PASS" in result.summary
    assert (
        "verification_scope=UNKNOWN" if legacy else "verification_scope=artifact_integrity_only"
    ) in result.summary
    assert ("未确认目标达成" if legacy else "任务语义未验证") in result.summary
    assert observe(env).summary == result.summary
    assert ledger(env) == before


def test_real_stale_revision_ref_is_not_presented(env):
    kernel, _, _, task_id, root = env
    path = root / "old.txt"
    path.write_text("historical fixture", encoding="utf-8")
    old = kernel.register_artifact(task_id=task_id, path=str(path), media_type="text/plain")
    TaskCoordinator(kernel).consider(task_id)
    kernel.bind_message(
        mission_id="delivery",
        session_ref="session",
        backend_native_id="session",
        message_id="correction",
        text="阶段产物，目标尚未达成，请继续",
        cwd=str(root),
    )
    require_scene(env)
    path = root / "new.mp4"
    path.write_bytes(b"new metadata fixture")
    new = kernel.register_artifact(
        task_id=task_id,
        path=str(path),
        media_type="video/mp4",
        metadata={"role": "video"},
    )
    before = ledger(env)
    result = observe(env)
    assert old["artifact_id"] not in result.summary
    assert new["artifact_id"] in result.summary
    assert "required kind=scene_video" in result.summary
    assert ledger(env) == before


@pytest.mark.parametrize(
    "failures", [None, {}, "DELIVERABLE_MISSING", [], [None, {}, float("inf")]]
)
def test_presentation_only_malformed_failures_do_not_invent_cause(env, monkeypatch, failures):
    # Explicit presentation-only stub; not evidence of a kernel verdict.
    outcome = {
        "lifecycle": "ACTIVE",
        "verification": "PARTIAL",
        "delivery": "MISSING",
        "repair_directive": {"failures": failures},
        "artifact_refs": [],
    }
    monkeypatch.setattr(TaskCoordinator, "consider", lambda self, task_id: outcome)
    result = observe(env)
    assert "required kind=" not in result.summary
    assert "DELIVERABLE_MISSING" not in result.summary
    assert "verification_scope=UNKNOWN" in result.summary
    assert "验收未过；未确认目标达成" in result.summary


def test_presentation_only_bounds_and_private_tails(env, monkeypatch):
    outcome = {
        "lifecycle": "ACTIVE",
        "verification": "FAIL",
        "delivery": "MISSING",
        "repair_directive": {
            "failures": [
                "DELIVERABLE_MISSING: required 交付物 scene_video 未在账本\nSECRET",
                "REQUIRED_FILE_MISSING: /private/SECRET\x1b[31m\r\n",
                "TRACKING_EXCEEDED: " + "SECRET" * 10000,
            ]
            + ["CHECK_FAILED: SECRET"] * 100
        },
        "artifact_refs": [
            {
                "artifact_id": "art_safe",
                "open_command": "rosclaw artifact open art_safe; SECRET",
                "path": "/private/SECRET",
                "metadata": {"secret": "SECRET"},
            },
            {"artifact_id": "art_bad\x1b[31m", "open_command": "SECRET"},
        ]
        + [{"artifact_id": "art_ok", "open_command": "rosclaw artifact open art_ok"}] * 100,
    }
    monkeypatch.setattr(TaskCoordinator, "consider", lambda self, task_id: outcome)
    result = observe(env)
    assert "required kind=scene_video" in result.summary
    assert "REQUIRED_FILE_MISSING" in result.summary
    assert "TRACKING_EXCEEDED" in result.summary
    assert "前 8 项诊断" in result.summary and "前 6 项 Ref" in result.summary
    assert "art_safe" in result.summary
    assert "SECRET" not in result.summary and "/private" not in result.summary
    assert not any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in result.summary)
    assert len(result.summary) < 2000
    assert observe(env).summary == result.summary


@pytest.mark.parametrize("with_code", [False, True])
def test_presentation_only_legacy_private_categories_are_counted(env, monkeypatch, with_code):
    # Presentation-only negative control, not a manufactured real-kernel verdict.
    private_failures = [
        "缺交付文件 /private/秘密.txt\x1b[31m\r\n令牌SECRET",
        "验收异常：输入提示SECRET\u202e/private/receipt.json",
        "lowercase_category: SECRET",
        "坏诊断" + "SECRET" * 10000,
        None,
        float("inf"),
    ]
    codes = ["DELIVERABLE_MISSING: required 交付物 scene_video 未在账本 SECRET"]
    failures = (codes if with_code else []) + private_failures
    outcome = {
        "lifecycle": "ACTIVE",
        "verification": "FAIL",
        "delivery": "PARTIAL",
        "repair_directive": {"failures": failures},
        "artifact_refs": [
            {
                "artifact_id": "art_current",
                "open_command": "rosclaw artifact open art_current",
                "path": "/private/SECRET",
            }
        ],
    }
    original = list(failures)
    monkeypatch.setattr(TaskCoordinator, "consider", lambda self, task_id: outcome)
    result = observe(env)
    assert "未分类失败诊断=6 项（原因未确认；原文不展示）" in result.summary
    assert ("required kind=scene_video" in result.summary) == with_code
    assert "verification=FAIL" in result.summary
    assert "verification_scope=UNKNOWN" in result.summary
    assert "task_semantic_verification=UNKNOWN" in result.summary
    assert "验收未过；未确认目标达成" in result.summary
    assert "rosclaw artifact open art_current" in result.summary
    for private in (
        "SECRET",
        "/private",
        "秘密.txt",
        "receipt.json",
        "输入提示",
        "lowercase_category",
    ):
        assert private not in result.summary
    assert not any(ord(c) < 32 or 127 <= ord(c) <= 159 or c == "\u202e" for c in result.summary)
    assert len(result.summary) < 2000
    assert observe(env).summary == result.summary
    assert failures == original


def test_presentation_only_unknown_count_is_limited_to_diagnostic_window(env, monkeypatch):
    # Explicit presentation-only bound check; no verifier/kernel replacement evidence.
    outcome = {
        "lifecycle": "ACTIVE",
        "verification": "FAIL",
        "delivery": "PARTIAL",
        "repair_directive": {"failures": ["私密SECRET\n"] * 9},
        "artifact_refs": [],
    }
    monkeypatch.setattr(TaskCoordinator, "consider", lambda self, task_id: outcome)
    result = observe(env)
    assert "未分类失败诊断=8 项" in result.summary
    assert "仅展示前 8 项诊断" in result.summary
    assert "SECRET" not in result.summary
    assert "required kind=" not in result.summary
    assert "verification_scope=UNKNOWN" in result.summary
    assert len(result.summary) < 2000
    assert observe(env).summary == result.summary
