"""W06 红测试（规格 2026-09-08 §10）：一次批准与真实隔离。

§10.1 负向探针：
- present ≠ usable——bwrap 可执行文件存在但 smoke 失败（云 VM
  user namespace 受限）时不得宣布隔离就绪；
- 无法形成真实权限隔离时声明降级为进程内 provenance——
  evidence 受信区 kernel-only 是该声明级别的实际执行边界。

§10.2 REAL 永不走 POLICY_AUTO（fake REAL 缺 Operator 无法
执行——行为链由 consent/admission 测试覆盖，此处锚定判定
函数不跨域）。
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace


class TestProbeHonesty:
    def test_ready_implies_real_smoke(self, tmp_path) -> None:
        """不变式：isolation_ready=True 必须有真实 smoke 通过
        （bwrap smoke_ok 或容器后端可用）——存在性不宣布隔离。"""
        from rosclaw.firstboot.os_isolation import probe_and_persist

        probe = probe_and_persist(tmp_path)
        bwrap_ok = bool((probe.get("bwrap") or {}).get("smoke_ok"))
        container_ok = bool(probe.get("container"))
        if probe["isolation_ready"]:
            assert bwrap_ok or container_ok, (
                f"无真实 smoke 通过却宣布隔离就绪: {probe}"
            )
        else:
            # 不就绪时必须有可诊断的诚实原因（不是空记录）。
            assert probe.get("bwrap") is not None
            detail = str((probe.get("bwrap") or {}).get("detail", ""))
            assert detail or container_ok is False

    def test_persisted_record_matches_probe(self, tmp_path) -> None:
        """落盘记录与探测一致（消费方读到什么就是什么）。"""
        import json

        from rosclaw.firstboot.os_isolation import probe_and_persist

        probe = probe_and_persist(tmp_path)
        record = json.loads(
            (tmp_path / "agent" / "os-isolation.json").read_text(
                encoding="utf-8"
            )
        )
        assert record["isolation_ready"] == probe["isolation_ready"]


class TestProvenanceBoundary:
    def test_evidence_zone_rejects_model_producer(self, tmp_path) -> None:
        """进程内 provenance 的实际边界：evidence 受信区只接受
        kernel 管道登记——模型不能自写受信证据（OS 隔离不可用
        时的诚实声明级别）。"""
        import pytest

        from rosclaw.storage.migrations import MigrationRunner
        from rosclaw.task_kernel.service import TaskKernel

        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        MigrationRunner().apply(conn, "sqlite")
        kernel = TaskKernel(conn, tmp_path)
        kernel.persist_input(
            mission_id="mis_1", session_ref="s1",
            message_id="msg_1", text="验证 evidence 边界",
        )
        bound = kernel.ensure_task_for_effect(
            mission_id="mis_1", session_ref="s1",
            backend_native_id="s1", cwd=str(tmp_path),
        )
        task_id = str(bound["task_id"])
        revision = int(bound["revision"])
        evidence_dir = (
            tmp_path / "runs" / task_id / f"r{revision}" / "evidence"
        )
        evidence_dir.mkdir(parents=True, exist_ok=True)
        fake = evidence_dir / "verify_self.json"
        fake.write_text('{"verdict": "PASS"}', encoding="utf-8")
        with pytest.raises(ValueError, match="EVIDENCE_KERNEL_ONLY"):
            kernel.register_artifact(
                task_id=task_id, path=str(fake),
                media_type="application/json",
                producer="model:rosclaw_artifact_register",
            )
        # kernel 管道可登记（边界是生产者身份，不是目录位置）。
        record = kernel.register_artifact(
            task_id=task_id, path=str(fake),
            media_type="application/json",
            producer="kernel:verifier",
        )
        assert record["artifact_id"]


class TestRealNeverPolicyAuto:
    def test_real_shadow_never_auto(self) -> None:
        """POLICY_AUTO 判定对 REAL/SHADOW 永远 False（政策授权
        不跨域——缺 Operator 时 REAL 无自动路径）。"""
        from rosclaw.agentd.pi_bridge.action_admission import (
            ActionAdmissionService,
        )

        admission = ActionAdmissionService.__new__(ActionAdmissionService)
        for mode in ("REAL", "SHADOW"):
            mission = SimpleNamespace(mode=SimpleNamespace(value=mode))
            assert admission._policy_auto_applies(mission, None) is False


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
