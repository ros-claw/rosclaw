"""0827 体验审计 P0-5 红测试：语义真实验收——PASS_NEAR_LIMIT。

0827 实证：最大误差 19.86mm / 阈值 20mm（99.3% 阈值占用）却显示
普通 PASS——低质量 PASS 是假成功。闭环断言：

1. tracking_grade：≥90% 阈值占用 → PASS_NEAR_LIMIT；<90% → PASS；
   超阈值 → FAIL；
2. 生产链端到端：阈值贴近实测误差时 outcome.verification ==
   "PASS_NEAR_LIMIT"（不是 PASS），verification.completed 事件
   与 verifications 账本带 grade——三处一致；
3. 宽松阈值下仍是普通 PASS（不误报）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestTrackingGrade:
    def test_near_limit_band(self) -> None:
        from rosclaw.task_kernel.embodied_verifier import tracking_grade

        # 0827 实证数字：19.86mm / 20mm = 99.3% → PASS_NEAR_LIMIT。
        assert tracking_grade(0.01986, 0.020) == "PASS_NEAR_LIMIT"
        assert tracking_grade(0.018, 0.020) == "PASS_NEAR_LIMIT"  # 90%
        assert tracking_grade(0.0179, 0.020) == "PASS"
        assert tracking_grade(0.010, 0.025) == "PASS"
        assert tracking_grade(0.0201, 0.020) == "FAIL"


def _measure_error(kernel, conn, tmp_path: Path) -> tuple[str, float]:
    """跑一次生产链，返回 (task_id, 实测 max_error_m)（验收账本
    checks_json 是误差事实的持久面）。"""
    from rosclaw.agentd.task_execution import TaskExecutionService
    from tests.agentd.test_r02_task_spec_deliverables import _draw_task

    task_id = _draw_task(kernel, tmp_path, "画一个五角星")
    kernel.note_tool_use(task_id, "rosclaw_task")
    TaskExecutionService(kernel=kernel, conn=conn, home=tmp_path).execute(
        task_id,
        recipe_inputs={"shape": "star5",
                       "center_m": [0.35, 0.25, 0.30], "scale_m": 0.10},
    )
    row = conn.execute(
        "SELECT checks_json FROM verifications WHERE task_id = ? "
        "AND status = 'PASS' ORDER BY rowid DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    assert row, "PASS 验收行缺失"
    checks = json.loads(str(row["checks_json"]))
    return task_id, float(checks["tracking_max_error_m"])


class TestNearLimitEndToEnd:
    def test_tight_threshold_yields_pass_near_limit(
        self, tmp_path: Path
    ) -> None:
        """阈值 = 实测误差的 1.02 倍（>90% 占用）→ outcome
        verification PASS_NEAR_LIMIT（账本无假 PASS）。"""
        from rosclaw.agentd.task_execution import TaskExecutionService
        from rosclaw.task_kernel.coordinator import TaskCoordinator
        from tests.agentd.test_r01_production_chain import _kernel

        kernel, conn = _kernel(tmp_path)
        _, measured = _measure_error(kernel, conn, tmp_path)
        assert measured > 0
        # 第二次任务：验收阈值贴近实测（利用率 ≈98%）。必须不同
        # session——_draw_task 的固定 mission/session/message 幂等
        # 键会返回同一（已终态）task。
        kernel.persist_input(
            mission_id="mis_1", session_ref="s2",
            message_id="msg_2", text="再画一个五角星（阈值收紧）",
        )
        bound = kernel.ensure_task_for_effect(
            mission_id="mis_1", session_ref="s2", backend_native_id="s2",
            cwd=str(tmp_path), body_id="sim/ur5e",
        )
        task_id = str(bound["task_id"])
        kernel.note_tool_use(task_id, "rosclaw_task")
        TaskExecutionService(kernel=kernel, conn=conn, home=tmp_path).execute(
            task_id,
            recipe_inputs={
                "shape": "star5",
                "center_m": [0.35, 0.25, 0.30], "scale_m": 0.10,
                "acceptance": {"max_tracking_error_m": measured * 1.02},
            },
        )
        outcome = TaskCoordinator(kernel).consider(task_id)
        assert outcome is not None
        assert outcome["verification"] == "PASS_NEAR_LIMIT", outcome
        assert outcome["lifecycle"] == "COMPLETED", outcome
        row = conn.execute(
            "SELECT checks_json FROM verifications WHERE task_id = ? "
            "AND status = 'PASS' ORDER BY rowid DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        assert row, "PASS 验收行缺失"
        checks = json.loads(str(row["checks_json"]))
        assert checks.get("grade") == "PASS_NEAR_LIMIT", checks

    def test_loose_threshold_stays_plain_pass(self, tmp_path: Path) -> None:
        """默认阈值（利用率 <90%）→ 普通 PASS（不误报 near-limit）。"""
        from rosclaw.task_kernel.coordinator import TaskCoordinator
        from tests.agentd.test_r01_production_chain import _kernel

        kernel, conn = _kernel(tmp_path)
        task_id, measured = _measure_error(kernel, conn, tmp_path)
        outcome = TaskCoordinator(kernel).consider(task_id)
        assert outcome is not None
        # 默认阈值 25mm 地板——实测 ~20mm 利用率 <90%。
        assert measured < 0.9 * 0.025, f"实测误差假设失效：{measured}"
        assert outcome["verification"] == "PASS", outcome


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
