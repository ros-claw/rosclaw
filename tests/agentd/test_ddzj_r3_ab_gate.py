"""大道至简 R3：A/B 比较门禁的比较器逻辑（合成夹具——真实比较
由 operator 带 key 跑，与 star_canary 同级）。

闭环断言：
1. B 成功率 < A → 阻断（负价值——方案原文「应直接阻断发布」）；
2. B ≥ A 且无冒充 → 通过（耗时/胶水劣势只降级证据强度，不阻断）；
3. B 组出现五角星冒充（star_impostor）→ 阻断（0905 假成功）；
4. 任一侧缺数据 → 不可判定（诚实：不算过）。
"""

from __future__ import annotations

import pytest

from scripts.ab_compare import RunResult, gate_verdict


def _r(group: str, ok: bool, wall: float = 100.0, glue: int = 0,
       impostor: bool = False) -> RunResult:
    return RunResult(
        group=group, task_id="t", ok=ok, wall_seconds=wall,
        tool_calls=5, glue_bytes=glue, visible_lines=50,
        evidence={"star_impostor": "p"} if impostor else {},
    )


class TestGateVerdict:
    def test_b_worse_than_a_blocks(self) -> None:
        ok, reasons = gate_verdict([_r("A", True), _r("B", False)])
        assert not ok
        assert any("负价值" in r for r in reasons)

    def test_b_equal_or_better_passes(self) -> None:
        ok, _ = gate_verdict([_r("A", True), _r("B", True)])
        assert ok

    def test_star_impostor_blocks(self) -> None:
        ok, reasons = gate_verdict([_r("A", True), _r("B", True, impostor=True)])
        assert not ok
        assert any("冒充" in r for r in reasons)

    def test_missing_side_is_not_a_pass(self) -> None:
        ok, reasons = gate_verdict([_r("B", True)])
        assert not ok
        assert any("缺数据" in r for r in reasons)

    def test_slower_or_more_glue_downgrades_not_blocks(self) -> None:
        ok, reasons = gate_verdict(
            [_r("A", True, wall=50, glue=0), _r("B", True, wall=200, glue=900)]
        )
        assert ok
        assert any("降一级" in r for r in reasons)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
