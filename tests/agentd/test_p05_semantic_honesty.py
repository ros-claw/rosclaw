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




# 大道至简 R0-2b：TestNearLimitEndToEnd（recipe 链 PASS_NEAR_LIMIT
# e2e）随生产 recipe 链退役——near-limit 分级纯函数
# （TestTrackingGrade）保留；模型面验收的阈值语义由 capability
# verify 结果 + Pi 的最终回答承接（R2 范围）。


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
