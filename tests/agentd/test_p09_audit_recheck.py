"""0827 复核（对抗自审）：HOME 导出。

大道至简 R0-2b：断链不沉默/重启恢复两腿随生产 recipe 链
（TaskExecutionService/resume_interrupted_executions）一起删除
——执行由 Pi 经通用原语工具驱动（OperationManager 的
reattach-or-LOST 是长进程的统一恢复面）。
"""

from __future__ import annotations

from pathlib import Path

import pytest


class TestHomeEnvExported:
    def test_chat_bootstrap_exports_rosclaw_home(self, tmp_path: Path) -> None:
        """agentd 进程必须导出 ROSCLAW_HOME（否则 PlanRef 生产/消费
        分裂或 conformance 误杀工具对——用户不会手工 export）。"""
        import os

        from rosclaw.agentd.cli import _ensure_home_env

        old = os.environ.pop("ROSCLAW_HOME", None)
        try:
            _ensure_home_env(tmp_path)
            assert os.environ.get("ROSCLAW_HOME") == str(tmp_path)
        finally:
            if old is not None:
                os.environ["ROSCLAW_HOME"] = old
            else:
                os.environ.pop("ROSCLAW_HOME", None)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
