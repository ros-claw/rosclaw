"""0827 体验审计 P0-7 红测试（Python 侧）：Pi 重试预算写入。

0827 实证：Kimi 403 配额错误被 Pi 自动重试 3 次（默认
retry.maxRetries=3）——确定性错误重复烧配额。setup 必须把重试
预算写进 Pi settings（maxRetries=1：瞬态一次；配额类措辞命中 Pi
NON_RETRYABLE 词表时零重试）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestRetryBudgetWritten:
    def test_setup_writes_retry_budget(self, tmp_path: Path) -> None:
        from rosclaw.agentd.onboarding import _write_retry_budget

        _write_retry_budget(tmp_path)
        settings = json.loads(
            (tmp_path / "agent" / "settings.json").read_text(
                encoding="utf-8"
            )
        )
        assert settings.get("retry", {}).get("maxRetries") == 1, settings

    def test_preserves_other_keys_and_idempotent(self, tmp_path: Path) -> None:
        from rosclaw.agentd.onboarding import _write_retry_budget

        settings_path = tmp_path / "agent" / "settings.json"
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text(
            json.dumps({"defaultThinkingLevel": "high", "retry": {"baseDelayMs": 1500}}),
            encoding="utf-8",
        )
        _write_retry_budget(tmp_path)
        _write_retry_budget(tmp_path)  # 幂等
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        assert settings["defaultThinkingLevel"] == "high", settings
        assert settings["retry"]["maxRetries"] == 1, settings
        assert settings["retry"]["baseDelayMs"] == 1500, settings


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
