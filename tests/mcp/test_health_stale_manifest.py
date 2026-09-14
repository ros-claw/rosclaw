"""W-2（二轮自审实证）：mcp health 陈旧安装条目韧性。

实证：宿主 ~/.rosclaw/mcp/installed.yaml 有 2026-06 安装的
realsense-d405——其 manifest 已从注册表下架（现只有 d455/
ros-mcp）。`rosclaw mcp health` 对整个检查硬失败（一行错误，
其余 server 全没查）。

闭环断言：manifest 缺失的条目必须成为该 server 的 failed 报告
（含可操作指引——uninstall 或重装），其余 server 继续检查；
不得整体异常退出。
"""

from __future__ import annotations

import pytest


class _StaleHub:
    """manifest 恒缺失（下架）——fetch_manifest 抛 ManifestNotFoundError。"""

    def fetch_manifest(self, manifest_id: str, version: str | None = None):
        from rosclaw.mcp.onboarding.hub_client import ManifestNotFoundError

        raise ManifestNotFoundError(f"Manifest not found: {manifest_id}@{version}")


def _stale_runner(tmp_path):
    from rosclaw.mcp.onboarding.health import HealthRunner
    from rosclaw.mcp.onboarding.installed import InstalledRegistry

    from rosclaw.mcp.onboarding.installed import InstalledRecord

    registry = InstalledRegistry(home=tmp_path)
    registry.add(
        InstalledRecord(
            server_name="realsense-d405",
            manifest_id="io.rosclaw.hardware.realsense-d405",
            name="realsense-d405",
            version="1.0.0",
            installed_at="2026-06-25T00:00:00Z",
            artifact_type="python",
            server_dir="",
            runtime_config_path="",
        )
    )
    return HealthRunner(home=tmp_path, registry=registry, hub=_StaleHub())


class TestStaleManifestResilience:
    def test_stale_entry_is_failed_report_not_exception(self, tmp_path) -> None:
        runner = _stale_runner(tmp_path)
        report = runner.check("realsense-d405")
        assert report.overall == "failed"
        assert report.checks, "陈旧条目无任何检查行——诊断空洞"
        manifest_check = report.checks[0]
        assert manifest_check.passed is False
        message = manifest_check.message
        # 可操作指引：说明陈旧（manifest 下架/缺失）+ 怎么处置。
        assert "下架" in message or "缺失" in message or "not found" in message.lower()
        assert "uninstall" in message or "重装" in message

    def test_check_all_continues_past_stale(self, tmp_path) -> None:
        runner = _stale_runner(tmp_path)
        # 不抛异常 + 每个已装 server 都有报告。
        reports = runner.check_all()
        assert len(reports) == 1
        assert reports[0].overall == "failed"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
