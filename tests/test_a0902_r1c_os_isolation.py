"""0902 审计 R1-c 红测试：setup/doctor 沙箱就绪（§5.3）——执行中甩给
用户 vs 启动前一次探测。

0902 实证：ubuntu VM 无 bwrap，用户跑到一半才被要求 export 全局
环境变量并重启。§5.3：setup/doctor 一次探测 bwrap/容器/文件系统
隔离/图形后端，自动选择可用后端并做真实 smoke test；无可用隔离时
在任务开始前提示一次，而不是运行到一半失败。

闭环断言：
1. doctor run_full 含 os_isolation 检查组（bwrap/容器/图形后端）；
2. bwrap 缺失 → WARN + 修复命令（不是混在一屏内部状态里无结论）；
3. bwrap 存在但 user namespace 受限（smoke 失败）→ 检出
   "present but broken"，不谎称可用；
4. 探测结果落盘 home/agent/os-isolation.json（任务开始前可消费）；
5. 无 OSMesa/EGL 图形后端 → WARN（headless 渲染需要；CI libosmesa6
   教训——装了 mujoco 不代表能渲染）。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest


def _run_full(home: Path):
    from rosclaw.firstboot.doctor import FirstbootDoctor

    return FirstbootDoctor(home=home).run_full(json_output=True)


class TestOsIsolationChecks:
    def test_full_includes_os_isolation_group(self, tmp_path: Path) -> None:
        result = _run_full(tmp_path)
        ids = {c.id for c in result.checks}
        assert "os_isolation.bwrap" in ids, "doctor 无 bwrap 检查"
        assert "os_isolation.container" in ids, "doctor 无容器后端检查"
        assert "os_isolation.graphics" in ids, "doctor 无图形后端检查"

    def test_bwrap_missing_warns_with_fix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        result = _run_full(tmp_path)
        bwrap = next(c for c in result.checks if c.id == "os_isolation.bwrap")
        from rosclaw.firstboot.doctor import CheckStatus

        assert bwrap.status in (CheckStatus.WARN, CheckStatus.FAIL)
        assert bwrap.fix, "无 bwrap 必须给修复命令（§5.3/R3.6 结论+修复）"
        assert "bwrap" in bwrap.fix or "bubblewrap" in bwrap.fix

    def test_bwrap_present_but_broken_smoke_detected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """bwrap 装了但 user namespace 受限（云 VM 常见）——smoke 必须
        检出，不得谎称隔离可用。"""
        import rosclaw.firstboot.os_isolation as oi

        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
        monkeypatch.setattr(oi, "_bwrap_smoke", lambda _path: (False, "user namespace 受限"))
        result = _run_full(tmp_path)
        bwrap = next(c for c in result.checks if c.id == "os_isolation.bwrap")
        from rosclaw.firstboot.doctor import CheckStatus

        assert bwrap.status != CheckStatus.PASS, "smoke 失败不得 PASS"
        assert "namespace" in bwrap.message or "smoke" in bwrap.message.lower()

    def test_readiness_written_to_home(self, tmp_path: Path) -> None:
        """探测落盘——chat/任务开始前可消费（§5.3 启动前处理）。"""
        _run_full(tmp_path)
        readiness = tmp_path / "agent" / "os-isolation.json"
        assert readiness.exists(), "os-isolation.json 未落盘"
        data = json.loads(readiness.read_text(encoding="utf-8"))
        assert "bwrap" in data and "checked_at" in data
        assert "isolation_ready" in data

    def test_graphics_backend_warns_when_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import rosclaw.firstboot.os_isolation as oi

        monkeypatch.setattr(oi, "_graphics_backend", lambda: None)
        result = _run_full(tmp_path)
        gfx = next(c for c in result.checks if c.id == "os_isolation.graphics")
        from rosclaw.firstboot.doctor import CheckStatus

        assert gfx.status == CheckStatus.WARN
        assert gfx.fix and ("osmesa" in gfx.fix.lower() or "egl" in gfx.fix.lower())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
