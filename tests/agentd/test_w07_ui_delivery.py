"""W07 红测试（规格 2026-09-08 §11.2/§11.3）：交付面 CLI 真实行为。

§11.2：help、合法 ID、未知 ID、文件缺失、导出——真实 handler
行为（不平行实现第二个 dispatcher）。
§11.3：无显示环境给可复制路径 + 导出提示（OSC 8 file URL 不是
依赖）；导出不静默覆盖既有文件。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


def _home_with_artifact(tmp_path: Path) -> tuple[Path, str, Path]:
    """真实账本：MissionStore DB + TaskKernel 登记的交付物。"""
    from rosclaw.storage.migrations import MigrationRunner
    from rosclaw.task_kernel.service import TaskKernel

    home = tmp_path / "home"
    db = home / "agentd" / "missions.db"
    db.parent.mkdir(parents=True)
    conn = sqlite3.connect(str(db), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    MigrationRunner().apply(conn, "sqlite")
    kernel = TaskKernel(conn, home)
    kernel.persist_input(
        mission_id="mis_1", session_ref="s1",
        message_id="msg_1", text="生成交付物",
    )
    bound = kernel.ensure_task_for_effect(
        mission_id="mis_1", session_ref="s1",
        backend_native_id="s1", cwd=str(tmp_path),
    )
    payload = tmp_path / "report.txt"
    payload.write_text("交付内容", encoding="utf-8")
    art = kernel.register_artifact(
        task_id=str(bound["task_id"]), path=str(payload),
        media_type="text/plain", producer="kernel:test",
    )
    conn.commit()
    conn.close()
    return home, str(art["artifact_id"]), payload


def _run(argv: list[str], home: Path, monkeypatch, capsys):
    """经 entrypoint._dispatch 真实路由（不绕过 dispatch 链）。"""
    monkeypatch.setenv("ROSCLAW_HOME", str(home))
    from rosclaw.entrypoint import _dispatch

    rc = _dispatch(argv)
    out = capsys.readouterr()
    return rc, out.out + out.err


class TestArtifactCliContract:
    def test_help_lists_subcommands(self, tmp_path, monkeypatch, capsys) -> None:
        home, _art, _p = _home_with_artifact(tmp_path)
        with pytest.raises(SystemExit) as exc:
            _run(["artifact", "--help"], home, monkeypatch, capsys)
        assert exc.value.code == 0
        text = capsys.readouterr().out
        for sub in ("list", "show", "path", "open", "export"):
            assert sub in text

    def test_list_and_open_valid_id_no_display(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """合法 ID + 无显示环境 → 给绝对路径 + 导出提示（不假装
        打开了查看器，OSC 8 file URL 不是依赖）。"""
        home, art_id, payload = _home_with_artifact(tmp_path)
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        rc, out = _run(["artifact", "list"], home, monkeypatch, capsys)
        assert rc == 0 and art_id in out
        rc, out = _run(["artifact", "open", art_id], home, monkeypatch, capsys)
        assert rc == 0
        assert str(payload) in out
        assert "artifact export" in out
        assert "已用系统默认程序打开" not in out, "无显示环境不得声称已打开"

    def test_unknown_id_rc2(self, tmp_path, monkeypatch, capsys) -> None:
        home, _art, _p = _home_with_artifact(tmp_path)
        rc, out = _run(
            ["artifact", "open", "art_nonexistent"], home, monkeypatch, capsys,
        )
        assert rc == 2 and "未知交付物" in out

    def test_missing_file_rc3(self, tmp_path, monkeypatch, capsys) -> None:
        home, art_id, payload = _home_with_artifact(tmp_path)
        payload.unlink()  # 登记后文件被删——诚实报缺失
        rc, out = _run(["artifact", "open", art_id], home, monkeypatch, capsys)
        assert rc == 3 and "缺失" in out

    def test_export_and_no_silent_overwrite(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """导出到指定路径；目标已存在时拒绝静默覆盖（§11.2/§6.4
        同一纪律——api.export 已是 EXPORT_TARGET_EXISTS）。"""
        home, art_id, payload = _home_with_artifact(tmp_path)
        dest = tmp_path / "out" / "copy.txt"
        rc, out = _run(
            ["artifact", "export", art_id, str(dest)],
            home, monkeypatch, capsys,
        )
        assert rc == 0 and dest.read_text(encoding="utf-8") == "交付内容"
        rc, out = _run(
            ["artifact", "export", art_id, str(dest)],
            home, monkeypatch, capsys,
        )
        assert rc != 0 and "已存在" in out, (
            "export 静默覆盖了既有文件"
        )

    def test_open_shortcut_rewrites_to_artifact_open(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """`rosclaw open <id>` 重写为 artifact open（同一 handler，
        无第二实现）——无显示环境同样给路径。"""
        home, art_id, payload = _home_with_artifact(tmp_path)
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        rc, out = _run(["open", art_id], home, monkeypatch, capsys)
        assert rc == 0 and str(payload) in out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
