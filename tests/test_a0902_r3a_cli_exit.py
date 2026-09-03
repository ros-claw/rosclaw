"""0902 审计 R3-a 红测试（§6.2 路径/CLI 收敛）：

1. `rosclaw open <artifact-id>` 最短主入口——等价 `rosclaw artifact
   open <id>`（TerminalPresenter 给的 open 命令必须一步到位）；
2. Ctrl-C 安静退出返回 130——Python traceback 不打到终端（进
   debug 日志）；普通异常路径不受影响的回归护栏。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


class TestOpenShortcut:
    def test_open_rewrites_to_artifact_open(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """`rosclaw open art_x` 与 `rosclaw artifact open art_x`
        同一 handler（不允许第二条实现）。"""
        from rosclaw import entrypoint

        calls: list[list[str]] = []

        def _spy(argv: list[str]) -> int:
            calls.append(list(argv))
            return 0

        monkeypatch.setattr(
            "rosclaw.agentd.cli.dispatch_artifact_argv", _spy
        )
        monkeypatch.setattr(sys, "argv", ["rosclaw", "open", "art_x"])
        assert entrypoint.main() == 0
        assert calls == [["artifact", "open", "art_x"]], calls

    def test_open_without_id_is_usage_error(
        self, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        from rosclaw import entrypoint

        monkeypatch.setattr(sys, "argv", ["rosclaw", "open"])
        # argparse 用法错误 = SystemExit(2) + stderr 用法提示。
        with pytest.raises(SystemExit) as exc_info:
            entrypoint.main()
        assert exc_info.value.code != 0
        assert "usage" in capsys.readouterr().err.lower()


class TestCtrlCCleanExit:
    def test_keyboard_interrupt_returns_130_no_traceback(
        self, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """Ctrl-C 安静退出 130——traceback 只进 debug 日志（§6.2）。"""
        from rosclaw import entrypoint

        def _boom(_argv: list[str]) -> int:
            raise KeyboardInterrupt

        monkeypatch.setattr(
            "rosclaw.root_cli.dispatch_root_cli", _boom
        )
        monkeypatch.setattr(sys, "argv", ["rosclaw", "commands"])
        rc = entrypoint.main()
        assert rc == 130, f"Ctrl-C 应返回 130，实际 {rc}"
        err = capsys.readouterr().err
        assert "Traceback" not in err, f"traceback 打到了终端: {err[:300]}"

    def test_other_exceptions_still_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """回归护栏：普通异常不被 130 吞掉（只拦 KeyboardInterrupt）。"""
        from rosclaw import entrypoint

        def _boom(_argv: list[str]) -> int:
            raise RuntimeError("real bug")

        monkeypatch.setattr("rosclaw.root_cli.dispatch_root_cli", _boom)
        monkeypatch.setattr(sys, "argv", ["rosclaw", "commands"])
        with pytest.raises(RuntimeError):
            entrypoint.main()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
