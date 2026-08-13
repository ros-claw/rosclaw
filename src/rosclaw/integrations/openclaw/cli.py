"""`rosclaw channel ...` — Channel 集成 CLI（Channel 设计 §46/§47）。

只提供 doctor（只读检查）与 setup 指引；ROSClaw CLI 是 OpenClaw 的
配置助手，不成为 Channel Runtime（§47）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_USAGE = """用法:
  rosclaw channel doctor [--home DIR] [--require-openclaw] [--no-acp-probe]
  rosclaw channel setup feishu        # 打印配置指引（不修改任何配置）
"""


def _arg_value(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None


def dispatch_channel_argv(argv: list[str]) -> int | None:
    if argv[:1] != ["channel"]:
        return None
    if len(argv) < 2 or argv[1] not in ("doctor", "setup"):
        print(_USAGE, file=sys.stderr)
        return 2

    home = Path(
        _arg_value(argv, "--home") or os.environ.get("ROSCLAW_HOME", Path.home() / ".rosclaw")
    )

    if argv[1] == "setup":
        channel = argv[2] if len(argv) > 2 else "feishu"
        guide = Path(__file__).resolve().parents[4] / "integrations" / "openclaw" / "README.md"
        print(
            f"ROSClaw 不直接配置 {channel} Channel（OpenClaw owns the Channel）。\n"
            f"按 integrations/openclaw/README.md 操作后运行 `rosclaw channel doctor` 验证。\n"
            f"指引文件: {guide}",
            file=sys.stderr,
        )
        return 0

    from rosclaw.integrations.openclaw.doctor import run_doctor

    report = run_doctor(
        home,
        require_openclaw="--require-openclaw" in argv,
        probe_acp="--no-acp-probe" not in argv,
    )
    print(report.render())
    return 1 if report.failed else 0
