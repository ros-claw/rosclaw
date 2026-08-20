"""Stage 2 直连 E2E（设计 §51）：ACPX → `rosclaw acp serve` → Native Agent。

不走 OpenClaw Gateway（gateway→acpx 的路由由 Feishu binding 在 Stage 3
验证）；用 OpenClaw 官方 ACPX 客户端驱动 rosclaw harness，验证 ACP 链路
与 Mission/Turn 落库。

用法：

    python acpx_direct_probe.py ["只回答：LIVE-ROSCLAW-ACP-OK"]

前置：
- `rosclaw channel doctor` 除 Feishu 外全绿（模型凭证已配置）
- OpenClaw 已安装 @openclaw/acpx（提供 acpx CLI）

验收（§51）：输出中出现预期回答；ROSClaw DB 出现 1 Mission + 1 Turn。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROSCLAW_BIN = os.environ.get(
    "ROSCLAW_BIN",
    str(Path(__file__).resolve().parents[3] / ".venv" / "bin" / "rosclaw"),
)
ROSCLAW_HOME = os.environ.get("ROSCLAW_HOME", str(Path.home() / ".rosclaw"))

PROMPT = sys.argv[1] if len(sys.argv) > 1 else "只回答：LIVE-ROSCLAW-ACP-OK"
EXPECT = "LIVE-ROSCLAW-ACP-OK"


def _find_acpx() -> str:
    # acpx CLI 是 @openclaw/acpx 插件包的依赖，位于嵌套 node_modules。
    candidates = sorted(
        Path.home().glob(
            ".openclaw/npm/projects/openclaw-acpx-*/node_modules/@openclaw/acpx/node_modules/.bin/acpx"
        )
    )
    if candidates:
        return str(candidates[-1])
    found = shutil.which("acpx")
    if found:
        return found
    raise SystemExit("acpx 未找到——先 `openclaw plugins install @openclaw/acpx`")


def main() -> int:
    acpx = _find_acpx()
    result = subprocess.run(
        [
            acpx,
            "--deny-all",
            "--non-interactive-permissions",
            "deny",
            "--timeout",
            "180",
            "--agent",
            f"{ROSCLAW_BIN} acp serve --home {ROSCLAW_HOME}",
            "exec",
            PROMPT,
        ],
        capture_output=True,
        text=True,
        timeout=240,
    )
    output = result.stdout + result.stderr
    print(output)
    if EXPECT in output and PROMPT == "只回答：LIVE-ROSCLAW-ACP-OK":
        print(f"[PASS] 收到预期回答 {EXPECT}")
        print("检查 ROSClaw DB：missions 表应新增 1 Mission + 对应 turn 事件。")
        return 0
    if PROMPT != "只回答：LIVE-ROSCLAW-ACP-OK":
        print("[INFO] 自定义 prompt——人工核对上方输出。")
        return 0
    print(f"[FAIL] 未收到 {EXPECT}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
