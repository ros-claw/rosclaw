"""PR-ADR-0012A 结构守门（调整方案 §十二.3）：用户无 engine 面。

ADR-0012A 决定：
1. Pi 是当前唯一默认 Harness Backend——用户没有任何 engine/backend
   选择面（CLI flag、配置键、TUI 命令、UI 命名一律禁止）；
2. Codex app-server 是"第二认证 Harness 候选"——不是 provider、
   不是 Worker；只有通过 HP3 Backend Conformance + 任务基准后才谈
   默认；旧 Worker 形态的 codex/acp 驱动（十五审 RF-3/RF-5，H9 后
   已无生产引用）删除，未来从 NativeHarnessBackend SPI 重写；
3. Provider 词表只含模型 API 提供方（kimi_cn/kimi_code/mock/...），
   永远不含 Harness 名称（codex/pi/app-server/claude-code）。

红测试先行——本测试在修复前必须为红：
- agentd/codex_driver.py 与 agentd/acp_driver.py 仍存在（孤儿
  Worker 形态驱动）；
- packages/rosclaw-agent/src/main.ts 文档注释仍写着不存在的
  `rosclaw chat --engine pi`。

任何重新引入 engine 选择面 / Codex-as-provider / Worker 形态
Harness 驱动的 PR 都会变红。
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "rosclaw"
TS_SRC = REPO / "packages" / "rosclaw-agent" / "src"

#: Harness 名称——永远不得作为 model provider 出现。
_HARNESS_NAMES = ("codex", "pi", "app-server", "app_server", "claude-code",
                  "claude_code", "deepseek-harness")

#: H9 后失去全部生产引用的 Worker 形态 Harness 驱动（ADR-0012A 删除）。
_DELETED_DRIVERS = [
    "agentd/codex_driver.py",
    "agentd/acp_driver.py",
]

#: Python CLI 入口文件。
_PY_CLI_FILES = ["cli.py", "root_cli.py", "entrypoint.py", "setup_cli.py"]


class TestNoEngineSurface:
    def test_python_cli_has_no_harness_engine_flag(self) -> None:
        """Python CLI 不得有 Harness engine/backend 选择面。

        合法的 --engine/--backend 只有仿真/运行时语义（choices ∈
        mujoco/isaac/mock/fixture/ros2）；任何 choices 含 Harness 名称
        （pi/codex/app-server/claude-code）的同名 flag 都是违规；
        --harness 一律禁止。
        """
        for name in _PY_CLI_FILES:
            text = (SRC / name).read_text(encoding="utf-8")
            assert '"--harness"' not in text, f"{name} 出现 --harness 用户面"
            for m in re.finditer(
                r'add_argument\(\s*"(--engine|--backend|--harness)"', text
            ):
                window = text[m.end(): m.end() + 300]
                choices_m = re.search(r"choices=\[([^\]]*)\]", window)
                choices = choices_m.group(1) if choices_m else ""
                for harness in _HARNESS_NAMES:
                    assert f'"{harness}"' not in choices, (
                        f"{name} 的 {m.group(1)} choices 混入 Harness "
                        f"名称 {harness!r}——用户不得选择 Harness"
                    )

    def test_ts_entry_has_no_engine_option(self) -> None:
        """TS chat 入口 parseArgs 不得接受 engine/backend/harness 选项，
        源码与注释都不得出现 Harness --engine 用户面。"""
        main = (TS_SRC / "main.ts").read_text(encoding="utf-8")
        assert "--engine" not in main, "main.ts 仍引用 --engine（含注释）"
        assert "--backend" not in main and "--harness" not in main
        parse = main[main.index("parseArgs"):]
        for flag in ("engine", "backend", "harness"):
            assert f'"--{flag}"' not in parse, f"parseArgs 接受 --{flag}"

    def test_no_engine_slash_command(self) -> None:
        """TUI 不得有 /engine 或 /backend 命令。"""
        commands = (TS_SRC / "extension" / "commands.ts").read_text(encoding="utf-8")
        for cmd in ('"/engine"', "'/engine", '"/backend"', "'/backend"):
            assert cmd not in commands, f"TUI 出现 {cmd} 命令"

    def test_provider_vocabulary_excludes_harness_names(self) -> None:
        """ModelProfile provider 词表不得含 Harness 名称（Codex 是
        Harness Backend 候选，不是 provider；Pi 同理）。"""
        profiles = (SRC / "agentd" / "models" / "profiles.py").read_text(
            encoding="utf-8"
        )
        providers = set(re.findall(r'provider="([^"]+)"', profiles))
        bad = providers & set(_HARNESS_NAMES)
        assert not bad, f"provider 词表混入 Harness 名称：{bad}"

    def test_worker_shape_harness_drivers_deleted(self) -> None:
        """Worker 形态的 Codex/ACP Harness 驱动已删除（H9 删除 Worker
        默认链后它们已无生产引用；未来 Codex 走 NativeHarnessBackend
        SPI + HP3 conformance 重写）。"""
        for rel in _DELETED_DRIVERS:
            assert not (SRC / rel).exists(), f"{rel} 仍存在（孤儿 Worker 驱动）"
        # 生产代码不得再引用这两个模块。
        for py in SRC.rglob("*.py"):
            text = py.read_text(encoding="utf-8")
            assert "codex_driver" not in text, f"{py} 引用 codex_driver"
            assert "acp_driver" not in text, f"{py} 引用 acp_driver"
