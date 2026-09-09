"""W11（规格 §15.1）hatchling 构建钩子：JS staging 注入。

- dist/js-stage 存在 → 注入 wheel 的 rosclaw/js_stage/<pkg>/；
- 缺失：默认跳过（editable/开发安装与 Python-only 构建合法——
  源码布局经 packages/<pkg>/dist 解析）；设
  ROSCLAW_REQUIRE_JS_STAGE=1（CI Build Package / 发布链）则
  硬错误（规格 §15.4 缺资产硬错误落在发布门禁上）。

静态 force-include 会被 editable 构建同样强制执行（CI 全灭
实证 2026-09-09）——注入只能在钩子里按产物存在性区分。
"""

from __future__ import annotations

import os
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

_PACKAGES = ("rosclaw-agent", "rosclaw-tui")
_ITEMS = ("dist", "package.json", "package-lock.json")


class JsStageHook(BuildHookInterface):
    """Inject the common JS staging into wheels when available."""

    def initialize(self, version: str, build_data: dict) -> None:
        if self.target_name != "wheel":
            return
        stage = Path(self.root) / "dist" / "js-stage"
        present = all(
            (stage / pkg / "dist" / "src" / "main.js").exists()
            for pkg in _PACKAGES
        )
        if not present:
            if os.environ.get("ROSCLAW_REQUIRE_JS_STAGE") == "1":
                raise RuntimeError(
                    "W11 §15.1/§15.4: wheel 构建缺 JS staging——先运行 "
                    "scripts/release/build_js_staging.sh（tar/wheel 同一"
                    "构建输入）"
                )
            return
        for pkg in _PACKAGES:
            for item in _ITEMS:
                build_data["force_include"][
                    f"dist/js-stage/{pkg}/{item}"
                ] = f"rosclaw/js_stage/{pkg}/{item}"

