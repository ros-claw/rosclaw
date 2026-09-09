"""W11 红测试（规格 2026-09-08 §15.1/§15.4）：打包断点修复。

§15.1：共同 staging——离线 tar 与 PyPI wheel 消费同一组 JS
产物；构建在专用工作目录（绝不 npm ci 开发 checkout 让其
依赖被删）。
§15.1：PyPI wheel 必须自包含已构建 JS（packages/rosclaw-agent
dist）——wheel force-include 不能假定自动包含。
§15.4：sdist 从自身内容可重建目标包（packages/scripts/js-stage
随 sdist 走）。
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _toml_section(text: str, header: str) -> str:
    """取 [header] 到下一个节头（行首 [）之间的文本。"""
    body = text.split(header, 1)[1]
    end = re.search(r"(?m)^\[", body)
    return body[: end.start()] if end else body


class TestDedicatedStagingBuild:
    def test_build_release_never_npm_ci_in_dev_checkout(self) -> None:
        """build_release.sh 不得在 packages/<pkg> 开发目录里执行
        npm ci（依赖被删状态事故源）——只能调共同 staging 脚本。"""
        script = (REPO_ROOT / "scripts" / "build_release.sh").read_text(
            encoding="utf-8"
        )
        # 开发 checkout npm ci 的具体形态：cd 进 packages/<pkg> 后 npm ci。
        offenders = re.findall(
            r'cd "?\$?\{?REPO_ROOT\}?/packages[^)]*npm ci', script,
        )
        assert not offenders, f"开发 checkout npm ci 残留: {offenders}"
        assert "build_js_staging.sh" in script, (
            "build_release.sh 未消费共同 JS staging"
        )

    def test_staging_script_builds_in_dedicated_workdir(self) -> None:
        """staging 脚本存在且构建目录是专用工作目录（dist/.js-build
        或调用方指定），不是 packages/ 开发目录。"""
        path = REPO_ROOT / "scripts" / "release" / "build_js_staging.sh"
        assert path.exists(), "缺共同 JS staging 脚本"
        text = path.read_text(encoding="utf-8")
        assert re.search(r'WORK=.*dist/\.js-build', text), (
            "staging 未默认专用工作目录"
        )
        # npm ci 只发生在 WORK 副本内。
        assert "rsync" in text and 'npm ci' in text
        assert not re.search(r'cd "\$REPO_ROOT/packages', text)


class TestWheelSelfContainment:
    def test_pyproject_registers_js_stage_hook(self) -> None:
        """wheel 的 JS 注入走自定义钩子（静态 force-include 会被
        editable 构建同样强制执行——CI 实证，必须按版本区分）。"""
        text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        hook_section = _toml_section(
            text, "[tool.hatch.build.hooks.custom]"
        )
        assert "hatch_js_stage_hook.py" in hook_section
        wheel_section = _toml_section(
            text, "[tool.hatch.build.targets.wheel.force-include]"
        )
        mapping_lines = [
            ln for ln in wheel_section.splitlines()
            if ln.strip().startswith('"')
        ]
        assert not any("js-stage" in ln for ln in mapping_lines), (
            "js-stage 不得静态 force-include（editable 构建会硬失败）"
        )
        assert not any("node_modules" in ln for ln in mapping_lines), (
            "node_modules 不得进 wheel（dev 依赖/体积）"
        )

    def test_hook_injects_for_wheel_skips_editable(
        self, tmp_path, monkeypatch
    ) -> None:
        """钩子行为：staging 在场注入 js_stage 映射；缺失默认跳过
        （editable/开发安装合法）；ROSCLAW_REQUIRE_JS_STAGE=1
        （发布门禁）缺失即硬错误。"""
        import sys

        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from hatch_js_stage_hook import JsStageHook

        hook = JsStageHook.__new__(JsStageHook)
        hook._BuildHookInterface__target_name = "wheel"
        hook._BuildHookInterface__root = str(tmp_path)
        import pytest

        # 缺 staging：默认跳过（editable/开发安装不硬失败——CI 全灭
        # 实证：静态 force-include 会误伤开发安装）。
        monkeypatch.delenv("ROSCLAW_REQUIRE_JS_STAGE", raising=False)
        data0 = {"force_include": {}}
        hook.initialize("standard", data0)
        assert data0["force_include"] == {}
        # 发布门禁（env=1）：缺 staging 硬错误并指向构建脚本。
        monkeypatch.setenv("ROSCLAW_REQUIRE_JS_STAGE", "1")
        with pytest.raises(RuntimeError, match="build_js_staging"):
            hook.initialize("standard", {"force_include": {}})
        monkeypatch.delenv("ROSCLAW_REQUIRE_JS_STAGE")
        # 有 staging → 注入映射（无 node_modules）。
        for pkg in ("rosclaw-agent", "rosclaw-tui"):
            entry = tmp_path / "dist" / "js-stage" / pkg / "dist" / "src"
            entry.mkdir(parents=True)
            (entry / "main.js").write_text("// built", encoding="utf-8")
        build_data = {"force_include": {}}
        hook.initialize("standard", build_data)
        assert any(
            v == "rosclaw/js_stage/rosclaw-agent/dist"
            for v in build_data["force_include"].values()
        )
        assert not any(
            "node_modules" in k for k in build_data["force_include"]
        )

    def test_sdist_self_contained_for_rebuild(self) -> None:
        """sdist 必须包含 packages/（JS 源码）与 scripts/release/
        （staging 构建工具）与 js-stage（预构建产物）——从自身
        内容可重建 wheel，不依赖本地 git checkout。"""
        text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        sdist_section = _toml_section(
            text, "[tool.hatch.build.targets.sdist]"
        )
        for required in ("/packages", "/scripts"):
            assert required in sdist_section, (
                f"sdist 缺 {required}（无法从自身重建 wheel）"
            )
        sdist_force = _toml_section(
            text, "[tool.hatch.build.targets.sdist.force-include]"
        )
        assert "js-stage" in sdist_force, (
            "js-stage 必须随 sdist（gitignore 产物须 force-include）"
        )


class TestWheelEmbeddedResolution:
    def test_package_entry_resolves_wheel_embedded(self, tmp_path) -> None:
        """pip 安装布局：env 无、仓库布局无、安装前缀无 → 解析
        wheel 内嵌 js_stage（rosclaw/js_stage/<pkg>/dist/src/main.js）。"""
        from rosclaw.agentd.pi_entry import _wheel_embedded_entry

        pkg_root = tmp_path / "rosclaw"
        entry = (
            pkg_root / "js_stage" / "rosclaw-agent" / "dist" / "src"
            / "main.js"
        )
        entry.parent.mkdir(parents=True)
        entry.write_text("// built", encoding="utf-8")
        assert _wheel_embedded_entry(
            "rosclaw-agent", package_root=pkg_root,
        ) == str(entry)
        assert _wheel_embedded_entry(
            "rosclaw-tui", package_root=pkg_root,
        ) is None

    def test_package_entry_order_env_repo_install_wheel(
        self, tmp_path, monkeypatch
    ) -> None:
        """解析顺序：env 优先；其后仓库/安装布局；wheel 内嵌兜底。"""
        from rosclaw.agentd import pi_entry

        monkeypatch.setenv("ROSCLAW_AGENT_ENTRY", "/env/main.js")
        assert pi_entry.package_entry(
            "rosclaw-agent", "ROSCLAW_AGENT_ENTRY"
        ) == "/env/main.js"
        monkeypatch.delenv("ROSCLAW_AGENT_ENTRY")
        pkg_root = tmp_path / "rosclaw"
        entry = (
            pkg_root / "js_stage" / "rosclaw-agent" / "dist" / "src"
            / "main.js"
        )
        entry.parent.mkdir(parents=True)
        entry.write_text("// built", encoding="utf-8")
        monkeypatch.setattr(
            pi_entry, "_wheel_package_root", lambda: pkg_root,
        )
        found = pi_entry.package_entry("rosclaw-agent", "ROSCLAW_AGENT_ENTRY")
        assert found is not None
        assert found.endswith("main.js")


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
