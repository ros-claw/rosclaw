"""G-1a（0916 三审 B-1 冒烟实证）：wheel JS 运行时闭包。

实证根因：wheel 内嵌 js_stage 只有 dist .js 没有 node_modules——
干净 venv 安装后 `rosclaw chat` 即死（Cannot find package
'@earendil-works/pi-coding-agent'）。"代码已存在"≠"生产路径已
接通"：此前的 wheel 从未在干净环境真实启动过 chat。

修复契约：
- 首跑 bootstrap：嵌入 package.json+lock+dist 复制到用户态运行时
  根（~/.rosclaw/js-runtime 或 ROSCLAW_JS_RUNTIME_ROOT），
  npm ci --omit=dev 装生产依赖；
- 幂等：lock digest + node_modules + entry 齐 → 不再跑 npm；
- lock 变化（升级）→ 重新 bootstrap；
- 失败诚实报错（JS_RUNTIME_BOOTSTRAP_FAILED + 原因 + 手动命令），
  不静默降级成"entry 存在但跑不起来"；
- 诊断面（js_runtime_state）只读无副作用——doctor 不触发 npm。
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from rosclaw.agentd import pi_entry


@pytest.fixture()
def fake_wheel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """假 wheel 布局：site-packages/rosclaw/js_stage/<pkg>/ 只有
    package.json/lock/dist（无 node_modules）+ 用户态运行时根。"""
    site = tmp_path / "site-packages" / "rosclaw"
    stage = site / "js_stage" / "rosclaw-agent"
    (stage / "dist" / "src").mkdir(parents=True)
    (stage / "package.json").write_text(
        json.dumps({"name": "rosclaw-agent", "version": "0.1.0"}),
        encoding="utf-8",
    )
    (stage / "package-lock.json").write_text('{"lockfileVersion": 3, "v": 1}')
    (stage / "dist" / "src" / "main.js").write_text("console.log('entry')")
    # postinstall 补丁器是运行时依赖（G-1a：缺它 npm ci 即死）。
    (stage / "patches").mkdir()
    (stage / "patches" / "apply-upstream-patches.mjs").write_text("// patch")
    runtime_root = tmp_path / "js-runtime"
    monkeypatch.setenv("ROSCLAW_JS_RUNTIME_ROOT", str(runtime_root))
    monkeypatch.setattr(pi_entry, "_wheel_package_root", lambda: site)
    # 仓库布局优先于 wheel 分支——测试在 repo 内跑，必须把
    # Path(__file__).parents[3] 引离真实仓库（否则命中 dev dist）。
    monkeypatch.setattr(
        pi_entry, "__file__", str(site / "agentd" / "pi_entry.py"),
    )
    return {
        "site": site, "stage": stage, "runtime_root": runtime_root,
    }


def _fake_npm_ok(root: Path) -> list[list[str]]:
    """记录调用并模拟 npm ci 产出 node_modules。"""
    calls: list[list[str]] = []

    def _runner(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
        calls.append(cmd)
        (cwd / "node_modules" / "pi-coding-agent").mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    _runner.calls = calls  # type: ignore[attr-defined]
    return _runner


class TestBootstrap:
    def test_first_run_bootstraps_and_resolves_runtime_entry(self, fake_wheel):
        npm = _fake_npm_ok(fake_wheel["runtime_root"])
        root = pi_entry.ensure_js_runtime("rosclaw-agent", npm_runner=npm)
        assert root == fake_wheel["runtime_root"] / "rosclaw-agent"
        assert len(npm.calls) == 1, "首跑必须跑一次 npm ci"
        assert npm.calls[0][0].endswith("npm") and npm.calls[0][1] == "ci"
        assert (root / "node_modules").is_dir()
        assert (root / "dist" / "src" / "main.js").exists()
        assert (root / "package-lock.json").exists()
        assert (root / "patches" / "apply-upstream-patches.mjs").exists(), (
            "postinstall 补丁器必须随 bootstrap 复制——缺它 npm ci 即死"
        )

    def test_second_run_is_idempotent(self, fake_wheel):
        npm = _fake_npm_ok(fake_wheel["runtime_root"])
        pi_entry.ensure_js_runtime("rosclaw-agent", npm_runner=npm)
        pi_entry.ensure_js_runtime("rosclaw-agent", npm_runner=npm)
        assert len(npm.calls) == 1, "lock digest 未变不得重跑 npm"

    def test_lock_change_rebootstraps(self, fake_wheel):
        npm = _fake_npm_ok(fake_wheel["runtime_root"])
        pi_entry.ensure_js_runtime("rosclaw-agent", npm_runner=npm)
        (fake_wheel["stage"] / "package-lock.json").write_text('{"v": 2}')
        pi_entry.ensure_js_runtime("rosclaw-agent", npm_runner=npm)
        assert len(npm.calls) == 2, "升级（lock 变化）必须重新 bootstrap"

    def test_npm_failure_is_honest(self, fake_wheel):
        def _bad_npm(cmd, cwd):
            return subprocess.CompletedProcess(cmd, 1, "", "ECONNREFUSED registry")

        with pytest.raises(pi_entry.JsRuntimeBootstrapError) as excinfo:
            pi_entry.ensure_js_runtime("rosclaw-agent", npm_runner=_bad_npm)
        message = str(excinfo.value)
        assert "JS_RUNTIME_BOOTSTRAP_FAILED" in message
        assert "ECONNREFUSED" in message
        assert "npm ci" in message, "必须给手动命令指引"

    def test_missing_npm_is_honest(self, fake_wheel, monkeypatch):
        monkeypatch.setattr(pi_entry.shutil, "which", lambda _name: None)
        with pytest.raises(pi_entry.JsRuntimeBootstrapError) as excinfo:
            pi_entry.ensure_js_runtime("rosclaw-agent")
        assert "npm" in str(excinfo.value)


class TestEntryResolution:
    def test_wheel_branch_uses_bootstrapped_runtime(self, fake_wheel):
        npm = _fake_npm_ok(fake_wheel["runtime_root"])
        entry = pi_entry.package_entry(
            "rosclaw-agent", "ROSCLAW_AGENT_ENTRY",
            bootstrap=True, npm_runner=npm,
        )
        assert entry is not None
        assert str(fake_wheel["runtime_root"]) in entry
        assert entry.endswith("dist/src/main.js")

    def test_state_probe_has_no_side_effects(self, fake_wheel):
        state = pi_entry.js_runtime_state("rosclaw-agent")
        assert state["embedded"] is True
        assert state["bootstrapped"] is False
        assert state["needs_bootstrap"] is True
        assert not fake_wheel["runtime_root"].exists(), (
            "诊断面不得触发 npm/写盘（doctor 本地-only）"
        )
