"""Pi dependency boundary tests (ADR-0008, 大纲 §19.1 — 2026-08-05 重构修订).

重构规格（rosclaw_native_agent重构.md §1）正式废止"禁止引入
pi-coding-agent"：Pi SDK 现在是 rosclaw-agent harness 的实现基础。
新边界：

- pi-coding-agent/pi-agent-core/pi-ai/pi-tui **只允许**出现在
  packages/rosclaw-agent（harness 包），且必须精确锁 0.83.0；
- 其他包（rosclaw-tui）仍只允许 pi-tui/pi-ai；
- hermes-agent/opencode 仍然全禁；
- Python 仍不得 import 任何外部 Agent 运行时。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
NODE_ROOT = REPO / "packages"

BANNED_NPM_PACKAGES = (
    "hermes-agent",
    "opencode",
)

# 重构修订：harness SDK 只允许出现在 rosclaw-agent，且精确锁定。
HARNESS_ONLY_PACKAGES = (
    "@earendil-works/pi-coding-agent",
    "@earendil-works/pi-agent-core",
)
HARNESS_PACKAGE_DIR = "rosclaw-agent"

ALLOWED_PI_PACKAGES = {
    "@earendil-works/pi-tui": "0.83.0",
    "@earendil-works/pi-ai": "0.83.0",
}

ALLOWED_HARNESS_PACKAGES = {
    "@earendil-works/pi-coding-agent": "0.83.0",
    "@earendil-works/pi-agent-core": "0.83.0",
    "@earendil-works/pi-ai": "0.83.0",
    "@earendil-works/pi-tui": "0.83.0",
}

BANNED_PYTHON_IMPORTS = re.compile(r"pi_agent_core|pi_coding_agent|hermes_agent|opencode_agent")


def _package_jsons() -> list[Path]:
    if not NODE_ROOT.exists():
        return []
    return [
        p
        for p in NODE_ROOT.rglob("package.json")
        if "node_modules" not in p.parts and "dist" not in p.parts
    ]


class TestNpmBoundary:
    def test_banned_packages_absent(self) -> None:
        if not _package_jsons():
            pytest.skip("node workspace not created yet")
        for path in _package_jsons():
            data = json.loads(path.read_text(encoding="utf-8"))
            is_harness = path.parent.name == HARNESS_PACKAGE_DIR
            for section in ("dependencies", "devDependencies", "peerDependencies"):
                for name in data.get(section, {}):
                    assert name not in BANNED_NPM_PACKAGES, (
                        f"{path}: banned package {name} in {section}"
                    )
                    if name in HARNESS_ONLY_PACKAGES:
                        assert is_harness, (
                            f"{path}: harness SDK {name} 只允许出现在 "
                            f"{HARNESS_PACKAGE_DIR}（重构规格 §2.4/§7）"
                        )

    def test_harness_packages_exactly_pinned(self) -> None:
        harness = NODE_ROOT / HARNESS_PACKAGE_DIR / "package.json"
        if not harness.exists():
            pytest.skip("rosclaw-agent not created yet")
        data = json.loads(harness.read_text(encoding="utf-8"))
        for section in ("dependencies",):
            for name, version in (data.get(section) or {}).items():
                if name in ALLOWED_HARNESS_PACKAGES:
                    assert version == ALLOWED_HARNESS_PACKAGES[name], (
                        f"{name} 必须精确锁 {ALLOWED_HARNESS_PACKAGES[name]}，"
                        f"got {version!r}（禁止 ^/~ 范围，重构规格 §7）"
                    )

    def test_pi_versions_exactly_pinned(self) -> None:
        if not _package_jsons():
            pytest.skip("node workspace not created yet")
        seen: set[str] = set()
        for path in _package_jsons():
            data = json.loads(path.read_text(encoding="utf-8"))
            for section in ("dependencies", "devDependencies"):
                for name, version in (data.get(section) or {}).items():
                    if name in ALLOWED_PI_PACKAGES:
                        seen.add(name)
                        assert version == ALLOWED_PI_PACKAGES[name], (
                            f"{path}: {name} must be exactly "
                            f"{ALLOWED_PI_PACKAGES[name]}, got {version!r} "
                            "(no ^ or ranges)"
                        )
                        assert not version.startswith(("^", "~"))

    def test_lockfile_no_banned_packages(self) -> None:
        for package in NODE_ROOT.glob("rosclaw-*"):
            lock = package / "package-lock.json"
            if not lock.exists():
                continue
            data = json.loads(lock.read_text(encoding="utf-8"))
            packages = (data.get("packages") or {}).keys()
            for name in BANNED_NPM_PACKAGES:
                assert not any(name in p for p in packages), f"lockfile contains {name}"


class TestPythonBoundary:
    def test_no_banned_python_imports(self) -> None:
        src = REPO / "src" / "rosclaw"
        offenders: list[str] = []
        for path in src.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith(("import ", "from ")) and BANNED_PYTHON_IMPORTS.search(
                    stripped
                ):
                    offenders.append(f"{path.relative_to(src)}: {stripped}")
        assert not offenders, "banned agent runtimes imported:\n" + "\n".join(offenders)


class TestRoleBoundaryStatements:
    """The TUI code (when it exists) must not cross its lane."""


    def test_pi_ai_providers_all_never_imported(self) -> None:
        """§14.13：pi-ai 必须按 provider 子路径懒加载，providers/all 会
        eager-load 所有 SDK。"""
        for package in NODE_ROOT.glob("rosclaw-*"):
            for path in package.rglob("*.ts"):
                if "node_modules" in path.parts or "dist" in path.parts:
                    continue
                for line in path.read_text(encoding="utf-8").splitlines():
                    stripped = line.strip()
                    if stripped.startswith(("import ", "export ")) or "import(" in stripped:
                        assert "providers/all" not in stripped, (
                            f"{path} imports providers/all: {stripped}"
                        )

    def test_tui_has_no_model_client(self) -> None:
        tui = NODE_ROOT / "rosclaw-tui"
        if not tui.exists():
            pytest.skip("tui not created yet")
        banned = re.compile(r"openai|anthropic|/v1/chat/completions|pi-ai", re.IGNORECASE)
        for path in tui.rglob("*.ts"):
            if "node_modules" in path.parts or "test" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            assert not banned.search(text), (
                f"{path} references a model API — TUI must only talk to agentd /v2"
            )
