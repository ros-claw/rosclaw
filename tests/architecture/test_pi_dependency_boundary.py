"""Pi dependency boundary tests (ADR-0008, 大纲 §19.1).

Scans npm dependency trees (when node/ exists) and Python imports to
enforce: only pi-tui/pi-ai may enter production, Pi/Codex/Hermes/OpenCode
Agent runtimes are banned, and package versions are exactly pinned.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
NODE_ROOT = REPO / "node"

BANNED_NPM_PACKAGES = (
    "@earendil-works/pi-agent-core",
    "@earendil-works/pi-coding-agent",
    "hermes-agent",
    "opencode",
)

ALLOWED_PI_PACKAGES = {
    "@earendil-works/pi-tui": "0.83.0",
    "@earendil-works/pi-ai": "0.83.0",
}

BANNED_PYTHON_IMPORTS = re.compile(r"pi_agent_core|pi_coding_agent|hermes_agent|opencode_agent")


def _package_jsons() -> list[Path]:
    if not NODE_ROOT.exists():
        return []
    return [
        p for p in NODE_ROOT.rglob("package.json") if "node_modules" not in p.parts
    ]


class TestNpmBoundary:
    def test_banned_packages_absent(self) -> None:
        if not _package_jsons():
            pytest.skip("node workspace not created yet")
        for path in _package_jsons():
            data = json.loads(path.read_text(encoding="utf-8"))
            for section in ("dependencies", "devDependencies", "peerDependencies"):
                for name in data.get(section, {}):
                    assert name not in BANNED_NPM_PACKAGES, (
                        f"{path}: banned package {name} in {section}"
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
        lock = NODE_ROOT / "package-lock.json"
        if not lock.exists():
            pytest.skip("lockfile not created yet")
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
    """The TUI/modeld code (when it exists) must not cross its lane."""

    def test_modeld_never_touches_hardware_paths(self) -> None:
        modeld = NODE_ROOT / "services" / "rosclaw-modeld"
        if not modeld.exists():
            pytest.skip("modeld not created yet")
        banned = re.compile(r"rosclawd|/dev/tty|serialport|can-utils|gpio|robot-sdk", re.IGNORECASE)
        for path in modeld.rglob("*.ts"):
            if "node_modules" in path.parts or "test" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            assert not banned.search(text), f"{path} references hardware paths"

    def test_tui_has_no_model_client(self) -> None:
        tui = NODE_ROOT / "apps" / "rosclaw-tui"
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
