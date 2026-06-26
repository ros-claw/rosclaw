"""Architecture test: non-body modules must not read body.yaml directly."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2] / "src" / "rosclaw"


FORBIDDEN_PATTERNS = [
    "~/.rosclaw/body/body.yaml",
    ".rosclaw/body/body.yaml",
    "body/body.yaml",
]

ALLOWED_PATHS = [
    "src/rosclaw/body",
]


def _is_allowed(path: Path) -> bool:
    rel = path.relative_to(PROJECT_ROOT.parents[1])
    return any(str(rel).startswith(a) for a in ALLOWED_PATHS)


def _walk_source_files(root: Path):
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".py"):
                yield Path(dirpath) / filename


def _contains_forbidden(source: str) -> list[str]:
    found: list[str] = []
    for pattern in FORBIDDEN_PATTERNS:
        if pattern in source:
            found.append(pattern)
    return found


@pytest.mark.parametrize(
    "module_dir",
    [
        "sandbox",
        "provider",
        "skill_manager",
        "memory",
        "dashboard",
        "mcp",
    ],
)
def test_no_direct_body_yaml_access(module_dir: str):
    module_path = PROJECT_ROOT / module_dir
    if not module_path.exists():
        pytest.skip(f"Module {module_dir} does not exist")
    violations: list[tuple[str, list[str]]] = []
    for path in _walk_source_files(module_path):
        source = path.read_text(encoding="utf-8")
        found = _contains_forbidden(source)
        if found:
            violations.append((str(path), found))
    assert not violations, f"Direct body.yaml references found: {violations}"
