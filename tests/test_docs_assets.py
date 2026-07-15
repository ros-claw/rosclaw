"""Regression tests that README/docs reference files actually exist.

Prevents dead links and missing examples from shipping in a release.
"""

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _markdown_files() -> list[Path]:
    return [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "README.zh.md",
        PROJECT_ROOT / "QUICKSTART.md",
        PROJECT_ROOT / "INSTALL.md",
        PROJECT_ROOT / "CONTRIBUTING.md",
        PROJECT_ROOT / "ARCHITECTURE.md",
        PROJECT_ROOT / "CHANGELOG.md",
    ]


def _extract_relative_paths(markdown: str) -> set[str]:
    """Find relative file paths referenced in markdown links and code."""
    paths = set()
    for match in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", markdown):
        path = match.group(2).split("#")[0].split("?")[0]
        if path and not path.startswith(("http://", "https://", "mailto:")):
            paths.add(path)
    for match in re.finditer(r"`([^`]+\.(?:json|md|yaml|yml|py|xml|urdf|mjcf))`", markdown):
        paths.add(match.group(1))
    return paths


@pytest.mark.parametrize("md_file", _markdown_files())
def test_referenced_files_exist(md_file: Path) -> None:
    if not md_file.exists():
        pytest.skip(f"{md_file.name} does not exist")
    text = md_file.read_text(encoding="utf-8")
    missing = []
    for rel_path in _extract_relative_paths(text):
        # Root-level MCP config files and the generated agent context snapshot
        # are local agent/runtime state and are intentionally ignored by
        # .gitignore.
        if rel_path in {".mcp.json", "mcp.json", ".rosclaw/agent/context.snapshot.json"}:
            continue
        # Skip generated output directories
        if rel_path.startswith("generated/") or rel_path.startswith("./generated/"):
            continue
        # Skip glob patterns (e.g. test_*.py)
        if "*" in rel_path:
            continue
        # Skip absolute or user-home paths; these are not project-relative
        if rel_path.startswith(("~/", "/")):
            continue
        target = PROJECT_ROOT / rel_path
        if not target.exists():
            missing.append(rel_path)
    assert not missing, f"{md_file.name} references missing files: {missing}"


def test_readme_critical_examples_exist() -> None:
    """High-value examples referenced by README must exist."""
    required = [
        "examples/actions/unsafe_reach.json",
        "docs/unitree_go2_sdk.md",
        "src/rosclaw/mcp/minimal_server.py",
    ]
    for rel in required:
        assert (PROJECT_ROOT / rel).exists(), f"critical example missing: {rel}"
