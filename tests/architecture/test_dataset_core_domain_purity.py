from __future__ import annotations

import ast
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2] / "src/rosclaw/dataset"
DOWNSTREAM_PREFIXES = ("rosclaw_soccer",)
DOMAIN_PATH_TOKENS = (
    "ballistic_contact",
    "football",
    "free_kick",
    "g1_",
    "goalkeeper",
    "stadium",
)


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            values.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            values.append(node.module)
    return tuple(values)


def test_dataset_core_has_no_downstream_import_or_domain_path() -> None:
    violations: list[str] = []
    for path in sorted(DATASET_ROOT.rglob("*.py")):
        relative = path.relative_to(DATASET_ROOT).as_posix().lower().replace("-", "_")
        source = path.read_text(encoding="utf-8").lower().replace("-", "_")
        for token in DOMAIN_PATH_TOKENS:
            if token in relative:
                violations.append(f"domain path: {relative} ({token})")
            if token in source:
                violations.append(f"domain source token: {relative} ({token})")
        for module in _imports(path):
            if any(
                module == prefix or module.startswith(prefix + ".")
                for prefix in DOWNSTREAM_PREFIXES
            ):
                violations.append(f"downstream import: {relative} -> {module}")

    assert violations == []
