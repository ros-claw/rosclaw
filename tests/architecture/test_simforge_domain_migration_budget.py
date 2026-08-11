from __future__ import annotations

import ast
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SIMFORGE_ROOT = REPOSITORY_ROOT / "src/rosclaw/simforge"
BASELINE_PATH = Path(__file__).with_name("fixtures") / "simforge_domain_debt.txt"
DOMAIN_TOKENS = ("g1_", "g1_goalforge", "goalforge", "unitree_")


def _domain_paths() -> set[str]:
    result: set[str] = set()
    for path in SIMFORGE_ROOT.rglob("*.py"):
        relative = path.relative_to(SIMFORGE_ROOT).as_posix().lower().replace("-", "_")
        if any(token in relative for token in DOMAIN_TOKENS):
            result.add(relative)
    return result


def test_simforge_domain_debt_can_only_decrease() -> None:
    """Quarantine inherited debt while task providers are extracted downstream."""

    baseline = {
        line.strip()
        for line in BASELINE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    observed = _domain_paths()

    assert observed - baseline == set(), (
        f"new embodiment/domain code entered SimForge Core: {sorted(observed - baseline)}"
    )


def test_simforge_registry_does_not_depend_on_growth_or_downstream_packages() -> None:
    path = SIMFORGE_ROOT / "registry.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)

    assert not any(
        module == prefix or module.startswith(prefix + ".")
        for module in imports
        for prefix in ("rosclaw.growth", "rosclaw_soccer")
    )
