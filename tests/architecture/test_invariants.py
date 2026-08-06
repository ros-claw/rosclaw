"""Static architecture invariant tests (ADR-0000 .. ADR-0006).

These tests do not exercise runtime behavior. They guard the process and
trust boundaries frozen by the ADRs: the cognitive plane (agentd, contracts,
team, operator) must never import hardware/privileged paths, legacy modules
stay marked, and public contracts never carry secret-like fields.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "rosclaw"

COGNITIVE_PACKAGES = ["agentd", "contracts", "team", "operator", "console"]

# Imports the cognitive plane must never make (ADR-0001 §3).
# NOTE: ``rosclaw.daemon.client`` / ``rosclaw.daemon.protocol`` are the
# *northbound IPC client* — the sanctioned channel to rosclawd, not a
# second physical boundary. ``rosclaw.kernel.contracts`` carries wire
# DTOs. Everything else in daemon/kernel is privileged southbound code.
FORBIDDEN_IMPORT_PATTERNS = [
    re.compile(r"rosclaw\.daemon\.(server|service|ledger|permits|session_manager"
               r"|worker_manager|watchdog|health|cli)\b"),
    re.compile(r"rosclaw\.daemon(?!\.client|\.protocol)\b"),
    re.compile(r"rosclaw\.control\b"),
    re.compile(r"rosclaw\.mcp_drivers\b"),
    re.compile(r"rosclaw\.kernel\.(action_gateway|registry|executors)\b"),
    re.compile(r"rosclaw\.sdk_to_mcp\b"),
    re.compile(r"serial\b"),
    re.compile(r"can(?:socket)?\b"),
    re.compile(r"gpio\b", re.IGNORECASE),
]

# Secret-like names banned from public contracts (ADR-0000 §2).
SECRET_FIELD_RE = re.compile(
    r"(api_key|secret|password|passwd|private_key|access_token|refresh_token"
    r"|bearer|permit_secret|hmac_key)",
    re.IGNORECASE,
)


def _python_files(package: str) -> list[Path]:
    root = SRC / package
    if not root.exists():
        return []
    files = sorted(root.rglob("*.py"))
    if package == "operator":
        # The daemon-side consent plane (upstream protocol/store/cli) is
        # trusted code by design — it may sit next to permit material. The
        # cognitive-plane restriction applies to the agentd-side broker.
        files = [f for f in files if f.name in ("broker.py", "__init__.py")]
    return files


def _imports_of(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


@pytest.mark.parametrize("package", COGNITIVE_PACKAGES)
def test_cognitive_plane_never_imports_hardware_paths(package: str) -> None:
    files = _python_files(package)
    if not files:
        pytest.skip(f"{package} not created yet")
    violations: list[str] = []
    for path in files:
        for mod in _imports_of(path):
            for pattern in FORBIDDEN_IMPORT_PATTERNS:
                if pattern.search(mod):
                    violations.append(f"{path.relative_to(SRC)} imports {mod}")
    assert not violations, "cognitive plane must not reach privileged paths:\n" + "\n".join(
        violations
    )


def test_public_contracts_have_no_secret_like_fields() -> None:
    files = _python_files("contracts")
    if not files:
        pytest.skip("contracts not created yet")
    violations: list[str] = []
    for path in files:
        if "test" in path.name:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            # dataclass / annotated fields
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                name = node.target.id
                if SECRET_FIELD_RE.search(name) and not name.endswith("_ref"):
                    violations.append(f"{path.relative_to(SRC)}: field {name!r}")
            # keyword argument names (values like scope labels are fine)
            if (
                isinstance(node, ast.keyword)
                and node.arg
                and SECRET_FIELD_RE.search(node.arg)
                and not node.arg.endswith("_ref")
            ):
                violations.append(f"{path.relative_to(SRC)}: kwarg {node.arg!r}")
            # dict keys in key position
            if isinstance(node, ast.Dict):
                for key in node.keys:
                    if (
                        isinstance(key, ast.Constant)
                        and isinstance(key.value, str)
                        and SECRET_FIELD_RE.search(key.value)
                        and not key.value.endswith("_ref")
                    ):
                        violations.append(f"{path.relative_to(SRC)}: dict key {key.value!r}")
    assert not violations, (
        "public contracts must carry credential references, never secrets:\n"
        + "\n".join(sorted(set(violations)))
    )


def test_legacy_modules_carry_maturity_marker() -> None:
    swarm_init = (SRC / "swarm" / "__init__.py").read_text(encoding="utf-8")
    assert "experimental_legacy" in swarm_init
    for legacy in ("ai_collaboration.py", "llm_provider.py"):
        text = (SRC / "agent_runtime" / legacy).read_text(encoding="utf-8")
        assert "experimental_legacy" in text, f"{legacy} missing maturity marker"


def test_adr_index_covers_all_adr_files() -> None:
    adr_dir = Path(__file__).resolve().parents[2] / "docs" / "adr"
    index = (adr_dir / "README.md").read_text(encoding="utf-8")
    for adr in sorted(adr_dir.glob("0*.md")):
        assert adr.name in index, f"{adr.name} not listed in docs/adr/README.md"
