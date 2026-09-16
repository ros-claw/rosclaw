"""ADR-0014 静态不变量：Simulation Harness 架构冻结。

守护规则：
- sim/ 永不 import 物理执行与授权面（daemon 南向、control、mcp_drivers、
  serial/can/gpio）——仿真不是第二条物理通道；
- sim/ 不得定义或引用 Harness Backend 命名（MuJoCo 不是
  NativeHarnessBackend）；
- 仿真契约不含 secret-like 字段（ADR-0000 §2）；
- 新模块带 experimental 成熟度标注（ADR-0000 §4）。
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "rosclaw"
SIM_DIR = SRC / "sim"

FORBIDDEN_IMPORT_PATTERNS = [
    re.compile(r"rosclaw\.daemon\b"),
    re.compile(r"rosclaw\.control\b"),
    re.compile(r"rosclaw\.mcp_drivers\b"),
    re.compile(r"serial\b"),
    re.compile(r"can(?:socket)?\b"),
    re.compile(r"gpio\b", re.IGNORECASE),
]

SECRET_FIELD_RE = re.compile(
    r"(api_key|secret|password|passwd|private_key|access_token|refresh_token"
    r"|bearer|permit_secret|hmac_key)",
    re.IGNORECASE,
)

HARNESS_BACKEND_RE = re.compile(r"NativeHarnessBackend|class\s+\w*HarnessBackend")


def _sim_python_files() -> list[Path]:
    return sorted(SIM_DIR.rglob("*.py"))


def _imports_of(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_sim_never_imports_physical_boundary() -> None:
    offenders: list[str] = []
    for path in _sim_python_files():
        for module in _imports_of(path):
            for pattern in FORBIDDEN_IMPORT_PATTERNS:
                if pattern.search(module):
                    offenders.append(f"{path.relative_to(SRC)} imports {module}")
    assert not offenders, "\n".join(offenders)


def test_sim_defines_no_harness_backend() -> None:
    offenders: list[str] = []
    for path in _sim_python_files():
        text = path.read_text(encoding="utf-8")
        if HARNESS_BACKEND_RE.search(text):
            offenders.append(str(path.relative_to(SRC)))
    assert not offenders, "\n".join(offenders)


def test_sim_contracts_no_secret_like_fields() -> None:
    contracts = SIM_DIR / "contracts.py"
    tree = ast.parse(contracts.read_text(encoding="utf-8"), filename=str(contracts))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            if SECRET_FIELD_RE.search(name) and not name.endswith("_ref"):
                offenders.append(name)
    assert not offenders, ", ".join(offenders)


def test_sim_contracts_only_depend_on_contracts_common() -> None:
    """契约模块依赖最轻：只允许 import rosclaw.contracts.common。"""
    contracts = SIM_DIR / "contracts.py"
    for module in _imports_of(contracts):
        if module.startswith("rosclaw."):
            assert module == "rosclaw.contracts.common", module


def test_maturity_markers() -> None:
    init_text = (SIM_DIR / "__init__.py").read_text(encoding="utf-8")
    assert "experimental" in init_text
    for name in ("contracts.py", "capabilities.py"):
        text = (SIM_DIR / name).read_text(encoding="utf-8")
        assert "experimental" in text, name
