"""L0: Static checks and architecture boundary tests."""
import ast
from pathlib import Path


def test_no_forbidden_imports():
    """AUTO-ARCH-001: Auto must not import private modules from other rosclaw components."""
    forbidden = [
        "rosclaw_practice.internal",
        "rosclaw_memory.internal",
        "rosclaw_darwin.internal",
        "rosclaw_sandbox.internal",
    ]
    base = Path(__file__).parent.parent / "rosclaw_auto"
    violations = []
    for pyfile in base.rglob("*.py"):
        if pyfile.name.startswith("test_"):
            continue
        try:
            tree = ast.parse(pyfile.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for f in forbidden:
                        if alias.name.startswith(f):
                            violations.append(f"{pyfile}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                for f in forbidden:
                    if mod.startswith(f):
                        violations.append(f"{pyfile}: from {mod} import ...")
    assert not violations, "Forbidden imports found: " + "\n".join(violations)


def test_no_hardcoded_api_keys():
    """AUTO-SEC-001: No API keys or secrets in source code."""
    base = Path(__file__).parent.parent.parent / "src" / "rosclaw" / "auto"
    # Check no ghp_ or sk- tokens in source
    violations = []
    for pyfile in base.rglob("*.py"):
        content = pyfile.read_text()
        if "ghp_" in content or "sk-" in content:
            violations.append(f"{pyfile}: potential leaked token")
    assert not violations, "Potential secrets found"


def test_all_core_models_have_to_dict():
    """AUTO-CORE-000: All core models must implement to_dict/from_dict."""
    import importlib
    core = importlib.import_module("rosclaw.auto.core")
    for name in dir(core):
        obj = getattr(core, name)
        if isinstance(obj, type) and hasattr(obj, "__dataclass_fields__"):
            assert hasattr(obj, "to_dict"), f"{name} missing to_dict"
            assert hasattr(obj, "from_dict"), f"{name} missing from_dict"


def test_dataclass_fields_not_empty():
    """AUTO-CORE-000b: Core models must have required fields."""
    from rosclaw.auto.core import AutoTask, FailureCase
    task = AutoTask(id="t1", name="pick", task_type="skill_tuning", robot_id="r1",
                    environment_id="e1", target_skill_id="s1")
    assert task.id == "t1"
    fc = FailureCase(id="f1", praxis_event_id="e1", task_id="t1", skill_id="s1")
    assert fc.failure_mode == ""
