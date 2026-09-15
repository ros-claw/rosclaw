"""Guard: validation harness scripts must be non-empty and parseable.

Regression test for the PR-SDB-140-3 incident: `retrieval_quality.py` was
committed as a 0-byte file (its content existed only on disk at qualification
time; an empty placeholder was committed instead).  CI was green because an
empty Python file passes every check — this test makes that impossible for
any script under validation/.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_DIRS = [
    REPO_ROOT / "validation" / "seekdb" / "scripts",
    REPO_ROOT / "validation" / "seekdb" / "benchmarks",
    REPO_ROOT / "validation" / "golden_flywheel" / "scripts",
]


def _scripts():
    for d in VALIDATION_DIRS:
        if d.is_dir():
            yield from sorted(d.rglob("*.py"))


def test_validation_scripts_exist():
    assert list(_scripts()), "validation script directories missing?"


def test_validation_scripts_non_empty_and_parseable():
    for script in _scripts():
        text = script.read_text(encoding="utf-8")
        assert text.strip(), f"{script} is empty (0-byte harness regression)"
        compile(text, str(script), "exec")  # raises SyntaxError if unparseable
