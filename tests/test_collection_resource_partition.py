"""The parallel and serial CI selectors retain every test exactly once."""

import os
import subprocess
import sys
from pathlib import Path


def test_embedded_fixture_closure_is_partitioned_without_skipping(tmp_path):
    # Exercise the actual project collection hooks in an isolated pytest run.
    # This scheduling fixture has no engine, network or native dependency.
    source = Path(__file__).parent / "conftest.py"
    (tmp_path / "conftest.py").write_text(
        source.read_text()
        + "\n@pytest.fixture\ndef shared_embedded_seekdb_target():\n    return {}\n"
    )
    config = tmp_path / "pytest.ini"
    config.write_text(
        "[pytest]\nmarkers =\n"
        "    embedded_seekdb: embedded-engine scheduling\n"
        "    perf_serial: serial timing checks\n"
    )
    (tmp_path / "test_scheduling.py").write_text(
        "import pytest\n"
        "@pytest.fixture\n"
        "def wrapped(shared_embedded_seekdb_target): return {}\n"
        "def test_direct(shared_embedded_seekdb_target): pass\n"
        "def test_wrapped(wrapped): pass\n"
        "@pytest.mark.parametrize('shared_embedded_seekdb_target', [{}], indirect=True, ids=['request'])\n"
        "def test_indirect(shared_embedded_seekdb_target): pass\n"
        "def test_ordinary(): pass\n"
        "@pytest.mark.perf_serial\n"
        "def test_timing(): pass\n"
    )

    def collect(expression):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-c",
                str(config),
                "--confcutdir",
                str(tmp_path),
                "--collect-only",
                "-q",
                "-m",
                expression,
            ],
            cwd=tmp_path,
            env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return {
            line for line in result.stdout.splitlines() if line.startswith("test_scheduling.py::")
        }

    all_tests = collect("")
    parallel = collect("not perf_serial and not embedded_seekdb")
    serial = collect("perf_serial or embedded_seekdb")
    assert parallel == {"test_scheduling.py::test_ordinary"}
    assert serial == {
        "test_scheduling.py::test_direct",
        "test_scheduling.py::test_wrapped",
        "test_scheduling.py::test_indirect[request]",
        "test_scheduling.py::test_timing",
    }
    assert not parallel & serial
    assert parallel | serial == all_tests
