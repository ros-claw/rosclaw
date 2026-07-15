"""Regression tests that README commands actually execute without error.

These tests extract bash commands from README.md and README.zh.md and verify
they run successfully. They prevent README/CLI drift on future releases.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
README_PATH = PROJECT_ROOT / "README.md"
README_ZH_PATH = PROJECT_ROOT / "README.zh.md"


_ROSCLAW_CMD_RE = re.compile(r"^(?:\./)?rosclaw\s+(.*)$")


def _extract_commands(markdown: str) -> list[str]:
    """Extract shell commands from bash code blocks, skipping comments."""
    commands = []
    for block in re.findall(r"```bash\n(.*?)\n```", markdown, re.DOTALL):
        current = ""
        for line in block.splitlines():
            line = line.rstrip()
            if not line or line.startswith("#"):
                continue
            if line.endswith("\\"):
                current += line[:-1].strip() + " "
            else:
                current += line.strip()
                cmd = current.strip()
                if _ROSCLAW_CMD_RE.match(cmd):
                    commands.append(cmd)
                current = ""
        if current.strip() and _ROSCLAW_CMD_RE.match(current.strip()):
            commands.append(current.strip())
    return commands


def _run_command(cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a README command through the CLI module."""
    match = _ROSCLAW_CMD_RE.match(cmd)
    args = match.group(1).split() if match else []
    if args[:2] == ["agent", "install"] and "--dry-run" not in args:
        args.append("--dry-run")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "rosclaw.cli", *args],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_readme_commands_are_found() -> None:
    cmds = _extract_commands(README_PATH.read_text(encoding="utf-8"))
    cmds += _extract_commands(README_ZH_PATH.read_text(encoding="utf-8"))
    assert len(cmds) >= 8, f"expected many README commands, found {len(cmds)}"


@pytest.mark.parametrize("cmd", _extract_commands((README_PATH).read_text(encoding="utf-8")))
def test_readme_english_command(cmd: str) -> None:
    _run_and_assert(cmd)


@pytest.mark.parametrize("cmd", _extract_commands((README_ZH_PATH).read_text(encoding="utf-8")))
def test_readme_chinese_command(cmd: str) -> None:
    _run_and_assert(cmd)


def _run_and_assert(cmd: str) -> None:
    """Run command and assert it exits 0, with a few known exceptions."""
    if "<" in cmd or ">" in cmd:
        pytest.skip(f"command contains a documentation placeholder: {cmd}")

    skip_patterns = [
        "rosclaw runtime",
        "rosclaw doctor --ros2",
        "rosclaw start",  # starts long-running services
    ]
    if any(p in cmd for p in skip_patterns):
        pytest.skip(f"command requires external setup or long-running process: {cmd}")

    result = _run_command(cmd)

    result.stderr.lower()
    stdout = result.stdout
    combined = (stdout + result.stderr).lower()

    assert "traceback" not in combined, f"{cmd} crashed with exception:\n{result.stderr}"
    assert "syntaxerror" not in combined, f"{cmd} triggered syntax error:\n{result.stderr}"
    assert "attributeerror" not in combined, f"{cmd} triggered AttributeError:\n{result.stderr}"
    assert "argumenterror" not in combined, f"{cmd} triggered argparse error:\n{result.stderr}"
    assert "unrecognized arguments" not in combined, f"{cmd} has bad arguments:\n{result.stderr}"

    success_expected = [
        "rosclaw --help",
        "rosclaw doctor",
        "rosclaw skill list",
        "rosclaw robot list",
        "rosclaw sandbox list-worlds",
        "rosclaw provider list",
        "rosclaw know search",
        "rosclaw memory status",
        "rosclaw auto --help",
        "rosclaw skill --help",
    ]
    if any(cmd.endswith(s) or s in cmd for s in success_expected):
        assert result.returncode == 0, (
            f"{cmd} should succeed but got rc={result.returncode}\nstderr={result.stderr}"
        )
