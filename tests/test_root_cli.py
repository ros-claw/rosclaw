"""NA-FIX-8（规格 §8/T-CLI）：root CLI 产品化。"""

from __future__ import annotations

import json
import subprocess
import sys

ENTRY = [sys.executable, "-m", "rosclaw.entrypoint"]


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([*ENTRY, *args], capture_output=True, text=True, timeout=120)


def test_bare_help_is_slim_and_product_level() -> None:
    result = _run()
    assert result.returncode == 0
    assert "Get started" in result.stdout
    assert "Native Agent" in result.stdout
    # 平铺的内部/科研命令不得出现在精简 help。
    for hidden in ("simforge", "darwin", "lerobot", "muscle", "hat-trick"):
        assert hidden not in result.stdout


def test_help_all_shows_full_legacy_list() -> None:
    result = _run("help", "--all")
    assert result.returncode == 0
    assert "simforge" in result.stdout
    assert "darwin" in result.stdout


def test_commands_json_schema_stable() -> None:
    result = _run("commands", "--json")
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["schema_version"] == "rosclaw.commands.v1"
    assert "chat" in data["registry"]
    assert "robot" in data["domain_groups"]
    assert "body" in data["domain_groups"]["robot"]


def test_topic_help_lists_domain_members() -> None:
    result = _run("help", "robot")
    assert result.returncode == 0
    assert "eurdf" in result.stdout
    result = _run("help", "nonsense-topic")
    assert result.returncode == 0
    assert "unknown topic" in result.stdout


def test_legacy_commands_still_execute() -> None:
    result = _run("--version")
    assert result.returncode == 0
    assert "rosclaw" in result.stdout
