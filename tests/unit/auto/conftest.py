"""Isolated subprocess runner for the integrated auto CLI."""

import os
import subprocess
import sys
import uuid
from collections.abc import Callable
from pathlib import Path

import pytest

from rosclaw.auto.core import Champion


def _subprocess_env() -> dict[str, str]:
    project_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    python_path = [str(project_root / "src")]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    return env


@pytest.fixture
def run_auto_cli(
    tmp_path: Path,
) -> Callable[..., subprocess.CompletedProcess[str]]:
    """Run an auto CLI module with an isolated local store."""
    env = _subprocess_env()

    def run(
        *args: str,
        module: str = "rosclaw.auto.cli",
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", module, *args],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env=env,
            check=False,
        )

    return run


@pytest.fixture
def run_isolated_python(
    tmp_path: Path,
) -> Callable[[str], subprocess.CompletedProcess[str]]:
    """Run a Python snippet against this checkout in a fresh interpreter."""
    env = _subprocess_env()

    def run(code: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env=env,
            check=False,
        )

    return run


@pytest.fixture
def store_test_champion():
    """Persist an explicitly unverified champion for read-only view tests."""

    def store(
        engine,
        skill_id: str,
        task_id: str,
        level: str = "sim",
        metrics: dict | None = None,
        parent_skill: str = "",
        patch_id: str = "",
        experiment_id: str = "",
    ) -> Champion:
        champion = Champion(
            id=f"champ_fixture_{uuid.uuid4().hex[:8]}",
            skill_id=skill_id,
            task_id=task_id,
            level=level,
            parent_skill_id=parent_skill,
            patch_id=patch_id,
            metrics=metrics or {},
            validation_summary={"promotion_verified": False, "test_fixture": True},
            experiment_id=experiment_id,
        )
        engine.champion_store.save_champion(champion)
        engine.lineage.record(
            skill_id=skill_id,
            parent_skill=parent_skill,
            patch_id=patch_id,
            experiment_id=experiment_id,
            result="fixture",
            metrics=metrics or {},
        )
        return champion

    return store
