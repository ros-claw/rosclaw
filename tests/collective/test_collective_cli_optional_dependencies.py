from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_collective_cli_import_does_not_require_torch() -> None:
    script = textwrap.dedent(
        """
        import builtins
        import sys

        sys.path.insert(0, "src")

        original_import = builtins.__import__

        def import_without_torch(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise ModuleNotFoundError("No module named 'torch'", name="torch")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = import_without_torch
        from rosclaw.collective.cli import dispatch_collective_argv

        assert dispatch_collective_argv(["not-collective"]) is None
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=_REPOSITORY_ROOT,
    )

    assert completed.returncode == 0, completed.stderr


def test_motion_prior_without_torch_returns_structured_fail_closed_error() -> None:
    script = textwrap.dedent(
        """
        import builtins
        import sys

        sys.path.insert(0, "src")

        original_import = builtins.__import__

        def import_without_torch(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise ModuleNotFoundError("No module named 'torch'", name="torch")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = import_without_torch
        from rosclaw.collective.cli import dispatch_collective_argv

        result = dispatch_collective_argv([
            "collective", "prior", "build",
            "--pilot-report", "pilot.json",
            "--dataset-root", "dataset",
            "--model-path", "g1.xml",
            "--output", "prior.npz",
        ])
        assert result == 2
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=_REPOSITORY_ROOT,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is False
    assert "rosclaw[rl]" in payload["error"]
