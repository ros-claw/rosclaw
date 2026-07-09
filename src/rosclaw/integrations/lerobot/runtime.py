"""LeRobot runtime discovery and inspection.

This module is intentionally free of heavy imports so that rosclaw-core can
inspect and describe LeRobot runtimes without importing torch or lerobot in
the current interpreter.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import semver

from rosclaw.integrations.lerobot.subprocess_runner import run_command, which

RuntimeMode = Literal["auto", "current-env", "isolated", "external"]
RuntimeState = Literal["not_found", "ready", "degraded", "error"]
LEROBOT_MIN_VERSION = semver.Version(0, 6, 0)
LEROBOT_MAX_VERSION = semver.Version(0, 7, 0)
LEROBOT_REQUIREMENT = ">=0.6,<0.7"
LEROBOT_INFO_MODULE = "lerobot.scripts.lerobot_info"


@dataclass
class PythonRuntimeInfo:
    """Basic information about a Python interpreter."""

    executable: Path
    version: str | None
    major: int | None
    minor: int | None
    ok: bool
    error: str | None = None


@dataclass
class LeRobotRuntime:
    """Describes a discovered or configured LeRobot runtime."""

    mode: RuntimeMode
    runtime_path: Path | None
    python_executable: Path
    pip_executable: Path | None
    lerobot_info_executable: Path | None
    python_version: str | None
    lerobot_version: str | None
    torch_version: str | None
    cuda_available: bool | None
    state: RuntimeState
    in_process_available: bool
    subprocess_available: bool
    lerobot_info_output: str = ""
    error: str | None = None


def inspect_python(python_executable: Path | str) -> PythonRuntimeInfo:
    """Return runtime info for a Python executable without importing it."""
    executable = Path(python_executable)
    if not executable.exists():
        return PythonRuntimeInfo(
            executable=executable,
            version=None,
            major=None,
            minor=None,
            ok=False,
            error=f"Python executable not found: {executable}",
        )

    result = run_command(
        [
            str(executable),
            "-c",
            "import sys; print(sys.version.split()[0]); print(sys.version_info.major); "
            "print(sys.version_info.minor)",
        ],
        timeout=30.0,
    )
    if not result.ok:
        return PythonRuntimeInfo(
            executable=executable,
            version=None,
            major=None,
            minor=None,
            ok=False,
            error=f"Failed to query Python version: {result.stderr}",
        )

    lines = result.stdout.splitlines()
    version = lines[0] if lines else None
    try:
        major = int(lines[1]) if len(lines) > 1 else None
        minor = int(lines[2]) if len(lines) > 2 else None
    except (ValueError, IndexError):
        major = minor = None

    return PythonRuntimeInfo(
        executable=executable,
        version=version,
        major=major,
        minor=minor,
        ok=True,
    )


def find_python312(preferred: Path | str | None = None) -> Path | None:
    """Locate a Python 3.12+ executable suitable for creating an isolated env."""
    candidates: list[str] = []
    if preferred is not None:
        candidates.append(str(preferred))
    candidates.extend(["python3.12", "python3", "python"])

    for candidate in candidates:
        path_str = candidate if Path(candidate).is_absolute() else which(candidate)
        if not path_str:
            continue
        info = inspect_python(path_str)
        if info.ok and info.major == 3 and info.minor is not None and info.minor >= 12:
            return info.executable
    return None


def resolve_lerobot_info(python_executable: Path | str) -> Path | None:
    """Find the best ``lerobot-info`` executable for a Python runtime.

    A global PATH binary is deliberately ignored because it may belong to a
    different Python environment.
    """
    python_path = Path(python_executable)

    # 1. Sibling binary inside the same bin/ directory as python.
    if python_path.name.startswith("python"):
        sibling = python_path.with_name("lerobot-info")
        if sibling.exists():
            return sibling

    # Fall back to the module entry point in the selected Python runtime.
    result = run_command(
        [
            str(python_path),
            "-c",
            "import importlib.util; import sys; "
            f"sys.exit(0 if importlib.util.find_spec('{LEROBOT_INFO_MODULE}') "
            "is not None else 1)",
        ],
        timeout=30.0,
    )
    if result.ok:
        return python_path

    return None


def _probe_lerobot_version(python_executable: Path | str) -> str | None:
    result = run_command(
        [
            str(python_executable),
            "-c",
            "import importlib; m = importlib.import_module('lerobot'); "
            "print(getattr(m, '__version__', 'unknown'))",
        ],
        timeout=30.0,
    )
    return result.stdout if result.ok else None


def is_supported_lerobot_version(version: str | None) -> bool:
    """Return whether a discovered LeRobot version is supported by this bridge."""
    if not version:
        return False
    try:
        parsed = semver.Version.parse(version)
    except ValueError:
        return False
    return LEROBOT_MIN_VERSION <= parsed < LEROBOT_MAX_VERSION


def _probe_torch_version(python_executable: Path | str) -> tuple[str | None, bool | None]:
    result = run_command(
        [
            str(python_executable),
            "-c",
            "import importlib.util; import sys; "
            "spec = importlib.util.find_spec('torch'); "
            "sys.exit(1 if spec is None else 0)",
        ],
        timeout=30.0,
    )
    if not result.ok:
        return None, None

    result = run_command(
        [
            str(python_executable),
            "-c",
            "import torch; print(torch.__version__); print(torch.cuda.is_available())",
        ],
        timeout=30.0,
    )
    if not result.ok:
        return None, None

    lines = result.stdout.splitlines()
    torch_version = lines[0] if lines else None
    cuda_available: bool | None = None
    if len(lines) > 1:
        cuda_available = lines[1].strip().lower() in ("true", "1", "yes")
    return torch_version, cuda_available


def inspect_lerobot_runtime(
    python_executable: Path | str,
    *,
    mode: RuntimeMode = "external",
    runtime_path: Path | str | None = None,
) -> LeRobotRuntime:
    """Inspect a candidate LeRobot runtime without importing in-process."""
    python_path = Path(python_executable)
    info = inspect_python(python_path)

    if not info.ok:
        return LeRobotRuntime(
            mode=mode,
            runtime_path=Path(runtime_path) if runtime_path else None,
            python_executable=python_path,
            pip_executable=None,
            lerobot_info_executable=None,
            python_version=info.version,
            lerobot_version=None,
            torch_version=None,
            cuda_available=None,
            state="error",
            in_process_available=False,
            subprocess_available=False,
            error=info.error,
        )

    if info.major != 3 or (info.minor is not None and info.minor < 12):
        return LeRobotRuntime(
            mode=mode,
            runtime_path=Path(runtime_path) if runtime_path else None,
            python_executable=python_path,
            pip_executable=None,
            lerobot_info_executable=None,
            python_version=info.version,
            lerobot_version=None,
            torch_version=None,
            cuda_available=None,
            state="error",
            in_process_available=False,
            subprocess_available=False,
            error=f"LeRobot requires Python >= 3.12; found {info.version}",
        )

    # Check LeRobot importability.
    import_result = run_command(
        [
            str(python_path),
            "-c",
            "import importlib.util; import sys; "
            "spec = importlib.util.find_spec('lerobot'); "
            "sys.exit(0 if spec is not None else 1)",
        ],
        timeout=30.0,
    )
    lerobot_importable = import_result.ok

    if not lerobot_importable:
        return LeRobotRuntime(
            mode=mode,
            runtime_path=Path(runtime_path) if runtime_path else None,
            python_executable=python_path,
            pip_executable=_guess_pip(python_path),
            lerobot_info_executable=resolve_lerobot_info(python_path),
            python_version=info.version,
            lerobot_version=None,
            torch_version=None,
            cuda_available=None,
            state="degraded",
            in_process_available=sys.executable == str(python_path) and False,
            subprocess_available=False,
            error="LeRobot package is not importable in this runtime",
        )

    lerobot_version = _probe_lerobot_version(python_path)
    if not is_supported_lerobot_version(lerobot_version):
        return LeRobotRuntime(
            mode=mode,
            runtime_path=Path(runtime_path) if runtime_path else None,
            python_executable=python_path,
            pip_executable=_guess_pip(python_path),
            lerobot_info_executable=None,
            python_version=info.version,
            lerobot_version=lerobot_version,
            torch_version=None,
            cuda_available=None,
            state="error",
            in_process_available=False,
            subprocess_available=False,
            error=(
                f"Unsupported LeRobot version {lerobot_version or 'unknown'}; "
                f"required {LEROBOT_REQUIREMENT}"
            ),
        )

    torch_version, cuda_available = _probe_torch_version(python_path)
    lerobot_info = resolve_lerobot_info(python_path)

    # Smoke test: run lerobot-info if we have a dedicated binary; otherwise
    # rely on the import check above as a subprocess smoke signal.
    subprocess_ok = False
    info_output = ""
    if lerobot_info == python_path:
        smoke = run_command(
            [str(python_path), "-m", LEROBOT_INFO_MODULE],
            timeout=60.0,
        )
        subprocess_ok = smoke.ok
        info_output = smoke.stdout
    elif lerobot_info is not None:
        smoke = run_command([str(lerobot_info)], timeout=60.0)
        subprocess_ok = smoke.ok
        info_output = smoke.stdout

    runtime_path_obj = Path(runtime_path) if runtime_path else None
    if runtime_path_obj is None and mode == "current-env":
        runtime_path_obj = python_path.parent.parent if python_path.name == "python" else None

    return LeRobotRuntime(
        mode=mode,
        runtime_path=runtime_path_obj,
        python_executable=python_path,
        pip_executable=_guess_pip(python_path),
        lerobot_info_executable=lerobot_info,
        python_version=info.version,
        lerobot_version=lerobot_version,
        torch_version=torch_version,
        cuda_available=cuda_available,
        state="ready" if subprocess_ok else "degraded",
        in_process_available=Path(sys.executable).resolve() == python_path.resolve(),
        subprocess_available=subprocess_ok,
        lerobot_info_output=info_output,
        error=None if subprocess_ok else "lerobot-info smoke test failed",
    )


def _guess_pip(python_executable: Path) -> Path | None:
    """Return the pip executable likely associated with a Python runtime."""
    candidate = python_executable.with_name("pip")
    if candidate.exists():
        return candidate
    candidate3 = python_executable.with_name("pip3")
    if candidate3.exists():
        return candidate3
    return None


def current_rosclaw_runtime() -> PythonRuntimeInfo:
    """Return runtime info for the Python interpreter running ROSClaw."""
    return inspect_python(sys.executable)
