"""Bounded import-scenario transport; execution requires separate review.

This is not an OS sandbox. Independent effects observation remains necessary.
No product modules are imported by the parent or by fixture discovery.
"""

import ast
from pathlib import Path

import pytest

_STAGE_CONSTANTS = {
    "A_cold": "COLD",
    "B_full_dependencies": "PUBLIC",
    "B_star_full_dependencies": "STAR",
    "C_dependency_denial_control": "NEGATIVE_EAGER",
    "C_full_dependencies_controls": "NEGATIVE_API",
}

_BOOTSTRAP = """
import resource
import sys
resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
resource.setrlimit(resource.RLIMIT_AS, (2147483648, 2147483648))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
"""


class ProcessBoundaryError(RuntimeError):
    """A real killed/reaped outcome, not a successful scenario result."""

    def __init__(self, reason, result, directory):
        super().__init__(reason)
        self.reason = reason
        self.result = result
        self.directory = directory


def _scenario_pairs(module_path):
    # Read only the immutable original prefix. Never import the test module.
    import hashlib

    prefix = Path(module_path).read_bytes()[:9573]
    if hashlib.sha256(prefix).hexdigest() != (
        "093fcdd53c64a98b4ba47ca687055c7ca31e0e716881e24604a9df764b089373"
    ):
        raise ValueError("original scenario prefix integrity mismatch")
    tree = ast.parse(prefix.decode("utf-8"))
    values = {}
    wanted = {"COMMON", *_STAGE_CONSTANTS.values()}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in wanted:
                if target.id in values:
                    raise ValueError("duplicate scenario constant")
                value = ast.literal_eval(node.value)
                if not isinstance(value, str):
                    raise ValueError("scenario must be literal text")
                values[target.id] = value
    if set(values) != wanted:
        raise ValueError("incomplete original scenario constants")
    return {stage: values["COMMON"] + values[name] for stage, name in _STAGE_CONSTANTS.items()}


def _source_tree():
    # Checkout-relative, never an audit path or an inherited source override.
    root = Path(__file__).resolve().parents[2] / "src"
    for relative in (
        "rosclaw/__init__.py",
        "rosclaw/practice/__init__.py",
        "rosclaw/perception/handcam/__init__.py",
    ):
        if not (root / relative).is_file():
            raise RuntimeError("BLOCKED: complete repository source tree required")
    return root.resolve(strict=True)


def _child_environment(directory, source_root):
    # Deliberately do not inherit credentials, Python startup, library injection,
    # proxy settings, or user configuration from the invoking shell.
    env = {
        "HOME": str(directory),
        "TMPDIR": str(directory),
        "TMP": str(directory),
        "TEMP": str(directory),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "JOBLIB_MULTIPROCESSING": "0",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    if source_root is not None:
        env["ROSCLAW_CANDIDATE_SRC"] = str(source_root)
    return env


def _run_bounded(source, source_root=None, timeout=90.0, output_limit=8388608):
    """Private transport seam, also used by finite inert contract controls.

    POSIX resource limits and pipe selectors are required, not silently omitted.
    -S avoids sitecustomize, usercustomize and executable .pth files; explicit
    interpreter installation directories retain ordinary dependency resolution.
    Editable-only third party packages requiring .pth execution must be installed
    normally for this contract. No dependency imports are swallowed.
    """
    import os
    import selectors
    import subprocess
    import sys
    import sysconfig
    import tempfile
    import time

    if os.name != "posix":
        raise RuntimeError("BLOCKED: POSIX resource limits/pipe selectors required")
    if not 0 < timeout <= 90 or not 0 < output_limit <= 8388608:
        raise ValueError("invalid transport bounds")
    dependency_paths = []
    for key in ("purelib", "platlib"):
        path = sysconfig.get_path(key)
        if path and path not in dependency_paths:
            dependency_paths.append(path)
    if source_root is not None:
        dependency_paths.insert(0, str(source_root))
    with tempfile.TemporaryDirectory(prefix="rosclaw-import-") as temporary:
        directory = Path(temporary)
        script = directory / "scenario.py"
        # No eval/exec wrapper: the original source is the literal script suffix.
        script.write_text(
            _BOOTSTRAP + "\nsys.path[:0] = " + repr(dependency_paths) + "\n" + source,
            encoding="utf-8",
        )
        argv = [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-X",
            "pycache_prefix=" + str(directory / "missing-pycache"),
            str(script),
        ]
        captured = [bytearray(), bytearray()]
        reason = None
        process = None
        selector = selectors.DefaultSelector()
        deadline = time.monotonic() + timeout
        try:
            process = subprocess.Popen(
                argv,
                cwd=directory,
                env=_child_environment(directory, source_root),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                close_fds=True,
            )
            process.stdin.close()
            for index, stream in enumerate((process.stdout, process.stderr)):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, index)
            total = 0
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    reason = "timeout"
                    break
                for key, _ in selector.select(min(remaining, 0.1)):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    available = output_limit - total
                    captured[key.data].extend(chunk[:available])
                    total += min(len(chunk), available)
                    if len(chunk) > available:
                        reason = "output_limit"
                        break
                if reason is not None:
                    break
            if reason is None:
                try:
                    process.wait(timeout=max(0.001, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    reason = "timeout"
        finally:
            # Kill only the child this call owns; no process-group/global cleanup.
            # All paths, including reader/selector exceptions, reap before removal.
            try:
                if process is not None:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=5)
            finally:
                selector.close()
                if process is not None:
                    process.stdout.close()
                    process.stderr.close()
        result = subprocess.CompletedProcess(
            argv,
            process.returncode,
            captured[0].decode("utf-8", errors="replace"),
            captured[1].decode("utf-8", errors="replace"),
        )
        if reason is not None:
            raise ProcessBoundaryError(reason, result, str(directory))
        return result


class ScenarioProcess:
    """Exactly the five original public pairs; transport is not the public API."""

    def __init__(self, pairs):
        self._pairs = pairs

    def __call__(self, stage, source):
        if not isinstance(stage, str) or stage not in self._pairs:
            raise ValueError("unknown stage")
        if not isinstance(source, str) or source != self._pairs[stage]:
            raise ValueError("source does not match stage")
        return _run_bounded(source, _source_tree())


@pytest.fixture(scope="function", autouse=False)
def root_owned_process():
    return ScenarioProcess(
        _scenario_pairs(Path(__file__).with_name("test_lightweight_public_imports.py"))
    )
