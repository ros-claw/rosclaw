"""Pytest configuration for ROSClaw test suite."""

import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

# ------------------------------------------------------------------
# ROS2 environment auto-detection (opt-in via env var)
# ------------------------------------------------------------------
# NOTE: Auto-injecting ROS2 paths during pytest collection triggers
# ROS2 pytest plugins (launch-testing-ros, ament-*) that conflict with
# standard test collection. ROS2 integration tests run in subprocesses
# via wrapper tests; they do not need this auto-detection.
# ------------------------------------------------------------------

if os.environ.get("ROSCLAW_TEST_ROS2"):
    _ros2_base = os.environ.get("ROS_DISTRO", "humble")
    if not _ros2_base.startswith("/"):
        _ros2_base = f"/opt/ros/{_ros2_base}"
    _py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    _ros2_local = "/tmp/ros2-local"

    _ros2_python_paths = [
        f"{_ros2_local}{_ros2_base}/local/lib/{_py_version}/dist-packages",
        f"{_ros2_base}/local/lib/{_py_version}/dist-packages",
    ]
    for _p in reversed(_ros2_python_paths):
        if os.path.isdir(_p) and _p not in sys.path:
            sys.path.insert(0, _p)
    _existing = os.environ.get("PYTHONPATH", "")
    _new_paths = [p for p in _ros2_python_paths if os.path.isdir(p) and p not in _existing]
    if _new_paths:
        os.environ["PYTHONPATH"] = ":".join(_new_paths + ([_existing] if _existing else []))

    _ros2_lib_paths = [
        f"{_ros2_local}{_ros2_base}/lib",
        f"{_ros2_base}/lib",
    ]
    _existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    _new_ld = [p for p in _ros2_lib_paths if os.path.isdir(p) and p not in _existing_ld]
    if _new_ld:
        os.environ["LD_LIBRARY_PATH"] = ":".join(_new_ld + ([_existing_ld] if _existing_ld else []))


# ------------------------------------------------------------------
# rosclaw-how service auto-start fixture
# ------------------------------------------------------------------

def _how_service_reachable(url: str, timeout: float = 2.0) -> bool:
    """Best-effort probe of the rosclaw-how health endpoint."""
    try:
        with urllib.request.urlopen(f"{url}/healthz", timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return False


def _seekdb_port_free(host: str, port: int, timeout: float = 1.0) -> bool:
    """Return True if ``host:port`` has no listening TCP socket."""
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            return sock.connect_ex((host, port)) != 0
    except OSError:
        return False


def _find_free_port(host: str = "127.0.0.1") -> int:
    """Ask the kernel for an ephemeral TCP port on ``host``."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _bundled_seekdb_binary() -> str | None:
    """Return the path to the pylibseekdb bundled ``seekdb`` binary, if any."""
    try:
        import pylibseekdb
    except Exception:  # noqa: BLE001
        return None
    binary = Path(pylibseekdb.__file__).resolve().parent / "seekdb"
    if binary.exists() and os.access(binary, os.X_OK):
        return str(binary)
    return None


def _seekdb_server_ready(host: str, port: int, timeout: float = 2.0) -> bool:
    """Probe the seekdb MySQL port by attempting a TCP connect."""
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            return sock.connect_ex((host, port)) == 0
    except OSError:
        return False


@pytest.fixture(scope="session")
def _rosclaw_how_service(tmp_path_factory):
    """Start a local rosclaw-how server if the package is installed.

    If a server is already reachable at the configured endpoint, reuse it.
    Otherwise try to start ``rosclaw-how-server`` in a subprocess and wait
    for it to become healthy.

    rosclaw-how's default embedded SeekDB binds port 2881. On hosts where
    another seekdb/OceanBase instance already owns that port, embedded mode
    hangs. We detect the conflict and automatically start a temporary seekdb
    server on a free port, then launch rosclaw-how in server mode.
    """
    default_url = "http://127.0.0.1:47820"
    url = os.environ.get("ROSCLAW_HOW_ENDPOINT", default_url)

    if _how_service_reachable(url):
        yield url
        return

    server_cmd = shutil.which("rosclaw-how-server")
    if server_cmd is None:
        pytest.skip("rosclaw-how-server not found; install rosclaw-how to run these tests")

    env = os.environ.copy()
    seekdb_proc = None
    base_dir = None

    # If the default embedded seekdb port is occupied, spin up a private server.
    if not _seekdb_port_free("127.0.0.1", 2881):
        seekdb_binary = _bundled_seekdb_binary()
        if seekdb_binary is None:
            pytest.skip(
                "Port 2881 is occupied and the bundled seekdb binary is unavailable; "
                "start a seekdb server manually or free port 2881"
            )
        free_port = _find_free_port("127.0.0.1")
        base_dir = tmp_path_factory.mktemp("rosclaw-how-seekdb")
        seekdb_proc = subprocess.Popen(
            [
                seekdb_binary,
                "--base-dir", str(base_dir),
                "--port", str(free_port),
                "--nodaemon",
                "--parameter", "memory_limit=1G",
                "--parameter", "log_disk_size=2G",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            if seekdb_proc.poll() is not None:
                stdout, _ = seekdb_proc.communicate()
                pytest.skip(f"seekdb server exited early:\n{stdout[-2000:]}")
            if _seekdb_server_ready("127.0.0.1", free_port, timeout=1.0):
                break
            time.sleep(0.5)
        else:
            if seekdb_proc.poll() is None:
                seekdb_proc.terminate()
                seekdb_proc.wait()
            pytest.skip("seekdb server did not become reachable within 60s")

        env["SEEKDB_MODE"] = "server"
        env["SEEKDB_HOST"] = "127.0.0.1"
        env["SEEKDB_PORT"] = str(free_port)
        env["SEEKDB_DATABASE"] = "rosclaw_how"

    proc = subprocess.Popen(
        [server_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    deadline = time.monotonic() + 120.0
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                stdout, _ = proc.communicate()
                pytest.skip(f"rosclaw-how-server exited early:\n{stdout[-2000:]}")
            if _how_service_reachable(url, timeout=2.0):
                yield url
                return
            time.sleep(1.0)
        pytest.skip("rosclaw-how-server did not become reachable within 120s")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if seekdb_proc is not None and seekdb_proc.poll() is None:
            seekdb_proc.terminate()
            try:
                seekdb_proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                seekdb_proc.kill()
                seekdb_proc.wait()
        if base_dir is not None and base_dir.exists():
            shutil.rmtree(base_dir, ignore_errors=True)


def pytest_runtest_setup(item):
    """Ensure log propagation is enabled so caplog works.

    Some pytest environments (e.g. with ROS launch-testing plugins)
    create loggers with propagate=False, which breaks the standard
    ``caplog`` fixture. We force propagation back on for all loggers
    before each test runs.
    """
    # Fix existing loggers
    for name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(name)
        if isinstance(logger, logging.Logger):
            logger.propagate = True
    # Also fix root logger children default
    logging.getLogger().propagate = True


def pytest_runtest_call(item):
    """Re-apply propagate fix right before the test body executes."""
    for name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(name)
        if isinstance(logger, logging.Logger):
            logger.propagate = True
    logging.getLogger().propagate = True


@pytest.fixture
def linked_realsense_workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace with a linked RealSense D405 body."""
    from rosclaw.body.service import BodyInstanceService

    workspace = tmp_path / "rosclaw_ws"
    service = BodyInstanceService(workspace=workspace)
    service.create_or_init(
        robot="realsense_d405",
        name="d405_lab_01",
        mode="registry",
        force=True,
        switch_active=True,
        update_registry=True,
    )
    return workspace


@pytest.fixture
def dummy_png(tmp_path: Path) -> Path:
    """Return a tiny PNG file path (valid enough for CLI image input)."""
    img = tmp_path / "frame.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    return img


@pytest.fixture
def fake_realsense_skill(monkeypatch, dummy_png):
    """Replace the builtin ``realsense_capture_rgbd`` skill with a fast stub.

    The real builtin is loaded dynamically by ``load_builtins`` as a separate
    module object, so monkeypatching the dotted import path used by the tests
    does not take effect.  Instead we patch ``load_builtins`` itself and inject
    a ``SkillEntry`` whose handler writes a color (and optional depth) artifact.
    """
    import shutil

    from rosclaw.skill_manager.registry import SkillEntry, SkillRegistry

    depth_png = dummy_png.parent / "depth.png"
    depth_png.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")

    def _fake_run(params):
        output_dir = Path(params.get("output_dir") or "./capture")
        output_dir.mkdir(parents=True, exist_ok=True)
        color = output_dir / "color.png"
        depth = output_dir / "depth.png"
        shutil.copy(dummy_png, color)
        shutil.copy(depth_png, depth)
        return {
            "status": "success",
            "skill": "realsense_capture_rgbd",
            "artifacts": {
                "color": str(color),
                "depth": str(depth),
            },
            "metrics": {"latency_ms": 10.0, "usb_mode": "USB3", "degraded": False},
        }

    def _fake_load_builtins(registry=None):
        if registry is None:
            registry = SkillRegistry()
        registry.register(
            SkillEntry(
                name="realsense_capture_rgbd",
                description="Fake RealSense RGB-D capture skill",
                skill_type="programmed",
                handler=_fake_run,
                metadata={"builtin": True},
                version="1.0.0",
            )
        )
        return registry, []

    monkeypatch.setattr("rosclaw.skill.builtins.load_builtins", _fake_load_builtins)
    monkeypatch.setattr("rosclaw.cli._image_dimensions", lambda _path: (640, 480))
