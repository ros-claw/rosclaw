"""Tests for persistent policy runtime config and socket server."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rosclaw.integrations.lerobot.policy_runtime.config import (
    get_daemon_status,
    is_process_alive,
    load_policy_runtime_config,
    read_pid_file,
    save_policy_runtime_config,
    terminate_process,
    write_pid_file,
)
from rosclaw.integrations.lerobot.policy_runtime.manager import PersistentRuntimeManager
from rosclaw.integrations.lerobot.policy_runtime.socket_server import (
    RuntimeSocketServer,
    send_request,
    start_daemon,
    stop_daemon,
)

_FAKE_WORKER_SOURCE = '''\
import sys
from rosclaw.integrations.lerobot.policy_runtime.protocol import encode_response, parse_line


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = parse_line(line)
        method = req.method
        rid = req.id
        if method == 'HELLO':
            result = {'status': 'ok', 'protocol_version': req.params.get('protocol_version')}
        elif method == 'PROBE':
            result = {'status': 'ok', 'policy_loaded': False}
        elif method == 'LOAD_POLICY':
            result = {'status': 'ok', 'policy_path': req.params.get('policy_path')}
        elif method == 'WARMUP':
            result = {'status': 'ok'}
        elif method == 'HEALTH':
            result = {'status': 'ok', 'policy_loaded': False, 'active_sessions': 0}
        elif method == 'SHUTDOWN':
            result = {'status': 'ok'}
        else:
            result = {'status': 'ok'}
        sys.stdout.write(encode_response(rid, result=result))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
'''


@pytest.fixture
def fake_worker_module(tmp_path: Path) -> tuple[Path, str]:
    """Create a fake worker package that echoes protocol responses."""
    pkg_dir = tmp_path / "fake_worker_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "fake_worker.py").write_text(_FAKE_WORKER_SOURCE)
    (pkg_dir / "__main__.py").write_text(
        "from fake_worker_pkg.fake_worker import main\nmain()\n"
    )
    return tmp_path, "fake_worker_pkg"


@pytest.fixture
def isolated_config(monkeypatch, tmp_path: Path):
    """Redirect runtime config files to a temp directory."""
    config_path = tmp_path / "lerobot_policy_runtime.yaml"
    pid_path = tmp_path / "lerobot_policy_runtime.pid"
    socket_path = tmp_path / "lerobot_policy_runtime.sock"
    monkeypatch.setattr(
        "rosclaw.integrations.lerobot.policy_runtime.config.get_policy_runtime_config_path",
        lambda: config_path,
    )
    monkeypatch.setattr(
        "rosclaw.integrations.lerobot.policy_runtime.config.get_policy_runtime_pid_path",
        lambda: pid_path,
    )
    monkeypatch.setattr(
        "rosclaw.integrations.lerobot.policy_runtime.socket_server.get_policy_runtime_socket_path",
        lambda: socket_path,
    )
    return config_path, pid_path, socket_path


def test_load_save_config(isolated_config) -> None:
    assert load_policy_runtime_config() == {}
    save_policy_runtime_config({"policy_path": "local/test", "device": "cpu"})
    config = load_policy_runtime_config()
    assert config["policy_path"] == "local/test"
    assert config["device"] == "cpu"
    assert "updated_at" in config


def test_pid_file_roundtrip(isolated_config) -> None:
    _, pid_path, _ = isolated_config
    write_pid_file(12345, pid_path)
    assert read_pid_file(pid_path) == 12345


def test_is_process_alive_current_process() -> None:
    assert is_process_alive(999999) is False
    assert is_process_alive(0) is False
    assert is_process_alive(-1) is False


def test_terminate_process_not_exists() -> None:
    assert terminate_process(999999) is True


def test_get_daemon_status_empty(isolated_config) -> None:
    status = get_daemon_status()
    assert status["running"] is False
    assert status["pid"] is None


def test_get_daemon_status_with_socket(isolated_config) -> None:
    _, pid_path, socket_path = isolated_config
    write_pid_file(999999, pid_path)
    socket_path.touch()
    save_policy_runtime_config({"socket_path": str(socket_path), "policy_path": "p"})
    status = get_daemon_status()
    assert status["running"] is False  # pid is fake
    assert status["socket_exists"] is True


def test_socket_server_request_response(fake_worker_module, isolated_config, tmp_path: Path) -> None:
    pkg_root, module_name = fake_worker_module
    config_path, pid_path, socket_path = isolated_config
    env = {"PYTHONPATH": str(pkg_root)}
    manager = PersistentRuntimeManager(
        sys.executable,
        worker_module=module_name,
        env=env,
        startup_timeout_sec=10.0,
        timeout_sec=10.0,
        shutdown_timeout_sec=5.0,
    )
    server = RuntimeSocketServer(manager, socket_path=socket_path)
    server.start()
    try:
        response = send_request(socket_path, "PROBE", {}, timeout_sec=10.0)
        assert response["status"] == "ok"
        assert response["policy_loaded"] is False
    finally:
        server.stop()


def test_start_stop_daemon(fake_worker_module, isolated_config, tmp_path: Path) -> None:
    pkg_root, module_name = fake_worker_module
    config_path, pid_path, socket_path = isolated_config
    env = {"PYTHONPATH": str(pkg_root)}
    manager = PersistentRuntimeManager(
        sys.executable,
        worker_module=module_name,
        env=env,
        startup_timeout_sec=10.0,
        timeout_sec=10.0,
        shutdown_timeout_sec=5.0,
    )
    server = start_daemon(
        manager,
        socket_path=socket_path,
        policy_path="local/test",
        python_executable=str(sys.executable),
    )
    try:
        config = load_policy_runtime_config()
        assert config["policy_path"] == "local/test"
        assert config["socket_path"] == str(socket_path)

        response = send_request(socket_path, "LOAD_POLICY", {"policy_path": "local/test"}, timeout_sec=10.0)
        assert response["status"] == "ok"
        assert response["policy_path"] == "local/test"
    finally:
        server.stop()
        if socket_path.exists():
            socket_path.unlink()


def test_stop_daemon_cleans_up(fake_worker_module, isolated_config, tmp_path: Path) -> None:
    pkg_root, module_name = fake_worker_module
    config_path, pid_path, socket_path = isolated_config
    env = {"PYTHONPATH": str(pkg_root)}
    manager = PersistentRuntimeManager(
        sys.executable,
        worker_module=module_name,
        env=env,
        startup_timeout_sec=10.0,
        timeout_sec=10.0,
        shutdown_timeout_sec=5.0,
    )
    server = start_daemon(manager, socket_path=socket_path)
    # Record the server PID so stop_daemon can find it.
    write_pid_file(server.manager.state.pid or 0, pid_path)
    try:
        result = stop_daemon(socket_path)
        assert result["running"] is False
    finally:
        if socket_path.exists():
            socket_path.unlink()
        if pid_path.exists():
            pid_path.unlink()
