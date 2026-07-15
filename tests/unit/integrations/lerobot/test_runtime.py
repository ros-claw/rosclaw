"""Unit tests for the persistent policy runtime protocol and manager."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rosclaw.integrations.lerobot.policy_runtime.client import RuntimeClient
from rosclaw.integrations.lerobot.policy_runtime.manager import PersistentRuntimeManager
from rosclaw.integrations.lerobot.policy_runtime.protocol import (
    RUNTIME_PROTOCOL_VERSION,
    RuntimeRequest,
    RuntimeResponse,
    encode_request,
    encode_response,
    parse_line,
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
        elif method == 'CREATE_SESSION':
            result = {'status': 'ok', 'session_id': req.params.get('session_id')}
        elif method == 'INFER':
            result = {
                'status': 'ok',
                'session_id': req.params.get('session_id'),
                'step_index': 0,
                'raw_action': {'values': [0.1], 'shape': [1], 'dtype': 'float32'},
                'processed_action': {'values': [0.2], 'shape': [1], 'dtype': 'float32'},
            }
        elif method == 'SHUTDOWN':
            result = {'status': 'ok'}
        else:
            result = {'status': 'ok'}
        sys.stdout.write(encode_response(rid, result=result))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
'''


def test_encode_and_parse_request() -> None:
    line = encode_request("HELLO", {"protocol_version": RUNTIME_PROTOCOL_VERSION}, "1")
    parsed = parse_line(line)
    assert isinstance(parsed, RuntimeRequest)
    assert parsed.method == "HELLO"
    assert parsed.id == "1"


def test_encode_and_parse_response() -> None:
    line = encode_response("1", result={"status": "ok"})
    parsed = parse_line(line)
    assert isinstance(parsed, RuntimeResponse)
    assert parsed.id == "1"
    assert parsed.result["status"] == "ok"


def test_parse_response_with_error() -> None:
    line = encode_response("2", error={"code": "oops", "message": "bad"})
    parsed = parse_line(line)
    assert isinstance(parsed, RuntimeResponse)
    assert parsed.error["code"] == "oops"


def test_parse_empty_line_returns_none() -> None:
    assert parse_line("") is None
    assert parse_line("   ") is None


def test_runtime_request_to_dict() -> None:
    req = RuntimeRequest(method="INFER", params={"x": 1}, id="42")
    data = req.to_dict()
    assert data["method"] == "INFER"
    assert data["params"] == {"x": 1}
    assert data["id"] == "42"


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
def manager(fake_worker_module, tmp_path: Path):
    pkg_root, module_name = fake_worker_module
    env = {"PYTHONPATH": str(pkg_root)}
    mgr = PersistentRuntimeManager(
        sys.executable,
        worker_module=module_name,
        env=env,
        startup_timeout_sec=10.0,
        timeout_sec=10.0,
        shutdown_timeout_sec=5.0,
    )
    yield mgr
    mgr.stop()


def test_manager_start_hello(manager: PersistentRuntimeManager) -> None:
    state = manager.start()
    assert state.state == "ready"
    assert manager.state.pid is not None


def test_manager_call_probe(manager: PersistentRuntimeManager) -> None:
    manager.start()
    result = manager.call("PROBE", {})
    assert result["status"] == "ok"


def test_manager_load_policy_and_infer(manager: PersistentRuntimeManager) -> None:
    manager.start()
    load = manager.call("LOAD_POLICY", {"policy_path": "local/test"})
    assert load["status"] == "ok"
    assert load["policy_path"] == "local/test"

    session = manager.call("CREATE_SESSION", {"session_id": "s1"})
    assert session["status"] == "ok"

    infer = manager.call(
        "INFER",
        {"session_id": "s1", "observation": {"observation.state": [0.0]}},
    )
    assert infer["status"] == "ok"
    assert infer["processed_action"]["values"] == [0.2]


def test_runtime_client_create_session(manager: PersistentRuntimeManager) -> None:
    manager.start()
    client = RuntimeClient(manager)
    client.load_policy("local/test")
    session = client.create_session("episode_1")
    assert session.session_id == "episode_1"

    infer = client.infer("episode_1", {"observation.state": [0.0]})
    assert infer["status"] == "ok"


def test_manager_context_manager(fake_worker_module, tmp_path: Path) -> None:
    pkg_root, module_name = fake_worker_module
    env = {"PYTHONPATH": str(pkg_root)}
    with PersistentRuntimeManager(
        sys.executable,
        worker_module=module_name,
        env=env,
        startup_timeout_sec=10.0,
        shutdown_timeout_sec=5.0,
    ) as mgr:
        assert mgr.state.state == "ready"
