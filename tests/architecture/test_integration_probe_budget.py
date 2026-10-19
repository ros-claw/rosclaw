"""Real framed Unix-socket delays; no daemon, action or hardware is started."""

import socket
import threading
import time

import pytest

from rosclaw.daemon.client import DaemonClient, DaemonUnavailableError
from rosclaw.daemon.protocol import make_response, receive_frame, send_frame
from scripts.integration_test import _integration_client


@pytest.mark.parametrize("old_short_budget", [True, False])
def test_fixture_tolerates_delayed_read_only_response_without_retrying(old_short_budget, tmp_path):
    path = tmp_path / "fixture.sock"
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(str(path))
    path.chmod(0o600)
    listener.listen(1)
    listener.settimeout(5)
    methods, errors = [], []

    def serve():
        try:
            with listener.accept()[0] as connection:
                request = receive_frame(connection)
                methods.append(request["method"])
                assert request["method"] == "runtime.status"
                time.sleep(2.2)
                try:
                    send_frame(
                        connection,
                        make_response(request["request_id"], result={"running": True}),
                    )
                except BrokenPipeError:
                    assert old_short_budget
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        client = (
            DaemonClient(socket_path=path, timeout_sec=2.0)
            if old_short_budget
            else _integration_client(path)
        )
        if old_short_budget:
            with pytest.raises(DaemonUnavailableError):
                client.get_runtime_status()
        else:
            assert client.get_runtime_status()["running"] is True
    finally:
        thread.join(timeout=6)
        listener.close()
    assert not thread.is_alive() and not errors
    assert methods == ["runtime.status"]
