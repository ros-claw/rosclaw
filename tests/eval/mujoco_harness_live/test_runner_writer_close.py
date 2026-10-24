"""Offline lifecycle checks; no provider, robot, or model is dispatched."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from benchmarks.harnessbench import runner


def test_public_quit_waits_for_actual_child_to_persist_before_reader(tmp_path):
    final = tmp_path / "final.json"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys,pathlib; assert sys.stdin.readline() == '/quit\\n'; "
            "pathlib.Path(sys.argv[1]).write_text('{\"closed\":true}'); sys.exit(0)",
            str(final),
        ],
        stdin=subprocess.PIPE,
        text=True,
    )
    sent = []

    def send(text):
        sent.append(text)
        proc.stdin.write(text.replace("\r", "\n"))
        proc.stdin.flush()

    try:
        runner._close_completed_native_turn(SimpleNamespace(proc=proc, send=send))
        assert sent == ["/quit\r"]
        assert proc.returncode == 0
        assert final.read_text() == '{"closed":true}'
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        proc.stdin.close()


@pytest.mark.parametrize("failure", ["timeout", "nonzero"])
def test_close_failure_refuses_oracle_and_still_cleans_up(tmp_path, monkeypatch, failure):
    calls = []

    class Proc:
        def poll(self):
            return None

        def wait(self, timeout):
            assert 0 <= timeout <= 10
            if failure == "timeout":
                raise subprocess.TimeoutExpired("fixture", timeout)
            return 7

    class Session:
        proc = Proc()

        def __init__(self, *args, **kwargs):
            pass

        def expect(self, *args, **kwargs):
            pass

        def send(self, text):
            calls.append(text)

        def stop(self, **kwargs):
            calls.append("cleanup")

    monkeypatch.setattr("tests.agentd.test_product_journey.PtySession", Session)
    monkeypatch.setattr(runner, "_wait_settled", lambda *a, **kw: 0)
    monkeypatch.setattr(runner, "_count_session_stats", lambda *a: (0, 0, 0))
    monkeypatch.setattr(runner.oracle, "judge", lambda *a, **kw: pytest.fail("oracle before close"))
    with pytest.raises(runner.RunInfraError) as caught:
        runner.run_leg("B", "U01", tmp_path, 1, model="kimi-coding")
    assert calls[-2:] == ["/quit\r", "cleanup"]
    assert calls.count("/quit\r") == 1
    assert "model_turn_completed" not in caught.value.partial
    assert "native_writer_closed" not in caught.value.partial
    assert caught.value.partial["failure_phase"] == "native_writer_close"


def test_already_exited_nonzero_child_is_not_sent_another_input():
    proc = SimpleNamespace(poll=lambda: 1, wait=lambda timeout: 1)
    with pytest.raises(RuntimeError, match="NATIVE_WRITER_CLOSE_NONZERO"):
        runner._close_completed_native_turn(
            SimpleNamespace(proc=proc, send=lambda *_: pytest.fail("sent after exit"))
        )


def test_timeout_cleanup_has_only_remaining_original_ten_second_budget(tmp_path, monkeypatch):
    clock = [0.0]
    cleanup = []

    class Proc:
        def poll(self):
            return None

        def wait(self, timeout):
            assert timeout == 10
            clock[0] = 10
            raise subprocess.TimeoutExpired("fixture", timeout)

    class Session:
        proc = Proc()

        def __init__(self, *args, **kwargs):
            pass

        def expect(self, *args, **kwargs):
            pass

        def send(self, text):
            pass

        def stop(self, *, timeout):
            cleanup.append(timeout)

    monkeypatch.setattr(runner.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr("tests.agentd.test_product_journey.PtySession", Session)
    monkeypatch.setattr(runner, "_wait_settled", lambda *a, **kw: 0)
    monkeypatch.setattr(runner, "_count_session_stats", lambda *a: (0, 0, 0))
    monkeypatch.setattr(runner.oracle, "judge", lambda *a, **kw: pytest.fail("oracle before close"))
    with pytest.raises(runner.RunInfraError):
        runner.run_leg("B", "U01", tmp_path, 1, model="kimi-coding")
    assert cleanup == [0.0]
