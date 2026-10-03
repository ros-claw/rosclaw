"""Long evidence roots must keep isolated, bindable native agent homes."""

import os
import shutil
import socket
from pathlib import Path

from benchmarks.harnessbench.runner import MODEL_PROFILES, _prepare_home_with_profile


def test_long_evidence_home_can_bind_and_keeps_persistent_files(tmp_path):
    evidence = tmp_path / ("evidence_" * 18) / "rh"
    home, env = _prepare_home_with_profile(evidence, MODEL_PROFILES["kimi-coding"])
    try:
        sockpath = home.absolute() / "run/pi-bridge.sock"
        assert len(os.fsencode(sockpath)) <= 107
        assert env["ROSCLAW_HOME"] == str(home)
        assert home.resolve() == evidence.resolve()
        assert (evidence / "agent/models.json").is_file()
        with socket.socket(socket.AF_UNIX) as sock:
            sock.bind(str(sockpath))
        assert (evidence / "run/pi-bridge.sock").exists()
    finally:
        if home != evidence:
            shutil.rmtree(home.parent)


def test_short_home_stays_in_place_and_long_homes_are_isolated(tmp_path):
    short = Path("/tmp") / ("hb-short-" + tmp_path.name[-12:])
    homes = []
    try:
        home, _ = _prepare_home_with_profile(short, MODEL_PROFILES["kimi-coding"])
        assert home == short
        for task in ("one", "two"):
            evidence = tmp_path / ("证据" * 35) / task / "rh"
            homes.append(_prepare_home_with_profile(evidence, MODEL_PROFILES["kimi-coding"])[0])
        assert homes[0] != homes[1]
        (homes[0] / "nonce").write_text("only-one")
        assert not (homes[1] / "nonce").exists()
    finally:
        shutil.rmtree(short, ignore_errors=True)
        for home in homes:
            if home.parent.parent == Path("/tmp"):
                shutil.rmtree(home.parent)


def test_native_launch_pins_trial_workspace_instead_of_enclosing_git_root(tmp_path, monkeypatch):
    from benchmarks.harnessbench import runner

    launches = []

    class Session:
        def __init__(self, argv, env, cwd=None, log_path=None):
            launches.append((argv, cwd))

        def expect(self, *args, **kwargs):
            pass

        def send(self, *args):
            pass

        def stop(self):
            pass

    monkeypatch.setattr("tests.agentd.test_product_journey.PtySession", Session)
    monkeypatch.setattr(runner, "_wait_settled", lambda *args, **kwargs: 0)
    monkeypatch.setattr(runner.oracle, "judge", lambda *args, **kwargs: {"verified_success": False})
    runner.run_leg("B", "U01", tmp_path, 1, model="kimi-coding")
    argv, cwd = launches[0]
    assert argv[argv.index("--workspace") + 1] == str(cwd.absolute())
    assert cwd == tmp_path / "b_u01_kimi-coding_run1"
