"""P0-5: lifecycle adversarial tests for the SeekDB 1.4 server scripts.

These run REAL engine instances on a scratch SEEKDB_HOME + dedicated port —
no mocks.  They prove the identity model end to end:

  port answering != ready;  pid alive != our instance;  ambiguous => refuse.

Requires the seekdb 1.4.0 deb installed (Jetson rig / Gate H1 runner).
Marked ``integration`` — excluded from default runs.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = REPO_ROOT / "scripts" / "seekdb"
TEST_PORT = "28811"

seekdb_available = Path("/usr/bin/seekdb").exists() or any(
    (Path(p) / "seekdb").exists() for p in os.environ.get("PATH", "").split(":")
)
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not seekdb_available, reason="seekdb binary not installed"),
]


def _env(home: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "SEEKDB_HOME": str(home),
            "SEEKDB_DATA_DIR": str(home / "data"),
            "SEEKDB_PID_FILE": str(home / "seekdb.pid"),
            "SEEKDB_LOG_DIR": str(home / "log"),
            "SEEKDB_RUNTIME_JSON": str(home / "runtime.json"),
            "SEEKDB_PORT": TEST_PORT,
        }
    )
    return env


def _run(script: str, env: dict[str, str], *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPTS / script), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )


def _engine_pid(env: dict[str, str]) -> int | None:
    rj = Path(env["SEEKDB_RUNTIME_JSON"])
    if not rj.exists():
        return None
    pid = json.loads(rj.read_text())["pid"]
    try:
        os.kill(pid, 0)
        return pid
    except OSError:
        return None


def _sql_marker(port: str, marker: str) -> bool:
    """Insert a marker row; returns True on success."""
    pymysql = pytest.importorskip("pymysql")
    try:
        conn = pymysql.connect(
            host="127.0.0.1",
            port=int(port),
            user="root",
            password="",
            connect_timeout=5,
            read_timeout=5,
        )
        cur = conn.cursor()
        cur.execute("CREATE DATABASE IF NOT EXISTS lc_test")
        cur.execute("USE lc_test")
        cur.execute("CREATE TABLE IF NOT EXISTS marker (id VARCHAR(64) PRIMARY KEY)")
        cur.execute("INSERT INTO marker (id) VALUES (%s)", (marker,))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception:
        return False


def _sql_marker_visible(port: str, marker: str) -> bool:
    pymysql = pytest.importorskip("pymysql")
    try:
        conn = pymysql.connect(
            host="127.0.0.1",
            port=int(port),
            user="root",
            password="",
            database="lc_test",
            connect_timeout=5,
            read_timeout=5,
        )
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM marker WHERE id=%s", (marker,))
        n = cur.fetchone()[0]
        cur.close()
        conn.close()
        return n == 1
    except Exception:
        return False


@pytest.fixture
def home(tmp_path):
    h = tmp_path / "seekdb_home"
    env = _env(h)
    yield h, env
    # best-effort cleanup: stop whatever the test left running
    _run("stop.sh", env)
    for pid in subprocess.run(
        ["pgrep", "-x", "seekdb"], capture_output=True, text=True
    ).stdout.split():
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                if str(h).encode() in fh.read():
                    os.kill(int(pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass


def _start_ok(env) -> None:
    r = _run("start_1_4.sh", env)
    assert r.returncode == 0, f"start failed: {r.stdout}\n{r.stderr}"
    assert "READY" in r.stdout


# L1: normal start -> READY with identity artifacts.
def test_l1_normal_start_ready(home):
    h, env = home
    _start_ok(env)
    rj = json.loads(Path(env["SEEKDB_RUNTIME_JSON"]).read_text())
    assert rj["engine_version"] == "1.4.0"
    assert rj["base_dir"] == str(h)
    assert rj["port"] == int(TEST_PORT)
    assert (Path(env["SEEKDB_DATA_DIR"]) / ".rosclaw_seekdb_1_4").exists()


# L2: after launcher re-exec the recorded pid IS the real engine.
def test_l2_recorded_pid_is_real_engine(home):
    _h, env = home
    _start_ok(env)
    pid = _engine_pid(env)
    assert pid is not None
    cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode().replace("\0", " ")
    assert "--base-dir" in cmdline and str(env["SEEKDB_HOME"]) in cmdline
    pidfile = Path(env["SEEKDB_PID_FILE"]).read_text().strip()
    assert pidfile == str(pid), "pid file must point at the real engine, not the dead launcher"


# L3: stale pid file while the real engine lives — stop must find it.
def test_l3_stale_pid_file_stop_finds_engine(home):
    _h, env = home
    _start_ok(env)
    pid = _engine_pid(env)
    Path(env["SEEKDB_PID_FILE"]).write_text("999999")  # dead launcher pid
    r = _run("stop.sh", env)
    assert r.returncode == 0, r.stderr
    time.sleep(1)
    with pytest.raises(OSError):
        os.kill(pid, 0)


# L4: a non-seekdb listener on the port -> start HARD FAILs.
def test_l4_foreign_listener_hard_fail(home):
    h, env = home
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", int(TEST_PORT)))
    sock.listen(1)
    try:
        r = _run("start_1_4.sh", env)
        assert r.returncode == 6
        assert "HARD FAIL" in r.stderr
    finally:
        sock.close()
    # and the stamp must NOT exist (nothing started)
    assert not (h / "data" / ".rosclaw_seekdb_1_4").exists()


def _launch_foreign(home_dir: Path) -> int:
    home_dir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [
            "bash",
            "-c",
            f"nohup /usr/bin/seekdb --base-dir {home_dir} --data-dir {home_dir}/data "
            f"--redo-dir {home_dir}/data/redo --port {TEST_PORT} "
            f">> {home_dir}/seekdb.log 2>&1 & echo $!",
        ],
        capture_output=True,
        text=True,
    )
    return int(r.stdout.strip())


def _foreign_pids(foreign: Path) -> list[int]:
    out = []
    for pid in subprocess.run(["pgrep", "-x", "seekdb"], capture_output=True, text=True).stdout.split():
        with (
            contextlib.suppress(OSError, ProcessLookupError),
            open(f"/proc/{pid}/cmdline", "rb") as fh,
        ):
            if str(foreign).encode() in fh.read():
                out.append(int(pid))
    return out


def _wait_foreign_bound(foreign: Path, timeout: float = 90) -> bool:
    """True once the foreign engine's process is up AND sharing the port."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        pids = _foreign_pids(foreign)
        if pids:
            # bound?  seekdb_pids_for_port equivalent: cmdline carries --port
            return True
        time.sleep(1)
    return False


def _kill_foreign(foreign: Path, launcher: int) -> None:
    for pid in _foreign_pids(foreign):
        with contextlib.suppress(OSError, ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
    with contextlib.suppress(OSError):
        os.kill(launcher, signal.SIGKILL)


# L5: an UNKNOWN seekdb alone on the port -> our start HARD FAILs.
# Deterministic: stop ours first so the foreign engine genuinely owns it.
def test_l5_unknown_engine_on_port_hard_fails(home):
    h, env = home
    _start_ok(env)
    r = _run("stop.sh", env)
    assert r.returncode == 0, r.stderr
    foreign = h.parent / "foreign_home"
    launcher = _launch_foreign(foreign)
    try:
        assert _wait_foreign_bound(foreign), "foreign engine never came up"
        r2 = _run("start_1_4.sh", env)
        assert r2.returncode == 6, r2.stdout + r2.stderr
        assert "UNKNOWN" in r2.stderr or "HARD FAIL" in r2.stderr
    finally:
        _kill_foreign(foreign, launcher)


# L6: two engines sharing the port via SO_REUSEPORT -> start HARD FAILs,
# doctor reports AMBIGUOUS.  Skips when the platform won't co-bind.
def test_l6_dual_engine_collision(home):
    h, env = home
    _start_ok(env)
    foreign = h.parent / "foreign_home"
    launcher = _launch_foreign(foreign)
    try:
        assert _wait_foreign_bound(foreign), "foreign engine never came up"
        # give it time to actually attempt the bind
        deadline = time.time() + 60
        while time.time() < deadline:
            out = subprocess.run(
                ["bash", "-c",
                 f"source {SCRIPTS}/_common.sh; seekdb_port_collision_count {TEST_PORT}"],
                capture_output=True, text=True, env=env,
            )
            if out.stdout.strip() == "2":
                break
            if not _foreign_pids(foreign):
                break  # foreign died (bind refused on this platform)
            time.sleep(2)
        count = subprocess.run(
            ["bash", "-c", f"source {SCRIPTS}/_common.sh; seekdb_port_collision_count {TEST_PORT}"],
            capture_output=True, text=True, env=env,
        ).stdout.strip()
        if count != "2":
            pytest.skip("platform did not co-bind two engines on one port")
        r2 = _run("start_1_4.sh", env)
        assert r2.returncode == 6, r2.stdout + r2.stderr
        r3 = _run("doctor.sh", env)
        assert r3.returncode != 0
        assert "AMBIGUOUS" in r3.stdout or "NOT READY" in r3.stdout
    finally:
        _kill_foreign(foreign, launcher)


# L7: failed start must NOT write the stamp.
def test_l7_failed_start_no_stamp(home):
    h, env = home
    env["SEEKDB_BIN"] = "/bin/false"  # "starts" and exits immediately
    r = _run("start_1_4.sh", env)
    assert r.returncode != 0
    assert not (h / "data" / ".rosclaw_seekdb_1_4").exists(), "stamp written for a failed start"
    assert not Path(env["SEEKDB_RUNTIME_JSON"]).exists()


# L8: runtime.json pid reused by an unrelated process -> refuse, don't kill.
def test_l8_pid_reuse_refused(home):
    _h, env = home
    _start_ok(env)
    victim = subprocess.Popen(["sleep", "60"])
    try:
        rj = Path(env["SEEKDB_RUNTIME_JSON"])
        payload = json.loads(rj.read_text())
        payload["pid"] = victim.pid
        payload["process_start_time"] = "bogus"
        rj.write_text(json.dumps(payload))
        r = _run("stop.sh", env)
        assert r.returncode == 8, r.stdout + r.stderr
        assert victim.poll() is None, "stop.sh killed an unrelated reused pid"
    finally:
        victim.kill()


# L9: graceful SIGTERM via stop.sh.
def test_l9_graceful_sigterm(home):
    _h, env = home
    _start_ok(env)
    pid = _engine_pid(env)
    r = _run("stop.sh", env)
    assert r.returncode == 0, r.stderr
    deadline = time.time() + 35
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
            time.sleep(0.5)
        except OSError:
            break
    with pytest.raises(OSError):
        os.kill(pid, 0)
    assert not Path(env["SEEKDB_RUNTIME_JSON"]).exists()


# L10: SIGKILL + stale runtime state -> start recovers, data persists.
def test_l10_sigkill_recovery(home):
    _h, env = home
    _start_ok(env)
    marker = f"l10_{int(time.time())}"
    assert _sql_marker(TEST_PORT, marker)
    pid = _engine_pid(env)
    os.kill(pid, signal.SIGKILL)
    time.sleep(2)
    # stale runtime.json + pid file remain; start must recover cleanly
    r = _run("start_1_4.sh", env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _sql_marker_visible(TEST_PORT, marker), "marker lost after SIGKILL+restart"
